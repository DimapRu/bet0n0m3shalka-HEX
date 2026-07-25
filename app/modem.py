#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Одна программа на оба устройства:
  python app/modem.py          — меню (передать / принять / файлы / шум / loopback)

Архитектура собрана из блоков, проверенных в labs/ (modem_lib):
  - синхронизация: chirp (FSK-кадры) / Schmidl-Cox (OFDM)
  - надёжный режим: CPFSK 100 бод (Лаба 3: BER=0 до SNR=-10 дБ)
  - быстрые режимы: OFDM QPSK/16QAM/64QAM (Лаба 5)
  - кадр: preamble+len+payload+CRC32 (Лаба 3)
  - FEC: Reed-Solomon + interleave (Лаба 4)
  - ARQ: SEQ+ACK+ретрансляции (Лаба 6)
  - авто-адаптация: таблица Лабы 7 + live-пробник шума
  - целостность: CRC32 на кадр + MD5 всего файла
"""
import os, sys, time, hashlib, argparse, subprocess


def _try_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def _ensure_deps():
    """При первом запуске докачивает зависимости через pip."""
    need = {'numpy': 'numpy', 'scipy': 'scipy',
            'sounddevice': 'sounddevice', 'reedsolo': 'reedsolo'}
    missing = [pip for mod, pip in need.items() if not _try_import(mod)]
    if not missing:
        return
    print(f"[setup] докачиваю зависимости: {', '.join(missing)} ...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
    except Exception as e:
        sys.exit(f"[setup] не удалось установить {missing}: {e}\n"
                 f"Поставьте вручную: pip install -r requirements.txt")


_ensure_deps()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'labs', 'common'))
import numpy as np
import modem_lib as m

FS = m.FS
RECV_DIR = os.path.join(os.path.dirname(__file__), 'received')
SEND_DIR = os.path.join(os.path.dirname(__file__), 'files_to_send')

# ---------------------------------------------------------------------------
# РЕЖИМЫ (Лаба 7). Пороги по SNR откалиброваны по измерениям (_dbg_modes):
# FSK ок уже @10 дБ, QPSK @15, 16QAM @20, 64QAM @30 — взяли пороги с запасом.
MODES = {
    'FSK':   {'min_snr': -100, 'rate': 100},       # страховка
    'QPSK':  {'min_snr': 12,   'rate': 25_000},
    '16QAM': {'min_snr': 18,   'rate': 50_000},
    '64QAM': {'min_snr': 28,   'rate': 75_000},
}
MODE_ORDER = ['FSK', 'QPSK', '16QAM', '64QAM']  # от надёжного к быстрому

PAYLOAD_BYTES = 32          # полезных байт в кадре (Лаба 6)
MAX_RETX = 6                # ретрансляций на кадр (Лаба 6)
ACK_TIMEOUT_S = 1.5
ACK_TIMEOUT_AIR_S = 3.0     # в эфире ответ идёт дольше (хвост записи, эхо)
HS_TRIES = 5                # попыток handshake, прежде чем решить, что приёмника нет
HS_TIMEOUT_S = 3.0
PROBE_EVERY = 8             # пробник шума каждые N кадров
HYSTERESIS_DB = 3.0         # гистерезис против "дёргания" режима

# FSK-тоны по "парам" (тест 5: развод частот)
PAIR_FSK = {1: (1200.0, 2200.0), 2: (3000.0, 4000.0),
            3: (1500.0, 2600.0), 4: (3400.0, 4300.0)}


def pick_mode(snr_db: float) -> str:
    """Выбор режима по SNR (таблица Лабы 7)."""
    mode = 'FSK'
    for name in MODE_ORDER:
        if snr_db >= MODES[name]['min_snr']:
            mode = name
    return mode


def _ofdm_cfg(mode: str) -> m.OFDMConfig:
    order = {'QPSK': 4, '16QAM': 16, '64QAM': 64}[mode]
    return m.OFDMConfig(fft_size=128, cp_len=32, num_data_carriers=48,
                        carrier_start=2, pilot_spacing=8, qam_order=order, fs=FS)


# ---------------------------------------------------------------------------
# TX / RX одного кадра (биты <-> звук)
# ---------------------------------------------------------------------------
def tx_frame_audio(bits: str, mode: str, pair: int) -> np.ndarray:
    if mode == 'FSK':
        mark, space = PAIR_FSK.get(pair, PAIR_FSK[1])
        chirp = m.generate_chirp(300, 3000, 0.05, fs=FS)
        body = m.generate_fsk(bits, mark, space, baud_rate=100, fs=FS,
                              continuous_phase=True)
        return np.concatenate([chirp, m.generate_silence(0.01, FS), body])
    cfg = _ofdm_cfg(mode)
    pre = m.ofdm_sync_preamble(cfg)
    bits = bits + '0' * ((-len(bits)) % cfg.bits_per_symbol)  # целые символы
    body = m.ofdm_modulate(bits, cfg)
    return np.concatenate([pre, body])  # без паузы: оффсет считается точно


def rx_frame_bits(audio: np.ndarray, mode: str, pair: int):
    """Возвращает dict packet_decode или None, если синхронизация не найдена."""
    if mode == 'FSK':
        mark, space = PAIR_FSK.get(pair, PAIR_FSK[1])
        chirp = m.generate_chirp(300, 3000, 0.05, fs=FS)
        if len(audio) <= len(chirp):
            return None
        corr, idx, val = m.cross_correlate(audio, chirp)
        if abs(val) < 0.3:
            return None
        start = idx + len(chirp) + int(0.01 * FS)
        seg = audio[start:]
        bits, conf = m.fsk_demodulate(seg, mark, space, 100, FS)
        return m.packet_decode(bits)
    cfg = _ofdm_cfg(mode)
    pre = m.ofdm_sync_preamble(cfg)
    if len(audio) <= len(pre):
        return None
    corr, idx, val = m.cross_correlate(audio, pre)
    if abs(val) < 0.3:
        return None
    start = idx + len(pre)
    sym_len = cfg.fft_size + cfg.cp_len
    n_sym = (len(audio) - start) // sym_len
    if n_sym < 1:
        return None
    seg = audio[start:start + n_sym * sym_len]
    bits, _ = m.ofdm_demodulate(seg, cfg)
    return m.packet_decode(bits)


# ---------------------------------------------------------------------------
# Кадры протокола: [1B type][1B seq][payload...] + FEC
# ---------------------------------------------------------------------------
T_HANDSHAKE, T_DATA, T_ACK, T_SWITCH, T_EOF = 1, 2, 3, 4, 5
T_PING = 6                  # проверка связи: приёмник отвечает T_ACK


def encode_frame(ftype: int, seq: int, payload: bytes) -> str:
    # фиксированное поле payload (32 Б, нулевой паддинг) -> постоянная
    # длина FEC-блока -> деинтерливинг без потери байтов
    payload = payload[:PAYLOAD_BYTES].ljust(PAYLOAD_BYTES, b'\x00')
    core = bytes([ftype, seq]) + payload
    fec = m.rs_encode(core, nsym=10)          # RS поверх (Лаба 4)
    fec = m.interleave(fec, rows=8)           # перемежение (Лаба 4)
    return m.packet_encode(fec)               # preamble+len+CRC32 (Лаба 3)


def decode_frame(dec: dict):
    """Из packet_decode -> (ftype, seq, payload) или None."""
    if not dec.get('found') or not dec.get('crc_ok'):
        return None
    fec_len = 2 + PAYLOAD_BYTES + 10          # длина RS-блока до перемежения
    fec = m.deinterleave(dec['payload'], rows=8, original_len=fec_len)
    core, nerr = m.rs_decode(fec, nsym=10)
    if nerr < 0 or len(core) < 2:
        return None
    return core[0], core[1], core[2:]


# ---------------------------------------------------------------------------
# Звук: реальный (sounddevice) или loopback-канал
# ---------------------------------------------------------------------------
class Channel:
    """Единый интерфейс: real (микрофон/динамик) или loopback (numpy+AWGN).

    Loopback — полудуплексный эфир без самопрослушки: две очереди
    (to_rx, to_tx). Сторона пишет в очередь собеседника, читает свою.
    Роль задаётся через role='tx'/'rx' (в однопоточном режиме не нужна).
    """
    _q = None          # {'to_rx': ndarray, 'to_tx': ndarray}
    _lock = None

    def __init__(self, loopback=False, loop_snr=20.0, role=None):
        self.loopback = loopback
        self.loop_snr = loop_snr
        self.role = role            # 'tx' или 'rx' (None = одиночный)
        if loopback:
            import threading
            if Channel._q is None:
                Channel._q = {'to_rx': np.zeros(0, np.float32),
                              'to_tx': np.zeros(0, np.float32)}
                Channel._lock = threading.Lock()
        else:
            import sounddevice as sd  # noqa
            self.sd = sd

    def _dst(self):
        # куда пишем: tx -> to_rx, rx -> to_tx, одиночный -> to_rx
        return 'to_tx' if self.role == 'rx' else 'to_rx'

    def _src(self):
        # откуда читаем: tx <- to_tx, rx <- to_rx, одиночный <- to_rx
        return 'to_tx' if self.role == 'tx' else 'to_rx'

    def play(self, audio: np.ndarray):
        if self.loopback:
            noisy = m.add_awgn(audio.astype(np.float32), self.loop_snr)
            with Channel._lock:
                d = self._dst()
                Channel._q[d] = np.concatenate([Channel._q[d], noisy])
        else:
            self.sd.play(audio, FS); self.sd.wait()

    def record(self, seconds: float) -> np.ndarray:
        if self.loopback:
            src = self._src()
            deadline = time.time() + seconds
            while time.time() < deadline:
                with Channel._lock:
                    if len(Channel._q[src]) > 0:
                        break
                time.sleep(0.005)
            time.sleep(0.02)  # дать дописать хвост кадра
            with Channel._lock:
                out = Channel._q[src].copy()
                Channel._q[src] = np.zeros(0, np.float32)
            return out
        rec = self.sd.rec(int(seconds * FS), samplerate=FS, channels=1,
                          dtype='float32')
        self.sd.wait()
        return rec[:, 0]

    def measure_noise(self, seconds: float = 2.0) -> float:
        rec = self.record(seconds)
        return float(np.sqrt(np.mean(rec ** 2)))


# ---------------------------------------------------------------------------
# ПЕРЕДАЧА файла
# ---------------------------------------------------------------------------
def cmd_send(chan: Channel, path: str, pair: int):
    with open(path, 'rb') as f:
        data = f.read()
    name = os.path.basename(path)[:12]          # короткое имя, чтобы handshake влезал
    md5 = hashlib.md5(data).hexdigest()[:8]     # 8 hex достаточно для контроля
    frames = [data[i:i + PAYLOAD_BYTES] for i in range(0, len(data), PAYLOAD_BYTES)]
    n = len(frames)
    print(f"[TX] файл '{name}': {len(data)} Б, MD5 {md5[:8]}..., кадров {n}")

    # 1) HANDSHAKE (всегда FSK — надёжно): имя|размер|md5|n.
    #    Проверка связи: шлём до HS_TRIES раз, ждём ответ приёмника.
    #    Молчит -> приёмника нет (телефон/диктофон не в счёт) -> отмена.
    hs = f"{name}|{len(data)}|{md5}|{n}".encode()
    hs_bits = encode_frame(T_HANDSHAKE, 0, hs)
    hs_audio = tx_frame_audio(hs_bits, 'FSK', pair)
    mode = None
    for t in range(1, HS_TRIES + 1):
        print(f"[TX] handshake {t}/{HS_TRIES} ... (на приёмнике: пункт 2 «Принять файл»)")
        chan.play(hs_audio)
        offer = chan.record(HS_TIMEOUT_S)
        dec = rx_frame_bits(offer, 'FSK', pair)
        fr = decode_frame(dec) if dec else None
        if fr and fr[0] == T_ACK:
            mode = fr[2].rstrip(b'\x00').decode(errors='ignore') or 'QPSK'
            break
    if mode is None:
        print("[TX] ПРИЁМНИК НЕ ОТВЕЧАЕТ — передача отменена.")
        print("     На втором устройстве запустите: python app/modem.py -> 2")
        print("     (если вы записываете на телефон — это запись, а не приёмник:")
        print("      телефон не отвечает на handshake, протоколу нужен живой приёмник)")
        return
    print(f"[TX] приёмник на связи, стартовый режим: {mode}")

    # 3) кадры данных + ARQ + live-адаптация
    t0 = time.time(); retx_total = 0
    for i, chunk in enumerate(frames):
        seq = i % 256
        sent = False
        # цикл деградации: не получили ACK за MAX_RETX — понижаем режим,
        # вплоть до FSK (RX поймёт переключение по FSK-декоду кадра)
        while not sent:
            # в эфире ждём ACK дольше: у записи есть "хвост", плюс эхо/паузы
            ack_to = ACK_TIMEOUT_S if chan.loopback else ACK_TIMEOUT_AIR_S
            for attempt in range(MAX_RETX):
                bits = encode_frame(T_DATA, seq, chunk)
                chan.play(tx_frame_audio(bits, mode, pair))
                ans = chan.record(ack_to)
                dec = rx_frame_bits(ans, 'FSK', pair)
                fr = decode_frame(dec) if dec else None
                if fr and fr[0] in (T_ACK, T_SWITCH) and fr[1] == seq:
                    sent = True
                    if fr[0] == T_SWITCH:  # приёмник просит сменить режим
                        new_mode = fr[2].rstrip(b'\x00').decode(errors='ignore')
                        if new_mode in MODES and new_mode != mode:
                            print(f"[TX] адаптация: {mode} -> {new_mode}")
                            mode = new_mode
                    break
                retx_total += 1
            if sent:
                break
            idx = MODE_ORDER.index(mode)
            if idx == 0:
                print(f"[TX] кадр {i}: НЕ ДОСТАВЛЕН даже в FSK — стоп.")
                return
            mode = MODE_ORDER[idx - 1]
            print(f"\n[TX] нет ACK за {MAX_RETX} попыток -> понижаю до {mode}")
        if (i + 1) % max(1, n // 20) == 0 or i + 1 == n:
            pct = (i + 1) * 100 // n
            bar = '#' * (pct // 5) + '-' * (20 - pct // 5)
            print(f"\r[TX] |{bar}| {pct}% ({i+1}/{n}), режим {mode}, "
                  f"ретрансов {retx_total}", end='', flush=True)
    print()

    # 4) EOF
    chan.play(tx_frame_audio(encode_frame(T_EOF, 0, b''), 'FSK', pair))
    dt = time.time() - t0
    print(f"[TX] готово за {dt:.1f} с, ~{len(data)*8/dt:.0f} бит/с, "
          f"ретранслировано {retx_total}")


# ---------------------------------------------------------------------------
# ПРИЁМ файла
# ---------------------------------------------------------------------------
def cmd_recv(chan: Channel, pair: int):
    os.makedirs(RECV_DIR, exist_ok=True)
    print("[RX] слушаю handshake...")
    while True:
        audio = chan.record(3.0)
        dec = rx_frame_bits(audio, 'FSK', pair)
        fr = decode_frame(dec) if dec else None
        if fr and fr[0] == T_HANDSHAKE:
            name, size, md5, n = fr[2].rstrip(b'\x00').decode().split('|')
            size, n = int(size), int(n)
            break
        if fr and fr[0] == T_PING:
            # проверка связи (пункт меню 6 у передатчика): просто отвечаем
            print("[RX] ping от передатчика -> отвечаю (связь есть)")
            chan.play(tx_frame_audio(encode_frame(T_ACK, fr[1], b''), 'FSK', pair))
    print(f"[RX] файл '{name}': {size} Б, кадров {n}, MD5 {md5[:8]}...")

    # стартовый SNR: в loopback он известен точно; в эфире — оценка по handshake
    if chan.loopback:
        snr = chan.loop_snr
    else:
        noise = float(np.sqrt(np.mean(audio[:int(0.5 * FS)] ** 2))) + 1e-12
        sig = float(np.sqrt(np.mean(audio ** 2)))
        snr = 20 * np.log10(sig / noise + 1e-12)
    mode = pick_mode(snr)
    print(f"[RX] стартовый SNR ~ {snr:.1f} дБ -> режим {mode}")
    chan.play(tx_frame_audio(encode_frame(T_ACK, 0, mode.encode()), 'FSK', pair))

    buf, expected_seq, retx = bytearray(), 0, 0
    snr_window = []
    t0 = time.time()
    while True:
        audio = chan.record(3.0)
        dec = rx_frame_bits(audio, mode, pair)
        fr = decode_frame(dec) if dec else None
        # fallback: EOF идёт всегда в FSK; кроме того TX мог сам понизить
        # режим до FSK после исчерпания ретрансов — ловим оба случая
        if (not fr or fr[0] not in (T_DATA, T_EOF)) and mode != 'FSK':
            dec_f = rx_frame_bits(audio, 'FSK', pair)
            fr_f = decode_frame(dec_f) if dec_f else None
            if fr_f and fr_f[0] == T_EOF:
                fr = fr_f
            elif fr_f and fr_f[0] == T_DATA and not fr:
                # TX понизил режим сам — принимаем FSK-кадр
                fr = fr_f
                if mode != 'FSK':
                    mode = 'FSK'
                    print(f"\n[RX] TX понизил режим -> перехожу на FSK")
        if fr and fr[0] == T_EOF:
            break
        if not fr or fr[0] != T_DATA:
            continue
        ftype, seq, payload = fr
        if seq == expected_seq % 256:
            chunk_len = min(PAYLOAD_BYTES, size - len(buf))
            buf.extend(payload[:max(0, chunk_len)])
            expected_seq += 1
            reply_type, reply_pl = T_ACK, b''
            # live-адаптация: каждые PROBE_EVERY кадров переоцениваем
            if expected_seq % PROBE_EVERY == 0:
                if chan.loopback:
                    snr_now = chan.loop_snr      # в цифровом канале SNR известен
                else:
                    # в эфире: фон меряем в паузе до кадра — берём хвост окна,
                    # где сигнала уже нет (кадр короче окна записи)
                    tail = audio[int(len(audio)*0.8):]
                    noise = float(np.sqrt(np.mean(tail**2))) + 1e-12
                    sig = float(np.sqrt(np.mean(audio**2)))
                    snr_now = 20*np.log10(sig/noise + 1e-12)
                snr_window.append(snr_now)
                cur_min = MODES[mode]['min_snr']
                idx = MODE_ORDER.index(mode)
                if snr_now < cur_min and idx > 0:  # упасть ниже
                    mode = MODE_ORDER[idx-1]; reply_type, reply_pl = T_SWITCH, mode.encode()
                elif idx < len(MODE_ORDER)-1 and snr_now > MODES[MODE_ORDER[idx+1]]['min_snr'] + HYSTERESIS_DB:
                    mode = MODE_ORDER[idx+1]; reply_type, reply_pl = T_SWITCH, mode.encode()
                print(f"\n[RX] SNR~{snr_now:.1f} дБ -> режим {mode}")
            chan.play(tx_frame_audio(encode_frame(reply_type, seq, reply_pl), 'FSK', pair))
            done = min(expected_seq, n)
            pct = done * 100 // n
            bar = '#' * (pct // 5) + '-' * (20 - pct // 5)
            print(f"\r[RX] |{bar}| {pct}% ({done}/{n}), режим {mode}", end='', flush=True)
        else:
            retx += 1
            chan.play(tx_frame_audio(encode_frame(T_ACK, seq, b''), 'FSK', pair))  # дубль-подтверждение
    print()

    data = bytes(buf[:size])
    got_md5 = hashlib.md5(data).hexdigest()[:8]
    ok = (got_md5 == md5)
    out_path = os.path.join(RECV_DIR, name)
    open(out_path, 'wb').write(data)
    dt = time.time() - t0
    print(f"[RX] принято за {dt:.1f} с, ретрансов {retx}")
    print(f"[RX] MD5: {got_md5[:8]}... vs {md5[:8]}...")
    print(f"[RX] ЦЕЛОСТНОСТЬ: {'OK OK' if ok else 'FAIL X'} -> {out_path}")


# ---------------------------------------------------------------------------
# Меню
# ---------------------------------------------------------------------------
def cmd_list():
    if not os.path.isdir(RECV_DIR):
        print("(пусто)"); return
    for f in sorted(os.listdir(RECV_DIR)):
        p = os.path.join(RECV_DIR, f)
        print(f"  {f:30s} {os.path.getsize(p)} Б")


def cmd_ping(chan: Channel, pair: int):
    """Проверка связи: шлём T_PING, приёмник (пункт 2) отвечает T_ACK."""
    bits = encode_frame(T_PING, 0, b'')
    audio = tx_frame_audio(bits, 'FSK', pair)
    print(f"[PING] ищу приёмник (пара {pair}) ...")
    print("       на втором устройстве: python app/modem.py -> 2) Принять файл")
    for t in range(1, 4):
        chan.play(audio)
        ans = chan.record(HS_TIMEOUT_S if not chan.loopback else ACK_TIMEOUT_S)
        dec = rx_frame_bits(ans, 'FSK', pair)
        fr = decode_frame(dec) if dec else None
        if fr and fr[0] == T_ACK:
            print(f"[PING] ПРИЁМНИК НА СВЯЗИ (ответ с попытки {t}) — можно передавать")
            return
        print(f"[PING] попытка {t}/3 — тишина")
    print("[PING] приёмник не найден. Проверьте: запущен ли пункт 2 на другом")
    print("       устройстве, громкость, номер пары (--pair), AGC/шумодав выключены.")


def pick_file_to_send() -> str | None:
    """Выбор файла для передачи:
    - показывает содержимое files_to_send/ (можно выбрать номером);
    - 'd' — системный диалог выбора файла (tkinter);
    - или ввести путь вручную.
    """
    os.makedirs(SEND_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(SEND_DIR)
                   if os.path.isfile(os.path.join(SEND_DIR, f)))
    print(f"\nФайлы в {SEND_DIR}:")
    for i, f in enumerate(files, 1):
        sz = os.path.getsize(os.path.join(SEND_DIR, f))
        print(f"  {i}) {f}  ({sz} Б)")
    print("  d) открыть диалог выбора файла...")
    print("  или введите путь к файлу вручную")
    c = input("> ").strip().strip('"')
    if c.lower() == 'd':
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
            p = filedialog.askopenfilename(
                title='Выберите файл для передачи', initialdir=SEND_DIR)
            root.destroy()
            return p or None
        except Exception as e:
            print(f"диалог недоступен ({e}) — введите путь вручную")
            return None
    if c.isdigit() and 1 <= int(c) <= len(files):
        return os.path.join(SEND_DIR, files[int(c) - 1])
    if os.path.isfile(c):
        return c
    if c:
        print("файл не найден")
    return None


# ---------------------------------------------------------------------------
# ОДНОСТОРОННИЙ РЕЖИМ (без приёмника/ACK): TX просто кричит файл в эфир,
# RX (или запись на телефон) собирает кадры. Всё в FSK — максимально надёжно.
# ---------------------------------------------------------------------------
BCAST_GAP_S = 0.15          # пауза между кадрами при вещании


def cmd_blast(chan: Channel, path: str, pair: int):
    with open(path, 'rb') as f:
        data = f.read()
    name = os.path.basename(path)[:12]
    md5 = hashlib.md5(data).hexdigest()[:8]
    frames = [data[i:i + PAYLOAD_BYTES] for i in range(0, len(data), PAYLOAD_BYTES)]
    n = len(frames)
    print(f"[BLAST-TX] '{name}': {len(data)} Б, MD5 {md5}, кадров {n}, FSK, БЕЗ ACK")
    print("[BLAST-TX] через 3 с — включи пункт 8 на приёмнике или диктофон!")
    time.sleep(3.0)
    t0 = time.time()
    # handshake дважды (вдруг приёмник включился посреди первого)
    hs = encode_frame(T_HANDSHAKE, 0, f"{name}|{len(data)}|{md5}|{n}".encode())
    chan.play(tx_frame_audio(hs, 'FSK', pair)); time.sleep(BCAST_GAP_S)
    chan.play(tx_frame_audio(hs, 'FSK', pair)); time.sleep(BCAST_GAP_S)
    for i, chunk in enumerate(frames):
        bits = encode_frame(T_DATA, i % 256, chunk)
        chan.play(tx_frame_audio(bits, 'FSK', pair))
        time.sleep(BCAST_GAP_S)
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"\r[BLAST-TX] {i+1}/{n}", end='', flush=True)
    chan.play(tx_frame_audio(encode_frame(T_EOF, 0, b''), 'FSK', pair))
    dt = time.time() - t0
    print(f"\n[BLAST-TX] передал всё за {dt:.1f} с (~{len(data)*8/dt:.0f} бит/с)")


def cmd_listen(chan: Channel, pair: int):
    os.makedirs(RECV_DIR, exist_ok=True)
    print("[BLAST-RX] слушаю одностороннюю передачу (FSK)... Ctrl+C — отмена")
    name, size, md5, n = None, 0, '', 0
    buf, expected, got = bytearray(), 0, 0
    t0 = time.time()
    try:
        while True:
            audio = chan.record(3.0)
            dec = rx_frame_bits(audio, 'FSK', pair)
            fr = decode_frame(dec) if dec else None
            if not fr:
                continue
            if fr[0] == T_HANDSHAKE and name is None:
                name, size, md5, n = fr[2].rstrip(b'\x00').decode().split('|')
                size, n = int(size), int(n)
                print(f"[BLAST-RX] файл '{name}': {size} Б, кадров {n}, MD5 {md5}")
                continue
            if fr[0] == T_DATA and name is not None:
                ftype, seq, payload = fr
                if seq == expected % 256:
                    take = min(PAYLOAD_BYTES, size - len(buf))
                    buf.extend(payload[:max(0, take)])
                    expected += 1; got += 1
                    print(f"\r[BLAST-RX] {got}/{n} кадров", end='', flush=True)
                continue
            if fr[0] == T_EOF:
                break
    except KeyboardInterrupt:
        print("\n[BLAST-RX] остановлено пользователем")
    print()
    if name is None:
        print("[BLAST-RX] ничего не услышал (нет handshake)")
        return
    data = bytes(buf[:size])
    got_md5 = hashlib.md5(data).hexdigest()[:8]
    ok = (got_md5 == md5 and got >= n)
    out = os.path.join(RECV_DIR, name)
    open(out, 'wb').write(data)
    dt = time.time() - t0
    print(f"[BLAST-RX] принято {got}/{n} кадров за {dt:.1f} с")
    print(f"[BLAST-RX] MD5: {got_md5} vs {md5} -> "
          f"{'OK' if ok else 'FAIL (потери — без ACK не восстановить)'}")
    print(f"[BLAST-RX] сохранено -> {out}")


def main():
    ap = argparse.ArgumentParser(description='Аудиомодем: передача файлов звуком')
    ap.add_argument('--pair', type=int, default=1, help='номер пары (развод частот, тест 5)')
    ap.add_argument('--loopback', action='store_true', help='тест без железа (numpy-канал)')
    ap.add_argument('--loop-snr', type=float, default=20.0, help='SNR loopback-канала, дБ')
    ap.add_argument('--send', metavar='FILE', help='сразу передать файл')
    ap.add_argument('--recv', action='store_true', help='сразу принять')
    args = ap.parse_args()

    chan = Channel(loopback=args.loopback, loop_snr=args.loop_snr)
    if args.loopback:
        print(f"[loopback] цифровой канал, SNR={args.loop_snr} дБ")

    if args.send:
        cmd_send(chan, args.send, args.pair); return
    if args.recv:
        cmd_recv(chan, args.pair); return

    while True:
        print("\n=== АУДИОМОДЕМ (HEX) ===")
        print("1) Передать файл (динамик)")
        print("2) Принять файл (микрофон)")
        print("3) Посмотреть принятые файлы")
        print("4) Замер фонового шума")
        print("5) Loopback-тест (без железа)")
        print("6) Проверка связи (есть ли приёмник рядом)")
        print("7) Передать файл БЕЗ подтверждений (односторонний, FSK)")
        print("8) Принять одностороннюю передачу (п.7)")
        print("0) Выход")
        c = input("> ").strip()
        if c == '1':
            p = pick_file_to_send()
            if p: cmd_send(chan, p, args.pair)
        elif c == '2': cmd_recv(chan, args.pair)
        elif c == '3': cmd_list()
        elif c == '4':
            rms = chan.measure_noise(2.0)
            # сигнал у нас ~0.5 RMS -> справочный SNR и рекомендуемый режим
            snr_ref = 20 * np.log10(0.5 / (rms + 1e-12))
            print(f"фоновый шум RMS = {rms:.5f} "
                  f"(~{20*np.log10(rms+1e-12):.1f} дБFS)")
            print(f"справочный SNR ~ {snr_ref:.1f} дБ -> режим {pick_mode(snr_ref)}")
        elif c == '5':
            snr_in = input(f"SNR канала, дБ [{args.loop_snr}]: ").strip()
            snr = float(snr_in) if snr_in else args.loop_snr
            lb_rx = Channel(loopback=True, loop_snr=snr, role='rx')
            lb_tx = Channel(loopback=True, loop_snr=snr, role='tx')
            test = os.path.join(os.path.dirname(__file__), '_loop_test.txt')
            with open(test, 'wb') as f:
                f.write(b'Loopback test file! ' * 20)
            import threading
            t = threading.Thread(target=cmd_recv, args=(lb_rx, args.pair), daemon=True)
            t.start(); time.sleep(0.3)
            cmd_send(lb_tx, test, args.pair); t.join(timeout=120)
        elif c == '6': cmd_ping(chan, args.pair)
        elif c == '7':
            p = pick_file_to_send()
            if p: cmd_blast(chan, p, args.pair)
        elif c == '8': cmd_listen(chan, args.pair)
        elif c == '0': break


if __name__ == '__main__':
    main()
