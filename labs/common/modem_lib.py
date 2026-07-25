"""
modem_lib.py — общая DSP-библиотека для всех лабораторных работ аудио-модема.

Содержит чистые numpy/scipy реализации:
- Генераторы сигналов (tone, FSK, MFSK, ASK, BPSK, chirp, noise)
- Детекторы (Goertzel, корреляция, FFT)
- Пакетирование (preamble + payload + CRC32)
- FEC (Reed-Solomon, interleaving)
- Симуляция канала (AWGN, reverb, attenuation)
- OFDM ядро (QAM mapping, IFFT/FFT, CP, pilots, Schmidl-Cox)
- Метрики (BER, SNR)
- Визуализация-хелперы

Все функции работают с float32 массивами в диапазоне [-1, 1].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal
import zlib

# ---------------------------------------------------------------------------
# Константы по умолчанию
# ---------------------------------------------------------------------------
FS = 48000  # частота дискретизации, Гц


# ===========================================================================
# 1. ГЕНЕРАТОРЫ СИГНАЛОВ
# ===========================================================================

def generate_tone(freq: float, duration: float, fs: int = FS,
                  amplitude: float = 0.9, phase: float = 0.0) -> NDArray[np.float32]:
    """Чистая синусоида."""
    t = np.arange(int(fs * duration)) / fs
    return (amplitude * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def generate_chirp(f0: float, f1: float, duration: float, fs: int = FS,
                   amplitude: float = 0.9, method: str = "linear") -> NDArray[np.float32]:
    """Частотная развёртка (sweep). method: linear / logarithmic / quadratic."""
    t = np.arange(int(fs * duration)) / fs
    y = sp_signal.chirp(t, f0=f0, f1=f1, t1=duration, method=method)
    return (amplitude * y).astype(np.float32)


def generate_noise(duration: float, fs: int = FS, noise_type: str = "white",
                   amplitude: float = 1.0, seed: int | None = None) -> NDArray[np.float32]:
    """Шум: white / pink / brown."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    white = rng.standard_normal(n)
    if noise_type == "white":
        y = white
    elif noise_type == "pink":
        # Voss-McCartney упрощённый: фильтр 1/f через накопление
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        y = sp_signal.lfilter(b, a, white)
    elif noise_type == "brown":
        y = np.cumsum(white)
        y /= np.max(np.abs(y)) + 1e-12
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    y = y / (np.max(np.abs(y)) + 1e-12)
    return (amplitude * y).astype(np.float32)


def generate_silence(duration: float, fs: int = FS) -> NDArray[np.float32]:
    return np.zeros(int(fs * duration), dtype=np.float32)


def _apply_fade(x: NDArray, fade_samples: int) -> NDArray:
    """Плавное нарастание/спадание (raised cosine) — убирает щелчки."""
    if fade_samples <= 0 or len(x) < 2 * fade_samples:
        return x
    w = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
    x = x.copy()
    x[:fade_samples] *= w
    x[-fade_samples:] *= w[::-1]
    return x


def generate_fsk(bits: str | NDArray, mark_freq: float = 1200.0,
                 space_freq: float = 2200.0, baud_rate: float = 100.0,
                 fs: int = FS, amplitude: float = 0.9,
                 continuous_phase: bool = False,
                 fade_ratio: float = 0.1) -> NDArray[np.float32]:
    """
    BFSK модуляция. bit 1 -> mark_freq, bit 0 -> space_freq.
    continuous_phase=True  -> CPFSK (без скачков фазы, уже спектр).
    fade_ratio            — доля символа на fade (только для не-CP).
    """
    if isinstance(bits, str):
        bit_arr = np.array([int(b) for b in bits], dtype=np.int8)
    else:
        bit_arr = np.asarray(bits, dtype=np.int8)
    sps = int(fs / baud_rate)  # samples per symbol
    n = len(bit_arr) * sps
    out = np.zeros(n, dtype=np.float64)
    if continuous_phase:
        # Интегрируем мгновенную частоту -> фаза непрерывна
        freqs = np.where(bit_arr == 1, mark_freq, space_freq)
        inst = np.repeat(freqs, sps)
        phase = 2 * np.pi * np.cumsum(inst) / fs
        out = amplitude * np.sin(phase)
    else:
        phase_acc = 0.0
        fade_n = int(sps * fade_ratio)
        for i, b in enumerate(bit_arr):
            f = mark_freq if b else space_freq
            t = np.arange(sps) / fs
            seg = amplitude * np.sin(2 * np.pi * f * t + phase_acc)
            seg = _apply_fade(seg, fade_n)
            out[i * sps:(i + 1) * sps] = seg
            phase_acc = 2 * np.pi * f * sps / fs  # сброс фазы каждый символ
    return out.astype(np.float32)


def generate_mfsk(bits: str | NDArray, num_tones: int = 8,
                  base_freq: float = 1000.0, tone_spacing: float = 300.0,
                  baud_rate: float = 100.0, fs: int = FS,
                  amplitude: float = 0.9) -> NDArray[np.float32]:
    """M-FSK: log2(num_tones) бит на символ."""
    if isinstance(bits, str):
        bit_arr = np.array([int(b) for b in bits], dtype=np.int8)
    else:
        bit_arr = np.asarray(bits, dtype=np.int8)
    bps = int(np.log2(num_tones))  # bits per symbol
    pad = (-len(bit_arr)) % bps
    if pad:
        bit_arr = np.concatenate([bit_arr, np.zeros(pad, dtype=np.int8)])
    symbols = bit_arr.reshape(-1, bps)
    sym_vals = symbols.dot(1 << np.arange(bps - 1, -1, -1))
    freqs = base_freq + sym_vals * tone_spacing
    sps = int(fs / baud_rate)
    out = np.zeros(len(freqs) * sps, dtype=np.float64)
    fade_n = int(sps * 0.1)
    for i, f in enumerate(freqs):
        t = np.arange(sps) / fs
        seg = amplitude * np.sin(2 * np.pi * f * t)
        seg = _apply_fade(seg, fade_n)
        out[i * sps:(i + 1) * sps] = seg
    return out.astype(np.float32)


def ask_modulate(bits: str | NDArray, carrier_freq: float = 2000.0,
                 baud_rate: float = 100.0, fs: int = FS,
                 amplitude: float = 0.9) -> NDArray[np.float32]:
    """OOK/ASK: 1 -> несущая, 0 -> тишина."""
    if isinstance(bits, str):
        bit_arr = np.array([int(b) for b in bits], dtype=np.int8)
    else:
        bit_arr = np.asarray(bits, dtype=np.int8)
    sps = int(fs / baud_rate)
    out = np.zeros(len(bit_arr) * sps, dtype=np.float64)
    fade_n = int(sps * 0.1)
    for i, b in enumerate(bit_arr):
        if b:
            t = np.arange(sps) / fs
            seg = amplitude * np.sin(2 * np.pi * carrier_freq * t)
            seg = _apply_fade(seg, fade_n)
            out[i * sps:(i + 1) * sps] = seg
    return out.astype(np.float32)


def psk_modulate(bits: str | NDArray, carrier_freq: float = 2000.0,
                 baud_rate: float = 100.0, fs: int = FS,
                 amplitude: float = 0.9) -> NDArray[np.float32]:
    """BPSK: 0 -> фаза 0, 1 -> фаза pi."""
    if isinstance(bits, str):
        bit_arr = np.array([int(b) for b in bits], dtype=np.int8)
    else:
        bit_arr = np.asarray(bits, dtype=np.int8)
    sps = int(fs / baud_rate)
    out = np.zeros(len(bit_arr) * sps, dtype=np.float64)
    fade_n = int(sps * 0.1)
    for i, b in enumerate(bit_arr):
        phase = np.pi if b else 0.0
        t = np.arange(sps) / fs
        seg = amplitude * np.sin(2 * np.pi * carrier_freq * t + phase)
        seg = _apply_fade(seg, fade_n)
        out[i * sps:(i + 1) * sps] = seg
    return out.astype(np.float32)


def line_code(bits: str | NDArray, encoding: str = "nrz",
              baud_rate: float = 100.0, fs: int = FS,
              amplitude: float = 1.0) -> NDArray[np.float32]:
    """Линейное кодирование: nrz / manchester / diff_manchester (меандр)."""
    if isinstance(bits, str):
        bit_arr = np.array([int(b) for b in bits], dtype=np.int8)
    else:
        bit_arr = np.asarray(bits, dtype=np.int8)
    sps = int(fs / baud_rate)
    half = sps // 2
    if encoding == "nrz":
        level = np.where(bit_arr == 1, amplitude, -amplitude)
        return np.repeat(level, sps).astype(np.float32)
    out = np.zeros(len(bit_arr) * sps, dtype=np.float64)
    state = 1.0  # для diff manchester
    for i, b in enumerate(bit_arr):
        if encoding == "manchester":
            # 1 -> high-low (переход вниз посередине), 0 -> low-high
            if b:
                seg = np.concatenate([np.full(half, amplitude), np.full(sps - half, -amplitude)])
            else:
                seg = np.concatenate([np.full(half, -amplitude), np.full(sps - half, amplitude)])
        elif encoding == "diff_manchester":
            # 0 -> переход в начале, 1 -> нет перехода; всегда переход посередине
            if b == 0:
                state = -state
            seg = np.concatenate([np.full(half, state * amplitude),
                                  np.full(sps - half, -state * amplitude)])
            state = -state
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        out[i * sps:(i + 1) * sps] = seg
    return out.astype(np.float32)


# ===========================================================================
# 2. ДЕТЕКТОРЫ И АНАЛИЗ
# ===========================================================================

def goertzel(x: NDArray, target_freq: float, fs: int = FS) -> float:
    """Алгоритм Гёрцеля — мощность на одной частоте. O(N), быстрее FFT для 1-2 тонов."""
    n = len(x)
    k = int(0.5 + n * target_freq / fs)
    w = 2 * np.pi * k / n
    cw, sw = np.cos(w), np.sin(w)
    coeff = 2 * cw
    s_prev = s_prev2 = 0.0
    for sample in x:
        s = sample + coeff * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    return float(power)


def goertzel_batch(x: NDArray, freqs: NDArray | list, fs: int = FS) -> NDArray:
    """Векторизованный Goertzel для набора частот."""
    return np.array([goertzel(x, f, fs) for f in freqs])


def cross_correlate(signal_rx: NDArray, template: NDArray) -> tuple[NDArray, int, float]:
    """
    Нормированная кросс-корреляция. Возвращает (corr, peak_idx, peak_value).
    peak_idx — позиция начала шаблона в signal_rx.
    """
    corr = sp_signal.fftconvolve(signal_rx, template[::-1], mode="valid")
    norm = np.linalg.norm(template)
    if norm > 0:
        # нормируем на локальную энергию сигнала
        win = np.ones(len(template))
        local_energy = np.sqrt(sp_signal.fftconvolve(signal_rx**2, win, mode="valid"))
        corr = corr / (local_energy * norm + 1e-12)
    peak_idx = int(np.argmax(np.abs(corr)))
    return corr, peak_idx, float(corr[peak_idx])


def fft_spectrum(x: NDArray, fs: int = FS) -> tuple[NDArray, NDArray]:
    """Односторонний амплитудный спектр."""
    n = len(x)
    window = sp_signal.windows.hann(n)
    X = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    mag = np.abs(X) * 2 / n
    return freqs, mag


def fsk_demodulate(x: NDArray, mark_freq: float = 1200.0,
                   space_freq: float = 2200.0, baud_rate: float = 100.0,
                   fs: int = FS) -> tuple[str, NDArray]:
    """
    Демодуляция BFSK через Goertzel. Возвращает (битовая строка, массив confidence).
    Ожидает, что x начинается точно с начала первого символа.
    """
    sps = int(fs / baud_rate)
    n_bits = len(x) // sps
    bits = []
    conf = np.zeros(n_bits)
    for i in range(n_bits):
        seg = x[i * sps:(i + 1) * sps]
        pm = goertzel(seg, mark_freq, fs)
        ps = goertzel(seg, space_freq, fs)
        bits.append("1" if pm > ps else "0")
        conf[i] = abs(pm - ps) / (pm + ps + 1e-12)
    return "".join(bits), conf


def mfsk_demodulate(x: NDArray, num_tones: int = 8,
                    base_freq: float = 1000.0, tone_spacing: float = 300.0,
                    baud_rate: float = 100.0, fs: int = FS) -> tuple[str, NDArray]:
    """Демодуляция M-FSK через Goertzel банк."""
    sps = int(fs / baud_rate)
    n_syms = len(x) // sps
    bps = int(np.log2(num_tones))
    freqs = base_freq + np.arange(num_tones) * tone_spacing
    bits = []
    conf = np.zeros(n_syms)
    for i in range(n_syms):
        seg = x[i * sps:(i + 1) * sps]
        powers = goertzel_batch(seg, freqs, fs)
        sym = int(np.argmax(powers))
        bits.append(format(sym, f"0{bps}b"))
        sorted_p = np.sort(powers)
        conf[i] = (sorted_p[-1] - sorted_p[-2]) / (sorted_p[-1] + 1e-12)
    return "".join(bits), conf


# ===========================================================================
# 3. ПАКЕТИРОВАНИЕ
# ===========================================================================

def bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


def bits_to_bytes(bits: str) -> bytes:
    pad = (-len(bits)) % 8
    bits = bits + "0" * pad
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def crc32_bits(data: bytes) -> str:
    return f"{zlib.crc32(data) & 0xFFFFFFFF:032b}"


def packet_encode(payload: bytes, preamble: str = "1010101010101010") -> str:
    """
    Кадр: preamble(16 бит) + length(16 бит) + payload + crc32(32 бита).
    Возвращает битовую строку.
    """
    length_bits = f"{len(payload):016b}"
    crc_bits = crc32_bits(payload)
    return preamble + length_bits + bytes_to_bits(payload) + crc_bits


def packet_decode(bits: str, preamble: str = "1010101010101010") -> dict:
    """
    Ищет preamble, извлекает payload, проверяет CRC.
    Возвращает dict: found, payload, length, crc_ok, start_idx, errors.
    """
    result = {"found": False, "payload": b"", "length": 0,
              "crc_ok": False, "start_idx": -1, "errors": []}
    idx = bits.find(preamble)
    if idx < 0:
        result["errors"].append("preamble_not_found")
        return result
    result["start_idx"] = idx
    pos = idx + len(preamble)
    if pos + 16 > len(bits):
        result["errors"].append("truncated_length")
        return result
    length = int(bits[pos:pos + 16], 2)
    result["length"] = length
    pos += 16
    payload_bits = bits[pos:pos + length * 8]
    if len(payload_bits) < length * 8:
        result["errors"].append("truncated_payload")
        return result
    pos += length * 8
    crc_bits = bits[pos:pos + 32]
    if len(crc_bits) < 32:
        result["errors"].append("truncated_crc")
        return result
    payload = bits_to_bytes(payload_bits)
    result["payload"] = payload
    expected_crc = f"{zlib.crc32(payload) & 0xFFFFFFFF:032b}"
    result["crc_ok"] = (crc_bits == expected_crc)
    if not result["crc_ok"]:
        result["errors"].append("crc_mismatch")
    result["found"] = True
    return result


# ===========================================================================
# 4. FEC — Reed-Solomon + Interleaving
# ===========================================================================

def rs_encode(data: bytes, nsym: int = 10) -> bytes:
    from reedsolo import RSCodec
    rsc = RSCodec(nsym)
    return bytes(rsc.encode(data))


def rs_decode(data: bytes, nsym: int = 10) -> tuple[bytes, int]:
    """Возвращает (decoded, num_errors_corrected). При фатальной ошибке — (b"", -1)."""
    from reedsolo import RSCodec, ReedSolomonError
    rsc = RSCodec(nsym)
    try:
        decoded, _, errata = rsc.decode(data)
        return bytes(decoded), len(errata)
    except ReedSolomonError:
        return b"", -1


def interleave(data: bytes, rows: int = 8) -> bytes:
    """Блочное перемежение по строкам."""
    cols = int(np.ceil(len(data) / rows))
    pad = rows * cols - len(data)
    arr = np.frombuffer(data + b"\x00" * pad, dtype=np.uint8).reshape(rows, cols)
    return arr.T.tobytes()


def deinterleave(data: bytes, rows: int = 8, original_len: int | None = None) -> bytes:
    cols = len(data) // rows
    arr = np.frombuffer(data, dtype=np.uint8).reshape(cols, rows).T
    out = arr.tobytes()
    if original_len is not None:
        out = out[:original_len]
    return out


# ===========================================================================
# 5. СИМУЛЯЦИЯ КАНАЛА
# ===========================================================================

def add_awgn(x: NDArray, snr_db: float, seed: int | None = None) -> NDArray[np.float32]:
    """Добавляет белый гауссов шум до заданного SNR (дБ)."""
    rng = np.random.default_rng(seed)
    sig_power = np.mean(x**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(x)) * np.sqrt(noise_power)
    return (x + noise).astype(np.float32)


def simulate_channel(x: NDArray, snr_db: float = 20.0,
                     reverb_amount: float = 0.0,
                     attenuation: float = 1.0,
                     fs: int = FS, seed: int | None = None) -> NDArray[np.float32]:
    """
    Модель акустического канала:
    - attenuation: затухание с расстоянием (1/r)
    - reverb: экспоненциальная реверберация (многолучёвость)
    - AWGN: шум
    """
    y = x.astype(np.float64) * attenuation
    if reverb_amount > 0:
        # Импульсная характеристика: прямой луч + затухающее эхо
        ir_len = int(0.05 * fs)  # 50 мс реверберация
        ir = np.exp(-np.linspace(0, 5, ir_len)) * reverb_amount
        ir[0] += 1.0
        y = sp_signal.fftconvolve(y, ir)[:len(y)]
    y = add_awgn(y.astype(np.float32), snr_db, seed)
    return y


def bandpass_filter(x: NDArray, lowcut: float, highcut: float,
                    fs: int = FS, order: int = 5) -> NDArray[np.float32]:
    """Полосовой фильтр Баттерворта."""
    sos = sp_signal.butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sp_signal.sosfilt(sos, x).astype(np.float32)


def estimate_snr(signal: NDArray, noise: NDArray) -> float:
    """SNR в дБ по известному сигналу и шуму."""
    sp = np.mean(signal**2)
    npw = np.mean(noise**2) + 1e-12
    return float(10 * np.log10(sp / npw))


# ===========================================================================
# 6. BER / МЕТРИКИ
# ===========================================================================

def ber_measure(tx_bits: str, rx_bits: str) -> dict:
    """Считает BER, выравнивая строки по минимальной длине."""
    n = min(len(tx_bits), len(rx_bits))
    if n == 0:
        return {"ber": 1.0, "errors": 0, "total": 0}
    errors = sum(1 for a, b in zip(tx_bits[:n], rx_bits[:n]) if a != b)
    return {"ber": errors / n, "errors": errors, "total": n}


def theoretical_ber_fsk(snr_db: float) -> float:
    """Теоретический BER некогерентного BFSK: 0.5 * exp(-Eb/N0 / 2)."""
    from scipy.special import erfc
    eb_n0 = 10 ** (snr_db / 10)
    return 0.5 * np.exp(-eb_n0 / 2)


def theoretical_ber_bpsk(snr_db: float) -> float:
    """Теоретический BER BPSK: 0.5 * erfc(sqrt(Eb/N0))."""
    from scipy.special import erfc
    eb_n0 = 10 ** (snr_db / 10)
    return 0.5 * erfc(np.sqrt(eb_n0))


# ===========================================================================
# 7. QAM / OFDM ЯДРО
# ===========================================================================

def qam_map(bits: str, order: int = 4) -> NDArray[np.complex128]:
    """
    QAM маппинг с Gray-кодированием. order: 4(QPSK), 16, 64.
    Возвращает комплексные символы, нормированные к средней мощности 1.
    """
    bps = int(np.log2(order))  # bits per symbol
    pad = (-len(bits)) % bps
    bits = bits + "0" * pad
    ints = np.array([int(bits[i:i + bps], 2) for i in range(0, len(bits), bps)])
    m = int(np.sqrt(order))
    i_bits = bps // 2
    # разбиваем на I (старшие) и Q (младшие)
    I = ints >> i_bits
    Q = ints & ((1 << i_bits) - 1)
    # Gray code ПО КАЖДОЙ оси отдельно
    I = I ^ (I >> 1)
    Q = Q ^ (Q >> 1)
    # преобразуем в уровни [-m+1, -m+3, ..., m-1]
    I_lvl = 2 * I - (m - 1)
    Q_lvl = 2 * Q - (m - 1)
    symbols = I_lvl + 1j * Q_lvl
    # нормализация к средней мощности 1
    norm = np.sqrt((2 / 3) * (order - 1))
    return symbols / norm


def qam_demap(symbols: NDArray[np.complex128], order: int = 4) -> str:
    """Обратный QAM demapping (hard decision, Gray)."""

    def gray2bin(g: int, width: int) -> int:
        b = 0
        while g:
            b ^= g
            g >>= 1
        return b & ((1 << width) - 1)

    bps = int(np.log2(order))
    m = int(np.sqrt(order))
    norm = np.sqrt((2 / 3) * (order - 1))
    s = symbols * norm
    i_bits = bps // 2
    bits_out = []
    for sym in s:
        I = int(np.clip(np.round((sym.real + m - 1) / 2), 0, m - 1))
        Q = int(np.clip(np.round((sym.imag + m - 1) / 2), 0, m - 1))
        # inverse Gray по каждой оси
        I = gray2bin(I, i_bits)
        Q = gray2bin(Q, i_bits)
        bits_out.append(format((I << i_bits) | Q, f"0{bps}b"))
    return "".join(bits_out)


class OFDMConfig:
    """Параметры OFDM-системы."""
    def __init__(self, fft_size: int = 128, cp_len: int = 32,
                 num_data_carriers: int = 48, carrier_start: int = 2,
                 pilot_spacing: int = 8, qam_order: int = 4, fs: int = FS):
        self.fft_size = fft_size
        self.cp_len = cp_len
        self.num_data_carriers = num_data_carriers
        self.carrier_start = carrier_start
        self.pilot_spacing = pilot_spacing
        self.qam_order = qam_order
        self.fs = fs
        # несущие
        all_carriers = np.arange(carrier_start, carrier_start + num_data_carriers)
        # Для ВЕЩЕСТВЕННОГО (акустического) сигнала несущие k и N-k связаны
        # Hermitian-симметрией: k не должна превышать N/2 и не зеркалиться на другую
        # занятую несущую, иначе они перезапишут друг друга.
        nyq = fft_size // 2
        if all_carriers.max() >= nyq:
            raise ValueError(
                f"Несущая {all_carriers.max()} >= Nyquist-индекса {nyq}: "
                f"для real-сигнала уменьшите num_data_carriers/carrier_start или увеличьте fft_size")
        mirrors = fft_size - all_carriers
        if np.intersect1d(all_carriers, mirrors).size:
            raise ValueError("Hermitian-зеркала несущих пересекаются с самими несущими")
        self.pilot_carriers = all_carriers[::pilot_spacing]
        self.data_carriers = np.setdiff1d(all_carriers, self.pilot_carriers)

    @property
    def bits_per_symbol(self) -> int:
        return len(self.data_carriers) * int(np.log2(self.qam_order))

    @property
    def symbol_duration(self) -> float:
        return (self.fft_size + self.cp_len) / self.fs

    @property
    def bitrate(self) -> float:
        return self.bits_per_symbol / self.symbol_duration


def ofdm_modulate(bits: str, cfg: OFDMConfig) -> NDArray[np.float32]:
    """OFDM модуляция: bits -> QAM -> IFFT -> +CP -> временной сигнал."""
    bps_sym = cfg.bits_per_symbol
    pad = (-len(bits)) % bps_sym
    bits = bits + "0" * pad
    n_symbols = len(bits) // bps_sym
    out = []
    pilot_val = 1.0 + 0.0j  # известный пилот
    for s in range(n_symbols):
        chunk = bits[s * bps_sym:(s + 1) * bps_sym]
        data_syms = qam_map(chunk, cfg.qam_order)
        # раскладываем по несущим
        spectrum = np.zeros(cfg.fft_size, dtype=np.complex128)
        spectrum[cfg.data_carriers] = data_syms
        spectrum[cfg.pilot_carriers] = pilot_val
        # Hermitian symmetry для вещественного выхода IFFT
        spectrum[cfg.fft_size - cfg.data_carriers] = np.conj(data_syms)
        spectrum[cfg.fft_size - cfg.pilot_carriers] = pilot_val
        time_sym = np.fft.ifft(spectrum).real
        # добавляем CP
        with_cp = np.concatenate([time_sym[-cfg.cp_len:], time_sym])
        out.append(with_cp)
    sig = np.concatenate(out)
    # нормализация
    sig = sig / (np.max(np.abs(sig)) + 1e-12) * 0.9
    return sig.astype(np.float32)


def ofdm_demodulate(x: NDArray, cfg: OFDMConfig,
                    equalize: bool = True) -> tuple[str, NDArray[np.complex128]]:
    """
    OFDM демодуляция: -CP -> FFT -> channel estimation (пилоты) -> QAM demap.
    Ожидает x, выровненный по началу символа.
    Возвращает (биты, полученные созвездия).
    """
    sym_len = cfg.fft_size + cfg.cp_len
    n_symbols = len(x) // sym_len
    bits = []
    constellations = []
    for s in range(n_symbols):
        seg = x[s * sym_len + cfg.cp_len:(s + 1) * sym_len]
        spectrum = np.fft.fft(seg)
        # channel estimate по пилотам
        if equalize:
            H_pilots = spectrum[cfg.pilot_carriers] / 1.0
            # интерполяция H на data carriers
            H_data = np.interp(cfg.data_carriers, cfg.pilot_carriers, H_pilots.real) \
                   + 1j * np.interp(cfg.data_carriers, cfg.pilot_carriers, H_pilots.imag)
            rx = spectrum[cfg.data_carriers] / (H_data + 1e-12)
        else:
            rx = spectrum[cfg.data_carriers]
        constellations.append(rx)
        bits.append(qam_demap(rx, cfg.qam_order))
    return "".join(bits), np.concatenate(constellations)


def schmidl_cox_sync(x: NDArray, cfg: OFDMConfig) -> tuple[NDArray, int]:
    """
    Schmidl-Cox синхронизация: метрика M(d) = |P(d)|^2 / R(d)^2.
    Требует preamble из двух одинаковых половин.
    Возвращает (метрика, индекс пика).
    """
    L = cfg.fft_size // 2
    P = np.zeros(len(x), dtype=np.complex128)
    R = np.zeros(len(x), dtype=np.float64)
    x_c = x.astype(np.complex128)
    for d in range(len(x) - 2 * L):
        P[d] = np.sum(x_c[d:d + L] * np.conj(x_c[d + L:d + 2 * L]))
        R[d] = np.sum(np.abs(x_c[d + L:d + 2 * L])**2)
    M = np.abs(P)**2 / (R**2 + 1e-12)
    peak = int(np.argmax(M))
    return M, peak


def ofdm_sync_preamble(cfg: OFDMConfig) -> NDArray[np.float32]:
    """Schmidl-Cox preamble: две идентичные половины."""
    L = cfg.fft_size // 2
    rng = np.random.default_rng(42)
    half = rng.standard_normal(L)
    pre = np.concatenate([half, half])
    pre = pre / np.max(np.abs(pre)) * 0.9
    return pre.astype(np.float32)


# ===========================================================================
# 8. ВИЗУАЛИЗАЦИЯ
# ===========================================================================

def plot_waveform(x: NDArray, fs: int = FS, ax=None, title: str = "",
                 max_duration: float = 0.05, color: str = "tab:blue"):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))
    n = min(len(x), int(fs * max_duration))
    t = np.arange(n) / fs * 1000  # мс
    ax.plot(t, x[:n], color=color, lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or "Waveform")
    ax.grid(True, alpha=0.3)
    return ax


def plot_spectrum(x: NDArray, fs: int = FS, ax=None, title: str = "",
                  fmax: float = 6000, color: str = "tab:blue"):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))
    freqs, mag = fft_spectrum(x, fs)
    mask = freqs <= fmax
    ax.plot(freqs[mask], 20 * np.log10(mag[mask] + 1e-12), color=color, lw=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title or "Spectrum")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-80, None)
    return ax


def plot_spectrogram(x: NDArray, fs: int = FS, ax=None, title: str = "",
                     fmax: float = 6000, nfft: int = 1024):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    # для коротких сигналов уменьшаем окно, иначе noverlap >= nperseg
    nperseg = min(nfft, len(x))
    nperseg = max(8, 1 << int(np.floor(np.log2(nperseg))))  # степень двойки
    f, t, Sxx = sp_signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = f <= fmax
    ax.pcolormesh(t * 1000, f[mask], 10 * np.log10(Sxx[mask] + 1e-12),
                  shading="auto", cmap="viridis")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or "Spectrogram")
    return ax


def plot_constellation(symbols: NDArray[np.complex128], ax=None,
                       title: str = "", color: str = "tab:blue", alpha: float = 0.5):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(symbols.real, symbols.imag, s=12, alpha=alpha, color=color)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title(title or "Constellation")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    return ax


def save_wav(path: str, x: NDArray, fs: int = FS):
    import soundfile as sf
    sf.write(path, x, fs)


def load_wav(path: str) -> tuple[NDArray, int]:
    import soundfile as sf
    data, fs = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, fs
