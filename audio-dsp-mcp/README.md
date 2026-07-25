# 🎛️ audio-dsp-mcp

> **MCP-сервер для DSP и аудио-модемов**: генерация сигналов, модуляция (FSK/M-FSK/ASK/BPSK), анализ, эквализация, FEC, симуляция канала, BER-кривые — всё через Model Context Protocol.

[![MCP](https://img.shields.io/badge/MCP-compatible-blue)](https://modelcontextprotocol.io)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tools](https://img.shields.io/badge/tools-43-orange.svg)](#-инструменты-43)

Разработан для проекта **программно-определяемого аудио-модема** — передачи данных между устройствами через динамик и микрофон (acoustic coupling). Подходит для CTF, исследований, лабораторных работ и прототипирования систем связи без радио.

---

## ✨ Возможности

- **Генерация сигналов** — тон, FSK, чирп, шум, ASK, BPSK, line codes
- **Модуляция** — FSK, **M-FSK (4/8/16 тонов)**, ASK, BPSK, NRZ, Manchester, Differential Manchester
- **Анализ** — FFT, спектрограмма, Goertzel, корреляция, eye diagram, constellation
- **Аудио I/O** — запись с микрофона, воспроизведение, loopback-тест
- **DSP** — фильтры Баттерворта, ресэмплинг, LMS-эквалайзер, измерение импульсной характеристики канала
- **Пакеты и FEC** — преамбула + CRC-16, Reed-Solomon
- **Канал и метрики** — симуляция акустического канала (AWGN + реверб + затухание + дропы), **BER-кривые vs SNR**, отчёт качества сигнала
- **Защита от ошибок проектирования** — предупреждение о неортогональности M-FSK тонов (spacing < baud)

---

## 🚀 Установка

```bash
pip install numpy scipy sounddevice soundfile matplotlib reedsolo
```

`sounddevice` нужен для аудио I/O (опционально — без него работает генерация/анализ файлов).

### Подключение к Cline / Roo Code / Claude Desktop

Добавьте в `mcp_settings.json`:

```json
{
  "mcpServers": {
    "audio-dsp-mcp": {
      "command": "python",
      "args": ["C:/path/to/audio-dsp-mcp/audio_dsp_server.py"],
      "disabled": false
    }
  }
}
```

---

## 🛠 Инструменты (43)

### 📡 Генерация сигналов
| Инструмент | Описание |
|---|---|
| `generate_tone` | Синус на заданной частоте |
| `generate_fsk` | FSK-модуляция битовой строки |
| `generate_chirp` | Линейный/логарифмический/квадратичный чирп |
| `generate_noise` | Белый/розовый/коричневый шум |
| `generate_silence` | Тишина |

### 🎚 Модуляция
| Инструмент | Описание |
|---|---|
| `ask_modulate` | ASK (амплитудная манипуляция) |
| `psk_modulate` | BPSK (фазовая манипуляция) |
| `line_code` | NRZ / Manchester / Diff-Manchester |
| `mfsk_modulate` | **M-FSK (4/8/16 тонов)** — 2/3/4 бита на символ, выше битрейт при той же скорости символов; предупреждает о неортогональных тонах |

### 🔬 Анализ
| Инструмент | Описание |
|---|---|
| `spectrogram` | Спектрограмма (PNG + данные) |
| `fft_analysis` | FFT-спектр с пиками |
| `goertzel` | Детекция одной частоты (алгоритм Гёрцеля) |
| `correlation` | Взаимная корреляция (поиск преамбулы) |
| `constellation_diagram` | Созвездие I/Q символов |
| `eye_diagram` | Глазковая диаграмма |

### 🎧 Аудио I/O
| Инструмент | Описание |
|---|---|
| `play_audio` | Воспроизведение через динамики |
| `record_audio` | Запись с микрофона |
| `loopback_test` | Play+Rec одновременно (задержка, SNR) |
| `load_audio` | Загрузка файла (WAV/MP3/FLAC) |
| `save_audio` | Сохранение в WAV |

### ⚙️ DSP
| Инструмент | Описание |
|---|---|
| `trim` | Обрезка по времени |
| `concatenate` | Склейка сигналов |
| `normalize` | Нормализация по пику |
| `add_noise` | Добавление AWGN с заданным SNR |
| `filter_design` | Проектирование фильтра Баттерворта |
| `apply_filter` | Применение фильтра |
| `resample` | Ресэмплинг |
| `impulse_response` | Импульсная характеристика канала (chirp/MLS) |
| `equalize_lms` | Адаптивный LMS-эквалайзер |

### 📦 Пакеты и FEC
| Инструмент | Описание |
|---|---|
| `packet_encode` | Преамбула + данные + CRC-16-CCITT |
| `packet_decode` | Синхронизация, извлечение данных, проверка CRC |
| `reed_solomon_encode` | Reed-Solomon FEC |
| `reed_solomon_decode` | Декодирование с исправлением ошибок |
| `ber_measure` | BER (прямой подсчёт или симуляция BPSK в AWGN) |

### 📶 Канал и метрики (соревновательные)
| Инструмент | Описание |
|---|---|
| `channel_simulate` | **Симуляция акустического канала**: AWGN до заданного SNR + реверберация + затухание с расстоянием + случайные дропы (тест без микрофона) |
| `ber_curve` | **BER vs SNR sweep** для FSK/M-FSK модема + PNG-график — ключевая метрика соревнования |
| `modem_quality` | Отчёт качества принятого сигнала: оценка SNR, уверенность, BER (по эталонному тексту), пропускная способность |

### 🔤 Текст и модем (end-to-end)
| Инструмент | Описание |
|---|---|
| `text_encode` | Текст → base64 (байты) для передачи |
| `text_decode` | base64 (байты) → текст |
| `fsk_demodulate` | FSK-демодуляция аудио → биты (Гёрцель побитово) |
| `mfsk_demodulate` | M-FSK-демодуляция аудио → биты (Гёрцель посимвольно) |
| `modem_tx` | **Полный передатчик**: текст → пакет → FSK-аудио (WAV) |
| `modem_rx` | **Полный приёмник**: аудио → синхронизация → демодуляция → текст |

---

## 📖 Примеры использования

### Передать "Hi" одной командой (рекомендуется)
```
modem_tx({ text: "Hi", baud_rate: 2000 })   → готовый WAV (base64) с чирп-синхронизацией, преамбулой и CRC
play_audio({ audio_data_b64: <поле data из modem_tx> })
```
Слово "Hi" здесь передаётся напрямую — его не нужно кодировать вручную.

### Принять одной командой
```
record_audio({ duration: 3 })
modem_rx({ audio_data_b64: <записанный сигнал>, baud_rate: 2000 })
→ { status: "ok", text: "Hi", crc_ok: true, ... }
```

### Что за "SGk=" в старом примере? (разбор по шагам)
`"SGk="` — это **base64-кодировка строки "Hi"**. Она нигде не прописана как текст, потому что это уже закодированные байты:
```
text_encode({ text: "Hi" })  →  { data_b64: "SGk=", bits: "0100100001101001" }
```
Ручной конвейер (эквивалент `modem_tx`). Здесь base64 **вообще не нужен** — `packet_encode` и `reed_solomon_encode` принимают `text` напрямую:
```
1. packet_encode({ text: "Hi" })                → преамбула + биты + CRC
2. generate_fsk({ bits: "...", mark_freq: 1200, space_freq: 2200, baud_rate: 2000 })
3. play_audio({ audio_data_b64: "..." })
```
Ручной приём (эквивалент `modem_rx`):
```
1. record_audio({ duration: 3 })
2. correlation({ audio_data_b64_ref: <chirp>, audio_data_b64_signal: <recorded> })  → найти начало пакета
3. fsk_demodulate({ audio_data_b64: ..., mark_freq: 1200, space_freq: 2200, baud_rate: 2000 })  → биты
4. packet_decode({ packet_bits: "..." })        → проверка CRC, готовое поле text = "Hi"
```
Base64-путь (`text_encode`/`text_decode`/`data_b64`) остаётся для совместимости и передачи произвольных байтов, но для обычного текста он больше не обязателен.

### Измерить BER при разных SNR
```
ber_measure({ tx_bits: "1010...", snr_range_db: [0, 5, 10, 15, 20] })
→ { results: [{snr_db, errors, ber}, ...], theoretical_ber: [...] }
```

### Оценить канал между динамиком и микрофоном
```
loopback_test({ duration: 2, chirp: true })
→ { delay_seconds, snr_db, rx_data }
```

---

## 🧪 Применение

- **Аудио-модемы** — передача данных через звук (speaker→mic, гидроакустика)
- **CTF / security research** — exfiltration через аудиоканал
- **Лабораторные работы** — FSK/PSK/QAM, BER, FEC, эквализация
- **Прототипирование** — быстрые эксперименты с DSP без Simulink

См. [`plans/audio-modem-lab-plan.md`](../plans/audio-modem-lab-plan.md) для 6 готовых лабораторных работ.

---

## 🏗 Архитектура

```
┌─────────────────────────────────────────────────┐
│           MCP Client (Cline/Roo)                │
└──────────────────┬──────────────────────────────┘
                   │ JSON-RPC over stdio
┌──────────────────▼──────────────────────────────┐
│         audio_dsp_server.py                     │
│  ┌──────────────────────────────────────────┐   │
│  │  AudioDSPTools (38 static methods)       │   │
│  │  ─ numpy + scipy + sounddevice           │   │
│  │  ─ soundfile + matplotlib                │   │
│  └──────────────────────────────────────────┘   │
│  TOOLS registry → TOOL_FUNCTIONS → handler      │
└─────────────────────────────────────────────────┘
```

- **Транспорт**: stdio (JSON-RPC 2.0, MCP protocol 2024-11-05)
- **Формат аудио**: base64-WAV в поле `data`
- **Изображения**: base64-PNG в поле `image_png_base64`

---

## 📦 Зависимости

| Пакет | Назначение | Обязательный |
|---|---|---|
| `numpy` | Базовые операции с сигналами | ✅ |
| `scipy` | Фильтры, чирпы, ресэмплинг | ✅ |
| `sounddevice` | Аудио I/O (play/record) | ⚠️ для I/O |
| `soundfile` | Чтение/запись WAV | ✅ |
| `matplotlib` | Спектрограммы, диаграммы | ⚠️ для визуализации |
| `reedsolo` | Reed-Solomon FEC | ⚠️ для FEC |

---

## 📝 Лицензия

MIT — см. [LICENSE](LICENSE).

## 🤝 Благодарности

Вдохновлено [amodem](https://github.com/romanz/amodem) и задачей *"How to transfer data between devices when usual communication means are unavailable"*.
