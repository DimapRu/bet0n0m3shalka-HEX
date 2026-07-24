# Audio Modem Lab Works — Plan & MCP Tool Reference

## Overview

**Project**: Software-defined audio modem over PC speakers/microphone  
**Stack**: Python + numpy + scipy + sounddevice + audio-dsp-mcp  
**Total Tools in audio-dsp-mcp**: 33 (24 original + 9 new)

---

## 6 Lab Works

### Lab 1: Signal Generation & Basic Modulation

| # | Task | MCP Tools Used | Description |
|---|------|---------------|-------------|
| 1.1 | Generate test tones | `generate_tone` | Sine waves at various frequencies (300–3000 Hz) |
| 1.2 | Generate FSK signal | `generate_fsk` | Binary data → 2-tone FSK (mark=1200, space=2200 Hz) |
| 1.3 | Generate ASK signal | `ask_modulate` | Carrier ON/OFF modulation |
| 1.4 | Generate BPSK signal | `psk_modulate` | Phase-shift keying with π transitions |
| 1.5 | Line coding | `line_code` | NRZ, Manchester, Differential Manchester |
| 1.6 | Visualize signals | `spectrogram`, `fft_analysis` | Check frequency content, time-frequency view |

**Goal**: Understand how digital bits become analog waveforms. Compare FSK/ASK/BPSK spectrum efficiency.

**Supporting tools**: `generate_silence`, `concatenate`, `generate_chirp`

---

### Lab 2: Audio Channel Measurement

| # | Task | MCP Tools Used | Description |
|---|------|---------------|-------------|
| 2.1 | Record/playback | `record_audio`, `play_audio` | Record ambient sound, play test tones |
| 2.2 | Loopback test | `loopback_test` | Play chirp → record → measure delay & SNR |
| 2.3 | Channel impulse response | `impulse_response` | Chirp/MLS → impulse & frequency response |
| 2.4 | Noise analysis | `record_audio`, `fft_analysis` | Record background noise, analyze spectrum |
| 2.5 | Signal quality tools | `eye_diagram` | Eye pattern from recorded signal |

**Goal**: Characterize the acoustic channel (delay, bandwidth, noise floor, multipath).

**Supporting tools**: `load_audio`, `save_audio`, `normalize`

---

### Lab 3: FSK Modem Implementation

| # | Task | MCP Tools Used | Description |
|---|------|---------------|-------------|
| 3.1 | Generate FSK frames | `packet_encode`, `generate_fsk` | Packetize data: preamble + payload + CRC → FSK |
| 3.2 | Play over speakers | `play_audio` | Send FSK signal through speakers |
| 3.3 | Record from mic | `record_audio` | Capture transmitted signal |
| 3.4 | Detect chirp sync | `generate_chirp`, `correlation` | Generate chirp preamble, correlate for sync |
| 3.5 | Demodulate bits | `goertzel`, `spectrogram` | Goertzel per bit window → bit decisions |
| 3.6 | Decode packets | `packet_decode` | Extract preamble, data, verify CRC |
| 3.7 | Measure errors | `ber_measure` | Compare tx_bits vs rx_bits → BER |

**Goal**: End-to-end FSK modem: data → audio → air → mic → data.

**Supporting tools**: `add_noise`, `filter_design`, `apply_filter`, `concatenate`

---

### Lab 4: Noise, Filtering & Equalization

| # | Task | MCP Tools Used | Description |
|---|------|---------------|-------------|
| 4.1 | Add controlled noise | `add_noise` | Mix AWGN at various SNR levels |
| 4.2 | Bandpass filtering | `filter_design`, `apply_filter` | Remove out-of-band noise |
| 4.3 | BER vs SNR curve | `ber_measure`, `add_noise`, `generate_fsk` | Sweep SNR 0–20 dB, measure BER |
| 4.4 | Real interference | `record_audio` | Record with music/talk in background |
| 4.5 | Adaptive equalization | `equalize_lms` | LMS filter to compensate channel distortion |
| 4.6 | Eye diagram analysis | `eye_diagram` | Check eye opening before/after equalization |

**Goal**: Quantify noise resilience. Apply filtering & equalization to improve BER.

**Supporting tools**: `spectrogram`, `fft_analysis`, `normalize`

---

### Lab 5: QAM & OFDM

| # | Task | MCP Tools Used | Description |
|---|------|---------------|-------------|
| 5.1 | QPSK/16-QAM constellation | `constellation_diagram` | Generate I/Q symbols, plot constellation |
| 5.2 | OFDM subcarrier design | `generate_tone`, `concatenate` | Generate multiple orthogonal carriers |
| 5.3 | OFDM symbol generation | `generate_tone`, `generate_fsk` | Map bits to subcarriers (QPSK per carrier) |
| 5.4 | Transmit OFDM | `play_audio` | Play OFDM symbol through speakers |
| 5.5 | FFT-based demodulation | `fft_analysis`, `spectrogram` | FFT per symbol → extract subcarrier phases |
| 5.6 | Channel estimation | `impulse_response`, `equalize_lms` | Pilot tones → estimate channel per subcarrier |

**Goal**: Higher data rate via parallel subcarriers. Understand OFDM principles.

**Supporting tools**: `generate_chirp`, `correlation`, `load_audio`, `save_audio`

---

### Lab 6: Complete System & FEC

| # | Task | MCP Tools Used | Description |
|---|------|---------------|-------------|
| 6.1 | Reed-Solomon coding | `reed_solomon_encode`, `reed_solomon_decode` | Add FEC to data packets |
| 6.2 | Adaptive rate selection | `ber_measure`, `impulse_response` | Measure channel → select best mode |
| 6.3 | Full system test | All relevant tools | End-to-end: file → FEC → packet → modulate → play → record → demod → decode → file |
| 6.4 | Benchmark | `ber_measure`, `loopback_test` | Compare BER for FSK vs PSK vs OFDM at multiple distances |
| 6.5 | Interference test | `record_audio`, `play_audio`, `spectrogram` | Modem working alongside music/speech |

**Goal**: Complete robust modem with forward error correction.

**Supporting tools**: `packet_encode`, `packet_decode`, `add_noise`, `spectrogram`, `eye_diagram`

---

## MCP Tool Reference Summary

### Signal Generation (9 tools)
| Tool | Purpose | Lab |
|------|---------|-----|
| `generate_tone` | Pure sine wave | 1, 5 |
| `generate_fsk` | Frequency Shift Keying | 1, 3, 4 |
| `generate_chirp` | Frequency sweep (sync) | 2, 3 |
| `generate_noise` | White/pink/brown noise | 2, 4 |
| `generate_silence` | Gap padding | 1 |
| `ask_modulate` | Amplitude Shift Keying | 1 |
| `psk_modulate` | Phase Shift Keying | 1 |
| `line_code` | NRZ/Manchester encoding | 1 |

### Signal Analysis (6 tools)
| Tool | Purpose | Lab |
|------|---------|-----|
| `spectrogram` | Time-frequency visualization | 1, 3, 4, 5 |
| `fft_analysis` | Frequency spectrum | 1, 2, 5 |
| `goertzel` | Single-tone detection | 3 |
| `correlation` | Sync/preamble detection | 3, 5 |
| `constellation_diagram` | I/Q visualization | 5 |
| `eye_diagram` | Signal quality analysis | 2, 4 |

### Audio I/O (4 tools)
| Tool | Purpose | Lab |
|------|---------|-----|
| `play_audio` | Play through speakers | 2, 3, 4, 5 |
| `record_audio` | Record from microphone | 2, 3, 4 |
| `load_audio` | Read from file | 2 |
| `save_audio` | Save to file | 2 |

### DSP Processing (7 tools)
| Tool | Purpose | Lab |
|------|---------|-----|
| `trim` | Cut audio segment | 1 |
| `concatenate` | Join audio clips | 1, 5 |
| `normalize` | Peak normalization | 2 |
| `add_noise` | Add AWGN at SNR | 4 |
| `filter_design` | Create Butterworth filter | 4 |
| `apply_filter` | Apply filter to audio | 4 |
| `resample` | Change sample rate | 4 |
| `equalize_lms` | Adaptive channel equalization | 4, 5 |
| `loopback_test` | Play-recorder measurement | 2, 6 |
| `impulse_response` | Channel measurement | 2, 5 |

### Packet & FEC (5 tools)
| Tool | Purpose | Lab |
|------|---------|-----|
| `packet_encode` | Preamble + data + CRC | 3, 6 |
| `packet_decode` | Frame sync & CRC check | 3, 6 |
| `reed_solomon_encode` | Forward error correction | 6 |
| `reed_solomon_decode` | FEC decode & correct | 6 |
| `ber_measure` | Bit Error Rate measurement | 3, 4, 6 |

---

## Additional MCP Servers for This Project

| Server | Tool | Purpose in Project |
|--------|------|--------------------|
| `mcp-sequential-thinking` | `sequentialthinking` | Plan complex debugging, OFDM design |
| `mcp-time` | `get_current_time` | Timestamp measurements, log files |
| `mcp-filesystem` | `read_file`, `write_file`, `search_files` | Manage test data, configs |
| `mcp-git` | `git_commit`, `git_log`, etc. | Version control code and experiments |
| `desktop-commander` | `read_file`, `write_file`, `execute_command` | File operations, Python scripts |
| `mcp-memory` | `create_entities`, `search_nodes` | Store experiment results as knowledge graph |
| `mcp-fetch` | `fetch` | Research academic papers on acoustic modems |
| `dimapoSerachMCP` | `web_search_exa`, `query-docs` | Search web for DSP algorithms, datasheets |
| `audio-dsp-mcp` | 33 tools (this server) | All signal processing, modulation, analysis |

---

## Implementation Roadmap

```
Lab 1 (Signal Generation)
  └── Lab 2 (Channel Measurement)
       ├── Lab 3 (FSK Modem)
       │    └── Lab 4 (Noise & Filtering)
       │         └── Lab 6 (FEC & Full System)
       └── Lab 5 (OFDM)
```

- **Lab 1→3→4→6**: FSK-based modem path (recommended to start)
- **Lab 2→5→6**: OFDM-based high-speed path
- Each lab is 2–4 hours of work
- Labs build on previous results — save WAV files and experimental data
