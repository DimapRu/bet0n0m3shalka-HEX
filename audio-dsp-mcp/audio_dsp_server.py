#!/usr/bin/env python3
"""
Audio DSP MCP Server — Software-defined audio modem toolkit.

Provides tools for: signal generation, analysis, audio I/O,
DSP processing, and modulation utilities for data-over-audio transmission.

Run:
    python audio_dsp_server.py
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import io
import json
import math
import os
import struct
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as sp_signal

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except Exception:
    HAS_SOUNDFILE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# MCP Protocol helpers
# ---------------------------------------------------------------------------

def mcp_log(msg: str) -> None:
    """Write a JSON-RPC notification for logging."""
    _write_message({
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {"level": "info", "data": msg},
    })


def mcp_error(msg: str) -> None:
    _write_message({
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {"level": "error", "data": msg},
    })


def _write_message(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    try:
        sys.stderr.write(line + "\n")
    except UnicodeEncodeError:
        sys.stderr.write(line.encode("ascii", "replace").decode("ascii") + "\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# DSP Utilities
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100
DEFAULT_CHUNK_DURATION = 1.0  # seconds


def _ensure_ndarray(data: Any, dtype=np.float64) -> np.ndarray:
    return np.asarray(data, dtype=dtype)


def _normalize(signal: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val * 0.95
    return signal


def _save_wav_bytes(signal: np.ndarray, samplerate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, signal, samplerate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class AudioDSPTools:
    """Collection of DSP tool methods that return dicts for JSON serialization."""

    # ---- Signal Generation ------------------------------------------------

    @staticmethod
    def generate_tone(frequency: float, duration: float, samplerate: int = SAMPLE_RATE,
                      amplitude: float = 0.9, phase: float = 0.0) -> dict:
        """Generate a sine wave tone."""
        t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        wav_bytes = _save_wav_bytes(signal, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "duration": duration,
            "samplerate": samplerate,
            "samples": len(signal),
        }

    @staticmethod
    def generate_fsk(bits: str, mark_freq: float = 1200, space_freq: float = 2200,
                     baud_rate: float = 300, duration_per_bit: float | None = None,
                     samplerate: int = SAMPLE_RATE, amplitude: float = 0.9) -> dict:
        """Generate FSK modulated signal from binary string (e.g. '10110010')."""
        if not bits:
            return {"status": "error", "message": "bits must not be empty"}
        if any(c not in "01" for c in bits):
            return {"status": "error", "message": "bits must contain only '0' and '1' characters"}
        if duration_per_bit is None:
            duration_per_bit = 1.0 / baud_rate
        samples_per_bit = int(samplerate * duration_per_bit)
        total_samples = len(bits) * samples_per_bit
        t = np.linspace(0, len(bits) * duration_per_bit, total_samples, endpoint=False)

        signal = np.zeros(total_samples)
        for i, bit in enumerate(bits):
            freq = mark_freq if bit in ("1", "1") else space_freq
            start = i * samples_per_bit
            end = (i + 1) * samples_per_bit
            bit_t = t[start:end] - t[start]
            bit_signal = amplitude * np.sin(2 * np.pi * freq * bit_t)
            # Fade in/out to reduce clicks
            fade_len = min(int(samples_per_bit * 0.05), 10)
            if fade_len > 0:
                bit_signal[:fade_len] *= np.linspace(0, 1, fade_len)
                bit_signal[-fade_len:] *= np.linspace(1, 0, fade_len)
            signal[start:end] = bit_signal

        wav_bytes = _save_wav_bytes(signal, samplerate)
        bitrate = len(bits) / (len(bits) * duration_per_bit)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "bits": bits,
            "bit_count": len(bits),
            "duration": len(bits) * duration_per_bit,
            "samplerate": samplerate,
            "baud_rate": baud_rate,
            "bitrate_bps": bitrate,
            "mark_freq": mark_freq,
            "space_freq": space_freq,
        }

    @staticmethod
    def generate_chirp(f0: float, f1: float, duration: float,
                       samplerate: int = SAMPLE_RATE, amplitude: float = 0.9,
                       method: str = "linear") -> dict:
        """Generate a chirp (frequency sweep) for synchronization."""
        t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
        if method == "linear":
            signal = amplitude * sp_signal.chirp(t, f0, duration, f1, method="linear")
        elif method == "logarithmic":
            signal = amplitude * sp_signal.chirp(t, f0, duration, f1, method="logarithmic")
        elif method == "quadratic":
            signal = amplitude * sp_signal.chirp(t, f0, duration, f1, method="quadratic")
        else:
            raise ValueError(f"Unknown chirp method: {method}")

        wav_bytes = _save_wav_bytes(signal, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "duration": duration,
            "samplerate": samplerate,
            "f0": f0,
            "f1": f1,
            "method": method,
        }

    @staticmethod
    def generate_noise(duration: float, noise_type: str = "white",
                       samplerate: int = SAMPLE_RATE, amplitude: float = 0.5) -> dict:
        """Generate noise (white, pink, brown)."""
        n_samples = int(samplerate * duration)
        if noise_type == "white":
            signal = amplitude * np.random.randn(n_samples)
        elif noise_type == "pink":
            # Pink noise via Voss-McCartney
            white = np.random.randn(n_samples)
            b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
            a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
            signal = amplitude * sp_signal.filtfilt(b, a, white)
        elif noise_type == "brown":
            white = np.random.randn(n_samples)
            signal = amplitude * np.cumsum(white)
            signal = _normalize(signal)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        wav_bytes = _save_wav_bytes(signal, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "duration": duration,
            "samplerate": samplerate,
            "noise_type": noise_type,
        }

    @staticmethod
    def generate_silence(duration: float, samplerate: int = SAMPLE_RATE) -> dict:
        """Generate silence."""
        n_samples = int(samplerate * duration)
        signal = np.zeros(n_samples)
        wav_bytes = _save_wav_bytes(signal, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "duration": duration,
            "samplerate": samplerate,
        }

    # ---- Signal Analysis --------------------------------------------------

    @staticmethod
    def spectrogram(audio_path: str | None = None, audio_data_b64: str | None = None,
                    samplerate: int = SAMPLE_RATE, nfft: int = 1024, hop_length: int | None = None,
                    title: str = "Spectrogram", return_image: bool = True) -> dict:
        """Generate a spectrogram as image (base64 PNG) or data."""
        if audio_data_b64:
            signal = _load_from_b64(audio_data_b64, samplerate)
        elif audio_path:
            signal, samplerate = _load_audio(audio_path)
        else:
            raise ValueError("Provide either audio_path or audio_data_b64")

        if hop_length is None:
            hop_length = nfft // 4

        freqs, times, Sxx = sp_signal.spectrogram(
            signal, samplerate, nperseg=nfft, noverlap=nfft - hop_length
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        result: dict[str, Any] = {
            "freq_range": [float(freqs[0]), float(freqs[-1])],
            "time_range": [float(times[0]), float(times[-1])],
            "shape": list(Sxx_db.shape),
        }

        if return_image and HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pcolormesh(times, freqs, Sxx_db, shading="gouraud")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_xlabel("Time [sec]")
            ax.set_title(title)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            result["image_png_base64"] = base64_encode(buf.getvalue())

        return result

    @staticmethod
    def fft_analysis(audio_path: str | None = None, audio_data_b64: str | None = None,
                     samplerate: int = SAMPLE_RATE, max_freq: float | None = None,
                     return_image: bool = True) -> dict:
        """FFT spectrum analysis."""
        if audio_data_b64:
            signal = _load_from_b64(audio_data_b64, samplerate)
        elif audio_path:
            signal, samplerate = _load_audio(audio_path)
        else:
            raise ValueError("Provide either audio_path or audio_data_b64")

        n = len(signal)
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, 1 / samplerate)
        magnitude = np.abs(fft) / n

        if max_freq:
            mask = freqs <= max_freq
            freqs = freqs[mask]
            magnitude = magnitude[mask]

        peaks = _find_peaks(freqs, magnitude, num_peaks=5)

        result: dict[str, Any] = {
            "sample_count": n,
            "samplerate": samplerate,
            "nyquist": samplerate / 2,
            "peaks": peaks,
        }

        if return_image and HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(freqs, magnitude)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Magnitude")
            ax.set_title("FFT Spectrum")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            result["image_png_base64"] = base64_encode(buf.getvalue())

        return result

    @staticmethod
    def goertzel(audio_path: str | None = None, audio_data_b64: str | None = None,
                 target_freq: float = 1000, samplerate: int = SAMPLE_RATE,
                 block_size: int | None = None) -> dict:
        """Goertzel algorithm for single-frequency detection."""
        if audio_data_b64:
            signal = _load_from_b64(audio_data_b64, samplerate)
        elif audio_path:
            signal, samplerate = _load_audio(audio_path)
        else:
            raise ValueError("Provide either audio_path or audio_data_b64")

        if block_size is None:
            block_size = len(signal)

        # Goertzel filter
        k = int(0.5 + (block_size * target_freq) / samplerate)
        omega = 2 * math.pi * k / block_size
        coeff = 2 * math.cos(omega)

        magnitude_per_block: list[float] = []
        for start in range(0, len(signal), block_size // 2):
            block = signal[start:start + block_size]
            if len(block) < block_size:
                block = np.pad(block, (0, block_size - len(block)))
            s_prev = 0.0
            s_prev2 = 0.0
            for sample in block:
                s = sample + coeff * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s
            power = s_prev2 ** 2 + s_prev ** 2 - coeff * s_prev * s_prev2
            magnitude_per_block.append(float(np.sqrt(power) / block_size))

        detected = max(magnitude_per_block) > 0.1
        return {
            "target_frequency": target_freq,
            "magnitude_per_block": magnitude_per_block,
            "max_magnitude": max(magnitude_per_block),
            "mean_magnitude": float(np.mean(magnitude_per_block)),
            "detected": bool(detected),
        }

    @staticmethod
    def correlation(audio_data_b64_ref: str, audio_data_b64_signal: str | None = None,
                    audio_path_signal: str | None = None, samplerate: int = SAMPLE_RATE) -> dict:
        """Cross-correlation for sync detection."""
        ref = _load_from_b64(audio_data_b64_ref, samplerate)
        if audio_data_b64_signal:
            signal = _load_from_b64(audio_data_b64_signal, samplerate)
        elif audio_path_signal:
            signal, samplerate = _load_audio(audio_path_signal)
        else:
            signal = ref.copy()

        corr = np.correlate(signal, ref, mode="valid")
        corr = corr / (np.linalg.norm(ref) + 1e-10)
        peak_idx = int(np.argmax(np.abs(corr)))
        peak_value = float(corr[peak_idx])

        return {
            "peak_index": peak_idx,
            "peak_value": peak_value,
            "peak_time_seconds": peak_idx / samplerate,
            "correlation_length": len(corr),
            "normalized": True,
        }

    # ---- Audio I/O --------------------------------------------------------

    @staticmethod
    def play_audio(audio_data_b64: str, samplerate: int = SAMPLE_RATE,
                   device: int | None = None, blocking: bool = True) -> dict:
        """Play audio through speakers."""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("sounddevice not available")
        signal = _load_from_b64(audio_data_b64, samplerate)
        sd.play(signal, samplerate, device=device)
        if blocking:
            sd.wait()
        return {"status": "played", "duration": len(signal) / samplerate}

    @staticmethod
    def record_audio(duration: float, samplerate: int = SAMPLE_RATE,
                     device: int | None = None, channels: int = 1) -> dict:
        """Record from microphone."""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("sounddevice not available")
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate,
                           channels=channels, device=device)
        sd.wait()
        if channels > 1:
            recording = recording.mean(axis=1)
        wav_bytes = _save_wav_bytes(recording, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "duration": duration,
            "samplerate": samplerate,
            "channels": channels,
        }

    # ---- File Operations --------------------------------------------------

    @staticmethod
    def load_audio(file_path: str) -> dict:
        """Load an audio file and return as WAV bytes."""
        signal, samplerate = _load_audio(file_path)
        duration = len(signal) / samplerate
        wav_bytes = _save_wav_bytes(signal, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "duration": duration,
            "samplerate": samplerate,
            "original_path": file_path,
        }

    @staticmethod
    def save_audio(audio_data_b64: str, file_path: str, samplerate: int = SAMPLE_RATE) -> dict:
        """Save audio data to file."""
        signal = _load_from_b64(audio_data_b64, samplerate)
        sf.write(file_path, signal, samplerate)
        return {
            "status": "saved",
            "path": file_path,
            "duration": len(signal) / samplerate,
            "samplerate": samplerate,
        }

    @staticmethod
    def trim(audio_data_b64: str, start_time: float, end_time: float,
             samplerate: int = SAMPLE_RATE) -> dict:
        """Trim audio to time range."""
        signal = _load_from_b64(audio_data_b64, samplerate)
        start_sample = int(start_time * samplerate)
        end_sample = int(end_time * samplerate)
        trimmed = signal[start_sample:end_sample]
        wav_bytes = _save_wav_bytes(trimmed, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "duration": len(trimmed) / samplerate,
            "samplerate": samplerate,
        }

    @staticmethod
    def concatenate(audio_list_b64: list[str], samplerates: list[int] | None = None) -> dict:
        """Concatenate multiple audio clips."""
        if samplerates is None:
            samplerates = [SAMPLE_RATE] * len(audio_list_b64)
        sr = samplerates[0]
        parts = [_load_from_b64(a, sr) for a in audio_list_b64]
        combined = np.concatenate(parts)
        wav_bytes = _save_wav_bytes(combined, sr)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "duration": len(combined) / sr,
            "samplerate": sr,
        }

    # ---- DSP Processing ---------------------------------------------------

    @staticmethod
    def normalize(audio_data_b64: str, samplerate: int = SAMPLE_RATE,
                  peak: float = 0.95) -> dict:
        """Normalize audio amplitude."""
        signal = _load_from_b64(audio_data_b64, samplerate)
        normalized = _normalize(signal) * peak
        wav_bytes = _save_wav_bytes(normalized, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "duration": len(normalized) / samplerate,
            "samplerate": samplerate,
        }

    @staticmethod
    def add_noise(audio_data_b64: str, snr_db: float = 20,
                  samplerate: int = SAMPLE_RATE) -> dict:
        """Add white noise at specified SNR."""
        signal = _load_from_b64(audio_data_b64, samplerate)
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        noisy = signal + noise
        wav_bytes = _save_wav_bytes(noisy, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "snr_db": snr_db,
            "duration": len(noisy) / samplerate,
            "samplerate": samplerate,
        }

    @staticmethod
    def filter_design(filter_type: str = "lowpass", cutoff_freq: float = 1000,
                      samplerate: int = SAMPLE_RATE, order: int = 5,
                      btype: str | None = None) -> dict:
        """Design a digital filter and return coefficients."""
        if btype is None:
            btype = filter_type  # 'lowpass', 'highpass', 'bandpass', 'bandstop'
        nyquist = samplerate / 2
        if isinstance(cutoff_freq, (int, float)):
            normal_cutoff = cutoff_freq / nyquist
            b, a = sp_signal.butter(order, normal_cutoff, btype=btype)
        else:
            raise ValueError("cutoff_freq must be a scalar")

        return {
            "b_coefficients": b.tolist(),
            "a_coefficients": a.tolist(),
            "filter_type": btype,
            "cutoff_freq": cutoff_freq,
            "order": order,
            "samplerate": samplerate,
        }

    @staticmethod
    def apply_filter(audio_data_b64: str, b: list[float], a: list[float],
                     samplerate: int = SAMPLE_RATE) -> dict:
        """Apply IIR/FIR filter to audio."""
        signal = _load_from_b64(audio_data_b64, samplerate)
        b_arr = np.array(b)
        a_arr = np.array(a)
        filtered = sp_signal.filtfilt(b_arr, a_arr, signal)
        wav_bytes = _save_wav_bytes(filtered, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "duration": len(filtered) / samplerate,
            "samplerate": samplerate,
        }

    @staticmethod
    def resample(audio_data_b64: str, orig_samplerate: int, target_samplerate: int) -> dict:
        """Resample audio to new sample rate."""
        signal = _load_from_b64(audio_data_b64, orig_samplerate)
        n_samples = int(len(signal) * target_samplerate / orig_samplerate)
        resampled = sp_signal.resample(signal, n_samples)
        wav_bytes = _save_wav_bytes(resampled, target_samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "duration": len(resampled) / target_samplerate,
            "samplerate": target_samplerate,
        }

    # ---- Modulation Utilities ---------------------------------------------

    @staticmethod
    def packet_encode(data_b64: str | None = None, preamble: str = "1010101010101010",
                      use_crc: bool = True, text: str | None = None) -> dict:
        """Encode data bytes into a packet with preamble and CRC.

        Provide either `data_b64` (base64 bytes) or `text` (encoded to UTF-8)."""
        if data_b64 is None:
            if text is None:
                return {"status": "error", "message": "Provide data_b64 or text"}
            data_bytes = text.encode("utf-8")
        else:
            data_bytes = _decode_b64(data_b64)
        bit_string = "".join(f"{b:08b}" for b in data_bytes)

        if use_crc:
            crc = _compute_crc16(data_bytes)
            crc_bits = f"{crc:016b}"
        else:
            crc_bits = ""

        packet_bits = preamble + bit_string + crc_bits
        return {
            "packet_bits": packet_bits,
            "payload_bits": bit_string,
            "crc_bits": crc_bits,
            "preamble_bits": preamble,
            "total_bits": len(packet_bits),
            "payload_bytes": len(data_bytes),
        }

    @staticmethod
    def packet_decode(packet_bits: str, preamble: str = "1010101010101010",
                      use_crc: bool = True) -> dict:
        """Decode a packet: detect preamble, extract data, verify CRC."""
        # Find preamble
        idx = packet_bits.find(preamble)
        if idx < 0:
            # Try with 1-bit error tolerance
            for i in range(len(packet_bits) - len(preamble) + 1):
                errors = sum(1 for j in range(len(preamble))
                             if packet_bits[i + j] != preamble[j])
                if errors <= 1:
                    idx = i
                    break

        if idx < 0:
            return {"status": "preamble_not_found", "packet_bits": packet_bits}

        start = idx + len(preamble)
        remaining = packet_bits[start:]

        if use_crc:
            if len(remaining) < 16:
                return {"status": "too_short", "packet_bits": packet_bits}
            # The bit stream may contain trailing garbage (e.g. decoded
            # silence) after the real packet, so the CRC is not necessarily
            # at the very end. Search payload lengths (whole bytes) and keep
            # the first one whose trailing 16 bits validate as CRC.
            best = None
            max_payload_bits = len(remaining) - 16
            # Limit search to a sane packet size to stay fast.
            max_payload_bits = min(max_payload_bits, 8 * 4096)
            for nbytes in range(0, max_payload_bits // 8 + 1):
                pb = remaining[: nbytes * 8]
                crc_bits = remaining[nbytes * 8: nbytes * 8 + 16]
                if len(crc_bits) < 16:
                    break
                received_crc = int(crc_bits, 2)
                if pb:
                    payload_bytes = int(pb, 2).to_bytes(nbytes, byteorder="big")
                else:
                    payload_bytes = b""
                if received_crc == _compute_crc16(payload_bytes):
                    best = (pb, payload_bytes, True)
                    break
            if best is None:
                # CRC never validated: fall back to taking all but last 16.
                payload_bits = remaining[:-16]
                received_crc = int(remaining[-16:], 2)
                if payload_bits:
                    payload_bytes = int(payload_bits, 2).to_bytes(
                        (len(payload_bits) + 7) // 8, byteorder="big"
                    )
                else:
                    payload_bytes = b""
                crc_ok = False
            else:
                payload_bits, payload_bytes, crc_ok = best
        else:
            payload_bits = remaining
            if payload_bits:
                payload_bytes = int(payload_bits, 2).to_bytes(
                    (len(payload_bits) + 7) // 8, byteorder="big"
                )
            else:
                payload_bytes = b""
            crc_ok = True

        return {
            "status": "ok" if crc_ok else "crc_error",
            "payload_base64": base64_encode(payload_bytes),
            "payload_bits": payload_bits,
            "payload_bytes": len(payload_bytes),
            "crc_ok": crc_ok,
            "preamble_offset": idx,
            "text": payload_bytes.decode("utf-8", errors="replace"),
        }

    @staticmethod
    def reed_solomon_encode(data_b64: str | None = None, nsym: int = 10,
                            text: str | None = None) -> dict:
        """Reed-Solomon FEC encode (requires reedsolo package).

        Provide either `data_b64` (base64 bytes) or `text` (encoded to UTF-8)."""
        try:
            from reedsolo import RSCodec
        except ImportError:
            return {"status": "error", "message": "reedsolo not installed. pip install reedsolo"}
        if data_b64 is None:
            if text is None:
                return {"status": "error", "message": "Provide data_b64 or text"}
            data = text.encode("utf-8")
        else:
            data = _decode_b64(data_b64)
        coder = RSCodec(nsym)
        encoded = coder.encode(data)
        return {
            "status": "ok",
            "encoded_base64": base64_encode(bytes(encoded)),
            "original_length": len(data),
            "encoded_length": len(encoded),
            "nsym": nsym,
        }

    @staticmethod
    def reed_solomon_decode(data_b64: str, nsym: int = 10) -> dict:
        """Reed-Solomon FEC decode."""
        try:
            from reedsolo import RSCodec
        except ImportError:
            return {"status": "error", "message": "reedsolo not installed. pip install reedsolo"}
        data = _decode_b64(data_b64)
        coder = RSCodec(nsym)
        try:
            decoded, ecc, errata = coder.decode(data)
            return {
                "status": "ok",
                "decoded_base64": base64_encode(bytes(decoded)),
                "decoded_length": len(decoded),
                "corrected_errors": len(errata),
                "text": bytes(decoded).decode("utf-8", errors="replace"),
            }
        except Exception as e:
            return {"status": "decode_failed", "message": str(e)}


    # ---- Advanced Modulation ------------------------------------------------

    @staticmethod
    def ask_modulate(bits: str, carrier_freq: float = 1000, baud_rate: float = 300,
                     samplerate: int = SAMPLE_RATE, amplitude: float = 0.9) -> dict:
        """ASK (Amplitude Shift Keying) modulate a binary string."""
        duration_per_bit = 1.0 / baud_rate
        samples_per_bit = int(samplerate * duration_per_bit)
        total_samples = len(bits) * samples_per_bit
        t = np.linspace(0, len(bits) * duration_per_bit, total_samples, endpoint=False)
        carrier = amplitude * np.sin(2 * np.pi * carrier_freq * t)
        envelope = np.zeros(total_samples)
        for i, bit in enumerate(bits):
            start = i * samples_per_bit
            end = (i + 1) * samples_per_bit
            level = 1.0 if bit in ("1", "1") else 0.15
            envelope[start:end] = level
        signal = carrier * envelope
        fade_len = min(int(samples_per_bit * 0.05), 10)
        if fade_len > 0:
            signal[:fade_len] *= np.linspace(0, 1, fade_len)
            signal[-fade_len:] *= np.linspace(1, 0, fade_len)
        wav_bytes = _save_wav_bytes(signal, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "bits": bits,
            "bit_count": len(bits),
            "carrier_freq": carrier_freq,
            "baud_rate": baud_rate,
            "duration": len(bits) * duration_per_bit,
            "samplerate": samplerate,
        }

    @staticmethod
    def psk_modulate(bits: str, carrier_freq: float = 1000, baud_rate: float = 300,
                     samplerate: int = SAMPLE_RATE, amplitude: float = 0.9) -> dict:
        """BPSK (Binary Phase Shift Keying) modulate a binary string."""
        duration_per_bit = 1.0 / baud_rate
        samples_per_bit = int(samplerate * duration_per_bit)
        total_samples = len(bits) * samples_per_bit
        t = np.linspace(0, len(bits) * duration_per_bit, total_samples, endpoint=False)
        signal = np.zeros(total_samples)
        for i, bit in enumerate(bits):
            start = i * samples_per_bit
            end = (i + 1) * samples_per_bit
            phase = 0.0 if bit in ("1", "1") else math.pi
            bit_t = t[start:end] - t[start]
            signal[start:end] = amplitude * np.sin(2 * np.pi * carrier_freq * bit_t + phase)
        # Soft transitions to reduce clicks
        for i in range(1, len(bits)):
            start = i * samples_per_bit
            fade_len = min(int(samples_per_bit * 0.08), 15)
            if fade_len > 0:
                signal[start:start + fade_len] *= np.linspace(0, 1, fade_len)
        wav_bytes = _save_wav_bytes(signal, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "bits": bits,
            "bit_count": len(bits),
            "carrier_freq": carrier_freq,
            "baud_rate": baud_rate,
            "duration": len(bits) * duration_per_bit,
            "samplerate": samplerate,
        }

    @staticmethod
    def line_code(bits: str, encoding: str = "nrz", baud_rate: float = 300,
                  samplerate: int = SAMPLE_RATE, amplitude: float = 1.0) -> dict:
        """Encode bits using line coding: nrz, manchester, diff_manchester."""
        duration_per_bit = 1.0 / baud_rate
        samples_per_bit = int(samplerate * duration_per_bit)
        total_samples = len(bits) * samples_per_bit

        encoding = encoding.lower().replace("-", "_")

        if encoding == "nrz":
            levels: list[float] = []
            for b in bits:
                levels.extend([amplitude if b == "1" else -amplitude] * samples_per_bit)
            signal = np.array(levels, dtype=np.float64)
        elif encoding == "manchester":
            signal = np.zeros(total_samples)
            half = samples_per_bit // 2
            for i, b in enumerate(bits):
                start = i * samples_per_bit
                if b == "1":
                    signal[start:start + half] = amplitude
                    signal[start + half:start + samples_per_bit] = -amplitude
                else:
                    signal[start:start + half] = -amplitude
                    signal[start + half:start + samples_per_bit] = amplitude
        elif encoding == "diff_manchester":
            signal = np.zeros(total_samples)
            half = samples_per_bit // 2
            last_level = amplitude
            for i, b in enumerate(bits):
                start = i * samples_per_bit
                if b == "0":
                    last_level = -last_level
                signal[start:start + half] = last_level
                signal[start + half:start + samples_per_bit] = -last_level
                last_level = -last_level
        else:
            raise ValueError(f"Unknown encoding: {encoding}. Use nrz, manchester, or diff_manchester")

        wav_bytes = _save_wav_bytes(signal, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "bits": bits,
            "encoding": encoding,
            "baud_rate": baud_rate,
            "duration": len(bits) * duration_per_bit,
            "samplerate": samplerate,
        }

    # ---- Advanced Analysis -------------------------------------------------

    @staticmethod
    def constellation_diagram(iq_data_b64: str | None = None,
                               iq_path: str | None = None,
                               title: str = "Constellation Diagram",
                               return_image: bool = True) -> dict:
        """Plot constellation diagram from I/Q samples."""
        if iq_data_b64:
            import base64 as _b64
            raw = _b64.b64decode(iq_data_b64)
            if len(raw) % np.dtype(np.complex128).itemsize != 0:
                return {
                    "status": "error",
                    "message": (
                        f"iq_data_b64 decodes to {len(raw)} bytes, which is not a "
                        f"multiple of the complex128 element size (16 bytes). "
                        "Provide raw base64-encoded complex128 I/Q samples."
                    ),
                }
            iq = np.frombuffer(raw, dtype=np.complex128)
            if iq.size == 0:
                return {"status": "error", "message": "iq_data_b64 contains no I/Q samples"}
        elif iq_path:
            data = np.load(iq_path)
            iq = data["iq"] if "iq" in data else data
        else:
            # Generate synthetic QPSK as demo
            symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex128)
            idx = np.random.randint(0, 4, 256)
            iq = symbols[idx] + 0.15 * np.random.randn(256)

        i_part = np.real(iq)
        q_part = np.imag(iq)
        iq_power = np.mean(np.abs(iq) ** 2)

        result: dict[str, Any] = {
            "num_symbols": len(iq),
            "mean_power": float(iq_power),
            "evm_percent": float(np.std(iq - np.mean(iq)) / np.sqrt(iq_power + 1e-10) * 100),
        }

        if return_image and HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(i_part, q_part, s=10, alpha=0.6)
            ax.axhline(0, color="gray", lw=0.5)
            ax.axvline(0, color="gray", lw=0.5)
            ax.set_xlabel("In-Phase (I)")
            ax.set_ylabel("Quadrature (Q)")
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            result["image_png_base64"] = base64_encode(buf.getvalue())

        return result

    @staticmethod
    def eye_diagram(audio_data_b64: str, symbol_rate: float,
                    samplerate: int = SAMPLE_RATE,
                    samples_per_symbol: int | None = None,
                    num_traces: int = 200,
                    title: str = "Eye Diagram",
                    return_image: bool = True) -> dict:
        """Generate an eye diagram for signal quality analysis."""
        signal = _load_from_b64(audio_data_b64, samplerate)
        if samples_per_symbol is None:
            samples_per_symbol = int(samplerate / symbol_rate)

        symbol_span = samples_per_symbol * 2
        traces: list[np.ndarray] = []
        stride = max(1, samples_per_symbol // 2)

        for i in range(0, len(signal) - symbol_span, stride):
            if len(traces) >= num_traces:
                break
            segment = signal[i:i + symbol_span]
            if len(segment) == symbol_span:
                traces.append(segment)

        result: dict[str, Any] = {
            "symbol_rate": symbol_rate,
            "samples_per_symbol": samples_per_symbol,
            "num_traces": len(traces),
        }

        if return_image and HAS_MATPLOTLIB and traces:
            fig, ax = plt.subplots(figsize=(10, 6))
            t = np.linspace(0, 2, symbol_span)
            for trace in traces:
                ax.plot(t, trace, color="blue", alpha=0.15, lw=0.5)
            ax.set_xlabel("Symbol Periods")
            ax.set_ylabel("Amplitude")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            result["image_png_base64"] = base64_encode(buf.getvalue())

        return result

    @staticmethod
    def ber_measure(tx_bits: str, rx_bits: str | None = None,
                    snr_range_db: list[float] | None = None,
                    num_bits: int = 1000) -> dict:
        """Measure BER between transmitted and received bits (direct or simulated)."""
        if rx_bits is not None:
            min_len = min(len(tx_bits), len(rx_bits))
            if min_len == 0:
                return {"error": "Empty bit strings"}
            errors = sum(1 for i in range(min_len) if tx_bits[i] != rx_bits[i])
            return {
                "mode": "direct",
                "bit_errors": errors,
                "total_bits": min_len,
                "ber": errors / min_len,
            }

        # Simulate BPSK BER over SNR range in AWGN
        if snr_range_db is None:
            snr_range_db = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

        import math as _math
        results: list[dict] = []
        for snr in snr_range_db:
            snr_linear = 10 ** (snr / 10)
            noise_std = 1.0 / np.sqrt(2 * snr_linear)
            tx_symbols = np.array([1 if b == "1" else -1 for b in tx_bits[:num_bits]], dtype=np.float64)
            noise = noise_std * np.random.randn(len(tx_symbols))
            rx_symbols = tx_symbols + noise
            rx_decoded = (rx_symbols >= 0).astype(np.int32)
            tx_int = (tx_symbols >= 0).astype(np.int32)
            errors = int(np.sum(rx_decoded != tx_int))
            results.append({
                "snr_db": snr,
                "errors": errors,
                "total_bits": len(tx_symbols),
                "ber": errors / len(tx_symbols) if len(tx_symbols) > 0 else 0,
            })

        theoretical = [0.5 * _math.erfc(_math.sqrt(10 ** (snr / 10))) for snr in snr_range_db]

        return {
            "mode": "simulated",
            "modulation": "bpsk",
            "results": results,
            "theoretical_ber": theoretical,
            "snr_range": snr_range_db,
        }

    # ---- Testing & Equalization --------------------------------------------

    @staticmethod
    def loopback_test(duration: float = 2.0, samplerate: int = SAMPLE_RATE,
                      device: int | None = None, chirp: bool = True) -> dict:
        """Play a test signal and record it back for delay/quality analysis."""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("sounddevice not available")

        if chirp:
            t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
            tx_audio = sp_signal.chirp(t, 200, duration, samplerate // 4, method="linear")
            tx_audio = _normalize(tx_audio)
        else:
            tx_audio = np.sin(2 * np.pi * 1000 * np.linspace(0, duration, int(samplerate * duration), endpoint=False))
            tx_audio = _normalize(tx_audio)

        recording = sd.playrec(tx_audio, samplerate, channels=1, device=device)
        sd.wait()
        recording = recording.flatten().astype(np.float64)

        corr = np.correlate(recording, tx_audio, mode="valid")
        peak_idx = int(np.argmax(np.abs(corr)))
        delay_sec = peak_idx / samplerate

        signal_power = np.mean(tx_audio ** 2)
        noise_power = np.mean((recording[:len(tx_audio)] - tx_audio) ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 1e-10 else float("inf")

        tx_wav = _save_wav_bytes(tx_audio, samplerate)
        rx_wav = _save_wav_bytes(_normalize(recording), samplerate)

        return {
            "format": "wav",
            "data": base64_encode(tx_wav),
            "rx_data": base64_encode(rx_wav),
            "delay_samples": peak_idx,
            "delay_seconds": delay_sec,
            "snr_db": snr_db,
            "duration": duration,
            "samplerate": samplerate,
        }

    @staticmethod
    def impulse_response(duration: float = 1.0, samplerate: int = SAMPLE_RATE,
                         method: str = "chirp",
                         device: int | None = None,
                         return_image: bool = True) -> dict:
        """Measure channel impulse response using chirp or MLS."""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("sounddevice not available")

        if method == "chirp":
            t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
            tx = sp_signal.chirp(t, 100, duration, samplerate // 4, method="linear")
            tx = _normalize(tx)
            recording = sd.playrec(tx, samplerate, channels=1, device=device)
            sd.wait()
            rx = recording.flatten().astype(np.float64)
            tx_rev = tx[::-1]
            ir = np.correlate(rx, tx_rev, mode="same") / (np.sum(tx ** 2) + 1e-10)
            ir = ir[:samplerate // 2]
        elif method == "mls":
            try:
                mls_bits = sp_signal.max_len_seq(10)[0]
                mls = mls_bits.flatten().astype(np.float64) * 2 - 1
            except AttributeError:
                n_len = 1023
                reg = 1
                mls = np.zeros(n_len, dtype=np.float64)
                for i in range(n_len):
                    mls[i] = 1.0 if (reg & 1) else -1.0
                    fb = ((reg >> 9) ^ (reg >> 6)) & 1
                    reg = (reg >> 1) | (fb << 9)
            n_mls = len(mls)
            repeats = max(1, int(samplerate * duration) // n_mls)
            tx = np.tile(mls, repeats)[:int(samplerate * duration)]
            recording = sd.playrec(tx, samplerate, channels=1, device=device)
            sd.wait()
            rx = recording.flatten().astype(np.float64)
            ir = np.correlate(rx, mls, mode="same") / (n_mls + 1e-10)
            ir = ir[:samplerate // 2]
        else:
            raise ValueError(f"Unknown method: {method}. Use 'chirp' or 'mls'")

        result: dict[str, Any] = {
            "method": method,
            "ir_length": len(ir),
            "samplerate": samplerate,
            "duration": len(ir) / samplerate,
            "peak_amplitude": float(np.max(np.abs(ir))),
            "peak_delay_ms": float(np.argmax(np.abs(ir)) / samplerate * 1000),
        }

        if return_image and HAS_MATPLOTLIB:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            t_ir = np.linspace(0, len(ir) / samplerate, len(ir))
            ax1.plot(t_ir * 1000, ir)
            ax1.set_xlabel("Time [ms]")
            ax1.set_ylabel("Amplitude")
            ax1.set_title(f"Impulse Response ({method})")
            ax1.grid(True, alpha=0.3)
            H = np.fft.rfft(ir)
            freqs = np.fft.rfftfreq(len(ir), 1 / samplerate)
            ax2.semilogx(freqs[1:], 20 * np.log10(np.abs(H[1:]) + 1e-10))
            ax2.set_xlabel("Frequency [Hz]")
            ax2.set_ylabel("Magnitude [dB]")
            ax2.set_title("Frequency Response")
            ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            result["image_png_base64"] = base64_encode(buf.getvalue())

        return result

    @staticmethod
    def equalize_lms(audio_data_b64: str, desired_b64: str | None = None,
                     filter_length: int = 32, step_size: float = 0.01,
                     samplerate: int = SAMPLE_RATE) -> dict:
        """LMS adaptive equalizer."""
        signal = _load_from_b64(audio_data_b64, samplerate)
        if desired_b64:
            desired = _load_from_b64(desired_b64, samplerate)
        else:
            desired = signal.copy()

        min_len = min(len(signal), len(desired))
        signal = signal[:min_len]
        desired = desired[:min_len]

        w = np.zeros(filter_length)
        error_history: list[float] = []
        output = np.zeros(min_len)

        for n in range(filter_length, min_len):
            x = signal[n - filter_length:n]
            y = np.dot(w, x)
            e = desired[n] - y
            w = w + 2 * step_size * e * x
            error_history.append(float(e ** 2))
            output[n] = y

        filtered = sp_signal.lfilter(w, [1.0], signal)
        mse = float(np.mean(error_history[-1000:])) if error_history else 0

        wav_bytes = _save_wav_bytes(_normalize(filtered), samplerate)
        result: dict[str, Any] = {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "filter_length": filter_length,
            "step_size": step_size,
            "converged_mse": mse,
            "iterations": len(error_history),
        }

        if HAS_MATPLOTLIB and error_history:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(error_history, lw=0.5)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Squared Error")
            ax.set_title("LMS Convergence")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            result["image_png_base64"] = base64_encode(buf.getvalue())

        return result

    # ---- Text helpers -----------------------------------------------------

    @staticmethod
    def text_encode(text: str, encoding: str = "utf-8") -> dict:
        """Encode text to base64 (helper for modem pipelines)."""
        data = text.encode(encoding)
        bits = "".join(f"{b:08b}" for b in data)
        return {
            "text": text,
            "data_b64": base64_encode(data),
            "bits": bits,
            "bytes": len(data),
            "encoding": encoding,
        }

    @staticmethod
    def text_decode(data_b64: str, encoding: str = "utf-8") -> dict:
        """Decode base64 data back to text."""
        data = _decode_b64(data_b64)
        text = data.decode(encoding, errors="replace")
        return {
            "text": text,
            "data_b64": data_b64,
            "bytes": len(data),
            "encoding": encoding,
        }

    # ---- Modem (end-to-end) -----------------------------------------------

    @staticmethod
    def fsk_demodulate(audio_data_b64: str, mark_freq: float = 1200,
                       space_freq: float = 2200, baud_rate: float = 300,
                       samplerate: int = SAMPLE_RATE,
                       threshold: float | None = None) -> dict:
        """Demodulate an FSK signal back to a bit string using Goertzel per bit."""
        signal = _load_from_b64(audio_data_b64, samplerate)
        samples_per_bit = int(samplerate / baud_rate)
        if samples_per_bit < 2:
            raise ValueError("baud_rate too high for samplerate")
        n_bits = len(signal) // samples_per_bit
        if n_bits == 0:
            return {"bits": "", "bit_count": 0, "samples_per_bit": samples_per_bit}

        def _goertzel_power(block: np.ndarray, freq: float) -> float:
            # Exact-frequency Goertzel (no DFT-bin rounding) — works even when
            # the bit window is short (high baud rate), where k-rounding would
            # collapse mark/space onto the same bin.
            omega = 2 * math.pi * freq / samplerate
            coeff = 2 * math.cos(omega)
            s_prev = 0.0
            s_prev2 = 0.0
            for sample in block:
                s = sample + coeff * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s
            return s_prev2 ** 2 + s_prev ** 2 - coeff * s_prev * s_prev2

        bits_chars: list[str] = []
        margins: list[float] = []
        for i in range(n_bits):
            block = signal[i * samples_per_bit:(i + 1) * samples_per_bit]
            p_mark = _goertzel_power(block, mark_freq)
            p_space = _goertzel_power(block, space_freq)
            bits_chars.append("1" if p_mark >= p_space else "0")
            total = p_mark + p_space
            margins.append(float(abs(p_mark - p_space) / total) if total > 0 else 0.0)

        bits = "".join(bits_chars)
        result: dict[str, Any] = {
            "bits": bits,
            "bit_count": n_bits,
            "samples_per_bit": samples_per_bit,
            "mark_freq": mark_freq,
            "space_freq": space_freq,
            "baud_rate": baud_rate,
            "mean_confidence": float(np.mean(margins)) if margins else 0.0,
            "min_confidence": float(np.min(margins)) if margins else 0.0,
        }
        if threshold is not None:
            low = sum(1 for m in margins if m < threshold)
            result["low_confidence_bits"] = low
        return result

    @staticmethod
    def modem_tx(text: str, mark_freq: float = 1200, space_freq: float = 2200,
                 baud_rate: float = 300, samplerate: int = SAMPLE_RATE,
                 use_crc: bool = True, use_fec: bool = False, fec_nsym: int = 10,
                 add_chirp_sync: bool = True, chirp_duration: float = 0.1,
                 pad_silence: float = 0.05) -> dict:
        """End-to-end text -> audio: text -> base64 -> (optional RS-FEC) -> packet -> FSK -> WAV."""
        data = text.encode("utf-8")
        if use_fec:
            try:
                from reedsolo import RSCodec
                data = bytes(RSCodec(fec_nsym).encode(data))
            except ImportError:
                return {"status": "error", "message": "reedsolo not installed. pip install reedsolo"}
        data_b64 = base64_encode(data)
        pkt = AudioDSPTools.packet_encode(data_b64, use_crc=use_crc)
        bits = pkt["packet_bits"]
        fsk = AudioDSPTools.generate_fsk(bits, mark_freq=mark_freq,
                                         space_freq=space_freq,
                                         baud_rate=baud_rate, samplerate=samplerate)
        body = _load_from_b64(fsk["data"], samplerate)

        parts: list[np.ndarray] = []
        if pad_silence > 0:
            parts.append(np.zeros(int(samplerate * pad_silence)))
        if add_chirp_sync:
            t = np.linspace(0, chirp_duration, int(samplerate * chirp_duration), endpoint=False)
            chirp_sig = _normalize(sp_signal.chirp(t, mark_freq * 0.5, chirp_duration,
                                                   space_freq * 1.5, method="linear"))
            parts.append(chirp_sig)
            parts.append(np.zeros(int(samplerate * pad_silence)))
        parts.append(body)
        if pad_silence > 0:
            parts.append(np.zeros(int(samplerate * pad_silence)))
        full = np.concatenate(parts) if parts else body

        wav_bytes = _save_wav_bytes(full, samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "text": text,
            "data_b64": data_b64,
            "packet_bits": bits,
            "bit_count": len(bits),
            "duration": len(full) / samplerate,
            "samplerate": samplerate,
            "baud_rate": baud_rate,
            "mark_freq": mark_freq,
            "space_freq": space_freq,
            "use_crc": use_crc,
            "use_fec": use_fec,
            "chirp_sync": add_chirp_sync,
            "chirp_duration": chirp_duration if add_chirp_sync else 0,
        }

    @staticmethod
    def modem_rx(audio_data_b64: str, mark_freq: float = 1200, space_freq: float = 2200,
                 baud_rate: float = 300, samplerate: int = SAMPLE_RATE,
                 use_crc: bool = True, use_fec: bool = False, fec_nsym: int = 10,
                 sync_chirp: bool = True, chirp_duration: float = 0.1,
                 preamble: str = "1010101010101010") -> dict:
        """End-to-end audio -> text: chirp sync -> FSK demod -> packet decode -> (optional RS-FEC) -> text."""
        signal = _load_from_b64(audio_data_b64, samplerate)
        samples_per_bit = int(samplerate / baud_rate)
        offset_samples = 0
        sync_detected = False

        if sync_chirp and chirp_duration > 0:
            t = np.linspace(0, chirp_duration, int(samplerate * chirp_duration), endpoint=False)
            chirp_ref = _normalize(sp_signal.chirp(t, mark_freq * 0.5, chirp_duration,
                                                   space_freq * 1.5, method="linear"))
            search_len = min(len(signal), int(samplerate * 2))
            corr = np.correlate(signal[:search_len], chirp_ref, mode="valid")
            if len(corr) > 0:
                peak = int(np.argmax(np.abs(corr)))
                offset_samples = peak + len(chirp_ref)
                sync_detected = True

        body = signal[offset_samples:]
        n_bits = len(body) // samples_per_bit
        body = body[:n_bits * samples_per_bit]
        if n_bits == 0:
            return {"status": "no_signal", "text": "", "sync_detected": sync_detected}

        # Bit-boundary phase alignment: the chirp-correlation offset may be a
        # few samples off, which smears mark/space energy across adjacent bits.
        # Scan sub-bit shifts and keep the one with the most decisive preamble.
        def _boundary_score(shift: int) -> float:
            seg = body[shift:shift + len(preamble) * samples_per_bit]
            if len(seg) < len(preamble) * samples_per_bit:
                return -1.0
            score = 0.0
            for bi, want in enumerate(preamble):
                blk = seg[bi * samples_per_bit:(bi + 1) * samples_per_bit]
                om_m = 2 * math.pi * mark_freq / samplerate
                om_s = 2 * math.pi * space_freq / samplerate
                def gp(o: float) -> float:
                    c = 2 * math.cos(o)
                    s1 = s2 = 0.0
                    for x in blk:
                        s0 = x + c * s1 - s2
                        s2, s1 = s1, s0
                    return s2 * s2 + s1 * s1 - c * s1 * s2
                pm, ps = gp(om_m), gp(om_s)
                denom = pm + ps
                if denom <= 0:
                    continue
                score += (pm - ps) / denom if want == "1" else (ps - pm) / denom
            return score

        scan = min(samples_per_bit, 64)
        best_shift = max(range(scan), key=_boundary_score) if scan > 1 else 0
        if best_shift:
            body = body[best_shift:]
            n_bits = len(body) // samples_per_bit
            body = body[:n_bits * samples_per_bit]
            offset_samples += best_shift

        import base64 as _b64mod
        buf = io.BytesIO()
        sf.write(buf, body, samplerate, format="WAV", subtype="FLOAT")
        demod = AudioDSPTools.fsk_demodulate(
            _b64mod.b64encode(buf.getvalue()).decode("ascii"),
            mark_freq=mark_freq, space_freq=space_freq,
            baud_rate=baud_rate, samplerate=samplerate,
        )
        bits = demod["bits"]

        pkt = AudioDSPTools.packet_decode(bits, preamble=preamble, use_crc=use_crc)
        if pkt.get("status") not in ("ok", "crc_error"):
            return {
                "status": pkt.get("status", "decode_failed"),
                "text": "",
                "bits": bits,
                "sync_detected": sync_detected,
                "offset_samples": offset_samples,
                "mean_confidence": demod.get("mean_confidence", 0.0),
            }

        payload = _decode_b64(pkt["payload_base64"])
        fec_errors = 0
        if use_fec:
            try:
                from reedsolo import RSCodec
                decoded, _, errata = RSCodec(fec_nsym).decode(payload)
                payload = bytes(decoded)
                fec_errors = len(errata)
            except ImportError:
                return {"status": "error", "message": "reedsolo not installed. pip install reedsolo"}
            except Exception as e:
                return {"status": "fec_failed", "message": str(e), "text": ""}

        text = payload.decode("utf-8", errors="replace")
        return {
            "status": "ok" if pkt.get("crc_ok", True) else "crc_error",
            "text": text,
            "crc_ok": pkt.get("crc_ok", True),
            "fec_corrected_errors": fec_errors,
            "sync_detected": sync_detected,
            "offset_samples": offset_samples,
            "bit_count": len(bits),
            "mean_confidence": demod.get("mean_confidence", 0.0),
            "min_confidence": demod.get("min_confidence", 0.0),
        }

    # ---- Advanced modem (M-FSK, channel sim, BER curve, quality) -----------

    @staticmethod
    def mfsk_modulate(bits: str, num_tones: int = 4, base_freq: float = 800,
                      tone_spacing: float = 400, baud_rate: float = 300,
                      samplerate: int = SAMPLE_RATE, amplitude: float = 0.9) -> dict:
        """Modulate a bit string with M-FSK (4/8/16 tones -> 2/3/4 bits per symbol).

        Higher bits-per-symbol than binary FSK => higher bitrate at the same
        symbol rate (competition plan: 300-1200 bps)."""
        num_tones = int(num_tones)
        if num_tones not in (4, 8, 16):
            return {"status": "error", "message": "num_tones must be 4, 8 or 16"}
        if not bits:
            return {"status": "error", "message": "bits must not be empty"}
        if any(c not in "01" for c in bits):
            return {"status": "error", "message": "bits must contain only '0' and '1' characters"}
        bits_per_symbol = int(round(math.log2(num_tones)))
        # pad bits to a whole number of symbols
        pad = (-len(bits)) % bits_per_symbol
        padded = bits + "0" * pad
        n_symbols = len(padded) // bits_per_symbol

        tones = [base_freq + i * tone_spacing for i in range(num_tones)]
        samples_per_symbol = max(2, int(samplerate / baud_rate))
        signal = np.zeros(n_symbols * samples_per_symbol)
        for s in range(n_symbols):
            chunk = padded[s * bits_per_symbol:(s + 1) * bits_per_symbol]
            sym = int(chunk, 2)
            freq = tones[sym]
            start = s * samples_per_symbol
            t = np.arange(samples_per_symbol) / samplerate
            seg = amplitude * np.sin(2 * np.pi * freq * t)
            fade = min(int(samples_per_symbol * 0.05), 10)
            if fade > 0:
                seg[:fade] *= np.linspace(0, 1, fade)
                seg[-fade:] *= np.linspace(1, 0, fade)
            signal[start:start + samples_per_symbol] = seg

        wav_bytes = _save_wav_bytes(signal, samplerate)
        duration = n_symbols / baud_rate
        result: dict[str, Any] = {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "bits": bits,
            "bit_count": len(bits),
            "padded_bits": padded,
            "num_tones": num_tones,
            "bits_per_symbol": bits_per_symbol,
            "symbol_count": n_symbols,
            "tones_hz": tones,
            "duration": duration,
            "samplerate": samplerate,
            "baud_rate": baud_rate,
            "bitrate_bps": bits_per_symbol * baud_rate,
        }
        # Non-coherent (energy) detectors need tone spacing >= 1/T_symbol to keep
        # the tones orthogonal; below that, adjacent tones leak into each other
        # and the demodulator will misdecode even on a clean channel.
        if tone_spacing < baud_rate:
            result["design_warnings"] = [
                (f"tone_spacing ({tone_spacing} Hz) < baud_rate ({baud_rate} Hz): "
                 f"tones are non-orthogonal at {baud_rate} baud (spacing/baud = "
                 f"{tone_spacing / baud_rate:.2f} < 1). A non-coherent detector will "
                 f"misdecode. Use tone_spacing >= baud_rate or lower the baud_rate.")
            ]
        return result

    @staticmethod
    def mfsk_demodulate(audio_data_b64: str, num_tones: int = 4,
                        base_freq: float = 800, tone_spacing: float = 400,
                        baud_rate: float = 300, samplerate: int = SAMPLE_RATE) -> dict:
        """Demodulate an M-FSK signal back to a bit string (Goertzel per symbol)."""
        num_tones = int(num_tones)
        if num_tones not in (4, 8, 16):
            return {"status": "error", "message": "num_tones must be 4, 8 or 16"}
        bits_per_symbol = int(round(math.log2(num_tones)))
        signal = _load_from_b64(audio_data_b64, samplerate)
        samples_per_symbol = max(2, int(samplerate / baud_rate))
        n_symbols = len(signal) // samples_per_symbol
        if n_symbols == 0:
            return {"bits": "", "bit_count": 0, "symbol_count": 0}
        tones = [base_freq + i * tone_spacing for i in range(num_tones)]

        def gp(block: np.ndarray, freq: float) -> float:
            omega = 2 * math.pi * freq / samplerate
            coeff = 2 * math.cos(omega)
            s_prev = 0.0
            s_prev2 = 0.0
            for sample in block:
                s = sample + coeff * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s
            return s_prev2 ** 2 + s_prev ** 2 - coeff * s_prev * s_prev2

        bits_chars: list[str] = []
        margins: list[float] = []
        for s in range(n_symbols):
            block = signal[s * samples_per_symbol:(s + 1) * samples_per_symbol]
            powers = [gp(block, f) for f in tones]
            best = int(np.argmax(powers))
            bits_chars.append(format(best, f"0{bits_per_symbol}b"))
            ordered = sorted(powers, reverse=True)
            total = sum(powers)
            margins.append(float((ordered[0] - ordered[1]) / total) if total > 0 and len(ordered) > 1 else 0.0)

        bits = "".join(bits_chars)
        result: dict[str, Any] = {
            "bits": bits,
            "bit_count": len(bits),
            "symbol_count": n_symbols,
            "num_tones": num_tones,
            "bits_per_symbol": bits_per_symbol,
            "tones_hz": tones,
            "mean_confidence": float(np.mean(margins)) if margins else 0.0,
            "min_confidence": float(np.min(margins)) if margins else 0.0,
        }
        if tone_spacing < baud_rate:
            result["design_warnings"] = [
                (f"tone_spacing ({tone_spacing} Hz) < baud_rate ({baud_rate} Hz): "
                 f"tones are non-orthogonal at {baud_rate} baud; expect symbol errors. "
                 f"Use tone_spacing >= baud_rate or lower the baud_rate.")
            ]
        return result

    @staticmethod
    def channel_simulate(audio_data_b64: str, snr_db: float = 20.0,
                         reverb_amount: float = 0.0, distance_attenuation: float = 1.0,
                         dropout_prob: float = 0.0, dropout_len_ms: float = 20.0,
                         samplerate: int = SAMPLE_RATE, seed: int | None = None) -> dict:
        """Simulate an acoustic channel: AWGN + simple reverb + distance loss + dropouts.

        Lets you test the modem's robustness without a real speaker/mic."""
        if seed is not None:
            np.random.seed(seed)
        signal = _load_from_b64(audio_data_b64, samplerate).astype(np.float64)

        # Distance attenuation (linear gain; 1.0 = no loss)
        signal = signal * max(0.0, distance_attenuation)

        # Reverb: a few exponentially decaying echoes
        if reverb_amount > 0:
            reverb_amount = min(1.0, reverb_amount)
            delays_ms = [30, 70, 120]
            for i, d_ms in enumerate(delays_ms):
                d = int(samplerate * d_ms / 1000)
                gain = reverb_amount * (0.6 ** (i + 1))
                if d < len(signal):
                    signal[d:] += gain * signal[:-d]

        # Dropouts: zero out random short gaps
        if dropout_prob > 0:
            if dropout_len_ms <= 0:
                return {"status": "error",
                        "message": "dropout_len_ms must be > 0 when dropout_prob > 0"}
            block_len = max(1, int(samplerate * dropout_len_ms / 1000))
            n_blocks = max(1, len(signal) // block_len)
            for b in range(n_blocks):
                if np.random.rand() < dropout_prob:
                    start = b * block_len
                    signal[start:start + block_len] = 0.0

        # AWGN to hit target SNR (measured on the processed signal power)
        sig_power = float(np.mean(signal ** 2))
        if sig_power > 0:
            noise_power = sig_power / (10 ** (snr_db / 10))
            signal = signal + np.random.randn(len(signal)) * np.sqrt(noise_power)

        measured_snr = 10 * math.log10(sig_power / (sig_power / (10 ** (snr_db / 10)))) if sig_power > 0 else 0.0
        wav_bytes = _save_wav_bytes(_normalize(signal), samplerate)
        return {
            "format": "wav",
            "data": base64_encode(wav_bytes),
            "snr_db": snr_db,
            "measured_snr_db": round(measured_snr, 2),
            "reverb_amount": reverb_amount,
            "distance_attenuation": distance_attenuation,
            "dropout_prob": dropout_prob,
            "samplerate": samplerate,
            "duration": len(signal) / samplerate,
        }

    @staticmethod
    def ber_curve(text: str = "Hello, acoustic modem!",
                  snr_list_db: list[float] | None = None,
                  mode: str = "fsk", num_tones: int = 4,
                  baud_rate: float = 300, mark_freq: float = 1200,
                  space_freq: float = 2200, base_freq: float = 800,
                  tone_spacing: float = 400, samplerate: int = SAMPLE_RATE,
                  return_image: bool = True) -> dict:
        """Sweep SNR and measure real BER of the FSK/M-FSK modem chain.

        This is the core competition metric: BER vs SNR. Returns per-SNR BER and
        a PNG plot."""
        if not text:
            return {"status": "error", "message": "text must not be empty"}
        if snr_list_db is None:
            snr_list_db = [0, 4, 8, 12, 16, 20]
        bits = "".join(format(b, "08b") for b in text.encode("utf-8"))

        curve: list[dict] = []
        for snr in snr_list_db:
            if mode == "mfsk":
                tx = AudioDSPTools.mfsk_modulate(bits, num_tones=num_tones,
                                                 base_freq=base_freq,
                                                 tone_spacing=tone_spacing,
                                                 baud_rate=baud_rate,
                                                 samplerate=samplerate)
                ref_bits = tx["padded_bits"]
                chan = AudioDSPTools.channel_simulate(tx["data"], snr_db=snr,
                                                      samplerate=samplerate, seed=42)
                rx = AudioDSPTools.mfsk_demodulate(chan["data"], num_tones=num_tones,
                                                   base_freq=base_freq,
                                                   tone_spacing=tone_spacing,
                                                   baud_rate=baud_rate,
                                                   samplerate=samplerate)
            else:
                tx = AudioDSPTools.generate_fsk(bits, mark_freq=mark_freq,
                                                space_freq=space_freq,
                                                baud_rate=baud_rate,
                                                samplerate=samplerate)
                ref_bits = bits
                chan = AudioDSPTools.channel_simulate(tx["data"], snr_db=snr,
                                                      samplerate=samplerate, seed=42)
                rx = AudioDSPTools.fsk_demodulate(chan["data"], mark_freq=mark_freq,
                                                  space_freq=space_freq,
                                                  baud_rate=baud_rate,
                                                  samplerate=samplerate)
            rx_bits = rx["bits"][:len(ref_bits)]
            n = min(len(ref_bits), len(rx_bits))
            errors = sum(1 for i in range(n) if ref_bits[i] != rx_bits[i]) if n else 0
            curve.append({
                "snr_db": snr,
                "bit_errors": errors,
                "total_bits": n,
                "ber": (errors / n) if n else 0.0,
                "mean_confidence": rx.get("mean_confidence", 0.0),
            })

        result: dict[str, Any] = {
            "mode": mode,
            "num_tones": num_tones if mode == "mfsk" else 2,
            "text": text,
            "baud_rate": baud_rate,
            "bitrate_bps": (int(round(math.log2(num_tones))) * baud_rate) if mode == "mfsk" else baud_rate,
            "curve": curve,
        }
        if mode == "mfsk" and tone_spacing < baud_rate:
            result["design_warnings"] = [
                (f"tone_spacing ({tone_spacing} Hz) < baud_rate ({baud_rate} Hz): "
                 f"non-orthogonal M-FSK tones; BER will be high even at high SNR.")
            ]
        if return_image and HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy([c["snr_db"] for c in curve],
                        [max(c["ber"], 1e-6) for c in curve], "o-", lw=2)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel("BER")
            ax.set_title(f"BER vs SNR — {mode.upper()} modem @ {baud_rate} baud")
            ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            result["image_png_base64"] = base64_encode(buf.getvalue())
        return result

    @staticmethod
    def modem_quality(audio_data_b64: str, reference_text: str | None = None,
                      mode: str = "fsk", num_tones: int = 4,
                      baud_rate: float = 300, mark_freq: float = 1200,
                      space_freq: float = 2200, base_freq: float = 800,
                      tone_spacing: float = 400, samplerate: int = SAMPLE_RATE) -> dict:
        """Quality report for a received modem signal: SNR, confidence, BER (if a
        reference text is given) and estimated throughput. For the demo/judges."""
        signal = _load_from_b64(audio_data_b64, samplerate)

        if mode == "mfsk":
            dem = AudioDSPTools.mfsk_demodulate(audio_data_b64, num_tones=num_tones,
                                                base_freq=base_freq,
                                                tone_spacing=tone_spacing,
                                                baud_rate=baud_rate,
                                                samplerate=samplerate)
            bitrate = int(round(math.log2(num_tones))) * baud_rate
        else:
            dem = AudioDSPTools.fsk_demodulate(audio_data_b64, mark_freq=mark_freq,
                                               space_freq=space_freq,
                                               baud_rate=baud_rate, samplerate=samplerate)
            bitrate = baud_rate

        # Estimate SNR from the strongest tone vs the rest of the band
        n = len(signal)
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1 / samplerate)
        tone_list = ([mark_freq, space_freq] if mode != "mfsk"
                     else [base_freq + i * tone_spacing for i in range(num_tones)])
        band_mask = np.zeros_like(spectrum, dtype=bool)
        for f in tone_list:
            band_mask |= (freqs > f - tone_spacing / 2) & (freqs < f + tone_spacing / 2)
        # peak tone power vs the median out-of-tone noise floor. Using the median
        # (not total out-of-band power) avoids counting the strong tones as
        # "noise" on a clean signal, which previously gave a bogus negative SNR.
        tone_power = float(np.max(spectrum[band_mask] ** 2)) if np.any(band_mask) else 0.0
        noise_power = float(np.median(spectrum[~band_mask] ** 2)) if np.any(~band_mask) else 0.0
        est_snr = 10 * math.log10(tone_power / noise_power) if noise_power > 0 else 60.0

        report: dict[str, Any] = {
            "mode": mode,
            "estimated_snr_db": round(est_snr, 2),
            "mean_confidence": dem.get("mean_confidence", 0.0),
            "min_confidence": dem.get("min_confidence", 0.0),
            "bit_count": dem.get("bit_count", 0),
            "duration_s": n / samplerate,
            "bitrate_bps": bitrate,
            "throughput_bytes_per_s": round(bitrate / 8.0, 1),
        }
        if reference_text is not None:
            ref_bits = "".join(format(b, "08b") for b in reference_text.encode("utf-8"))
            rx_bits = dem["bits"]
            m = min(len(ref_bits), len(rx_bits))
            errors = sum(1 for i in range(m) if ref_bits[i] != rx_bits[i]) if m else 0
            report["reference_text"] = reference_text
            report["bit_errors"] = errors
            report["ber"] = (errors / m) if m else 0.0
            report["message_recovered"] = errors == 0
        return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def base64_encode(data: bytes) -> str:
    return base64_encode_impl(data)


def base64_encode_impl(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


def _decode_b64(data_b64: str) -> bytes:
    import base64
    return base64.b64decode(data_b64)


def _load_from_b64(data_b64: str, samplerate: int) -> np.ndarray:
    """Decode base64 WAV and return float64 signal."""
    import base64
    raw = base64.b64decode(data_b64)
    buf = io.BytesIO(raw)
    try:
        signal, sr = sf.read(buf)
        if sr != samplerate:
            # resample
            n = int(len(signal) * samplerate / sr)
            signal = sp_signal.resample(signal, n)
        return signal.astype(np.float64)
    except Exception:
        # Try as raw float64 array
        return np.frombuffer(raw, dtype=np.float64)


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio from file, return (signal, samplerate)."""
    signal, samplerate = sf.read(path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return signal.astype(np.float64), samplerate


def _compute_crc16(data: bytes) -> int:
    """CRC-16-CCITT."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = (crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1
        crc &= 0xFFFF
    return crc


def _find_peaks(freqs: np.ndarray, magnitude: np.ndarray,
                num_peaks: int = 5) -> list[dict]:
    """Find the largest spectral peaks."""
    indices = np.argsort(magnitude)[-num_peaks:]
    indices = indices[::-1]
    return [
        {"frequency": float(freqs[i]), "magnitude": float(magnitude[i])}
        for i in indices if magnitude[i] > 0
    ]


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

# Tool registry
TOOLS: dict[str, dict] = {
    "generate_fsk": {
        "name": "generate_fsk",
        "description": "Generate FSK modulated audio signal from binary string",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bits": {"type": "string", "description": "Binary string (e.g., '10110010')"},
                "mark_freq": {"type": "number", "description": "Mark frequency (Hz)", "default": 1200},
                "space_freq": {"type": "number", "description": "Space frequency (Hz)", "default": 2200},
                "baud_rate": {"type": "number", "description": "Baud rate (bps)", "default": 300},
                "amplitude": {"type": "number", "description": "Signal amplitude (0-1)", "default": 0.9},
            },
            "required": ["bits"],
        },
    },
    "generate_tone": {
        "name": "generate_tone",
        "description": "Generate a sine wave tone",
        "inputSchema": {
            "type": "object",
            "properties": {
                "frequency": {"type": "number", "description": "Frequency in Hz"},
                "duration": {"type": "number", "description": "Duration in seconds"},
                "amplitude": {"type": "number", "description": "Amplitude (0-1)", "default": 0.9},
            },
            "required": ["frequency", "duration"],
        },
    },
    "generate_chirp": {
        "name": "generate_chirp",
        "description": "Generate a frequency sweep (chirp) for synchronization",
        "inputSchema": {
            "type": "object",
            "properties": {
                "f0": {"type": "number", "description": "Start frequency (Hz)"},
                "f1": {"type": "number", "description": "End frequency (Hz)"},
                "duration": {"type": "number", "description": "Duration in seconds"},
                "method": {"type": "string", "enum": ["linear", "logarithmic", "quadratic"], "default": "linear"},
            },
            "required": ["f0", "f1", "duration"],
        },
    },
    "generate_noise": {
        "name": "generate_noise",
        "description": "Generate noise (white/pink/brown) for testing",
        "inputSchema": {
            "type": "object",
            "properties": {
                "duration": {"type": "number", "description": "Duration in seconds"},
                "noise_type": {"type": "string", "enum": ["white", "pink", "brown"], "default": "white"},
            },
            "required": ["duration"],
        },
    },
    "generate_silence": {
        "name": "generate_silence",
        "description": "Generate silence padding",
        "inputSchema": {
            "type": "object",
            "properties": {
                "duration": {"type": "number", "description": "Duration in seconds"},
            },
            "required": ["duration"],
        },
    },
    "spectrogram": {
        "name": "spectrogram",
        "description": "Generate spectrogram from audio (base64 or file path)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string", "description": "Base64-encoded WAV audio"},
                "audio_path": {"type": "string", "description": "Path to audio file"},
                "nfft": {"type": "integer", "description": "FFT size", "default": 1024},
                "return_image": {"type": "boolean", "description": "Return PNG image", "default": True},
            },
        },
    },
    "fft_analysis": {
        "name": "fft_analysis",
        "description": "FFT spectrum analysis",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "audio_path": {"type": "string"},
                "max_freq": {"type": "number", "description": "Max frequency to display"},
                "return_image": {"type": "boolean", "default": True},
            },
        },
    },
    "goertzel": {
        "name": "goertzel",
        "description": "Goertzel algorithm for single-frequency detection",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "audio_path": {"type": "string"},
                "target_freq": {"type": "number", "description": "Target frequency (Hz)"},
            },
            "required": ["target_freq"],
        },
    },
    "correlation": {
        "name": "correlation",
        "description": "Cross-correlation for sync/preamble detection",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64_ref": {"type": "string", "description": "Reference signal (preamble)"},
                "audio_data_b64_signal": {"type": "string", "description": "Signal to search in"},
                "audio_path_signal": {"type": "string", "description": "Or path to signal file"},
            },
            "required": ["audio_data_b64_ref"],
        },
    },
    "play_audio": {
        "name": "play_audio",
        "description": "Play audio through speakers (requires sounddevice)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string", "description": "Base64-encoded WAV audio"},
                "blocking": {"type": "boolean", "description": "Wait for playback to finish", "default": True},
            },
            "required": ["audio_data_b64"],
        },
    },
    "record_audio": {
        "name": "record_audio",
        "description": "Record audio from microphone",
        "inputSchema": {
            "type": "object",
            "properties": {
                "duration": {"type": "number", "description": "Recording duration in seconds"},
                "samplerate": {"type": "integer", "default": 44100},
            },
            "required": ["duration"],
        },
    },
    "load_audio": {
        "name": "load_audio",
        "description": "Load audio file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to audio file"},
            },
            "required": ["file_path"],
        },
    },
    "save_audio": {
        "name": "save_audio",
        "description": "Save audio to file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "file_path": {"type": "string"},
                "samplerate": {"type": "integer", "default": 44100},
            },
            "required": ["audio_data_b64", "file_path"],
        },
    },
    "trim": {
        "name": "trim",
        "description": "Trim audio to time range",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"},
            },
            "required": ["audio_data_b64", "start_time", "end_time"],
        },
    },
    "concatenate": {
        "name": "concatenate",
        "description": "Concatenate audio clips",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_list_b64": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["audio_list_b64"],
        },
    },
    "normalize": {
        "name": "normalize",
        "description": "Normalize audio amplitude",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "peak": {"type": "number", "default": 0.95},
            },
            "required": ["audio_data_b64"],
        },
    },
    "add_noise": {
        "name": "add_noise",
        "description": "Add white noise at specified SNR for testing",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "snr_db": {"type": "number", "description": "SNR in dB", "default": 20},
            },
            "required": ["audio_data_b64"],
        },
    },
    "filter_design": {
        "name": "filter_design",
        "description": "Design digital filter (Butterworth)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filter_type": {"type": "string", "enum": ["lowpass", "highpass", "bandpass", "bandstop"], "default": "lowpass"},
                "cutoff_freq": {"type": "number", "description": "Cutoff frequency (Hz)"},
                "samplerate": {"type": "integer", "default": 44100},
                "order": {"type": "integer", "default": 5},
            },
            "required": ["cutoff_freq"],
        },
    },
    "apply_filter": {
        "name": "apply_filter",
        "description": "Apply filter to audio",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "b": {"type": "array", "items": {"type": "number"}, "description": "Filter numerator coefficients"},
                "a": {"type": "array", "items": {"type": "number"}, "description": "Filter denominator coefficients"},
            },
            "required": ["audio_data_b64", "b", "a"],
        },
    },
    "resample": {
        "name": "resample",
        "description": "Resample audio",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "orig_samplerate": {"type": "integer"},
                "target_samplerate": {"type": "integer"},
            },
            "required": ["audio_data_b64", "orig_samplerate", "target_samplerate"],
        },
    },
    "packet_encode": {
        "name": "packet_encode",
        "description": "Encode data bytes into a packet with preamble and CRC",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data_b64": {"type": "string", "description": "Base64-encoded data bytes (optional if text given)"},
                "text": {"type": "string", "description": "Plain text to encode (optional if data_b64 given)"},
                "preamble": {"type": "string", "default": "1010101010101010"},
                "use_crc": {"type": "boolean", "default": True},
            },
        },
    },
    "packet_decode": {
        "name": "packet_decode",
        "description": "Decode a packet: detect preamble, extract data, verify CRC",
        "inputSchema": {
            "type": "object",
            "properties": {
                "packet_bits": {"type": "string", "description": "Bit string with preamble + payload + CRC"},
                "preamble": {"type": "string", "default": "1010101010101010"},
                "use_crc": {"type": "boolean", "default": True},
            },
            "required": ["packet_bits"],
        },
    },
    "reed_solomon_encode": {
        "name": "reed_solomon_encode",
        "description": "Reed-Solomon FEC encode (requires reedsolo)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data_b64": {"type": "string", "description": "Base64-encoded data bytes (optional if text given)"},
                "text": {"type": "string", "description": "Plain text to encode (optional if data_b64 given)"},
                "nsym": {"type": "integer", "default": 10},
            },
        },
    },
    "reed_solomon_decode": {
        "name": "reed_solomon_decode",
        "description": "Reed-Solomon FEC decode",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data_b64": {"type": "string"},
                "nsym": {"type": "integer", "default": 10},
            },
            "required": ["data_b64"],
        },
    },
    "ask_modulate": {
        "name": "ask_modulate",
        "description": "ASK (Amplitude Shift Keying) modulate a binary string",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bits": {"type": "string", "description": "Binary string (e.g., '10110010')"},
                "carrier_freq": {"type": "number", "description": "Carrier frequency (Hz)", "default": 1000},
                "baud_rate": {"type": "number", "description": "Baud rate (bps)", "default": 300},
                "amplitude": {"type": "number", "description": "Signal amplitude (0-1)", "default": 0.9},
            },
            "required": ["bits"],
        },
    },
    "psk_modulate": {
        "name": "psk_modulate",
        "description": "BPSK (Binary Phase Shift Keying) modulate a binary string",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bits": {"type": "string", "description": "Binary string (e.g., '10110010')"},
                "carrier_freq": {"type": "number", "description": "Carrier frequency (Hz)", "default": 1000},
                "baud_rate": {"type": "number", "description": "Baud rate (bps)", "default": 300},
                "amplitude": {"type": "number", "description": "Signal amplitude (0-1)", "default": 0.9},
            },
            "required": ["bits"],
        },
    },
    "line_code": {
        "name": "line_code",
        "description": "Encode bits using line coding: nrz, manchester, diff_manchester",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bits": {"type": "string", "description": "Binary string"},
                "encoding": {"type": "string", "enum": ["nrz", "manchester", "diff_manchester"], "default": "nrz"},
                "baud_rate": {"type": "number", "description": "Baud rate", "default": 300},
                "amplitude": {"type": "number", "default": 1.0},
            },
            "required": ["bits"],
        },
    },
    "constellation_diagram": {
        "name": "constellation_diagram",
        "description": "Plot constellation diagram from I/Q samples",
        "inputSchema": {
            "type": "object",
            "properties": {
                "iq_data_b64": {"type": "string", "description": "Base64-encoded complex128 I/Q samples"},
                "iq_path": {"type": "string", "description": "Path to .npy/.npz file with I/Q samples"},
                "title": {"type": "string", "default": "Constellation Diagram"},
                "return_image": {"type": "boolean", "default": True},
            },
        },
    },
    "eye_diagram": {
        "name": "eye_diagram",
        "description": "Generate an eye diagram for signal quality analysis",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "symbol_rate": {"type": "number", "description": "Symbol rate in Hz"},
                "samplerate": {"type": "integer", "default": 44100},
                "num_traces": {"type": "integer", "default": 200},
                "return_image": {"type": "boolean", "default": True},
            },
            "required": ["audio_data_b64", "symbol_rate"],
        },
    },
    "ber_measure": {
        "name": "ber_measure",
        "description": "Measure BER between transmitted and received bits, or simulate BPSK BER over SNR",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tx_bits": {"type": "string", "description": "Transmitted bit string"},
                "rx_bits": {"type": "string", "description": "Received bit string (omit for simulated AWGN BER)"},
                "snr_range_db": {"type": "array", "items": {"type": "number"}, "description": "SNR range for simulation"},
                "num_bits": {"type": "integer", "default": 1000},
            },
            "required": ["tx_bits"],
        },
    },
    "loopback_test": {
        "name": "loopback_test",
        "description": "Play a test signal and record it back for delay/quality analysis",
        "inputSchema": {
            "type": "object",
            "properties": {
                "duration": {"type": "number", "description": "Duration in seconds", "default": 2.0},
                "device": {"type": "integer", "description": "Audio device index"},
                "chirp": {"type": "boolean", "default": True},
            },
        },
    },
    "impulse_response": {
        "name": "impulse_response",
        "description": "Measure channel impulse response using chirp or MLS",
        "inputSchema": {
            "type": "object",
            "properties": {
                "duration": {"type": "number", "default": 1.0},
                "method": {"type": "string", "enum": ["chirp", "mls"], "default": "chirp"},
                "device": {"type": "integer"},
                "return_image": {"type": "boolean", "default": True},
            },
        },
    },
    "equalize_lms": {
        "name": "equalize_lms",
        "description": "LMS adaptive equalizer",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "desired_b64": {"type": "string", "description": "Desired signal (omit for auto-equalization)"},
                "filter_length": {"type": "integer", "default": 32},
                "step_size": {"type": "number", "default": 0.01},
                "samplerate": {"type": "integer", "default": 44100},
            },
            "required": ["audio_data_b64"],
        },
    },
    "text_encode": {
        "name": "text_encode",
        "description": "Encode text to base64 and bit string (helper for modem pipelines)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to encode (e.g. 'Hi')"},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["text"],
        },
    },
    "text_decode": {
        "name": "text_decode",
        "description": "Decode base64 data back to text",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data_b64": {"type": "string", "description": "Base64-encoded data"},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["data_b64"],
        },
    },
    "fsk_demodulate": {
        "name": "fsk_demodulate",
        "description": "Demodulate FSK audio back to a bit string (Goertzel per bit)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "mark_freq": {"type": "number", "default": 1200},
                "space_freq": {"type": "number", "default": 2200},
                "baud_rate": {"type": "number", "default": 300},
                "samplerate": {"type": "integer", "default": 44100},
                "threshold": {"type": "number", "description": "Confidence threshold for flagging weak bits"},
            },
            "required": ["audio_data_b64"],
        },
    },
    "modem_tx": {
        "name": "modem_tx",
        "description": "End-to-end text to audio: text -> base64 -> (optional RS-FEC) -> packet with preamble+CRC -> FSK -> WAV with chirp sync",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to transmit (e.g. 'Hi')"},
                "mark_freq": {"type": "number", "default": 1200},
                "space_freq": {"type": "number", "default": 2200},
                "baud_rate": {"type": "number", "default": 300},
                "samplerate": {"type": "integer", "default": 44100},
                "use_crc": {"type": "boolean", "default": True},
                "use_fec": {"type": "boolean", "default": False},
                "fec_nsym": {"type": "integer", "default": 10},
                "add_chirp_sync": {"type": "boolean", "default": True},
                "chirp_duration": {"type": "number", "default": 0.1},
                "pad_silence": {"type": "number", "default": 0.05},
            },
            "required": ["text"],
        },
    },
    "modem_rx": {
        "name": "modem_rx",
        "description": "End-to-end audio to text: chirp sync detect -> FSK demodulate -> packet decode (CRC) -> (optional RS-FEC) -> text",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "mark_freq": {"type": "number", "default": 1200},
                "space_freq": {"type": "number", "default": 2200},
                "baud_rate": {"type": "number", "default": 300},
                "samplerate": {"type": "integer", "default": 44100},
                "use_crc": {"type": "boolean", "default": True},
                "use_fec": {"type": "boolean", "default": False},
                "fec_nsym": {"type": "integer", "default": 10},
                "sync_chirp": {"type": "boolean", "default": True},
                "chirp_duration": {"type": "number", "default": 0.1},
                "preamble": {"type": "string", "default": "1010101010101010"},
            },
            "required": ["audio_data_b64"],
        },
    },
    "mfsk_modulate": {
        "name": "mfsk_modulate",
        "description": "Modulate a bit string with M-FSK (4/8/16 tones -> 2/3/4 bits per symbol) for higher bitrate than binary FSK",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bits": {"type": "string", "description": "Binary string e.g. '10110010'"},
                "num_tones": {"type": "integer", "enum": [4, 8, 16], "default": 4},
                "base_freq": {"type": "number", "default": 800},
                "tone_spacing": {"type": "number", "default": 400},
                "baud_rate": {"type": "number", "default": 300},
                "samplerate": {"type": "integer", "default": 44100},
                "amplitude": {"type": "number", "default": 0.9},
            },
            "required": ["bits"],
        },
    },
    "mfsk_demodulate": {
        "name": "mfsk_demodulate",
        "description": "Demodulate an M-FSK signal back to a bit string (Goertzel per symbol)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "num_tones": {"type": "integer", "enum": [4, 8, 16], "default": 4},
                "base_freq": {"type": "number", "default": 800},
                "tone_spacing": {"type": "number", "default": 400},
                "baud_rate": {"type": "number", "default": 300},
                "samplerate": {"type": "integer", "default": 44100},
            },
            "required": ["audio_data_b64"],
        },
    },
    "channel_simulate": {
        "name": "channel_simulate",
        "description": "Simulate an acoustic channel: AWGN at target SNR + reverb + distance attenuation + dropouts, for robustness testing without a mic",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "snr_db": {"type": "number", "default": 20.0},
                "reverb_amount": {"type": "number", "default": 0.0},
                "distance_attenuation": {"type": "number", "default": 1.0},
                "dropout_prob": {"type": "number", "default": 0.0},
                "dropout_len_ms": {"type": "number", "default": 20.0},
                "samplerate": {"type": "integer", "default": 44100},
                "seed": {"type": "integer", "description": "Optional RNG seed for reproducibility"},
            },
            "required": ["audio_data_b64"],
        },
    },
    "ber_curve": {
        "name": "ber_curve",
        "description": "Sweep SNR and measure real BER of the FSK/M-FSK modem chain, returning per-SNR BER and a PNG plot (core competition metric)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "default": "Hello, acoustic modem!"},
                "snr_list_db": {"type": "array", "items": {"type": "number"}, "description": "SNR points to test (dB)"},
                "mode": {"type": "string", "enum": ["fsk", "mfsk"], "default": "fsk"},
                "num_tones": {"type": "integer", "enum": [4, 8, 16], "default": 4},
                "baud_rate": {"type": "number", "default": 300},
                "mark_freq": {"type": "number", "default": 1200},
                "space_freq": {"type": "number", "default": 2200},
                "base_freq": {"type": "number", "default": 800},
                "tone_spacing": {"type": "number", "default": 400},
                "samplerate": {"type": "integer", "default": 44100},
                "return_image": {"type": "boolean", "default": True},
            },
        },
    },
    "modem_quality": {
        "name": "modem_quality",
        "description": "Quality report for a received modem signal: estimated SNR, confidence, BER (if reference text given) and throughput",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audio_data_b64": {"type": "string"},
                "reference_text": {"type": "string", "description": "Optional expected text to compute BER against"},
                "mode": {"type": "string", "enum": ["fsk", "mfsk"], "default": "fsk"},
                "num_tones": {"type": "integer", "enum": [4, 8, 16], "default": 4},
                "baud_rate": {"type": "number", "default": 300},
                "mark_freq": {"type": "number", "default": 1200},
                "space_freq": {"type": "number", "default": 2200},
                "base_freq": {"type": "number", "default": 800},
                "tone_spacing": {"type": "number", "default": 400},
                "samplerate": {"type": "integer", "default": 44100},
            },
            "required": ["audio_data_b64"],
        },
    },
}

# Map tool names to functions
TOOL_FUNCTIONS: dict[str, Any] = {
    "generate_tone": AudioDSPTools.generate_tone,
    "generate_fsk": AudioDSPTools.generate_fsk,
    "generate_chirp": AudioDSPTools.generate_chirp,
    "generate_noise": AudioDSPTools.generate_noise,
    "generate_silence": AudioDSPTools.generate_silence,
    "spectrogram": AudioDSPTools.spectrogram,
    "fft_analysis": AudioDSPTools.fft_analysis,
    "goertzel": AudioDSPTools.goertzel,
    "correlation": AudioDSPTools.correlation,
    "play_audio": AudioDSPTools.play_audio,
    "record_audio": AudioDSPTools.record_audio,
    "load_audio": AudioDSPTools.load_audio,
    "save_audio": AudioDSPTools.save_audio,
    "trim": AudioDSPTools.trim,
    "concatenate": AudioDSPTools.concatenate,
    "normalize": AudioDSPTools.normalize,
    "add_noise": AudioDSPTools.add_noise,
    "filter_design": AudioDSPTools.filter_design,
    "apply_filter": AudioDSPTools.apply_filter,
    "resample": AudioDSPTools.resample,
    "packet_encode": AudioDSPTools.packet_encode,
    "packet_decode": AudioDSPTools.packet_decode,
    "reed_solomon_encode": AudioDSPTools.reed_solomon_encode,
    "reed_solomon_decode": AudioDSPTools.reed_solomon_decode,
    "ask_modulate": AudioDSPTools.ask_modulate,
    "psk_modulate": AudioDSPTools.psk_modulate,
    "line_code": AudioDSPTools.line_code,
    "constellation_diagram": AudioDSPTools.constellation_diagram,
    "eye_diagram": AudioDSPTools.eye_diagram,
    "ber_measure": AudioDSPTools.ber_measure,
    "loopback_test": AudioDSPTools.loopback_test,
    "impulse_response": AudioDSPTools.impulse_response,
    "equalize_lms": AudioDSPTools.equalize_lms,
    "text_encode": AudioDSPTools.text_encode,
    "text_decode": AudioDSPTools.text_decode,
    "fsk_demodulate": AudioDSPTools.fsk_demodulate,
    "modem_tx": AudioDSPTools.modem_tx,
    "modem_rx": AudioDSPTools.modem_rx,
    "mfsk_modulate": AudioDSPTools.mfsk_modulate,
    "mfsk_demodulate": AudioDSPTools.mfsk_demodulate,
    "channel_simulate": AudioDSPTools.channel_simulate,
    "ber_curve": AudioDSPTools.ber_curve,
    "modem_quality": AudioDSPTools.modem_quality,
}

# ---------------------------------------------------------------------------
# JSON-RPC Handler
# ---------------------------------------------------------------------------


def handle_request(request: dict) -> dict | None:
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "experimental": {},
                },
                "serverInfo": {
                    "name": "audio-dsp-mcp",
                    "version": "0.1.0",
                },
            },
        }

    if method == "notifications/initialized":
        mcp_log("Audio DSP MCP server initialized")
        return None  # no response for notifications

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [t for t in TOOLS.values()],
            },
        }

    if method == "tools/call":
        tool_name = request["params"]["name"]
        arguments = request["params"].get("arguments", {})

        if tool_name not in TOOL_FUNCTIONS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
            }

        try:
            result = TOOL_FUNCTIONS[tool_name](**arguments)
            # Return the full result (including base64 audio/image payloads) as a
            # single text content block. Emitting {"type": "resource"} blocks
            # without resource.uri/resource.blob violates the MCP content schema
            # and strict clients (Roo Code Zod validation) reject the whole tool
            # result. Keeping "data" in the JSON also preserves tool chaining
            # (e.g. modem_tx -> modem_rx needs audio_data_b64 in the text output).
            content: list[dict] = [{
                "type": "text",
                "text": json.dumps(result, ensure_ascii=False, default=str),
            }]

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": content},
            }
        except Exception as e:
            mcp_error(f"Error calling {tool_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)},
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main():
    mcp_log("Audio DSP MCP server starting...")
    mcp_log(f"Python: {sys.version}")
    mcp_log(f"sounddevice: {HAS_SOUNDDEVICE}, soundfile: {HAS_SOUNDFILE}, matplotlib: {HAS_MATPLOTLIB}")

    buffer = ""
    for line in sys.stdin:
        buffer += line
        try:
            request = json.loads(buffer)
            buffer = ""
        except json.JSONDecodeError:
            continue  # incomplete JSON, wait for more

        response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
