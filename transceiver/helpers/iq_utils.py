import numpy as np


def complex_to_interleaved_int16(data: np.ndarray) -> np.ndarray:
    """Convert complex samples to interleaved int16 with proper clipping."""
    real = np.clip(np.round(np.real(data)), -32768, 32767)
    imag = np.clip(np.round(np.imag(data)), -32768, 32767)
    interleaved = np.empty(real.size + imag.size, dtype=np.int16)
    interleaved[0::2] = real.astype(np.int16)
    interleaved[1::2] = imag.astype(np.int16)
    return interleaved


def save_interleaved(
    filename: str, data: np.ndarray, amplitude: float = 10000.0
) -> None:
    """Save complex64 data as interleaved int16."""
    if data.ndim != 1:
        raise ValueError("Mehrkanal-Daten müssen vor dem Speichern gemischt werden.")
    max_abs = np.max(np.abs(data)) if np.any(data) else 1.0
    scale = amplitude / max_abs if max_abs > 1e-9 else 1.0
    scaled = data * scale
    interleaved = complex_to_interleaved_int16(scaled)
    interleaved.tofile(filename)
