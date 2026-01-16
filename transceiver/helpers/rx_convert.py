#!/usr/bin/env python3
"""
Konvertiert Binär-IQ-Dateien zwischen
  • SC16 (int16, interleaved I/Q, 4 B/Sample)  und
  • FC32 (float32, I+Q hintereinander, 8 B/Sample).

Unterstützt optional mehrere Kanäle und NumPy-Dateien.

Standard-Ziel ohne Option:  SC16
Dateiname erhält immer '_conv' vor der Endung.
"""

from pathlib import Path
import argparse
import numpy as np

INT16_MAX = 32767.0  # ±32767 … ±32768


def _reshape_channels(
    data: np.ndarray, channels: int, layout: str
) -> np.ndarray:
    if channels <= 1:
        return data
    if data.size % channels:
        raise ValueError("Kanallayout passt nicht zur Datenlänge.")
    if layout == "blocked":
        return data.reshape(channels, -1)
    if layout == "interleaved":
        return data.reshape(-1, channels).T
    raise ValueError(f"Unbekanntes Kanallayout: {layout}")


def _flatten_channels(data: np.ndarray, layout: str) -> np.ndarray:
    if data.ndim == 1:
        return data
    if layout == "blocked":
        return data.reshape(-1)
    if layout == "interleaved":
        return data.T.reshape(-1)
    raise ValueError(f"Unbekanntes Kanallayout: {layout}")


def detect_format(path: Path) -> str:
    """Grober Heuristik-Test nach Dateigröße & Float-Probe."""
    if path.suffix.lower() in {".npy", ".npz"}:
        return "numpy"
    size = path.stat().st_size
    if size % 8 == 0:  # könnte fc32 oder sc16*2
        with path.open("rb") as f:
            probe = np.frombuffer(f.read(8), dtype=np.float32)
        if np.isfinite(probe).all() and np.max(np.abs(probe)) < 1e5:
            return "fc32"
    if size % 4 == 0:
        return "sc16"
    raise ValueError("Unbekanntes Dateiformat / Größe passt nicht.")


def load_fc32(path: Path, channels: int = 1, layout: str = "blocked") -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % 2:
        raise ValueError("FC32-Datei muss gerade Anzahl float32 enthalten.")
    data = arr.view(np.complex64)
    return _reshape_channels(data, channels, layout)


def load_sc16(path: Path, channels: int = 1, layout: str = "blocked") -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size % 2:
        raise ValueError("SC16-Datei hat ungerade I/Q-Werte.")
    raw = raw.reshape(-1, 2).astype(np.float32)
    data = raw[:, 0] + 1j * raw[:, 1]
    return _reshape_channels(data, channels, layout)


def load_numpy(
    path: Path,
    channels: int = 1,
    layout: str = "blocked",
    mmap_mode: str | None = None,
) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False, mmap_mode=mmap_mode) as npz:
            if not npz.files:
                raise ValueError("NPZ-Datei enthält keine Arrays.")
            data = npz[npz.files[0]]
    else:
        data = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)

    if not np.iscomplexobj(data):
        if data.ndim >= 1 and data.shape[-1] == 2:
            data = data[..., 0] + 1j * data[..., 1]
        else:
            raise ValueError("NumPy-Datei enthält keine komplexen IQ-Daten.")

    data = np.asarray(data)
    if data.ndim == 1:
        return _reshape_channels(data, channels, layout)
    if data.ndim == 2:
        if channels <= 1:
            return data.reshape(-1)
        if data.shape[0] == channels:
            return data
        if data.shape[1] == channels:
            return data.T
        return _reshape_channels(data.reshape(-1), channels, layout)
    raise ValueError("NumPy-Datei hat ein unerwartetes Datenformat.")


def load_iq_file(
    path: Path,
    channels: int = 1,
    layout: str = "blocked",
    mmap_mode: str | None = None,
) -> np.ndarray:
    src_fmt = detect_format(path)
    if src_fmt == "numpy":
        return load_numpy(
            path, channels=channels, layout=layout, mmap_mode=mmap_mode
        )
    if src_fmt == "fc32":
        return load_fc32(path, channels=channels, layout=layout)
    return load_sc16(path, channels=channels, layout=layout)


def save_sc16(
    path: Path,
    data: np.ndarray,
    amplitude: float | None = None,
    layout: str = "blocked",
):
    """Speichert *data* im SC16-Format.

    Wenn ``amplitude`` angegeben ist, wird das Signal so skaliert, dass
    dessen Maximalwert diese Amplitude erreicht.  Standardmäßig wird – wie
    bisher – auf den vollen int16-Bereich (±32767) skaliert.
    """

    target = INT16_MAX if amplitude is None else float(amplitude)
    scale = target / (np.abs(data).max() or 1.0)
    data = data * scale
    flat = _flatten_channels(data, layout)
    real = np.clip(np.round(flat.real), -32768, 32767).astype(np.int16)
    imag = np.clip(np.round(flat.imag), -32768, 32767).astype(np.int16)
    out = np.empty(real.size + imag.size, dtype=np.int16)
    out[0::2], out[1::2] = real, imag
    out.tofile(path)
    print(f"→ SC16   |   Skaliert um Faktor {scale:.4f}")


def save_fc32(path: Path, data: np.ndarray, layout: str = "blocked"):
    flat = _flatten_channels(data, layout)
    out = flat.astype(np.complex64) / (INT16_MAX + 1.0)
    out.view(np.float32).tofile(path)
    print("→ FC32   |   Durch 32768 normiert")


def main():
    ap = argparse.ArgumentParser(
        description="IQ-Format-Konverter (Default-Ziel: sc16)")
    ap.add_argument("infile", help="Quelldatei (SC16 oder FC32)")
    ap.add_argument("--to", choices=["sc16", "fc32"],
                    default="sc16", help="Ziel­format (Default sc16)")
    ap.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help=(
            "Zielamplitude für SC16-Ausgabe. Ohne Angabe wird auf ±32767 "
            "skaliert."
        ),
    )
    ap.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Anzahl der Kanäle (Default: 1)",
    )
    ap.add_argument(
        "--layout",
        choices=["blocked", "interleaved"],
        default="blocked",
        help="Kanallayout (blocked oder interleaved)",
    )
    args = ap.parse_args()

    infile = Path(args.infile).resolve()
    src_fmt = detect_format(infile)
    dst_fmt = args.to

    # Laden
    if src_fmt == "numpy":
        data = load_numpy(infile, channels=args.channels, layout=args.layout)
    elif src_fmt == "fc32":
        data = load_fc32(infile, channels=args.channels, layout=args.layout)
    else:
        data = load_sc16(infile, channels=args.channels, layout=args.layout)

    # Ausgabedatei
    out_name = infile.with_name(infile.stem + "_conv" + infile.suffix)

    # Speichern
    if dst_fmt == "sc16":
        save_sc16(out_name, data, args.amplitude, layout=args.layout)
    else:
        save_fc32(out_name, data, layout=args.layout)

    print(f"{len(data):,} Samples geschrieben ➜ {out_name}")

if __name__ == "__main__":
    main()
