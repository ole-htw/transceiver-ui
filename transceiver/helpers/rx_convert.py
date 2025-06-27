#!/usr/bin/env python3
"""
Konvertiert Binär-IQ-Dateien zwischen
  • SC16 (int16, interleaved I/Q, 4 B/Sample)  und
  • FC32 (float32, I+Q hintereinander, 8 B/Sample).

Standard-Ziel ohne Option:  SC16
Dateiname erhält immer '_conv' vor der Endung.
"""

from pathlib import Path
import argparse
import numpy as np

INT16_MAX = 32767.0                 # ±32767 … ±32768

def detect_format(path: Path) -> str:
    """Grober Heuristik-Test nach Dateigröße & Float-Probe."""
    size = path.stat().st_size
    if size % 8 == 0:                        # könnte fc32 oder sc16*2
        with path.open("rb") as f:
            probe = np.frombuffer(f.read(8), dtype=np.float32)
        if np.isfinite(probe).all() and np.max(np.abs(probe)) < 1e5:
            return "fc32"
    if size % 4 == 0:
        return "sc16"
    raise ValueError("Unbekanntes Dateiformat / Größe passt nicht.")

def load_fc32(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % 2:
        raise ValueError("FC32-Datei muss gerade Anzahl float32 enthalten.")
    return arr.view(np.complex64)

def load_sc16(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size % 2:
        raise ValueError("SC16-Datei hat ungerade I/Q-Werte.")
    raw = raw.reshape(-1, 2).astype(np.float32)
    return raw[:, 0] + 1j * raw[:, 1]

def save_sc16(path: Path, data: np.ndarray):
    scale = INT16_MAX / (np.abs(data).max() or 1.0)
    data = data * scale
    real = np.clip(np.round(data.real), -32768, 32767).astype(np.int16)
    imag = np.clip(np.round(data.imag), -32768, 32767).astype(np.int16)
    out = np.empty(real.size + imag.size, dtype=np.int16)
    out[0::2], out[1::2] = real, imag
    out.tofile(path)
    print(f"→ SC16   |   Skaliert um Faktor {scale:.4f}")

def save_fc32(path: Path, data: np.ndarray):
    out = data.astype(np.complex64) / (INT16_MAX + 1.0)
    out.view(np.float32).tofile(path)
    print("→ FC32   |   Durch 32768 normiert")

def main():
    ap = argparse.ArgumentParser(
        description="IQ-Format-Konverter (Default-Ziel: sc16)")
    ap.add_argument("infile", help="Quelldatei (SC16 oder FC32)")
    ap.add_argument("--to", choices=["sc16", "fc32"],
                    default="sc16", help="Ziel­format (Default sc16)")
    args = ap.parse_args()

    infile = Path(args.infile).resolve()
    src_fmt = detect_format(infile)
    dst_fmt = args.to

    # Laden
    data = load_fc32(infile) if src_fmt == "fc32" else load_sc16(infile)

    # Ausgabedatei
    out_name = infile.with_name(infile.stem + "_conv" + infile.suffix)

    # Speichern
    if dst_fmt == "sc16":
        save_sc16(out_name, data)
    else:
        save_fc32(out_name, data)

    print(f"{len(data):,} Samples geschrieben ➜ {out_name}")

if __name__ == "__main__":
    main()

