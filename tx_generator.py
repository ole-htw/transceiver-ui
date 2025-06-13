#!/usr/bin/env python3
"""
Erweitertes Waveform‑Generator‑Skript
------------------------------------
Unterstützt jetzt drei Wellenformen:
    • sinus      – Reiner Sinus‑Ton
    • zadoffchu  – Zadoff–Chu‑Sequenz (komplex, zykloideiphase)
    • chirp      – Linear aufsteigender Up‑Chirp (f0 → f1)

Neu: Option **--no-zeros**
    Gibt an, dass *keine* Null‑Samples an die Wellenform angehängt werden.
    Standard ist weiterhin das alte Verhalten (Waveform + Null‑Sequenz).

Die erzeugte Folge wird als interleaved int16 (IQ IQ …) in eine Binärdatei
 geschrieben.
"""

import argparse
import math
from typing import Optional
from pathlib import Path
from datetime import datetime

import numpy as np

# ---------- Hilfsfunktionen --------------------------------------------------


def gcd(a: int, b: int) -> int:
    """Größter gemeinsamer Teiler (kompatibel zu Python <3.5)."""
    a, b = abs(int(a)), abs(int(b))
    try:
        return math.gcd(a, b)
    except AttributeError:  # pragma: no cover – Fallback für sehr alte Pythons
        while b:
            a, b = b, a % b
        return a


def find_prime_near(target: int, search_up: bool = True) -> int:
    """Sucht die nächste Primzahl in der Umgebung von *target*.

    Wird nur für Zadoff‑Chu genutzt (N muss nicht zwingend prim, aber
    teilerfremd zu q sein; eine Primzahl ist der einfachste Weg dahin).
    Begrenzung: ±1000 um *target*, sonst Warnung.
    """
    if target <= 1:
        return 2

    num = target + 1 if search_up else target - 1
    step = 1 if search_up else -1

    while num > 1:
        if num > 2 and num % 2 == 0:
            num += step
            continue
        for i in range(3, int(math.sqrt(num)) + 1, 2):
            if num % i == 0:
                break
        else:
            return num
        num += step
        if abs(num - target) > 1000:
            print(f"Warnung: Keine Primzahl nahe {target}.")
            return target + step
    return 2


def _pretty(val: float) -> str:
    """Hilfsfunktion f\xc3\xbcr verk\xc3\xbcrzte numerische Strings."""
    abs_v = abs(val)
    if abs_v >= 1e6 and abs_v % 1e6 == 0:
        return f"{int(val/1e6)}M"
    if abs_v >= 1e3 and abs_v % 1e3 == 0:
        return f"{int(val/1e3)}k"
    return f"{int(val)}"


def generate_filename(args) -> Path:
    """Erzeuge einen Dateinamen basierend auf den Parametern."""
    parts = [args.waveform]
    if args.waveform == "sinus":
        parts.append(f"f{_pretty(args.f)}")
    elif args.waveform == "zadoffchu":
        parts.append(f"q{args.q}")
    elif args.waveform == "chirp":
        parts.append(f"{_pretty(args.f0)}_{_pretty(args.f1)}")
    parts.append(f"N{args.samples}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "_".join(parts) + f"_{stamp}.bin"
    return Path(args.output_dir) / name


# ---------- Waveform‑Generator ----------------------------------------------


def generate_waveform(
    waveform: str,
    fs: float,
    f: float,
    N: int,
    q: int = 1,
    f0: Optional[float] = None,
    f1: Optional[float] = None,
) -> np.ndarray:
    """Erzeugt komplexe Samples einer der unterstützten Wellenformen."""

    if N <= 0:
        raise ValueError("N muss > 0 sein.")

    n = np.arange(N)

    w = waveform.lower()

    if w == "sinus":  # ---------- Sinus -----------------------------------
        signal = np.exp(2j * np.pi * f * n / fs)

    elif w == "zadoffchu":  # -------- Zadoff–Chu --------------------------
        if q == 0:
            raise ValueError("Zadoff‑Chu‑Parameter q darf nicht 0 sein.")
        if gcd(q, N) != 1:
            print(f"WARNUNG: q={q} nicht teilerfremd zu N={N}.")
        if N % 2:
            signal = np.exp(-1j * np.pi * q * n**2 / N)
        else:
            signal = np.exp(-1j * np.pi * q * n * (n + 1) / N)

    elif w == "chirp":  # -------------- Linear Up‑Chirp -------------------
        if f0 is None or f1 is None:
            raise ValueError("Für Chirp müssen f0 und f1 gesetzt sein.")
        # Zeitbasis in Sekunden
        t = n / fs
        T = N / fs  # Gesamtdauer
        k = (f1 - f0) / T  # Steigung (Hz pro s)
        # Phase φ(t) = 2π (f0 t + ½ k t²)
        phi = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
        signal = np.exp(1j * phi)

    else:
        raise ValueError(f"Unbekannte Wellenform: {waveform}")

    return signal.astype(np.complex64)


# ---------- Hauptprogramm ----------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Erzeugt eine Wellen­folge (sinus, zadoffchu oder chirp) "
            "gefolgt optional von einer gleichen Anzahl Null‑Samples und speichert "
            "sie als interleaved int16 (IQIQ…) in eine Datei."
        )
    )

    # Grundparameter
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Zieldatei. Wenn nicht angegeben, wird der Dateiname automatisch generiert.",
    )
    parser.add_argument(
        "--output-dir",
        default="signals/tx",
        help="Basisverzeichnis f\xc3\xbcr generierte Dateien (Default: signals/tx)",
    )
    parser.add_argument(
        "--waveform",
        choices=["sinus", "zadoffchu", "chirp"],
        default="sinus",
        help="Wellenform (Standard: sinus)",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=25e6,
        help="Abtastrate in Samples/s (Standard: 25e6)",
    )

    # Sinus‑spezifisch
    parser.add_argument(
        "--f",
        type=float,
        default=1e6,
        help="Frequenz in Hz – nur für sinus (Standard: 1e6)",
    )

    # Zadoff–Chu‑spezifisch
    parser.add_argument(
        "--q", type=int, default=1, help="Zadoff‑Chu‑Parameter q (Standard: 1)"
    )

    # Chirp‑spezifisch
    parser.add_argument(
        "--f0",
        type=float,
        default=0.0,
        help="Startfrequenz für Chirp in Hz (Standard: 0)",
    )
    parser.add_argument(
        "--f1",
        type=float,
        default=None,
        help="Endfrequenz für Chirp in Hz (Standard: fs/2 − 1 Hz)",
    )

    # Gemeinsame Parameter
    parser.add_argument(
        "--amplitude",
        type=float,
        default=10000.0,
        help="Ziel‑Amplitude nach Skalierung (Standard: 10000)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=40000,
        help="Anzahl Samples der Wellenform (Standard: 40000)",
    )

    # Neue Option: Null-Sequenz weglassen
    parser.add_argument(
        "--no-zeros",
        action="store_true",
        help="Keine Null‑Samples anhängen (Standard: deaktiviert)",
    )

    args = parser.parse_args()

    if args.filename is None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.filename = str(generate_filename(args))
    else:
        Path(args.filename).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pre‑Flight‑Checks
    # ------------------------------------------------------------------
    if args.waveform == "chirp" and args.f1 is None:
        args.f1 = args.fs / 2 - 1  # Maximal fast bis Nyquist

    N_waveform = args.samples

    # Blockgröße ggf. an Primzahl anpassen (nur ZC)
    if args.waveform == "zadoffchu":
        prime = find_prime_near(N_waveform, search_up=True)
        if prime != N_waveform:
            print(f"Info: samples={N_waveform} angepasst auf Primzahl {prime} für ZC.")
            N_waveform = prime

    append_zeros = not args.no_zeros
    total_samples = N_waveform * (2 if append_zeros else 1)

    # ------------------------------------------------------------------
    if append_zeros:
        print(
            f"Erzeuge {N_waveform} Samples {args.waveform} + {N_waveform} Null‑Samples."
        )
    else:
        print(f"Erzeuge {N_waveform} Samples {args.waveform} (ohne Null‑Samples).")

    if args.waveform == "chirp":
        print(f"  Chirp: {args.f0/1e6:.3f} MHz → {args.f1/1e6:.3f} MHz")

    # Wellenform generieren
    waveform_signal = generate_waveform(
        args.waveform,
        args.fs,
        args.f,
        N_waveform,
        args.q,
        f0=args.f0,
        f1=args.f1,
    )

    if append_zeros:
        zeros = np.zeros(N_waveform, dtype=np.complex64)
        final_signal = np.concatenate([waveform_signal, zeros])
    else:
        final_signal = waveform_signal

    # Skalierung
    max_abs = np.max(np.abs(final_signal)) if np.any(final_signal) else 1.0
    scale = args.amplitude / max_abs if max_abs > 1e-9 else 1.0
    scaled = final_signal * scale

    # Interleaved int16 schreiben
    real_i16 = np.int16(np.round(np.real(scaled)))
    imag_i16 = np.int16(np.round(np.imag(scaled)))
    interleaved = np.empty(real_i16.size + imag_i16.size, dtype=np.int16)
    interleaved[0::2] = np.clip(real_i16, -32768, 32767)
    interleaved[1::2] = np.clip(imag_i16, -32768, 32767)
    interleaved.tofile(args.filename)

    # ------------------------------------------------------------------
    print("-" * 40)
    print(f"Datei:          {args.filename}")
    print(f"Samples gesamt: {total_samples}")
    print(f"Amplitude Ziel: {args.amplitude:.0f} (Skalierung: {scale:.2f})")
    print("-" * 40)


if __name__ == "__main__":
    main()
