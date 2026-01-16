#!/usr/bin/env python3
"""
Erweitertes Waveform-Generator-Skript
------------------------------------
Unterstützt jetzt drei Wellenformen:
    • sinus      – Reiner Sinus-Ton
    • zadoffchu  – Zadoff–Chu-Sequenz (komplex, zykloideiphase)
    • chirp      – Linear aufsteigender Up-Chirp (f0 → f1)

Option **--no-zeros**
    Gibt an, dass *keine* Null-Samples an die Wellenform angehängt werden.
    Standard ist weiterhin das alte Verhalten (Waveform + Null-Sequenz).

Option **--oversampling**
    Oversampling-Faktor nur für Zadoff-Chu. Wird korrekt als:
        Symbole (1 Sample/Symbol) -> (Upsampling + RRC-FIR in einem Schritt)
    implementiert (scipy.signal.upfirdn).

Die erzeugte Folge wird als interleaved int16 (IQ IQ …) in eine Binärdatei
geschrieben.
"""

import argparse
import math
from typing import Optional
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.signal import upfirdn


# ---------- Hilfsfunktionen --------------------------------------------------


def gcd(a: int, b: int) -> int:
    """Größter gemeinsamer Teiler (kompatibel zu Python <3.5)."""
    a, b = abs(int(a)), abs(int(b))
    try:
        return math.gcd(a, b)
    except AttributeError:  # pragma: no cover – Fallback für sehr alte Pythons
        while b:
            a, b = b, a % b
        return a


def find_prime_near(target: int, search_up: bool = True) -> int:
    """Sucht die nächste Primzahl in der Umgebung von *target*.

    Wird nur für Zadoff-Chu genutzt (N muss nicht zwingend prim, aber
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
    """Hilfsfunktion für verkürzte numerische Strings."""
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
        if args.oversampling != 1:
            parts.append(f"os{args.oversampling}")
    elif args.waveform == "chirp":
        parts.append(f"{_pretty(args.f0)}_{_pretty(args.f1)}")

    if args.waveform == "zadoffchu":
        parts.append(f"Nsym{args.samples}")
        if args.oversampling != 1:
            parts.append(f"Nsamp{args.samples * args.oversampling}")
    else:
        parts.append(f"N{args.samples}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "_".join(parts) + f"_{stamp}.bin"
    return Path(args.output_dir) / name


def rrc_coeffs(beta: float, span: int, sps: int = 1) -> np.ndarray:
    """Koeffizienten für einen Root-Raised-Cosine-Filter erzeugen.

    span: Filterlänge in Symbolen (typ. 6..12)
    sps:  Samples per Symbol (Oversampling-Faktor)
    """
    if span <= 0:
        return np.array([1.0], dtype=np.float32)
    if sps <= 0:
        raise ValueError("sps muss >= 1 sein.")
    if beta < 0 or beta > 1:
        raise ValueError("beta muss im Bereich [0, 1] liegen.")

    num_taps = span * sps + 1              # span = Gesamtlänge in Symbolen
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps
    h = np.zeros_like(t, dtype=np.float64)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-12:
            h[i] = (
                beta
                / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
                )
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(
                np.pi * ti * (1.0 + beta)
            )
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den

    # Energie-Normierung (üblich bei Pulse-Shaping)
    h /= np.sqrt(np.sum(h**2) + 1e-30)
    return h.astype(np.float32)


def _trim_to_length(x: np.ndarray, length: int) -> np.ndarray:
    """Trimmt oder paddet (mit Nullen) auf genau 'length' Samples."""
    if length <= 0:
        return x[:0]
    if len(x) == length:
        return x
    if len(x) > length:
        return x[:length]
    out = np.zeros(length, dtype=x.dtype)
    out[: len(x)] = x
    return out


# ---------- Waveform-Generator ----------------------------------------------


def generate_waveform(
    waveform: str,
    fs: float,
    f: float,
    N: int,
    q: int = 1,
    f0: Optional[float] = None,
    f1: Optional[float] = None,
    rrc_beta: float = 0.25,
    rrc_span: int = 6,
    oversampling: int = 1,
) -> np.ndarray:
    """Erzeugt komplexe Samples einer der unterstützten Wellenformen.

    Wichtigste Fixes ggü. deiner Version:
    - RRC-Taps werden erzeugt, bevor sie benutzt werden.
    - Für Zadoff-Chu ist Oversampling korrekt als upfirdn(RRC, symbols, up=OS)
      implementiert (Upsampling + FIR in einem Schritt).
    - Kein doppeltes Filtern mehr.
    - Für den oversampleten ZC-Fall wird Delay kompensiert und auf exakt
      N*oversampling Samples getrimmt.
    """
    if N <= 0:
        raise ValueError("N muss > 0 sein.")
    if oversampling <= 0:
        raise ValueError("oversampling muss >= 1 sein")

    w = waveform.lower()

    # ---------- Sinus ---------------------------------------------------------
    if w == "sinus":
        n = np.arange(N)
        sig = np.exp(2j * np.pi * f * n / fs).astype(np.complex64)

        # Optional: RRC auf "normales" kontinuierliches Sinussignal ist meist nicht sinnvoll,
        # aber wir behalten dein Verhalten bei: Wenn rrc_span > 0, wird gefiltert.
        if rrc_span > 0:
            h = rrc_coeffs(rrc_beta, rrc_span, sps=1).astype(np.float32)
            sig = np.convolve(sig, h, mode="same").astype(np.complex64)
        return sig

    # ---------- Chirp ---------------------------------------------------------
    if w == "chirp":
        if f0 is None or f1 is None:
            raise ValueError("Für Chirp müssen f0 und f1 gesetzt sein.")
        n = np.arange(N)
        t = n / fs
        T = N / fs
        k = (f1 - f0) / T
        phi = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
        sig = np.exp(1j * phi).astype(np.complex64)

        if rrc_span > 0:
            h = rrc_coeffs(rrc_beta, rrc_span, sps=1).astype(np.float32)
            sig = np.convolve(sig, h, mode="same").astype(np.complex64)
        return sig

    # ---------- Zadoff–Chu ----------------------------------------------------
    if w == "zadoffchu":
        if q == 0:
            raise ValueError("Zadoff-Chu-Parameter q darf nicht 0 sein.")
        if gcd(q, N) != 1:
            print(f"WARNUNG: q={q} nicht teilerfremd zu N={N}.")

        n = np.arange(N)
        if N % 2:
            symbols = np.exp(-1j * np.pi * q * n**2 / N).astype(np.complex64)
        else:
            symbols = np.exp(-1j * np.pi * q * n * (n + 1) / N).astype(np.complex64)

        # Oversampling-Faktor entspricht "samples per symbol" für die Pulse-Shaping-Stufe
        sps = int(oversampling)

        if rrc_span > 0:
            h = rrc_coeffs(rrc_beta, rrc_span, sps=sps).astype(np.float32)
        else:
            # Kein Pulse-Shaping: upfirdn mit [1] macht nur Zero-Stuffing.
            # Hinweis: Das ist *kein* "glattes" Interpolieren.
            h = np.array([1.0], dtype=np.float32)

        if sps > 1:
            # Upsampling + FIR in einem Schritt
            y = upfirdn(h, symbols, up=sps).astype(np.complex64)

            delay = (len(h) - 1) // 2
            start = delay
            stop = start + N * sps
            y = y[start:stop]                      # schneidet direkt auf Zielbereich
            y = _trim_to_length(y, N * sps).astype(np.complex64)
            return y
            
        else:
            # Kein Oversampling -> einfach filtern (same-Länge)
            if len(h) > 1:
                y = np.convolve(symbols, h, mode="same").astype(np.complex64)
            else:
                y = symbols
            return y

    raise ValueError(f"Unbekannte Wellenform: {waveform}")


# ---------- Hauptprogramm ----------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Erzeugt eine Wellen­folge (sinus, zadoffchu oder chirp) "
            "gefolgt optional von einer gleichen Anzahl Null-Samples und speichert "
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
        help="Basisverzeichnis für generierte Dateien (Default: signals/tx)",
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

    # Sinus-spezifisch
    parser.add_argument(
        "--f",
        type=float,
        default=1e6,
        help="Frequenz in Hz – nur für sinus (Standard: 1e6)",
    )

    # Zadoff–Chu-spezifisch + RRC
    parser.add_argument("--q", type=int, default=1, help="Zadoff-Chu-Parameter q (Standard: 1)")
    parser.add_argument(
        "--rrc-beta",
        type=float,
        default=0.25,
        help="RRC-Rollofffaktor (Standard: 0.25)",
    )
    parser.add_argument(
        "--rrc-span",
        type=int,
        default=6,
        help="RRC-Filterspan in Symbolen (Standard: 6; 0 deaktiviert)",
    )

    # Chirp-spezifisch
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
        help="Ziel-Amplitude nach Skalierung (Standard: 10000)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=40000,
        help="Anzahl Samples der Wellenform (Zadoff-Chu: Symbolanzahl, Standard: 40000)",
    )
    parser.add_argument(
        "--oversampling",
        type=int,
        default=1,
        help="Oversampling-Faktor nur für Zadoff-Chu (Standard: 1)",
    )

    # Null-Sequenz weglassen
    parser.add_argument(
        "--no-zeros",
        action="store_true",
        help="Keine Null-Samples anhängen (Standard: deaktiviert)",
    )

    args = parser.parse_args()

    if args.filename is None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.filename = str(generate_filename(args))
    else:
        Path(args.filename).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pre-Flight-Checks
    # ------------------------------------------------------------------
    if args.waveform == "chirp" and args.f1 is None:
        args.f1 = args.fs / 2 - 1

    N_waveform = args.samples
    N_output = N_waveform

    if args.waveform == "zadoffchu":
        # Dein altes Verhalten beibehalten: nur bei OS<=1 auf Primzahl anpassen
        if args.oversampling <= 1:
            prime = find_prime_near(N_waveform, search_up=True)
            if prime != N_waveform:
                print(f"Info: samples={N_waveform} angepasst auf Primzahl {prime} für ZC.")
                N_waveform = prime

        N_output = N_waveform * max(1, int(args.oversampling))

    append_zeros = not args.no_zeros

    if append_zeros:
        print(f"Erzeuge {N_output} Samples {args.waveform} + {N_output} Null-Samples")
    else:
        print(f"Erzeuge {N_output} Samples {args.waveform} (ohne Null-Samples)")

    if args.waveform == "zadoffchu" and args.oversampling != 1:
        print(f"  Oversampling: {args.oversampling}× (Upsampling + RRC via upfirdn)")

    if args.waveform == "chirp":
        print(f"  Chirp: {args.f0/1e6:.3f} MHz → {args.f1/1e6:.3f} MHz")

    # Wellenform generieren
    waveform_signal = generate_waveform(
        args.waveform,
        args.fs,
        args.f,
        N_waveform,
        args.q,
        f0=args.f0,
        f1=args.f1,
        rrc_beta=args.rrc_beta,
        rrc_span=args.rrc_span,
        oversampling=args.oversampling,
    ).astype(np.complex64)

    # (Sicherstellen, dass die Länge genau den Erwartungen entspricht)
    waveform_signal = _trim_to_length(waveform_signal, N_output).astype(np.complex64)

    final_len = len(waveform_signal)
    if append_zeros:
        zeros = np.zeros(final_len, dtype=np.complex64)
        final_signal = np.concatenate([waveform_signal, zeros])
    else:
        final_signal = waveform_signal

    total_samples = final_signal.size

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

    print("-" * 40)
    print(f"Datei:          {args.filename}")
    print(f"Samples gesamt: {total_samples}")
    print(f"Amplitude Ziel: {args.amplitude:.0f} (Skalierung: {scale:.2f})")
    print("-" * 40)


if __name__ == "__main__":
    main()

