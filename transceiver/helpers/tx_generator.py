#!/usr/bin/env python3
"""
Erweitertes Waveform-Generator-Skript
------------------------------------
Unterstützt jetzt fünf Wellenformen:
    • sinus      – Reiner Sinus-Ton
    • doppelsinus – Zwei halbe Sinustöne (f1 + f2)
    • zadoffchu  – Zadoff–Chu-Sequenz (komplex, zykloideiphase)
    • chirp      – Linear aufsteigender Up-Chirp (f0 → f1)
    • ofdm_preamble – OFDM-Präambel mit CP (BPSK, deterministisch)

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

from .iq_utils import complex_to_interleaved_int16


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
    elif args.waveform == "doppelsinus":
        parts.append(f"{_pretty(args.f)}_{_pretty(args.f2)}")
    elif args.waveform == "zadoffchu":
        parts.append(f"q{args.q}")
        if args.oversampling != 1:
            parts.append(f"os{args.oversampling}")
    elif args.waveform == "chirp":
        parts.append(f"{_pretty(args.f0)}_{_pretty(args.f1)}")
    elif args.waveform == "ofdm_preamble":
        parts.append(f"nfft{args.ofdm_nfft}")
        parts.append(f"cp{args.ofdm_cp}")
        parts.append(f"sym{args.ofdm_symbols}")
        if args.ofdm_short_repeats:
            parts.append(f"short{args.ofdm_short_repeats}")
    elif args.waveform == "pseudo_noise":
        parts.append(f"pn{_pretty(args.pn_chip_rate)}")
        parts.append(f"seed{args.pn_seed}")

    if args.waveform == "zadoffchu":
        parts.append(f"Nsym{args.samples}")
        if args.oversampling != 1:
            parts.append(f"Nsamp{args.samples * args.oversampling}")
    elif args.waveform == "ofdm_preamble":
        parts.append(f"N{args.samples}")
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
    
    eps = 1e-8
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0, atol=eps):
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1.0 / (4.0 * beta), atol=eps):
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
    ofdm_nfft: int = 64,
    ofdm_cp_len: int = 16,
    ofdm_active_subcarriers: int = 52,
    ofdm_num_symbols: int = 2,
    ofdm_short_repeats: int = 10,
    pn_chip_rate: float = 1e6,
    pn_seed: int = 1,
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
    w = waveform.lower()
    if w != "ofdm_preamble" and N <= 0:
        raise ValueError("N muss > 0 sein.")
    if oversampling <= 0:
        raise ValueError("oversampling muss >= 1 sein")

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

    # ---------- Doppelsinus ---------------------------------------------------
    if w == "doppelsinus":
        if f1 is None:
            raise ValueError("Für Doppelsinus müssen f und f2 gesetzt sein.")
        n = np.arange(N)
        sig = (
            0.5 * np.exp(2j * np.pi * f * n / fs)
            + 0.5 * np.exp(2j * np.pi * f1 * n / fs)
        ).astype(np.complex64)

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

    # ---------- OFDM-Preamble -------------------------------------------------
    if w == "ofdm_preamble":
        if ofdm_nfft <= 0:
            raise ValueError("ofdm_nfft muss > 0 sein.")
        if ofdm_cp_len < 0:
            raise ValueError("ofdm_cp_len muss >= 0 sein.")
        if ofdm_cp_len >= ofdm_nfft:
            raise ValueError("ofdm_cp_len muss < ofdm_nfft sein.")
        if ofdm_active_subcarriers <= 0:
            raise ValueError("ofdm_active_subcarriers muss > 0 sein.")
        if ofdm_active_subcarriers > ofdm_nfft - 1:
            raise ValueError("ofdm_active_subcarriers muss <= nfft-1 sein.")
        if ofdm_num_symbols <= 0:
            raise ValueError("ofdm_num_symbols muss > 0 sein.")
        if ofdm_short_repeats < 0:
            raise ValueError("ofdm_short_repeats muss >= 0 sein.")

        rng = np.random.default_rng(0)
        lower_count = ofdm_active_subcarriers // 2
        upper_count = ofdm_active_subcarriers - lower_count
        neg_bins = np.arange(ofdm_nfft - lower_count, ofdm_nfft)
        pos_bins = np.arange(1, upper_count + 1)
        active_bins = np.concatenate([neg_bins, pos_bins])

        bpsk = rng.choice([-1.0, 1.0], size=(ofdm_num_symbols, ofdm_active_subcarriers))
        symbols = []
        for idx in range(ofdm_num_symbols):
            spectrum = np.zeros(ofdm_nfft, dtype=np.complex64)
            spectrum[active_bins] = bpsk[idx].astype(np.complex64)
            spectrum[0] = 0.0
            time_symbol = np.fft.ifft(spectrum).astype(np.complex64)
            if ofdm_cp_len:
                cp = time_symbol[-ofdm_cp_len:]
                time_symbol = np.concatenate([cp, time_symbol]).astype(np.complex64)
            symbols.append(time_symbol)

        payload = np.concatenate(symbols).astype(np.complex64)
        if ofdm_short_repeats > 0:
            short_block = np.tile(symbols[0], ofdm_short_repeats).astype(np.complex64)
            payload = np.concatenate([short_block, payload]).astype(np.complex64)
        return payload.astype(np.complex64)

    # ---------- Pseudo-Noise -------------------------------------------------
    if w == "pseudo_noise":
        if pn_chip_rate <= 0:
            raise ValueError("pn_chip_rate muss > 0 sein.")
        samples_per_chip = max(1, int(round(fs / pn_chip_rate)))
        num_chips = int(np.ceil(N / samples_per_chip))
        rng = np.random.default_rng(pn_seed)
        chips = rng.choice([-1.0, 1.0], size=num_chips)
        tiled = np.repeat(chips, samples_per_chip)[:N]
        return tiled.astype(np.complex64)

    raise ValueError(f"Unbekannte Wellenform: {waveform}")


# ---------- Hauptprogramm ----------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
        "Erzeugt eine Wellen­folge (sinus, doppelsinus, zadoffchu, chirp, "
        "ofdm_preamble oder pseudo_noise) "
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
        choices=[
            "sinus",
            "doppelsinus",
            "zadoffchu",
            "chirp",
            "ofdm_preamble",
            "pseudo_noise",
        ],
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
        help="Frequenz in Hz – für sinus/doppelsinus (Standard: 1e6)",
    )
    parser.add_argument(
        "--f2",
        type=float,
        default=None,
        help="Zweite Frequenz in Hz – nur für doppelsinus (Standard: None)",
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

    # OFDM-spezifisch
    parser.add_argument(
        "--ofdm-nfft",
        type=int,
        default=64,
        help="OFDM-NFFT (Standard: 64)",
    )
    parser.add_argument(
        "--ofdm-cp",
        type=int,
        default=16,
        help="OFDM-CP-Länge (Standard: 16)",
    )
    parser.add_argument(
        "--ofdm-active",
        type=int,
        default=52,
        help="Aktive OFDM-Subträger (Standard: 52, DC bleibt 0)",
    )
    parser.add_argument(
        "--ofdm-symbols",
        type=int,
        default=2,
        help="OFDM-Symbole (Standard: 2)",
    )
    parser.add_argument(
        "--ofdm-short-repeats",
        type=int,
        default=10,
        help="OFDM-Short-Repeats für erstes Symbol (Standard: 10)",
    )
    parser.add_argument(
        "--pn-chip-rate",
        type=float,
        default=1e6,
        help="PN Chip-Rate in Hz (Standard: 1e6)",
    )
    parser.add_argument(
        "--pn-seed",
        type=int,
        default=1,
        help="PN Seed (Standard: 1)",
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
    if args.waveform == "doppelsinus" and args.f2 is None:
        raise ValueError("Für Doppelsinus muss --f2 gesetzt sein.")
    if args.waveform == "ofdm_preamble":
        if args.ofdm_nfft <= 0:
            raise ValueError("--ofdm-nfft muss > 0 sein.")
        if args.ofdm_cp < 0:
            raise ValueError("--ofdm-cp muss >= 0 sein.")
        if args.ofdm_cp >= args.ofdm_nfft:
            raise ValueError("--ofdm-cp muss < --ofdm-nfft sein.")
        if args.ofdm_active <= 0:
            raise ValueError("--ofdm-active muss > 0 sein.")
        if args.ofdm_active > args.ofdm_nfft - 1:
            raise ValueError("--ofdm-active muss <= nfft-1 sein.")
        if args.ofdm_symbols <= 0:
            raise ValueError("--ofdm-symbols muss > 0 sein.")
        if args.ofdm_short_repeats < 0:
            raise ValueError("--ofdm-short-repeats muss >= 0 sein.")

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
    elif args.waveform == "ofdm_preamble":
        N_output = (
            (args.ofdm_nfft + args.ofdm_cp)
            * (args.ofdm_symbols + args.ofdm_short_repeats)
        )

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
    f1_value = args.f1 if args.waveform != "doppelsinus" else args.f2
    waveform_signal = generate_waveform(
        args.waveform,
        args.fs,
        args.f,
        N_waveform,
        args.q,
        f0=args.f0,
        f1=f1_value,
        rrc_beta=args.rrc_beta,
        rrc_span=args.rrc_span,
        oversampling=args.oversampling,
        ofdm_nfft=args.ofdm_nfft,
        ofdm_cp_len=args.ofdm_cp,
        ofdm_active_subcarriers=args.ofdm_active,
        ofdm_num_symbols=args.ofdm_symbols,
        ofdm_short_repeats=args.ofdm_short_repeats,
        pn_chip_rate=args.pn_chip_rate,
        pn_seed=args.pn_seed,
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
    interleaved = complex_to_interleaved_int16(scaled)
    interleaved.tofile(args.filename)

    print("-" * 40)
    print(f"Datei:          {args.filename}")
    print(f"Samples gesamt: {total_samples}")
    print(f"Amplitude Ziel: {args.amplitude:.0f} (Skalierung: {scale:.2f})")
    print("-" * 40)


if __name__ == "__main__":
    main()
