#!/usr/bin/env python3
"""
rx_instfreq.py – Instantane Frequenz aus I/Q-Binärdatei plotten

Dateiformat: int16, interleaved (I,Q,I,Q,…)
Aufrufbeispiel:
    python rx_instfreq.py capture.bin --rate 200e6 --samples 400000 --smooth 201
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def instantaneous_frequency(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Liefert f_inst[n] für jedes Sample-Intervall (Länge = len(x)-1).
    Formel: f[n] = (fs / 2π) · Δφ[n],  φ = unwrap(angle(x))
    """
    phase = np.unwrap(np.angle(x))
    dphi = np.diff(phase)                 # rad pro Sample
    return fs * dphi / (2 * np.pi)        # Hz


def main() -> None:
    p = argparse.ArgumentParser(
        description="Instantane Frequenz eines RX-Binärfiles anzeigen")
    p.add_argument("filename", type=str,
                   help="Pfad zur RX-Binärdatei (int16, I/Q)")
    p.add_argument("--rate",   type=float, required=True,
                   help="Abtastrate fs in Hz (z. B. 200e6)")
    p.add_argument("--samples", type=int, default=None,
                   help="Anzahl auszuwertender Samples (default: alle)")
    p.add_argument("--smooth",  type=int, default=0,
                   help="Optionale gleitende Mittelung (Fensterbreite in Samples)")
    args = p.parse_args()

    # ---------- Datei einlesen ------------------------------------------------
    raw = np.fromfile(args.filename, dtype=np.int16)
    if raw.size % 2:
        raise ValueError("ungerade Anzahl int16 – I/Q-Paar fehlt")

    iq = raw.reshape((-1, 2))
    rx = iq[:, 0] + 1j * iq[:, 1]

    if args.samples:
        rx = rx[: args.samples]

    # ---------- Instantane Frequenz berechnen --------------------------------
    f_inst = instantaneous_frequency(rx, args.rate)

    # Glättung (einfaches Moving-Average) optional
    if args.smooth and args.smooth > 1:
        k = args.smooth
        kernel = np.ones(k) / k
        f_inst = np.convolve(f_inst, kernel, mode="same")

    # ---------- Zeitachse -----------------------------------------------------
    t = np.arange(f_inst.size) / args.rate  # Sekunden

    # ---------- Plot ----------------------------------------------------------
    plt.figure()
    plt.plot(t * 1e6, f_inst / 1e6)
    plt.xlabel("Zeit [µs]")
    plt.ylabel("Instantane Frequenz [MHz]")
    plt.title(f"Instantane Frequenz: {args.filename}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

