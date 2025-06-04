#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Frequenzspektrum eines RX-Binärfiles anzeigen"
    )
    parser.add_argument("filename", type=str,
                        help="Pfad zur RX-Binärdatei (int16, 2 Kanäle: I und Q)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Anzahl der zu plottenden Samples (Standard: alle)")
    parser.add_argument("--rate", type=float, default=None,
                        help="Abtastrate in Hz (für richtige Frequenzachse)")
    args = parser.parse_args()

    # Datei einlesen
    data = np.fromfile(args.filename, dtype=np.int16)
    if data.size % 2 != 0:
        raise ValueError("Datei enthält ungerade Anzahl von int16-Werten")

    # In (I,Q)-Paare umformen und komplexes Signal bilden
    iq = data.reshape((-1, 2))
    rx = iq[:,0] + 1j*iq[:,1]

    # Optional Samples beschränken
    if args.samples is not None:
        rx = rx[:args.samples]

    N = rx.size

    # FFT und zentrieren
    spec = np.fft.fftshift(np.fft.fft(rx))
    mag = 20*np.log10(np.abs(spec) / N)

    # Frequenzachse
    if args.rate:
        fs = args.rate
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
        xlabel = "Frequenz (Hz)"
    else:
        freqs = np.fft.fftshift(np.fft.fftfreq(N))
        xlabel = "Normierte Frequenz"

    # Plot
    plt.figure()
    plt.plot(mag)
    plt.title(f"Frequenzspektrum: {args.filename}")
    plt.xlabel(xlabel)
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

