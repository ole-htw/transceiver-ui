#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Visualisierung eines RX Binärfiles analog zum MATLAB-Skript"
    )
    parser.add_argument("filename", type=str,
                        help="Pfad zur RX Binärdatei")
    parser.add_argument("--samples", type=int, default=None,
                        help="Anzahl der zu plottenden Samples (Standard: Alle Samples)")
    parser.add_argument("--format", type=str, choices=["int16", "complex64"], default="int16",
                        help="Eingabeformat der Datei: 'int16' für sc16 oder 'complex64' für fc32 (default: int16)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Skalierungsfaktor für die Amplitude (default: 1.0)")
    args = parser.parse_args()

    try:
        if args.format == "int16":
            # Einlesen der int16-Daten (sc16)
            data = np.fromfile(args.filename, dtype=np.int16)
            if data.size % 2 != 0:
                print("Die Anzahl der Datenwerte ist ungerade. Die Datei hat vermutlich ein falsches Format.")
                return
            data = data.reshape((-1, 2))
            rx_sig = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)
        else:
            # Einlesen der complex64-Daten (fc32)
            rx_sig = np.fromfile(args.filename, dtype=np.complex64)
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei: {e}")
        return

    # Falls --samples angegeben wurde, nur diese Anzahl an Samples plotten
    if args.samples is not None:
        rx_sig = rx_sig[:args.samples]

    # Skalierung anwenden
    rx_sig = rx_sig * args.scale

    # Erstellen des Plots und setzen des Fenstertitels auf den Dateinamen
    fig = plt.figure()
    fig.canvas.manager.set_window_title(args.filename)
    plt.plot(np.real(rx_sig), label='Realteil')
    plt.plot(np.imag(rx_sig), label='Imaginärteil')
    plt.grid(True)
    plt.xlim(0, len(rx_sig))
    plt.title('Raw RX waveform')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

