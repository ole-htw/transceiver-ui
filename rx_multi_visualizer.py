#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools

def main():
    p = argparse.ArgumentParser(
        description=("Zeigt 1-4 parallele RX-Kanäle aus einer Bin-Datei "
                     "(Layout: I0 Q0 I1 Q1 …, int16)."))
    p.add_argument("filename", type=str, help="Pfad zur Binärdatei")
    p.add_argument("--channels", type=int, choices=range(1,5), default=1,
                   help="Anzahl der Kanäle in der Datei (1-4, Standard 1)")
    p.add_argument("--samples", type=int, default=None,
                   help="Anzahl komplexer Samples, die geplottet werden sollen")
    args = p.parse_args()

    ch        = args.channels
    col_count = 2 * ch              # I- und Q-Werte je Kanal

    # --- Datei laden --------------------------------------------------------
    raw = np.fromfile(args.filename, dtype=np.int16)
    if raw.size % col_count != 0:
        raise ValueError("Dateigröße passt nicht zu channel-Anzahl.")

    data = raw.reshape((-1, col_count))        # [samples , 2*channels]

    # optional kürzen
    if args.samples is not None:
        data = data[:args.samples, :]

    # --- Komplexe Arrays je Kanal bilden ------------------------------------
    sigs = []
    for k in range(ch):
        I = data[:, 2*k]
        Q = data[:, 2*k + 1]
        sigs.append(I + 1j*Q)

    # --- Plot ---------------------------------------------------------------
    fig, axes = plt.subplots(ch, 1, sharex=True,
                             figsize=(10, 2.5*ch), squeeze=False)
    fig.canvas.manager.set_window_title(args.filename)

    colours = itertools.cycle(("tab:blue", "tab:orange",
                               "tab:green", "tab:red"))

    for idx, (ax, sig, color) in enumerate(zip(axes.flat, sigs, colours)):
        ax.plot(np.real(sig), label=f"Ch {idx} I", color=color, lw=0.7)
        ax.plot(np.imag(sig), label=f"Ch {idx} Q",
                color=color, ls="--", lw=0.7)
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1, 0].set_xlabel("Sample Index")
    fig.suptitle("RX Waveforms (komplex)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

