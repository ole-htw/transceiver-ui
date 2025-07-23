#!/usr/bin/env python3
#
# Copyright 2017-2018 Ettus Research, a National Instruments Company
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
RX samples to file using Python API
"""

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import uhd
from uhd.usrp import dram_utils
from uhd.types import StreamCMD, StreamMode


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--args", default="", type=str)
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Zieldatei. Wenn nicht angegeben, wird automatisch ein Name vergeben.",
    )
    parser.add_argument(
        "--output-dir",
        default="signals/rx",
        help="Basisverzeichnis f\xc3\xbcr Auto-Dateinamen (Default: signals/rx)",
    )
    parser.add_argument("-f", "--freq", type=float, required=True)
    parser.add_argument("-r", "--rate", default=1e6, type=float)
    parser.add_argument("-d", "--duration", default=5.0, type=float)
    parser.add_argument("-c", "--channels", default=[0], nargs="+", type=int)
    parser.add_argument("-g", "--gain", type=int, default=10)
    parser.add_argument(
        "-n",
        "--numpy",
        default=False,
        action="store_true",
        help="Save output file in NumPy format (default: No)",
    )
    parser.add_argument(
        "--dram",
        action="store_true",
        help="If given, will attempt to stream via DRAM",
    )
    parser.add_argument(
        "--rrc",
        action="store_true",
        help="Apply Root-Raised-Cosine filtering to the received signal",
    )
    parser.add_argument(
        "--rrc-beta",
        type=float,
        default=0.25,
        help="RRC roll-off factor (default: 0.25)",
    )
    parser.add_argument(
        "--rrc-span",
        type=int,
        default=6,
        help="RRC filter span in symbols (default: 6; 0 disables)",
    )

    args = parser.parse_args()

    if args.output_file is None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"rx_{int(args.freq)}Hz_{stamp}.bin"
        args.output_file = str(Path(args.output_dir) / base)
    else:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    return args


def rrc_coeffs(beta: float, span: int, sps: int = 1) -> np.ndarray:
    """Return Root-Raised-Cosine filter coefficients."""
    N = span * sps
    t = np.arange(-N, N + 1) / sps
    h = np.zeros_like(t, dtype=np.float64)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-10:
            h[i] = (
                beta
                / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(
                np.pi * ti * (1 + beta)
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h**2))
    return h.astype(np.float32)


def multi_usrp_rx(args):
    """
    multi_usrp based RX example
    """
    usrp = uhd.usrp.MultiUSRP(args.args)
    num_samps = int(np.ceil(args.duration * args.rate))
    if not isinstance(args.channels, list):
        args.channels = [args.channels]
    samps = usrp.recv_num_samps(
        num_samps, args.freq, args.rate, args.channels, args.gain
    )
    if args.rrc and args.rrc_span > 0:
        h = rrc_coeffs(args.rrc_beta, args.rrc_span)
        if samps.ndim == 1:
            samps = np.convolve(samps, h, mode="same")
        else:
            samps = np.stack([np.convolve(ch, h, mode="same") for ch in samps])
    with open(args.output_file, "wb") as out_file:
        if args.numpy:
            np.save(out_file, samps, allow_pickle=False, fix_imports=False)
        else:
            samps.tofile(out_file)


def rfnoc_dram_rx(args):
    """
    rfnoc_graph + replay-block based RX example
    """
    # Init graph
    graph = uhd.rfnoc.RfnocGraph(args.args)
    num_samps = int(np.ceil(args.duration * args.rate))
    if graph.get_num_mboards() > 1:
        print(
            "ERROR: This example only supports DRAM streaming on a single "
            "motherboard."
        )
        return
    # Init radios and replay block
    available_radio_chans = [
        (radio_block_id, chan)
        for radio_block_id in graph.find_blocks("Radio")
        for chan in range(graph.get_block(radio_block_id).get_num_output_ports())
    ]
    radio_chans = [available_radio_chans[x] for x in args.channels]
    print("Receiving from radio channels:", end="")
    print("\n* ".join((f"{r}:{c}" for r, c in radio_chans)))
    dram = dram_utils.DramReceiver(graph, radio_chans, cpu_format="fc32")
    replay = dram.replay_blocks[0]
    print(f"Using replay block {replay.get_block_id()}")
    for (radio, radio_chan), ddc_info in zip(
        dram.radio_chan_pairs, dram.ddc_chan_pairs
    ):
        radio.set_rx_frequency(args.freq, radio_chan)
        radio.set_rx_gain(args.gain, radio_chan)
        if ddc_info:
            ddc, ddc_chan = ddc_info
            ddc.set_output_rate(args.rate, ddc_chan)
        else:
            radio.set_rate(args.rate)
    # Overwrite default memory regions to maximize available memory
    mem_per_ch = int(replay.get_mem_size() / len(args.channels))
    mem_regions = [
        (idx * mem_per_ch, mem_per_ch) for idx, _ in enumerate(args.channels)
    ]
    dram.mem_regions = mem_regions

    data = np.zeros((len(radio_chans), num_samps), dtype=np.complex64)
    stream_cmd = StreamCMD(StreamMode.num_done)
    stream_cmd.stream_now = True
    stream_cmd.num_samps = num_samps
    dram.issue_stream_cmd(stream_cmd)
    rx_md = uhd.types.RXMetadata()
    dram.recv(data, rx_md)
    if args.rrc and args.rrc_span > 0:
        h = rrc_coeffs(args.rrc_beta, args.rrc_span)
        for idx in range(data.shape[0]):
            data[idx] = np.convolve(data[idx], h, mode="same")
    with open(args.output_file, "wb") as out_file:
        if args.numpy:
            np.save(out_file, data, allow_pickle=False, fix_imports=False)
        else:
            data.tofile(out_file)


def main():
    """RX samples and write to file"""
    args = parse_args()

    if args.dram:
        rfnoc_dram_rx(args)
    else:
        multi_usrp_rx(args)


if __name__ == "__main__":
    main()
