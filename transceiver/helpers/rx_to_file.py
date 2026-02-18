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
import sys
import numpy as np
import uhd
from uhd.usrp import dram_utils
from uhd.types import StreamCMD, StreamMode


def parse_args(argv=None):
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
    args = parser.parse_args(argv)

    if args.output_file is None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"rx_{int(args.freq)}Hz_{stamp}.bin"
        args.output_file = str(Path(args.output_dir) / base)
    else:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    return args


def ensure_uhd_frame_sizes(args_str, recv_frame_size=8000, send_frame_size=8000):
    """Ensure recv/send frame sizes are present in UHD device args."""
    components = [part for part in args_str.split(",") if part]
    if not any(part.startswith("recv_frame_size=") for part in components):
        components.append(f"recv_frame_size={recv_frame_size}")
    if not any(part.startswith("send_frame_size=") for part in components):
        components.append(f"send_frame_size={send_frame_size}")
    return ",".join(components)


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
        radio.set_rx_antenna("TX/RX", radio_chan)
        print(
            f"RX antenna for channel {radio_chan}: {radio.get_rx_antenna(radio_chan)}",
            file=sys.stderr,
        )
        radio.set_rx_frequency(args.freq, radio_chan)
        radio.set_rx_gain(args.gain, radio_chan)
        if ddc_info:
            ddc, ddc_chan = ddc_info
            ddc.set_output_rate(args.rate, ddc_chan)
        else:
            radio.set_rate(args.rate)
    # Overwrite default memory regions to maximize available memory
    mem_per_ch = int(replay.get_mem_size() / len(args.channels))
    bytes_per_sample = 8  # fc32: complex64 = 8 bytes per sample
    max_samps_per_ch = mem_per_ch // bytes_per_sample
    if num_samps > max_samps_per_ch:
        requested_samps = args.duration * args.rate
        print(
            "WARNING: Requested samples exceed DRAM capacity per channel "
            f"({requested_samps:.0f} > {max_samps_per_ch}). "
            "Clamping to available DRAM capacity.",
            file=sys.stderr,
        )
        num_samps = max_samps_per_ch
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
    with open(args.output_file, "wb") as out_file:
        if args.numpy:
            np.save(out_file, data, allow_pickle=False, fix_imports=False)
        else:
            data.tofile(out_file)


def main(args=None):
    """RX samples and write to file"""
    args = args or parse_args()
    args.args = ensure_uhd_frame_sizes(args.args)

    if args.dram:
        rfnoc_dram_rx(args)
    else:
        multi_usrp_rx(args)


if __name__ == "__main__":
    main()
