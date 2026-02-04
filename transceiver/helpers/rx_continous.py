#!/usr/bin/env python3
"""
Continuous rolling record into Replay DRAM (ring-buffer emulation) + periodic snippet downloads.

Key idea:
- Radio streams continuously into Replay (record buffer).
- A background monitor restarts recording (record_restart) before the buffer fills up,
  effectively overwriting old data -> ring-buffer behavior (emulated).
- A snippet loop periodically reads "the most recent" N seconds from Replay to host.

NOTE:
Replay will back-pressure when the record buffer is full; restarting avoids that. :contentReference[oaicite:1]{index=1}
"""

import os
import sys
import time
import argparse
import threading
from datetime import datetime

import numpy as np
import uhd


# -----------------------------
# Args / Utilities
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--args", "-a", type=str, default="",
                   help="Device args to use when connecting to the USRP.")
    p.add_argument("--block", "-b", type=str, default="0/Replay#0",
                   help="Replay block to use. Defaults to \"0/Replay#0\".")
    p.add_argument("-c", "--radio-channels", default=["0/Radio#0:0"], nargs="+",
                   help="List radios plus their channels (defaults to \"0/Radio#0:0\")")
    p.add_argument("-f", "--freq", type=float, required=True, nargs="+", help="Center frequency")
    p.add_argument("-g", "--gain", type=int, default=[10], nargs="+", help="Gain (dB)")
    p.add_argument("--antenna", nargs="+", help="Antenna")
    p.add_argument("-r", "--rate", type=float, help="Sampling rate (defaults to radio rate)")

    # Ring buffer params (per port)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ring-seconds", type=float,
                     help="Ring buffer length in seconds (per port).")
    grp.add_argument("--ring-bytes", type=int,
                     help="Ring buffer size in bytes (per port). Must fit into per-port stride.")

    # Snippet params
    p.add_argument("--snippet-seconds", type=float, required=True,
                   help="Snippet length in seconds.")
    p.add_argument("--snippet-interval", type=float, required=True,
                   help="How often to download snippets (seconds).")

    # Safety / alignment
    p.add_argument("--guard-seconds", type=float, default=0.01,
                   help="Safety guard behind current write pointer before reading (seconds).")
    p.add_argument("--restart-margin-seconds", type=float, default=None,
                   help="Restart recording this many seconds before the buffer fills. "
                        "Default: max(snippet+guard, 0.25*ring).")

    # Output
    p.add_argument("-o", "--output-prefix", required=True,
                   help="Prefix for snippet files (directory optional). "
                        "Example: /data/cap/snips/run1 -> run1_s000001_ch0.npy")
    p.add_argument("-n", "--numpy", default=False, action="store_true",
                   help="Save snippets as .npy (default: raw .dat).")
    p.add_argument("--cpu-format", default="sc16", choices=["sc16", "fc32"],
                   help="Host data format for received snippets.")
    p.add_argument("--pkt-size", type=int, default=None,
                   help="Max packet size in bytes for Replay playback (optional).")

    p.add_argument("--max-snippets", type=int, default=0,
                   help="Stop after this many snippet rounds (0 = run forever).")

    p.add_argument("--delay", type=float, default=0.5,
                   help="Start streaming after this delay (seconds).")

    return p.parse_args()


def lcm(a: int, b: int) -> int:
    import math
    return abs(a * b) // math.gcd(a, b)


def ensure_parent_dir(path_prefix: str):
    parent = os.path.dirname(os.path.abspath(path_prefix))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# -----------------------------
# RFNoC Graph helpers (mostly from your script)
# -----------------------------
def enumerate_radios(graph, radio_chans):
    radio_id_chan_pairs = [
        (r.split(':', 2)[0], int(r.split(':', 2)[1])) if ':' in r else (r, 0)
        for r in radio_chans
    ]
    available_radios = graph.find_blocks("Radio")
    radio_chan_pairs = []
    for bid, chan in radio_id_chan_pairs:
        if bid not in available_radios:
            raise RuntimeError(f"'{bid}' is not a valid radio block ID!")
        radio_chan_pairs.append((uhd.rfnoc.RadioControl(graph.get_block(bid)), chan))
    return radio_chan_pairs


def connect_radios(graph, replay, radio_chan_pairs, freqs, gains, antennas, rate):
    if rate is None:
        rate = radio_chan_pairs[0][0].get_rate()
    print(f"Requested rate: {rate/1e6:.2f} Msps")

    actual_rate = None
    for replay_port_idx, (radio, chan) in enumerate(radio_chan_pairs):
        print(f"Connecting {radio.get_unique_id()}:{chan} -> {replay.get_unique_id()}:{replay_port_idx}")

        radio.set_rx_frequency(freqs[replay_port_idx % len(freqs)], chan)
        radio.set_rx_gain(gains[replay_port_idx % len(gains)], chan)
        if antennas is not None:
            radio.set_rx_antenna(antennas[replay_port_idx % len(antennas)], chan)

        print(f"--> Radio settings: fc={radio.get_rx_frequency(chan)/1e6:.2f} MHz, "
              f"gain={radio.get_rx_gain(chan)} dB, antenna={radio.get_rx_antenna(chan)}")

        radio_to_replay_graph = uhd.rfnoc.connect_through_blocks(
            graph,
            radio.get_unique_id(), chan,
            replay.get_unique_id(), replay_port_idx
        )

        ddc_block = next((
            (x.dst_blockid, x.dst_port)
            for x in radio_to_replay_graph
            if uhd.rfnoc.BlockID(x.dst_blockid).get_block_name() == 'DDC'
        ), None)

        if ddc_block is not None:
            print(f"Found DDC block on channel {chan}.")
            this_rate = uhd.rfnoc.DdcBlockControl(graph.get_block(ddc_block[0])).set_output_rate(rate, chan)
        else:
            this_rate = radio.set_rate(rate)

        if actual_rate is None:
            actual_rate = this_rate
        elif actual_rate != this_rate:
            raise RuntimeError("Unexpected rate mismatch across channels.")

    return actual_rate


# -----------------------------
# Ring record + snippet download
# -----------------------------
def compute_snip_ranges(base: int, ring_bytes: int, record_pos: int,
                        snippet_bytes: int, guard_bytes: int, align: int):
    """
    Compute one or two (offset,size) ranges that represent the last snippet_bytes
    ending at (record_pos - guard_bytes), with wrap-around inside [base, base+ring_bytes).
    All returned offsets/sizes are aligned to `align`.
    """
    end = record_pos - guard_bytes
    if end < base:
        return []  # not enough data yet
    if end > base + ring_bytes:
        end = base + ring_bytes

    end = base + ((end - base) // align) * align
    if end <= base:
        return []

    have = end - base
    if have >= snippet_bytes:
        start = end - snippet_bytes
        return [(start, snippet_bytes)]

    # wrap: take tail from end of ring + head from beginning
    head = have                      # bytes available at beginning up to 'end'
    tail = snippet_bytes - head      # remaining bytes from ring end
    tail_start = (base + ring_bytes) - tail
    return [(tail_start, tail), (base, head)]


def recv_exact_1ch(rx_streamer, num_items: int, dtype, pkt_items: int):
    """
    Receive exactly num_items samples from a 1-channel rx_streamer.
    Returns a 1D numpy array of length num_items.
    """
    md = uhd.types.RXMetadata()
    out = np.empty((1, num_items), dtype=dtype)
    tmp = np.empty((1, pkt_items), dtype=dtype)

    got = 0
    while got < num_items:
        n = rx_streamer.recv(tmp, md, 1.0)
        if md.error_code == uhd.types.RXMetadataErrorCode.timeout:
            raise RuntimeError("Snippet download recv() timed out.")
        if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
            # For host downloads from Replay this should be rare, but handle it.
            print("WARNING: Overflow during snippet download (possible host/network bottleneck).")
        elif md.error_code != uhd.types.RXMetadataErrorCode.none:
            raise RuntimeError(f"recv() error: {md.strerror()}")

        if n > 0:
            take = min(n, num_items - got)
            out[:, got:got + take] = tmp[:, :take]
            got += take

    return out[0].copy()


def play_and_download_chunk(replay, rx_streamer, port: int,
                            offset_bytes: int, size_bytes: int,
                            item_size: int, dtype, pkt_items: int):
    """
    Configure Replay playback for (offset,size) on output port `port`,
    then download exactly that chunk through rx_streamer (1ch).
    """
    if size_bytes <= 0:
        return np.empty((0,), dtype=dtype)

    # Configure playback window for this port
    replay.config_play(offset_bytes, size_bytes, port)

    num_items = size_bytes // item_size
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_items
    stream_cmd.stream_now = False
    stream_cmd.time_spec = uhd.types.TimeSpec(0.0)

    # Issue command via streamer (as in your original script)
    rx_streamer.issue_stream_cmd(stream_cmd)
    return recv_exact_1ch(rx_streamer, num_items, dtype, pkt_items)


def snippet_filename(prefix: str, snip_idx: int, port: int, use_numpy: bool):
    ext = "npy" if use_numpy else "dat"
    return f"{prefix}_s{snip_idx:06d}_ch{port}.{ext}"


def record_monitor(stop_evt: threading.Event,
                   replay, port: int,
                   ring_bytes: int, restart_margin_bytes: int,
                   lock: threading.Lock):
    """
    Restart recording before the buffer fills to emulate ring-buffer overwrite.
    Uses the same lock as snippet reads to avoid restarting mid-read.
    """
    while not stop_evt.is_set():
        try:
            fullness = replay.get_record_fullness(port)
            if fullness >= ring_bytes - restart_margin_bytes:
                with lock:
                    replay.record_restart(port)
            # Optional: pull RX async metadata (overruns etc.)
            md = uhd.types.RXMetadata()
            while replay.get_record_async_metadata(md, 0.0):
                # Only print if something noteworthy
                if md.error_code != uhd.types.RXMetadataErrorCode.none:
                    print(f"[port {port}] record async md: {md.strerror()}")
        except Exception as e:
            print(f"[port {port}] monitor error: {e}")
        time.sleep(0.01)


def main():
    args = parse_args()
    ensure_parent_dir(args.output_prefix)

    graph = uhd.rfnoc.RfnocGraph(args.args)
    replay = uhd.rfnoc.ReplayBlockControl(graph.get_block(args.block))
    radio_chan_pairs = enumerate_radios(graph, args.radio_channels)

    rate = connect_radios(graph, replay, radio_chan_pairs,
                          args.freq, args.gain, args.antenna, args.rate)
    print(f"Using rate: {rate/1e6:.3f} Msps")

    num_ports = len(radio_chan_pairs)

    # Streamer args: wire format from Replay is sc16; CPU format selectable
    stream_args = uhd.usrp.StreamArgs(args.cpu_format, "sc16")

    # Create 1-channel rx_streamer per port so each port can have independent config_play()
    rx_streamers = []
    for port in range(num_ports):
        s = graph.create_rx_streamer(1, stream_args)
        graph.connect(replay.get_unique_id(), port, s, 0)
        rx_streamers.append(s)

    graph.commit()

    # Replay memory layout: split memory into per-port strides (like your original script)
    mem_size = replay.get_mem_size()
    mem_stride = mem_size // num_ports

    # Replay memory alignment constraints
    word_size = int(replay.get_word_size())  # bytes
    # Replay stores sc16 in memory; item size depends on play_type/record_type
    replay_item_size = int(replay.get_record_item_size(0))  # bytes per item (after we set type)
    # We'll set types explicitly per port to be consistent
    for port in range(num_ports):
        replay.set_record_type("sc16", port)
        replay.set_play_type("sc16", port)

    # Item size for playback (should match record type)
    item_size = int(replay.get_play_item_size(0))
    align = lcm(word_size, item_size)

    # Determine ring buffer size per port
    if args.ring_seconds is not None:
        ring_bytes = int(args.ring_seconds * rate * item_size)
    else:
        ring_bytes = int(args.ring_bytes)

    ring_bytes = (ring_bytes // align) * align
    if ring_bytes <= 0:
        raise RuntimeError("ring_bytes ends up <= 0 after alignment.")

    if ring_bytes > mem_stride:
        ring_bytes = (mem_stride // align) * align
        print(f"WARNING: ring_bytes exceeds per-port stride; reducing to {ring_bytes} bytes")

    # Snippet sizes
    snippet_bytes = int(args.snippet_seconds * rate * item_size)
    guard_bytes = int(args.guard_seconds * rate * item_size)

    snippet_bytes = max(align, (snippet_bytes // align) * align)
    guard_bytes = max(0, (guard_bytes // align) * align)

    if snippet_bytes + guard_bytes > ring_bytes:
        raise RuntimeError(
            f"snippet_bytes({snippet_bytes}) + guard_bytes({guard_bytes}) > ring_bytes({ring_bytes}). "
            "Increase ring size or reduce snippet/guard."
        )

    # Restart margin (bytes)
    if args.restart_margin_seconds is None:
        # default: enough room for one full snippet read + guard, but at least 25% of ring
        restart_margin_bytes = max(snippet_bytes + guard_bytes, ring_bytes // 4)
    else:
        restart_margin_bytes = int(args.restart_margin_seconds * rate * item_size)
        restart_margin_bytes = max(align, (restart_margin_bytes // align) * align)

    if restart_margin_bytes >= ring_bytes:
        restart_margin_bytes = ring_bytes // 2

    print(f"Replay mem total: {mem_size/1024/1024:.1f} MiB, stride/port: {mem_stride/1024/1024:.1f} MiB")
    print(f"Ring/port: {ring_bytes/1024/1024:.2f} MiB  (align={align} bytes)")
    print(f"Snippet: {snippet_bytes} bytes ({args.snippet_seconds}s), interval: {args.snippet_interval}s, guard: {guard_bytes} bytes")
    print(f"Restart margin: {restart_margin_bytes} bytes")

    # Configure record buffers per port (start at each stride base)
    base_offsets = []
    for port in range(num_ports):
        base = port * mem_stride
        base = (base // align) * align
        base_offsets.append(base)
        replay.record(base, ring_bytes, port)

    # Packet sizing for playback (optional)
    for port in range(num_ports):
        if args.pkt_size is not None:
            ipp = max(1, args.pkt_size // item_size)
            replay.set_max_items_per_packet(ipp, port)

    # Determine receive packet items
    pkt_items = []
    for port in range(num_ports):
        ipp = int(replay.get_max_items_per_packet(port))
        if ipp <= 0:
            ipp = int(replay.get_max_packet_size(port)) // item_size
        pkt_items.append(max(1, ipp))

    # Start continuous streaming from radios into Replay
    now = graph.get_mb_controller().get_timekeeper(0).get_time_now()
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = False
    stream_cmd.time_spec = now + uhd.types.TimeSpec(args.delay)
    print("Starting continuous RX stream into Replay...")
    for radio, chan in radio_chan_pairs:
        radio.issue_stream_cmd(stream_cmd, chan)

    # Background record monitors (wrap before full)
    stop_evt = threading.Event()
    locks = [threading.Lock() for _ in range(num_ports)]
    monitors = []
    for port in range(num_ports):
        t = threading.Thread(
            target=record_monitor,
            args=(stop_evt, replay, port, ring_bytes, restart_margin_bytes, locks[port]),
            daemon=True,
        )
        t.start()
        monitors.append(t)

    # Host dtype for snippet arrays
    # (Keep original behavior: sc16 as packed uint32; fc32 as complex64)
    if args.cpu_format == "fc32":
        dtype = np.complex64
    else:
        dtype = np.uint32

    # Snippet loop
    print("Entering snippet download loop. Ctrl+C to stop.")
    snip_idx = 1
    next_t = time.monotonic() + args.snippet_interval

    try:
        while True:
            if args.max_snippets and snip_idx > args.max_snippets:
                break

            now_m = time.monotonic()
            if now_m < next_t:
                time.sleep(min(0.05, next_t - now_m))
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            for port in range(num_ports):
                with locks[port]:
                    pos = int(replay.get_record_position(port))
                    base = base_offsets[port]

                    ranges = compute_snip_ranges(
                        base=base,
                        ring_bytes=ring_bytes,
                        record_pos=pos,
                        snippet_bytes=snippet_bytes,
                        guard_bytes=guard_bytes,
                        align=align,
                    )

                    if not ranges:
                        print(f"[s{snip_idx:06d} ch{port}] not enough data yet")
                        continue

                    parts = []
                    for off, sz in ranges:
                        parts.append(
                            play_and_download_chunk(
                                replay=replay,
                                rx_streamer=rx_streamers[port],
                                port=port,
                                offset_bytes=off,
                                size_bytes=sz,
                                item_size=item_size,
                                dtype=dtype,
                                pkt_items=pkt_items[port],
                            )
                        )
                    data = np.concatenate(parts) if len(parts) > 1 else parts[0]

                out_fn = snippet_filename(args.output_prefix + "_" + ts, snip_idx, port, args.numpy)
                if args.numpy:
                    with open(out_fn, "wb") as f:
                        np.save(f, data)
                else:
                    # raw dump (native endianness)
                    data.tofile(out_fn)

                print(f"[s{snip_idx:06d} ch{port}] wrote {len(data)} items -> {out_fn}")

            snip_idx += 1
            next_t += args.snippet_interval

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Stop radio continuous streaming
        try:
            stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            stop_cmd.stream_now = True
            for radio, chan in radio_chan_pairs:
                radio.issue_stream_cmd(stop_cmd, chan)
        except Exception as e:
            print(f"Stop streaming error: {e}")

        stop_evt.set()
        time.sleep(0.1)

    return 0


if __name__ == "__main__":
    sys.exit(main())

