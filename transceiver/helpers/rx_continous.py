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

import io
import os
import sys
import time
import argparse
import threading
import queue
from datetime import datetime

import numpy as np
import uhd


# -----------------------------
# Args / Utilities
# -----------------------------
def parse_args(argv=None):
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
    p.add_argument("--memory-only", action="store_true",
                   help="Disable file output; emit snippet arrays via callback "
                        "(direct call) or stdout framing (subprocess).")
    p.add_argument("--stdout-binary", action="store_true",
                   help="When used with --memory-only, write binary snippet frames to stdout. "
                        "Disabled by default to keep stdout text-only in subprocess mode.")
    p.add_argument("-o", "--output-prefix", required=False,
                   help="Prefix for snippet files (directory optional). Required unless "
                        "--memory-only is set. Example: /data/cap/snips/run1 -> "
                        "run1_s000001_ch0.npy")
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

    return p.parse_args(argv)


def lcm(a: int, b: int) -> int:
    import math
    return abs(a * b) // math.gcd(a, b)


def ensure_parent_dir(path_prefix: str):
    parent = os.path.dirname(os.path.abspath(path_prefix))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def log(message: str, *, memory_only: bool):
    stream = sys.stderr if memory_only else sys.stdout
    print(message, file=stream)


def emit_snippet(data: np.ndarray, *, port: int, snip_idx: int, ts: str,
                 args, callback):
    if not args.memory_only:
        out_fn = snippet_filename(args.output_prefix + "_" + ts, snip_idx, port, args.numpy)
        if args.numpy:
            with open(out_fn, "wb") as f:
                np.save(f, data)
        else:
            # raw dump (native endianness)
            data.tofile(out_fn)
        log(f"[s{snip_idx:06d} ch{port}] wrote {len(data)} items -> {out_fn}",
            memory_only=args.memory_only)
        return

    if callback is not None:
        callback(data=data, port=port, snip_idx=snip_idx, timestamp=ts)
        return

    if not args.stdout_binary:
        log(
            "[memory-only] stdout binary disabled; no snippet data emitted "
            "(use --stdout-binary to enable).",
            memory_only=True,
        )
        return

    header = f"SNIP {snip_idx} {port} {ts} {len(data)}\n"
    sys.stdout.buffer.write(header.encode("utf-8"))
    buf = io.BytesIO()
    np.save(buf, data)
    sys.stdout.buffer.write(buf.getvalue())
    sys.stdout.buffer.flush()

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


def connect_radios(graph, replay, radio_chan_pairs, freqs, gains, antennas, rate, *, memory_only: bool):
    if rate is None:
        rate = radio_chan_pairs[0][0].get_rate()
    log(f"Requested rate: {rate/1e6:.2f} Msps", memory_only=memory_only)

    actual_rate = None
    for replay_port_idx, (radio, chan) in enumerate(radio_chan_pairs):
        log(f"Connecting {radio.get_unique_id()}:{chan} -> {replay.get_unique_id()}:{replay_port_idx}",
            memory_only=memory_only)

        radio.set_rx_frequency(freqs[replay_port_idx % len(freqs)], chan)
        radio.set_rx_gain(gains[replay_port_idx % len(gains)], chan)
        if antennas is not None:
            radio.set_rx_antenna(antennas[replay_port_idx % len(antennas)], chan)

        log(f"--> Radio settings: fc={radio.get_rx_frequency(chan)/1e6:.2f} MHz, "
            f"gain={radio.get_rx_gain(chan)} dB, antenna={radio.get_rx_antenna(chan)}",
            memory_only=memory_only)

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
            log(f"Found DDC block on channel {chan}.", memory_only=memory_only)
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
                        snippet_bytes: int, guard_bytes: int, align: int,
                        wrapped: bool):
    """
    Compute one or two (offset,size) ranges that represent the last snippet_bytes
    ending at (record_pos - guard_bytes), with wrap-around inside [base, base+ring_bytes).
    All returned offsets/sizes are aligned to `align`.

    IMPORTANT: Before the first record_restart(), we do NOT allow wrap reads
    because the end-of-ring does not contain the newest data yet.
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

    # Before first restart: no wrap; only read when we have a full contiguous snippet
    if have < snippet_bytes and not wrapped:
        return []

    if have >= snippet_bytes:
        start = end - snippet_bytes
        return [(start, snippet_bytes)]

    # wrap (only valid once wrapped==True)
    head = have
    tail = snippet_bytes - head
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
            print("WARNING: Overflow during snippet download (possible host/network bottleneck).",
                  file=sys.stderr)
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

    replay.config_play(offset_bytes, size_bytes, port)

    num_items = size_bytes // item_size
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_items
    stream_cmd.stream_now = True

    # Wichtig: Replay-Port steuern (nicht der Streamer)
    replay.issue_stream_cmd(stream_cmd, port)

    return recv_exact_1ch(rx_streamer, num_items, dtype, pkt_items)


def port_download_worker(
    *,
    port: int,
    worker_args: dict,
    item_size: int,
    pkt_items: int,
    job_queue,
    result_queue,
    stop_event,
    port_lock,
):
    """Per-port producer/consumer worker that downloads snippets in-process."""
    try:
        graph = worker_args["graph"]
        replay = worker_args["replay"]
        stream_args = uhd.usrp.StreamArgs(worker_args["cpu_format"], "sc16")

        rx_streamer = graph.create_rx_streamer(1, stream_args)
        graph.connect(replay.get_unique_id(), port, rx_streamer, 0)
        graph.commit()

        replay.set_play_type("sc16", port)
        if worker_args["pkt_size"] is not None:
            ipp = max(1, int(worker_args["pkt_size"]) // item_size)
            replay.set_max_items_per_packet(ipp, port)

        dtype = np.complex64 if worker_args["cpu_format"] == "fc32" else np.uint32

        while not stop_event.is_set():
            try:
                job = job_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if job is None:
                break

            try:
                with port_lock:
                    parts = [
                        play_and_download_chunk(
                            replay=replay,
                            rx_streamer=rx_streamer,
                            port=port,
                            offset_bytes=off,
                            size_bytes=sz,
                            item_size=item_size,
                            dtype=dtype,
                            pkt_items=pkt_items,
                        )
                        for off, sz in job["ranges"]
                    ]

                data = np.concatenate(parts) if len(parts) > 1 else parts[0]
                result_queue.put({
                    "ok": True,
                    "port": port,
                    "snip_idx": job["snip_idx"],
                    "ts": job["ts"],
                    "data": data,
                })
            except Exception as e:
                result_queue.put({
                    "ok": False,
                    "port": port,
                    "snip_idx": job["snip_idx"],
                    "ts": job["ts"],
                    "error": str(e),
                })
    except Exception as e:
        result_queue.put({
            "ok": False,
            "port": port,
            "snip_idx": -1,
            "ts": "",
            "error": f"worker init failed: {e}",
        })
    finally:
        try:
            rx_streamer = locals().get("rx_streamer")
            if rx_streamer is not None:
                del rx_streamer
        except Exception:
            pass



def snippet_filename(prefix: str, snip_idx: int, port: int, use_numpy: bool):
    ext = "npy" if use_numpy else "dat"
    return f"{prefix}_s{snip_idx:06d}_ch{port}.{ext}"


def record_monitor(stop_evt: threading.Event,
                   replay, port: int,
                   ring_bytes: int, restart_margin_bytes: int,
                   lock: threading.Lock,
                   wrapped_flags: list):
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
                    wrapped_flags[port] = True

            # Optional: pull RX async metadata (overruns etc.)
            while True:
                md = replay_get_record_async_md(replay, 0.0)
                if md is None:
                    break
                if md.error_code != uhd.types.RXMetadataErrorCode.none:
                    print(f"[port {port}] record async md: {md.strerror()}")
        except Exception as e:
            print(f"[port {port}] monitor error: {e}", file=sys.stderr)

        time.sleep(0.01)



def replay_get_record_async_md(replay, timeout: float = 0.0):
    """
    UHD/pyuhd Kompatibilität:
    - "pythonisch": md = replay.get_record_async_metadata(timeout) -> md | None | False
    - "C++-like":   ok = replay.get_record_async_metadata(md, timeout) -> bool
    Gibt RXMetadata zurück oder None.
    """
    # Versuch 1: pythonische Variante (nur timeout)
    try:
        md = replay.get_record_async_metadata(timeout)
        if md is None or md is False:
            return None
        if isinstance(md, bool):
            return None
        return md
    except TypeError:
        pass  # vermutlich alte Signatur

    # Versuch 2: C++-artige Variante (md-out-parameter)
    md = uhd.types.RXMetadata()
    ok = replay.get_record_async_metadata(md, timeout)
    return md if ok else None


def main(callback=None, args=None, stop_event=None):
    args = parse_args() if args is None else args
    if not args.memory_only and not args.output_prefix:
        raise RuntimeError("--output-prefix is required unless --memory-only is set.")
    if args.output_prefix and not args.memory_only:
        ensure_parent_dir(args.output_prefix)

    graph = uhd.rfnoc.RfnocGraph(args.args)
    replay = uhd.rfnoc.ReplayBlockControl(graph.get_block(args.block))
    radio_chan_pairs = enumerate_radios(graph, args.radio_channels)

    rate = connect_radios(graph, replay, radio_chan_pairs,
                          args.freq, args.gain, args.antenna, args.rate,
                          memory_only=args.memory_only)
    log(f"Using rate: {rate/1e6:.3f} Msps", memory_only=args.memory_only)

    num_ports = len(radio_chan_pairs)
    wrapped_flags = [False] * num_ports

    graph.commit()

    # Replay memory layout: split memory into per-port strides (like your original script)
    mem_size = replay.get_mem_size()
    mem_stride = mem_size // num_ports

    # Replay memory alignment constraints
    word_size = int(replay.get_word_size())  # bytes

    # Types explizit setzen, bevor item sizes abgefragt werden
    for port in range(num_ports):
        replay.set_record_type("sc16", port)
        replay.set_play_type("sc16", port)

    item_size = int(replay.get_play_item_size(0))  # bytes per item
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
        log(f"WARNING: ring_bytes exceeds per-port stride; reducing to {ring_bytes} bytes",
            memory_only=args.memory_only)

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

    log(f"Replay mem total: {mem_size/1024/1024:.1f} MiB, stride/port: {mem_stride/1024/1024:.1f} MiB",
        memory_only=args.memory_only)
    log(f"Ring/port: {ring_bytes/1024/1024:.2f} MiB  (align={align} bytes)",
        memory_only=args.memory_only)
    log(f"Snippet: {snippet_bytes} bytes ({args.snippet_seconds}s), interval: {args.snippet_interval}s, guard: {guard_bytes} bytes",
        memory_only=args.memory_only)
    log(f"Restart margin: {restart_margin_bytes} bytes", memory_only=args.memory_only)

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
    log("Starting continuous RX stream into Replay...", memory_only=args.memory_only)
    for radio, chan in radio_chan_pairs:
        radio.issue_stream_cmd(stream_cmd, chan)

    # Background record monitors (wrap before full)
    stop_evt = stop_event or threading.Event()
    locks = [threading.Lock() for _ in range(num_ports)]

    monitors = []
    for port in range(num_ports):
        t = threading.Thread(
            target=record_monitor,
            args=(stop_evt, replay, port, ring_bytes, restart_margin_bytes, locks[port], wrapped_flags),
        )
        t.start()
        monitors.append(t)

    worker_args = {
        "graph": graph,
        "replay": replay,
        "cpu_format": args.cpu_format,
        "pkt_size": args.pkt_size,
    }
    job_queues = [queue.Queue() for _ in range(num_ports)]
    result_queue = queue.Queue()
    workers = []
    for port in range(num_ports):
        p = threading.Thread(
            target=port_download_worker,
            kwargs={
                "port": port,
                "worker_args": worker_args,
                "item_size": item_size,
                "pkt_items": pkt_items[port],
                "job_queue": job_queues[port],
                "result_queue": result_queue,
                "stop_event": stop_evt,
                "port_lock": locks[port],
            },
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Snippet loop
    log("Entering snippet download loop. Ctrl+C to stop.", memory_only=args.memory_only)
    snip_idx = 1
    next_t = time.monotonic() + args.snippet_interval
    last_positions = [None] * num_ports
    no_progress_counts = [0] * num_ports

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                stop_evt.set()

            if stop_evt.is_set():
                break
            if args.max_snippets and snip_idx > args.max_snippets:
                break

            now_m = time.monotonic()
            if now_m < next_t:
                time.sleep(min(0.05, next_t - now_m))
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            pending_results = 0
            for port in range(num_ports):
                with locks[port]:
                    pos = int(replay.get_record_position(port))
                    if last_positions[port] is not None and pos == last_positions[port]:
                        no_progress_counts[port] += 1
                        if no_progress_counts[port] == 1 or no_progress_counts[port] % 20 == 0:
                            log(
                                f"[s{snip_idx:06d} ch{port}] no new samples recorded yet; "
                                "waiting for replay buffer to fill",
                                memory_only=args.memory_only,
                            )
                        continue
                    last_positions[port] = pos
                    no_progress_counts[port] = 0
                    base = base_offsets[port]

                    ranges = compute_snip_ranges(
                        base=base,
                        ring_bytes=ring_bytes,
                        record_pos=pos,
                        snippet_bytes=snippet_bytes,
                        guard_bytes=guard_bytes,
                        align=align,
                        wrapped=wrapped_flags[port],
                    )

                    if not ranges:
                        log(f"[s{snip_idx:06d} ch{port}] not enough data yet",
                            memory_only=args.memory_only)
                        continue
                job_queues[port].put({
                    "port": port,
                    "snip_idx": snip_idx,
                    "ts": ts,
                    "ranges": ranges,
                })
                pending_results += 1

            received_results = 0
            while received_results < pending_results and not stop_evt.is_set():
                if stop_event is not None and stop_event.is_set():
                    stop_evt.set()
                    break
                try:
                    result = result_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if result.get("ok"):
                    data = result["data"]

                    emit_snippet(
                        data,
                        port=result["port"],
                        snip_idx=result["snip_idx"],
                        ts=result["ts"],
                        args=args,
                        callback=callback,
                    )
                    del data
                else:
                    log(
                        f"[s{result.get('snip_idx', -1):06d} ch{result.get('port', -1)}] "
                        f"worker error: {result.get('error', 'unknown error')}",
                        memory_only=args.memory_only,
                    )
                received_results += 1

            snip_idx += 1
            next_t += args.snippet_interval

    except KeyboardInterrupt:
        log("\nStopping...", memory_only=args.memory_only)

    finally:
        # Stop radio continuous streaming
        try:
            stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            stop_cmd.stream_now = True
            for radio, chan in radio_chan_pairs:
                radio.issue_stream_cmd(stop_cmd, chan)
        except Exception as e:
            log(f"Stop streaming error: {e}", memory_only=args.memory_only)

        stop_evt.set()

        for q in job_queues:
            q.put(None)

        for t in monitors:
            t.join(timeout=2.0)

        for p in workers:
            p.join(timeout=5.0)

        time.sleep(0.1)

        try:
            if graph is not None:
                graph.release()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
