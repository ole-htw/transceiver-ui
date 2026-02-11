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
import multiprocessing as mp
import queue
import inspect
import traceback
import contextlib
from _thread import LockType
from datetime import datetime

import numpy as np
import uhd


class SnippetRecvTimeoutError(RuntimeError):
    """Raised when Replay snippet downloads time out on recv()."""


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
    p.add_argument("--allow-wrap-read", action="store_true",
                   help="Allow snippet reads that cross the replay ring wrap boundary. "
                        "Disabled by default, so only fully contiguous snippets are emitted.")

    # Safety / alignment
    p.add_argument("--guard-seconds", type=float, default=0.01,
                   help="Safety guard behind current write pointer before reading (seconds).")
    p.add_argument("--restart-margin-seconds", type=float, default=None,
                   help="Restart recording this many seconds before the buffer fills. "
                        "Default: max(snippet+guard+safety, 0.25*ring), "
                        "safety defaults to 15% of ring.")

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
    p.add_argument("--no-progress-recovery-threshold", type=int, default=50,
                   help="Trigger recovery if record position does not advance this many checks.")
    p.add_argument("--recovery-retry-delay", type=float, default=0.05,
                   help="Delay after a recovery action before next snippet attempt (seconds).")
    p.add_argument("--recovery-rearm-record", action="store_true",
                   help="During recovery also re-arm Replay recording via replay.record(...).")
    p.add_argument("--graph-reset-after-recovery-failures", type=int, default=0,
                   help="Optional full streaming reset after N failed recovery attempts per port (0=disabled).")

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


@contextlib.contextmanager
def timed_lock(lock: LockType, *, path: str, memory_only: bool,
               contention_threshold_s: float = 0.001):
    """
    Acquire `lock` and log observed contention per lock path.

    Thread-safety assumption:
    - We only serialize the minimal UHD Replay calls that mutate per-port replay state.
    - Waiting time is measured around lock acquire to validate real-world parallelism gains.
    """
    wait_start = time.monotonic()
    lock.acquire()
    wait_s = time.monotonic() - wait_start
    if wait_s >= contention_threshold_s:
        log(
            f"[lock:{path}] contention observed, waited {wait_s * 1e3:.2f} ms",
            memory_only=memory_only,
        )
    try:
        yield
    finally:
        lock.release()


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


def _build_worker_args_dict(args):
    return {
        "args": args.args,
        "block": args.block,
        "cpu_format": args.cpu_format,
        "memory_only": args.memory_only,
        "stdout_binary": args.stdout_binary,
        "output_prefix": args.output_prefix,
        "numpy": args.numpy,
    }


def _read_worker_loop(
    port: int,
    req_q,
    res_q,
    replay,
    rx_streamer,
    replay_port_lock: LockType,
    *,
    cpu_format: str,
    memory_only: bool,
):
    try:
        item_size = int(replay.get_play_item_size(0))
        replay.set_play_type("sc16", port)
        ipp = int(replay.get_max_items_per_packet(port))
        if ipp <= 0:
            ipp = int(replay.get_max_packet_size(port)) // item_size
        pkt_items = max(1, ipp)
        dtype = np.complex64 if cpu_format == "fc32" else np.uint32

        while True:
            msg = req_q.get()
            if msg.get("kind") == "shutdown":
                break

            result = {
                "kind": "read_result",
                "port": port,
                "snip_idx": msg.get("snip_idx"),
                "timestamp": msg.get("timestamp"),
                "ranges": msg.get("ranges", []),
                "status": "ok",
                "error": None,
                "data": None,
            }
            try:
                parts = []
                for r in msg.get("ranges", []):
                    parts.append(
                        play_and_download_chunk(
                            replay=replay,
                            rx_streamer=rx_streamer,
                            port=port,
                            offset_bytes=int(r["offset"]),
                            size_bytes=int(r["size"]),
                            item_size=item_size,
                            dtype=dtype,
                            pkt_items=pkt_items,
                            replay_port_lock=replay_port_lock,
                            lock_path=f"port{port}/config_play+issue_stream_cmd.read_worker",
                            memory_only=memory_only,
                        )
                    )
                if not parts:
                    result["data"] = np.empty((0,), dtype=dtype)
                elif len(parts) == 1:
                    result["data"] = parts[0]
                else:
                    result["data"] = np.concatenate(parts)
            except Exception as exc:
                result["status"] = "error"
                result["error"] = f"{type(exc).__name__}: {exc}"
                result["traceback"] = traceback.format_exc(limit=5)

            res_q.put(result)
    except Exception as exc:
        res_q.put(
            {
                "kind": "read_result",
                "port": port,
                "snip_idx": None,
                "timestamp": None,
                "ranges": [],
                "status": "error",
                "error": f"worker_init_failed: {type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
                "data": None,
            }
        )


def _emitter_worker_loop(emit_q, worker_args):
    args = argparse.Namespace(**worker_args)
    while True:
        msg = emit_q.get()
        if msg.get("kind") == "shutdown":
            break
        if msg.get("kind") != "emit_request":
            continue
        emit_snippet(
            msg["data"],
            port=msg["port"],
            snip_idx=msg["snip_idx"],
            ts=msg["timestamp"],
            args=args,
            callback=None,
        )


def _start_emitter_worker(args, callback):
    if callback is not None:
        def _thread_emitter():
            while True:
                msg = emit_q.get()
                if msg.get("kind") == "shutdown":
                    break
                if msg.get("kind") != "emit_request":
                    continue
                emit_snippet(
                    msg["data"],
                    port=msg["port"],
                    snip_idx=msg["snip_idx"],
                    ts=msg["timestamp"],
                    args=args,
                    callback=callback,
                )

        emit_q = queue.Queue(maxsize=16)
        emitter = threading.Thread(target=_thread_emitter, daemon=True)
        emitter.start()
        return emit_q, emitter, "thread"

    ctx = mp.get_context("spawn")
    emit_q = ctx.Queue(maxsize=16)
    emitter = ctx.Process(
        target=_emitter_worker_loop,
        args=(emit_q, _build_worker_args_dict(args)),
        daemon=True,
    )
    emitter.start()
    return emit_q, emitter, "process"

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
                        wrapped: bool, allow_wrap_read: bool = False):
    """
    Compute one or two (offset,size) ranges that represent the last snippet_bytes
    ending at (record_pos - guard_bytes), with wrap-around inside [base, base+ring_bytes).
    All returned offsets/sizes are aligned to `align`.

    IMPORTANT: Before the first record_restart(), we do NOT allow wrap reads
    because the end-of-ring does not contain the newest data yet.

    If allow_wrap_read is False, wrap snippets are skipped even after wrapped=True.
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

    if have < snippet_bytes and not allow_wrap_read:
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
            raise SnippetRecvTimeoutError(
                f"Snippet download recv() timed out after {got}/{num_items} items."
            )
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
                            item_size: int, dtype, pkt_items: int,
                            replay_port_lock: LockType | None = None,
                            lock_path: str | None = None,
                            memory_only: bool = False):
    """
    Configure Replay playback for (offset,size) on output port `port`,
    then download exactly that chunk through rx_streamer (1ch).

    Thread-safety assumption:
    - `config_play()` + `issue_stream_cmd()` are serialized per Replay output port.
    - Different ports may run these calls concurrently.
    """
    if size_bytes <= 0:
        return np.empty((0,), dtype=dtype)

    num_items = size_bytes // item_size
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_items
    stream_cmd.stream_now = True

    if replay_port_lock is None:
        replay.config_play(offset_bytes, size_bytes, port)
        # Wichtig: Replay-Port steuern (nicht der Streamer)
        replay.issue_stream_cmd(stream_cmd, port)
    else:
        path = lock_path or f"port{port}/config_play+issue_stream_cmd"
        with timed_lock(replay_port_lock, path=path, memory_only=memory_only):
            replay.config_play(offset_bytes, size_bytes, port)
            # Wichtig: Replay-Port steuern (nicht der Streamer)
            replay.issue_stream_cmd(stream_cmd, port)

    return recv_exact_1ch(rx_streamer, num_items, dtype, pkt_items)



def snippet_filename(prefix: str, snip_idx: int, port: int, use_numpy: bool):
    ext = "npy" if use_numpy else "dat"
    return f"{prefix}_s{snip_idx:06d}_ch{port}.{ext}"


def record_monitor(stop_evt: threading.Event,
                   replay, port: int,
                   ring_bytes: int, restart_margin_bytes: int,
                   wrapped_flags: list,
                   state_lock: threading.Lock,
                   replay_port_lock: threading.Lock,
                   restart_times: list,
                   memory_only: bool):
    """
    Restart recording before the buffer fills to emulate ring-buffer overwrite.

    Thread-safety assumption:
    - `record_restart(port)` is scoped to one Replay port and only needs a per-port lock.
    - Global serialization is intentionally avoided to allow independent port progress.
    """
    while not stop_evt.is_set():
        try:
            fullness = replay.get_record_fullness(port)
            if fullness >= ring_bytes - restart_margin_bytes:
                now_m = time.monotonic()
                with timed_lock(
                    replay_port_lock,
                    path=f"port{port}/record_restart.monitor",
                    memory_only=memory_only,
                ):
                    replay.record_restart(port)
                with state_lock:
                    wrapped_flags[port] = True
                    prev_restart = restart_times[port]
                    restart_times[port] = now_m
                if prev_restart is None:
                    log(f"[port {port}] record_restart() issued", memory_only=memory_only)
                else:
                    delta = now_m - prev_restart
                    log(
                        f"[port {port}] record_restart() issued, dt since last restart: {delta:.3f}s",
                        memory_only=memory_only,
                    )

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


def log_recovery_event(*, port: int, reason: str, position: int, fullness: int,
                       ring_bytes: int, restart_margin_bytes: int, snippet_bytes: int,
                       guard_bytes: int, attempts: int, memory_only: bool):
    log(
        f"[recovery ch{port}] reason={reason}, attempts={attempts}, pos={position}, "
        f"fullness={fullness}, ring={ring_bytes}, margin={restart_margin_bytes}, "
        f"snippet={snippet_bytes}, guard={guard_bytes}",
        memory_only=memory_only,
    )


def recover_port(*, replay, port: int, base: int, ring_bytes: int,
                 restart_margin_bytes: int, snippet_bytes: int, guard_bytes: int,
                 replay_port_lock: threading.Lock, wrapped_flags: list,
                 args, reason: str,
                 recovery_attempts: list, recovery_failures: list):
    """Recover one Replay port after stalled capture.

    Thread-safety assumption:
    - `record_restart(port)` and optional `record(base, ring_bytes, port)` only mutate one port.
      They are therefore protected by the same per-port lock.
    """
    recovery_attempts[port] += 1
    attempts = recovery_attempts[port]
    position = int(replay.get_record_position(port))
    fullness = int(replay.get_record_fullness(port))
    log_recovery_event(
        port=port,
        reason=reason,
        position=position,
        fullness=fullness,
        ring_bytes=ring_bytes,
        restart_margin_bytes=restart_margin_bytes,
        snippet_bytes=snippet_bytes,
        guard_bytes=guard_bytes,
        attempts=attempts,
        memory_only=args.memory_only,
    )

    try:
        with timed_lock(
            replay_port_lock,
            path=f"port{port}/recover.record_restart+record",
            memory_only=args.memory_only,
        ):
            replay.record_restart(port)
            if args.recovery_rearm_record:
                replay.record(base, ring_bytes, port)
        wrapped_flags[port] = True
        recovery_failures[port] = 0
        time.sleep(args.recovery_retry_delay)
        return True
    except Exception as exc:
        recovery_failures[port] += 1
        log(
            f"[recovery ch{port}] failed ({reason}), consecutive failures="
            f"{recovery_failures[port]}: {exc}",
            memory_only=args.memory_only,
        )
        return False


def reset_rfnoc_streaming(*, graph, replay, radio_chan_pairs, base_offsets,
                          ring_bytes: int, args,
                          replay_global_lock: threading.Lock,
                          replay_port_locks: list[LockType]):
    """Best-effort clean restart of continuous streaming and Replay record arming.

    Thread-safety assumption:
    - Graph stop/start is a global control-plane transition.
    - During replay re-arm we hold a short global lock and then each per-port lock to
      keep the global section minimal while avoiding races with per-port monitor/recovery.
    """
    log("[recovery] resetting RFNoC streaming graph state", memory_only=args.memory_only)
    stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    stop_cmd.stream_now = True
    for radio, chan in radio_chan_pairs:
        radio.issue_stream_cmd(stop_cmd, chan)

    with timed_lock(
        replay_global_lock,
        path="global/reset_rfnoc_streaming",
        memory_only=args.memory_only,
    ):
        for port, base in enumerate(base_offsets):
            with timed_lock(
                replay_port_locks[port],
                path=f"port{port}/reset.record",
                memory_only=args.memory_only,
            ):
                replay.record(base, ring_bytes, port)

    now = graph.get_mb_controller().get_timekeeper(0).get_time_now()
    start_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    start_cmd.stream_now = False
    start_cmd.time_spec = now + uhd.types.TimeSpec(args.delay)
    for radio, chan in radio_chan_pairs:
        radio.issue_stream_cmd(start_cmd, chan)

    time.sleep(max(args.recovery_retry_delay, 0.05))


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

    # Build Replay->Host playback streamers in-process. Multiprocess workers would
    # each open a separate UHD/RFNoC session and can fail to claim the same device
    # (e.g. worker_init_failed: No devices found for addr=...).
    read_streamers = []
    for port in range(num_ports):
        stream_args = uhd.usrp.StreamArgs(args.cpu_format, "sc16")
        rx_streamer = graph.create_rx_streamer(1, stream_args)
        graph.connect(replay.get_unique_id(), port, rx_streamer, 0)
        read_streamers.append(rx_streamer)

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

    snippet_interval_bytes = int(args.snippet_interval * rate * item_size)
    if snippet_interval_bytes <= 0:
        raise RuntimeError("snippet_interval is too small for the selected rate/item-size.")
    if snippet_bytes > snippet_interval_bytes:
        raise RuntimeError(
            f"snippet_bytes({snippet_bytes}) > bytes produced per interval({snippet_interval_bytes}). "
            "This will likely cause Replay backpressure. Reduce snippet_seconds or increase snippet_interval."
        )
    if snippet_bytes > int(0.8 * snippet_interval_bytes):
        log(
            "WARNING: snippet_seconds is close to snippet_interval at this rate "
            f"({snippet_bytes}/{snippet_interval_bytes} bytes per interval). "
            "This increases backpressure risk.",
            memory_only=args.memory_only,
        )
    if snippet_bytes + guard_bytes > ring_bytes // 2:
        log(
            "WARNING: snippet+guard uses more than 50% of ring buffer; "
            "consider a larger ring or shorter snippets to reduce replay pressure.",
            memory_only=args.memory_only,
        )

    # Restart margin (bytes)
    user_margin_missing = args.restart_margin_seconds is None
    if args.restart_margin_seconds is None:
        # default: enough room for one full snippet read + guard, but at least 25% of ring
        restart_margin_bytes = max(snippet_bytes + guard_bytes, ring_bytes // 4)
    else:
        restart_margin_bytes = int(args.restart_margin_seconds * rate * item_size)
        restart_margin_bytes = max(align, (restart_margin_bytes // align) * align)

    safety_bytes = max(align, ((ring_bytes * 15) // 100 // align) * align)
    min_restart_margin_bytes = snippet_bytes + guard_bytes + safety_bytes
    if min_restart_margin_bytes >= ring_bytes:
        raise RuntimeError(
            f"restart margin requirement is impossible: min_needed={min_restart_margin_bytes} >= ring={ring_bytes}. "
            "Increase ring size or reduce snippet/guard settings."
        )

    if restart_margin_bytes < min_restart_margin_bytes:
        reason = "missing" if user_margin_missing else "too small"
        log(
            "WARNING: restart_margin_seconds was "
            f"{reason}; raising restart margin to safe minimum {min_restart_margin_bytes} bytes "
            f"(snippet {snippet_bytes} + guard {guard_bytes} + safety {safety_bytes}).",
            memory_only=args.memory_only,
        )
        restart_margin_bytes = min_restart_margin_bytes

    if restart_margin_bytes >= ring_bytes:
        fallback_margin = max(align, (ring_bytes // 2 // align) * align)
        restart_margin_bytes = max(min_restart_margin_bytes, fallback_margin)
        log(
            "WARNING: restart margin exceeded ring buffer; clamping to 50% of ring.",
            memory_only=args.memory_only,
        )

    log(f"Replay mem total: {mem_size/1024/1024:.1f} MiB, stride/port: {mem_stride/1024/1024:.1f} MiB",
        memory_only=args.memory_only)
    log(f"Ring/port: {ring_bytes/1024/1024:.2f} MiB  (align={align} bytes)",
        memory_only=args.memory_only)
    log(f"Snippet: {snippet_bytes} bytes ({args.snippet_seconds}s), interval: {args.snippet_interval}s, guard: {guard_bytes} bytes",
        memory_only=args.memory_only)
    log(f"Safety bytes: {safety_bytes} bytes (15% of ring)", memory_only=args.memory_only)
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
    state_locks = [threading.Lock() for _ in range(num_ports)]
    replay_global_lock = threading.Lock()
    replay_port_locks = [threading.Lock() for _ in range(num_ports)]
    restart_times = [None] * num_ports
    monitors = []
    for port in range(num_ports):
        t = threading.Thread(
            target=record_monitor,
            args=(
                stop_evt,
                replay,
                port,
                ring_bytes,
                restart_margin_bytes,
                wrapped_flags,
                state_locks[port],
                replay_port_locks[port],
                restart_times,
                args.memory_only,
            ),
            daemon=True,
        )
        t.start()
        monitors.append(t)

    read_req_queues = [queue.Queue(maxsize=2) for _ in range(num_ports)]
    read_result_queue = queue.Queue(maxsize=max(8, num_ports * 4))
    read_workers = []
    for port in range(num_ports):
        proc = threading.Thread(
            target=_read_worker_loop,
            args=(
                port,
                read_req_queues[port],
                read_result_queue,
                replay,
                read_streamers[port],
                replay_port_locks[port],
            ),
            kwargs={
                "cpu_format": args.cpu_format,
                "memory_only": args.memory_only,
            },
            daemon=True,
        )
        proc.start()
        read_workers.append(proc)

    emit_q, emitter, emitter_kind = _start_emitter_worker(args, callback)

    # Snippet loop
    log("Entering snippet download loop. Ctrl+C to stop.", memory_only=args.memory_only)
    snip_idx = 1
    next_t = time.monotonic() + args.snippet_interval
    last_positions = [None] * num_ports
    no_progress_counts = [0] * num_ports
    recovery_attempts = [0] * num_ports
    recovery_failures = [0] * num_ports
    pending = {}

    def handle_read_result(result_msg):
        port = int(result_msg.get("port", -1))
        local_snip_idx = result_msg.get("snip_idx")
        pending.pop(port, None)
        status = result_msg.get("status")
        if status == "ok":
            data = result_msg.get("data")
            log(
                f"[s{local_snip_idx:06d} ch{port}] snippet download done "
                f"({0 if data is None else len(data)} items)",
                memory_only=args.memory_only,
            )
            emit_q.put(
                {
                    "kind": "emit_request",
                    "port": port,
                    "snip_idx": local_snip_idx,
                    "timestamp": result_msg.get("timestamp"),
                    "ranges": result_msg.get("ranges", []),
                    "status": "ok",
                    "error": None,
                    "data": data,
                }
            )
            return

        error = result_msg.get("error", "unknown")
        log(
            f"[s{local_snip_idx if local_snip_idx is not None else 0:06d} ch{port}] "
            f"worker error: {error}",
            memory_only=args.memory_only,
        )
        recover_port(
            replay=replay,
            port=port,
            base=base_offsets[port],
            ring_bytes=ring_bytes,
            restart_margin_bytes=restart_margin_bytes,
            snippet_bytes=snippet_bytes,
            guard_bytes=guard_bytes,
            replay_port_lock=replay_port_locks[port],
            wrapped_flags=wrapped_flags,
            args=args,
            reason="worker_error",
            recovery_attempts=recovery_attempts,
            recovery_failures=recovery_failures,
        )
        if (
            args.graph_reset_after_recovery_failures > 0
            and recovery_failures[port] >= args.graph_reset_after_recovery_failures
        ):
            reset_rfnoc_streaming(
                graph=graph,
                replay=replay,
                radio_chan_pairs=radio_chan_pairs,
                base_offsets=base_offsets,
                ring_bytes=ring_bytes,
                args=args,
                replay_global_lock=replay_global_lock,
                replay_port_locks=replay_port_locks,
            )
            recovery_failures[port] = 0

    try:
        while True:
            if stop_evt.is_set():
                break
            if args.max_snippets and snip_idx > args.max_snippets:
                break

            now_m = time.monotonic()
            if now_m < next_t:
                while True:
                    try:
                        handle_read_result(read_result_queue.get_nowait())
                    except queue.Empty:
                        break
                time.sleep(min(0.05, next_t - now_m))
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            for port in range(num_ports):
                with state_locks[port]:
                    pos = int(replay.get_record_position(port))
                    if last_positions[port] is not None and pos == last_positions[port]:
                        no_progress_counts[port] += 1
                        if no_progress_counts[port] == 1 or no_progress_counts[port] % 20 == 0:
                            log(
                                f"[s{snip_idx:06d} ch{port}] no new samples recorded yet; "
                                "waiting for replay buffer to fill",
                                memory_only=args.memory_only,
                            )
                        if no_progress_counts[port] >= args.no_progress_recovery_threshold:
                            recover_port(
                                replay=replay,
                                port=port,
                                base=base_offsets[port],
                                ring_bytes=ring_bytes,
                                restart_margin_bytes=restart_margin_bytes,
                                snippet_bytes=snippet_bytes,
                                guard_bytes=guard_bytes,
                                replay_port_lock=replay_port_locks[port],
                                wrapped_flags=wrapped_flags,
                                args=args,
                                reason="no_progress_threshold",
                                recovery_attempts=recovery_attempts,
                                recovery_failures=recovery_failures,
                            )
                            no_progress_counts[port] = 0
                            if (
                                args.graph_reset_after_recovery_failures > 0
                                and recovery_failures[port] >= args.graph_reset_after_recovery_failures
                            ):
                                reset_rfnoc_streaming(
                                    graph=graph,
                                    replay=replay,
                                    radio_chan_pairs=radio_chan_pairs,
                                    base_offsets=base_offsets,
                                    ring_bytes=ring_bytes,
                                    args=args,
                                    replay_global_lock=replay_global_lock,
                                    replay_port_locks=replay_port_locks,
                                )
                                recovery_failures[port] = 0
                        continue
                    last_positions[port] = pos
                    no_progress_counts[port] = 0
                    base = base_offsets[port]
                    wrapped = wrapped_flags[port]

                    ranges = compute_snip_ranges(
                        base=base,
                        ring_bytes=ring_bytes,
                        record_pos=pos,
                        snippet_bytes=snippet_bytes,
                        guard_bytes=guard_bytes,
                        align=align,
                        wrapped=wrapped,
                        allow_wrap_read=args.allow_wrap_read,
                    )

                    if not ranges:
                        if wrapped and not args.allow_wrap_read:
                            wrap_ranges = compute_snip_ranges(
                                base=base,
                                ring_bytes=ring_bytes,
                                record_pos=pos,
                                snippet_bytes=snippet_bytes,
                                guard_bytes=guard_bytes,
                                align=align,
                                wrapped=wrapped,
                                allow_wrap_read=True,
                            )
                            if len(wrap_ranges) == 2:
                                log(
                                    f"[s{snip_idx:06d} ch{port}] skip due to wrap boundary "
                                    f"(port={port} snip_idx={snip_idx} record_pos={pos})",
                                    memory_only=args.memory_only,
                                )
                                continue
                        log(f"[s{snip_idx:06d} ch{port}] not enough data yet",
                            memory_only=args.memory_only)
                        continue
                if port in pending:
                    log(
                        f"[s{snip_idx:06d} ch{port}] worker busy; skipping request",
                        memory_only=args.memory_only,
                    )
                    continue

                msg = {
                    "kind": "read_request",
                    "port": port,
                    "snip_idx": snip_idx,
                    "timestamp": ts,
                    "ranges": [{"offset": off, "size": sz} for off, sz in ranges],
                    "status": "queued",
                    "error": None,
                }
                read_req_queues[port].put(msg)
                pending[port] = {"snip_idx": snip_idx, "queued_t": time.monotonic()}

            while True:
                try:
                    handle_read_result(read_result_queue.get_nowait())
                except queue.Empty:
                    break

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
        for q in read_req_queues:
            try:
                q.put_nowait({"kind": "shutdown"})
            except Exception:
                pass
        try:
            emit_q.put_nowait({"kind": "shutdown"})
        except Exception:
            pass

        for proc in read_workers:
            proc.join(timeout=2.0)
            if proc.is_alive():
                log("WARNING: read worker thread did not exit cleanly", memory_only=args.memory_only)

        if emitter_kind == "process":
            emitter.join(timeout=2.0)
            if emitter.is_alive():
                emitter.terminate()
        else:
            emitter.join(timeout=2.0)

        time.sleep(0.1)

        try:
            if graph is not None:
                graph.release()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
