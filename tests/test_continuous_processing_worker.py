import multiprocessing as mp

import numpy as np

from transceiver.helpers.continuous_processing import continuous_processing_worker


def test_continuous_worker_is_importable_from_non_main_module():
    assert continuous_processing_worker.__module__ == "transceiver.helpers.continuous_processing"


def test_continuous_worker_spawn_startup_and_shutdown():
    ctx = mp.get_context("spawn")
    tasks = ctx.Queue()
    results = ctx.Queue()
    proc = ctx.Process(target=continuous_processing_worker, args=(tasks, results), daemon=True)
    proc.start()
    tasks.put(None)
    proc.join(timeout=5)
    assert proc.exitcode == 0


def test_continuous_worker_keeps_full_resolution_for_plot_data():
    ctx = mp.get_context("spawn")
    tasks = ctx.Queue()
    results = ctx.Queue()
    proc = ctx.Process(target=continuous_processing_worker, args=(tasks, results), daemon=True)
    proc.start()

    sample_count = 8192
    channel_1 = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    channel_2 = np.linspace(1.0, 0.0, sample_count, dtype=np.float32)
    frame = np.vstack([channel_1, channel_2]).astype(np.complex64)

    tasks.put(
        {
            "data": frame,
            "fs": 1.0,
            "rx_channel_view": "Kanal 1",
        }
    )

    result = results.get(timeout=5)

    tasks.put(None)
    proc.join(timeout=5)

    assert proc.exitcode == 0
    assert result["plot_data"].shape[0] == sample_count
