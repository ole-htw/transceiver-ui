import multiprocessing as mp

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
