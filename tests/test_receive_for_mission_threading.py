from __future__ import annotations

import sys
import threading
import time
import types
import queue


def _install_pyqtgraph_stub() -> None:
    if "pyqtgraph" in sys.modules:
        return

    qt_mod = types.ModuleType("pyqtgraph.Qt")

    class _PgDummy:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, _name: str):
            return self

    class _QtWidgetsDummy:
        QDialog = type("QDialog", (), {})

        def __getattr__(self, name: str):
            return type(name, (), {})

    qt_mod.QtCore = _PgDummy()
    qt_mod.QtGui = _PgDummy()
    qt_mod.QtWidgets = _QtWidgetsDummy()

    pg_mod = types.ModuleType("pyqtgraph")
    pg_mod.Qt = qt_mod
    pg_mod.exporters = types.ModuleType("pyqtgraph.exporters")
    pg_mod.ScatterPlotItem = type("ScatterPlotItem", (), {})
    pg_mod.PlotWidget = type("PlotWidget", (), {})
    pg_mod.InfiniteLine = type("InfiniteLine", (), {})

    def _pg_getattr(_name: str):
        return _PgDummy()

    pg_mod.__getattr__ = _pg_getattr  # type: ignore[attr-defined]

    sys.modules["pyqtgraph"] = pg_mod
    sys.modules["pyqtgraph.Qt"] = qt_mod
    sys.modules["pyqtgraph.exporters"] = pg_mod.exporters


_install_pyqtgraph_stub()
sys.modules.setdefault("uhd", types.ModuleType("uhd"))

from transceiver.__main__ import TransceiverUI


class _DummyWidget:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def configure(self, **kwargs) -> None:
        self.calls.append(kwargs)


def test_run_rx_thread_single_uses_shared_subprocess_executor_and_single_postprocessing(monkeypatch) -> None:
    ui = object.__new__(TransceiverUI)
    ui._out_queue = queue.Queue()
    ui._cmd_running = True
    ui._rx_running = True
    ui._proc = object()
    ui._ui = lambda callback: callback()
    ui._reset_rx_buttons = lambda: None
    displayed: list[tuple[object, float, str]] = []
    ui._display_rx_plots = lambda data, rate, target_tab="Single": displayed.append((data, rate, target_tab))
    executed: list[dict[str, object]] = []
    ui._execute_rx_subprocess = (
        lambda **kwargs: executed.append(kwargs) or 0
    )
    monkeypatch.setattr(
        "transceiver.__main__.rx_convert.load_iq_file",
        lambda *_args, **_kwargs: [1, 2, 3],
    )

    result = TransceiverUI._run_rx_thread(
        ui,
        ["--output-file", "signals/rx/single.bin"],
        channels=1,
        rate=1_000_000.0,
        backend_only=False,
        mission_mode=False,
    )

    assert result["ok"] is True
    assert result["output_file"] == "signals/rx/single.bin"
    assert len(executed) == 1
    assert displayed and displayed[0][2] == "Single"


def test_run_rx_thread_mission_uses_shared_subprocess_executor_without_single_ui_postprocessing() -> None:
    ui = object.__new__(TransceiverUI)
    ui._out_queue = queue.Queue()
    ui._mission_rx_proc = object()
    ui._mission_rx_running = True
    ui._cleanup_mission_rx = lambda terminate=False: setattr(ui, "_mission_rx_running", False)
    ui._ui = lambda callback: callback()
    displayed: list[object] = []
    ui._display_rx_plots = lambda *_args, **_kwargs: displayed.append("called")
    executed: list[dict[str, object]] = []
    ui._execute_rx_subprocess = (
        lambda **kwargs: executed.append(kwargs) or 0
    )

    result = TransceiverUI._run_rx_thread(
        ui,
        ["--output-file", "signals/rx/mission.bin"],
        channels=1,
        rate=1_000_000.0,
        backend_only=True,
        mission_mode=True,
    )

    assert result["ok"] is True
    assert result["output_file"] == "signals/rx/mission.bin"
    assert len(executed) == 1
    assert displayed == []


def test_run_rx_thread_single_without_explicit_output_file_sets_result_output_file(monkeypatch) -> None:
    ui = object.__new__(TransceiverUI)
    ui._out_queue = queue.Queue()
    ui._cmd_running = True
    ui._rx_running = True
    ui._proc = object()
    ui._ui = lambda callback: callback()
    ui._reset_rx_buttons = lambda: None
    captured_args: list[list[str]] = []
    ui._execute_rx_subprocess = lambda **kwargs: captured_args.append(list(kwargs["arg_list"])) or 0
    monkeypatch.setattr(
        "transceiver.__main__.rx_convert.load_iq_file",
        lambda *_args, **_kwargs: [1, 2, 3],
    )

    result = TransceiverUI._run_rx_thread(
        ui,
        ["-a", "type=b200"],
        channels=1,
        rate=1_000_000.0,
        backend_only=True,
        mission_mode=False,
    )

    assert result["ok"] is True
    assert isinstance(result["output_file"], str)
    assert str(result["output_file"]).startswith("signals/rx/rx_auto_")
    assert str(result["output_file"]).endswith(".bin")
    assert captured_args and "--output-file" in captured_args[0]
    out_idx = captured_args[0].index("--output-file")
    assert captured_args[0][out_idx + 1] == result["output_file"]


def test_receive_for_mission_uses_worker_path_and_waits_for_result() -> None:
    ui = object.__new__(TransceiverUI)
    ui._cmd_running = False
    ui._mission_rx_running = False
    ui._mission_rx_proc = None
    ui.rx_stop = _DummyWidget()
    ui.rx_button = _DummyWidget()
    ui._ui = lambda callback: callback()
    process_queue_calls: list[str] = []
    ui._process_queue = lambda: process_queue_calls.append("called")
    ui._cleanup_mission_rx = lambda terminate=False: setattr(ui, "_mission_rx_running", False)
    ui._build_receive_arg_list_for_worker = (
        lambda *, output_file: (["-a", "type=b200"], 2, 2_000_000.0)
    )

    captured: dict[str, object] = {}

    def _fake_start_rx_worker(
        *,
        arg_list,
        channels,
        rate,
        point_context=None,
        on_complete=None,
        backend_only=False,
    ):
        captured["arg_list"] = arg_list
        captured["channels"] = channels
        captured["rate"] = rate
        captured["point_context"] = point_context
        captured["caller_thread"] = threading.current_thread().name
        captured["backend_only"] = backend_only

        def _runner():
            time.sleep(0.05)
            if on_complete is not None:
                on_complete({"ok": True, "output_file": "signals/rx/mission/demo.bin"})

        worker = threading.Thread(target=_runner, daemon=True, name="rx-worker-test")
        worker.start()
        return worker

    ui._start_rx_worker = _fake_start_rx_worker

    started = time.perf_counter()
    result = TransceiverUI.receive_for_mission(
        ui,
        output_file="signals/rx/mission/demo.bin",
        point_context=types.SimpleNamespace(global_index=7),
    )
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.045
    assert captured["arg_list"] == ["-a", "type=b200"]
    assert captured["channels"] == 2
    assert captured["rate"] == 2_000_000.0
    assert captured["backend_only"] is True
    assert process_queue_calls == []
    assert result == {"ok": True, "output_file": "signals/rx/mission/demo.bin"}
    assert ui.rx_stop.calls == []
    assert ui.rx_button.calls == []


def test_receive_for_mission_can_run_multiple_times_sequentially() -> None:
    ui = object.__new__(TransceiverUI)
    ui._cmd_running = False
    ui._mission_rx_running = False
    ui._mission_rx_proc = None
    ui.rx_stop = _DummyWidget()
    ui.rx_button = _DummyWidget()
    ui._ui = lambda callback: callback()
    ui._process_queue = lambda: None
    ui._cleanup_mission_rx = lambda terminate=False: setattr(ui, "_mission_rx_running", False)
    ui._build_receive_arg_list_for_worker = (
        lambda *, output_file: (["--output-file", output_file], 1, 1_000_000.0)
    )

    run_calls: list[dict[str, object]] = []

    def _fake_start_rx_worker(
        *,
        arg_list,
        channels,
        rate,
        point_context=None,
        on_complete=None,
        backend_only=False,
    ):
        run_calls.append(
            {
                "arg_list": list(arg_list),
                "channels": channels,
                "rate": rate,
                "backend_only": backend_only,
                "point_index": getattr(point_context, "global_index", None),
            }
        )
        if on_complete is not None:
            on_complete({"ok": True, "output_file": arg_list[-1]})
        return types.SimpleNamespace(name="rx-worker-test")

    ui._start_rx_worker = _fake_start_rx_worker

    first = TransceiverUI.receive_for_mission(
        ui,
        output_file="signals/rx/mission/p1.bin",
        point_context=types.SimpleNamespace(global_index=1),
    )
    second = TransceiverUI.receive_for_mission(
        ui,
        output_file="signals/rx/mission/p2.bin",
        point_context=types.SimpleNamespace(global_index=2),
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert [call["arg_list"][-1] for call in run_calls] == [
        "signals/rx/mission/p1.bin",
        "signals/rx/mission/p2.bin",
    ]
    assert [call["backend_only"] for call in run_calls] == [True, True]


def test_review_measurement_for_mission_applies_same_interpolation_as_single_receive(monkeypatch) -> None:
    import numpy as np
    import queue
    import transceiver.__main__ as app_module

    ui = object.__new__(TransceiverUI)
    ui._ui = lambda callback: callback()
    ui._out_queue = queue.Queue()
    ui.latest_fs = 10.0
    ui.rx_channel_2 = types.SimpleNamespace(get=lambda: False)
    ui.rx_xcorr_normalized_enable = types.SimpleNamespace(get=lambda: False)
    ui._select_rx_display_data = lambda loaded: (loaded, "CH0")
    ui._get_crosscorr_reference = lambda: (np.array([3.0, 4.0], dtype=float), "TX")

    interpolation_calls: list[tuple[np.ndarray, float, np.ndarray]] = []

    def _fake_apply_crosscorr_interpolation(data, fs, ref_data, crosscorr_compare=None, **_kwargs):
        interpolation_calls.append((data.copy(), fs, ref_data.copy()))
        return data * 2.0, ref_data * 3.0, crosscorr_compare, fs * 2.0

    ui._apply_crosscorr_interpolation = _fake_apply_crosscorr_interpolation
    ui.rx_interpolation_enable = types.SimpleNamespace(get=lambda: True)

    monkeypatch.setattr(
        app_module.rx_convert,
        "load_iq_file",
        lambda *_args, **_kwargs: np.array([1.0, 2.0], dtype=float),
    )

    class _DummyApp:
        def activeWindow(self):
            return None

    monkeypatch.setattr(app_module.pg, "mkQApp", lambda: _DummyApp())
    monkeypatch.setattr(app_module.QtWidgets.QApplication, "topLevelWidgets", lambda: [], raising=False)

    captured_ctx_input: dict[str, np.ndarray] = {}

    def _fake_build_ctx(data, ref_data, **_kwargs):
        captured_ctx_input["data"] = np.asarray(data)
        captured_ctx_input["ref_data"] = np.asarray(ref_data)
        return {
            "lags": np.array([0.0]),
            "mag": np.array([1.0]),
            "los_idx": 0,
            "echo_indices": [],
        }

    monkeypatch.setattr(app_module, "_build_crosscorr_ctx", _fake_build_ctx)

    class _DummyDialog:
        def __init__(self, **_kwargs):
            self.confirmed = False
            self.selected_los_idx = None
            self.selected_echo_indices = []
            self.manual_lags = {}
            self.echo_delays = []

        def raise_(self):
            return None

        def activateWindow(self):
            return None

        def exec(self):
            return 0

    monkeypatch.setattr(app_module, "MissionMeasurementReviewDialog", _DummyDialog)

    outcome = TransceiverUI.review_measurement_for_mission(
        ui,
        point_label="P1",
        output_file="signals/rx/mission/demo.bin",
    )

    assert interpolation_calls and interpolation_calls[0][1] == 10.0
    np.testing.assert_allclose(captured_ctx_input["data"], np.array([2.0, 4.0]))
    np.testing.assert_allclose(captured_ctx_input["ref_data"], np.array([9.0, 12.0]))
    assert outcome["approved"] is False


def test_review_measurement_for_mission_uses_persisted_tx_reference_without_single_rx(
    monkeypatch, tmp_path
) -> None:
    import numpy as np
    import queue
    import transceiver.__main__ as app_module

    tx_path = tmp_path / "persisted_tx.bin"
    np.array([10, 20, 30, 40], dtype=np.int16).tofile(tx_path)

    ui = object.__new__(TransceiverUI)
    ui._ui = lambda callback: callback()
    ui._out_queue = queue.Queue()
    ui.latest_fs = 10.0
    ui.tx_data = np.array([], dtype=np.complex64)
    ui._cached_tx_path = None
    ui._cached_tx_data = np.array([], dtype=np.complex64)
    ui._cached_tx_load_error_path = None
    ui.tx_file = types.SimpleNamespace(get=lambda: str(tx_path))
    ui.rx_channel_2 = types.SimpleNamespace(get=lambda: False)
    ui.rx_xcorr_normalized_enable = types.SimpleNamespace(get=lambda: False)
    ui.rx_interpolation_enable = types.SimpleNamespace(get=lambda: False)
    ui._select_rx_display_data = lambda loaded: (loaded, "CH0")
    ui._apply_crosscorr_interpolation = (
        lambda data, fs, ref_data, crosscorr_compare=None, **_kwargs: (data, ref_data, crosscorr_compare, fs)
    )

    monkeypatch.setattr(
        app_module.rx_convert,
        "load_iq_file",
        lambda *_args, **_kwargs: np.array([1.0, 2.0], dtype=float),
    )

    class _DummyApp:
        def activeWindow(self):
            return None

    monkeypatch.setattr(app_module.pg, "mkQApp", lambda: _DummyApp())
    monkeypatch.setattr(app_module.QtWidgets.QApplication, "topLevelWidgets", lambda: [], raising=False)
    monkeypatch.setattr(app_module.messagebox, "showerror", lambda *_args, **_kwargs: None)

    captured_ctx_input: dict[str, np.ndarray] = {}

    def _fake_build_ctx(data, ref_data, **_kwargs):
        captured_ctx_input["ref_data"] = np.asarray(ref_data)
        return {
            "lags": np.array([0.0]),
            "mag": np.array([1.0]),
            "los_idx": 0,
            "echo_indices": [],
        }

    monkeypatch.setattr(app_module, "_build_crosscorr_ctx", _fake_build_ctx)

    class _DummyDialog:
        def __init__(self, **_kwargs):
            self.confirmed = False
            self.selected_los_idx = None
            self.selected_echo_indices = []
            self.manual_lags = {}
            self.echo_delays = []

        def raise_(self):
            return None

        def activateWindow(self):
            return None

        def exec(self):
            return 0

    monkeypatch.setattr(app_module, "MissionMeasurementReviewDialog", _DummyDialog)

    outcome = TransceiverUI.review_measurement_for_mission(
        ui,
        point_label="P1",
        output_file="signals/rx/mission/demo.bin",
    )

    np.testing.assert_allclose(captured_ctx_input["ref_data"], np.array([10.0 + 20.0j, 30.0 + 40.0j]))
    assert ui.tx_data.size == 2
    assert outcome["reason"] != "missing_tx_reference"


def test_get_live_echo_distances_for_mission_preview_reads_cached_field_and_limits() -> None:
    ui = object.__new__(TransceiverUI)
    ui._cont_thread = types.SimpleNamespace(is_alive=lambda: True)
    ui._last_continuous_payload = {
        "mission_preview_echo_distances_m": [1.5, "3.0", float("nan"), -2.0, "bad", 4.5],
    }
    ui._get_crosscorr_reference = lambda: (_ for _ in ()).throw(RuntimeError("must not be called"))

    result = TransceiverUI.get_live_echo_distances_for_mission_preview(ui, limit=2)

    assert result == [1.5, 3.0]


def test_get_live_echo_distances_for_mission_preview_handles_invalid_payloads_robustly() -> None:
    ui = object.__new__(TransceiverUI)
    ui._cont_thread = types.SimpleNamespace(is_alive=lambda: True)

    ui._last_continuous_payload = None
    assert TransceiverUI.get_live_echo_distances_for_mission_preview(ui) == []

    ui._last_continuous_payload = {"mission_preview_echo_distances_m": "invalid"}
    assert TransceiverUI.get_live_echo_distances_for_mission_preview(ui) == []

    ui._last_continuous_payload = {"mission_preview_echo_distances_m": [1.5]}
    ui._cont_thread = types.SimpleNamespace(is_alive=lambda: False)
    assert TransceiverUI.get_live_echo_distances_for_mission_preview(ui) == []


def test_render_continuous_payload_caches_mission_preview_distances_without_recompute() -> None:
    ui = object.__new__(TransceiverUI)
    ui.latest_fs = 1.0
    ui._cont_rendered_frames = 0
    ui._cont_runtime_config = {}
    ui._rx_cont_pg_state = {}
    ui.rx_xcorr_normalized_enable = types.SimpleNamespace(set=lambda _value: None)
    ui.rx_interpolation_enable = types.SimpleNamespace(set=lambda _value: None)
    ui.rx_interpolation_method = types.SimpleNamespace(set=lambda _value: None)
    ui._set_rx_interpolation_factor_text = lambda _value: None
    ui._rx_interpolation_factor_text = lambda: "2"
    ui._on_rx_interpolation_toggle = lambda recompute=False: None
    ui._update_rx_interpolation_status = lambda interpolation_applied=False: None
    ui._display_rx_plots = lambda *_args, **_kwargs: None

    calls = {"count": 0}

    def _fake_compute(_payload):
        calls["count"] += 1
        return [2.25, 4.5]

    ui._compute_mission_preview_echo_distances_from_payload = _fake_compute

    payload = {
        "plot_data": [1.0, 2.0, 3.0],
        "plot_ref_data": [1.0, 0.0],
        "fs": 2.0,
        "frame_ts": 0.0,
        "processing_ms": 0.0,
        "interpolation_applied": False,
    }

    TransceiverUI._render_continuous_payload(ui, payload)
    TransceiverUI._render_continuous_payload(ui, payload)

    assert calls["count"] == 1
    assert ui._last_continuous_payload["mission_preview_echo_distances_m"] == [2.25, 4.5]
