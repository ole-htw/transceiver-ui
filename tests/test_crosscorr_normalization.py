import sys
import types

import numpy as np


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

from transceiver.__main__ import (
    TransceiverUI,
    _build_crosscorr_ctx,
    _classify_visible_xcorr_peaks,
    _find_echo_marker_slot_near_lag,
    _format_echo_delay_display,
    _format_rx_stats_rows,
    _update_echo_indices_after_manual_drag,
)


def _make_reference_and_rx() -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0 + 0.0j, 0.5 - 0.25j, -0.25 + 0.1j, 0.1 + 0.2j], dtype=np.complex128)
    channel = np.zeros(40, dtype=np.complex128)
    channel[10] = 2.0
    channel[17] = 0.8
    rx = np.convolve(channel, ref, mode="full")
    return ref, rx


def test_crosscorr_normalization_sets_peak_to_one_for_nonempty_signal() -> None:
    ref, rx = _make_reference_and_rx()

    ctx = _build_crosscorr_ctx(rx, ref, normalize=True)

    mag = ctx["mag"]
    assert isinstance(mag, np.ndarray)
    assert mag.size > 0
    assert np.isfinite(mag).all()
    assert np.isclose(np.max(mag), 1.0)


def test_classify_visible_xcorr_peaks_keeps_echoes_with_stricter_los_threshold() -> None:
    mag = np.zeros(180, dtype=float)
    mag[90] = 1.0
    mag[110] = 0.25

    highest_idx, los_idx, echo_indices = _classify_visible_xcorr_peaks(
        mag,
        repetition_period_samples=200,
        peaks_before=0,
        peaks_after=1,
        min_rel_height=0.0,
        los_min_rel_height=0.3,
    )

    assert highest_idx == 90
    assert los_idx == 90
    assert echo_indices == [110]


def test_crosscorr_normalization_applies_to_comparison_trace_mag2() -> None:
    ref, rx = _make_reference_and_rx()
    rx_compare = 0.35 * rx

    ctx = _build_crosscorr_ctx(
        rx,
        ref,
        crosscorr_compare=rx_compare,
        normalize=True,
    )

    mag2 = ctx["mag2"]
    assert isinstance(mag2, np.ndarray)
    assert mag2.size > 0
    assert np.isfinite(mag2).all()
    assert np.isclose(np.max(mag2), 1.0)


def test_crosscorr_normalization_handles_zero_signal_without_nan() -> None:
    ref = np.array([1.0 + 0.0j, -0.2 + 0.3j], dtype=np.complex128)
    zero = np.zeros(16, dtype=np.complex128)

    ctx = _build_crosscorr_ctx(zero, ref, crosscorr_compare=zero, normalize=True)

    mag = ctx["mag"]
    mag2 = ctx["mag2"]
    assert isinstance(mag, np.ndarray)
    assert isinstance(mag2, np.ndarray)
    assert np.isfinite(mag).all()
    assert np.isfinite(mag2).all()
    assert float(np.max(mag)) == 0.0
    assert float(np.max(mag2)) == 0.0


def test_crosscorr_normalization_keeps_los_and_echo_indices_stable() -> None:
    ref, rx = _make_reference_and_rx()

    ctx_raw = _build_crosscorr_ctx(rx, ref, normalize=False)
    ctx_normalized = _build_crosscorr_ctx(rx, ref, normalize=True)

    assert ctx_raw["los_idx"] == ctx_normalized["los_idx"]
    assert ctx_raw["echo_indices"] == ctx_normalized["echo_indices"]
    assert ctx_raw["peak_source_highest_idx"] == ctx_normalized["peak_source_highest_idx"]


def test_crosscorr_context_exposes_background_noise_level() -> None:
    ref, rx = _make_reference_and_rx()

    ctx = _build_crosscorr_ctx(rx, ref, normalize=False)

    noise_level = ctx.get("background_noise_level")
    assert isinstance(noise_level, float)
    assert np.isfinite(noise_level)
    assert noise_level >= 0.0


def test_format_echo_delay_display_with_and_without_interpolation_factor() -> None:
    plain = _format_echo_delay_display(12, interpolation_enabled=False, interpolation_factor=2.0)
    assert plain == "12 samp (18.0 m)"

    interp_x2 = _format_echo_delay_display(12, interpolation_enabled=True, interpolation_factor=2.0)
    assert interp_x2 == "6 samp (9.0 m) (interp. Raster)"

    interp_x15 = _format_echo_delay_display(12, interpolation_enabled=True, interpolation_factor=1.5)
    assert interp_x15 == "8 samp (12.0 m) (interp. Raster)"

    interp_invalid = _format_echo_delay_display(12, interpolation_enabled=True, interpolation_factor="bad")
    assert interp_invalid == "12 samp (18.0 m) (interp. Raster)"

    interp_zero = _format_echo_delay_display(12, interpolation_enabled=True, interpolation_factor=0)
    assert interp_zero == "12 samp (18.0 m) (interp. Raster)"


def test_format_rx_stats_rows_scales_echo_distance_when_interpolated() -> None:
    stats = {
        "f_low": 100.0,
        "f_high": 200.0,
        "amp": 1.0,
        "bw": 50.0,
        "echo_delay": 12,
    }

    rows = _format_rx_stats_rows(
        stats,
        interpolation_enabled=True,
        interpolation_factor=2.0,
    )

    as_dict = dict(rows)
    assert as_dict["LOS-Echo"] == "6 samp (9.0 m) (interp. Raster)"


def test_format_rx_stats_rows_does_not_scale_echo_when_interpolation_not_applied() -> None:
    """Interpolation toggle may be active, but rows must follow runtime data path."""
    stats = {
        "f_low": 100.0,
        "f_high": 200.0,
        "amp": 1.0,
        "bw": 50.0,
        "echo_delay": 12,
    }

    rows = _format_rx_stats_rows(
        stats,
        interpolation_enabled=False,
        interpolation_factor=2.0,
    )

    as_dict = dict(rows)
    assert as_dict["LOS-Echo"] == "12 samp (18.0 m)"


def test_update_echo_indices_after_manual_drag_updates_selected_marker_slot() -> None:
    lags = np.array([-20, -10, 0, 10, 20], dtype=float)
    updated = _update_echo_indices_after_manual_drag(
        lags,
        echo_indices=[1, 3, 4],
        marker_slot=1,
        lag_value=0.1,
    )
    assert updated == [1, 2, 4]


def test_update_echo_indices_after_manual_drag_keeps_overlapping_slots() -> None:
    lags = np.array([-20, -10, 0, 10, 20], dtype=float)
    updated = _update_echo_indices_after_manual_drag(
        lags,
        echo_indices=[1, 3, 4],
        marker_slot=2,
        lag_value=10.0,
    )
    assert updated == [1, 3, 3]


def test_find_echo_marker_slot_near_lag_returns_nearest_slot_within_threshold() -> None:
    lags = np.array([-20, -10, 0, 10, 20], dtype=float)
    marker_slot = _find_echo_marker_slot_near_lag(
        lags,
        echo_indices=[1, 3, 4],
        target_lag=9.2,
        max_lag_distance=1.5,
    )
    assert marker_slot == 1


def test_find_echo_marker_slot_near_lag_returns_none_outside_threshold() -> None:
    lags = np.array([-20, -10, 0, 10, 20], dtype=float)
    marker_slot = _find_echo_marker_slot_near_lag(
        lags,
        echo_indices=[1, 3, 4],
        target_lag=0.0,
        max_lag_distance=0.5,
    )
    assert marker_slot is None


class _DummyEntryWidget:
    def __init__(self, text: str) -> None:
        self._text = text
        self.entry = object()

    def get(self) -> str:
        return self._text


def test_rx_interpolation_factor_prefers_continuous_value_on_continuous_tab() -> None:
    dummy = types.SimpleNamespace()
    dummy.rx_interpolation_factor_single = _DummyEntryWidget("2")
    dummy.rx_interpolation_factor_cont = _DummyEntryWidget("7")
    dummy.focus_get = lambda: None
    dummy._get_rx_active_tab = lambda: "Continuous"

    value = TransceiverUI._rx_interpolation_factor_text(dummy)  # type: ignore[arg-type]
    assert value == "7"


def test_rx_interpolation_factor_prefers_single_value_on_single_tab() -> None:
    dummy = types.SimpleNamespace()
    dummy.rx_interpolation_factor_single = _DummyEntryWidget("5")
    dummy.rx_interpolation_factor_cont = _DummyEntryWidget("2")
    dummy.focus_get = lambda: None
    dummy._get_rx_active_tab = lambda: "Single"

    value = TransceiverUI._rx_interpolation_factor_text(dummy)  # type: ignore[arg-type]
    assert value == "5"


def test_review_manual_los_drag_updates_echo_distances() -> None:
    dialog = types.SimpleNamespace()
    dialog._lags = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    dialog._manual_lags = {"los": None, "echo": None}
    dialog._selected_los_idx = 0
    dialog._selected_echo_indices = [2, 3]
    dialog._base_echo_indices = [2, 3]
    dialog._render_plot = lambda: None

    from transceiver.__main__ import MissionMeasurementReviewDialog

    MissionMeasurementReviewDialog._apply_manual_lag(dialog, "los", 10.0)

    delays = MissionMeasurementReviewDialog.echo_delays.fget(dialog)
    assert dialog._selected_los_idx == 1
    assert delays == [10, 20]


def test_review_manual_echo_click_updates_first_echo_distance() -> None:
    dialog = types.SimpleNamespace()
    dialog._lags = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    dialog._manual_lags = {"los": None, "echo": None}
    dialog._selected_los_idx = 0
    dialog._selected_echo_indices = [2, 3]
    dialog._base_echo_indices = [2, 3]
    dialog._render_plot = lambda: None

    from transceiver.__main__ import MissionMeasurementReviewDialog

    MissionMeasurementReviewDialog._apply_manual_lag(dialog, "echo", 10.0)

    delays = MissionMeasurementReviewDialog.echo_delays.fget(dialog)
    assert dialog._selected_echo_indices == [1, 3]
    assert delays == [10, 30]


def test_review_drag_preview_updates_echo_distances_live() -> None:
    dialog = types.SimpleNamespace()
    dialog._lags = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    dialog._manual_lags = {"los": None, "echo": None}
    dialog._selected_los_idx = 0
    dialog._selected_echo_indices = [2, 3]
    dialog._base_echo_indices = [2, 3]
    dialog._update_peak_label_positions = lambda: None
    dialog._update_stats_label = lambda: None

    from transceiver.__main__ import MissionMeasurementReviewDialog

    MissionMeasurementReviewDialog._preview_manual_lag(dialog, "los", 10.0)

    delays = MissionMeasurementReviewDialog.echo_delays.fget(dialog)
    assert dialog._selected_los_idx == 1
    assert delays == [10, 20]


def test_review_echo_drag_preview_updates_selected_slot_live() -> None:
    dialog = types.SimpleNamespace()
    dialog._lags = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    dialog._manual_lags = {"los": None, "echo": None}
    dialog._selected_los_idx = 0
    dialog._selected_echo_indices = [2, 3]
    dialog._base_echo_indices = [2, 3]
    dialog._update_peak_label_positions = lambda: None
    dialog._update_stats_label = lambda: None

    from transceiver.__main__ import MissionMeasurementReviewDialog

    MissionMeasurementReviewDialog._preview_manual_echo_lag(dialog, 1, 10.0)

    delays = MissionMeasurementReviewDialog.echo_delays.fget(dialog)
    assert dialog._selected_echo_indices == [2, 1]
    assert delays == [20, 10]


def test_review_echo_numbers_swap_when_markers_cross() -> None:
    from transceiver.__main__ import MissionMeasurementReviewDialog

    dialog = types.SimpleNamespace()
    dialog._lags = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    dialog._selected_echo_indices = [2, 1]
    dialog._echo_marker_slots_by_lag = lambda: MissionMeasurementReviewDialog._echo_marker_slots_by_lag(dialog)

    slots_by_lag = MissionMeasurementReviewDialog._echo_marker_slots_by_lag(dialog)
    numbers = MissionMeasurementReviewDialog._echo_numbers_by_marker_slot(dialog)

    assert slots_by_lag == [1, 0]
    assert numbers == {1: 1, 0: 2}


def test_review_echo_delays_hide_duplicates_for_overlapping_markers() -> None:
    from transceiver.__main__ import MissionMeasurementReviewDialog

    dialog = types.SimpleNamespace()
    dialog._lags = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    dialog._selected_los_idx = 0
    dialog._selected_echo_indices = [2, 2, 3]
    dialog._unique_echo_indices = MissionMeasurementReviewDialog._unique_echo_indices

    delays = MissionMeasurementReviewDialog.echo_delays.fget(dialog)

    assert delays == [20, 30]


def test_review_remove_echo_marker_near_lag_removes_matching_marker() -> None:
    from transceiver.__main__ import MissionMeasurementReviewDialog

    dialog = types.SimpleNamespace()
    dialog._lags = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    dialog._selected_echo_indices = [1, 2, 3]
    dialog._base_echo_indices = [1, 2, 3]
    dialog._manual_lags = {"los": None, "echo": 20}
    dialog._render_plot = lambda: None
    dialog._plot = types.SimpleNamespace(
        getViewBox=lambda: types.SimpleNamespace(viewRange=lambda: ((0.0, 100.0), (0.0, 1.0)))
    )

    removed = MissionMeasurementReviewDialog._remove_echo_marker_near_lag(dialog, 20.5)

    assert removed is True
    assert dialog._selected_echo_indices == [1, 3]
    assert dialog._base_echo_indices == [1, 3]


def test_review_remove_echo_marker_near_lag_resets_manual_echo_when_last_removed() -> None:
    from transceiver.__main__ import MissionMeasurementReviewDialog

    dialog = types.SimpleNamespace()
    dialog._lags = np.array([0.0, 10.0, 20.0], dtype=float)
    dialog._selected_echo_indices = [2]
    dialog._base_echo_indices = [2]
    dialog._manual_lags = {"los": None, "echo": 20}
    dialog._render_plot = lambda: None
    dialog._plot = types.SimpleNamespace(
        getViewBox=lambda: types.SimpleNamespace(viewRange=lambda: ((0.0, 100.0), (0.0, 1.0)))
    )

    removed = MissionMeasurementReviewDialog._remove_echo_marker_near_lag(dialog, 20.0)

    assert removed is True
    assert dialog._selected_echo_indices == []
    assert dialog._manual_lags["echo"] is None


def test_review_add_echo_marker_near_lag_adds_marker_at_nearest_lag() -> None:
    from transceiver.__main__ import MissionMeasurementReviewDialog

    dialog = types.SimpleNamespace()
    dialog._lags = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    dialog._selected_echo_indices = [1, 3]
    dialog._base_echo_indices = [1, 3]
    dialog._manual_lags = {"los": None, "echo": None}
    dialog._render_plot = lambda: None
    dialog._plot = types.SimpleNamespace(
        getViewBox=lambda: types.SimpleNamespace(viewRange=lambda: ((0.0, 100.0), (0.0, 1.0)))
    )

    added = MissionMeasurementReviewDialog._add_echo_marker_near_lag(dialog, 21.0)

    assert added is True
    assert dialog._selected_echo_indices == [1, 3, 2]
    assert dialog._base_echo_indices == [1, 3, 2]
    assert dialog._manual_lags["echo"] == 20


def test_review_add_echo_marker_near_lag_ignores_existing_marker() -> None:
    from transceiver.__main__ import MissionMeasurementReviewDialog

    dialog = types.SimpleNamespace()
    dialog._lags = np.array([0.0, 10.0, 20.0], dtype=float)
    dialog._selected_echo_indices = [1]
    dialog._base_echo_indices = [1]
    dialog._manual_lags = {"los": None, "echo": None}
    dialog._render_plot = lambda: None
    dialog._plot = types.SimpleNamespace(
        getViewBox=lambda: types.SimpleNamespace(viewRange=lambda: ((0.0, 100.0), (0.0, 1.0)))
    )

    added = MissionMeasurementReviewDialog._add_echo_marker_near_lag(dialog, 11.0)

    assert added is False
    assert dialog._selected_echo_indices == [1]
    assert dialog._base_echo_indices == [1]
    assert dialog._manual_lags["echo"] is None
