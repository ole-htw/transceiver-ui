import sys
import types

import numpy as np


def _install_pyqtgraph_stub() -> None:
    if "pyqtgraph" in sys.modules:
        return

    qt_mod = types.ModuleType("pyqtgraph.Qt")
    qt_mod.QtCore = object()
    qt_mod.QtGui = object()

    class _PgDummy:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, _name: str):
            return self

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
    _build_crosscorr_ctx,
    _format_echo_delay_display,
    _format_rx_stats_rows,
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


def test_format_echo_delay_display_with_and_without_interpolation_factor() -> None:
    plain = _format_echo_delay_display(12, interpolation_enabled=False, interpolation_factor=2.0)
    assert plain == "12 samp (18.0 m)"

    interp_x2 = _format_echo_delay_display(12, interpolation_enabled=True, interpolation_factor=2.0)
    assert interp_x2 == "12 samp (9.0 m) (interp. Raster)"

    interp_x15 = _format_echo_delay_display(12, interpolation_enabled=True, interpolation_factor=1.5)
    assert interp_x15 == "12 samp (12.0 m) (interp. Raster)"

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
    assert as_dict["LOS-Echo"] == "12 samp (9.0 m) (interp. Raster)"
