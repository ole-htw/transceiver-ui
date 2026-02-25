import types
import unittest
from unittest.mock import patch

import numpy as np

from transceiver.helpers.interpolation import _apply_rx_interpolation


class TestRxInterpolation(unittest.TestCase):
    def test_returns_original_when_disabled_or_empty(self) -> None:
        signal = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
        out, fs_new = _apply_rx_interpolation(
            signal, fs=1_000.0, enabled=False, method="interp1d", factor_expr="2"
        )
        self.assertTrue(np.array_equal(out, signal))
        self.assertEqual(fs_new, 1_000.0)

        empty = np.array([], dtype=np.complex64)
        out, fs_new = _apply_rx_interpolation(
            empty, fs=1_000.0, enabled=True, method="interp1d", factor_expr="2"
        )
        self.assertEqual(out.size, 0)
        self.assertEqual(fs_new, 1_000.0)

    def test_rejects_invalid_factors(self) -> None:
        signal = np.array([1.0, 2.0], dtype=np.float32)
        for factor_expr in ["", "0", "-1", "0.5", "abc"]:
            with self.subTest(factor_expr=factor_expr):
                with self.assertRaises(ValueError):
                    _apply_rx_interpolation(
                        signal,
                        fs=100.0,
                        enabled=True,
                        method="interp1d",
                        factor_expr=factor_expr,
                    )

    def test_interp1d_complex_interpolates_and_scales_fs(self) -> None:
        signal = np.array([0 + 0j, 10 + 10j], dtype=np.complex64)

        class _FakeInterp:
            @staticmethod
            def interp1d(x, y, kind="linear"):
                def _fn(x_new):
                    return np.interp(x_new, x, y)

                return _fn

        with patch("transceiver.helpers.interpolation.importlib.util.find_spec", return_value=object()):
            with patch(
                "transceiver.helpers.interpolation.importlib.import_module",
                return_value=_FakeInterp,
            ):
                out, fs_new = _apply_rx_interpolation(
                    signal,
                    fs=200.0,
                    enabled=True,
                    method="interp1d",
                    factor_expr="3/2",
                )

        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(fs_new, 300.0)
        self.assertTrue(np.iscomplexobj(out))

    def test_resample_poly_uses_fraction_ratio(self) -> None:
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        captured = {}

        def _resample_poly(arr, up, down):
            captured["up"] = up
            captured["down"] = down
            return arr

        fake_signal_mod = types.SimpleNamespace(resample_poly=_resample_poly)

        with patch("transceiver.helpers.interpolation.importlib.util.find_spec", return_value=object()):
            with patch(
                "transceiver.helpers.interpolation.importlib.import_module",
                return_value=fake_signal_mod,
            ):
                out, fs_new = _apply_rx_interpolation(
                    signal,
                    fs=400.0,
                    enabled=True,
                    method="resample_poly",
                    factor_expr="1.25",
                )

        self.assertTrue(np.array_equal(out, signal))
        self.assertEqual(captured["up"], 5)
        self.assertEqual(captured["down"], 4)
        self.assertAlmostEqual(fs_new, 500.0)

    def test_numerical_failures_fallback_with_warning(self) -> None:
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        class _BrokenInterp:
            @staticmethod
            def interp1d(*_args, **_kwargs):
                raise RuntimeError("boom")

        with patch("transceiver.helpers.interpolation.importlib.util.find_spec", return_value=object()):
            with patch(
                "transceiver.helpers.interpolation.importlib.import_module",
                return_value=_BrokenInterp,
            ):
                with self.assertLogs(level="WARNING"):
                    out, fs_new = _apply_rx_interpolation(
                        signal,
                        fs=400.0,
                        enabled=True,
                        method="interp1d",
                        factor_expr="2",
                    )

        self.assertTrue(np.array_equal(out, signal))
        self.assertEqual(fs_new, 400.0)


if __name__ == "__main__":
    unittest.main()
