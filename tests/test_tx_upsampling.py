import types
import unittest
from unittest.mock import patch

import numpy as np

from transceiver.helpers.interpolation import apply_tx_upsampling


class TestTxUpsampling(unittest.TestCase):
    def test_rejects_target_rate_not_above_input_rate(self) -> None:
        signal = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)

        for fs_target in [200.0, 199.0, -1.0]:
            with self.subTest(fs_target=fs_target):
                with self.assertRaises(ValueError):
                    apply_tx_upsampling(signal, fs_in=200.0, fs_target=fs_target)

    def test_resample_poly_uses_fraction_ratio_for_target_rate(self) -> None:
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
                out, fs_out = apply_tx_upsampling(
                    signal,
                    fs_in=200e6,
                    fs_target=245.76e6,
                    method="resample_poly",
                )

        self.assertTrue(np.array_equal(out, signal))
        self.assertEqual(captured["up"], 768)
        self.assertEqual(captured["down"], 625)
        self.assertAlmostEqual(fs_out, 245.76e6)

    def test_empty_signal_remains_empty(self) -> None:
        signal = np.array([], dtype=np.complex64)
        out, fs_out = apply_tx_upsampling(signal, fs_in=200.0, fs_target=245.76)

        self.assertEqual(out.size, 0)
        self.assertEqual(fs_out, 245.76)

    def test_raises_when_scipy_unavailable(self) -> None:
        signal = np.array([1.0, 2.0], dtype=np.float32)

        with patch("transceiver.helpers.interpolation.importlib.util.find_spec", return_value=None):
            with self.assertRaises(RuntimeError):
                apply_tx_upsampling(signal, fs_in=200.0, fs_target=245.76)


if __name__ == "__main__":
    unittest.main()
