import tempfile
import unittest

import numpy as np

from transceiver.helpers.iq_utils import (
    complex_to_interleaved_int16,
    save_interleaved,
)


class TestInterleavedInt16(unittest.TestCase):
    def test_complex_to_interleaved_int16_clips(self) -> None:
        data = np.array([40000 + 0j, 0 - 40000j], dtype=np.complex64)
        interleaved = complex_to_interleaved_int16(data)
        expected = np.array([32767, 0, 0, -32768], dtype=np.int16)
        np.testing.assert_array_equal(interleaved, expected)

    def test_save_interleaved_clips_in_file(self) -> None:
        data = np.array([1 + 0j, 0 - 1j], dtype=np.complex64)
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            save_interleaved(tmp.name, data, amplitude=40000.0)
            raw = np.fromfile(tmp.name, dtype=np.int16)
        expected = np.array([32767, 0, 0, -32768], dtype=np.int16)
        np.testing.assert_array_equal(raw, expected)


if __name__ == "__main__":
    unittest.main()
