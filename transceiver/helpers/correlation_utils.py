import numpy as np


def apply_manual_lags(
    lags: np.ndarray,
    los_idx: int | None,
    echo_idx: int | None,
    manual_lags: dict[str, int | None] | None,
) -> tuple[int | None, int | None]:
    """Return marker indices adjusted by manual lag selections."""
    if manual_lags is None or lags.size == 0:
        return los_idx, echo_idx
    manual_los = manual_lags.get("los")
    manual_echo = manual_lags.get("echo")
    min_lag = float(lags.min())
    max_lag = float(lags.max())
    if manual_los is not None and min_lag <= manual_los <= max_lag:
        los_idx = int(np.abs(lags - manual_los).argmin())
    if manual_echo is not None and min_lag <= manual_echo <= max_lag:
        echo_idx = int(np.abs(lags - manual_echo).argmin())
    return los_idx, echo_idx


def xcorr_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the full cross-correlation of *a* and *b* using FFT."""
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    A = np.fft.fft(a, nfft)
    B = np.fft.fft(b, nfft)
    cc = np.fft.ifft(A * np.conj(B))
    return np.concatenate((cc[-(len(b) - 1) :], cc[: len(a)]))


def autocorr_fft(x: np.ndarray) -> np.ndarray:
    """Return the full autocorrelation of *x* using FFT."""
    return xcorr_fft(x, x)


def find_los_echo(cc: np.ndarray) -> tuple[int | None, int | None]:
    """Return indices of the LOS peak and the first echo in ``cc``."""
    mag = np.abs(cc)
    if mag.size == 0:
        return None, None

    los = int(np.argmax(mag))
    echo = None
    for i in range(los + 1, len(mag) - 1):
        if mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]:
            echo = int(i)
            break

    if echo is None and los + 1 < len(mag):
        echo = int(np.argmax(mag[los + 1 :]) + los + 1)

    return los, echo


def find_local_maxima_around_peak(
    cc: np.ndarray,
    center_idx: int | None = None,
    *,
    peaks_before: int = 3,
    peaks_after: int = 3,
    min_rel_height: float = 0.1,
) -> list[int]:
    """Return local maxima indices around a center peak (before + after)."""
    mag = np.abs(cc)
    if mag.size < 3:
        return []
    if center_idx is None:
        center_idx = int(np.argmax(mag))
    center_idx = int(np.clip(center_idx, 0, mag.size - 1))

    center_mag = float(mag[center_idx])
    min_height = max(0.0, float(min_rel_height)) * center_mag

    # Constrain the search window to the lobe of ``center_idx``. Otherwise,
    # side-lobes of another strong LOS peak can be incorrectly attributed to
    # the selected LOS peak.
    left_bound = 0
    right_bound = mag.size - 1
    # Only very dominant neighbours should clip the search window. A lower
    # threshold can suppress valid echoes around the selected LOS peak.
    strong_peak_min_height = 0.9 * center_mag
    strong_peaks = [
        i
        for i in range(1, mag.size - 1)
        if (
            i != center_idx
            and mag[i] >= mag[i - 1]
            and mag[i] >= mag[i + 1]
            and mag[i] >= strong_peak_min_height
        )
    ]
    left_strong = max((i for i in strong_peaks if i < center_idx), default=None)
    right_strong = min((i for i in strong_peaks if i > center_idx), default=None)

    if left_strong is not None and left_strong + 1 < center_idx:
        left_bound = left_strong + int(np.argmin(mag[left_strong : center_idx + 1]))
    if right_strong is not None and center_idx + 1 < right_strong:
        right_bound = center_idx + int(np.argmin(mag[center_idx : right_strong + 1]))

    local_maxima = [
        i
        for i in range(max(1, left_bound), min(mag.size - 1, right_bound + 1))
        if (
            mag[i] >= mag[i - 1]
            and mag[i] >= mag[i + 1]
            and mag[i] >= min_height
        )
    ]
    if not local_maxima:
        return []

    before = [i for i in local_maxima if i < center_idx]
    after = [i for i in local_maxima if i > center_idx]

    before_sel = before[-max(0, peaks_before) :]
    after_sel = after[: max(0, peaks_after)]
    return before_sel + [center_idx] + after_sel


def lag_overlap(
    data_len: int, ref_len: int, lag: int
) -> tuple[int, int, int]:
    """Return (data_start, ref_start, length) for a given lag."""
    if lag >= 0:
        r_start = lag
        s_start = 0
        length = min(data_len - r_start, ref_len)
    else:
        r_start = 0
        s_start = -lag
        length = min(data_len, ref_len - s_start)
    return r_start, s_start, length
