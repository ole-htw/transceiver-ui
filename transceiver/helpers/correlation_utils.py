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
    repetition_period_samples: int | None = None,
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

    if repetition_period_samples is not None and repetition_period_samples > 1:
        half_period = max(1, int(round(repetition_period_samples / 2.0)))
        left_bound = max(0, center_idx - half_period)
        right_bound = min(mag.size - 1, center_idx + half_period)
    else:
        # Fallback segmentation: walk from the center outwards until a local
        # minimum is reached on each side *and* a new dominant lobe is found
        # after that minimum. This keeps weaker echoes in the current group.
        left_bound = 0
        right_bound = mag.size - 1
        main_lobe_threshold = 0.8 * center_mag

        for idx in range(center_idx - 1, 0, -1):
            if mag[idx] <= mag[idx - 1] and mag[idx] <= mag[idx + 1]:
                has_new_main_lobe = any(
                    mag[j] >= mag[j - 1]
                    and mag[j] >= mag[j + 1]
                    and mag[j] >= main_lobe_threshold
                    for j in range(idx - 1, 0, -1)
                )
                if has_new_main_lobe:
                    left_bound = idx
                    break

        for idx in range(center_idx + 1, mag.size - 1):
            if mag[idx] <= mag[idx - 1] and mag[idx] <= mag[idx + 1]:
                has_new_main_lobe = any(
                    mag[j] >= mag[j - 1]
                    and mag[j] >= mag[j + 1]
                    and mag[j] >= main_lobe_threshold
                    for j in range(idx + 1, mag.size - 1)
                )
                if has_new_main_lobe:
                    right_bound = idx
                    break

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


def filter_peak_indices_to_period_group(
    lags: np.ndarray,
    peak_indices: list[int],
    anchor_idx: int | None,
    period_samples: int | None,
) -> list[int]:
    """Keep only peaks that belong to the same repetition group as ``anchor_idx``.

    With repeated TX sequences, cross-correlation maxima appear every
    ``period_samples``. This helper keeps only the peak indices inside a
    half-period window around the selected anchor lag, so marker points remain
    on one peak group and do not jump between adjacent repetitions.
    """
    if anchor_idx is None or lags.size == 0:
        return [int(i) for i in peak_indices]
    if period_samples is None or period_samples <= 1:
        return [int(i) for i in peak_indices]

    anchor_idx = int(np.clip(anchor_idx, 0, lags.size - 1))
    anchor_lag = float(lags[anchor_idx])
    half_period = float(period_samples) / 2.0

    filtered = []
    for idx in peak_indices:
        idx = int(idx)
        if idx < 0 or idx >= lags.size:
            continue
        if abs(float(lags[idx]) - anchor_lag) <= half_period:
            filtered.append(idx)
    return filtered


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
