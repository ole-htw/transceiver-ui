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
    """Return LOS + first echo indices using the shared peak grouping logic."""
    _highest_idx, los_idx, echo_indices, _group_indices = classify_peak_group(
        cc,
        peaks_before=0,
        peaks_after=1,
        min_rel_height=0.1,
    )
    echo_idx = echo_indices[0] if echo_indices else None
    return los_idx, echo_idx


def classify_peak_group(
    cc: np.ndarray,
    *,
    peaks_before: int = 3,
    peaks_after: int = 3,
    min_rel_height: float = 0.1,
    repetition_period_samples: int | None = None,
) -> tuple[int | None, int | None, list[int], list[int]]:
    """Return (highest_idx, los_idx, echo_indices, group_indices)."""
    mag = np.abs(cc)
    if mag.size == 0:
        return None, None, [], []

    highest_idx = int(np.argmax(mag))
    peak_indices = find_local_maxima_around_peak(
        cc,
        center_idx=highest_idx,
        peaks_before=peaks_before,
        peaks_after=peaks_after,
        min_rel_height=min_rel_height,
        repetition_period_samples=repetition_period_samples,
    )
    if not peak_indices:
        peak_indices = [highest_idx]

    group_indices = sorted({int(idx) for idx in peak_indices})
    if highest_idx not in group_indices:
        group_indices.append(highest_idx)
        group_indices.sort()

    los_idx = int(group_indices[0])
    echo_indices = [int(idx) for idx in group_indices[1:]]
    return highest_idx, los_idx, echo_indices, group_indices


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


def resolve_manual_los_idx(
    lags: np.ndarray,
    base_los_idx: int | None,
    manual_lags: dict[str, int | None] | None,
    *,
    peak_group_indices: list[int] | None = None,
    highest_idx: int | None = None,
    period_samples: int | None = None,
) -> tuple[int | None, bool]:
    """Return LOS idx with optional validation against the active peak group.

    When ``manual_lags['los']`` points outside the currently dominant
    cross-correlation group, the manual LOS is ignored and ``manual_lags['los']``
    is reset to ``None`` so subsequent frames snap back to the current group.
    """
    if lags.size == 0:
        return base_los_idx, False
    if manual_lags is None or manual_lags.get("los") is None:
        return base_los_idx, False

    manual_los = float(manual_lags["los"])
    min_lag = float(lags.min())
    max_lag = float(lags.max())
    if manual_los < min_lag or manual_los > max_lag:
        return base_los_idx, False

    manual_idx = int(np.abs(lags - manual_los).argmin())
    allow_manual = True

    if peak_group_indices:
        normalized = {
            int(idx)
            for idx in peak_group_indices
            if 0 <= int(idx) < lags.size
        }
        if normalized and manual_idx not in normalized:
            allow_manual = False

    if (
        allow_manual
        and highest_idx is not None
        and period_samples is not None
        and period_samples > 1
    ):
        highest_idx = int(np.clip(highest_idx, 0, lags.size - 1))
        highest_lag = float(lags[highest_idx])
        half_period = float(period_samples) / 2.0
        if abs(manual_los - highest_lag) > half_period:
            allow_manual = False

    if not allow_manual:
        manual_lags["los"] = None
        return base_los_idx, True

    return manual_idx, False


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
