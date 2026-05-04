import numpy as np


def _smooth_edge(x: np.ndarray, window: int) -> np.ndarray:
    """Moving average with edge padding to avoid boundary dips."""
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1

    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(np.asarray(x, dtype=float), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _robust_baseline_sigma(x: np.ndarray) -> tuple[float, float]:
    """Return median baseline and MAD-scaled robust sigma."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0, 0.0

    baseline = float(np.median(x))
    mad = float(np.median(np.abs(x - baseline)))
    sigma = 1.4826 * mad
    return baseline, sigma


def find_left_flank_shoulder_candidates_from_mag(
    mag: np.ndarray,
    *,
    center_idx: int,
    left_bound: int,
    right_bound: int,
    min_height: float,
    short_window: int = 5,
    long_window: int = 31,
    slope_window: int = 3,
    residual_sigma_factor: float = 2.0,
    kink_sigma_factor: float = 1.5,
    noise_sigma_factor: float = 0.8,
    min_peak_distance: int = 3,
    snap_window: int = 3,
) -> list[int]:
    """Detect weak LOS-like shoulders on the rising flank before center_idx.

    This catches cases where LOS is not a strict local maximum, but only a
    small hump or slope kink on the rising flank of a stronger later path.
    """
    if mag.size < 9:
        return []

    mag_f = np.asarray(mag, dtype=float)

    center_idx = int(np.clip(center_idx, 0, mag.size - 1))
    left_bound = int(np.clip(left_bound, 0, mag.size - 1))
    right_bound = int(np.clip(right_bound, 0, mag.size - 1))

    search_right = min(center_idx - max(1, int(min_peak_distance)), right_bound)
    if search_right - left_bound < 8:
        return []

    left_mag_segment = mag_f[left_bound : search_right + 1]
    mag_baseline, mag_sigma = _robust_baseline_sigma(left_mag_segment)

    # Do NOT force shoulders to reach the same relative height as real peaks.
    # LOS can be much weaker than the dominant correlation maximum.
    center_mag = float(mag_f[center_idx])
    relative_floor = 0.015 * center_mag

    height_threshold = max(
        relative_floor,
        mag_baseline + max(0.0, float(noise_sigma_factor)) * mag_sigma,
        0.25 * float(min_height),
    )

    # Compress dynamic range so the huge main peak does not dominate slope
    # and residual calculations.
    scale = max(1e-12, mag_baseline + mag_sigma)
    work = np.log1p(mag_f / scale)

    short = _smooth_edge(work, short_window)
    long = _smooth_edge(work, long_window)

    residual = short - long
    slope = np.gradient(short)

    res_segment = residual[left_bound : search_right + 1]
    res_baseline, res_sigma = _robust_baseline_sigma(res_segment)

    slope_segment = slope[left_bound : search_right + 1]
    _slope_baseline, slope_sigma = _robust_baseline_sigma(slope_segment)

    res_threshold = residual_sigma_factor * max(res_sigma, 1e-12)
    kink_threshold = kink_sigma_factor * max(slope_sigma, 1e-12)
    slope_eps = 0.2 * max(slope_sigma, 1e-12)

    candidates: set[int] = set()

    start = max(left_bound + int(slope_window), 1)
    stop = min(search_right - int(slope_window), mag.size - 2)

    for i in range(start, stop + 1):
        if float(mag_f[i]) < height_threshold:
            continue

        # Small hump after removing broad rising trend.
        is_residual_hump = (
            residual[i] >= residual[i - 1]
            and residual[i] >= residual[i + 1]
            and residual[i] - res_baseline >= res_threshold
        )

        left_slope = float(np.median(slope[i - slope_window : i]))
        right_slope = float(np.median(slope[i + 1 : i + 1 + slope_window]))

        # Kink on a rising flank: slope was clearly positive and then drops.
        is_rising_kink = (
            left_slope > slope_eps
            and (left_slope - right_slope) >= kink_threshold
        )

        if not (is_residual_hump or is_rising_kink):
            continue

        # Snap to strongest residual, not strongest magnitude, otherwise the
        # marker drifts toward the big LOS/main lobe peak.
        snap_left = max(left_bound, i - int(snap_window))
        snap_right = min(search_right, i + int(snap_window))
        snap_candidates = list(range(snap_left, snap_right + 1))

        if snap_candidates:
            i = int(max(snap_candidates, key=lambda j: float(residual[j])))

        if float(mag_f[i]) >= height_threshold:
            candidates.add(int(i))

    return _suppress_nearby_candidates(
        sorted(candidates),
        mag_f,
        min_distance=max(
            int(min_peak_distance),
            2 * int(snap_window) + 1,
        ),
    )

def _suppress_nearby_candidates(
    indices: list[int],
    mag: np.ndarray,
    min_distance: int,
) -> list[int]:
    """Merge nearby candidate peaks and keep only the strongest per cluster."""
    if not indices:
        return []

    min_distance = max(0, int(min_distance))
    sorted_indices = sorted(
        {int(idx) for idx in indices if 0 <= int(idx) < mag.size}
    )
    if not sorted_indices:
        return []

    clusters: list[list[int]] = [[sorted_indices[0]]]
    for idx in sorted_indices[1:]:
        if idx - clusters[-1][-1] <= min_distance:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])

    kept = [int(max(cluster, key=lambda j: float(mag[j]))) for cluster in clusters]
    return sorted(kept)


def _smooth_edge(x: np.ndarray, window: int) -> np.ndarray:
    """Moving average with edge padding, so small windows do not shift peaks."""
    window = max(1, int(window))
    if window <= 1 or x.size == 0:
        return np.asarray(x, dtype=float).copy()
    if window % 2 == 0:
        window += 1

    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(
        np.pad(np.asarray(x, dtype=float), (pad, pad), mode="edge"),
        kernel,
        mode="valid",
    )


def _robust_baseline_sigma(x: np.ndarray) -> tuple[float, float]:
    """Return median baseline and MAD-based sigma."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0, 0.0
    baseline = float(np.median(x))
    mad = float(np.median(np.abs(x - baseline)))
    return baseline, 1.4826 * mad


def _peak_search_bounds(
    mag: np.ndarray,
    *,
    center_idx: int,
    repetition_period_samples: int | None = None,
) -> tuple[int, int]:
    """Return inclusive [left_bound, right_bound] used for peak search."""
    center_mag = float(mag[center_idx])
    if repetition_period_samples is not None and repetition_period_samples > 1:
        half_period = max(1, int(round(repetition_period_samples / 2.0)))
        left_bound = max(0, center_idx - half_period)
        right_bound = min(mag.size - 1, center_idx + half_period)
        return left_bound, right_bound

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
    return left_bound, right_bound


def find_left_flank_los_candidates_from_mag(
    mag: np.ndarray,
    *,
    center_idx: int,
    left_bound: int,
    right_bound: int,
    min_rel_height: float = 0.005,
    short_window: int = 5,
    long_window: int = 31,
    slope_window: int = 3,
    residual_sigma_factor: float = 3.0,
    kink_sigma_factor: float = 2.5,
    noise_sigma_factor: float = 1.3,
    min_peak_distance: int = 3,
    snap_window: int = 3,
) -> list[int]:
    """Detect weak LOS-like bumps/kinks on the rising flank before center_idx.

    This is the important extension for cases where LOS is not a strict local
    maximum but only a small shoulder on the left flank of a later dominant
    correlation peak.

    The detector works on log-compressed magnitude and looks for:
    - a local high-pass residual hump on the broad rising flank, or
    - a kink where the positive slope drops noticeably.
    """
    if mag.size < 9:
        return []

    mag_f = np.asarray(mag, dtype=float)
    center_idx = int(np.clip(center_idx, 0, mag_f.size - 1))
    left_bound = int(np.clip(left_bound, 0, mag_f.size - 1))
    right_bound = int(np.clip(right_bound, 0, mag_f.size - 1))

    search_right = min(center_idx - max(1, int(min_peak_distance)), right_bound)
    if search_right - left_bound < 8:
        return []

    # Use the early part of the left side as noise/floor estimate. Using the
    # whole rising flank would overestimate the noise floor and hide weak LOS.
    left_span = search_right - left_bound + 1
    noise_stop = left_bound + max(5, left_span // 4)
    noise_stop = min(noise_stop, search_right + 1)
    noise_segment = mag_f[left_bound:noise_stop]
    if noise_segment.size < 3:
        noise_segment = mag_f[left_bound : search_right + 1]

    floor, noise_sigma = _robust_baseline_sigma(noise_segment)
    center_mag = float(mag_f[center_idx])

    # Weak LOS can be far below the dominant peak, so this floor is deliberately
    # lower than the usual strict peak min_height.
    amp_threshold = max(
        floor + max(0.0, float(noise_sigma_factor)) * noise_sigma,
        max(0.0, float(min_rel_height)) * center_mag,
    )

    smooth_mag = _smooth_edge(mag_f, max(3, short_window))
    above = [
        i
        for i in range(left_bound, search_right + 1)
        if smooth_mag[i] >= amp_threshold
    ]
    if not above:
        return []

    # Start slightly before the first sustained rise above the floor.
    search_left = max(left_bound, above[0] - max(2, snap_window))

    # Log compression keeps the huge main peak from dominating the derivative.
    scale = max(1e-12, floor + noise_sigma)
    work = np.log1p(mag_f / scale)

    short = _smooth_edge(work, max(3, short_window))
    long = _smooth_edge(work, max(5, long_window))
    residual = short - long
    slope = np.gradient(short)

    res_segment = residual[search_left : search_right + 1]
    res_baseline, res_sigma = _robust_baseline_sigma(res_segment)

    slope_segment = slope[search_left : search_right + 1]
    _slope_baseline, slope_sigma = _robust_baseline_sigma(slope_segment)
    slope_eps = max(1e-12, 0.2 * slope_sigma)
    kink_threshold = max(1e-12, float(kink_sigma_factor) * slope_sigma)

    candidates: set[int] = set()

    start = max(search_left + slope_window, 1)
    stop = min(search_right - slope_window, mag_f.size - 2)
    for i in range(start, stop + 1):
        if mag_f[i] < amp_threshold:
            continue

        # Small hump after trend removal.
        is_residual_hump = (
            residual[i] >= residual[i - 1]
            and residual[i] >= residual[i + 1]
            and residual[i] - res_baseline
            >= float(residual_sigma_factor) * max(res_sigma, 1e-12)
        )

        # Kink on a rising flank: slope is positive before the candidate and
        # drops afterwards. The point itself does not need to be a magnitude max.
        left_slope = float(np.median(slope[i - slope_window : i]))
        right_slope = float(np.median(slope[i + 1 : i + 1 + slope_window]))
        is_rising_kink = (
            left_slope > slope_eps
            and (left_slope - right_slope) >= kink_threshold
        )

        if not (is_residual_hump or is_rising_kink):
            continue

        # Snap to strongest residual, not strongest magnitude. Snapping to
        # magnitude would pull the marker toward the later dominant peak.
        snap_left = max(search_left, i - max(0, int(snap_window)))
        snap_right = min(search_right, i + max(0, int(snap_window)))
        snap_candidates = list(range(snap_left, snap_right + 1))
        if snap_candidates:
            i = int(max(snap_candidates, key=lambda j: float(residual[j])))

        if mag_f[i] >= amp_threshold:
            candidates.add(int(i))

    # Suppress using residual strength, not magnitude, to keep shoulder position.
    raw = sorted(candidates)
    if not raw:
        return []

    min_distance = max(1, int(min_peak_distance), 2 * max(0, int(snap_window)) + 1)
    clusters: list[list[int]] = [[raw[0]]]
    for idx in raw[1:]:
        if idx - clusters[-1][-1] <= min_distance:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])

    kept = [int(max(cluster, key=lambda j: float(residual[j]))) for cluster in clusters]
    return sorted(kept)


def find_shoulder_candidates_from_mag(
    mag: np.ndarray,
    *,
    center_idx: int,
    strict_peak_indices: list[int],
    min_height: float,
    left_bound: int,
    right_bound: int,
    include_left_shoulders: bool = True,
    smooth_window: int = 5,
    slope_window: int = 3,
    slope_threshold_factor: float = 1.0,
    slope_eps_factor: float = 0.2,
    min_peak_distance: int = 2,
    snap_window: int = 2,
    noise_sigma_factor: float = 1.0,
) -> list[int]:
    """Return shoulder-like candidates from slope inflections within one group.

    Strict local maxima should be selected first. Shoulders are supplemental and
    do not consume peaks_before / peaks_after quotas.

    This version supports:
    - right-side shoulders between peaks,
    - left-side shoulders before the dominant peak,
    - weak LOS humps on the rising flank.
    """
    if mag.size < 7:
        return []

    mag_f = np.asarray(mag, dtype=float)

    left_bound = int(np.clip(left_bound, 0, mag.size - 1))
    right_bound = int(np.clip(right_bound, 0, mag.size - 1))
    center_idx = int(np.clip(center_idx, 0, mag.size - 1))

    if right_bound - left_bound < 6:
        return []

    smooth_window = max(3, int(smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1

    slope_window = max(1, int(slope_window))
    min_peak_distance = max(1, int(min_peak_distance))
    snap_window = max(0, int(snap_window))

    smooth_mag = _smooth_edge(mag_f, smooth_window)
    slope = np.gradient(smooth_mag)

    slope_segment = slope[left_bound : right_bound + 1]
    slope_baseline, slope_sigma = _robust_baseline_sigma(slope_segment)

    slope_threshold = max(
        1e-12,
        float(slope_threshold_factor) * slope_sigma,
    )
    slope_eps = max(
        1e-12,
        float(slope_eps_factor) * slope_sigma,
    )

    mag_segment = mag_f[left_bound : right_bound + 1]
    mag_baseline, mag_sigma = _robust_baseline_sigma(mag_segment)

    strict = sorted(
        {
            int(i)
            for i in strict_peak_indices
            if left_bound <= int(i) <= right_bound
        }
    )
    if not strict:
        return []

    def _inside_search_interval(idx: int) -> bool:
        # Allow LOS shoulders on the left rising flank.
        if include_left_shoulders and left_bound < idx < center_idx:
            return True

        # Keep previous behaviour: shoulders between strict peaks.
        return any(a < idx < b for a, b in zip(strict[:-1], strict[1:]))

    candidates: set[int] = set()

    start = max(left_bound + slope_window, 1)
    stop = min(right_bound - slope_window, mag.size - 2)

    for i in range(start, stop + 1):
        if not _inside_search_interval(i):
            continue

        if any(abs(i - p) <= min_peak_distance for p in strict):
            continue

        if float(mag_f[i]) < float(min_height):
            continue

        left_slope = float(np.median(slope[i - slope_window : i]))
        mid_slope = float(slope[i])
        right_slope = float(np.median(slope[i + 1 : i + 1 + slope_window]))

        # Shoulder on falling flank.
        is_right_candidate = (
            left_slope < -slope_eps
            and right_slope < -slope_eps
            and mid_slope > left_slope
            and mid_slope > right_slope
            and (mid_slope - max(left_slope, right_slope)) >= slope_threshold
        )

        # Improved shoulder on rising flank:
        # not a true maximum, but a meaningful slope break / kink.
        is_left_candidate = (
            include_left_shoulders
            and i < center_idx
            and left_slope > slope_eps
            and (left_slope - right_slope) >= slope_threshold
        )

        if not (is_right_candidate or is_left_candidate):
            continue

        mag_prominence = float(mag_f[i]) - mag_baseline
        if mag_sigma > 1e-12:
            if mag_prominence < max(0.0, float(noise_sigma_factor)) * mag_sigma:
                continue
        elif mag_prominence <= 0.0:
            continue

        snap_left = max(left_bound, i - snap_window)
        snap_right = min(right_bound, i + snap_window)
        snap_candidates = [j for j in range(snap_left, snap_right + 1) if j not in strict]

        if snap_candidates:
            if is_right_candidate:
                # On falling side, snap to local magnitude.
                i = int(max(snap_candidates, key=lambda j: float(mag_f[j])))
            else:
                # On rising side, avoid snapping into the main peak.
                # Prefer the strongest slope break.
                i = int(
                    max(
                        snap_candidates,
                        key=lambda j: float(abs(np.gradient(smooth_mag)[j])),
                    )
                )

        if any(abs(i - p) <= min_peak_distance for p in strict):
            continue

        if left_bound <= i <= right_bound and float(mag_f[i]) >= float(min_height):
            candidates.add(int(i))

    raw_candidates = sorted(candidates)

    merged = _suppress_nearby_candidates(
        raw_candidates,
        mag_f,
        min_distance=max(min_peak_distance, 2 * snap_window + 1),
    )

    return merged

    def _inside_search_interval(idx: int) -> bool:
        # Allow a shoulder on the left rising flank before the dominant peak.
        if include_left_shoulders and left_bound < idx < center_idx:
            return True

        # Existing behavior: shoulders between strict peaks.
        return any(a < idx < b for a, b in zip(strict[:-1], strict[1:]))

    candidates: set[int] = set()

    for i in range(
        max(left_bound + slope_window, 1),
        min(right_bound - slope_window, mag.size - 2) + 1,
    ):
        if not _inside_search_interval(i):
            continue
        if any(abs(i - p) <= min_peak_distance for p in strict):
            continue
        if float(mag[i]) < float(min_height):
            continue

        left_slope = float(np.median(slope[i - slope_window : i]))
        mid_slope = float(slope[i])
        right_slope = float(np.median(slope[i + 1 : i + 1 + slope_window]))

        # Right-side shoulder: falling flank briefly flattens or rises.
        is_right_candidate = (
            left_slope < -slope_eps
            and right_slope < -slope_eps
            and mid_slope > left_slope
            and mid_slope > right_slope
            and (mid_slope - max(left_slope, right_slope)) >= slope_threshold
        )

        # Left-side shoulder: rising flank forms a small bump/kink.
        # This is intentionally different from the old "mid_slope is a local
        # minimum" rule. Here the relevant signal is a positive slope that
        # noticeably drops after the candidate.
        is_left_candidate = (
            include_left_shoulders
            and i < center_idx
            and left_slope > slope_eps
            and (left_slope - right_slope) >= slope_threshold
        )

        if not (is_right_candidate or is_left_candidate):
            continue

        mag_prominence = float(mag[i]) - mag_baseline
        if mag_sigma > 1e-12:
            if mag_prominence < max(0.0, float(noise_sigma_factor)) * mag_sigma:
                continue
        elif mag_prominence <= 0.0:
            continue

        if is_right_candidate:
            # For normal right-side shoulders, snap to magnitude.
            snap_left = max(left_bound, i - snap_window)
            snap_right = min(right_bound, i + snap_window)
            snap_candidates = [
                j for j in range(snap_left, snap_right + 1) if j not in strict
            ]
            if snap_candidates:
                i = int(
                    snap_candidates[
                        int(np.argmax([mag[j] for j in snap_candidates]))
                    ]
                )

        # For left shoulders, do not snap to magnitude, because that would drift
        # toward the dominant peak. Keep the kink location.

        if any(abs(i - p) <= min_peak_distance for p in strict):
            continue
        if left_bound <= i <= right_bound and float(mag[i]) >= float(min_height):
            candidates.add(int(i))

    raw_candidates = sorted(candidates)
    merged = _suppress_nearby_candidates(
        raw_candidates,
        mag,
        min_distance=max(min_peak_distance, 2 * snap_window + 1),
    )
    return merged


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


def find_los_echo(
    cc: np.ndarray,
    *,
    repetition_period_samples: int | None = None,
) -> tuple[int | None, int | None]:
    """Return LOS + first echo indices using a single magnitude conversion."""
    return find_los_echo_from_mag(
        np.abs(cc),
        repetition_period_samples=repetition_period_samples,
    )


def find_los_echo_from_mag(
    mag: np.ndarray,
    *,
    repetition_period_samples: int | None = None,
) -> tuple[int | None, int | None]:
    """Return LOS + first echo indices.

    LOS is treated as the earliest significant arrival in the dominant group,
    not necessarily as a strict local maximum.
    """
    _highest_idx, los_idx, echo_indices, _group_indices = classify_peak_group_from_mag(
        mag,
        peaks_before=4,
        peaks_after=1,
        min_rel_height=0.01,
        repetition_period_samples=repetition_period_samples,
        include_shoulders=True,
    )

    echo_indices = filter_echo_indices_by_noise_prominence(
        mag,
        los_idx=los_idx,
        echo_indices=echo_indices,
        repetition_period_samples=repetition_period_samples,
        noise_sigma_factor=0.3,
        min_echo_lag_samples=2,
    )

    echo_indices = _suppress_nearby_candidates(
        echo_indices,
        mag,
        min_distance=2,
    )

    echo_idx = echo_indices[0] if echo_indices else None
    return los_idx, echo_idx


def filter_echo_indices_by_noise_prominence(
    mag: np.ndarray,
    *,
    los_idx: int | None,
    echo_indices: list[int],
    repetition_period_samples: int | None = None,
    noise_sigma_factor: float = 0.3,
    min_echo_lag_samples: int = 2,
) -> list[int]:
    """Keep echo peaks that stand out from global background noise.

    The filtering does not depend on LOS peak height. Instead it compares each
    candidate echo against a robust global baseline (median) and global noise
    spread (MAD-scaled sigma estimate).
    """
    if mag.size == 0 or not echo_indices:
        return []

    los_idx_int = int(los_idx) if los_idx is not None else None
    cleaned_indices = sorted(
        {
            int(idx)
            for idx in echo_indices
            if (
                0 <= int(idx) < mag.size
                and (
                    los_idx_int is None
                    or int(idx) > los_idx_int + max(0, int(min_echo_lag_samples))
                )
            )
        }
    )
    if not cleaned_indices:
        return []

    mag_global = np.asarray(mag, dtype=float)
    global_baseline, noise_sigma = _robust_baseline_sigma(mag_global)

    filtered: list[int] = []
    for idx in cleaned_indices:
        prominence = float(mag[idx]) - global_baseline
        if prominence <= 0.0:
            continue
        if noise_sigma <= 1e-12:
            if prominence > 0.0:
                filtered.append(int(idx))
            continue
        if prominence >= max(0.0, float(noise_sigma_factor)) * noise_sigma:
            filtered.append(int(idx))
    return filtered


def classify_peak_group(
    cc: np.ndarray,
    *,
    peaks_before: int = 3,
    peaks_after: int = 3,
    min_rel_height: float = 0.1,
    repetition_period_samples: int | None = None,
) -> tuple[int | None, int | None, list[int], list[int]]:
    """Return (highest_idx, los_idx, echo_indices, group_indices)."""
    return classify_peak_group_from_mag(
        np.abs(cc),
        peaks_before=peaks_before,
        peaks_after=peaks_after,
        min_rel_height=min_rel_height,
        repetition_period_samples=repetition_period_samples,
    )


def classify_peak_group_from_mag(
    mag: np.ndarray,
    *,
    peaks_before: int = 3,
    peaks_after: int = 3,
    min_rel_height: float = 0.1,
    repetition_period_samples: int | None = None,
    include_shoulders: bool = False,
) -> tuple[int | None, int | None, list[int], list[int]]:
    """Return (highest_idx, los_idx, echo_indices, group_indices)."""
    if mag.size == 0:
        return None, None, [], []

    highest_idx = int(np.argmax(mag))
    peak_indices = find_local_maxima_around_peak_from_mag(
        mag,
        center_idx=highest_idx,
        peaks_before=peaks_before,
        peaks_after=peaks_after,
        min_rel_height=min_rel_height,
        repetition_period_samples=repetition_period_samples,
        include_shoulders=include_shoulders,
    )
    if not peak_indices:
        peak_indices = [highest_idx]

    group_indices = sorted({int(idx) for idx in peak_indices})
    if highest_idx not in group_indices:
        group_indices.append(highest_idx)
        group_indices.sort()

    # LOS is intentionally the earliest significant candidate in the active
    # group, not necessarily the largest magnitude peak.
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
    include_shoulders: bool = False,
) -> list[int]:
    """Return local maxima indices around a center peak (before + after)."""
    return find_local_maxima_around_peak_from_mag(
        np.abs(cc),
        center_idx=center_idx,
        peaks_before=peaks_before,
        peaks_after=peaks_after,
        min_rel_height=min_rel_height,
        repetition_period_samples=repetition_period_samples,
        include_shoulders=include_shoulders,
    )


def find_local_maxima_around_peak_from_mag(
    mag: np.ndarray,
    center_idx: int | None = None,
    *,
    peaks_before: int = 3,
    peaks_after: int = 3,
    min_rel_height: float = 0.1,
    repetition_period_samples: int | None = None,
    include_shoulders: bool = False,
) -> list[int]:
    """Return local maxima and optional shoulder candidates around center peak."""
    if mag.size < 3:
        return []

    mag_f = np.asarray(mag, dtype=float)

    if center_idx is None:
        center_idx = int(np.argmax(mag_f))
    center_idx = int(np.clip(center_idx, 0, mag_f.size - 1))

    center_mag = float(mag_f[center_idx])
    min_height = max(0.0, float(min_rel_height)) * center_mag

    left_bound, right_bound = _peak_search_bounds(
        mag_f,
        center_idx=center_idx,
        repetition_period_samples=repetition_period_samples,
    )

    local_maxima = [
        i
        for i in range(max(1, left_bound), min(mag_f.size - 1, right_bound + 1))
        if (
            mag_f[i] >= mag_f[i - 1]
            and mag_f[i] >= mag_f[i + 1]
            and (mag_f[i] > mag_f[i - 1] or mag_f[i] > mag_f[i + 1])
            and mag_f[i] >= min_height
        )
    ]

    before = [i for i in local_maxima if i < center_idx]
    after = [i for i in local_maxima if i > center_idx]

    before_count = max(0, int(peaks_before))
    after_count = max(0, int(peaks_after))

    before_sel = before[-before_count:] if before_count > 0 else []
    after_sel = after[:after_count] if after_count > 0 else []

    selected = before_sel + [center_idx] + after_sel

    if not include_shoulders:
        return selected

    shoulder_candidates = find_shoulder_candidates_from_mag(
        mag_f,
        center_idx=center_idx,
        strict_peak_indices=selected,
        min_height=min_height,
        left_bound=left_bound,
        right_bound=right_bound,
        include_left_shoulders=True,
        smooth_window=5,
        slope_window=3,
        slope_threshold_factor=1.0,
        slope_eps_factor=0.2,
        min_peak_distance=2,
        snap_window=2,
        noise_sigma_factor=0.8,
    )

    left_flank_candidates = find_left_flank_shoulder_candidates_from_mag(
        mag_f,
        center_idx=center_idx,
        left_bound=left_bound,
        right_bound=right_bound,
        min_height=min_height,
        short_window=5,
        long_window=31,
        slope_window=3,
        residual_sigma_factor=2.0,
        kink_sigma_factor=1.5,
        noise_sigma_factor=0.8,
        min_peak_distance=3,
        snap_window=3,
    )

    return sorted({*selected, *shoulder_candidates, *left_flank_candidates})


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
    constrain_to_peak_group: bool = True,
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

    if constrain_to_peak_group and peak_group_indices:
        normalized = {
            int(idx)
            for idx in peak_group_indices
            if 0 <= int(idx) < lags.size
        }
        if normalized and manual_idx not in normalized:
            allow_manual = False

    if (
        constrain_to_peak_group
        and allow_manual
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

