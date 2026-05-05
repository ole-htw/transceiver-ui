import numpy as np


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



def _robust_sigma_1d(x: np.ndarray) -> float:
    """Return a robust sigma estimate for a 1-D array."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad

    if sigma <= 1e-12 and x.size >= 4:
        q25, q75 = np.percentile(x, [25.0, 75.0])
        sigma = float((q75 - q25) / 1.349) if q75 > q25 else 0.0

    if sigma <= 1e-12 and x.size >= 2:
        sigma = float(np.std(x))

    return max(0.0, sigma)


def _as_finite_1d(y: np.ndarray) -> np.ndarray:
    """Return a finite float copy, linearly interpolating non-finite samples."""
    y = np.asarray(y, dtype=float).copy()
    if y.size == 0 or np.all(np.isfinite(y)):
        return y

    finite = np.isfinite(y)
    if not np.any(finite):
        return np.zeros_like(y, dtype=float)

    x = np.arange(y.size)
    y[~finite] = np.interp(x[~finite], x[finite], y[finite])
    return y


def _odd_window(value: int, *, minimum: int = 3) -> int:
    """Return an odd integer window >= minimum."""
    value = max(int(minimum), int(value))
    if value % 2 == 0:
        value += 1
    return value


def _smooth_1d_edge(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding, preserving array length."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y.copy()

    window = _odd_window(window, minimum=1)
    if window <= 1:
        return y.copy()

    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    return np.convolve(np.pad(y, pad, mode="edge"), kernel, mode="valid")


def _merge_close_indices(indices: list[int], *, max_gap: int) -> list[list[int]]:
    """Merge sorted indices into runs when consecutive hits are close."""
    if not indices:
        return []

    max_gap = max(1, int(max_gap))
    sorted_indices = sorted({int(i) for i in indices})
    runs: list[list[int]] = [[sorted_indices[0]]]

    for idx in sorted_indices[1:]:
        if idx - runs[-1][-1] <= max_gap:
            runs[-1].append(idx)
        else:
            runs.append([idx])

    return runs


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
    shoulder_merge_distance: int | None = None,
    residual_window: int | None = None,
    max_shoulders_per_interval: int = 1,
) -> list[int]:
    """Return shoulder-like peak candidates within the active peak group.

    Detection and placement are deliberately separated:

    * detection uses local signed slope relief on each monotonic flank,
    * placement uses the + -> - zero crossing of the local slope residual
      plus the maximum of the locally integrated slope residual.

    This removes the common half-lobe bias:
    descending-flank shoulders move right from the early relief knee toward the
    hidden echo center, while ascending-flank shoulders move left from the late
    relief knee toward the hidden echo center.
    """
    work_mag = _as_finite_1d(mag)
    n = work_mag.size
    if n < 7:
        return []

    left_bound = int(np.clip(left_bound, 0, n - 1))
    right_bound = int(np.clip(right_bound, 0, n - 1))
    if right_bound - left_bound < 6:
        return []

    center_idx = int(np.clip(center_idx, 0, n - 1))
    smooth_window = _odd_window(smooth_window, minimum=3)
    slope_window = max(1, int(slope_window))
    min_peak_distance = max(1, int(min_peak_distance))
    snap_window = max(0, int(snap_window))

    if residual_window is None:
        residual_window = max(
            snap_window + 1,
            smooth_window,
            2 * slope_window + 1,
        )
    residual_window = max(1, int(residual_window))

    if shoulder_merge_distance is None:
        shoulder_merge_distance = max(
            min_peak_distance + 1,
            smooth_window,
            2 * slope_window + 1,
            residual_window,
        )
    shoulder_merge_distance = max(1, int(shoulder_merge_distance))
    max_shoulders_per_interval = max(1, int(max_shoulders_per_interval))

    smooth_mag = _smooth_1d_edge(work_mag, smooth_window)
    slope = np.gradient(smooth_mag)

    relief_window = _odd_window(
        max(
            3 * smooth_window,
            2 * residual_window + 1,
            4 * slope_window + 3,
        ),
        minimum=3,
    )
    slope_background = _smooth_1d_edge(slope, relief_window)
    slope_relief = slope - slope_background

    strict = sorted(
        {
            int(i)
            for i in strict_peak_indices
            if left_bound <= int(i) <= right_bound
        }
    )
    if len(strict) < 2:
        return []

    strict_set = set(strict)

    def _too_close_to_strict(idx: int) -> bool:
        return any(abs(int(idx) - peak_idx) <= min_peak_distance for peak_idx in strict)

    def _valid_marker_idx(idx: int, segment_left: int, segment_right: int) -> bool:
        idx = int(idx)
        return (
            segment_left <= idx <= segment_right
            and left_bound <= idx <= right_bound
            and idx not in strict_set
            and not _too_close_to_strict(idx)
            and float(work_mag[idx]) >= float(min_height)
        )

    high_freq_residual = work_mag - smooth_mag
    global_mag_noise = _robust_sigma_1d(
        high_freq_residual[left_bound : right_bound + 1]
    )
    global_mag_span = float(np.ptp(smooth_mag[left_bound : right_bound + 1]))

    residual_floor_global = max(
        0.0,
        float(noise_sigma_factor) * global_mag_noise,
        0.0025 * global_mag_span,
    )

    def _crossing_idx(
        *,
        idx0: int,
        idx1: int,
        level: float = 0.0,
    ) -> int:
        idx0 = int(idx0)
        idx1 = int(idx1)
        y0 = float(slope_relief[idx0] - level)
        y1 = float(slope_relief[idx1] - level)
        denom = y1 - y0

        if abs(denom) <= 1e-12:
            return idx1

        frac = float(np.clip(-y0 / denom, 0.0, 1.0))
        return int(np.clip(round(idx0 + frac * (idx1 - idx0)), 0, n - 1))

    def _slope_zero_center(
        *,
        run: list[int],
        core_idx: int,
        segment_left: int,
        segment_right: int,
        mode: str,
        margin: int,
    ) -> int:
        core_idx = int(core_idx)
        search_span = max(
            residual_window + len(run) + 1,
            shoulder_merge_distance,
            smooth_window + 2 * slope_window + 1,
        )

        segment_left = int(segment_left)
        segment_right = int(segment_right)
        search_left = max(segment_left + margin, left_bound + margin)
        search_right = min(segment_right - margin, right_bound - margin)

        if search_right < search_left:
            return int(np.clip(core_idx, segment_left, segment_right))

        if mode == "descending":
            start = int(np.clip(max(core_idx, run[-1]), search_left, search_right))
            end = int(np.clip(start + search_span, search_left, search_right))

            for j in range(start + 1, end + 1):
                if slope_relief[j - 1] > 0.0 and slope_relief[j] <= 0.0:
                    return _crossing_idx(idx0=j - 1, idx1=j)

            xs = np.arange(start, end + 1, dtype=int)

        elif mode == "ascending":
            start = int(np.clip(min(core_idx, run[0]), search_left, search_right))
            end = int(np.clip(start - search_span, search_left, search_right))

            for j in range(start, end, -1):
                if slope_relief[j - 1] > 0.0 and slope_relief[j] <= 0.0:
                    return _crossing_idx(idx0=j - 1, idx1=j)

            xs = np.arange(end, start + 1, dtype=int)

        else:
            raise ValueError("mode must be 'descending' or 'ascending'")

        if xs.size == 0:
            return int(np.clip(core_idx, segment_left, segment_right))

        best = int(xs[np.argmin(np.abs(slope_relief[xs]))])
        return int(np.clip(best, segment_left, segment_right))

    def _integrated_relief_best(
        *,
        event_left: int,
        event_right: int,
        segment_left: int,
        segment_right: int,
        fallback_idx: int,
    ) -> tuple[int | None, float]:
        segment_left = int(np.clip(segment_left, left_bound, right_bound))
        segment_right = int(np.clip(segment_right, left_bound, right_bound))

        event_left = int(np.clip(event_left, segment_left, segment_right))
        event_right = int(np.clip(event_right, segment_left, segment_right))
        if event_right - event_left < 2:
            return (
                (int(fallback_idx), 0.0)
                if _valid_marker_idx(fallback_idx, segment_left, segment_right)
                else (None, 0.0)
            )

        xs = np.arange(event_left, event_right + 1, dtype=int)
        component = np.cumsum(slope_relief[xs].astype(float))

        if component.size > 1:
            t = np.linspace(0.0, 1.0, component.size)
            component = component - (component[0] + t * (component[-1] - component[0]))

        smooth_component = _smooth_1d_edge(
            component,
            _odd_window(min(smooth_window, max(1, component.size)), minimum=1),
        )

        valid = np.array(
            [
                _valid_marker_idx(int(x), segment_left, segment_right)
                for x in xs
            ],
            dtype=bool,
        )

        if not np.any(valid):
            return (
                (int(fallback_idx), 0.0)
                if _valid_marker_idx(fallback_idx, segment_left, segment_right)
                else (None, 0.0)
            )

        xs_valid = xs[valid]
        values_valid = smooth_component[valid]
        best_local = int(np.argmax(values_valid))
        best_idx = int(xs_valid[best_local])

        edge_level = float(np.median(smooth_component))
        gain = float(values_valid[best_local] - edge_level)

        if gain < residual_floor_global and _valid_marker_idx(
            fallback_idx, segment_left, segment_right
        ):
            return int(fallback_idx), gain

        return best_idx, gain

    def _linear_residual_snap(
        *,
        center_idx_est: int,
        event_left: int,
        event_right: int,
        segment_left: int,
        segment_right: int,
    ) -> tuple[int | None, float]:
        segment_left = int(np.clip(segment_left, left_bound, right_bound))
        segment_right = int(np.clip(segment_right, left_bound, right_bound))
        event_left = int(np.clip(event_left, segment_left, segment_right))
        event_right = int(np.clip(event_right, segment_left, segment_right))
        center_idx_est = int(np.clip(center_idx_est, event_left, event_right))

        snap_radius = max(snap_window, 1)
        snap_left = max(event_left, center_idx_est - snap_radius)
        snap_right = min(event_right, center_idx_est + snap_radius)
        if snap_right < snap_left:
            return (
                (center_idx_est, 0.0)
                if _valid_marker_idx(center_idx_est, segment_left, segment_right)
                else (None, 0.0)
            )

        xs = np.arange(event_left, event_right + 1, dtype=int)
        if event_right == event_left:
            baseline = np.array([float(smooth_mag[event_left])], dtype=float)
        else:
            t = (xs.astype(float) - float(event_left)) / float(event_right - event_left)
            baseline = (
                float(smooth_mag[event_left])
                + t * (float(smooth_mag[event_right]) - float(smooth_mag[event_left]))
            )

        residual = smooth_mag[xs] - baseline
        snap_xs = np.arange(snap_left, snap_right + 1, dtype=int)
        valid = np.array(
            [
                _valid_marker_idx(int(x), segment_left, segment_right)
                for x in snap_xs
            ],
            dtype=bool,
        )

        if not np.any(valid):
            return (
                (center_idx_est, 0.0)
                if _valid_marker_idx(center_idx_est, segment_left, segment_right)
                else (None, 0.0)
            )

        residual_lookup = {
            int(x): float(residual[int(x) - event_left])
            for x in xs
        }
        values = np.array(
            [residual_lookup[int(x)] for x in snap_xs[valid]],
            dtype=float,
        )
        best_idx = int(snap_xs[valid][int(np.argmax(values))])
        best_residual = float(np.max(values))
        return best_idx, best_residual

    def _segment_candidates(
        *,
        segment_left: int,
        segment_right: int,
        mode: str,
    ) -> list[tuple[int, float]]:
        segment_left = int(np.clip(segment_left, left_bound, right_bound))
        segment_right = int(np.clip(segment_right, left_bound, right_bound))

        if segment_right - segment_left < max(6, 2 * slope_window + 2):
            return []

        delta = float(smooth_mag[segment_right] - smooth_mag[segment_left])

        if mode == "descending":
            if delta >= 0.0:
                return []
            signed_relief = slope_relief
            expected_slope_sign = -1.0
        elif mode == "ascending":
            if delta <= 0.0:
                return []
            signed_relief = -slope_relief
            expected_slope_sign = 1.0
        else:
            raise ValueError("mode must be 'descending' or 'ascending'")

        segment_len = max(1, segment_right - segment_left)
        trend_slope = abs(delta) / float(segment_len)

        margin = max(1, slope_window, min_peak_distance)
        search_left = max(segment_left + margin, left_bound + margin)
        search_right = min(segment_right - margin, right_bound - margin)
        if search_right < search_left:
            return []

        signed_segment = signed_relief[search_left : search_right + 1]
        slope_segment = slope[search_left : search_right + 1]

        relief_sigma = _robust_sigma_1d(signed_segment)
        slope_sigma = _robust_sigma_1d(slope_segment)

        if relief_sigma <= 1e-12:
            relief_sigma = max(
                1e-12,
                0.01 * float(np.ptp(signed_segment))
                if signed_segment.size > 1
                else 1e-12,
            )

        if slope_sigma <= 1e-12:
            slope_sigma = max(
                1e-12,
                0.01 * float(np.ptp(slope_segment))
                if slope_segment.size > 1
                else 1e-12,
            )

        threshold_from_mad = max(0.0, float(slope_threshold_factor)) * relief_sigma
        trend_floor = 0.025 * trend_slope
        trend_cap = 0.35 * trend_slope
        local_threshold = max(
            1e-12,
            trend_floor,
            min(threshold_from_mad, trend_cap),
        )

        sign_eps = max(
            1e-12,
            min(
                max(0.0, float(slope_eps_factor)) * slope_sigma,
                0.35 * trend_slope if trend_slope > 0.0 else np.inf,
            ),
        )

        raw_hits: list[int] = []
        hit_score: dict[int, float] = {}

        for i in range(search_left, search_right + 1):
            i = int(i)
            if _too_close_to_strict(i):
                continue
            if float(work_mag[i]) < float(min_height):
                continue

            score_i = float(signed_relief[i])
            if score_i < local_threshold:
                continue

            left_slice = slope[max(segment_left, i - slope_window) : i]
            right_slice = slope[
                i + 1 : min(segment_right + 1, i + 1 + slope_window)
            ]
            if left_slice.size == 0 or right_slice.size == 0:
                continue

            left_slope = float(np.median(left_slice))
            right_slope = float(np.median(right_slice))

            if expected_slope_sign > 0.0:
                sign_ok = left_slope >= -sign_eps and right_slope >= -sign_eps
            else:
                sign_ok = left_slope <= sign_eps and right_slope <= sign_eps

            if not sign_ok:
                continue

            raw_hits.append(i)
            hit_score[i] = score_i

        if not raw_hits:
            return []

        candidates: list[tuple[int, float]] = []
        side_band = max(slope_window, smooth_window // 2, 1)

        for run in _merge_close_indices(raw_hits, max_gap=shoulder_merge_distance):
            core_idx = int(max(run, key=lambda idx: hit_score.get(int(idx), 0.0)))
            event_score = float(hit_score.get(core_idx, 0.0))

            left_band = signed_relief[
                max(segment_left, run[0] - side_band) : run[0]
            ]
            right_band = signed_relief[
                run[-1] + 1 : min(segment_right + 1, run[-1] + 1 + side_band)
            ]

            side_values: list[float] = []
            if left_band.size:
                side_values.append(float(np.median(left_band)))
            if right_band.size:
                side_values.append(float(np.median(right_band)))

            boundary_level = max(side_values) if side_values else 0.0
            event_contrast = event_score - boundary_level

            if (
                event_contrast < 0.12 * local_threshold
                and event_score < 1.50 * local_threshold
            ):
                continue

            zero_idx = _slope_zero_center(
                run=run,
                core_idx=core_idx,
                segment_left=segment_left,
                segment_right=segment_right,
                mode=mode,
                margin=margin,
            )

            event_left = max(
                segment_left + margin,
                min(run[0], zero_idx) - residual_window,
            )
            event_right = min(
                segment_right - margin,
                max(run[-1], zero_idx) + residual_window,
            )

            best_idx, component_gain = _integrated_relief_best(
                event_left=event_left,
                event_right=event_right,
                segment_left=segment_left,
                segment_right=segment_right,
                fallback_idx=zero_idx,
            )
            if best_idx is None:
                continue

            max_center_shift = max(
                residual_window,
                smooth_window,
                2 * slope_window + 1,
            )
            if abs(best_idx - zero_idx) > max_center_shift and _valid_marker_idx(
                zero_idx, segment_left, segment_right
            ):
                best_idx = int(zero_idx)

            snap_idx, snap_residual = _linear_residual_snap(
                center_idx_est=best_idx,
                event_left=event_left,
                event_right=event_right,
                segment_left=segment_left,
                segment_right=segment_right,
            )
            if (
                snap_idx is not None
                and abs(snap_idx - best_idx) <= max(1, snap_window)
                and snap_residual >= residual_floor_global
            ):
                best_idx = int(snap_idx)

            if not _valid_marker_idx(best_idx, segment_left, segment_right):
                if _valid_marker_idx(zero_idx, segment_left, segment_right):
                    best_idx = int(zero_idx)
                elif _valid_marker_idx(core_idx, segment_left, segment_right):
                    best_idx = int(core_idx)
                else:
                    continue

            placement_gain = max(
                float(component_gain),
                float(snap_residual),
                residual_floor_global,
                0.01 * global_mag_span,
                1e-12,
            )
            combined_score = (
                event_score
                * max(event_contrast, 0.20 * local_threshold, 1e-12)
                * placement_gain
            )
            candidates.append((int(best_idx), float(combined_score)))

        candidates.sort(key=lambda item: item[1], reverse=True)

        kept: list[tuple[int, float]] = []
        for idx, score in candidates:
            if any(
                abs(idx - kept_idx) <= shoulder_merge_distance
                for kept_idx, _ in kept
            ):
                continue
            kept.append((idx, score))

        return kept

    all_candidates: list[tuple[int, float]] = []

    for a, b in zip(strict[:-1], strict[1:]):
        a = int(a)
        b = int(b)
        if b - a < 6:
            continue

        interval_left = max(left_bound, a)
        interval_right = min(right_bound, b)
        if interval_right - interval_left < 6:
            continue

        valley = int(
            interval_left
            + np.argmin(smooth_mag[interval_left : interval_right + 1])
        )

        interval_candidates: list[tuple[int, float]] = []

        interval_candidates.extend(
            _segment_candidates(
                segment_left=interval_left,
                segment_right=valley,
                mode="descending",
            )
        )

        if include_left_shoulders:
            interval_candidates.extend(
                _segment_candidates(
                    segment_left=valley,
                    segment_right=interval_right,
                    mode="ascending",
                )
            )

        interval_candidates.sort(key=lambda item: item[1], reverse=True)

        kept_in_interval: list[tuple[int, float]] = []
        for idx, score in interval_candidates:
            if any(
                abs(idx - kept_idx) <= shoulder_merge_distance
                for kept_idx, _ in kept_in_interval
            ):
                continue

            kept_in_interval.append((idx, score))

            if len(kept_in_interval) >= max_shoulders_per_interval:
                break

        all_candidates.extend(kept_in_interval)

    if not all_candidates:
        return []

    all_candidates.sort(key=lambda item: item[1], reverse=True)

    final: list[tuple[int, float]] = []
    for idx, score in all_candidates:
        if _too_close_to_strict(idx):
            continue

        if any(
            abs(idx - kept_idx) <= shoulder_merge_distance
            for kept_idx, _ in final
        ):
            continue

        final.append((idx, score))

    return sorted(int(idx) for idx, _ in final)

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
    """Return LOS + first echo indices using the shared peak grouping logic."""
    _highest_idx, los_idx, echo_indices, _group_indices = classify_peak_group_from_mag(
        mag,
        peaks_before=0,
        peaks_after=1,
        min_rel_height=0.0,
        repetition_period_samples=repetition_period_samples,
    )
    echo_indices = filter_echo_indices_by_noise_prominence(
        mag,
        los_idx=los_idx,
        echo_indices=echo_indices,
        repetition_period_samples=repetition_period_samples,
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
    global_baseline = float(np.median(mag_global))
    global_mad = float(np.median(np.abs(mag_global - global_baseline)))
    noise_sigma = 1.4826 * global_mad

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
    """Return local maxima indices around a center peak (before + after)."""
    if mag.size < 3:
        return []
    if center_idx is None:
        center_idx = int(np.argmax(mag))
    center_idx = int(np.clip(center_idx, 0, mag.size - 1))

    center_mag = float(mag[center_idx])
    min_height = max(0.0, float(min_rel_height)) * center_mag

    left_bound, right_bound = _peak_search_bounds(
        mag,
        center_idx=center_idx,
        repetition_period_samples=repetition_period_samples,
    )

    local_maxima = [
        i
        for i in range(max(1, left_bound), min(mag.size - 1, right_bound + 1))
        if (
            mag[i] >= mag[i - 1]
            and mag[i] >= mag[i + 1]
            and (mag[i] > mag[i - 1] or mag[i] > mag[i + 1])
            and mag[i] >= min_height
        )
    ]
    if not local_maxima:
        return []

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
        mag,
        center_idx=center_idx,
        strict_peak_indices=selected,
        min_height=min_height,
        left_bound=left_bound,
        right_bound=right_bound,
    )
    return sorted({*selected, *shoulder_candidates})


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

