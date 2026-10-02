"""Intensity diagnostics and reference comparisons for manual tuning.

Two complementary metrics are computed for each trial:

- ``modulation_stats``: P98-P2 span of the ROI mean intensity. Sensitive to
  any slow drift of focus or illumination, so reference and test windows are
  only comparable when the manual Z sweep is faithfully repeated.
- ``xy_residual_stats``: extent of the detrended anisotropy locus (ax, ay)
  reconstructed from the four mosaic channel means. A moving-average
  detrend removes slow drift (the anisotropy of a static dipole is
  intensity-independent, so only the interference term moves it) while
  preserving the fast hand-wobble traversal of the residual interference
  arc/loop. This is the drift-immune interference metric and it also gives
  the picture a span number cannot: an open arc means the Z sweep did not
  cover a full fringe, a shrinking closed locus means the cancellation is
  working.
"""

import json
from pathlib import Path

import numpy as np


def roi_channel_means(values: np.ndarray, phase_x: int, phase_y: int) -> tuple:
    """Split a mosaic ROI into its four analyser sub-grids and mean each one.

    Channel order is 0, 90, 45, 135 for sensor mosaic [90, 45; 135, 0] with
    the supplied ROI origin parity, matching the diagnostic columns.
    """
    values = np.asarray(values)
    return (
        values[1 - phase_y::2, 1 - phase_x::2],
        values[phase_y::2, phase_x::2],
        values[phase_y::2, 1 - phase_x::2],
        values[1 - phase_y::2, phase_x::2],
    )


class IntensityDiagnostic:
    COLUMNS = (
        "received_s", "processed_s", "frame_index", "raw_mean_dn", "analysed_mean_dn",
        "i0_mean_dn", "i90_mean_dn", "i45_mean_dn", "i135_mean_dn",
        "raw_min_dn", "raw_max_dn", "zero_fraction", "saturated_fraction",
    )
    MAX_SAMPLES = 500_000

    def __init__(self):
        self.active = False
        self.data = None
        self.count = 0
        self.metadata = {}

    def begin(self, now: float, duration: float, metadata: dict) -> None:
        if self.active:
            raise ValueError("A diagnostic recording is already running.")
        if not np.isfinite(now) or not np.isfinite(duration) or not 10.0 <= duration <= 120.0:
            raise ValueError("Diagnostic duration must be 10 to 120 seconds.")
        self.data = np.empty((self.MAX_SAMPLES, len(self.COLUMNS)), dtype=np.float64)
        self.count = 0
        self.started_at = float(now)
        self.duration = float(duration)
        self.metadata = {
            "schema_version": 1,
            "requested_duration_s": self.duration,
            "host_started_at_perf_counter": self.started_at,
            "timestamp_basis": "Host SDK buffer receipt and GUI processing; not sensor exposure timestamps",
            "pixel_format_assumption": "Mono12 stored in uint16; saturation threshold 4095",
            "channel_order": "0,90,45,135; sensor mosaic [90,45;135,0] using supplied ROI origin",
            "columns": list(self.COLUMNS),
            "initial_settings": metadata,
            "events": [],
            "pre_start_frames_excluded": 0,
            "nonfinite_timestamps_excluded": 0,
        }
        self.active = True

    def mark(self, now: float, label: str, settings: dict | None = None) -> None:
        if self.active:
            self.metadata["events"].append({
                "host_time_s": float(now) - self.started_at,
                "label": str(label),
                "settings": settings,
            })

    def add(self, raw: np.ndarray, analysed_mean: float, received_at: float,
            processed_at: float, frame_index: int, origin: tuple[int, int]) -> None:
        if not self.active:
            return
        if not np.isfinite(received_at) or not np.isfinite(processed_at):
            self.metadata["nonfinite_timestamps_excluded"] += 1
            return
        elapsed = float(received_at) - self.started_at
        if elapsed < 0.0:
            self.metadata["pre_start_frames_excluded"] += 1
            return
        if elapsed >= self.duration:
            return
        values = np.asarray(raw)
        if values.ndim != 2 or min(values.shape) < 2:
            self.mark(processed_at, "Invalid ROI shape", {"shape": list(values.shape)})
            return
        phase_x, phase_y = int(origin[0]) % 2, int(origin[1]) % 2
        channel_grids = roi_channel_means(values, phase_x, phase_y)
        self.data[self.count] = (
            elapsed, float(processed_at) - self.started_at, frame_index,
            np.mean(values, dtype=np.float64), analysed_mean,
            *(np.mean(channel, dtype=np.float64) for channel in channel_grids),
            np.min(values), np.max(values),
            np.count_nonzero(values == 0) / values.size,
            np.count_nonzero(values >= 4095) / values.size,
        )
        self.count += 1
        if self.count >= self.MAX_SAMPLES:
            self.finish(processed_at, "sample_limit")

    def finish(self, now: float, reason: str) -> None:
        if self.active:
            self.active = False
            self.metadata["stop_reason"] = str(reason)
            self.metadata["host_stopped_s"] = float(now) - self.started_at
            self.metadata["sample_count"] = self.count

    def save(self, path: Path) -> None:
        if self.active or self.data is None:
            raise ValueError("Stop the diagnostic recording before saving.")
        metadata = json.dumps(self.metadata, allow_nan=False)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("wb") as output:
                np.savez(output, samples=self.data[:self.count], columns=np.asarray(self.COLUMNS), metadata_json=np.asarray(metadata))
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()


class IntensityTuner:
    def __init__(self):
        self.reference = None
        self.active = None
        self.last = None
        self.trials = []

    def reset(self) -> None:
        self.reference = None
        self.active = None
        self.last = None

    def begin(self, kind: str, label: str, context: dict, now: float, *, duration: float = 5.0, target: float = 10.0) -> None:
        if self.active is not None:
            raise ValueError("A trial is already running.")
        if kind not in ("reference", "test"):
            raise ValueError("Unknown trial type.")
        if not np.isfinite(duration) or not 2.0 <= duration <= 30.0:
            raise ValueError("Trial duration must be between 2 and 30 seconds.")
        if not np.isfinite(target) or target <= 1.0:
            raise ValueError("Reduction target must be greater than one.")
        if len(self.trials) >= 40:
            raise ValueError("Save and clear the 40 retained trials before continuing.")
        if kind == "test":
            if self.reference is None:
                raise ValueError("Record a valid reference first.")
            if self.reference["context"] != context or self.reference["duration_s"] != duration:
                raise ValueError("Settings or duration changed; record a new reference.")
        else:
            self.reference = None
        self.last = None
        self.active = {
            "id": len(self.trials) + 1,
            "kind": kind,
            "label": str(label).strip()[:80] or kind.title(),
            "context": dict(context),
            "duration_s": float(duration),
            "target": float(target),
            "start_at": float(now) + 0.5,
            "reference_id": self.reference["id"] if self.reference else None,
            "times_s": [],
            "intensity_dn": [],
            "channels": [],
            "clipped": False,
            "warnings": [],
            "max_delivery_age_s": 0.0,
        }

    def add(self, value: float, received_at: float, processed_at: float, *,
            clipped: bool = False, channels=None) -> None:
        trial = self.active
        if trial is None:
            return
        if not np.all(np.isfinite((value, received_at, processed_at))):
            if "Nonfinite sample" not in trial["warnings"]:
                trial["warnings"].append("Nonfinite sample")
            return
        elapsed = float(received_at) - trial["start_at"]
        if not 0.0 <= elapsed < trial["duration_s"]:
            return
        if trial["times_s"] and elapsed <= trial["times_s"][-1]:
            if "Non-increasing timestamps" not in trial["warnings"]:
                trial["warnings"].append("Non-increasing timestamps")
            return
        age = max(0.0, float(processed_at) - float(received_at))
        trial["max_delivery_age_s"] = max(trial["max_delivery_age_s"], age)
        if age > 0.25 and "GUI backlog exceeds 250 ms" not in trial["warnings"]:
            trial["warnings"].append("GUI backlog exceeds 250 ms")
        if channels is None:
            if trial["channels"]:
                if "Channel feed stopped" not in trial["warnings"]:
                    trial["warnings"].append("Channel feed stopped")
        else:
            values = tuple(float(item) for item in channels)
            if len(values) != 4 or not np.all(np.isfinite(values)):
                if "Invalid channel data" not in trial["warnings"]:
                    trial["warnings"].append("Invalid channel data")
                values = (float("nan"),) * 4
            trial["channels"].append(values)
        trial["times_s"].append(elapsed)
        trial["intensity_dn"].append(float(value))
        trial["clipped"] |= bool(clipped)

    def update(self, now: float) -> dict | None:
        trial = self.active
        if trial is None:
            return self.last
        elapsed = max(0.0, float(now) - trial["start_at"])
        complete = elapsed >= trial["duration_s"]
        warnings = list(trial["warnings"])
        stats = None
        if len(trial["intensity_dn"]) >= 32:
            stats = modulation_stats(trial["intensity_dn"], clipped=trial["clipped"])
            if stats["clipped"]:
                warnings.append("Clipped pixels")
            if stats["mean"] <= 0.0:
                warnings.append("Nonpositive intensity")
            if stats["span"] <= 0.0:
                warnings.append("Flat trace: verify Z sweep")
        if complete:
            if stats is None:
                warnings.append("Insufficient samples")
            expected = float(trial["context"].get("fps", 0.0)) * trial["duration_s"]
            times = trial["times_s"]
            if not times or times[0] > 0.25 or trial["duration_s"] - times[-1] > 0.25 or len(times) < 0.9 * expected:
                warnings.append("Incomplete capture")
            xy_stats = None
            if trial["channels"]:
                try:
                    xy_stats = xy_residual_stats(times, trial["channels"], duration_s=trial["duration_s"])
                    warnings.extend(xy_stats["warnings"])
                except ValueError as exc:
                    warnings.append(f"XY residual unavailable: {exc}")
            # A trial with no channel feed at all stays valid: the span metric
            # remains usable on its own (pre-XY workflow compatibility).
        else:
            xy_stats = None
        comparison = None
        if stats is not None and trial["kind"] == "test" and self.reference is not None:
            comparison = compare_modulation(self.reference["stats"], stats, target=trial["target"])
            warnings.extend(comparison["warnings"])
            comparison["target_reached"] &= complete and not warnings
            reference_xy = self.reference.get("xy")
            if xy_stats is not None and reference_xy is not None:
                comparison.update(compare_xy(reference_xy, xy_stats, target=trial["target"]))
                warnings.extend(comparison["xy_warnings"])
        result = {
            **trial,
            "stats": stats,
            "xy": xy_stats,
            "comparison": comparison,
            "warnings": list(dict.fromkeys(warnings)),
            "complete": complete,
            "elapsed_s": min(elapsed, trial["duration_s"]),
        }
        if complete:
            self.active = None
            self.last = result
            self.trials.append(result)
            if trial["kind"] == "reference" and not warnings:
                self.reference = result
        return result


def modulation_stats(values: np.ndarray, *, clipped: bool = False) -> dict:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 32:
        raise ValueError("At least 32 intensity samples are required.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Intensity samples must all be finite.")
    lower, upper = np.percentile(samples, [2.0, 98.0])
    mean = float(np.mean(samples))
    span = float(upper - lower)
    return {
        "samples": int(samples.size),
        "mean": mean,
        "p02": float(lower),
        "p98": float(upper),
        "span": span,
        "relative_span": span / mean if mean > 0.0 else None,
        "clipped": bool(clipped),
    }


def compare_modulation(reference: dict, current: dict, *, target: float = 10.0) -> dict:
    if not np.isfinite(target) or target <= 1.0:
        raise ValueError("Reduction target must be finite and greater than one.")
    warnings = []
    if reference["clipped"] or current["clipped"]:
        warnings.append("Clipped pixels")
    if reference["mean"] <= 0.0 or current["mean"] <= 0.0:
        warnings.append("Nonpositive intensity")
    if reference["span"] <= 0.0:
        warnings.append("Reference has no measurable variation")
    if current["span"] <= 0.0:
        warnings.append("Flat trace: verify Z sweep")
    mean_ratio = current["mean"] / reference["mean"] if reference["mean"] > 0.0 else None
    if mean_ratio is not None and not 0.5 <= mean_ratio <= 2.0:
        warnings.append("Brightness changed by more than 2x")
    reduction = reference["span"] / current["span"] if current["span"] > 0.0 else None
    relative = current["relative_span"]
    relative_reduction = (
        reference["relative_span"] / relative
        if relative is not None and relative > 0.0 and reference["relative_span"] is not None
        else None
    )
    reached = (
        not warnings
        and reduction is not None
        and relative_reduction is not None
        and reduction >= target
        and relative_reduction >= target
    )
    return {
        "reduction": reduction,
        "relative_reduction": relative_reduction,
        "mean_ratio": mean_ratio,
        "target": float(target),
        "target_reached": bool(reached),
        "warnings": warnings,
    }


def anisotropy_series(channels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample (ax, ay, r) from an (n, 4) array of channel means (0, 90, 45, 135).

    Samples with nonpositive pair sums or nonfinite channels come back NaN so
    callers can filter them explicitly.
    """
    values = np.asarray(channels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Channels must be an (n, 4) array in order 0, 90, 45, 135.")
    i0, i90, i45, i135 = (values[:, index] for index in range(4))
    sum_x, sum_y = i0 + i90, i45 + i135
    good = (
        np.all(np.isfinite(values), axis=1)
        & (sum_x > 0.0) & (sum_y > 0.0)
    )
    ax = np.full(values.shape[0], np.nan)
    ay = np.full(values.shape[0], np.nan)
    r = np.full(values.shape[0], np.nan)
    np.divide(i0 - i90, sum_x, out=ax, where=good)
    np.divide(i45 - i135, sum_y, out=ay, where=good)
    with np.errstate(invalid="ignore"):
        r = np.hypot(ax, ay)
    return ax, ay, r


def moving_average_residual(times_s, values, timescale_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Subtract a moving-average trend from a near-uniformly sampled series.

    The residual keeps everything faster than roughly 1 / timescale_s (the
    hand-wobble traversal of the interference locus) and discards slower
    drift (focus, illumination). Edges are handled by reflection; the
    returned mask excludes half the detrend window at each end.
    """
    times = np.asarray(times_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or values.shape != times.shape:
        raise ValueError("Matching one-dimensional times and values are required.")
    if not np.isfinite(timescale_s) or timescale_s <= 0.0:
        raise ValueError("Detrend timescale must be positive and finite.")
    span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    fps = len(times) / span if span > 0.0 else 0.0
    window = int(round(timescale_s * fps))
    if window < 3 or 2 * window >= len(values):
        raise ValueError(f"Not enough samples for a {timescale_s:g} s detrend window.")
    padded = np.pad(values, window, mode="reflect")
    trend = np.convolve(padded, np.ones(2 * window + 1) / (2 * window + 1), mode="valid")
    interior = slice(window, len(values) - window)
    return (values - trend)[interior]


def xy_residual_stats(times_s, channels, *, duration_s: float) -> dict:
    """Drift-immune interference metrics from detrended anisotropy samples.

    Extent is the P98-P2 box diagonal of the detrended (ax, ay) locus: the
    size of the residual interference arc/loop traced during the manual Z
    sweep. Detrending uses half the trial duration (clamped to 2-10 s) so the
    slow drift that defeats the raw P98-P2 intensity span is removed.
    """
    times = np.asarray(times_s, dtype=np.float64)
    values = np.asarray(channels, dtype=np.float64)
    if times.ndim != 1 or values.ndim != 2 or values.shape[0] != len(times) or values.shape[1] != 4:
        raise ValueError("Matching times and (n, 4) channel data are required.")
    ax, ay, r = anisotropy_series(values)
    finite = np.isfinite(ax) & np.isfinite(ay)
    warnings = []
    if not np.any(finite):
        raise ValueError("All anisotropy samples are invalid.")
    if np.count_nonzero(finite) < 0.9 * len(times):
        warnings.append("Over 10% invalid anisotropy samples")
    detrend_s = float(np.clip(0.4 * float(duration_s), 1.0, 10.0))
    ax_finite, ay_finite = ax[finite], ay[finite]
    t_finite = times[finite]
    fallback_used = False
    while True:
        try:
            ax_res = moving_average_residual(t_finite, ax_finite, detrend_s)
            ay_res = moving_average_residual(t_finite, ay_finite, detrend_s)
            break
        except ValueError:
            if detrend_s <= 0.125 * float(duration_s) + 1e-9:
                raise ValueError("Not enough samples for any detrend window") from None
            detrend_s = max(0.125 * float(duration_s), 0.5 * detrend_s)
            fallback_used = True
    if fallback_used:
        warnings.append(f"Reduced detrend window to {detrend_s:.2g} s")
    extent_x = float(np.percentile(ax_res, 98.0) - np.percentile(ax_res, 2.0))
    extent_y = float(np.percentile(ay_res, 98.0) - np.percentile(ay_res, 2.0))
    extent = float(np.hypot(extent_x, extent_y))
    r_mean = float(np.nanmean(r[finite]))
    relative_extent = extent / r_mean if r_mean > 0.01 else None
    if relative_extent is None:
        warnings.append("Anisotropy radius below 0.01: normalized extent unavailable")
    return {
        "samples": int(np.count_nonzero(finite)),
        "detrend_s": detrend_s,
        "extent": extent,
        "extent_x": extent_x,
        "extent_y": extent_y,
        "r_mean": r_mean,
        "relative_extent": relative_extent,
        "warnings": warnings,
        # Downsampled residual locus for plotting (ax, ay) pairs.
        "plot_ax": ax_res[:: max(1, len(ax_res) // 2000)].tolist(),
        "plot_ay": ay_res[:: max(1, len(ay_res) // 2000)].tolist(),
    }


def compare_xy(reference: dict, current: dict, *, target: float = 10.0) -> dict:
    """Extent-based reduction between a reference and a test trial."""
    warnings = []
    extent_reduction = None
    relative_reduction = None
    if reference.get("extent") and current.get("extent") and reference["extent"] > 0.0 and current["extent"] > 0.0:
        extent_reduction = reference["extent"] / current["extent"]
    else:
        warnings.append("XY extent not measurable in both trials")
    ref_rel = reference.get("relative_extent")
    cur_rel = current.get("relative_extent")
    if ref_rel and cur_rel and ref_rel > 0.0 and cur_rel > 0.0:
        relative_reduction = ref_rel / cur_rel
    xy_target_reached = bool(
        not warnings
        and extent_reduction is not None
        and extent_reduction >= target
        and (relative_reduction is None or relative_reduction >= target)
    )
    return {
        "extent_reduction": extent_reduction,
        "extent_relative_reduction": relative_reduction,
        "xy_target_reached": xy_target_reached,
        "xy_warnings": warnings,
    }


def display_envelope(times: np.ndarray, values: np.ndarray, columns: int = 600) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or values.shape != times.shape or columns < 1:
        raise ValueError("Matching one-dimensional arrays and positive columns are required.")
    if len(values) <= 2 * columns:
        return times, values
    boundaries = np.linspace(0, len(values), columns + 1, dtype=int)
    indices = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        block = values[start:stop]
        indices.extend(sorted((start + int(np.argmin(block)), start + int(np.argmax(block)))))
    return times[indices], values[indices]