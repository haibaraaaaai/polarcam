"""File labels and fixed-scale camera brightness diagnostics."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from time import strftime

import numpy as np


CAMERA_FULL_SCALE = 4095
BRIGHTNESS_EDGES = np.array([*range(0, 251, 25), 255], dtype=float)
_UINT16_PREVIEW_LUT = np.rint(
    np.clip(np.arange(65536, dtype=np.float32), 0, CAMERA_FULL_SCALE)
    * (255 / CAMERA_FULL_SCALE)
).astype(np.uint8)
_UINT16_PREVIEW_LUT.setflags(write=False)


def filename_label(value: str) -> str:
    """Keep a short, Windows-safe label; never let it change the directory."""
    return re.sub(r"[^\w-]+", "_", str(value), flags=re.UNICODE).strip("_-")[:64]


def capture_path(directory: Path, prefix: str, label: str = "", *, timestamp=None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = filename_label(label)
    stem = f"{prefix}_{timestamp or strftime('%Y%m%d-%H%M%S')}"
    if suffix:
        stem += "_" + suffix
    path = directory / (stem + ".npy")
    index = 2
    while path.exists() or path.with_suffix(".json").exists():
        path = directory / f"{stem}_{index}.npy"
        index += 1
    return path


def preview_u8(frame: np.ndarray) -> np.ndarray:
    """Mono12 -> 0..255, with no per-image autoscaling; uint8 is already scaled."""
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError("Expected a grayscale frame")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if arr.dtype == np.uint16:
        return _UINT16_PREVIEW_LUT[arr]
    work = np.nan_to_num(arr.astype(np.float32), nan=0, posinf=CAMERA_FULL_SCALE, neginf=0)
    return np.rint(np.clip(work, 0, CAMERA_FULL_SCALE) * (255 / CAMERA_FULL_SCALE)).astype(np.uint8)


def save_raw_peak(path: Path, peak: np.ndarray, frames: int) -> dict:
    """One small companion image retains raw clipping evidence after BG subtraction."""
    dest = path.parent / "diagnostics" / (path.stem + "_raw_peak.npy")
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, peak, allow_pickle=False)
    return {"path": dest.relative_to(path.parent).as_posix(),
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "frames": int(frames), "full_scale": CAMERA_FULL_SCALE}


def brightness_source(path: Path, metadata: dict | list[dict], saved_peak: np.ndarray, frames: int):
    """Prefer raw diagnostics, otherwise add back an exactly identified background."""
    if isinstance(metadata, list):
        # The directly adjacent sidecar precedes legacy *_meta.json/parent metadata.
        metadata = next((item for item in metadata if isinstance(item, dict)), {})
    output = metadata.get("fetch_frames_output", {})
    diagnostic = output.get("raw_peak") or metadata.get("actual", {}).get("raw_peak")
    diagnostic_issue = False
    if diagnostic:
        try:
            candidate = path.parent / diagnostic["path"]
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != diagnostic["sha256"]:
                raise ValueError("hash mismatch")
            peak = np.load(candidate, allow_pickle=False)
            if peak.shape != saved_peak.shape or int(diagnostic["frames"]) != frames:
                raise ValueError("shape/frame-count mismatch")
            if not np.all(np.isfinite(peak)):
                raise ValueError("nonfinite pixels")
            return peak, float(diagnostic["full_scale"]), True, "Raw camera pixels"
        except (OSError, ValueError, KeyError, TypeError):
            diagnostic_issue = True
    bg = metadata.get("background", {})
    subtracted = bg.get("background_subtracted", output.get("background_subtracted"))
    full_scale = 255 if saved_peak.dtype == np.uint8 else CAMERA_FULL_SCALE
    if subtracted is False and saved_peak.dtype != np.uint8:
        return saved_peak, full_scale, True, "Raw camera pixels"
    if subtracted is True and saved_peak.dtype != np.uint8:
        recovered = _restore_peak_background(path, metadata, saved_peak)
        if recovered is not None:
            return recovered, full_scale, True, "Exact BG added back for clipping check; zero-clipped pixels are upper bounds"
    reason = "Raw diagnostic missing/invalid" if diagnostic_issue else "BG-subtracted" if subtracted else "Raw provenance unavailable"
    return saved_peak, full_scale, False, f"{reason}; saved pixels only"


def _restore_peak_background(path: Path, metadata: dict, saved_peak: np.ndarray):
    """Invert integer-profile subtraction for bright pixels; never guess a legacy BG.

    Positive saved peaks recover exactly. A zero peak only bounds the original
    by the BG value; accept that bound only when it cannot hide saturation.
    Fractional profiles have rounding ambiguity and need the raw diagnostic.
    """
    bg = metadata.get("background", {})
    output = metadata.get("fetch_frames_output", {})
    digest = bg.get("background_profile_sha256") or output.get("background_profile_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
        return None
    source = bg.get("background_profile_path") or output.get("background_profile_path")
    candidates = [Path(source)] if source else []
    for parent in (path.parent, *list(path.parents[1:4])):
        candidates.append(parent / "backgrounds" / "used" / (digest + ".npy"))
    for candidate in candidates:
        try:
            payload = candidate.read_bytes()
            if hashlib.sha256(payload).hexdigest().lower() != digest.lower():
                continue
            import io
            profile = np.asarray(np.load(io.BytesIO(payload), allow_pickle=False), dtype=float)
            if profile.ndim != 2 or not np.isfinite(profile).all() or (profile < 0).any():
                continue
            if profile.shape != saved_peak.shape:
                roi = metadata.get("roi") or output.get("roi") or {}
                x, y = int(roi.get("OffsetX", roi.get("x", -1))), int(roi.get("OffsetY", roi.get("y", -1)))
                if min(x, y) < 0:
                    continue
                h, w = saved_peak.shape
                profile = profile[y:y+h, x:x+w]
            if profile.shape != saved_peak.shape or not np.array_equal(profile, np.rint(profile)):
                continue
            restored = np.asarray(saved_peak, dtype=float) + profile
            if ((saved_peak == 0) & (profile >= CAMERA_FULL_SCALE - 1)).any():
                continue
            if not np.isfinite(restored).all() or (restored > CAMERA_FULL_SCALE).any():
                continue
            return restored
        except (OSError, ValueError, TypeError):
            continue
    return None


def spot_brightness(peak: np.ndarray, bounds, full_scale=CAMERA_FULL_SCALE) -> dict:
    values = []
    for x0, x1, y0, y1 in bounds:
        patch = peak[y0:y1, x0:x1]
        values.append(float(np.max(patch)) if patch.size else np.nan)
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    scaled = np.clip(values * (255 / full_scale), 0, 255)
    counts, _ = np.histogram(scaled, BRIGHTNESS_EDGES)
    return {"total": len(values), "near": int(np.sum(scaled >= 250)),
            "clipped": int(np.sum(values >= full_scale)), "counts": counts.tolist()}
