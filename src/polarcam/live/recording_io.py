"""Read timing metadata and separate v3 phase markers from measurement frames."""

import json
import math
from pathlib import Path

import numpy as np


def read_recording_metadata(path: Path) -> list[dict]:
    candidates = [
        path.with_suffix(".json"),
        path.with_name(f"{path.stem}_meta.json"),
        path.parent / "meta.json",
    ]
    metadata = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read recording metadata {candidate}: {exc}") from exc
        if isinstance(payload, dict):
            metadata.append(payload)
    return metadata


def _nested_value(payload: object, keys: tuple[str, ...]) -> object:
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def recording_fps(metadata: list[object]) -> float | None:
    paths = (
        ("actual", "fps"),
        ("actual", "timing", "resulting_fps"),
        ("actual", "timing_snapshot", "resulting_fps"),
        ("actual_fps",),
        ("resulting_fps",),
        ("fetch_frames_output", "actual_fps"),
        ("timing", "resulting_fps"),
        ("timing_snapshot", "resulting_fps"),
        ("actual", "timing", "fps"),
        ("actual", "timing_snapshot", "fps"),
        ("timing", "fps"),
        ("timing_snapshot", "fps"),
        ("fps",),
    )
    for keys in paths:
        for payload in metadata:
            value = _nested_value(payload, keys)
            if isinstance(value, bool):
                continue
            try:
                fps = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fps) and fps > 0.0:
                return fps
    return None


def recording_phase_marker(metadata: list[object]) -> bool | None:
    for payload in metadata:
        for keys in (
            ("actual", "phase_marker_appended"),
            ("phase_marker_appended",),
            ("fetch_frames_output", "phase_marker_appended"),
        ):
            value = _nested_value(payload, keys)
            if isinstance(value, bool):
                return value
    return None


def strip_phase_marker(
    frames: np.ndarray,
    marker_appended: bool | None = None,
    roi_meta: dict | None = None,
) -> np.ndarray:
    if marker_appended is False:
        return frames
    valid_marker = False
    if frames.ndim == 3 and frames.shape[0] >= 2:
        marker = frames[-1]
        positions = np.argwhere(marker[:2, :2] == 1)
        if len(positions) == 1 and np.count_nonzero(marker) == 1:
            valid_marker = True
    if not valid_marker:
        if marker_appended is True:
            raise ValueError("Recording metadata declares a phase marker, but the last frame is not one.")
        return frames
    if roi_meta is not None:
        marker_y, marker_x = positions[0]
        roi_meta["phase_x"] = int(marker_x)
        roi_meta["phase_y"] = int(marker_y)
    return frames[:-1]