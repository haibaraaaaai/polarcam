"""Keep the active background's source attached to the exact NPY bytes loaded."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np


def read_background_profile(path: Path) -> tuple[np.ndarray, dict]:
    path = Path(path).resolve()
    payload = path.read_bytes()
    profile = np.asarray(np.load(io.BytesIO(payload), allow_pickle=False), dtype=np.float32)
    if profile.ndim != 2 or not np.isfinite(profile).all():
        raise ValueError(f"Background must be a finite 2D array: {path}")
    fingerprint = hashlib.sha256(payload).hexdigest()
    source = None
    try:
        saved = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        # An old sidecar or an externally replaced active NPY must not lend its
        # source name to different pixels. Legacy files retain an unknown source.
        if isinstance(saved, dict) and saved.get("background_profile_sha256") == fingerprint:
            source = saved.get("background_profile_source_path")
    except (OSError, ValueError):
        pass
    return profile, {
        "background_profile_path": str(path),
        "background_profile_file": path.name,
        "background_profile_sha256": fingerprint,
        "background_profile_source_path": source,
    }


def save_active_background(profile: np.ndarray, path: Path, source_path: Path | None = None) -> dict:
    path = Path(path).resolve()
    arr = np.asarray(profile, dtype=np.float32)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError("Background must be a finite 2D array.")
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    payload = buffer.getvalue()
    metadata = {
        "background_profile_path": str(path),
        "background_profile_file": path.name,
        "background_profile_sha256": hashlib.sha256(payload).hexdigest(),
        "background_profile_source_path": str(Path(source_path).resolve()) if source_path else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npy.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    sidecar = path.with_suffix(".json")
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(sidecar)
    return metadata
