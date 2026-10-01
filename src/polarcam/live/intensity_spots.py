"""Native polarization statistics and explicitly folded orientation distributions."""

import numpy as np


AVERAGING_METHODS = ("3D Cartesian", "Anisotropy X/Y", "Four intensities")


def channel_means(frame, bounds, origin=(0, 0)):
    values = np.asarray(frame)
    result = np.full((len(bounds), 4), np.nan, dtype=np.float64)
    for index, (left, right, top, bottom) in enumerate(bounds):
        crop = values[top:bottom, left:right]
        if crop.ndim != 2 or min(crop.shape) < 2:
            continue
        phase_x = (int(origin[0]) + left) % 2
        phase_y = (int(origin[1]) + top) % 2
        channels = (crop[1-phase_y::2, 1-phase_x::2], crop[phase_y::2, phase_x::2],
                    crop[phase_y::2, 1-phase_x::2], crop[1-phase_y::2, phase_x::2])
        result[index] = [np.mean(channel, dtype=np.float64) for channel in channels]
    return result


def anisotropy(intensities):
    values = np.asarray(intensities, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError("Expected channels in I0, I90, I45, I135 order.")
    denominator = values[..., (0, 2)] + values[..., (1, 3)]
    numerator = values[..., (0, 2)] - values[..., (1, 3)]
    valid = np.isfinite(values).all(axis=-1) & (values >= 0).all(axis=-1) & (denominator > 0).all(axis=-1)
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=valid[..., None])
    return result


def unit_vectors(xy, theta_lut, continuous=False):
    values = np.asarray(xy, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected a sequence of X/Y pairs.")
    radius = np.linalg.norm(values, axis=1)
    valid = np.isfinite(values).all(axis=1) & (radius <= float(theta_lut["r_max"]))
    result = np.full((len(values), 3), np.nan)
    if not valid.any():
        return result
    theta = np.interp(radius[valid], theta_lut["r"], theta_lut["theta_rad"])
    doubled_phi = np.arctan2(values[valid, 1], values[valid, 0])
    phi = np.unwrap(doubled_phi) / 2 if continuous else np.mod(doubled_phi / 2, np.pi)
    result[valid] = np.column_stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))
    return result


def average_directions(intensities, theta_lut, method="3D Cartesian", window_frames=None):
    values = np.asarray(intensities, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 4 or values.shape[0] < 1:
        raise ValueError("Expected nonempty frame/spot/channel data.")
    if method not in AVERAGING_METHODS:
        raise ValueError("Unknown averaging method.")
    window = len(values) if window_frames is None else int(window_frames)
    if window < 1:
        raise ValueError("Averaging window must be positive.")
    starts = np.arange(0, len(values), window)
    stops = np.minimum(starts + window, len(values))
    directions = np.full((len(starts), values.shape[1], 3), np.nan)
    valid_counts = np.zeros(directions.shape[:2], dtype=np.int64)
    resultant = np.full(directions.shape[:2], np.nan)
    xy = anisotropy(values)
    for spot_index in range(values.shape[1]):
        vectors = unit_vectors(xy[:, spot_index], theta_lut, continuous=True) if method == "3D Cartesian" else None
        for window_index, (start, stop) in enumerate(zip(starts, stops)):
            if method == "3D Cartesian":
                segment = vectors[start:stop]
                valid = np.isfinite(segment).all(axis=1)
                valid_counts[window_index, spot_index] = np.count_nonzero(valid)
                if not valid.any():
                    continue
                mean = np.mean(segment[valid], axis=0)
                length = float(np.linalg.norm(mean))
                resultant[window_index, spot_index] = length
                if length <= 1e-8:
                    continue
                direction = mean / length
            else:
                valid = np.isfinite(xy[start:stop, spot_index]).all(axis=1)
                valid_counts[window_index, spot_index] = np.count_nonzero(valid)
                if not valid.any():
                    continue
                if method == "Anisotropy X/Y":
                    mean_xy = np.mean(xy[start:stop, spot_index][valid], axis=0)
                else:
                    mean_xy = anisotropy(np.mean(values[start:stop, spot_index][valid], axis=0))
                direction = unit_vectors(mean_xy[None, :], theta_lut)[0]
            if not np.isfinite(direction).all():
                continue
            phi = np.mod(np.arctan2(direction[1], direction[0]), np.pi)
            transverse = np.hypot(direction[0], direction[1])
            directions[window_index, spot_index] = (transverse*np.cos(phi), transverse*np.sin(phi), direction[2])
    projected = directions[..., :2] / (1 + directions[..., 2:3])
    return {"directions": directions, "projected": projected, "valid_counts": valid_counts,
            "resultant_length": resultant, "window_start": starts, "window_stop": stops}


def projected_density(projected, bins=64):
    points = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    counts, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=bins, range=((-1, 1), (0, 1)))
    area = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    density = counts / (len(points) * area) if len(points) else counts
    return density, x_edges, y_edges