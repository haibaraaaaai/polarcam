"""APD photometry and optical models used by Unit_sphere_viewer.ipynb.

Correction order is inverse detector gains, then the supplied inverse optical
matrix. No offset, background subtraction, recentering, or stretching is applied.
Block channel means precede anisotropy ratios. Polar angles are measured from the
optical axis and folded to [0, 90] degrees; they are not hinge opening angles.
Neither state counting nor a unique 3D orientation trajectory is inferred here.
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Fourkas coefficients (published convention)
# ----------------------------------------------------------------------------

def fourkas_ABC(NA: float = 1.3, n: float = 1.33,
                NA_in: float | None = None) -> tuple[float, float, float]:
    """A, B, C in the published convention I_k = Itot[A + B sin^2t + C sin^2t cos2(fk-f)].

    NA_in=None  -> full pupil.  NA_in=float -> annular pupil (BFP hole), e.g. 0.39.
    """
    inner_na = 0.0 if NA_in is None else NA_in
    if not np.all(np.isfinite([NA, n, inner_na])) or not 0 <= inner_na < NA <= n:
        raise ValueError("Require finite 0 <= NA_in < NA <= n")
    a_out = np.arcsin(NA / n)
    if NA_in is None:
        ca = np.cos(a_out)
        A = 1/6 - ca/4 + ca**3/12
        B = ca/8 - ca**3/8
        C = 7/48 - ca/16 - ca**2/16 - ca**3/48
        return A, B, C
    # Annular pupil = full cone MINUS inner cone (published A, B, C are
    # antiderivative differences from 0 to alpha, so additivity is exact).
    # Verified: NA=1.3, n=1.33, NA_in=0.39 -> A=0.114177, B=0.014946,
    # C=0.118900, r_max=0.9208 (matches the paper content map r_sat=0.921).
    a_in = np.arcsin(NA_in / n)
    ca_o, ca_i = np.cos(a_out), np.cos(a_in)
    A = (1/6 - ca_o/4 + ca_o**3/12) - (1/6 - ca_i/4 + ca_i**3/12)
    B = (ca_o/8 - ca_o**3/8) - (ca_i/8 - ca_i**3/8)
    C = (7/48 - ca_o/16 - ca_o**2/16 - ca_o**3/48) - (7/48 - ca_i/16 - ca_i**2/16 - ca_i**3/48)
    return A, B, C


def fourkas_ABC_fresnel(NA=1.3, n_water=1.33, n_oil=1.515, NA_in=0.38,
                        quadrature_order=128):
    """A, B, C for an ideal circular annulus and one lossless water/oil interface.

    The Cartesian pupil field weight is sqrt(cos(alpha_oil))/cos(alpha_water).
    Azimuth is integrated analytically and pupil radius q = n*sin(alpha) by
    Gauss-Legendre quadrature with area element q*dq. A common intensity scale
    is omitted; equal media recover fourkas_ABC in its original normalization.
    This does not model an irregular hole, multilayers, or near-field emission.
    """
    parameters = [NA, NA_in, n_water, n_oil]
    if not np.all(np.isfinite(parameters)) or not 0 <= NA_in < NA <= min(n_water, n_oil):
        raise ValueError("Require finite 0 <= NA_in < NA <= min(n_water, n_oil)")
    if (isinstance(quadrature_order, bool) or int(quadrature_order) != quadrature_order
            or quadrature_order < 8):
        raise ValueError("quadrature_order must be an integer >= 8")
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    pupil_radius = NA_in + (nodes + 1) * (NA - NA_in) / 2
    radial_weights = weights * (NA - NA_in) / 2
    sin_water = pupil_radius / n_water
    cos_water = np.sqrt(1 - sin_water**2)
    cos_oil = np.sqrt(1 - (pupil_radius / n_oil)**2)
    transmission_s = 2 * n_water * cos_water / (n_water * cos_water + n_oil * cos_oil)
    transmission_p = 2 * n_water * cos_water / (n_oil * cos_water + n_water * cos_oil)
    pupil_field_weight = np.sqrt(cos_oil) / cos_water
    measure = radial_weights * pupil_radius * pupil_field_weight**2 / n_water**2
    radial_field = transmission_p * cos_water
    coeff_a = np.sum(measure * transmission_p**2 * sin_water**2) / 4
    coeff_a_plus_b = np.sum(measure * (radial_field**2 + transmission_s**2)) / 8
    coeff_c = np.sum(measure * (radial_field + transmission_s)**2) / 16
    return float(coeff_a), float(coeff_a_plus_b - coeff_a), float(coeff_c)


def r_max_of(A, B, C):
    return C / (A + B)


def load_tdms(path, channel_order=("90", "45", "135", "0"), fs=None):
    """Load Hugh's 4-channel APD TDMS recording.

    Native channel order in Hugh's files is [90, 45, 135, 0] (PyQtRod
    get_pol_ind convention). Pass fs to override; default from the
    wf_increment property (4 us -> 250 kHz).
    """
    from nptdms import TdmsFile
    tf = TdmsFile.read(path)
    group = tf.groups()[0]
    chans = group.channels()
    if len(chans) != 4:
        raise ValueError(f"expected 4 channels, got {len(chans)}")
    raw = [ch[:] for ch in chans]
    if fs is None:
        inc = chans[0].properties.get("wf_increment", 4e-6)
        fs = 1.0 / float(inc)
    out = {}
    for name, sig in zip(channel_order, raw):
        out[f"c{name}"] = np.asarray(sig, dtype=np.float64)
    n = min(len(v) for v in out.values())
    for k in list(out):
        out[k] = out[k][:n]
    t = np.arange(n) / fs
    return {"c0": out["c0"], "c45": out["c45"], "c90": out["c90"], "c135": out["c135"],
            "t": t, "fps": fs, "meta": {}, "source": "tdms", "n_frames": int(n)}


# ----------------------------------------------------------------------------
# Corrections
# ----------------------------------------------------------------------------

def fit_gains_tmatrix(rec, T=None):
    """Hugh's order: fit per-channel gains on RAW channels with the T-matrix
    inside the objective (pair-sum equalisation AFTER T @ diag(a) raw), c0 gain
    fixed at 1. Linear least squares. Returns gains dict."""
    if T is None:
        T = apd_tmatrix()
    c0, c90, c45, c135 = rec["c0"], rec["c90"], rec["c45"], rec["c135"]
    stack = np.stack([c0, c90, c45, c135], axis=1)          # (N,4) logical [0,90,45,135]
    finite_rows = np.all(np.isfinite(stack), axis=1)
    if not np.all(finite_rows):
        stack = stack[finite_rows]
    if len(stack) < 3:
        raise ValueError("At least three finite samples are required to fit channel gains")
    # corrected = T @ (a * raw).  Pair-sum objective: (corr0 + corr90 - corr45 - corr135)^2
    # = (w @ T @ diag(a) @ raw_i)^2 summed, with w = [1,1,-1,-1].
    # Linear in a: basis vectors v_j = w @ T @ (e_j * stack[:, j]).
    WT = np.array([1.0, 1.0, -1.0, -1.0]) @ T               # (4,)
    cols = WT * stack                                        # (N,4), column j = WT_j * raw_j
    a = np.ones(4)
    # solve min || sum_j a_j cols[:,j] ||^2 with a_0 = 1 (reference)
    M = cols[:, 1:]
    y = -cols[:, 0]
    u, *_ = np.linalg.lstsq(M, y, rcond=None)
    a = np.concatenate([[1.0], u])
    # stack order [0, 90, 45, 135] -> gains dict
    return {"0": float(a[0]), "90": float(a[1]), "45": float(a[2]), "135": float(a[3])}


def block_channel_means(rec, n_block, *, return_flags=False):
    """Block-average the corrected CHANNELS then form anisotropies (ratio of
    means - the physically correct estimator, consistent with an integrating
    detector and with the RB-notes background model).

    Returns (ax, ay, t, fps_eff), optionally followed by per-block validity flags.
    No channel offsets, clipping, or sparse sampling are applied.
    """
    if isinstance(n_block, bool) or int(n_block) != n_block or n_block < 1:
        raise ValueError("n_block must be a positive integer")
    n_block = int(n_block)
    def bm(x, n):
        m = len(x) // n * n
        return np.asarray(x, dtype=float)[:m].reshape(-1, n).mean(axis=1)
    c0 = bm(rec["c0"], n_block); c90 = bm(rec["c90"], n_block)
    c45 = bm(rec["c45"], n_block); c135 = bm(rec["c135"], n_block)
    den_x = c0 + c90; den_y = c45 + c135
    channel_values = np.stack((c0, c90, c45, c135))
    nonfinite = ~np.all(np.isfinite(channel_values), axis=0)
    nonpositive_pair_sum = (den_x <= 0) | (den_y <= 0)
    negative_channel = np.any(channel_values < 0, axis=0)
    usable_ratio = ~nonfinite & ~nonpositive_pair_sum
    ax = np.divide(c0 - c90, den_x, out=np.full_like(den_x, np.nan), where=usable_ratio)
    ay = np.divide(c45 - c135, den_y, out=np.full_like(den_y, np.nan), where=usable_ratio)
    t = bm(np.asarray(rec["t"]), n_block)
    result = (ax, ay, t, rec["fps"] / n_block)
    if return_flags:
        flags = {"valid": usable_ratio & ~negative_channel,
                 "nonfinite": nonfinite,
                 "nonpositive_pair_sum": nonpositive_pair_sum,
                 "negative_channel": negative_channel}
        return (*result, flags)
    return result


def apply_gains(rec, gains):
    r = dict(rec)
    for k in ("c0", "c45", "c90", "c135"):
        r[k] = rec[k] * gains[k[1:]]
    return r


def apd_tmatrix():
    """4x4 correction matrix in logical order [c0, c90, c45, c135]:
    corrected = T @ I_measured.

    DEFAULT: the inverse matrix Hugh Bowman sent by email on 2026-09-15
    ("I_actual = sent matrix * I_measured", basis I = (0,90,45,135)) - this is
    the supplied setup calibration, distinct from the BFM/PyQtRod matrix."""
    return np.array([
        [ 2.56410256,  0.0,         0.0,         0.0        ],
        [ 0.0,         2.65305220,  0.0,         0.0        ],
        [-0.04047189,  0.04187587,  1.87478350, -0.00045528],
        [-0.04045247,  0.04185578, -0.00046721, 1.82373830],
    ])


def apply_tmatrix(rec, T=None):
    """Apply the APD T-matrix. Channel order logical [0, 90, 45, 135]."""
    if T is None:
        T = apd_tmatrix()
    stack = np.stack([rec["c0"], rec["c90"], rec["c45"], rec["c135"]], axis=1)
    corr = stack @ T.T
    return {"c0": corr[:, 0], "c90": corr[:, 1], "c45": corr[:, 2], "c135": corr[:, 3],
            "t": rec["t"], "fps": rec["fps"], "meta": rec.get("meta", {}),
            "source": rec.get("source"), "n_frames": rec["n_frames"]}


def correct_apd_channels(rec, gains=None, T=None):
    """Return (corrected_recording, inverse_gains): gain first, inverse matrix next.

    No clipping or background manipulation is performed. Future background
    subtraction in corrected-channel units belongs after this operation.
    """
    if T is None:
        T = apd_tmatrix()
    if gains is None:
        gains = fit_gains_tmatrix(rec, T)
    return apply_tmatrix(apply_gains(rec, gains), T), gains


def theta_from_r(r, A, B, C, clip_r=False, warn_over=False, *, return_flags=False):
    """Acute polar angle in degrees; invalid radii are NaN unless clipping is explicit.

    Optional flags describe the original radii even when clip_r=True.
    """
    r = np.asarray(r, dtype=np.float64)
    if not np.all(np.isfinite([A, B, C])) or A <= 0 or A + B <= 0 or C <= 0:
        raise ValueError("Fourkas coefficients must give positive finite intensities")
    rmax = r_max_of(A, B, C)
    tolerance = 16 * np.finfo(float).eps * max(1.0, abs(rmax))
    nonfinite = ~np.isfinite(r)
    negative_radius = r < 0
    over_range = r > rmax + tolerance
    valid = ~nonfinite & ~negative_radius & ~over_range
    if warn_over and np.any(over_range):
        print(f"  [theta_from_r] {np.count_nonzero(over_range)}/{r.size} radii "
              f"exceed model r_max={rmax:.4f}; check calibration, background, and noise")
    usable = ~nonfinite if clip_r else valid
    radius_used = np.where(usable, np.clip(r, 0.0, rmax), np.nan)
    sine_squared = np.clip(A * radius_used / (C - B * radius_used), 0.0, 1.0)
    theta = np.degrees(np.arcsin(np.sqrt(sine_squared)))
    if return_flags:
        return theta, {"valid": valid, "nonfinite": nonfinite,
                       "negative_radius": negative_radius, "over_range": over_range}
    return theta


def extract_radial_excursions(ax, ay, times, circle_center, circle_radius, *,
                             core_center=None, core_fraction=0.12,
                             outer_fraction=0.8, min_core_samples=3,
                             min_outer_samples=3, valid=None, max_gap_s=None):
    """Outward first passages from a central disk to a fitted-circle radius fraction.

    Both distances are in measured anisotropy coordinates, not rod angles.
    A confirmed core visit arms one excursion. It starts at the last core sample
    and ends at the first sample of a confirmed outer run (end_idx inclusive).
    Returning to the core aborts that attempt; another confirmed core visit is
    required after each completed passage. Invalid samples and time gaps censor
    attempts instead of joining unrelated paths. No smoothing is performed.
    """
    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    times = np.asarray(times, dtype=float)
    if ax.ndim != 1 or ax.shape != ay.shape or ax.shape != times.shape:
        raise ValueError("ax, ay, and times must be matching one-dimensional arrays")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite and strictly increasing")
    circle_center = np.asarray(circle_center, dtype=float)
    core_center = circle_center.copy() if core_center is None else np.asarray(core_center, dtype=float)
    if (circle_center.shape != (2,) or core_center.shape != (2,)
            or not np.all(np.isfinite(circle_center)) or not np.all(np.isfinite(core_center))):
        raise ValueError("circle_center and core_center must be finite coordinate pairs")
    if (not np.all(np.isfinite([circle_radius, core_fraction, outer_fraction]))
            or circle_radius <= 0 or not 0 < core_fraction < outer_fraction <= 1):
        raise ValueError("Require positive circle_radius and 0 < core_fraction < outer_fraction <= 1")
    if np.linalg.norm(core_center - circle_center) / circle_radius + core_fraction >= outer_fraction:
        raise ValueError("The central disk must lie strictly inside the outer threshold")
    for sample_count in (min_core_samples, min_outer_samples):
        if isinstance(sample_count, bool) or int(sample_count) != sample_count or sample_count < 1:
            raise ValueError("Residence checks require positive integer sample counts")
    usable = np.isfinite(ax) & np.isfinite(ay)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != ax.shape:
            raise ValueError("valid must match the sample arrays")
        usable &= valid
    if max_gap_s is None:
        max_gap_s = 1.5 * np.median(np.diff(times)) if len(times) > 1 else np.inf
    if np.isnan(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    gap_before = np.concatenate(([False], np.diff(times) > max_gap_s)) if len(times) else np.array([], bool)
    core_mask = np.hypot(ax - core_center[0], ay - core_center[1]) <= core_fraction * circle_radius
    radius_fraction = np.hypot(ax - circle_center[0], ay - circle_center[1]) / circle_radius
    paths = []
    aborted = censored = attempts = 0
    core_run = outer_run = 0
    armed = False
    start_idx = last_core_idx = first_outer_idx = None
    for sample_index in range(len(times)):
        if gap_before[sample_index] or not usable[sample_index]:
            censored += int(start_idx is not None)
            armed = False
            start_idx = last_core_idx = first_outer_idx = None
            core_run = outer_run = 0
            if not usable[sample_index]:
                continue
        if core_mask[sample_index]:
            if start_idx is not None:
                aborted += 1
                start_idx = first_outer_idx = None
                outer_run = 0
                armed = False
            core_run += 1
            last_core_idx = sample_index
            if core_run >= min_core_samples:
                armed = True
            continue
        core_run = 0
        if not armed:
            continue
        if start_idx is None:
            start_idx = last_core_idx
            attempts += 1
        if radius_fraction[sample_index] >= outer_fraction:
            if outer_run == 0:
                first_outer_idx = sample_index
            outer_run += 1
            if outer_run >= min_outer_samples:
                paths.append({
                    "start_idx": int(start_idx),
                    "end_idx": int(first_outer_idx),
                    "confirm_idx": sample_index,
                    "duration_s": float(times[first_outer_idx] - times[start_idx]),
                    "n_samples": int(first_outer_idx - start_idx + 1),
                })
                start_idx = first_outer_idx = None
                outer_run = 0
                armed = False
        else:
            outer_run = 0
            first_outer_idx = None
    censored += int(start_idx is not None)
    return {"paths": paths, "attempts": attempts, "aborted": aborted,
            "censored": censored, "invalid_samples": int((~usable).sum()),
            "time_gaps": int(gap_before.sum()), "core_center": core_center.copy(),
            "core_radius": float(core_fraction * circle_radius),
            "outer_radius": float(outer_fraction * circle_radius)}


def radial_crossing_profiles(ax, ay, times, excursions, circle_center, circle_radius, fractions):
    """One interpolated outward crossing per passage and radial gate.

    Bearings are anisotropy-plane angles around circle_center, NOT rod phi.
    Interior gates use first outward crossings; the target gate uses the
    confirmed endpoint crossing. Interpolation is local to an observed adjacent
    sample pair and does not assert that an undersampled path was resolved.
    """
    points = np.column_stack((ax, ay)).astype(float)
    times = np.asarray(times, dtype=float)
    circle_center = np.asarray(circle_center, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if (fractions.ndim != 1 or not np.all(np.isfinite(fractions))
            or np.any(fractions <= 0) or np.any(np.diff(fractions) <= 0)):
        raise ValueError("fractions must be finite, positive, and strictly increasing")
    if not np.isfinite(circle_radius) or circle_radius <= 0:
        raise ValueError("circle_radius must be positive and finite")
    shape = (len(excursions["paths"]), len(fractions))
    crossings = np.full((*shape, 2), np.nan)
    crossing_times = np.full(shape, np.nan)
    for event_index, event in enumerate(excursions["paths"]):
        start_idx, end_idx = event["start_idx"], event["end_idx"]
        segment = points[start_idx:end_idx + 1]
        radius = np.linalg.norm(segment - circle_center, axis=1)
        for gate_index, fraction in enumerate(fractions):
            target = fraction * circle_radius
            brackets = np.flatnonzero((radius[:-1] < target) & (radius[1:] >= target))
            if len(brackets) == 0:
                continue
            if np.isclose(target, excursions["outer_radius"], rtol=1e-12, atol=0):
                local_index = len(segment) - 2
                if local_index not in brackets:
                    continue
            else:
                local_index = int(brackets[0])
            offset = segment[local_index] - circle_center
            increment = segment[local_index + 1] - segment[local_index]
            quadratic = float(increment @ increment)
            linear = float(2 * offset @ increment)
            constant = float(offset @ offset - target**2)
            weight = (-linear + np.sqrt(max(0., linear**2 - 4 * quadratic * constant))) / (2 * quadratic)
            weight = float(np.clip(weight, 0, 1))
            crossings[event_index, gate_index] = segment[local_index] + weight * increment
            absolute_index = start_idx + local_index
            crossing_times[event_index, gate_index] = (
                times[absolute_index] + weight * (times[absolute_index + 1] - times[absolute_index]))
    offsets = crossings - circle_center
    bearings = np.degrees(np.arctan2(offsets[..., 1], offsets[..., 0])) % 360
    return {"points": crossings, "times": crossing_times, "bearing_deg": bearings,
            "fractions": fractions.copy()}


def fit_origin_separator(ax, ay, *, radial_band=(0.5, 0.75),
                         angle_bounds_deg=(115.0, 170.0), smoothing_deg=3.0,
                         angular_step_deg=0.5, valid=None):
    """Find a low-count upper-left ray; its complete separator passes through zero.

    Fit the smoothed angular occupancy in an origin-centered annulus, restricted
    to the supplied sector between the intended B/C populations. A boundary or
    monotonic minimum is flagged, not presented as evidence for a density valley.
    Smoothing is of the angular histogram only, never of time-ordered samples.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    ax, ay = np.asarray(ax, dtype=float), np.asarray(ay, dtype=float)
    if ax.ndim != 1 or ax.shape != ay.shape:
        raise ValueError("ax and ay must be matching one-dimensional arrays")
    lower_r, upper_r = radial_band
    lower_angle, upper_angle = angle_bounds_deg
    if (not np.all(np.isfinite([lower_r, upper_r, lower_angle, upper_angle,
                               smoothing_deg, angular_step_deg]))
            or not 0 <= lower_r < upper_r or not 0 < lower_angle < upper_angle < 180
            or smoothing_deg <= 0 or not 0 < angular_step_deg <= upper_angle - lower_angle):
        raise ValueError("Invalid radial band, angular sector, or smoothing resolution")
    usable = np.isfinite(ax) & np.isfinite(ay)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != ax.shape:
            raise ValueError("valid must match the samples")
        usable &= valid
    radius = np.hypot(ax, ay)
    selected = usable & (radius >= lower_r) & (radius <= upper_r)
    if not np.any(selected):
        raise ValueError("No valid samples in the separator annulus")
    angles = np.degrees(np.arctan2(ay[selected], ax[selected])) % 360
    bin_count = max(4, int(np.ceil(360 / angular_step_deg)))
    edges = np.linspace(0., 360., bin_count + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    counts, _ = np.histogram(angles, bins=edges)
    density = gaussian_filter1d(counts.astype(float), smoothing_deg / (360 / bin_count), mode="wrap")
    candidates = np.flatnonzero((centers >= lower_angle) & (centers <= upper_angle))
    if not np.any(counts[candidates]):
        raise ValueError("No samples in the requested separator sector")
    local_minima, _ = find_peaks(-density)
    interior = np.intersect1d(candidates, local_minima)
    chosen = int(interior[np.argmin(density[interior])]) if len(interior) else int(candidates[np.argmin(density[candidates])])
    angle_deg = float(centers[chosen])
    angle = np.radians(angle_deg)
    normal = np.array([-np.sin(angle), np.cos(angle)])
    return {"angle_deg": angle_deg, "normal": normal,
            "angles_deg": centers, "counts": counts, "smoothed_counts": density,
            "radial_band": tuple(radial_band), "angle_bounds_deg": tuple(angle_bounds_deg),
            "interior_minimum": bool(len(interior)), "annulus_samples": int(selected.sum()),
            "smoothing_deg": float(smoothing_deg)}


def fit_origin_core_radius(ax, ay, *, quantile=0.95, right_radius_limit=0.30, valid=None):
    """Smallest origin-centered disk covering a chosen fraction of near-origin right-side samples.

    The right half-plane isolates A for this empirical radius criterion. This is
    not an optical correction or a state-boundary inference; the quantile and
    selection limit must be reported. No points or the resulting disk are clipped.
    """
    ax, ay = np.asarray(ax, dtype=float), np.asarray(ay, dtype=float)
    if ax.ndim != 1 or ax.shape != ay.shape:
        raise ValueError("ax and ay must be matching one-dimensional arrays")
    if not 0 < quantile < 1 or not np.isfinite(right_radius_limit) or right_radius_limit <= 0:
        raise ValueError("Require 0 < quantile < 1 and a positive radius limit")
    usable = np.isfinite(ax) & np.isfinite(ay)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != ax.shape:
            raise ValueError("valid must match the samples")
        usable &= valid
    radius = np.hypot(ax, ay)
    selected = usable & (ax >= 0) & (radius <= right_radius_limit)
    if not np.any(selected):
        raise ValueError("No valid right-side samples for the central disk")
    fitted_radius = float(np.quantile(radius[selected], quantile))
    if fitted_radius <= 0:
        raise ValueError("The fitted central radius is zero")
    return {"radius": fitted_radius, "quantile": quantile,
            "right_radius_limit": right_radius_limit, "selected_samples": int(selected.sum())}


def ellipse_region_clearances(region, origin_radius, separator_normal, side):
    """Exact filled-ellipse clearance from a line and an origin-centered disk."""
    from scipy.optimize import brentq

    center = np.asarray(region["center"], dtype=float)
    axes = np.asarray(region["semiaxes"], dtype=float)
    angle = np.radians(region.get("angle_deg", 0.))
    normal = np.asarray(separator_normal, dtype=float)
    if (center.shape != (2,) or axes.shape != (2,) or normal.shape != (2,)
            or not np.all(np.isfinite(np.r_[center, axes, normal, angle, origin_radius]))
            or np.any(axes <= 0) or np.linalg.norm(normal) == 0 or origin_radius < 0
            or side not in (-1, 1)):
        raise ValueError("Invalid ellipse, disk, separator, or side")
    normal = normal / np.linalg.norm(normal)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    projected_normal = rotation.T @ normal
    support = np.linalg.norm(axes * projected_normal)
    line_gap = float(side * (center @ normal) - support)
    local_center = rotation.T @ center
    if np.sum((local_center / axes)**2) <= 1:
        distance_to_origin = 0.
    else:
        squared_axes = axes**2
        multiplier = brentq(
            lambda value: np.sum((axes * local_center / (squared_axes + value))**2) - 1,
            0., float(np.linalg.norm(axes * local_center)), xtol=1e-13,
        )
        distance_to_origin = float(np.linalg.norm(local_center * multiplier / (squared_axes + multiplier)))
    return {"line_gap": line_gap, "disk_gap": distance_to_origin - origin_radius}


def fit_equal_area_ellipses(ax, ay, a_radius, separator_normal, *, valid=None,
                           histogram_step=0.01, clearance=0.005, max_aspect=4.0,
                           maxiter=160, popsize=12, seed=0):
    """Maximize B/C occupancy on a spatial histogram under exact geometric constraints.

    A is a full origin-centered disk. Each ellipse has semiaxis product R_A**2,
    is at least clearance from A and the separator, and stays on its own side.
    B is the positive-normal side, C negative. The bounded differential-evolution
    search is an approximate occupancy maximum, not a certified global optimum.
    Histogram binning accelerates fitting only; classify the original samples
    separately for occupancy, residence, and transition measurements.
    """
    from scipy.optimize import differential_evolution, NonlinearConstraint

    ax, ay = np.asarray(ax, dtype=float), np.asarray(ay, dtype=float)
    normal = np.asarray(separator_normal, dtype=float)
    if ax.ndim != 1 or ax.shape != ay.shape:
        raise ValueError("ax and ay must be matching one-dimensional arrays")
    if (normal.shape != (2,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) == 0
            or not np.all(np.isfinite([a_radius, histogram_step, clearance, max_aspect]))
            or a_radius <= 0 or histogram_step <= 0 or clearance <= 0 or max_aspect < 1):
        raise ValueError("Invalid radius, histogram spacing, clearance, aspect ratio, or normal")
    normal = normal / np.linalg.norm(normal)
    usable = np.isfinite(ax) & np.isfinite(ay)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != ax.shape:
            raise ValueError("valid must match the samples")
        usable &= valid
    if not np.any(usable):
        raise ValueError("No valid samples for ellipse fitting")
    selected_x, selected_y = ax[usable], ay[usable]
    edges = []
    for values in (selected_x, selected_y):
        lower = int(np.floor(values.min() / histogram_step))
        upper = max(lower + 1, int(np.ceil(values.max() / histogram_step)))
        edges.append(np.arange(lower, upper + 1) * histogram_step)
    histogram, x_edges, y_edges = np.histogram2d(selected_x, selected_y, bins=edges)
    x_indices, y_indices = np.nonzero(histogram)
    points = np.column_stack(((x_edges[x_indices] + x_edges[x_indices + 1]) / 2,
                              (y_edges[y_indices] + y_edges[y_indices + 1]) / 2))
    weights = histogram[x_indices, y_indices]
    regions = {"A": {"center": (0., 0.), "semiaxes": (a_radius, a_radius), "angle_deg": 0.}}
    diagnostics = {}
    padding = a_radius * np.sqrt(max_aspect)

    def unpack(parameters):
        center_x, center_y, log_aspect, angle_deg = parameters
        major = a_radius * np.exp(log_aspect / 2)
        minor = a_radius * np.exp(-log_aspect / 2)
        return {"center": (center_x, center_y), "semiaxes": (major, minor), "angle_deg": angle_deg}

    for side_index, (name, side) in enumerate((("B", 1), ("C", -1))):
        eligible = (side * (points @ normal) > clearance) & (np.linalg.norm(points, axis=1) > a_radius + clearance)
        fit_points, fit_weights = points[eligible], weights[eligible]
        if not len(fit_points):
            raise ValueError(f"No eligible histogram bins for region {name}")
        normalization = float(fit_weights.sum())

        def objective(parameters):
            center_x, center_y, log_aspect, angle_deg = parameters
            angle = np.radians(angle_deg)
            offset_x = fit_points[:, 0] - center_x
            offset_y = fit_points[:, 1] - center_y
            major = a_radius * np.exp(log_aspect / 2)
            minor = a_radius * np.exp(-log_aspect / 2)
            along = (np.cos(angle) * offset_x + np.sin(angle) * offset_y) / major
            across = (-np.sin(angle) * offset_x + np.cos(angle) * offset_y) / minor
            return -float(fit_weights[along**2 + across**2 <= 1].sum()) / normalization

        def margins(parameters):
            gaps = ellipse_region_clearances(unpack(parameters), a_radius, normal, side)
            return np.array([gaps["line_gap"], gaps["disk_gap"]])

        bounds = [(float(fit_points[:, axis].min() - padding), float(fit_points[:, axis].max() + padding))
                  for axis in range(2)] + [(0., float(np.log(max_aspect))), (0., 180.)]
        result = differential_evolution(
            objective, bounds, constraints=(NonlinearConstraint(margins, clearance, np.inf),),
            seed=int(seed) + side_index, maxiter=int(maxiter), popsize=int(popsize),
            polish=False, tol=1e-4, atol=1e-7,
        )
        ellipse = unpack(result.x)
        gaps = ellipse_region_clearances(ellipse, a_radius, normal, side)
        if not np.isfinite(result.fun) or min(gaps.values()) < clearance - 1e-10:
            raise RuntimeError(f"No feasible equal-area ellipse for {name}; adjust fit limits")
        regions[name] = ellipse
        diagnostics[name] = {**gaps, "histogram_count": -float(result.fun) * normalization,
                             "converged": bool(result.success), "iterations": int(result.nit),
                             "message": str(result.message), "aspect_ratio": float(np.exp(result.x[2]))}
    return {"regions": regions, "diagnostics": diagnostics,
            "area": float(np.pi * a_radius**2), "clearance": clearance,
            "histogram_step": histogram_step, "max_aspect": max_aspect}


def classify_region_cores(ax, ay, regions, *, valid=None, scale=1.0):
    """Label explicit elliptical cores; -1 is unassigned and -2 is invalid.

    Region insertion order defines nonnegative codes. Each region specifies
    center, semiaxes, and optional angle_deg (counterclockwise from ax).
    Scale changes semiaxes only. No nearest-center assignment or smoothing is
    applied. Samples inside overlapping cores raise instead of depending on
    dictionary order. Labels describe measured regions, not physical states.
    """
    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    if ax.ndim != 1 or ay.shape != ax.shape:
        raise ValueError("ax and ay must be matching one-dimensional arrays")
    if not regions or len(regions) > np.iinfo(np.int16).max:
        raise ValueError("At least one region and at most 32767 regions are required")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    usable = np.isfinite(ax) & np.isfinite(ay)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != ax.shape:
            raise ValueError("valid must match the sample arrays")
        usable &= valid
    labels = np.full(ax.shape, -1, dtype=np.int16)
    labels[~usable] = -2
    for region_code, (region_name, specification) in enumerate(regions.items()):
        center = np.asarray(specification["center"], dtype=float)
        semiaxes = np.asarray(specification["semiaxes"], dtype=float)
        angle_deg = float(specification.get("angle_deg", 0.0))
        if (center.shape != (2,) or semiaxes.shape != (2,)
                or not np.all(np.isfinite(center)) or not np.all(np.isfinite(semiaxes))
                or np.any(semiaxes <= 0) or not np.isfinite(angle_deg)):
            raise ValueError(f"Invalid ellipse specification for region {region_name}")
        cosine, sine = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
        offset_x, offset_y = ax - center[0], ay - center[1]
        along = (cosine * offset_x + sine * offset_y) / (scale * semiaxes[0])
        across = (-sine * offset_x + cosine * offset_y) / (scale * semiaxes[1])
        inside = usable & (along**2 + across**2 <= 1.0)
        overlap = inside & (labels >= 0)
        if np.any(overlap):
            raise ValueError(f"Region {region_name} overlaps another core at "
                             f"{np.count_nonzero(overlap)} samples; adjust the region boundaries")
        labels[inside] = region_code
    return {"labels": labels, "region_names": tuple(regions)}


def region_residence_statistics(labels, times, rate, region_names, *, min_visit_samples=1):
    """Measured core occupancy and uninterrupted core visits on a native time grid.

    Labels use -1 for unassigned and -2 for invalid, as classify_region_cores.
    Missing timestamps break visits and do not contribute observed time. Visits
    touching invalid data, time gaps, or recording edges are flagged as censored
    and excluded from complete-visit duration summaries. Minimum visit length
    selects runs retrospectively; occupancy and the input labels never change.
    No unassigned interval is bridged or assigned to the previous core.
    """
    labels = np.asarray(labels)
    times = np.asarray(times, dtype=float)
    region_names = tuple(region_names)
    if labels.ndim != 1 or labels.shape != times.shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be integer codes matching one-dimensional times")
    if (not region_names or len(set(region_names)) != len(region_names)
            or any(name in ("unassigned", "invalid") for name in region_names)):
        raise ValueError("region_names must be unique and exclude reserved names")
    if np.any(labels < -2) or np.any(labels >= len(region_names)):
        raise ValueError("Unknown region label")
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    if (isinstance(min_visit_samples, bool) or int(min_visit_samples) != min_visit_samples
            or min_visit_samples < 1):
        raise ValueError("min_visit_samples must be a positive integer")
    increments = np.diff(times)
    if not np.all(np.isfinite(times)) or np.any(increments <= 0):
        raise ValueError("times must be finite and strictly increasing")
    interval = 1.0 / rate
    gaps = increments > 1.5 * interval
    if np.any(~gaps & ~np.isclose(increments, interval, rtol=1e-6, atol=1e-12)):
        raise ValueError("Samples must be on the stated uniform time grid apart from gaps")
    sample_count = len(labels)
    gap_before = np.concatenate(([False], gaps)) if sample_count else np.array([], dtype=bool)
    if sample_count:
        starts = np.concatenate(([0], np.flatnonzero((labels[1:] != labels[:-1]) | gaps) + 1))
        stops = np.concatenate((starts[1:], [sample_count]))
        codes = labels[starts]
        left_censored = (starts == 0) | gap_before[starts] | (labels[np.maximum(starts - 1, 0)] == -2)
        next_indices = np.minimum(stops, sample_count - 1)
        right_censored = (stops == sample_count) | gap_before[next_indices] | (labels[next_indices] == -2)
    else:
        starts = stops = np.array([], dtype=int)
        codes = np.array([], dtype=int)
        left_censored = right_censored = np.array([], dtype=bool)
    lengths = stops - starts
    durations = lengths / rate
    retained = (codes >= 0) & (lengths >= min_visit_samples)
    censored = left_censored | right_censored
    valid_count = int(np.count_nonzero(labels != -2))
    sample_counts = np.bincount(labels.astype(np.int64) + 2, minlength=len(region_names) + 2)
    summary = {}
    categories = list(enumerate(region_names)) + [(-1, "unassigned"), (-2, "invalid")]
    for code, name in categories:
        region_runs = codes == code
        retained_runs = region_runs & retained
        complete_durations = durations[retained_runs & ~censored]
        count = int(sample_counts[code + 2])
        summary[name] = {
            "samples": count,
            "time_s": count / rate,
            "fraction_recorded": count / sample_count if sample_count else np.nan,
            "fraction_valid": count / valid_count if valid_count and code != -2 else np.nan,
            "n_runs": int(np.count_nonzero(region_runs)),
            "n_censored_runs": int(np.count_nonzero(region_runs & censored)),
            "n_short_runs": int(np.count_nonzero(region_runs & (lengths < min_visit_samples))),
            "n_retained_complete_runs": len(complete_durations),
            "retained_time_s": float(np.sum(durations[retained_runs])),
            "complete_durations_s": complete_durations,
            "median_complete_s": float(np.median(complete_durations)) if len(complete_durations) else np.nan,
        }
    return {
        "summary": summary,
        "runs": {"start_idx": starts, "stop_idx": stops, "region_code": codes,
                 "duration_s": durations, "left_censored": left_censored,
                 "right_censored": right_censored, "retained": retained},
        "observed_time_s": sample_count / rate,
        "valid_time_s": valid_count / rate,
        "span_s": float(times[-1] - times[0] + interval) if sample_count else 0.,
        "time_gaps": int(np.count_nonzero(gaps)),
        "minimum_visit_s": min_visit_samples / rate,
    }


def region_transition_counts(residence, region_names=("A", "B", "C")):
    """Count successive committed core visits and observed B/C passages via or bypassing A.

    Commitment is the residence result's minimum uninterrupted run length. An
    unassigned interval is allowed between visits but is never relabeled. Any
    invalid-data or timestamp-gap boundary resets the chain. For B/C passages,
    the source is the last committed B or C visit before the next committed
    opposite endpoint, retaining all intervening A contacts, even uncommitted
    ones. 'no_A_sample' is observational, not proof of a physical direct route.
    """
    region_names = tuple(region_names)
    if len(region_names) != 3 or set(region_names) != {"A", "B", "C"}:
        raise ValueError("Exactly A, B, and C region names are required")
    runs = residence["runs"]
    codes = np.asarray(runs["region_code"])
    starts, stops = runs["start_idx"], runs["stop_idx"]
    retained = runs["retained"]
    a_code, b_code, c_code = (region_names.index(name) for name in ("A", "B", "C"))
    counts = np.zeros((3, 3), dtype=int)
    committed_visit_counts = np.zeros(3, dtype=int)
    previous_code = None
    endpoint = None
    a_samples = unassigned_samples = 0
    max_a_samples = 0
    committed_a = False
    passages = []
    for run_index, code in enumerate(codes):
        if runs["left_censored"][run_index] or code == -2:
            previous_code = None
            endpoint = None
            a_samples = unassigned_samples = max_a_samples = 0
            committed_a = False
        if code == -2:
            continue
        length = int(stops[run_index] - starts[run_index])
        if retained[run_index]:
            committed_visit_counts[code] += 1
            if previous_code is not None and previous_code != code:
                counts[previous_code, code] += 1
            previous_code = int(code)
        if endpoint is not None:
            if code == a_code:
                a_samples += length
                max_a_samples = max(max_a_samples, length)
                committed_a |= bool(retained[run_index])
            elif code == -1:
                unassigned_samples += length
        if retained[run_index] and code in (b_code, c_code):
            if endpoint is not None and endpoint["code"] != code:
                category = "committed_A" if committed_a else ("brief_A" if a_samples else "no_A_sample")
                passages.append({
                    "from": region_names[endpoint["code"]], "to": region_names[code],
                    "start_idx": endpoint["stop_idx"] - 1,
                    "end_idx": int(starts[run_index]),
                    "source_run": endpoint["run"], "destination_run": run_index,
                    "category": category, "a_samples": a_samples,
                    "max_a_run_samples": max_a_samples,
                    "unassigned_samples": unassigned_samples,
                })
            endpoint = {"code": int(code), "stop_idx": int(stops[run_index]), "run": run_index}
            a_samples = unassigned_samples = max_a_samples = 0
            committed_a = False
        if runs["right_censored"][run_index]:
            previous_code = None
            endpoint = None
            a_samples = unassigned_samples = max_a_samples = 0
            committed_a = False
    categories = ("no_A_sample", "brief_A", "committed_A")
    bc_counts = {direction: dict.fromkeys(categories, 0) for direction in ("B->C", "C->B")}
    for passage in passages:
        bc_counts[f"{passage['from']}->{passage['to']}"][passage["category"]] += 1
    return {"counts": counts, "region_names": region_names,
            "committed_visit_counts": committed_visit_counts,
            "bc_counts": bc_counts, "bc_passages": passages,
            "minimum_visit_s": residence["minimum_visit_s"]}
