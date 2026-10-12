"""Annular dipole collection and direct angle inversion. See docs/angle_reconstruction.md.

Convention: I(psi) = scale * [A + B*sin(theta)^2
                             + C*sin(theta)^2*cos(2*(psi-phi))].
Angles passed to/returned by the numerical functions are radians.
"""

from functools import lru_cache

import numpy as np


MODEL_VERSION = "annular-fresnel-bfp-v1"
COEFFICIENT_CONVENTION = "I=scale*(A+B*sin(theta)^2+C*sin(theta)^2*cos(2*(psi-phi)))"
CHANNEL_ORDER = ("0", "90", "45", "135")


@lru_cache(maxsize=32)
def annular_coefficients(na_out=1.3, na_in=0.39, n_sample=1.33,
                         n_immersion=1.515, quadrature_order=128):
    """Integrate one ideal circular annulus, with Fresnel and BFP field mapping.

    q = n_sample*sin(alpha_sample) = n_immersion*sin(alpha_immersion).
    Field mapping is sqrt(cos(alpha_immersion))/cos(alpha_sample).
    The measure is q*dq/n_sample^2 times the squared field mapping.
    Glass and oil are represented by one effective, lossless index. No near-field
    substrate emission, multiple reflections, or measured asymmetric mask is used.
    Equal sample/immersion indices recover homogeneous Fourkas collection.
    """
    if (not np.all(np.isfinite([na_out, na_in, n_sample, n_immersion]))
            or not 0 <= na_in < na_out < min(n_sample, n_immersion)):
        raise ValueError("Require finite 0 <= NA_in < NA_out < both refractive indices.")
    if (isinstance(quadrature_order, bool) or int(quadrature_order) != quadrature_order
            or quadrature_order < 8):
        raise ValueError("Quadrature order must be an integer >= 8.")
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    q = na_in + (nodes + 1) * (na_out - na_in) / 2
    dq = weights * (na_out - na_in) / 2
    sin_s = q / n_sample
    cos_s = np.sqrt(1 - sin_s**2)
    cos_o = np.sqrt(1 - (q / n_immersion)**2)
    ts = 2*n_sample*cos_s / (n_sample*cos_s + n_immersion*cos_o)
    tp = 2*n_sample*cos_s / (n_immersion*cos_s + n_sample*cos_o)
    measure = q*dq * cos_o/cos_s**2 / n_sample**2
    p = tp*cos_s
    a = np.sum(measure * tp**2 * sin_s**2) / 4
    a_plus_b = np.sum(measure * (p**2 + ts**2)) / 8
    c = np.sum(measure * (p + ts)**2) / 16
    return float(a), float(a_plus_b-a), float(c)


def theta_model(n_sample=1.33, *, label="water Fresnel + BFP + hole",
                na_out=1.3, na_in=0.39, n_immersion=1.515):
    """JSON-safe optical parameters; no detector/path calibration is implied."""
    a, b, c = annular_coefficients(na_out, na_in, n_sample, n_immersion)
    return {
        "model_version": MODEL_VERSION, "coefficient_convention": COEFFICIENT_CONVENTION,
        "label": label, "n_medium": float(n_sample), "n_immersion": float(n_immersion),
        "na_out": float(na_out), "na_hole": float(na_in),
        "A": a, "B": b, "C": c, "r_max": c/(a+b),
        "mask": "centred circular annulus", "quadrature_order": 128,
        "interface": "single lossless sample to glass/oil effective interface",
        "field_mapping": "sqrt(cos(alpha_immersion))/cos(alpha_sample)",
        "channel_calibration": "not included in optical coefficients",
    }


def camera_theta_models():
    return {
        "water": theta_model(),
        "glycerol50": theta_model(1.398, label="50% glycerol Fresnel + BFP + hole"),
    }


def radius_from_theta(theta, a, b, c):
    """Forward anisotropy radius, for a fixed ideal dipole."""
    values = np.asarray(theta, dtype=np.float64)
    sin2 = np.sin(values)**2
    result = c*sin2/(a+b*sin2)
    return np.where(np.isfinite(values) & (values >= 0) & (values <= np.pi/2), result, np.nan)


def theta_from_radius(radius, a, b, c):
    """Analytic inverse in radians; invalid radii stay NaN, not clipped to 90 deg."""
    if not np.all(np.isfinite([a, b, c])) or min(a, a+b, c) <= 0:
        raise ValueError("Coefficients must give a positive, monotonic collection curve.")
    r = np.asarray(radius, dtype=np.float64)
    r_max = c/(a+b)
    valid = np.isfinite(r) & (r >= 0) & (r <= r_max)
    safe_r = np.where(valid, r, 0.0)
    sin2 = a*safe_r/(c-b*safe_r)
    # Clip only floating-point round-off after checking the measured radius.
    theta = np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0)))
    return np.where(valid, theta, np.nan)


def polarcam_apd_inverse():
    """Hugh's supplied inverse in [0,90,45,135] order; already an inverse.

    September 14/15 email and SI section 2; the SI calls this the complete
    applied correction. Do not invert again or assume it calibrates the camera.
    """
    return np.array([
        [2.56410256, 0., 0., 0.],
        [0., 2.65305220, 0., 0.],
        [-0.04047189, 0.04187587, 1.87478350, -0.00045528],
        [-0.04045247, 0.04185578, -0.00046721, 1.82373830],
    ])


def correct_polarcam_apd(channels, residual_inverse_gains=None):
    """Apply the supplied matrix once, optionally after explicit residual scaling.

    Input last axis must be [0,90,45,135], in detector units with any measured
    offsets already removed. No gain fitting, clipping, or offset guessing occurs.
    Supplied multipliers are inverse gains (multiply by them; do not divide).
    """
    values = np.asarray(channels, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError("Expected channels in [0,90,45,135] order.")
    g = np.ones(4) if residual_inverse_gains is None else np.asarray(residual_inverse_gains, dtype=float)
    if g.shape != (4,) or not np.all(np.isfinite(g) & (g > 0)):
        raise ValueError("Residual inverse gains must be four finite positive multipliers.")
    return (values*g) @ polarcam_apd_inverse().T


def pair_sum_imbalance(corrected_channels):
    """Dimensionless diagnostic (c0+c90-c45-c135)/(c0+c90+c45+c135)."""
    values = np.asarray(corrected_channels, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError("Expected channels in [0,90,45,135] order.")
    total = values.sum(axis=-1)
    valid = np.isfinite(values).all(axis=-1) & (values >= 0).all(axis=-1) & (total > 0)
    result = np.full(total.shape, np.nan)
    return np.divide(values @ np.array([1., 1., -1., -1.]), total, out=result, where=valid)
