"""Synthetic validation of the interference-tuning metrics.

Simulates the ROI mean intensity and the four mosaic channel means under:
  - a rod + background with per-channel interference, cycle-averaged over an
    exposure equal to one triangle drive cycle (the sinc residual),
  - a manual back-and-forth Z sweep (wobble) that traverses the residual
    interference locus,
  - slow common illumination crawl that inflates the raw intensity span but
    leaves the anisotropy locus nearly untouched,
  - shot-like multiplicative noise.

Checks:
  1. XY extent tracks |sinc(phi0/2)| and vanishes at the one-fringe amplitude
     (phi0 = 2 pi total phase sweep per drive cycle).
  2. Near the null the raw P98-P2 span is dominated by slow drift and grossly
     under-reports the suppression, while the XY extent reduction stays
     faithful.
  3. An under-covered Z sweep yields a shorter arc (reduced extent) even
     though the residual amplitude is unchanged - the closure caveat, i.e.
     why the arc must be closed (wobble through a full fringe) before the
     extent is read as a suppression number.

Run with pytest, or directly: python tests/test_interference_metrics.py
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polarcam.live.intensity_tuning import (  # noqa: E402
    IntensityTuner,
    xy_residual_stats,
)


FPS = 800.0
DURATION_S = 5.0


def _simulate(channels_base, b, *, phi0, wobble_amp, crawl_depth=0.0, crawl_hz=0.1,
              noise=0.002, wobble_hz=1.5, seed=0):
    """Return (times, intensity, channels) for one trial window.

    channels_base: (i0, i90, i45, i135) rod intensities at the parked state,
    without interference. b: per-channel background. phi0: total interference
    phase sweep per drive cycle (radians); the cycle average of cos(delta)
    over a triangle sweep is sinc(phi0/2) * cos(delta_centre). The wobble
    moves delta_centre: delta_c = wobble_amp * sin(2 pi f t) (radians).
    crawl_depth: fractional common illumination crawl at crawl_hz.
    """
    rng = np.random.default_rng(seed)
    n = int(FPS * DURATION_S)
    t = np.arange(n) / FPS
    sinc = np.sinc(phi0 / (2.0 * np.pi))  # np.sinc(x) = sin(pi x)/(pi x)
    delta_c = wobble_amp * np.sin(2.0 * np.pi * wobble_hz * t)
    cross = sinc * np.cos(delta_c)
    crawl = 1.0 + crawl_depth * np.sin(2.0 * np.pi * crawl_hz * t)
    channels = np.empty((n, 4))
    intensity = np.zeros(n)
    for j in range(4):
        rod = channels_base[j]
        signal = rod + b[j] + 2.0 * np.sqrt(rod * b[j]) * cross
        signal *= crawl  # common illumination crawl: all channels together
        signal *= 1.0 + rng.normal(0.0, noise, n)
        channels[:, j] = signal
        intensity += signal
    intensity /= 4.0
    return t, intensity, channels


def _feed(t, intensity, channels, tuner, kind, label):
    tuner.begin(kind, label, {"fps": FPS}, 0.0, duration=DURATION_S, target=10.0)
    start = 0.5
    for index in range(len(t)):
        tuner.add(
            float(intensity[index]), start + float(t[index]), start + float(t[index]),
            channels=tuple(float(v) for v in channels[index]),
        )
    return tuner.update(start + DURATION_S + 1.0)


def _channel_base():
    # Rod anisotropy r ~ 0.13 like the recorded data, equal per-channel backgrounds.
    rod = np.array([430.0, 560.0, 470.0, 540.0])
    background = np.array([30.0, 30.0, 30.0, 30.0])
    return rod, background


def test_extent_tracks_sinc_null():
    rod, background = _channel_base()
    # phi0 from 0.5*pi (half a fringe) to 3*pi; null at phi0 = 2*pi (one fringe pp).
    phis = np.linspace(0.5 * np.pi, 3.0 * np.pi, 11)
    extents = []
    for phi0 in phis:
        t, _, channels = _simulate(rod, background, phi0=phi0, wobble_amp=6.0, seed=1)
        stats = xy_residual_stats(t, channels, duration_s=DURATION_S)
        extents.append(stats["extent"])
    extents = np.asarray(extents)
    null_index = int(np.argmin(extents))
    # Null must sit at phi0 = 2*pi (index 6 of 11).
    assert null_index == 6, (phis, extents)
    # And the null extent must be small compared with the phi0 = pi extent
    # (the leftover is the shot-noise floor of the metric).
    assert extents[6] < 0.25 * extents[2], (extents[2], extents[6])


def test_span_under_reports_near_null_but_extent_holds():
    rod, background = _channel_base()
    tuner = IntensityTuner()
    # Reference: oscillation off (phi0 = 0), full fringe contrast traversed by
    # the wobble, mild illumination crawl as any real session has.
    t, intensity, channels = _simulate(
        rod, background, phi0=0.0, wobble_amp=6.0, crawl_depth=0.05, seed=2)
    reference = _feed(t, intensity, channels, tuner, "reference", "ref")
    # Test: one-fringe amplitude (true cancellation) but a 20% slow common
    # illumination crawl at 0.1 Hz during the window - exactly the drift that
    # defeats the span (the anisotropy ratios are invariant to it, so the
    # extent metric is not).
    t2, intensity2, channels2 = _simulate(
        rod, background, phi0=2.0 * np.pi, wobble_amp=6.0, crawl_depth=0.20, seed=3)
    result = _feed(t2, intensity2, channels2, tuner, "test", "nulled + crawl")
    comparison = result["comparison"]
    # The extent metric sees the true null (residual at noise floor).
    assert comparison["extent_reduction"] >= 10.0, comparison["extent_reduction"]
    # The span metric is polluted by the crawl and reports far less.
    assert comparison["reduction"] < 10.0, comparison["reduction"]
    # The two metrics disagree wildly: the span reduction is drift-limited
    # (crawl trace, not residual interference) while the extent reduction
    # reflects the true null. A lone span number cannot tell what it measured.
    assert comparison["reduction"] < 0.5 * comparison["extent_reduction"], (
        comparison["reduction"], comparison["extent_reduction"])


def test_wobble_coverage_sets_closure():
    rod, background = _channel_base()
    # Same residual amplitude (phi0 = 0), but a wobble that only sweeps a
    # fraction of a fringe: the traced arc is short, the extent under-reads.
    extents = {}
    for wobble_amp in (6.0, 1.0):
        t, _, channels = _simulate(rod, background, phi0=0.0, wobble_amp=wobble_amp, seed=4)
        stats = xy_residual_stats(t, channels, duration_s=DURATION_S)
        extents[wobble_amp] = stats["extent"]
    assert extents[1.0] < 0.6 * extents[6.0], extents


def test_tuner_end_to_end_reduction():
    rod, background = _channel_base()
    tuner = IntensityTuner()
    t, intensity, channels = _simulate(rod, background, phi0=0.0, wobble_amp=6.0, noise=0.001, seed=5)
    reference = _feed(t, intensity, channels, tuner, "reference", "ref")
    assert reference["complete"] and reference["xy"] is not None
    t2, intensity2, channels2 = _simulate(rod, background, phi0=2.0 * np.pi, wobble_amp=6.0, noise=0.001, seed=6)
    result = _feed(t2, intensity2, channels2, tuner, "test", "one fringe")
    comparison = result["comparison"]
    assert comparison["reduction"] > 5.0, comparison["reduction"]
    assert comparison["extent_reduction"] > 5.0, comparison["extent_reduction"]
    assert comparison["xy_target_reached"], comparison
    assert result["xy"]["extent"] < 0.2 * reference["xy"]["extent"]


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS {name}")
    print("all metrics tests passed")
