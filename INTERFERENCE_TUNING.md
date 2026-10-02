# Interference tuning: physics, metrics, and workflow

This document explains the interference-cancellation step of the APD/Polarcam
nanorod system, why the original span-only tuning number misleads, and how to
use the upgraded Interference tuning panel and the offline analyser.

---

## 1. What the cancellation does

The excitation laser scatters off a nanorod and also reaches the detector
coherently from parasitic reflections (dominantly objective back-reflection).
The per-channel measured intensity is

    I_j = I_rod,j + b_j + 2 sqrt(I_rod,j * b_j) * cos(delta)

with delta the rod-versus-background optical phase difference. Vibrating the
stage Z with a triangle wave at the stage resonance f0 (~830 Hz) while setting
the camera exposure to one drive cycle (1/f0) turns every recorded frame into
a cycle average. For a triangle sweep covering a total interference phase
phi0 per cycle, the cycle average of the interference term is

    2 sqrt(I_rod,j b_j) * sinc(phi0 / 2) * cos(delta_centre),   sinc(x) = sin(x)/x

so:

- sweep **less than one fringe** (peak-to-peak Z excursion below
  lambda / 2n ≈ 240 nm for the He-Ne line in water): sinc > 0, signal sits
  **above** the fringe mid-level;
- sweep **exactly one fringe** (phi0 = 2 pi): sinc(π) = 0 — the interference
  term vanishes **for any** delta_centre. This is the null. At the null the
  static Z position stops mattering;
- sweep **past one fringe**: sinc < 0 — the signal sits **below** mid. If your
  signal "always dips below mid", you are overshooting the null, not failing.

Two practical limits:

- f0 is a mechanical resonance of the stage and drifts (Hugh's handover:
  depends on the tightness of the stage screws). If the exposure is no longer
  1/f0, fringes reappear even at the correct amplitude.
- Larger amplitude is not better: the rod samples its axial intensity profile
  during the sweep, so the mean signal itself starts to oscillate. The useful
  amplitude is "just past one fringe", not "as much as possible".

## 2. Why the raw P98-P2 span is not enough

The panel's original metric is the P98-P2 span of the ROI mean intensity over
the trial window, compared between an oscillation-off reference and a test
amplitude. Two problems, both visible in real recordings
(`intensity_diagnostic_20261001-190946.npz`, 60 s at ~800 fps, medium drive):

1. **Slow drift dominates.** The mean intensity crawled by ~30 DN over the
   minute (all spectral power below ~0.3 Hz), while the span grew from 25 DN
   to 39 DN. A span number therefore mixes the interference residual with
   whatever the focus and illumination were doing during that window. Unless
   the manual Z sweep is repeated identically between reference and test
   windows — the "Comparable Z sweeps" checkbox — the reduction factor is not
   meaningful. The saved schema said as much: "not proof of optical
   cancellation".
2. **No picture.** A span cannot distinguish a genuinely suppressed residual
   from a window where the wobble happened to be small.

## 3. The XY residual metric (new)

The diagnostic recorder already stores the four mosaic channel means
(i0, i90, i45, i135) per frame. From these the anisotropy of the ROI is

    ax = (i0 - i90) / (i0 + i90),   ay = (i45 - i135) / (i45 + i135)

For a static dipole this anisotropy is intensity-independent — drift moves the
background-to-signal ratio only weakly — while the interference cross term
moves it along a residual locus whose size is the residual interference
amplitude. Subtracting a moving average (the detrend) removes the slow drift
and keeps the hand-wobble traversal of the locus. The metrics:

- **XY extent**: P98-P2 box diagonal of the detrended (ax, ay) locus — the
  size of the residual interference arc/loop, in anisotropy units.
- **XY reduction**: reference extent / test extent — the drift-immune
  suppression factor.

Validated on synthetic data with known physics
(`tests/test_interference_metrics.py`):

- the extent traces the sinc curve and vanishes exactly at the one-fringe
  amplitude (phi0 = 2 pi);
- with a 20% slow illumination crawl in the test window, the span reduction
  reads ~5x while the true suppression is >10x — the extent metric holds;
- an under-covered Z sweep produces a shorter arc even at full residual
  amplitude, which is why the locus must be **closed** before the extent is
  read as a suppression number (next section).

## 4. How to run a tuning session

1. Focus the rod, start magnifier analysis (Mode 2), background subtraction
   OFF, oscillation off.
2. Open Interference tuning. Set Window (s) — 5 s is a good default.
3. **Record reference**: while it runs, wobble the stage Z by hand
   back-and-forth through **at least one full fringe** (~240 nm peak-to-peak;
   you can watch the fringes in the live intensity plot). The reference is
   your "oscillation off" baseline; its XY locus is the full-strength
   interference loop.
4. Switch on the signal generator at some amplitude. **Test amplitude** while
   wobbling Z the same way.
5. Read the panel:
   - **Open arc** in the XY plot → the wobble did not cover a full fringe.
     Wiggle Z further; the extent number is not trustworthy yet.
   - **Closed loop, large** → amplitude still far from the null; turn Vp-p up.
   - **Closed loop, small / collapsing to a blob** → close to the null. The
     XY reduction column tells you how many times better than the reference.
   - The extent-per-trial bar chart builds up the measured sinc null curve as
     you sweep amplitudes; the dashed line is the target.
6. The best trial is the smallest closed loop, not merely the largest
   reduction: a tiny arc can look like a tiny extent (see 5 in the plot
   legend note).

Reading the reduction honestly: report it as a ratio against your own
reference, recorded in the same session with the same wobble discipline. The
extents from different sessions are not comparable.

## 5. Offline analysis of diagnostic recordings

The panel's **Record diagnostic** saves a 60-120 s npz with per-frame channel
means. Analyse it without the GUI:

    python -m polarcam.live.interference_npz_analysis RECORDING.npz

or several at once, with `--out DIR`, `--window S`, `--step S`. Outputs
(next to the file unless `--out`): a PNG with the intensity trace (events
marked), the raw and detrended XY loci, sliding span and extent, and a JSON
summary with per-segment statistics and reduction factors. Events marked in
the panel (Record reference / Test amplitude / Mark event) split the recording
into segments automatically — mark an event whenever you change the amplitude.

The offline analyser needs the package importable: run from the repo root with
`PYTHONPATH=src` or with the package installed.

## 6. What changed in the code

- `src/polarcam/live/intensity_tuning.py`
  - `roi_channel_means`: shared mosaic de-multiplexing (matches the
    diagnostic's channel mapping).
  - `anisotropy_series`, `moving_average_residual`, `xy_residual_stats`,
    `compare_xy`: the new metric stack.
  - `IntensityTuner`: trials now record per-sample channel means; completed
    trials carry an `xy` block; test-trial comparisons carry
    `extent_reduction` / `extent_relative_reduction` / `xy_target_reached`.
    A trial recorded without any channel feed stays valid (span-only
    workflow), so this is backwards compatible.
- `src/polarcam/live/Spinners_gui_live.py`
  - the live intensity callback feeds channel means into the tuner;
  - the Interference tuning window shows the detrended residual locus
    (reference grey, latest trial colour) and the per-trial extent bar chart;
  - the trials table has XY extent and XY reduction columns; the status line
    reports the XY reduction; trial JSON saves are schema_version 2.
- `src/polarcam/live/interference_npz_analysis.py`: new offline analyser.
- `tests/test_interference_metrics.py`: synthetic validation (sinc null,
  drift immunity, closure caveat, end-to-end tuner reduction).

## 7. Lab checklist for the first run

- [ ] Pull this branch on the lab machine; `python -m
      polarcam.live.interference_npz_analysis intensity_diagnostic_*.npz`
      reproduces yesterday's recording as a sanity check.
- [ ] Start the GUI normally; the tuning window now shows the XY plot frame
      (placeholder until the first trial completes).
- [ ] Record a reference with a deliberate full-fringe wobble; confirm the
      reference locus is a closed loop before recording any test.
- [ ] Sweep amplitudes: one test per Vp-p setting, same wobble. Watch the
      bar chart for the extent minimum.
- [ ] Confirm the exposure is still 1/f0 (the settings line shows the actual
      exposure and fps readback) — re-check after any stage screw adjustment.
- [ ] Save trials (JSON) and a diagnostic around the best amplitude; the
      before/after XY plot of reference vs best trial is the record figure.
