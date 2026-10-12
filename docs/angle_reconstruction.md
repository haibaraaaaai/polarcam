# Angle reconstruction for Polarcam and its APDs

The camera and the APDs on the Polarcam microscope share the collection optics:
finite NA, Fresnel transmission, BFP field mapping and the perforated mirror.
Use the circular annulus **NA 0.39–1.30** for both. Correct each detector branch
before calculating anisotropy. Hugh's APD matrix does not calibrate the camera.

This note assumes negligible APD background, and that camera background and
coherent interference have been dealt with experimentally. These are conditions
for the reconstruction, not effects removed by the optical coefficients. A rod
is approximated as one linear electric dipole. Theta is its acute angle to the
optical axis (0° vertical, 90° parallel to the coverslip), not automatically the
DNA hinge opening angle. Phi is known modulo 180°; the measurements do not
resolve the sign of the out-of-plane tilt.

## Which hole and which calibration

| Path | Inner NA | Channel correction before anisotropy |
| --- | --- | --- |
| Polarcam camera | 0.39 | Camera calibration if independently measured; never the APD matrix |
| APDs on the Polarcam microscope | 0.39 | Hugh's supplied inverse; optional justified residual gain correction |
| Separate APD microscope next door | Reported as 0.38 | Its own calibration and hole geometry; outside this implementation |

The manuscript and SI specify 0.39 from the imaged BFP on Polarcam. Hugh used
0.39 in the April/August correspondence. The evidence for 0.38 is Daping's
9 May email reporting a calculation from Martin's documentation, quoted in
Hugh's 11 May reply; it is not an independently rechecked measurement here.
Hugh also reported negligible effects of hole irregularity on Polarcam, but a
more asymmetric hole on the separate APD rig. His 9 September operating handover
describes background/interference procedures, not a numerical hole calibration.

## Shared optical model

Use this coefficient convention throughout; it differs from the SI's lowercase
inversion coefficients and from the APD optical transfer matrix:

\[
I_\psi=K\{A+B\sin^2\theta+C\sin^2\theta\cos[2(\psi-\phi)]\}.
\]

Here K is an unknown common brightness scale. For corrected intensities in
**[0°, 90°, 45°, 135°]** order:

\[
X=\frac{I_0-I_{90}}{I_0+I_{90}},\quad
Y=\frac{I_{45}-I_{135}}{I_{45}+I_{135}},\quad r=\sqrt{X^2+Y^2}.
\]

The direct forward and inverse relations are

\[
r=\frac{C\sin^2\theta}{A+B\sin^2\theta},\qquad
\theta=\arcsin\sqrt{\frac{Ar}{C-Br}},\qquad
\phi=\tfrac12\operatorname{atan2}(Y,X)\pmod\pi.
\]

The physical range is **0 ≤ r ≤ C/(A+B)**. Above-range values are invalid;
they are not pushed onto 90°. At r=0, phi is undefined. Sensitivity to noise
becomes large near 90°, so small radius errors can produce large angle errors.

### Deriving A B C

Let q be pupil radius in NA units, with q = n_s sin(alpha_s) = n_o sin(alpha_o).
Here s denotes the sample and o the effective glass/oil medium. Define

\[
c_s=\sqrt{1-(q/n_s)^2},\quad c_o=\sqrt{1-(q/n_o)^2},\quad
t_s=\frac{2n_sc_s}{n_sc_s+n_oc_o},\quad
t_p=\frac{2n_sc_s}{n_oc_s+n_sc_o}.
\]

Fresnel transmission multiplies the s and p **fields**, not their intensities.
The BFP field mapping is **sqrt(c_o)/c_s**, as in the optical schematic. With
p = t_p c_s and the pupil-area measure

\[
d\mu=\frac{q\,dq}{n_s^2}\frac{c_o}{c_s^2},
\]

integrate from q=0.39 to q=1.30:

\[
A=\tfrac14\int t_p^2(q/n_s)^2\,d\mu,\qquad
A+B=\tfrac18\int(p^2+t_s^2)\,d\mu,\qquad
C=\tfrac1{16}\int(p+t_s)^2\,d\mu.
\]

Azimuth has been integrated analytically. A common intensity normalization is
irrelevant to angle inversion. The implementation uses 128-point Gauss–Legendre
quadrature in q, cached once per optical geometry. The GUI then evaluates the
inverse formula directly for every spot or time point; no per-frame optical
simulation or interpolated angle lookup is needed.

For water **n_s=1.33**, effective glass/oil **n_o=1.515**, and NA **0.39–1.30**:

| Coefficient | Value |
| --- | ---: |
| A | 0.0977982598560731 |
| B | 0.0114807729168106 |
| C | 0.1013763500772505 |
| Maximum r | 0.9276834494677721 |

The GUI also retains the nominal 50% glycerol index 1.398, with coefficients
derived by the same function. It is a different medium from the manuscript's
99.5% glycerol (1.47); choose the actual medium rather than the curve that gives
an expected angle. Water uses 1.33 to match the manuscript convention.

To compare with the SI: its inverse is asin(sqrt(a*r/(b-c*r))). In that notation
(a,b,c) is proportional to **(A,C,B)**, not (A,B,C). Its J notation has
a=2J3, b=J1−J2 and c=J1+J2−2J3. Do not interchange these conventions.

This is one lossless effective interface, a centred circular hole and an ideal
dipole. It does not model multilayer reflections, near-field substrate coupling,
residual polarization retardance, aggregates, or incomplete spatial collection
by a small camera extraction window. No extra Snell-law correction is applied
to the reconstructed *rod* angle: refraction is already in the collection model.

A measured asymmetric BFP mask can be used in a full field integration. In
general it introduces phi-dependent terms, so it cannot simply replace this
annulus with another universal set of three coefficients. Both theta and phi
would then require the more general forward model. That extension is not used
in the present GUI.

## APDs on Polarcam

Reorder TDMS acquisition channels first: **ai0=90°, ai1=45°, ai2=135°, ai3=0°**.
All following vectors and matrices use **[0°,90°,45°,135°]**.

The forward instrument model is I_measured = D H I_ideal, where H describes
optical transmission/mixing and D describes detector scaling. Thus the correction
order is **H inverse after D inverse**. Hugh confirmed on 15 September that the
matrix he sent is already an inverse. The SI calls the following matrix the
complete applied correction:

```text
M_ref = [[ 2.56410256,  0,           0,           0          ],
         [ 0,           2.65305220,  0,           0          ],
         [-0.04047189,  0.04187587,  1.87478350, -0.00045528 ],
         [-0.04045247,  0.04185578, -0.00046721,  1.82373830 ]]

I_corrected = M_ref @ I_measured
```

**Apply it once; do not invert it again.** Hugh's 14 September reply also confirms
that this numerical inverse already has the correct sign convention; do not
apply another beta sign flip based on the older bearing-paper derivation.

### Checking relative gains

After correction, test **c0+c90 ≈ c45+c135**, allowing for measurement noise.
A convenient diagnostic is their difference divided by their sum. This is a
necessary consistency check, not proof that the angle or calibration is correct.

If detector settings have changed from the reference calibration, additional
positive *residual inverse-gain multipliers* g may be justified:

\[
I_{\rm corrected}=M_{\rm ref}\,\operatorname{diag}(g)\,I_{\rm measured}.
\]

The default is g=(1,1,1,1). These are multiplicative corrections, not the forward
detector gains. Do not apply a second full D inverse on top of a matrix that
already includes it. Existing analysis functions that automatically fit gains
must be interpreted as fitting residual scaling relative to M_ref.

With w=(1,1,−1,−1), a residual fit minimises
sum_t [w M_ref diag(g) I_measured(t)]², fixing g0=1 to remove the unobservable
common scale. Fit against the **corrected** pair sums, with M_ref inside the
objective; balancing raw pairs before applying M_ref is a different operation.

Use a calibration trace with sufficiently diverse polarization states, such as
a suitable rotating rod. A stationary orientation, even with many noisy frames,
does not reliably identify three relative gains. Check positivity, conditioning,
stability between windows and pair-sum agreement on held-out samples. Do not
refit each stationary hinge to force equality or force a desired angle peak.

Any measured offsets must be removed in their own channel units. A background
measured in raw units belongs before the linear correction; a background already
transformed by the same correction can be subtracted afterwards. None is inferred
from pair-sum equality alone. Preserve negative corrected values as invalid
photometry rather than silently clipping them to zero.

## Camera and software use

For the camera: extract the four channel means with the correct sensor parity,
apply any separately established camera calibration, calculate X/Y, then use the
shared optical inverse. Hugh's four-arm APD matrix and APD gains are not applied.
The main GUI currently assumes the camera channel response and optical
polarization compensation are adequate; the new optical coefficients do not
establish that calibration. Inspect pair-sum balance as a diagnostic here too.

Background removal, any-pixel saturation rejection, spot detection, extraction
window and time averaging remain separate operations. Adding the background back
for a saturation check does **not** mean adding it back to the angle signal.
For one stationary-spot angle, average the corrected four intensities first;
the GUI's existing X/Y and Cartesian averaging choices answer different questions
when the rod moves and remain explicitly selectable.

The maintained implementation is [src/polarcam/optics.py](../src/polarcam/optics.py).
For post-measurement APD processing, with a final channel axis in the stated order:

```python
from polarcam.optics import theta_model, theta_from_radius, correct_polarcam_apd, pair_sum_imbalance
from polarcam.live.intensity_spots import anisotropy
import numpy as np

corrected = correct_polarcam_apd(raw_channels)  # matrix once; no automatic gain fit
balance = pair_sum_imbalance(corrected)
xy = anisotropy(corrected)                     # or block-mean channels first
model = theta_model()                          # water, NA_in=0.39
theta_deg = np.degrees(theta_from_radius(np.linalg.norm(xy, axis=-1),
                                        model['A'], model['B'], model['C']))
```

The main live GUI uses this same optical model for spot theta traces,
frame-stack theta/projection statistics and live/stationary angle estimates.
Analysis exports retain the coefficient convention, numerical coefficients,
refractive indices, pupil geometry and model version `annular-fresnel-bfp-v1`.
Restart an already running GUI to load the change.

Historical notebooks, generated galleries and separate experimental GUI utilities
are not retroactively converted by this change. Some use NA_in=0.38, another
coefficient convention, fitted gains, or angle clipping; record those choices
when comparing old results. The replaced main-GUI water curve had maximum
r=0.82166. For the same r=0.39473, it gave 27.22°, whereas this model gives
39.15°. That discrepancy was not a small Fresnel correction.

## Evidence and numerical checks

- *Polarcam Manuscript V2 2026-09-08 RB*, angle-reconstruction section, and
  *SI outline*, sections 2–3: hole geometry, APD matrix, conventions.
- Hugh, *Angle Reconstruction — background characterisation*, 29 April 2026;
  *Using total intensity to estimate background / latest version of Fourkas*,
  11 May 2026, quoting Daping's 9 May note: setup distinction and hole evidence.
- Hugh, *TDMS files*, 14–15 September 2026: acquisition order, supplied inverse
  and beta sign confirmation.
- [Hugh's measured-hole simulation](https://github.com/Hugh-Bowman/Dipole_through_bfp_jagged_hole_optics_simulation)
  provides the separate mask-based investigation; its existence does not validate
  a Fresnel/BFP convention by itself.
- The new implementation recovers analytic annular Fourkas coefficients when
  sample and immersion indices are equal; agrees with independent two-dimensional
  vector-field integration; converges with quadrature order; and passes angle
  round-trip, invalid-radius and APD matrix/gain-order checks. These are software
  and model-consistency checks, not experimental validation of absolute angles.
