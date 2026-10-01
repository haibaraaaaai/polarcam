# Lab Setup And Acceptance

## Stage 1: Open The Application

Use the `testing/v3-rewrite` checkout. Double-click **Polarcam Lab** on the desktop
after installing the shortcut below, or open [Launch-Polarcam.cmd](Launch-Polarcam.cmd)
directly. This launches the actual application, not the angle-analysis tool. It
does not automatically start camera acquisition. A diagnostic console stays open
while the app runs; if startup fails, retain its error text before closing it.

Recordings and active backgrounds default to `runs/` inside the repository.
Use an explicit data folder on a suitable local disk for experiments. Do not use
Hugh's working installation or its data directory for these initial tests.
The working directory controls relative output, but file dialogs can choose other
locations. Verify each tab's save folder before capture.

### Another Windows PC

1. Install Python with Tk support, version >=3.12.10 (a tested 3.12 or 3.13 build
   is preferable to changing Python versions immediately before an experiment).
2. Clone this repository and select `testing/v3-rewrite`. For an existing checkout,
   preserve local work before fetching and pulling with `--ff-only`.
3. Open PowerShell in the repository root and create a project-local environment
   if it does not already exist:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

4. For camera use, install IDS peak Cockpit, SDK, and the camera drivers first.
   Confirm the device works in Cockpit, then close Cockpit to release the camera:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[camera]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\Launch-Polarcam.cmd --help
```

5. Create the desktop shortcut, optionally choosing a data disk:

```powershell
.\Install-DesktopShortcut.ps1
# Or choose an experiment root before creating the shortcut:
# .\Install-DesktopShortcut.ps1 -DataDir "D:\Polarcam Data"
```

If local policy blocks a trusted downloaded script, use the approved script policy
for your machine; do not change machine-wide execution policy for this application.
The CMD launcher works directly without running the PowerShell installer. The
installer refuses to overwrite an existing **Polarcam Lab** shortcut. To change
the repository or data location, explicitly remove that shortcut and recreate it.
Moving the checkout or deleting its `.venv` invalidates the shortcut. Neither the
environment, calibration files, nor experiment recordings travel automatically
with Git. In VS Code also select this repository's `.venv` interpreter.

### Tcl/Tk Startup

On the lab PC, Python 3.13 could import Tkinter but could not locate `init.tcl`
when creating a window. The Tcl/Tk files were present under the base Python
installation. The application launcher now supplies those matching runtime paths
to its child process on Windows, without changing system settings or replacing
existing `TCL_LIBRARY` / `TK_LIBRARY` overrides. Use the launcher or
`python -m polarcam` for the main app to get this startup configuration.

If the runtime files are actually missing, repair that Python installation with
Tcl/Tk support. Do not copy Tcl/Tk from another Python or Conda installation. Other
standalone scripts do not automatically use the main launcher's environment fix.

## Today's Staged Plan

The experiment is scheduled for 2026-10-02. Stage 1 is startup and desktop access;
it is not certification of camera acquisition or scientific results. Keep the old
installation untouched as a fallback until the required lab checks pass.

Update 2026-10-01: the user chose to skip the general Stage 2 acceptance pass and
defer a broad rebuild. Its rows below remain an unverified reference, not completed
tests. Work moved directly to investigating magnifier-intensity lag; see the
measurements and recommended local fixes in [REWRITE_NOTES.md](REWRITE_NOTES.md).

| Stage | Workflow / Owning Functions | Acceptance Check | Status |
| --- | --- | --- | --- |
| 1 | Desktop launcher, `polarcam.cli.main`, `BasicVideoPlayer.__init__` | Launch with the intended environment/data folder; no startup exception; close cleanly | Passed 2026-10-01: shortcut installed, GUI smoke check passed, real app opened and responding |
| 2 | Live feed: `_start_live_feed`, `_apply_live_settings` | Acquire; change exposure/analog gain; check actual camera readback, not just entry text; stop/start | Needs camera |
| 2 | Background: `_select_background_profile_stack`, `_capture_background_profile_from_live_settings`, `_subtract_background_frame`, `fetch_frames._subtract_background` | Use matching full-sensor background; confirm cropping, subtraction, saved correction status and reproducible profile info | Needs sample/background |
| 2 | Magnifier/spot targeting: `_on_live_click`, `_live_update_xy_preview`, `_roi_from_target_center` | Magnifier follows the intended rod; sensor coordinates/ROI agree; distinguish display stretch from saved counts | Needs camera |
| 2 | Interference/intensity trace and guided tuner | Start/stop on magnified ROI; compare fresh reference/test sweeps and measure backlog | Start/stop, caching, plot reuse and 10x tuner implemented; camera-free checks pass; physical suppression not yet validated |
| 2 | Full-frame capture/spot finding: `_on_fetch_frames`, `_capture_frames_to_npy`, `_recon_worker` | Save a short full stack and JSON; reopen with correct FPS/count; find/select rods | Needs camera/sample |
| 2 | Single-spot capture: `_start_spotrec`, `_stop_spotrec`, `_capture_stationary_mode` | Short ROI recording saves and reopens; actual ROI/FPS/exposure/gains agree; no background/phase ambiguity | Needs camera/sample |
| 2 | XY plots: `_append_xy_frame`, `_xy_phi_stats_from_frame`, `_spotrec_refresh_plots` | Same pixels/window give consistent channel signs and XY in detection and recording | Automated signs checked; real data pending |
| 2 | Auto exposure/FPS: `Controller.desaturate`, backend timing and ROI controls, `_capture_auto_spot_series` | Establish which workflow actually adjusts exposure automatically; verify clamped FPS after ROI changes | Do not assume automatic exposure is wired into capture |
| 3 | Alternative intensity-only detector for all rods | Compare detections on a real full-frame stack; keep rotating-rod detector available | Planned, not implemented |
| 3 | All-rod theta distribution and stereographic scatter/density; offline replay | Save raw stack plus metadata, reprocess offline; agree optical model, masks, per-rod weighting, projection convention | Planned, not implemented |
| Later | Spot cycler, persistent capture, buffer reuse, scientific fits | Separate measured performance and numerical acceptance | Deferred from startup stage |

The main workflows above live in
[src/polarcam/live/Spinners_gui_live.py](src/polarcam/live/Spinners_gui_live.py),
[src/polarcam/live/fetch_frames.py](src/polarcam/live/fetch_frames.py), and
[src/polarcam/live/backend/ids_backend.py](src/polarcam/live/backend/ids_backend.py).
See [REWRITE_NOTES.md](REWRITE_NOTES.md) for earlier decisions and known limitations.

### Verification On 2026-10-01

- `Launch-Polarcam.cmd --help` passes without creating a GUI or accessing a camera.
- All 36 camera-free tests pass, including real Tk window creation, callback
  checks, and shutdown in a temporary data directory; none skipped on this PC.
- Subsequent magnifier/default fixes extend this to 46 passing tests: analyser
  Stop resumes full-frame view, centre and image updates survive repeated cycles,
  and internal stops/window closure do not restart a camera.
- The guided tuner and performance work extend the suite to 66 tests, including
  synthetic suppression ratios, freshness/quality guards, JSON export, queued
  frame timestamps, native precision, cache/plot reuse and sidebar scrolling.
- Runtime and IDS Python-binding imports pass in this checkout's Python 3.13.0
  `.venv`. No package upgrade was needed or performed for this startup stage.
- The installed desktop shortcut was opened; the application window is responding.
  The data root is `C:\workspace\polarcam\runs`, with existing tab-specific
  subdirectories selected by the application. Check the displayed save folders.
- Camera acquisition, control readbacks, correction provenance, recording
  integrity, and live intensity responsiveness still need the Stage 2 lab trials.

## Current Defaults And Magnifier Controls

- Exposure defaults to **1.2 ms** across the main live view, magnifier analysis,
  full-frame capture and spot capture; gain defaults to **1.0**.
  This is the initial interference-tuning exposure. The user's reported procedure
  then uses 0.6 ms and twice the tuned drive voltage for acquisition; see below.
- Background subtraction defaults to **OFF** in the main live and save controls.
  Enable it explicitly when a matching profile is ready. The live checkbox
  controls the intensity analyser; the save checkboxes control recorded data.
- Start analysing enables the magnifier at its selected centre. **Stop analysing**
  closes the ROI stream, then resumes full-frame live feed at that same centre.
  The magnifier remains enabled and updates from fresh live frames.
- Tab-change/capture/shutdown cleanup does not use that automatic return action.
- Close and relaunch **Polarcam Lab** after stopping acquisition to load these
  changes. Existing windows are not updated in place and have not been interrupted.
- At 1.2 ms, the physical FPS ceiling is roughly 833 or less. Requested FPS fields
  remain unchanged; use actual readback and allow longer frame-count captures.
- The analyser now caches its background lookup/crop and reuses its plot. Its
  intensity values are native camera DN, not the former 8-bit display counts.
  Preview images still use 8-bit display conversion; existing recordings and
  unrelated polarization/angle calculations are unchanged.

## Intensity-Based Spot Analysis

2026-10-01: interference troubleshooting is deferred until tomorrow. Added an
intensity-selected workflow in the existing **Spot analysis** tab; the original
**Time variation** mode remains available and is still the startup default.

### Capture And Replay

1. Safely stop acquisition and relaunch Polarcam to load the changes. In **Spot
   analysis**, choose **Detection = Intensity** before fetching or opening a stack.
2. Set **Recording = Sample** or **Background**, and **Interference drive (manual)
   = On / Off / Unknown**. Add known frequency, amplitude, servo and sample details
   in **Condition note**. These fields record operator declarations; they do not
   operate the generator or confirm that cancellation occurred. They are separate
   from the older global **Sound on / vibrating** checkbox.
3. Set exposure, gain, FPS and frame count/duration, then **Choose save folder**.
   For retained raw measurements leave **Save with background subtraction** OFF.
   **Fetch frames** saves the frame stack as NPY plus a JSON sidecar and reopens it.
   The helper preserves native capture dtype; an explicitly marked phase-marker
   frame may follow the measurement frames and is stripped on replay.
4. Record a background stack separately with **Recording = Background** and the
   relevant interference state. This labels and preserves the stack; it does NOT
   automatically install it as the application's stored correction profile.
   The existing subtraction option still uses the stored profile and records both
   requested and actually applied subtraction. Do not subtract twice on replay.
5. **Select AVI/NPY** reopens an existing stack without operating the camera. The
   source JSON metadata is retained; current capture controls do not relabel it.
   Missing FPS still prompts. Native NPY is preferred over lossy AVI.
6. Detection averages every measurement frame, averages each four-polarizer cell
   spatially to remove mosaic bias, and runs the existing DoG/area/edge detector
   on that intensity map. It does not require temporal variation or hollow XY
   trajectories. Candidates are ranked by mean intensity; motion/hollowness
   filters and automatic camera inspection are disabled in intensity mode.
7. Adjust **DoG k** and **Spot window**, then **Update analysis** to replay.
   Lower DoG k includes fainter candidates and more noise. The inherited initial
   threshold is 8; inspect the selected spots on the actual sample before using
   the distribution. The odd display window uses the next smaller even raw crop
   for equal-size channel measurements (19 -> 18 pixels). Switching Detection
   and pressing Update analysis reruns the chosen mode on the same source.

No full-stack copy is made for NPY analysis: it is memory-mapped, accumulated into
a float64 mean, then replayed to extract channel means at detected coordinates.
Saved pixels are not rescaled for measurement. Spatial crop and sensor-origin
parities both enter the `[90,45;135,0]` channel assignment. Origin comes from the
sidecar, then a saved phase marker, otherwise a full-frame (0,0) assumption that
is recorded in the export. Cropped files without origin information require
verification. AVI retains the existing decoded-frame cache.

### Averaging And Distribution

The supervisor requested averaging before projection and density, with uncertainty
about whether averaging should occur in I(1-4), anisotropy X/Y, or 3D Cartesian
coordinates. All three are available under **Intensity orientation**:

- **3D Cartesian** (default): reconstruct each frame's ROI-channel X/Y, map its
  radius through the selected theta model, choose a continuous azimuth branch,
  average unit vectors in each window, and normalize the resultant direction.
  Preserve the unnormalized resultant length as a spread/cancellation diagnostic.
- **Anisotropy X/Y**: average the per-frame normalized pair differences first,
  then reconstruct one unit vector per rod/window.
- **Four intensities**: average each native channel first, form normalized pair
  differences from those means, then reconstruct one unit vector. This generally
  weights brighter frames differently from averaging X/Y.

**Frames / average = All** gives one equally weighted direction per detected rod
over the whole stack. A positive integer gives nonoverlapping windows; the last
partial window is included and its bounds/valid counts are exported. Invalid
channels, nonpositive pair sums, out-of-model reconstructed radii and vanishing
Cartesian resultants are not silently assigned an orientation. The selected
method determines whether the model validity check is before or after averaging.

**All-rod distribution** shows the unit-sphere representatives, stereographic
scatter and a normalized 64x64 histogram density. Projection is
`(nx, ny) / (1 + nz)`. Density is probability per PROJECTED-PLANE area, with one
vote per valid rod/window, not density per solid angle and not a smoothed KDE.
The displayed acquisition labels come from the saved stack, not today's controls.

Scientific conventions and limits:

- Theta uses the existing explicitly selectable `water` or `glycerol50` finite-NA
  model and its 0.1-degree LUT. Coefficients are exported. These models have NOT
  been experimentally calibrated for tomorrow's sample; no APD calibration or
  unrelated theta/unwrap/sphere-fitting code was changed.
- Polarization azimuth is modulo 180 degrees and tilt/polarity is ambiguous.
  Cartesian averaging unwraps doubled azimuth across valid frames before taking
  the half angle, avoiding the 179/1-degree discontinuity. This chooses a continuous
  representative, not uniquely measured 3D polarity. It assumes no unresolved
  azimuth changes over 90 degrees between valid frames, including gaps.
- Final representatives have nonnegative z and phi folded to [0,180) degrees,
  so the projection is an upper half disk. Fold boundaries remain conventions;
  this is not an oriented-vector full-sphere measurement. Small mean resultant
  lengths and substantial missing samples require inspection, not overinterpretation.
- No local-background estimation, rod tracking, or calibrated orientation-error
  model is added. Bright-background bias, overlapping objects and drift outside a
  fixed crop can change the result. Compare methods on the same saved stack.

**Save analysis** writes an atomic, pickle-free NPZ containing:
`channels` (frames x rods x 4, I0/I90/I45/I135), `centers`, raw `bounds`, `mean_frame`,
`directions`, `projected`, `valid_counts`, `resultant_length`, `window_start/stop`,
`density`, histogram edges, and `metadata_json`. Metadata includes the source path,
source JSON, detector settings, model coefficients, averaging/projection rules,
origin source and validity totals. Non-applicable resultant lengths are NaN.
The original NPY/JSON stack remains separate and unchanged; preserve both for
offline reanalysis. The capture's JSON records purpose/interference conditions,
camera readbacks and requested/applied subtraction, not these later analysis choices.

Measurement-trace storage is capped at 256 MiB, with an explicit error rather than
silently discarding rods/frames. A distribution is capped at 100,000 rod/window
averages; increase the averaging window when needed. The stack capture helper
still buffers its recording before saving, and distribution rendering/export can
briefly occupy the GUI. No cycler or asynchronous disk streaming was added.

Validation: 92 camera-free tests pass. Added stationary-spot detection, native
precision, all channel-origin parities, averaging-order/boundary tests, invalid
samples, density normalization, NPY replay, metadata/export, mode switching and
cancellation/error coverage. A synthetic five-frame 1536x1024 stack found all 96
stationary spots in about 0.10 s excluding plotting; all three methods returned
normalized distributions. Tk checked at 1200x800 and distribution at 860x540.
No real camera or generator was operated; tomorrow's sample remains unvalidated.

## Guided Interference Tuning

Latest 2026-10-01 update: manual Z sweeps proved difficult to reproduce, and the
user reports an approximately five-second intensity oscillation with no deliberate
Z motion. Use **Intensity diagnostic** first (below); no reference or sweep
acknowledgement is needed. A tuning score alone cannot diagnose this observation.

Stop acquisition, close the old process and reopen **Polarcam Lab** to load this
feature. It measures reduction relative to your own reference; it does not set
the generator voltage or read either physical knob.

1. In Live video, start preview and select the rod/magnifier centre. Use gain 1,
  subtraction OFF, and the intended exposure in **Magnifier capture settings**.
  Start with the 1.2 ms tuning procedure described below.
2. Open **Interference tuning** from the top bar or analyser section, then use
  **Start analysing** if needed. The live sidebar now scrolls to expose lower
  controls. Wait for actual exposure, FPS, gains and delivery-age readouts.
3. Establish your reference drive condition using the lab's procedure, with
  clearly visible fringes during a small repeatable Z sweep around focus. Do
  not use an already-suppressed or stationary-Z trace as the reference.
  Enter a reference label; optionally note measured frequency, waveform and
  attenuation in **Drive setup note**. That note is manual metadata, not readback.
4. Leave **Window (s) = 5** and **Target (x) = 10** initially. Confirm comparable
  Z sweeps with amplitude held fixed, click **Record reference**, and repeat
  those sweeps. There is a 0.5-second settling exclusion followed by the capture.
5. Adjust Vpp to a candidate within the established safe range, then HOLD IT
  FIXED. Enter the measured Vpp or an explicitly estimated trial label in
  **Trial / Vpp label**, confirm the sweep checkbox again, and click **Test
  amplitude**. Repeat the same Z range and similar speed while the ratio updates.
6. Compare completed trials and retest the best setting. A final **>=10x observed
  target** requires both the raw and brightness-normalized ratios to reach 10.
  Partial-window ratios are provisional and never mark the target reached.
7. Use **Save trials** between tests or after stopping analysis. It saves JSON
  containing all retained native ROI-mean samples, host timestamps, actual camera
  settings, manual labels, scores and warnings. It does not save raw image stacks.
  Up to 40 completed trials are retained in memory; save before Clear trials or
  exiting the app. Cancel or closing the tuning window discards an active partial
  trial but does not stop the analyser or delete completed trials.

The metric is `span = P98(intensity) - P2(intensity)` and
`reduction = reference_span / test_span`. The normalized ratio compares
`span / mean` between the reference and test, helping distinguish suppression from
simple dimming. The best-trial label is ranked by the normalized ratio among valid
completed tests sharing the current reference. The plot scale stays fixed to the
reference while it is active. Progress-bar completion alone is not a pass.

These percentiles measure residual intensity modulation, not an isolated optical
interference term. The handover's 20x/6x statements do not define this exact metric;
10x here is a user-selected benchmark, not a prediction or hardware guarantee.
Without Z readback, the app cannot verify scan coverage or know that a knob moved.
Repeat comparable sweeps; stopping Z or defocusing can still create misleading data.

Tuning requires background subtraction OFF and actual ROI/timing/gain readbacks.
Stopping/restarting analysis, changing camera/ROI settings, changing the drive
setup note, or toggling subtraction invalidates the current reference. Changing
trial duration also requires a new reference. At 0.6 ms, record a new 0.6 ms
reference; do not reuse the 1.2 ms baseline. Keep changing Vpp labels separate
from the fixed drive-setup note.

The target is withheld for flat/nonpositive traces, clipped ROI pixels (0 or
>=4095 under the backend's Mono12 contract), a mean-brightness change beyond a
factor of two, invalid data, incomplete capture or delivery backlog above 250 ms.
Fewer than 32 samples cannot be scored; completion requires endpoints within
250 ms of the requested window and at least 90% of the count expected from actual
FPS. No recent frame for 500 ms invalidates the comparison. These are conservative
quality screens, not proof of identical focus or absence of lost sensor frames.

Frame timestamps are host `perf_counter` values at SDK buffer receipt, before
GUI queuing, not camera exposure timestamps. Older queued frames cannot enter a
new trial; delays before the SDK supplies a buffer remain unmeasured. Acquisition
is not throttled or thinned for display. Only the plotted line uses a peak-preserving
envelope; scores and saved traces use every accepted measurement sample.

Synthetic microbenchmarks on this PC: 1640 analyser callbacks took about 14 ms with
no profile and 32 ms with a cached profile, versus the earlier 1.3-2.0 seconds with
repeated background path lookups. Plotting a 16400-sample trace took about 30 ms
median versus 69 ms before. These exclude real-camera throughput/latency and do
not establish achieved physical suppression. Background changes made outside the
GUI require restarting/toggling analysis correction or selecting the profile again.

## Intensity Diagnostic Recording

This is an independent recorder inside **Interference tuning**, not another
suppression test. It records without a reference, Z sweep, tuning target, or
complete camera readbacks. Clipped, flat, delayed and questionable measurements
are retained for diagnosis rather than excluded by tuning quality screens.

1. Restart Polarcam after safely stopping acquisition to load the new controls.
  Start live view, choose the magnifier region, then **Start analysing**.
2. In **Trial / Vpp label**, identify the condition, e.g. `Z untouched, drive A`.
  In **Drive setup note**, enter the frequency, Vpp or dial estimate, attenuation,
  waveform and servo state you actually know. Do not guess measured values.
3. In **Intensity diagnostic**, leave **Duration (s) = 60**, press **Record
  diagnostic**, and choose the output `.npz` file. This duration is separate
  from the five-second tuning-trial duration. Recording starts after the dialog.
4. For the first recording, leave Z and drive amplitude untouched. Sixty seconds
  should include about twelve repetitions if the observed period is five seconds.
  Prefer background OFF for a direct raw comparison, but diagnostics also work
  with subtraction on and record both raw and analysed intensity.
5. **Mark event** stores the current trial label, drive note and settings with a
  host timestamp. Use it for an observed change or deliberate action; it does
  not read physical controls or automatically detect their movement.
6. Capture ends and saves automatically after the selected duration. **Stop and
  save** ends only diagnostic capture, leaving the analyser running. Stopping
  the analyser or closing the tuning window also finishes/saves a partial diagnostic.
  The saved path, sample count and stop reason appear below the controls.
7. If saving fails, data stay in memory and a new diagnostic/window close is
  blocked. Resolve the reported filesystem error and use **Retry save**. A forced
  process termination still loses unsaved RAM; this is not a crash-safe recorder.

The pickle-free NPZ contains `samples` (float64 rows), `columns` (column names),
and `metadata_json` (JSON string). Per-frame columns are:

- `received_s`, `processed_s`: host SDK-buffer receipt and GUI processing time,
  relative to diagnostic start. Their difference exposes GUI delivery backlog.
- `frame_index`: the analyser's processed-frame index, NOT a hardware frame ID.
- `raw_mean_dn`, `analysed_mean_dn`: native magnifier-ROI mean before and after
  any enabled background correction. Neither comes from stretched display pixels.
- `i0_mean_dn`, `i90_mean_dn`, `i45_mean_dn`, `i135_mean_dn`: separate RAW channel
  means with global ROI plus local-crop parity accounted for.
- `raw_min_dn`, `raw_max_dn`, `zero_fraction`, `saturated_fraction`: raw ROI
  statistics. Saturation is assumed at 4095 under the current Mono12 backend
  contract, not at uint16's 65535 limit.

Metadata includes initial settings and timestamped ROI/timing/gain readbacks,
background-state events, camera/callback errors and manual markers. Missing
readbacks remain missing; samples are not re-timed to make an assumed FPS fit.
Old frames received before diagnostic start are counted but excluded. In-window
delayed/non-monotonic samples are retained. SDK receipt times still cannot reveal
sensor exposure time or latency before the SDK returns the buffer. Full raw images,
Z displacement and electrical generator voltage are NOT recorded.

Duration is 10-120 seconds, with a 500,000-sample hard cap and explicit stop reason.
The preallocated buffer is about 52 MB, and NPZ output writes only retained rows.
No file is written per frame; saving occurs at completion and can briefly occupy
the UI. A no-frame recording still saves an empty array and metadata. The sampled
window ends at the requested host time; delayed delivery beyond the 250 ms finish
grace may remain uncaptured. Do not interpret the diagnostic as proof of zero loss.

Example offline inspection (without opening a camera):

```python
import json
import numpy as np
import matplotlib.pyplot as plt

with np.load("intensity_diagnostic.npz", allow_pickle=False) as recording:
   samples = recording["samples"]
   columns = {name: index for index, name in enumerate(recording["columns"])}
   metadata = json.loads(str(recording["metadata_json"]))

received = samples[:, columns["received_s"]]
raw_mean = samples[:, columns["raw_mean_dn"]]
plt.plot(received, raw_mean)
plt.xlabel("Host SDK receipt time (s)")
plt.ylabel("Raw ROI mean (DN)")
plt.show()
```

## Interpreting A Slow Intensity Wave

Generator frequency sets the repetition rate of the voltage command. Vpp sets
its excursion around the DC offset; through the PI amplifier and piezo this changes
stage-displacement amplitude. The mapping depends on load, frequency, resonance,
feedback mode and amplifier limits. Twice the command voltage is not a verified
doubling of stage displacement or guaranteed interference cancellation.

Changing optical path changes phase, producing fringes. The camera averages
intensity over each exposure and samples those averages at the actual frame rate.
A residual intensity component at `f_I` can appear at a slow alias frequency
`abs(f_I - k * fps)` for an appropriate integer `k`. For example, 830 Hz residual
intensity sampled at 829.8 fps could look like a 0.2 Hz (five-second) oscillation.
This is an illustrative hypothesis, not a diagnosis or measured device frequency.
The intensity waveform may contain harmonics of stage motion.

Exposure averaging can suppress a periodic component: for a sinusoidal intensity
component, attenuation is proportional to `abs(sin(pi*f_I*T)/(pi*f_I*T))`. Exact
period averaging assumes stable periodic motion; 1.2 ms alone does not establish
the actual period or waveform. Larger drive amplitude changes optical phase
excursion and harmonic content, not just brightness or generator frequency.

Other explanations include real slow Z/focus drift, illumination fluctuations,
mechanical instability, or delayed/bursty delivery and plotting. The diagnostic
lets us compare raw intensity/channel traces, actual settings and host arrival
timing; it cannot independently identify the physical cause from one clip.

After the stationary first clip, useful comparisons change ONE quantity at a time:
repeat at the same settings, or make a small permitted camera-FPS change while
holding exposure and hardware drive fixed and recording actual readback. A slow
frequency that moves with FPS is evidence for sampling aliasing. Changing exposure
can test integration sensitivity but also changes counts/noise, so interpret with
care. Only compare nearby permitted drive amplitudes with frequency/exposure/FPS
and Z held fixed; reduced stationary fluctuation alone is not proof of robust
cancellation across Z. The five-second period does not by itself say higher/lower Vpp.

Validation: 75 camera-free tests pass, including raw channels at all ROI parities,
delay/clipping retention, no-reference diagnostics, empty stalled recordings,
automatic and manual saving, and failed-save retention/close guards. A synthetic
active diagnostic took about 120 ms per 1640 callbacks including frame creation
and saved all 8200 test samples. This is not a measured real-camera throughput.

## First Diagnostic Result: 2026-10-01

Analysed the user's stationary-knob recording
[runs/intensity_diagnostic_20261001-190946.npz](runs/intensity_diagnostic_20261001-190946.npz)
read-only. The condition note says "diagnostic, no z change, medium voltage, no
voltage change". Actual generator frequency, Vpp, servo state and displacement
were not measured or recorded. `sound_on_manual: false` comes from the legacy
manual Sound on / vibrating checkbox, not a hardware readback. It does not prove
that the drive was off, especially given the recording's medium-voltage note.

- 47,927 rows cover 59.999 s, with 798.781 frames/s from host receipt times,
  consistent with the 798.791 fps readback (47,927.46 expected rows in 60 s).
  Exposure readback is 1194.270 us, analog/digital gains both 1, background OFF.
  Camera ROI is 256x14; intensity uses a 14x14 crop. No setting-change/error
  events occurred; the only event is the automatic duration stop.
- All values are finite; receipt times increase and analyser indices are
  consecutive. Raw and analysed means are identical and equal the average of
  the four channel means. No zero/saturated pixels; largest raw pixel is 2749 DN.
  These checks do not establish hardware-frame completeness or SDK latency.
- GUI delivery age: median 0.362 ms, 95th percentile 28.06 ms, maximum 84.58 ms.
  Largest receipt interval is 9.00 ms. Intensity/delivery-age correlation is
  -0.018. No evidence here for a multi-second GUI queue causing the slow trace.
- First/last ten-second raw means are 499.834/531.022 DN, a 6.24% increase.
  Whole-record P2-P98 is 492.635-542.059 DN, a span of 9.61% of mean brightness.
  The trace contains irregular drift and excursions rather than a stable sine.
- The Hann-windowed, linearly detrended periodogram has a local peak near
  0.1833 Hz (5.45 s), but stronger broad low-frequency power near 0.05 Hz.
  Fitting a fixed 0.1833 Hz sine plus a linear trend to 50 ms means removes only
  2.8% of the variance remaining after a linear trend alone. A 60-second trace
  cannot establish a stable 20-second cycle from the lower-frequency peak.
- All four channels largely move together (raw pairwise correlations 0.64-0.89),
  though their relative contributions also change. This is not solely intensity
  exchanging between polarization channels. Smaller peaks near 10.12 Hz and
  harmonics are present; their physical origin is not identified.

Analysis used NumPy and SciPy on the project `.venv`, without opening a camera.
Frame-order spectra use the measured mean receipt rate, not sensor timestamps.
The 0.05/0.1833 Hz powers agree within 1% with independent 50 ms host-time-bin
spectra, so these slow features are not an artefact of the chosen timing basis.
Saved overview:
[runs/intensity_diagnostic_20261001-190946_analysis.png](runs/intensity_diagnostic_20261001-190946_analysis.png).
Original data and generated plot remain in ignored `runs/`; these numerical
findings persist in the notes when committed, but the files do not travel in Git.

Conclusion: the slow change is in recorded native intensities, not merely preview
stretching/background subtraction or multi-second GUI backlog. This does not
prove an optical/mechanical cause or exclude acquisition effects before receipt.
Slide thickness/interfaces, focus or lateral drift, illumination, drive response
and aliasing remain possibilities. One ROI-statistics recording cannot distinguish
them or determine higher/lower Vpp. The earlier 830/829.8 Hz alias example does
not match the actual camera rate: a pure 830 Hz intensity fundamental would alias
near 31.21 Hz here, not 0.2 Hz. Actual drive frequency/harmonics remain unknown.

The user may return on 2026-10-02 with the actual experimental sample. Start with
another stationary-knob 60-second recording and document the known drive setup.
An otherwise matched drive-disabled/drive-enabled comparison, using the established
lab procedure and leaving camera settings/Z unchanged, can test drive dependence;
the generator's minimum amplitude dial is not OFF. If the pattern persists, a
small permitted FPS change at fixed exposure/drive can test sampling dependence.
Do not diagnose a thick slide or adjust voltage direction from this file alone.
No application code or hardware settings changed during this analysis.

## Reference Lab Trial (Skipped)

Retained for later validation; the user chose not to run this general pass now.

- Close other camera applications. Start with a short capture, not the full experiment.
- Confirm preview, exposure/gain readback, magnifier position, and saving to the
  intended disk. Do not enable automated multi-spot capture yet.
- Record whether background subtraction was requested AND actually applied;
  preserve the background file and its acquisition settings alongside the trial.
- Reopen the saved stack; compare frame count, FPS, native intensity range, ROI,
  and XY traces. Keep the original recording and JSON unmodified for debugging.
- Record each result/error here or in the rewrite notes. A passing camera-free
  suite cannot establish hardware timing, absence of lost frames, or scientific
  validity. Do not proceed to tomorrow's long experiment on startup checks alone.

  ## Interference And Background Order

  Hugh's supplied handover (9.9.26), read on 2026-10-01, specifies:

  1. Start with gain=1 and **Subtract background OFF**. Start live view and select
    the magnifier region before starting the intensity analyser.
  2. Use the un-subtracted intensity trace for the interference-suppression check.
    The handover calls for exposure `1/f0` with the applied stage oscillation and
    a compatible frame rate. For its 830 Hz example that is about 1.20 ms exposure,
    for initial tuning; check the actual generator frequency and camera readback.
  3. User clarification on 2026-10-01: tune the drive voltage at 1.2 ms first,
    then switch exposure to 0.6 ms and set drive voltage to twice that tuned value.
    This is Hugh's reported faster-acquisition procedure, not an automated action
    in the application. Verify fringe suppression again under the final settings.
  4. Then capture a background for each sample with oscillation on, recording that
    state in metadata and moving the stage in x/y during the minimum projection.
    Match exposure, gain and pixel format to the intended measurement.
  5. Enable subtraction for subsequent corrected measurements. Preserve the profile
    and settings. If settings change, check whether a new background is required.

  Terminology remains to be confirmed: at fixed frequency, doubling voltage changes
  the drive amplitude, not the temporal period. At 830 Hz, 1.2 ms spans about one
  mechanical oscillation and 0.6 ms about half. "Two periods" may mean optical
  fringe periods swept by the larger displacement, not two mechanical cycles
  inside the exposure. Do not assume exact cancellation from voltage scaling
  alone; it depends on waveform, actual motion and exposure timing.

  Defaults remain 1.2 ms as requested; no automatic voltage/exposure coupling was
  added. Live preview, magnifier analysis and recording have separate exposure
  fields. Set the relevant fields to 0.6 ms and stop/restart the analyser to apply
  its capture exposure. Live-view Apply alone does not update analyser settings.
  Capture the background at the final measurement settings, not only at the
  earlier 1.2 ms tuning setting. Log actual drive frequency and voltage separately;
  the sound/vibration checkbox is not a record of those numeric settings.

  Newly launched windows now default to subtraction OFF. In an older open window, for the initial
  interference check, stop analysing, untick the **main Live video** subtraction
  checkbox, and restart. The separate save-with-subtraction option does not control
  this analyser. A missing background causes repeated searches, not a requirement
  to capture one first; a captured profile also incurs the current lookup overhead.

  Static background subtraction is not the same as coherent-interference suppression.
  It cannot remove a phase-varying interference term, and zero-clipping after
  subtraction can distort the observed fringe amplitude. Tune interference first.
  The handover says the background capture is three seconds; this version's capture
  routine uses six seconds. The later defaults/transition changes above do not
  modify background capture or correction math.

## Equipment Photo Check: 2026-10-01

Latest user clarification: the equipment was recently brought back into use and
the old inspection label was not updated. This is the user's reported context,
not an independently verified electrical inspection. Have the owner regularize
the label/status; the photograph alone is not evidence of a hardware fault.
The user also reports switching previously OFF servo switches ON. Hugh's verified
handover explicitly specifies the driven Z channel open-loop, servo OFF, making
the changed servo mode a concrete troubleshooting lead.

The user reported failure to suppress interference and supplied two HEIC photos.
They show a TTi TG120 function generator and a PI piezo amplifier/servo controller.
**The PI controller has a yellow "DO NOT USE AFTER" label with an apparent 2023
date. Stop the test using the lab's normal shutdown procedure and ask the equipment
owner/technician to confirm current clearance before operating or adjusting it
further.** The photograph does not establish the present inspection status or a
hardware fault. Do not remove/bypass the label or assume the unit is cleared.

Only after clearance, the relevant checks are the actual generator frequency,
waveform, voltage and offset at the controller input, and the intended Z servo
mode from the approved setup procedure. The photo shows the 1 kHz frequency range,
but the analog dial does not establish exactly 830 Hz. The output has two -20 dB
attenuator switches; knob position alone does not establish delivered voltage.
Do not change attenuation to obtain a larger signal without confirming limits.
The Z servo switch position and full cable route cannot be established reliably
from these views. Have a qualified operator verify them rather than guessing.

Also distinguish a visual scaling effect from failed suppression: the intensity
plot rescales its Y axis each refresh. Compare the numeric `p98-p2` intensity span
at the same exposure, gain, ROI and comparable slow Z scan; a smaller fluctuation
can still fill the plot. The rolling ten-second window retains older data, and
the previously measured processing backlog can delay updates. No hardware or
application settings were changed during this photo inspection.

### Servo, Frequency And Amplitude Checks

- Servo ON closes the position-feedback loop for that channel; OFF leaves
  open-loop drive, not an unpowered actuator. Changing this mode changes the
  response to the external generator, including high-frequency behavior. It
  does not universally mean the servo simply cancels all commanded motion.
- The photograph suggests CH3 is Z and has the external control cable, but verify
  axis mapping and routing locally. Hugh specifies Z servo OFF for this routine;
  do not infer that all three channels should be switched together. Reduce or
  disable AC drive using the approved procedure before a mode change; the stage
  position can jump. Leave ZERO trims and amplifier offsets alone while identifying
  the setup. Do not reconnect high-voltage piezo cables under drive.
- Verify generator frequency with a suitable oscilloscope or frequency counter
  on the TG120 low-voltage MAIN OUT / controller input signal. Use a correctly
  rated high-impedance probe/input and proper grounding; get lab help if unsure.
  Never connect a normal scope/counter to the PI PZT output (marked up to 120 V).
  A 50-ohm termination changes the generator loading/amplitude; do not introduce
  one inadvertently into the established drive circuit.
- Use the scope's frequency measurement or measure adjacent triangle peaks:
  `f = 1/T`. 833.33 Hz gives 1.200 ms; 830 Hz gives 1.205 ms. Around 0.2 ms/div
  displays roughly six divisions per cycle. On the TG120's 1 kHz range the
  frequency multiplier is about 0.83, but that analog setting is approximate.
  Do not estimate the drive frequency from the exposure-averaged camera trace:
  it can suppress or alias the oscillation.
- Approximately 830 Hz was Hugh's observed stage resonance, not a universal
  cancellation frequency. Prefer the verified resonance/known-good setup and
  match the initial exposure to its measured period. Electrical waveform
  verification does not establish the actual mechanical displacement waveform.
- Confirm triangle waveform, normal symmetric operation, established attenuation,
  and permitted offset/amplitude. Then hold frequency/waveform/offset fixed while
  tuning amplitude. A -20 dB attenuation change changes voltage by a factor of ten;
  maximum knob position is neither a displacement measurement nor a safe target.
- At gain 1 and background OFF, use the magnifier analyser's own exposure control
  for the initial 1.2 ms (or measured-period) exposure. With AC drive absent,
  perform a small slow Z scan around focus to identify fringe maxima/minima.
  Hugh's rough step starts at a fringe maximum and uses the lowest drive amplitude
  that brings the average intensity to the fringe midpoint. His fine step adjusts
  amplitude while repeating the small slow Z scan to minimize fringe contrast.
  Stay within the established input, actuator and travel limits.
- More amplitude is not monotonically better: residual coherent contrast can
  have minima and rise again. Larger motion can also modulate focus and brightness.
  Cancellation targets the rapid fringe modulation, not all intensity changes
  across a large defocus scan or measurement noise. Use axis values and `p98-p2`,
  with comparable Z scans and time for the rolling window/lag to settle.
- Only after establishing suppression at 1.2 ms, test the user's reported 0.6 ms
  / twice-tuned-voltage procedure within approved limits and verify it again.
  It is not a guaranteed cancellation rule. No settings were changed by this note.

### TG120 Amplitude Scale Clarification

Verified against the manufacturer's
[TG120 manual, Issue 9](https://resources.aimtti.com/manuals/TG120_Instruction_Manual-Iss9.pdf),
Specification and Operation sections, on 2026-10-01:

- **AMPLITUDE / Vpk-pk is the correct continuous amplitude adjustment.** The
  attenuation switches select its output range. Neither Symmetry nor DC Offset
  is another amplitude control: symmetry changes rise/fall durations and offset
  changes the centre voltage. For the ordinary triangle, symmetry control should
  be disabled (the photograph appears to show OFF). Keep the established offset;
  do not use it to seek cancellation. Zero offset is the centre detent, not a
  recommendation to move the stage bias without checking the setup.
- The published output ranges into a high-impedance load are 1-20 Vpp at 0 dB,
  0.1-2 Vpp at -20 dB, and 0.01-0.2 Vpp at -40 dB. Into 50 ohms these values halve.
  The photographed dial's low marking is 2; neither that marking nor the minimum
  stop is an output-off setting. The earlier shorthand "increase from zero" must
  not be read as "the minimum amplitude knob setting is zero volts".
- The manual does not provide an amplitude-knob calibration accuracy or guarantee
  a linear angle-to-voltage mapping. The triangle-wave linearity specification
  describes waveform shape, not knob calibration. Do not double the angle turned
  or assume a position halfway across the dial is half the measured output.
- For the reported doubling procedure, measure the tuned voltage peak-to-peak at
  the low-voltage MAIN OUT/controller-input signal with the established load and
  correctly rated high-impedance probing. Keep waveform, frequency, offset and
  attenuation fixed, then target twice that measured Vpp only within approved
  controller/actuator limits. A -20 dB switch is a factor of ten, not two; do not
  flip ranges live to chase a doubled reading beyond the current knob range.
- The user reports much louder stage sound after returning servo OFF, resembling
  Hugh's previous operation. This supports the servo-response explanation, but
  sound does not verify safe displacement, drive voltage or fringe suppression.

No hardware settings or application code changed during this clarification.

### Software-Assisted Tuning Status

Implemented later on 2026-10-01 with a default 10x target; see
[Guided Interference Tuning](#guided-interference-tuning) above. The original
proposal and subsequent implementation evidence are retained in
[REWRITE_NOTES.md](REWRITE_NOTES.md). Physical suppression remains to be tested.

## Notebook Reference For Stage 3

The supplied unit-sphere notebook is an APD/TDMS analysis reference, not yet part
of this repository's runtime. It can inform projection/validity design, but its
APD inverse matrix and fitted gains must not be assumed to calibrate the camera.
Inspect the relevant cells before reusing formulas: physical azimuth is half the
anisotropy angle, and invalid samples must not silently become valid angles.
No notebook code, full-stack analysis, or export movie is executed during Stage 1.