# Rewrite Notes

## Goal: Performance First

Optimize reliable camera ROI/FPS transitions and buffer save/drain throughput.
Clarity is useful, but performance takes precedence over structural tidiness.
Do not reorganize the application merely to make its file layout look cleaner.
Correctness, frame ownership, and recording integrity must not be traded away.

Planned workflow: select several spots; set the camera ROI for one spot; record
about 10 seconds at high FPS; switch to the next as quickly and reliably as
possible; repeat after the last spot. For example, ten samples each receive
successive recording slices instead of dedicating the whole experiment to one
sample. A five-second recording followed by a five-second reconfiguration would
defeat the purpose. The spot cycler is a new feature, not implemented in v3.

No hardware transition benchmark or cycler implementation was made in this step.

## Lab Session: 2026-10-01

### Immediate Goal And Stage Boundaries

An experiment is scheduled for 2026-10-02, and Hugh's older application is failing
in the lab. The user wants a usable version here today to avoid fixing the same
behavior in two installations; the old installation remains a fallback. Work in
stages, beginning with launch and practical acquisition checks, not a broad rewrite.
The performance-first cycler objective above remains the longer-term goal.

The current stage is **startup complete; general Stage 2 checks skipped by the
user; magnifier start/stop and requested defaults fixed in code and tested with
fake cameras; guided 10x interference tuner and analyser performance fixes
implemented; independent diagnostic intensity recording added after the user
reported slow modulation with stationary Z; first real diagnostic reviewed and
slow raw-intensity drift confirmed, but physical cause still unknown**.
Interference investigation is paused until tomorrow. The current implementation
task is now complete: **intensity-selected spot analysis, stack condition metadata,
and all-rod sphere/projection/density with three averaging choices**; 92 camera-free
tests pass. The new scientific workflow still needs real-sample validation.
[LAB_SETUP.md](LAB_SETUP.md) is the function/workflow checklist, machine setup
guide, and place to record lab results. Do not treat startup as experiment readiness.

### Stage 1 Completed

- Added [Launch-Polarcam.cmd](Launch-Polarcam.cmd), using this checkout's `.venv`
  and default `runs/` data root. Errors leave the diagnostic console open. Command
  arguments allow an alternative `--data-dir`; no camera opens automatically.
- Added [Install-DesktopShortcut.ps1](Install-DesktopShortcut.ps1) and installed
  **Polarcam Lab** on this user's desktop. The installer refuses to overwrite an
  existing shortcut and supports an explicit data directory.
- Found an actual startup blocker: Tcl could not find `init.tcl` even though it
  was installed under `C:\Program Files\Python313\tcl`. Updated `polarcam.cli`
  to pass matching Tcl/Tk paths to the child process on Windows, respecting
  explicit overrides. No system environment or scientific processing was changed.
- Added real Tk GUI startup/shutdown coverage and two launcher environment tests.
  All 36 tests pass in the current lab `.venv` (Python 3.13.0), with no skips.
- Verified imports of the application dependencies and IDS Python bindings.
  No dependency upgrade was performed during this stage.
- Opened the real application via the desktop shortcut and verified its window
  is responding. Acquisition was not started automatically.

### Latest: Intensity-Selected Rods And Orientation Distribution

The user deferred interference diagnosis until 2026-10-02 and requested a workflow
mirroring Spot analysis, but selecting intensity rather than temporal variation,
saving frame stacks and recording background/interference conditions. Their
supervisor wants averaging before sphere projection/density and is undecided
between averaging I(1-4), anisotropy X/Y or unit-sphere Cartesian coordinates.

Implemented in the existing Spot analysis tab, preserving Time variation mode:

- Detection selector: full-stack native mean -> average four-polarizer cells ->
  existing DoG/area/edge detector. Stationary rods survive; candidates are ordered
  by brightness. No motion/hollowness filters or automatic camera inspection in
  Intensity mode. Update analysis replays the stack and supports mode switching.
- Fetch frames reuses existing native NPY stack capture and JSON readbacks, adding
  snapshotted manual Sample/Background purpose, interference On/Off/Unknown, note
  and detector choice. Requested/applied background correction stays separate.
  Background role labels a stack; it does not replace the stored correction profile.
- Per-frame channel means use raw crop + sensor origin parity; NPY measurement
  avoids display rescaling and full-stack copies. Source metadata/phase-marker
  origin survives reload, with explicit full-frame assumption if absent.
- All-rod distribution uses one mean per rod by default (All frames), or explicit
  frame windows. Three choices: 3D Cartesian (default), Anisotropy X/Y, Four
  intensities. Existing water/glycerol50 theta models are selected explicitly.
- Unit-sphere, stereographic scatter and projected-plane histogram density views;
  atomic NPZ export retains raw channel traces, geometry, means, valid counts,
  resultants, windows, densities and complete analysis/source metadata.
- Added isolated intensity_spots.py for numerical reducers; actual detector owner
  is Detection_alg_offline.py, imported as detect_spinners by the GUI. The similarly
  named detect_spinners.py is not the GUI's active detector.

Do not overclaim the science: these theta models are not calibrated for the new
sample. Modulo-180 azimuth is unwrapped on its doubled phase before Cartesian
averaging, then the representative is folded into phi [0,180), z >= 0. This handles
the branch boundary but does not uniquely recover 3D polarity/tilt sign or arbitrary
motion across gaps. The density is probability per projected area, NOT per solid
angle. Rod detection bias, background contamination and fixed-crop drift remain.
See [LAB_SETUP.md](LAB_SETUP.md#intensity-based-spot-analysis) for exact workflow,
formulas, limits and export schema. No deferred global science model was changed.

Validation: 92 camera-free tests pass, including end-to-end native synthetic NPY
replay/export, static spots, all ROI parities, averaging choices, boundary handling,
invalid data, density normalization, source switching, decode errors/cancellation.
96 stationary synthetic rods in a 5x1024x1536 stack were found in about 0.10 s
excluding plotting; all methods produced normalized distributions. Main Tk UI
checked at 1200x800, distribution at 860x540. No hardware was operated or user
process restarted. Changes remain local, uncommitted and unpushed.

Next: restart safely, choose Detection = Intensity, capture/replay the actual
sample with explicit condition notes, inspect candidates and compare the three
averaging choices on the same stack. Preserve stack NPY + JSON + analysis NPZ.
Resume the earlier interference diagnosis only when the user is ready; do not
replace it with another hard-to-repeat manual Z-sweep score.

### Earlier: First Real Diagnostic Reviewed

Reviewed the 2026-10-01 stationary-knob recording with 47,927 samples over 60 s.
See [LAB_SETUP.md](LAB_SETUP.md#first-diagnostic-result-2026-10-01) for numerical
findings, analysis method, local data/plot paths and interpretation limits.
Actual readbacks: 798.791 fps, 1194.270 us, gains 1, background OFF. Raw and
analysed means agree; no clipping, setting changes, error events or multi-second
GUI backlog. These are recording checks, not general hardware acceptance.

The raw signal has a 6.24% first-to-last ten-second mean increase, irregular
excursions and largely shared changes across polarization channels. A local
0.1833 Hz (5.45 s) peak exists, but broad slower drift dominates. A fixed sine
at that frequency explains only 2.8% of linearly detrended 50 ms mean variance.
Host-time-bin and frame-order low-frequency spectra agree. No unique physical
cause or voltage direction follows; slide thickness is unverified. Generator
frequency/Vpp were not measured, so the earlier numerical alias example is not
this recording's diagnosis. No application code or equipment was changed.

The user may return on 2026-10-02 with the actual experimental sample. Next:
repeat stationary recording with known condition notes; compare drive disabled/
enabled only through established lab procedure with camera/Z held fixed. Later
consider a small FPS-only change to test aliasing. Do not insist on another
manual Z-sweep score. Data and overview plot are in ignored runs/; preserve or
transfer those separately for cross-machine reanalysis. Current work is local,
not committed or pushed in this analysis step.

### Earlier: Independent Diagnostic Intensity Recording

The user finds manual-sweep scoring hard to interpret and reports large, roughly
five-second sinusoidal intensity variation at some drive amplitudes even without
deliberately changing Z. They requested temporary intensity recordings to separate
manual-motion effects, equipment behavior and script faults. No voltage direction
or physical cause can be inferred from that observation alone.

Implemented an independent **Intensity diagnostic** section in the tuning window:
60 seconds by default (10-120 s), Record diagnostic / Stop and save / Mark event /
Retry save. It requires only a running analyser, not a sweep checkbox, reference,
10x score, background-OFF state or complete readbacks. The tuning workflow stays
available but is not required for diagnosis. See
[LAB_SETUP.md](LAB_SETUP.md#intensity-diagnostic-recording) for user steps and format.

- Capture raw native ROI-mean and analysed/corrected intensity, all four RAW
  polarization-channel means with sensor+crop parity, min/max, zero and 4095-limit
  fractions, host receipt/processing times, and analyser frame index on each frame.
- A preallocated 500,000-row float64 buffer (about 52 MB) prevents unbounded
  diagnostic memory growth. No per-frame disk I/O; sample cap stops explicitly.
  In-window delayed, flat and clipped data are retained, not rejected by tuning
  quality screens. No-frame captures still save metadata and an empty array.
- The user chooses the NPZ path before capture starts. Auto-stop, manual stop,
  analyser stop and tuning-window close save the captured rows plus JSON metadata
  atomically. A save failure retains data, prevents replacement/window teardown
  and provides Retry save. This is not crash-safe disk streaming; forced termination
  still loses unsaved RAM and final writing can briefly block the UI.
- Metadata contains initial settings, frame shape, raw ROI and timestamped camera
  ROI/timing/gain updates, background changes, camera/callback errors, and manual
  labels/drive notes. Generator voltage and physical Z are not read by the program.
  A host timestamp fallback is flagged if an emitter omits its receipt timestamp.
- Host receipt times begin before GUI delivery, but after SDK buffer retrieval.
  They are not sensor exposure timestamps or hardware frame IDs. Time-window
  boundaries and the final 250 ms delivery grace do not guarantee all sensor frames
  were captured. No raw image stack or spatial tracking was added in this step.
- All 75 camera-free tests pass; diagnostic UI checked at 730x640. Synthetic frame
  generation plus diagnostic callbacks measured about 120 ms per 1640 samples,
  and all 8200 test rows exported. No real camera was opened, restarted or adjusted.

Explained drive physics: frequency sets voltage-command repetition, Vpp sets
voltage excursion and (through the amplifier/piezo/stage transfer response)
displacement/optical-phase excursion. Resonance, servo mode and limits matter;
increasing Vpp does not monotonically improve cancellation or establish displacement.

The five-second wave could be residual periodic intensity aliasing/beating against
the actual frame rate: e.g. 830 Hz minus 829.8 fps gives 0.2 Hz, but these are an
illustrative pair, not measurements. Exposure averaging, harmonics, genuine focus/
illumination drift and queued delivery also need consideration. The current task
does not implement a spectral diagnosis or a higher/lower-voltage recommendation.

The first stationary 60-second recording has now been inspected; see the latest
result above. Continue with controlled comparisons, recording actual readbacks;
do not use the unstable manual Z sweep as the primary comparison until this
signal is understood.

### Earlier: Guided 10x Tuner Implemented

The user approved software-assisted tuning and suggested a 10x benchmark based
on the handover's approximate 20x/6x reduction expectations. Implemented a
reference/test trial workflow in the existing analyser, without generator control
or modifying optical reconstruction. Full user steps and metric definitions are
in [LAB_SETUP.md](LAB_SETUP.md#guided-interference-tuning).

- **Interference tuning** opens a small window from the main top bar or analyser.
  It includes analyser Start/Stop, manual trial/Vpp labels and fixed-drive notes,
  duration (default 5 s), target (default 10x), comparable-sweep acknowledgement,
  reference/test/cancel actions, live ratios, completed trial history and export.
- The first 0.5 s of each trial is settling time. Only subsequent frames within
  that trial window count. Change amplitude BETWEEN trials; repeat the same small
  Z sweep while each trial records. Software does not know the physical Z or Vpp.
- Reduction is reference P98-P2 divided by test P98-P2, in native ROI-mean counts.
  Also show the ratio of (span/mean). Target indication requires both >=10 (or the
  selected target), a completed capture and no quality warnings. The handover
  does not establish that its quoted reduction used this exact percentile metric.
- Native uint16/Mono12 means now feed this analyser; 8-bit conversion is preview
  only. This is scoped to intensity tuning, not the deferred general NPY grayscale
  conversion or XY/theta calculations. Old 8-bit count readouts are not numerically
  interchangeable with the new native DN values.
- Actual ROI, exposure, FPS and analog/digital gain readbacks are required. Reference
  and test must use matching settings and duration, with background subtraction
  OFF. Stop/restart, metadata changes, subtraction changes or drive-note edits
  invalidate the current reference. Completed trials remain for export.
- Flat/nonpositive data, any 0 or >=4095 ROI pixel, >2x mean-brightness change,
  >250 ms GUI delivery age, nonfinite data or incomplete captures cannot meet the
  target. Complete windows need >=32 samples, first/last coverage within 250 ms,
  and >=90% of the expected count. No frame for 500 ms invalidates the comparison.
  These are documented conservative screens, not calibrated optical validity.
- The backend adds a `frame_timed` signal containing the existing owned array and
  host receipt time. A single queued relay publishes both it and the unchanged
  original array-only `frame` signal. Existing recorders keep their API. Tests
  exercise identity, timing and queued delivery; no IDS acquisition code was
  executed against real hardware. Host times do not measure sensor exposure or
  delay before the SDK returns a buffer.
- Background discovery is cached for the analyser, including a missing profile;
  ROI crop reuse avoids per-frame filesystem calls. Selecting a new active profile,
  toggling correction or restarting refreshes it. Out-of-band file edits are not
  polled. Other live-view background lookup paths were not broadly rewritten.
- Reuse the Matplotlib figure, axes, line and canvas; avoid repeated tight-layout
  calls. A display-only min/max envelope retains peaks without thinning scores or
  saved samples. With a reference, plot limits stay fixed to its range.
- Added sidebar scrolling after a real layout check showed the old fixed-height
  live rail hid the analyser controls at a normal 1600x1000 window. A top-bar entry
  makes the tuner reachable without a full interface redesign.
- Save trials writes JSON with native mean traces (not full image stacks), host
  times, actual camera settings, labels, reference IDs and results. Up to 40 complete
  trials are retained; Clear/exit discard unsaved in-memory records. Cancel/closing
  the tuner discards an incomplete trial. A new reference is needed at 0.6 ms.

After-change wall-clock microbenchmarks: five batches of 1640 synthetic 14x256
frames, reducing a 14x14 magnifier window, took medians of 13.48 ms with background
off, 13.86 ms with a missing profile, and 32.10 ms with a cached profile. The earlier
baseline was 22.09 / 1322.16 / 2048.03 ms. Reused plotting with 16400 points measured
30.01 ms median / 32.87 ms p95 over 20 warmed calls, versus 68.77 / 79.94 ms before.
No real-camera or closed-loop physical-tuning speed claim follows from these probes.

Verification: 66 camera-free tests cover known reductions, dimming/flat/clipping
rejection, settling/stale/incomplete windows, metadata invalidation, native counts,
plot/background reuse, GUI actions, export and queued frame relay. The tuner was
visually checked with synthetic 20x data at 730x520, and sidebar reachability is
tested. No running camera session was interrupted and no external hardware settings
were changed. Local edits are not automatically committed/pushed; restart the
application after stopping acquisition to use them.

### GUI Provenance And Gain Default

- Compared the main GUI with the local source under
  `C:\Polarcam Software\Polarcam_v3\polarcam_live` on 2026-10-01. All six
  screen-building methods (`_build_ui` and the five tab builders) have identical
  Python syntax trees. The visible layout is inherited, not translated or rebuilt.
- Both main GUI classes have 217 methods; 14 method bodies differ, including
  initialization, channel fixes, recording reload, and shutdown guards. This is
  a method-level comparison, not a claim that all remaining behavior is validated.
- The work so far is importing and stabilizing Hugh's application. Keeping the
  working GUI avoids a cosmetic rebuild; capture performance and scientific
  consistency are still the substantive follow-up work.
- At the user's request, changed the main live-view default gain from 20 to 1.0.
  Capture gain defaults were already 1.0. The GUI startup test checks the new
  value. Existing open windows and camera settings were not altered; relaunch
  after stopping acquisition, or set Gain to 1 and Apply in the current window.

### Magnifier Transition And Default Fixes

Later on 2026-10-01, the user requested exposure 1.2 ms, subtraction off, and an
automatic return to live preview when stopping magnifier intensity analysis.

- All main live/capture exposure controls now start at 1.2 ms, including magnifier
  analysis, fetch frames, spot recording and stationary capture. Gain stays 1.0.
  All main live/capture background-subtraction controls now start OFF, including
  the live processing flag. Standalone tool defaults were not independently changed.
- Requested FPS fields remain unchanged; 1.2 ms allows at most about 833 FPS
  before other camera limits. Verify actual readbacks; requesting 1640/2000 does
  not override exposure. Frame-count captures may consequently take longer.
- Starting the intensity analyser enables the ordinary Magnifier checkbox and
  retains the selected sensor centre. Previously the analyser could show a ROI
  image even with that checkbox off, while normal live view would blank it.
- The Stop button now calls `_on_stop_live_intensity_analysis`: it stops/closes
  the ROI controller, clears pending ROI previews, then resumes the full-frame
  stream with the same centre and magnifier enabled. No previously detected spot
  can overwrite that centre during this return-to-live action.
- Internal stops during tab changes, capture handoffs and window shutdown still
  use `_stop_live_intensity_analysis` without restarting a stream. Live start is
  ignored while intensity analysis owns the camera; the Stop button does not
  start preview during another capture. Synchronous analyser-start exceptions
  clean up and restore a previously running full-frame stream.
- Added a post-Qt-event stop check and cancellation of the UI pump before Tk
  destruction. This prevents a stopped analyser from rescheduling its timer and
  removes a stale-timer warning exposed by the full GUI test run.
- All 46 camera-free tests pass (ten new lifecycle tests). They invoke real Tk
  controls with fake cameras, check close-before-open order, repeated cycles,
  centre preservation and magnifier image updates. No real camera session was
  stopped, restarted or reconfigured. Hardware errors reported only by signals,
  native SDK hangs and overall acquisition latency are not resolved by this step.
- These are lifecycle/default fixes, not the background-cache or plot-reuse
  optimization discussed below. Relaunch the app after stopping acquisition;
  the already-open process retains the previous code and defaults.

### Stage 2: Existing Workflows (Skipped)

On 2026-10-01, after confirming that the GUI is inherited from Hugh's application,
the user chose to skip this general acceptance stage and defer any broad rebuild.
The checklist remains a reference, not evidence of passed hardware tests. The
current request is to investigate the lag in the magnifier intensity analyser.

Check live preview, actual exposure/gain settings, magnifier selection and zoom to
found spots; background selection/capture, subtraction and saved profile provenance;
full-frame stack recording during spot finding; short single-spot recording;
consistent XY maps in detection and recording. Verify which workflows actually
auto-adjust exposure/FPS instead of assuming requested values were applied.

The interference/intensity-trace control **Start analysing magnified region** is
reported to lag in Hugh's application. Measure the corresponding path here before
changing it. It currently combines per-frame callbacks with Qt event pumping and
periodic plotting. Camera-free measurements below isolate substantial avoidable
costs; no performance fix for that control has been applied yet.

### Magnifier Intensity Lag: Investigation

Investigated `_start_live_intensity_analysis`, `_live_intensity_on_frame`,
`_live_intensity_tick`, `_live_render_intensity_plot`, the background loader, and
the IDS worker-to-facade frame signal. No camera was opened, no live session was
interrupted, and no application code was changed for this investigation.

The button stops/closes the full-field preview camera and opens another controller
for the small ROI, by default requesting 1640 FPS. Camera open/configuration can
explain an initial pause, but the ongoing lag has distinct per-frame/UI causes.
This mode computes mean intensity versus time; it is not recording/dumping a stack
or running spot detection/theta reconstruction on each frame.

**Measured on the current lab PC, Python 3.13.0, synthetic data:**

| Operation | Measurement | Interpretation |
| --- | --- | --- |
| 1640 callbacks, background disabled | 22.09 ms median (13.47 us/frame) | About 2.2% of one second of UI time at 1640 FPS |
| 1640 callbacks, background enabled but absent | 1322.16 ms median (806.20 us/frame) | Searching missing profile paths alone can prevent keeping up |
| 1640 callbacks, background enabled and cached | 2048.03 ms median (1248.80 us/frame) | The cache does not eliminate repeated filesystem work |
| Actual plot method, 164 samples | 59.26 ms median; 66.11 ms p95 | Figure/layout creation is expensive even for a short trace |
| Actual plot method, 16400 samples | 68.77 ms median; 79.94 ms p95 | Requested 10 refreshes/s would consume about 69% of the UI thread |

Frame timings used five batches of 1640 synthetic uint16 14x256 ROI frames,
reducing a 14x14 magnifier window. This shape is a probe, not a measurement of the
actual camera's current ROI. The temporary background was a 512x512 float32
array. Plot timings used two warmups then 20 actual calls with a hidden Tk label,
the normal Matplotlib/PIL/Tk conversion, and a synthetic 60 Hz signal plus noise.
These are wall-clock microbenchmarks under this machine's load, not hardware
throughput guarantees or measurements of the running application's delay.

**Isolated causes:**

1. Background path discovery is inside every intensity callback, even after the
  array has been cached. `_load_background_profile` calls
  `_find_background_profile_path` and compares resolved paths before returning
  cached data. One cached-profile callback made four `Path.resolve` calls and
  one `Path.exists` call, in addition to other path construction.
2. In a second controlled probe (five batches of 500 callbacks), the unchanged
  cached path took 1462.78 us/frame. Replacing only `_load_background_profile`
  with a cached-array return in the temporary benchmark process took 85.68
  us/frame. All subtraction, clipping, 8-bit conversion, mean calculation, and
  deque updates remained; the result stayed exactly 62 counts. This roughly
  17x callback reduction is evidence for caching, not an implemented app speedup.
3. The camera worker queues a full frame to the facade for every acquired frame.
  A synthetic signal probe with the actual facade/worker classes (never opened)
  delivered zero of 100 emitted frames before `processEvents`, then all 100 on
  the main/UI thread after pumping. The GUI's single-item preview queue is
  downstream of these callbacks; it does not bound this Qt event backlog.
4. Each plot refresh creates a new Figure, axes, labels, grid, line and canvas,
  runs `tight_layout`, draws all up-to-16400 points, converts to an image and
  constructs a new Tk image. This runs in the same Tk callback that pumps Qt
  events. The `after(5, ...)` is scheduled after the work, not a 5 ms guarantee.

At 1640 FPS the frame interval is about 0.61 ms. Background-enabled callbacks
alone exceeded that budget in these probes, before plotting. During a 69 ms plot
draw, roughly 113 new frames could arrive at that rate. This explains a plausible
growing lag, not merely a low display refresh rate. Actual arrival rate, queue
depth and capture timestamps have not been measured on the live camera. The
displayed time is currently frame-count/FPS, so it cannot establish display age.

**Recommended small fixes, not yet applied:** cache the selected background and
its ROI crop outside the frame callback, invalidating explicitly when the profile
or ROI changes; reuse the plot figure/canvas/artists and update only data; retain
all intensity samples but optionally render a per-screen-column min/max envelope
so high-frequency peaks remain visible. If needed after measurement, accumulate
samples away from the GUI and send bounded batches/latest preview to it. Do not
silently lower acquisition FPS, skip measurement frames, or change correction
math to make the display look faster. A full GUI rebuild is unnecessary for these
bottlenecks. Genuine SDK open/reconfiguration time is a separate follow-up.

### Handover Clarification: Interference Before Background

Read the user's supplied "Hugh Bowman handover 9.9.26.docx" from the desktop on
2026-10-01. The Word file itself is outside Git; the relevant workflow is preserved
here for another machine. This supersedes any assumption that the intensity
analyser requires a saved background before use.

- The handover explicitly starts with gain=1 and background subtraction OFF.
- In "INTERFERENCE BACKGROUND SUPPRESSION", the magnifier intensity analyser is
  used to observe fringes while varying z and tune their suppression. This comes
  before the separate background-measurement section.
- The handover specifies exposure equal to one period of the applied oscillation
  (`1/f0`), with a frame rate slow enough to allow that exposure. Its approximately
  830 Hz example corresponds to approximately 1.20 ms, not the former 0.6 ms default.
  Use the actual generator frequency; the camera FPS and exposure are separate
  quantities. No hardware setting was changed by this clarification.
- "BACKGROUND MEASURMENT" says a new background is needed for every sample:
  oscillation on, record that state in metadata, move the stage in x/y during the
  minimum-projection capture, then enable background subtraction. Check that the
  capture and measurement exposure/gains/pixel format match; retest if they change.
- The handover describes a three-second background capture; the current imported
  `_capture_background_profile_from_live_settings` uses six seconds and a temporal
  minimum. Selecting an existing background stack instead computes its mean.
  Preserve that distinction when testing or revising background provenance.

Interpretation: the initial fringe-suppression check should use un-subtracted
camera intensities. Subtracting a static intensity image cannot cancel the
phase-dependent coherent interference term; the physical oscillation/exposure
averaging is intended to address that. The later background subtraction removes
the remaining static contribution. The current subtraction also clips negative
values to zero, which can distort a fringe's apparent depth when a profile is
mismatched. These are different operations, not interchangeable corrections.

At the time of this investigation, "Subtract background" was ON by default, including for the
intensity analyser. Even when no background was selected in this session it looks
for an existing profile; if none exists it copies the frame unchanged and retries
the lookup on every subsequent frame. On this inspection no profile existed under
either `runs/` or `src/polarcam/live/`, the desktop launcher's default search roots.
This supports the missing-background lag scenario; it is not a direct measurement
of the live camera queue. Taking a profile first does not fix repeated path work.

For the immediate interference check, stop the analyser, untick the main Live
video **Subtract background** checkbox (not the separate save-with-subtraction
option), then restart the analyser on the selected magnifier region. Plot rebuilding
can still cause lag. No runtime or default-checkbox changes were made in this turn.

Subsequent update: the user requested subtraction OFF by default and exposure
1.2 ms. Those defaults and the analyser-to-live transition are now implemented;
see "Magnifier Transition And Default Fixes" above. The per-frame lookup overhead
still exists if subtraction is enabled; plot rebuilding has not been optimized.

### Two-Stage Exposure/Voltage Procedure: User Clarification

On 2026-10-01 the user clarified Hugh's intended procedure: first tune the stage
drive voltage using 1.2 ms exposure, then change to 0.6 ms exposure and twice the
tuned voltage for faster acquisition. Record this as a reported lab procedure,
not as a validated automatic cancellation rule. The user's description was that
the larger voltage covers "two periods instead of one".

At fixed generator frequency, amplitude does not change the mechanical period:
830 Hz gives approximately 1.205 ms per oscillation, so a 0.6 ms exposure covers
about half a mechanical cycle, not two. The intended periods may instead be
optical interference-fringe periods traversed with the larger stage displacement.
Confirm that interpretation with Hugh if needed; verify suppression at the final
exposure and amplitude rather than assuming linear voltage/displacement response
or phase-independent cancellation. Waveform and stage dynamics matter.

No application defaults or runtime settings changed for this clarification.
Keep 1.2 ms as the initial tuning default until the user requests otherwise.
Main live preview, magnifier analysis and recording exposures are independent;
the analyser reads the magnifier capture settings when started, not live Apply.
After tuning, set the intended acquisition controls to 0.6 ms and restart the
analyser to apply that exposure. Capture background at the final exposure/gain
and illumination/oscillation conditions. A background made at 1.2 ms must not be
assumed suitable for 0.6 ms without validation.

The application does not control or automatically double the external generator
voltage. The sound/vibration checkbox records only an on/off observation, not
frequency, waveform or amplitude. Record those numeric settings separately for
now; explicit tuning/acquisition presets and richer metadata are possible later
improvements, not implemented in this clarification.

### Equipment Photos: Interference Troubleshooting

On 2026-10-01, the user supplied IMG_6118.HEIC and IMG_6119.HEIC from the desktop
and reported inability to suppress interference. Windows codecs converted them
to temporary PNG viewing copies; originals and the running equipment were not
changed. The photos are local and not part of Git.

The second photo shows a PI piezo controller with a yellow "DO NOT USE AFTER"
label and an apparent 2023 date. Advised stopping the test via the lab's normal
shutdown procedure and asking the equipment owner/technician to confirm current
clearance before further operation or adjustments. This is not proof of a fault
or a determination of current inspection status; do not assume clearance.

The first photo shows a TTi TG120 generator on the 1 kHz range. Its analog dial
does not establish an exact frequency. Two -20 dB output attenuators are visible,
so amplitude-knob position is not a voltage measurement. Full wiring, actual Z
servo state, delivered voltage/offset and stage motion cannot be verified from
these photos. Once cleared, use approved lab checks for those quantities; do not
blindly increase amplitude, switch attenuation, or move stage screws to compensate.

Software observation to explain possible apparent non-cancellation: the analyser
Y axis is automatically rescaled from each displayed window's intensity extrema.
Use its numeric `p98-p2` span and axis values at fixed exposure/gain/ROI and
comparable Z scans, not apparent plot height alone. The ten-second rolling buffer
and known event backlog can delay visual feedback. Suppression is not expected to
remove all measurement noise. No runtime code change was made for this question.

### Equipment Follow-Up: Servo Mode And Amplitude Tuning

The user clarified that the equipment's old inspection label had not been updated
after it was brought back into use; do not treat the date alone as proof of a
fault. This remains user-reported context, not independent inspection clearance.
More importantly, the user reports turning servo switches ON when they had been
OFF on arrival. A fresh read of Hugh's handover confirms Z **servo OFF/open-loop**,
**triangle** drive and **approximately 830 Hz stage resonance** for this procedure.

Explained that servo ON is position feedback, not a piezo enable switch, and can
change the high-frequency transfer response. Verify which channel is Z (CH3 is
suggested by the photo, not established by tracing) before restoring the documented
mode using the lab's procedure with AC drive reduced/disabled. Do not switch every
axis or adjust calibration trims/offsets blindly.

Frequency verification should use the TG120's low-voltage output with a suitable
scope/counter, high-impedance loading, appropriate probe and grounding, not the
PI high-voltage PZT outputs or the aliased/exposure-averaged camera signal.
833.33 Hz is a 1.200 ms period; 830 Hz is 1.205 ms. Exact 833 Hz is not magic:
Hugh chose approximately 830 Hz for mechanical response, then matched exposure.

The user's expectation that maximum Vpk-pk should flatten all intensity changes
with Z is not the intended tuning rule. Hugh says to increase from zero to the
lowest amplitude giving the fringe midpoint from a starting maximum, then minimize
contrast during a small slow Z scan. Contrast minima are not necessarily monotonic
with amplitude, stage motion may not reproduce the electrical triangle near
resonance, and focus-envelope/noise changes remain. Record voltage/attenuation,
frequency, waveform, exposure and comparable numeric contrast, not graph height.

The expanded practical checklist is in [LAB_SETUP.md](LAB_SETUP.md), under
"Servo, Frequency And Amplitude Checks". No equipment settings or application
code changed for this discussion; the running camera session remains untouched.

### TG120 Knob Clarification And Servo Observation

The user reports that servo OFF made the stage substantially louder, matching
their memory of Hugh's setup. This is consistent with a changed mechanical
response, not independent proof of cancellation or safe motion.

Consulted the manufacturer's TG120 Issue 9 manual (Specification and Operation):
the amplitude control plus attenuators gives published high-impedance ranges
1-20 Vpp, 0.1-2 Vpp and 0.01-0.2 Vpp; 50-ohm termination halves them. The knob
minimum is not zero/output-off, despite the earlier handover wording "from zero".
The photo's low marking is 2. The manual does not establish an accurate linear
angle-to-amplitude calibration. The reported doubled-voltage acquisition step
should use twice the measured MAIN OUT Vpp under the same load/attenuation,
within the approved PI input and actuator limits, not twice the rotation of the
knob or a -20 dB switch (which changes voltage tenfold).

Clarified that Vpk-pk IS the continuous amplitude tuning control. Symmetry changes
triangle shape/duty and should not be used to change amplitude; normal triangle
operation uses symmetry OFF. DC offset sets the centre voltage and is not an
amplitude control either; preserve the established bias, do not adjust it blindly.
Detailed instructions and the manual link are in [LAB_SETUP.md](LAB_SETUP.md).
No application changes or equipment adjustments were made by the assistant.

### Proposed Guided Interference Tuner

Historical proposal, subsequently approved and implemented on 2026-10-01. See
"Latest: Guided 10x Tuner Implemented" above for the current implementation and
scope. The constraints below remain useful design context, not current status.

On 2026-10-01 the user reported that manual interference tuning was still not
working and asked whether software could find a suitable Vpp while they move Z
and change the amplitude knob. This is a design discussion; no tuner, generator
control or measurement change has been implemented.

The existing intensity analyser has frame-derived mean intensity and a rolling
ten-second trace, but no Z-position or generator-amplitude readback. It reports
`p98-p2` after 8-bit conversion and redraws with automatic vertical limits. There
is no integrated voltage-control/readback path. Consequently software can compare
observations and recommend a labelled trial, but cannot infer actual Vpp or
distinguish an incomplete Z sweep from successful suppression using this trace
alone. It must not claim a calibrated automatic optimum.

Recommended small next feature: a guided manual-knob, automatic-scoring mode in
the existing intensity panel, rather than another GUI or broad rebuild:

1. Keep ROI, exposure, gain, frequency, waveform and attenuation fixed. Record a
  reference with a comparable small Z sweep around focus and known drive state.
2. Enter the current Vpp (measured, or explicitly labelled a dial estimate) or a
  trial number. Hold it fixed while collecting a fresh short trial containing
  repeated back-and-forth Z sweeps through the same range. Do not include data
  from changing the amplitude in that trial.
3. Display robust intensity variation in native counts, brightness-normalized
  variation, mean level and clipping/low-signal warnings. Compare with the
  reference and show variation between repeat sweeps, not just one lowest value.
  Without Z readback, label this residual intensity modulation under the user's
  sweep, not an isolated measurement of coherent fringe contrast.
4. Collect a few candidate amplitudes within the established safe range, revisit
  the best candidates, and report the best tested setting or no clear improvement.
  Do not treat saturation, defocusing the rod, stopping Z, or less scan coverage
  as evidence of cancellation. Manual confirmation of comparable sweeps remains
  necessary without position measurements.
5. Save the trial traces, exposure/gain/ROI/FPS, manually supplied generator
  settings, labels and scores for replay. Retune/verify separately at 0.6 ms;
  do not compare its raw-count span directly against the 1.2 ms reference.

Avoid mixing continuous changes in Vpp and Z into a single rolling minimum:
focus, phase coverage, scan speed, noise and stale queued frames confound it.
First remove the documented plot/background-path lag or otherwise establish fresh
acquisition boundaries; the existing frame-count/FPS time axis cannot certify
data age after knob changes. Preserve native precision and all measurement samples;
display thinning/plot reuse must not change scores or scientific sampling. A
continuous score could be advisory, while retained best-setting comparisons use
fixed-amplitude trials. Fully automatic search would need a verified controllable
generator/DAQ and suitably known or controlled Z motion, not just software added
to this uninstrumented manual setup. A useful minimum is possible; exact zero
contrast and a unique voltage are not guaranteed by the optical/mechanical system.

### Stage 3: New Features Requested

- Alternative pure-intensity detection of all rods, alongside the existing
  rotating-rod detector. Tune against a real sample and retain a clear mode choice.
- Stereographic scatter/density of all detected rods to inspect the theta
  distribution for a possible nanohinge-related peak. Agree whether weighting is
  one value per rod or accumulated time samples; these answer different questions.
- Offline reprocessing from saved native frame stacks and metadata, so plots do
  not have to be captured during an experiment. Add an offline analysis surface
  for the new v3 workflow when implementing this stage, not the deleted legacy code.
- The supplied unit-sphere notebook is a local APD/TDMS reference in Downloads,
  not a tracked runtime dependency. No notebook was imported, modified, or run.
  Do not transfer its APD mixing matrix or fitted gains to the camera without
  calibration evidence. Check physical half-angle, optical model validity, and
  density/projection conventions before reuse. It is not a verified camera
  intensity-spot-finding implementation merely because it contains density fitting.

### Next Lab Action

The broad Stage 2 checklist was skipped by user decision, not completed. Relaunch
after stopping acquisition to use the new independent diagnostic recorder. Leave
Z and Vpp unchanged for a 60-second capture, save the NPZ and inspect raw intensity,
channel means, timing and metadata before further tuning. The reference/10x workflow
remains available, but is not required and cannot diagnose the slow wave by itself.
No hardware timing, frame-loss, or scientific-parity claim is made yet.

## Update: 2026-09-19

- The user checked with Hugh: the correct sensor convention is `[90,45;135,0]`;
  the inverted I0/I90 assignments were errors, not an intentional convention.
- Corrected the four identified mappings: live `_capture_auto_spot_series`, live
  `_spot_playback_windows`, and angle-analysis `_xy_phi_from_gray_bounds` and
  `_append_xy_from_frame`. Normal live reduction and shared reconstruction were
  already correct and are unchanged. Existing saved data is not rewritten.
- Added four channel regression tests, including mocked capture with all four
  ROI parities and signed playback differences before squaring. Added one preview
  contrast characterization test and extended the recorder test to verify native
  preview-file values. All 33 camera-free tests pass (29 live plus four launcher).
- Installed missing declared OpenCV and npTDMS dependencies with the editable
  project install in this checkout's `.venv`; the earlier lab environment did not
  travel with Git. Validation used this checkout's Python 3.12 environment.
- The user reported unusually bright selected-spot displays during a lab test.
  Source tracing and a synthetic example confirm automatic preview stretching;
  see the brightness section below. No experimental recording was supplied for
  inspection. Brightness behavior was investigated, not changed.
- Global ROI parity in `_xy_phi_stats_from_frame`, other standalone readers,
  intensity conversion, calibration, angular calculations, and camera performance
  remain separate follow-ups. No cycler or backend changes were made.

## Handoff: 2026-09-15

- Working branch: `testing/v3-rewrite`.
- Reviewed import baseline: `bb47c6490379f4c8ae90cbfe6d3e4315dad2792c`.
- Original v3 revision: `5fe41d5bdd9f089a441f30554888c0b4773e62a7`.
- Review covered all 12 standalone scripts and the backend/controller code.
- Changes in this step are limited to the authorized reliability fixes below.
- No changes to channel signs, intensity scaling, calibration coefficients,
  angular unwrapping, sphere fitting, or the IDS backend in this step.
- No real recordings or external calibration files were available for validation.
- The user is leaving the lab and will ask Hugh about the conventions and bring
  recordings later. Do not assume either the reply or data is already available.
- This file is the portable conversation context. Local Copilot chat history,
  editor state, and local memory are not required to continue on another machine.

| Review Finding | Decision / Status |
| --- | --- |
| 1. Inconsistent channel conventions | Resolved with Hugh and four inverted mappings corrected on 2026-09-19; global ROI parity remains a follow-up. |
| 2. Recording reload loses FPS / includes marker | Fixed in the main live and angle-analysis NPY readers. |
| 3. Frame-dependent intensity downscaling | Deferred at the user's request; explanation below. |
| 4. Missing theta assets / silent model fallback | Deferred; inspect the local calibration assets later. |
| 5. Angular unwrapping and sphere-axis fitting | Deferred; validate each calculation separately. |
| 6. Capture waits indefinitely after errors / stalls | Added immediate error propagation and an inactivity watchdog. |
| 7. Old analysis workers outlive source reset | Added guarded shutdown; source reset waits for actual worker exit. |

## Completed Work Before This Handoff

- Checked the repository-local `.venv` and confirmed VS Code selected it, not a
  shared parent environment. This lab environment runs Python 3.13.0.
- Updated `jupyter_server` to 2.21.1 and `tifffile` to 2026.9.15. Installed the
  missing v3 runtime packages OpenCV and npTDMS plus optional IDS Python bindings.
  Refreshed the editable project install so its dependency metadata matched v3.
- Verified runtime imports, Python compilation, and `pip check`. OpenCV is
  4.14.0.94, deliberately below the available 5.x release because the project
  requires `<5`; npTDMS is 1.11.0. The virtual environment itself is not in Git.
- Removed pre-v3 `legacy/`, older `offline/` and `reference_files/` material, and
  the obsolete roadmap and configuration. They remain recoverable from Git
  history; no history was rewritten and the `LICENSE` was retained.
- Committed and pushed the v3 baseline/cleanup as `bb47c64` on
  `origin/testing/v3-rewrite`. `main` was not changed.
- Read the full maintained script set and recorded the review findings here.
  Numerical examples below were checked with synthetic data, not lab recordings.
- Implemented only findings 2, 6, and 7, added the shared recording reader helpers
  and 24 regression tests, and updated the README/changelog. The four launcher
  tests also pass, for 28 passing camera-free tests in total.
- The reliability changes and this expanded handoff are being delivered together
  in the next commit on the same testing branch. Use
  `git log -1 --oneline -- REWRITE_NOTES.md` to identify that handoff commit.

## Resume On Another Machine

Use the existing repository checkout or clone
`https://github.com/haibaraaaaai/polarcam.git`. Check for local work before switching
branches; do not discard it or force a reset. For a clean checkout:

```powershell
git fetch origin
git switch testing/v3-rewrite
git pull --ff-only
```

Open this repository folder in VS Code. Read this file and
[README.md](README.md), then check `git status` in case work has continued since
this handoff. The code and notes travel with Git; `.venv`, recordings, background
profiles, and excluded calibration assets do not.

Use a separate `.venv` inside this repository. Select a supported Python with Tk
(Python >=3.12.10); Python 3.13.0 was used for the checks here. If the project venv
does not exist, create it using that interpreter:

```powershell
python -m venv .venv
```

Then install and verify using its explicit interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Select `.venv/Scripts/python.exe` in VS Code too. The camera-free tests do not
require a camera or the IDS bindings. For actual camera use, install the IDS SDK
and drivers first, then install `.[camera]` as described in the README. Do not open
the camera just to run the tests. Runtime versions can resolve differently on a
new machine: the project currently has dependency ranges, not a reproduced lab
lockfile. At home, focus on saved data, simulation, and the agreed analysis issue.

### Prompt For A New Copilot Chat

```text
We are continuing the Polarcam rewrite from another machine. Read README.md and
REWRITE_NOTES.md first, then LAB_SETUP.md; this chat may have no local history.

The latest lab session is 2026-10-01, with an experiment planned for 2026-10-02.
Stage 1 is complete: desktop launch, a lab Tcl/Tk startup fix, a responding GUI,
analyser start/stop fixes, a guided 10x tuner and an independent diagnostic recorder.
Latest: intensity-selected spot analysis with full native stacks, manual
Sample/Background and interference On/Off/Unknown metadata, and all-rod sphere,
stereographic scatter and projected density. Three averaging orders are available;
3D Cartesian is default. Read LAB_SETUP.md's Intensity-Based Spot Analysis section
for the folded orientation conventions and unvalidated theta-model assumptions.
92 camera-free tests pass; real-sample validation is next. No hardware was operated.
The first stationary 60-second diagnostic NPZ has been
reviewed: 47,927 frames, 798.791 fps readback, 1194.270 us, gains 1, background OFF.
Slow changes are in native intensities, with no clipping or multi-second GUI queue.
Irregular drift dominates a smaller ~5.45 s spectral feature; this does not diagnose
slide thickness, hardware fault, aliasing or a voltage direction. The user may
return on 2026-10-02 with the actual sample. Read the numerical findings in
LAB_SETUP.md before proposing another comparison; do not require manual Z sweeps.
Defaults are now 1.2 ms exposure, gain 1, and background subtraction off. Stop
analysing returns to full-frame live view at the same magnifier centre. The app
must be relaunched when code changes; the assistant did not operate a real camera.
Reviewing the user's recording does not certify general hardware readiness.
The user skipped general Stage 2 checks and asked to investigate magnifier-intensity
lag. Background caching and plot reuse are now implemented, with camera-free
before/after measurements. The guided tuner compares fixed-amplitude manual Z
sweeps with matching actual camera settings, uses native counts and host receipt
timestamps, and exports traces/results. Physical suppression is not yet validated.
Intensity-only rod finding and offline all-rod distributions are now implemented;
validate them on the actual sample rather than building a second implementation.
Do not claim skipped checks passed. Ask for the current lab result or next action.

Work on testing/v3-rewrite and preserve existing work. Performance is the main
goal: fast, reliable camera ROI/FPS changes and buffer saving for a future spot
cycler that records about 10 seconds per selected spot and repeats the list.
Do not restructure for its own sake, and do not sacrifice recording integrity.

Findings 1, 2, 6, and 7 have targeted fixes and camera-free regression tests. Hugh
confirmed [90,45;135,0]; the four inverted channel mappings are corrected. Global
ROI parity remains a follow-up. General NPY intensity scaling (3), calibration
assets (4), and angular calculations (5) are deferred; only the intensity analyser
was moved to native counts for tuning. Spot-preview brightness scaling has been
investigated but not changed. New scientific features must make their conventions
explicit; do not silently adopt the APD notebook's calibration for the camera.
Do not start the cycler or unrelated numerical changes automatically.

Check the branch/worktree and the repository-local Python environment, then give
a brief status based on the notes. Ask whether I have sample data or a chosen
next issue before continuing the deferred work. Keep decisions,
evidence, tests, and open questions in REWRITE_NOTES.md as we proceed.
```

### Questions To Take To Hugh

1. Resolved 2026-09-19: Hugh confirmed `[90,45;135,0]`. The swapped I0/I90
  assignments were errors and the four identified paths have been corrected.
2. What is the physical zero-angle reference and positive rotation direction?
   Are any saved recordings flipped, transposed, rotated, or shifted by an odd
   sensor ROI origin? Which path should serve as the reference for each format?
3. What are the actual bit depth, saved-value scaling, and background-subtraction
   conventions in the older recordings? Do they have sidecars and phase markers?
   This is clarification only; the intensity-scaling change is still on hold.
4. Which local theta-model/calibration files are current for each optical setup
   and medium? Bring the actual parameter files and provenance when available;
   do not infer the intended coefficients from a GUI label alone.
5. For later discussion, what assumptions were intended for the sphere-axis fit
   and Brownian classification, and are there known-good examples or simulations?

The detailed function map and numerical examples below support these questions.

### Data To Bring

- A small raw mosaic NPY recording and all associated JSON files, keeping the
  original filenames and relative folder layout. Old recordings are useful.
- Known acquisition FPS, exposure/gain, sensor pixel format/bit depth, ROI offsets
  and dimensions, and whether correction or rescaling occurred before saving.
- A brief note on what should be visible or measured, such as a stationary rod,
  known rotation direction/speed, or a result Hugh regards as a useful reference.
- Later, the calibration assets and camera model/SDK details for the lab setup.
- Transfer datasets separately from normal code commits; keep reference data
  read-only and write derived output somewhere else. Provide the new local paths.

### Next-Session Checklist

- [x] Receive Hugh's channel-convention explanation and record the decision.
- [ ] Inspect a supplied old recording and its metadata without modifying it.
- [ ] Agree expected results from physics/simulation rather than treating every
  existing output as correct.
- [ ] Select one next issue explicitly; points 3, 4, and 5 remain on hold until then.
- [ ] At the lab, measure ROI/FPS switching and writer throughput before claiming
  a cycler speed improvement or choosing buffer-reuse changes.

## Implemented Reliability Fixes

### Recording Reload

[src/polarcam/live/recording_io.py](src/polarcam/live/recording_io.py) supplies
shared sidecar-FPS and phase-marker handling. The readers use actual/readback FPS
fields rather than a requested FPS or a frame-size heuristic. Unknown FPS must be
chosen explicitly: the main live reader prompts, and angle analysis offers its
existing 77 / 1600 / Manual choices. An FPS prompt cancellation aborts loading.

V3 appends an artificial all-zero image with one pixel equal to one in its top-left
2x2 block to encode ROI parity. The two readers now exclude that image. An explicit
`phase_marker_appended: false` preserves a legitimate one-hot final measurement;
`true` requires a valid marker. With no marker metadata, the v3 signature is used
for backward compatibility. That heuristic is inherently ambiguous for an old,
genuinely one-hot final image, so explicit metadata is preferable.

Marker removal returns a view, not a full-stack copy. This change does not modify
native saved pixel values, rescale the analysis, or resolve ROI/channel-sign
conventions. Other standalone readers still need a separate consistency pass.

### Capture Failure Handling

[src/polarcam/live/fetch_frames.py](src/polarcam/live/fetch_frames.py) now connects
the camera error signal before opening the camera and reports frame-processing
failures. A monotonic inactivity watchdog covers both failure to receive a first
frame and a stream that stops partway through. Cleanup disconnects callbacks and
stops/closes the controller on failure as well as success.

`--frame-timeout` / `frame_timeout_s` defaults to five seconds **without a frame**.
It is not a five-second pause and does not stop a healthy ten-second recording.
No new fixed wait was added. Stop flags remain able to finish a partial recording,
and hardware-clamped FPS remains accepted. Intentionally slow acquisition needs a
larger timeout. Reported camera errors abort instead of masquerading as success.

Limits: the watchdog runs while Python pumps events; it cannot interrupt a blocked
native SDK call or disk write. Recording still accumulates in RAM and writes after
stopping. A capture error currently raises without salvaging a partial recording;
streaming persistence belongs in the upcoming recorder work.

### Analysis Worker Shutdown

The live application now requests cancellation before clearing caches or releasing
the source. It uses one shared, at-most-one-second join budget for the decoder and
reconstructor, with no minimum wait when they finish quickly. If either remains
alive, their references and buffers are retained, cancellation stays set, and
source replacement / GUI destruction is refused with a retry status message.
The next successful source start, not shutdown, clears cancellation. Cancelled
reconstruction skips its final analysis stage and can stop while seeding traces.

This is a bounded defensive fix, not a general asynchronous job scheduler. It does
not claim cancellation support for every auxiliary worker or camera subprocess.

## Channel Conventions: Confirmed By Hugh

Hugh confirmed this sensor mosaic on 2026-09-19, as relayed by the user:

```text
90   45
135   0
```

For this grid, the normal definitions are:

```text
X = (mean(I0) - mean(I90)) / (mean(I0) + mean(I90))
Y = (mean(I45) - mean(I135)) / (mean(I45) + mean(I135))
phi = 0.5 * atan2(Y, X)
```

The implementations add a small denominator epsilon. Local window parity also
matters: a crop starting on an odd sensor row/column has a shifted mosaic.

| Implementation | Effective Grid At Even Origin | Purpose / Callers |
| --- | --- | --- |
| Live `BasicVideoPlayer._xy_phi_stats_from_raw_window` | `[90,45;135,0]` | Normal live/spot/stationary XY and phi. Used by `_append_xy_frame`, `_append_xy_from_frame`, and the window wrapper `_xy_phi_stats_from_frame`. |
| Live `_capture_auto_spot_series` | `[90,45;135,0]` (corrected) | Fresh high-FPS captures for optional automatic inspection of the top spots; results override their plotted trajectories. |
| Live `_spot_playback_windows` | `[90,45;135,0]` (corrected) | Diagnostic S-space playback; computes squared energy, not signed phi. |
| Angle `AngleDistributionApp._xy_phi_from_gray_bounds` | `[90,45;135,0]` (corrected) | Batch inspection-file XY/phi through `_analyze_inspection_path`. |
| Angle `_append_xy_from_frame` | `[90,45;135,0]` (corrected) | XY/phi for single-file inspection and detected widefield spots in `_process_npy_worker`. |
| Shared `make_qu_reconstructor` | `[90,45;135,0]` at even origin | Raw Q/U on the overlapping intersection grid, used for S-map detection in both applications. |

Source locations:
[src/polarcam/live/Spinners_gui_live.py](src/polarcam/live/Spinners_gui_live.py),
[src/polarcam/live/angle_distribution_analysis.py](src/polarcam/live/angle_distribution_analysis.py),
[src/polarcam/live/pol_reconstruction.py](src/polarcam/live/pol_reconstruction.py).

Swapping only I0/I90 gives `X_alternative = -X_sensor`, while Y is unchanged.
Therefore phi becomes `90 degrees - phi`, modulo 180 degrees. It reflects the
angle; it is not merely a constant offset, and it reverses signed rotation.
The radius and squared S-map energy can look unchanged, hiding the disagreement.

Verified synthetic example: repeat the raw block `[10,70;30,90]`. Normal live,
corrected angle analysis, and corrected auto-capture now give X=0.8, Y=0.4,
phi=13.282526 degrees. Before the fix, the latter paths gave X=-0.8, Y=0.4,
phi=76.717474 degrees. The shared XY reconstructor agrees with normal live.
Playback's signed X changes from -80 to +80; Y remains +40 and squared energy
remains 8000. Previously exported angles are not automatically corrected;
reanalysis of raw recordings uses the corrected convention.

Additional point-1 follow-up: `_xy_phi_stats_from_frame` receives ROI metadata but
passes only its local crop origin to the channel reducer. The parity of the global
ROI origin is not applied there. GUI ROI requests are made even, but arbitrary
older ROI recordings and actual camera readback still need a defined policy.

## Selected-Spot Brightness: Investigated 2026-09-19

- `_spotrec_update_preview` in the live GUI always calls `to_u8_preview(window,
  lo_pct=0.0, hi_pct=99.5)`. This applies before and during "Spot examine"
  recording and is independent of the live "Display stretch" checkbox.
- `Detection_alg_offline.to_u8_preview` computes percentiles over nonzero pixels
  in the current window, maps the lower/upper values to 0/255, and clips. Exact
  zeros remain black. This is relative contrast, not absolute sensor brightness.
  The scale can change as the crop, frame, or recording mode changes.
- Verified example: `[0,10,20;30,40,50]` becomes `[0,0,64;128,192,255]` at those
  percentiles. Multiplying all input values by four produces the same preview.
  The helper does not modify the input array.
- `_update_spot_view` and `_update_stationary_view` also stretch their windows
  (0th/100th percentiles), but those thumbnails show the S-map, not raw intensity.
- The live magnifier instead uses `_apply_live_display_stretch`, optional and
  disabled by default, with user-selected low/high bounds. The spot-recording
  preview's automatic scaling is not disabled by that control.
- During recording, `_spotrec_preview_tick` first applies `_to_gray_u8` to the
  preview file. Its frame-maximum-dependent conversion is the separate deferred
  point 3 below, and remains unchanged.
- `fetch_frames` saves native uint8/uint16 values to the recording and preview
  NPY files; contrast scaling happens after loading the preview in the GUI.
  Optional background subtraction does change saved values when enabled, and
  recording exposure/gain settings affect the acquired signal. The Save action
  copies the recording, not the rendered preview. Synthetic tests verify native
  save/preview preservation; today's experimental values have not been checked.

Recommendation for a later authorized display change: offer a fixed sensor-range
preview alongside optional auto-contrast, so brightness is comparable across spots
and recording states. Do not infer saturation or absolute brightness from the
current stretched preview. No display or intensity-scaling change was made here.

## Intensity Scaling: Deferred Point 3

The issue is the heuristic in `_to_gray_u8`, not ordinary display contrast:

1. A uint16 frame whose maximum is <=255 is converted without dividing.
2. A uint16 frame whose maximum is 256..4095 is shifted right by four bits.
3. The choice is made separately for every frame and the result feeds analysis.

Reproduced example: an unchanged raw pixel of 240 becomes 240 in the first frame
and 15 in the second solely because another pixel went from <=255 to 256. The
unnormalized Q/U range map and brightness thresholds can see this as a large change.
An exact common scale would cancel in normalized XY ratios, but integer truncation
does not cancel exactly. This is not a claim that all XY angles change by 16x.

A likely eventual separation is native-depth measurement processing and explicit
fixed-depth preview conversion. **Do not implement that until point 3 is discussed.**

## Other Deferred Findings

- Calibration: the angle tool currently loads only its built-in models when the
  external parameter JSON is absent, yet offers more models in its selector.
  Selecting an unavailable water model silently uses `hole+fresnel`. Inspect the
  local lab assets before deciding model identity, defaults, and failure behavior.
- Angles: default `np.unwrap(phi)` is wrong for the pi-periodic phi returned by
  half-atan2. A constant-speed synthetic rotation gave an MSD slope about 0.085
  instead of 2. The angle tool contains a double-angle-first unwrapping path.
- Sphere fitting: the largest-eigenvalue axis rule gave about 89.97 degrees axis
  error on a perfect 70-degree cone about Z, versus about 0.008 degrees for a
  15-degree cone. Define the intended model and valid regime before changing it.
- Scientific validation: manual-rotation matching favors measured changes close
  to the expected manual rotation. Its agreement plot is not an independent test.
- Saturation reporting uses uint16's maximum in some paths, although Mono12
  saturation is 4095. Treat saturation/validity metrics as a later numerical audit.

## Cycler Performance Work

Current code evidence, not hardware benchmarks:

- Each `fetch_frames` subprocess creates a controller, opens the SDK/camera,
  configures it, starts, stops, closes, and only then writes the stack. A cycler
  should normally keep one camera session open across spots and cycles.
- [src/polarcam/live/backend/ids_backend.py](src/polarcam/live/backend/ids_backend.py)
  already accepts ROI changes while running, but `_apply_roi_payload` pauses
  acquisition, clears the pool, forces a pixel format, rebuilds buffers, and
  resumes. `_announce_and_queue` clears the pool too. A timing change can trigger
  another pause/rebuild. Measure these steps before combining/reusing them.
- [src/polarcam/live/fetch_frames.py](src/polarcam/live/fetch_frames.py) collects
  frames in a list, then stacks and concatenates them for the phase marker. Those
  whole-recording copies and synchronous saving should not delay the next ROI.
- Candidate approach: persistent acquisition owner, a coordinated ROI/FPS update,
  reusable buffers when payload permits, and bounded asynchronous chunk writing.
  Frame ownership must stay valid when SDK buffers are requeued. If storage cannot
  keep up, report backpressure/failure rather than silently dropping measurements.
- Each segment needs spot/cycle identity, actual ROI and timing, sample count,
  pixel format, and first/last timestamps or frame IDs where supported. Mark the
  transition so frames from the old ROI cannot enter the new segment.

Measure at the lab: last valid frame of spot A to first valid frame of spot B;
ROI apply time; FPS apply/readback time; buffer drain/rearm time; enqueue and disk
flush time; sustained writer throughput; peak RAM/queue occupancy; dropped or
misassigned frames. Report median and tail latency over repeated cycles, not just
one successful switch. There is no defensible latency promise without these data.

Do not start with a GUI-toolkit replacement or a large class/file split. Extract
only the ownership and numerical functions that help this acquisition workflow.

## Data And Validation

Old recordings are useful now, especially raw mosaic NPY plus accompanying JSON.
They can validate reading, metadata, ROI extraction, sample counts, and analysis
regressions. Without JSON, provide known FPS, sensor/ROI origin, pixel depth, and
any preprocessing. AVI can exercise playback but does not establish native-depth
scientific parity, especially if encoded lossily. Do not overwrite reference data.

At home, synthetic mosaics with known channel intensities, rotation direction,
speed, noise, and ROI parity can define expected analysis results. Hugh can help
resolve intended conventions and model assumptions, rather than treating every
current result as ground truth. Real SDK/hardware testing is still needed for
switch latency, frame loss, buffer reuse, and recording throughput.

Current automated checks: 29 live tests plus four launcher tests, run with the
repository-local virtual environment (Python 3.12 on the 2026-09-19 checkout;
the original lab checks used Python 3.13.0):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use temporary synthetic recordings, fake camera signals/clocks, and
worker shutdown probes; no camera is opened. They do not validate the deferred
science or hardware performance. Existing Pylance warnings in angle analysis for
two script-local imports and the optional `Image.Image` annotation remain outside
this change; runtime script imports and the new helper pass the tests.

Next discussion: choose whether to change spot-preview brightness presentation,
address a deferred numerical issue, or begin the measured persistent-camera/segment-writer work. Keep this file
updated when decisions change so home and lab work share the same assumptions.