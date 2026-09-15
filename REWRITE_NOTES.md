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
| 1. Inconsistent channel conventions | Explain and ask Hugh; calculations unchanged. |
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
REWRITE_NOTES.md first; this chat may not have any previous local history.

Work on testing/v3-rewrite and preserve existing work. Performance is the main
goal: fast, reliable camera ROI/FPS changes and buffer saving for a future spot
cycler that records about 10 seconds per selected spot and repeats the list.
Do not restructure for its own sake, and do not sacrifice recording integrity.

Findings 2, 6, and 7 have targeted fixes and camera-free regression tests. Channel
conventions (1) await clarification from Hugh; intensity scaling (3), calibration
assets (4), and angular calculations (5) are explicitly deferred. Do not change
those or start the cycler automatically. No hardware performance is validated yet.

Check the branch/worktree and the repository-local Python environment, then give
a brief status based on the notes. Ask whether I have Hugh's reply, sample data,
or a chosen next issue before continuing the deferred work. Keep decisions,
evidence, tests, and open questions in REWRITE_NOTES.md as we proceed.
```

### Questions To Take To Hugh

1. For raw frames with the sensor grid `[90,45;135,0]`, why do angle analysis and
   live auto-inspection use `[0,45;135,90]` while normal live analysis does not?
   Is the sign difference intentional for older data or a coordinate convention?
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

- [ ] Receive Hugh's channel-convention explanation and record the decision.
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

## Channel Conventions: Question For Hugh

The user's stated sensor mosaic is:

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
| Live `_capture_auto_spot_series` | `[0,45;135,90]` | Fresh high-FPS captures for optional automatic inspection of the top spots; results override their plotted trajectories. |
| Live `_spot_playback_windows` | `[0,45;135,90]` | Diagnostic S-space playback; computes squared energy, not signed phi. |
| Angle `AngleDistributionApp._xy_phi_from_gray_bounds` | `[0,45;135,90]` | Batch inspection-file XY/phi through `_analyze_inspection_path`. |
| Angle `_append_xy_from_frame` | `[0,45;135,90]` | XY/phi for single-file inspection and detected widefield spots in `_process_npy_worker`. |
| Shared `make_qu_reconstructor` | `[90,45;135,0]` at even origin | Raw Q/U on the overlapping intersection grid, used for S-map detection in both applications. |

Source locations:
[src/polarcam/live/Spinners_gui_live.py](src/polarcam/live/Spinners_gui_live.py),
[src/polarcam/live/angle_distribution_analysis.py](src/polarcam/live/angle_distribution_analysis.py),
[src/polarcam/live/pol_reconstruction.py](src/polarcam/live/pol_reconstruction.py).

Swapping only I0/I90 gives `X_alternative = -X_sensor`, while Y is unchanged.
Therefore phi becomes `90 degrees - phi`, modulo 180 degrees. It reflects the
angle; it is not merely a constant offset, and it reverses signed rotation.
The radius and squared S-map energy can look unchanged, hiding the disagreement.

Verified synthetic example: repeat the raw block `[10,70;30,90]`. Normal live
reduction gives X=0.8, Y=0.4, phi=13.282526 degrees. Angle analysis gives X=-0.8,
Y=0.4, phi=76.717474 degrees. The shared XY reconstructor agrees with normal live.

Ask Hugh: "For the camera grid [90,45;135,0], was the inverted X convention in
angle analysis and auto-inspection intentional, perhaps for older recordings or
an image-axis convention? Should these paths agree with normal live analysis?"

Additional point-1 follow-up: `_xy_phi_stats_from_frame` receives ROI metadata but
passes only its local crop origin to the channel reducer. The parity of the global
ROI origin is not applied there. GUI ROI requests are made even, but arbitrary
older ROI recordings and actual camera readback still need a defined policy.

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

Current automated checks: 24 reliability tests plus four launcher tests, run with
the repository-local virtual environment (Python 3.13.0 on this machine):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use temporary synthetic recordings, fake camera signals/clocks, and
worker shutdown probes; no camera is opened. They do not validate the deferred
science or hardware performance. Existing Pylance warnings in angle analysis for
two script-local imports and the optional `Image.Image` annotation remain outside
this change; runtime script imports and the new helper pass the tests.

Next discussion: settle point 1 with Hugh, then choose the next numerical issue
or begin the measured persistent-camera/segment-writer work. Keep this file
updated when decisions change so home and lab work share the same assumptions.