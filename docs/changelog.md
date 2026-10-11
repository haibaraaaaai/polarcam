# Changelog

All notable changes to this project will be documented in this file.

## 2026-10-11 Background profile provenance (awaiting lab validation)

- Active background selection now preserves the source filename and NPY SHA-256;
  the GUI shows the source, and frame/spot/stationary sidecars retain the recorder's
  loaded profile identity even if the active file is replaced later.
- Recorders receive the GUI's explicit profile path and expected hash. Missing,
  changed or incompatible profiles stop capture instead of claiming subtraction.
- Historical sidecars are unchanged; their repeated active-file path cannot
  identify earlier wide/narrow selections. Regression checks were added but not
  run; source inspection only, with camera validation deferred to the lab.

## 2026-10-11 Workspace cleanup and operating manual

- Removed one-off `_tmp/` editing scripts and the redundant `requirements.txt`.
- Renamed `tools/` to `scripts/` and moved the desktop shortcut installer there;
  adjusted its repository-relative paths. The launcher remains at the root.
- Kept the existing software tests in `tests/`; no tests or camera execution were
  performed. README now lists the separate command for standalone metric checks
  and notes the older test expectations affected by the 10 October changes.
- Consolidated root documentation into README plus `Polarcam User Manual.docx`.
  The manual develops a copy of Hugh Bowman's 9 September handover into the
  operating guide; the Desktop original is unchanged. It distinguishes the
  physical controls, software declarations and pending lab validation.
- Archived LAB_SETUP and REWRITE_NOTES as dated evidence; moved interference
  methods here under `interference.md`. The separate 10 October camera update
  has been incorporated into the operating manual and this change record.
- Application source, tests, analysis projects and measurement files were not
  changed by this cleanup. The Word document was rendered with Microsoft Word
  and visually reviewed; that is document QA, not camera validation.

## 2026-10-10 Camera controls awaiting validation

Commit `10cc419` was reviewed as source only, with no tests, GUI launch or camera
execution, at the user's request. Its intended changes are:

- Shared requested exposure 0.6 ms and analogue gain 20 across live, magnifier,
  full-frame, spot and stationary controls. Camera readbacks still determine what
  was applied; supported gain limits may clamp the request.
- Common hinges/bfm session selection, with frames/spots/stationary directories
  beneath `data/<experiment>/polarcam/<session>/`. Background profiles remain
  beneath the data root's `backgrounds/`. Historical imports are not renamed.
- One manual On/Off/Unknown drive declaration, shared role and notes, with a
  snapshot taken at acquisition start for frame and spot sidecars. The legacy
  sound field derives from that declaration; Unknown is null. It does not switch
  hardware or verify cancellation.
- Human stationary review explicitly distinguishes pending/accepted/rejected;
  files are retained, with a separate `review.json`. Acquisition metadata keeps
  its original unreviewed status.
- Explicit magnifier ROI/settings application and restart, live-target transfer
  to Spot examine, requested/acquired ROI previews and fixed ROI during capture.
  Sizes use even sensor pixels, with odd sizes rounded down. An even-crop
  off-by-one error was corrected in the source.
- Manual spot addition/removal reruns the loaded-file analysis. Added spots bypass
  automatic motion/ring/directionality filters; edits persist for the current app
  session and are included in analysis metadata, without modifying raw input.
- Magnifier time plots can show native mean intensity, X or Y with ROI-origin
  parity and the displayed subtraction setting; they are diagnostic anisotropies.

The operating manual retains all eight pending lab checks: settings/readbacks,
destinations/metadata, drive states, magnifier controls, spot ROI, manual edits,
human review and camera lifecycle. Large GUI restructuring, automatic biological
classification and hardware drive-amplitude control were outside this change.

## 2026-10-10 Research workspace organisation

Commit `1af2f29` introduced experiment/detector data folders, verified local
imports, dated analysis directories, source-only analysis versioning and ignore
rules for raw/generated/private files. Desktop source data were retained. The
local import manifests record SHA-256 verification; measurements were not pushed.

## Earlier unreleased changes before 10 October

The entries below are preserved historical change descriptions. Defaults and
validation claims belong to those earlier versions, not the current untested
camera update. Refer to README and the user manual for current status.
### Added
- Per-spot Theta(t) beside Phi(t), using the All-rod distribution reconstruction
  and selected finite-NA model, with invalid-sample gaps and model-change refresh.
- Interference tuning: drift-immune XY residual metric. Trials record the four
  mosaic channel means; the detrended anisotropy locus extent gives an
  interference measure that slow focus/illumination drift cannot fake, plus the
  picture the span number lacked (open arc = Z wobble did not cover a full
  fringe). The tuning window shows the residual locus and per-trial extent
  chart; trials table gains XY extent / XY reduction columns; trial JSON saves
  are schema_version 2. New offline analyser
  `python -m polarcam.live.interference_npz_analysis RECORDING.npz` for
  diagnostic recordings, and `tests/test_interference_metrics.py` validating
  the metric against the known sinc null. See INTERFERENCE_TUNING.md.

### Changed
- Limit Qt event delivery inside the Tk GUI to queued calls and deferred cleanup,
  mitigating the native event-pump boundary implicated in a reported GIL abort.
  Real queued-frame/Tk subprocess tests pass; the original hardware crash has not
  been reproduced or confirmed fixed. Standalone capture event pumping is unchanged.
- Use native camera counts for magnifier intensity measurements; keep 8-bit
  conversion and peak-preserving display reduction out of numerical scoring.
- Default the main GUI's live, magnifier, full-frame and spot-capture exposure to
  1.2 ms, with background subtraction off in all main live/capture controls.
- Set the main live-view gain default to 1.0 instead of 20, matching the existing
  capture gain defaults. Existing running windows keep their current settings.
- Adopt Hugh Bowman's Polarcam Live application as the maintained baseline,
  imported from Polarcam_v3 commit `5fe41d5bdd9f089a441f30554888c0b4773e62a7`.
- Preserve the initial 17-file upstream snapshot in commit
  `bb47c6490379f4c8ae90cbfe6d3e4315dad2792c` before subsequent targeted fixes.
- Replace the superseded Qt application's active modules with the live baseline;
  earlier code remains in Git history.
- Route `polarcam`, `polarcam-gui`, and `python -m polarcam` to the live GUI, with
  an optional `--data-dir` working directory and camera-free launcher tests.
- Align install dependencies with the imported application and ignore generated
  runtime output. Calibration, data, simulations, environments, and builds are
  excluded from this import.

### Fixed
- Cache analyser background discovery/crops (including missing profiles), reuse
  Matplotlib figures/canvases, and keep reference plot limits fixed while tuning.
- Make the live-controls sidebar scrollable so lower capture/analyser controls
  are reachable at normal window sizes.
- Make Stop analysing close the ROI stream and resume full-frame live view at
  the same magnifier centre, with magnification enabled. Internal stops for tab
  changes, capture handoffs and shutdown do not trigger an automatic restart.
- Prevent live preview from starting while intensity analysis owns the camera,
  recover the previous live stream after synchronous analyser-start errors, and
  cancel pending GUI timer work during shutdown.
- Resolve installed Windows Tcl/Tk runtime paths from the launcher's own base
  Python installation when no explicit override exists; fixes GUI startup on the
  lab Python 3.13 environment without changing system environment variables.
- Correct inverted I0/I90 assignments in live auto-inspection, diagnostic spot
  playback, and both angle-analysis reducers to Hugh's confirmed `[90,45;135,0]`
  sensor convention. Saved raw recordings are unchanged.
- Use recorded FPS and exclude phase-marker frames when reopening NPY recordings
  in the live and angle-analysis tools. Require explicit FPS when it is unknown.
- Propagate camera errors and time out stalled frame streams in the capture
  helper, without adding a success-path wait or limiting recording duration.
- Keep analysis cancellation set and retain worker resources until threads exit;
  refuse source replacement or GUI destruction while analysis is still stopping.

### Added
- Intensity-based spot selection from full-stack native mean intensity, without
  temporal-motion or hollowness filters; original time-variation mode retained.
- Explicit Sample/Background and manual interference On/Off/Unknown condition
  metadata alongside native frame stacks and requested/applied background state.
- All-rod unit-sphere, stereographic and projected-density analysis with Cartesian,
  anisotropy or four-intensity averaging, selectable existing theta models,
  per-window validity and reproducible NPZ export. Models remain uncalibrated for
  the new sample and folded orientation conventions are documented.
- Native-stack replay, stationary detection, channel parity, averaging, density,
  metadata and cancellation coverage; complete camera-free suite now has 92 tests.
- Independent intensity diagnostics without a tuning reference or Z sweep:
  native raw/analysed and four-channel means, host timing, clipping statistics,
  manual event markers and readback history in automatically saved NPZ files.
- Bounded diagnostic memory, partial/empty capture export and failed-save retention;
  camera-free coverage now totals 75 tests. No generator control is added.
- Guided interference reference/test trials with live reduction ratios, a default
  10x observed-modulation target, quality/context guards, best-trial history and
  JSON export of native mean traces and metadata. Generator/Z controls stay manual.
- Host receipt timestamps through an additive frame signal, preserving existing
  array-only camera subscribers and excluding stale frames from fresh trials.
- Tuning, performance-path, timestamp relay and layout regression checks; the
  current complete camera-free suite has 66 tests.
- Ten fake-camera GUI lifecycle tests for analyser-to-live transitions, changing
  magnifier images, repeated cycles, startup errors and shutdown. Together with
  updated startup-default assertions, all 46 tests pass on the current lab PC.
- Windows desktop launcher and shortcut installer, plus `LAB_SETUP.md` covering
  another-PC setup and staged lab checks ahead of the 2026-10-02 experiment.
- Camera-free full-window startup/shutdown test and launcher Tcl/Tk environment
  tests. The suite now has 36 passing tests on the current lab PC.
- Channel-convention regression tests and spot-preview contrast characterization;
  verify native preview-file values and document existing display stretching.
- Camera-free regression tests for recording reload, capture failure handling,
  inactivity deadlines, and worker shutdown.
- `REWRITE_NOTES.md` documenting the performance-first spot-cycler goal, channel
  conventions, deferred findings, and validation tasks for home and lab.

### Removed
- Remove `legacy/`, the earlier `offline/` and `reference_files/` material, and
  `ideas.md` so the working tree contains only the v3 rewrite baseline.
- Remove the notebook-only dependency extra and obsolete legacy tooling exclusions.

### Notes
- The source import does not merge v3's Git history or establish licensing for
  its contributions. See the README for provenance and attribution.

## [0.1.0] - 2026-08-25
### Added
- MIT license and this changelog.
- `requirements.txt` mirroring the `pyproject.toml` dependencies plus the
  notebook tooling used in `offline/`.

### Notes
- The IDS peak SDK wheels (`ids-peak`, `ids-peak-ipl`) are installed separately
  with `--no-deps` after the IDS peak Cockpit.
- `legacy/` holds the pre-rewrite sources for reference only and is not
  maintained.
