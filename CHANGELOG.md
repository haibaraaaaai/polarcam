# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Changed
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
