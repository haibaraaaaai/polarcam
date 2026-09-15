# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Changed
- Adopt Hugh Bowman's Polarcam Live application as the maintained baseline,
  imported from Polarcam_v3 commit `5fe41d5bdd9f089a441f30554888c0b4773e62a7`.
- Preserve 17 upstream Python files unchanged under `src/polarcam/live/`.
- Replace the superseded Qt application's active modules with the live baseline;
  earlier code remains in Git history.
- Route `polarcam`, `polarcam-gui`, and `python -m polarcam` to the live GUI, with
  an optional `--data-dir` working directory and camera-free launcher tests.
- Align install dependencies with the imported application and ignore generated
  runtime output. Calibration, data, simulations, environments, and builds are
  excluded from this import.

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
