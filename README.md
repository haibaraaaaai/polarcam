*# Polarcam

Live polarization-camera acquisition and spot analysis. The maintained application
is Hugh Bowman's Polarcam Live implementation, imported from Polarcam_v3 as a
baseline for further development. Its acquisition and analysis methods have not
been rewritten in this import.

## Install and Run

Use Python 3.12.10 or later with Tk support. From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m polarcam --data-dir .\runs
```

`polarcam` and `polarcam-gui` launch the same live application. The launcher starts
the original script with the current Python interpreter, preserving its local
imports and camera subprocess layout. `--help` does not open the GUI or camera.

`--data-dir` selects the working directory for recordings, backgrounds, and
analysis output; it defaults to the current directory. Use a dedicated directory
outside any known-working reference installation. This option is not a sandbox:
file dialogs and explicitly selected output paths can still use other locations.

Live capture additionally requires the IDS peak SDK and camera drivers. Install
IDS peak Cockpit first, then install the Python camera bindings in this environment:

```powershell
python -m pip install -e ".[camera]"
```

The camera SDK is not needed to analyze saved AVI/NPY files. Runtime dependencies
are declared in [pyproject.toml](pyproject.toml); [requirements.txt](requirements.txt)
installs that same project. The original machine's package pins were not imported
as a lockfile, and its Python environment has not been reproduced.

## Application

- Live camera preview, exposure/gain controls, and spot inspection.
- AVI/NPY loading, spot detection, anisotropy reconstruction, and rotation analysis.
- Spot recording through the bundled camera helper.
- Standalone angle-distribution, TDMS, manual-rotation, and recording-comparison tools.

The original standalone scripts remain under [src/polarcam/live](src/polarcam/live).
For example, from the repository root:

```powershell
python src/polarcam/live/angle_distribution_analysis.py
python src/polarcam/live/tdms_manual_scaling_xy_gui.py
```

The TDMS utility retains an upstream machine-specific default input path; select
your own file. Auxiliary scripts remain an imported baseline, not a portability
or behavior cleanup.

## Source Import and Attribution

Imported on 2026-09-15 from the local Polarcam_v3 checkout:

- Source revision: `5fe41d5bdd9f089a441f30554888c0b4773e62a7` (2026-09-10).
- Source directory: `polarcam_live/`.
- Destination: `src/polarcam/live/`, preserving relative paths.
- Scope: the 12 top-level Python scripts, three `backend/` Python files, and two
  `Controlling/controller/` Python files, 17 upstream files in total.
- All 17 files were imported byte-for-byte. Only package markers, the external
  launcher, dependency metadata, and documentation were added or adapted.

Hugh Bowman developed the v3 live application and analysis workflow, building on
Daping Xu's earlier Polarcam camera software and rewrite. Upstream history also
contains contributions recorded under the author name Hugh T. This is a source
snapshot import, not a merge of the v3 commit history. Its source revision identifies
the baseline in the retained local v3 repository.

The existing [LICENSE](LICENSE) is retained unchanged. The imported v3 checkout
contained no top-level project license. Attribution here does not resolve the
license for those contributions; confirm permission and applicable notices before
redistributing the combined application. This import does not assign a new license
to Hugh's contributions.

### Excluded From This Import

- Calibration/theta-model assets, background profiles, recordings, and datasets.
- Simulation/offline experiments and the dataset-specific analysis-script collection.
- Virtual environments, Git metadata, build/distribution output, and generated plots.
- V3's obsolete `src/polarcam` and `Polarcam_v3_Hugh` application copies.

Missing calibration assets mean this is not an equivalent deployment of the lab
installation: external theta models and background corrections require their
corresponding files. No files were changed in the local v3 or Polarcam Software
reference installations.

## Earlier Versions

The superseded Qt application, 2024 legacy implementation, offline utilities and
notebooks, reference files, and old rewrite roadmap are preserved in Git history
at `6c36303084b4ff799a277166df016b8b62e9c53d`.

These earlier materials have been removed from the current source tree so the
rewrite starts with only the maintained v3 live baseline. Existing Git history
has not been rewritten.

## Development Checks

After installing the project, run the camera-free launcher tests:

```powershell
python -m unittest discover -s tests -v
```

The import was checked for matching source hashes and Python syntax. Launcher
tests cover the selected interpreter, script layout, working directory, help, and
exit status. These checks do not establish camera timing, hardware behavior, or
scientific parity with the lab installation; those require separate validation.
