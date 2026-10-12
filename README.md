# Polarcam

Polarization-camera acquisition, spot inspection and offline nanorod analysis.
This README describes the software, repository and analysis entry points.
The root [Polarcam User Manual](Polarcam%20User%20Manual.docx) covers everyday
operation: opening the program, illumination, stage controls, interference tuning,
backgrounds, recording modes and saving measurements. It is an editable working
manual based on Hugh Bowman's 9 September 2026 handover.

**Current status:** camera-free GUI review and synthetic/fake-camera checks passed
on 12 October 2026. Camera timing, gain readbacks, stage controls and recording
transitions still need the lab check. The controls below supersede the older
session/review descriptions in the working Word manual.

## Install and launch

Use Python >=3.12.10 with Tk support. In PowerShell at the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\Launch-Polarcam.cmd
```

For live acquisition, first install the IDS peak SDK, camera drivers and Cockpit
on the lab PC. Confirm camera access in Cockpit, close it, then install the bindings:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[camera]"
```

The SDK is not required for offline saved-file analysis. Dependencies are declared
in [pyproject.toml](pyproject.toml); there is no separate requirements list to maintain.
The old lab environment has not been reproduced as a pinned lockfile.

[Launch-Polarcam.cmd](Launch-Polarcam.cmd) uses this checkout's `.venv` and defaults
to its `data/` folder. It retains a diagnostic console. To choose a data disk or
create the **Polarcam Lab** desktop shortcut:

```powershell
.\Launch-Polarcam.cmd --data-dir "D:\Polarcam Data"
.\scripts\Install-DesktopShortcut.ps1 -DataDir "D:\Polarcam Data"
```

The shortcut installer refuses to overwrite an existing shortcut. Relative
`-DataDir` values are resolved against the repository root. Moving the checkout
requires recreating the shortcut. `python -m polarcam`, `polarcam` and
`polarcam-gui` use the same application; `--help` does not open the GUI or camera.
With direct module/command invocation, the default data folder is `./data`
relative to the current working directory. File dialogs can select other locations.

## Repository layout

| Item | Purpose | Git policy |
| --- | --- | --- |
| `src/` | Camera application and reusable processing code | Track |
| `analysis/` | Dated research analyses, figures and notebooks | Track selected source and clean notebook inputs; ignore generated/private material |
| `data/` | Recordings, backgrounds and local imports | Ignore recordings; track the catalogue and folder documentation |
| `scripts/` | Maintenance utilities: shortcut installation and verified data import | Track |
| `tests/` | Automated software correctness checks using synthetic data/fake cameras | Track |
| `docs/` | Technical methods, changelog and historical records | Track |
| `Polarcam User Manual.docx` | Editable operating manual | Track |
| `.venv/` | This machine's installed Python environment | Ignore |

Data layout is described in [data/README.md](data/README.md), and analysis
versioning in [analysis/README.md](analysis/README.md). Recordings default to
`data/<experiment>/polarcam/YYYY-MM-DD/`, with `frames/`, `backgrounds/`, `spots/`
and `stationary/` underneath. **Recording folder** changes this parent for every
capture type. Existing data are not moved.

## Recording controls

- Exposure starts at **1.2 ms**, analogue gain at **20**, and magnifier is on.
  Exposure/gain fields share values across tabs. Use **Apply** to update a running
  live feed; new captures use the current values. Display stretch is separate.
- Enter a **Filename label** such as `hinge_widefield` before capture. Frame stacks,
  backgrounds and spot captures append it to their timestamps; notes are saved
  in JSON. Filenames remain unique. Spot recordings save automatically; **Save
  copy** makes another copy, and **Close recording** only closes the loaded result.
- Select/capture the appropriate background. Its source name and short hash are
  shown. Enabling live subtraction also enables the shared **Save with background
  subtraction** setting. Saving can still be switched off independently. Turning
  live subtraction off does not silently change the save choice. Used profiles
  are preserved in `backgrounds/used/`, and capture JSON records the exact hash.
- Spot analysis counts all detected candidates before motion/shape filtering.
  Each spot contributes its brightest pixel over the measured frames and current
  extraction window. The fixed 0–255 scale maps Mono12 0–4095; it ignores display
  stretch. **250+** means near full scale (~4015 raw); **clipped** means 4095.
  Counts appear in 25-unit bins, with a final 250–255 bin. Counts depend on detection;
  for static rods choose **Intensity**. Overlapping windows/hot pixels can affect
  the counts, so inspect the image too.
- BG-subtracted captures keep one raw maximum-projection image in a `diagnostics/`
  subfolder, referenced and hash-checked through their JSON. Keep it with the
  recording to recover accurate per-spot clipping counts. If it is absent, the
  program can add back the exact hash-verified integer background, including the
  sensor ROI offset. Zero-clipped pixels remain upper bounds; unverified or
  ambiguous backgrounds leave sensor clipping unknown.
- The live **Time trace** selector offers intensity, X, Y, or **X + Y anisotropy**
  together. Display reduction preserves extrema; measurements retain native samples.
- For frame-stack populations, choose **Intensity** detection, then **Theta /
  projection**. The theta tab shows 5° counts and spherical-area-corrected density;
  the sphere/projection tab is retained. **Frames / average = All** gives one mean
  orientation per rod using the selected averaging method and theta model.
  Any clipped extraction pixel excludes the whole rod from these distributions;
  legacy files without raw clipping evidence are marked provisional. Save analysis
  exports the angles, bins, exclusions and settings alongside the original channels.
- The good/bad/pending review tab and automatic review labels have been removed.
  Old recording folders are retained.

Before the next measurement, check actual exposure/gain readbacks, switch wide/narrow
backgrounds, and make one labelled frame stack and spot recording. Verify their
folder, JSON background identity, and brightness counts against the live image.

To copy a later dataset while retaining the source and verifying SHA-256 hashes:

```powershell
.\.venv\Scripts\python.exe scripts/import_local_data.py "E:\New hinge data" "data\hinges\polarcam\YYYY-MM-DD"
```

Run this from the repository root. The destination must be inside this checkout's
`data/` folder. Differing existing files cause an error; provenance is added to
local `data/index.csv`. This utility is not part of camera launch.

## Analysis entry points

[Angle reconstruction](docs/angle_reconstruction.md) documents the camera and
Polarcam APD paths, hole geometry, Fresnel/BFP coefficients and APD inverse/gain
conventions. The live GUI now uses their shared direct optical inverse; old
analysis galleries retain their original models.

Install notebook support in this environment when needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[analysis]"
.\.venv\Scripts\python.exe -m jupyterlab analysis
```

The maintained paper trials are in `analysis/paper_analysis_20261008/`:

- [APD hinge trial](analysis/paper_analysis_20261008/APD_hinge_publication_trial.ipynb)
  uses the legacy Hugh APD recordings.
- [Camera population trial](analysis/paper_analysis_20261008/Framestack_population_publication_trial.ipynb)
  uses camera frame stacks.
- `paper_figures.py` and `camera_trial.py` contain the associated processing.

Earlier dated projects retain their own entry scripts. Local generated reports,
figures, executed notebook backups and private correspondence are not portable
through Git. Configure local input paths and check the model/correction choices
before rerunning an analysis. This cleanup does not revise the analysis projects.

Standalone application utilities remain under `src/polarcam/live/`, for example:

```powershell
.\.venv\Scripts\python.exe src/polarcam/live/angle_distribution_analysis.py
.\.venv\Scripts\python.exe src/polarcam/live/tdms_manual_scaling_xy_gui.py
.\.venv\Scripts\python.exe -m polarcam.live.interference_npz_analysis RECORDING.npz
```

The TDMS utility retains an upstream machine-specific default; select your own
file. See [interference methods](docs/interference.md) for the diagnostic analysis.
The live application's distribution histogram and paper notebook plots use
different density conventions; consult the manual and each analysis's methods.

## Development and validation

`tests/` checks program behaviour; it is separate from experimental analysis.
Both commands are needed for the current suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests/test_interference_metrics.py
```

The second file contains standalone functions that `unittest` discovery does not
run. The 12 October camera-free pass covered 132 unittest checks and four metric
checks. Synthetic checks do not establish actual FPS, lost-frame behaviour, stage
response or experimental accuracy. Remaining historical findings are recorded in
[the rewrite notes](docs/archive/REWRITE_NOTES.md#other-deferred-findings).

## Provenance and history

Hugh Bowman's `polarcam_live/` implementation was imported from local Polarcam_v3
revision `5fe41d5bdd9f089a441f30554888c0b4773e62a7` (10 September 2026), on
15 September. The 17 upstream Python files were initially copied byte-for-byte
into `src/polarcam/live/` in commit `bb47c6490379f4c8ae90cbfe6d3e4315dad2792c`;
subsequent targeted changes mean the current tree is no longer identical.
Hugh developed the v3 application on Daping Xu's earlier camera software; the
upstream history also records contributions under Hugh T. This is a snapshot
import, not a merge of upstream history.

The existing [LICENSE](LICENSE) is unchanged. The imported checkout had no
top-level project licence; this import does not assign a new licence to Hugh's
contributions. Permission and applicable notices for redistribution remain to be
resolved. Calibration assets, backgrounds, recordings and the original Python
environment were not included in that source import. A Git checkout alone is
therefore not an equivalent deployment of the lab installation.

Earlier application versions are retained in Git at
`6c36303084b4ff799a277166df016b8b62e9c53d`. See the
[changelog](docs/changelog.md) for dated changes and
[archived lab notes](docs/archive/LAB_SETUP.md) for historical observations.
The Word manual is the maintained operating guide; archived logs are not current instructions.
