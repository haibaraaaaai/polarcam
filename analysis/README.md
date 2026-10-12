# Analysis workspace

Each dated directory is a distinct analysis snapshot; do not merge their scientific
assumptions simply to reduce the number of folders.

- `analysis_2026-10-06_firstpass`: original broad acquisition/background audit.
- `analysis_2026-10-07_bg_theta_gallery`: October angle/background gallery.
- `analysis_2026-10-07_october_consistency`: timing and interference-flag audit.
- `analysis_2026-10-07_september_oscillation`: September oscillation review.
- `paper_analysis_20261008`: APD/camera publication-figure trials and local email review.
- `analysis_2026-10-10_friday`: October/September spot, theta, saturation and
  interference comparisons; each script retains its stated historical model.

Git contains executable analysis source, notebook inputs and dependencies. Notebook
outputs are cleared before committing. Executed notebook backups, plots, tables,
caches, extracted documents, private email and narrative reports remain local and
ignored. In particular, nothing in `source_review/` belongs on GitHub.

Raw inputs are under `../data/hinges`, separated by detector. Dataset-relative paths
replace historical Desktop paths in maintained analysis scripts. Historical generated
reports and manifests are preserved as records and may contain old paths; rerun the
relevant analysis to regenerate them. Original notebook snapshots in `original_scripts`
retain their original code, with outputs removed; use the trial notebooks to run the
updated pipeline. The paper review's executed notebooks are in `executed_notebooks/`.

Recommended GitHub scope: application source + analysis source + these conventions.
Do not commit raw data, generated output or entire Python environments. Commit locally
first; publishing is a separate step. Scientific calibration and hardware changes
still require lab validation, regardless of Git status.
