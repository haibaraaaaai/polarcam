# Data

```
hinges/
  polarcam/YYYY-MM-DD/      # frame stacks and spot recordings
  apd_polarcam/legacy_hugh/
bfm/
  polarcam/
  apd_polarcam/
```

Within a camera date, backgrounds, stationary batches and derived outputs have
their own folders. Original filenames and JSON sidecars are unchanged.
`auxiliary/` holds old stop flags/logs; `duplicates/` retains a verified extra copy.

Local `index.csv` lists files, labels, notes, old/source paths and import hashes.
Filter `kind = frame stack` or `spot recording` to hide supporting files.
`unknown` and `hinges_unconfirmed` are not verified sample labels; `conflict`
means recorded drive fields disagree. All drive states are operator declarations.
Edit index labels as details are confirmed; keep the original sidecars intact.

Only this guide, the catalogue and four `.gitkeep` placeholders enter Git.
Recordings and the index stay local. Git does not back them up.
