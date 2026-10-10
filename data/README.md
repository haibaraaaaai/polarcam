# Local data layout

```
data/
  hinges/
    polarcam/import_2026-10-09/       # camera export, including older sessions
    polarcam/legacy_workspace_runs/  # former repo runs/ and root diagnostic
    apd_polarcam/legacy_hugh/         # Hugh APD recordings and accompanying reference material
    reference/                      # additional local sample/reference material
  bfm/
    polarcam/
    apd_polarcam/
  local/                            # private import hashes and path manifests
```

The date on an import folder is an acquisition/export label, not a claim that every
file inside was acquired that day. The October export also contains September and
early-October data. Preserve original basenames, subfolders and sidecars. In particular,
`temp`, `good`, `bad` and `pending` are historical application/review labels, not
validated sample identities or automatic scientific quality decisions.

Original Desktop inputs remain intact. Imported files are SHA-256 verified copies;
`local/import_*.json` records each original path, destination, size and hash. Raw JSON
sidecars are preserved byte-for-byte, including stale original absolute paths. Resolve
adjacent files locally before using those historical paths. No hinge/control label is
inferred solely from an angle or folder named good/bad.

This directory is excluded from Git except this policy, `catalog.json` and the BFM
placeholder README. A Git commit does not back up data. Keep the original/OneDrive
copies until a separate data-backup policy is established.
