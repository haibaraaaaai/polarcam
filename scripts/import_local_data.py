"""Copy local data with SHA-256 verification; preserve source names and sidecars.

Usage: python scripts/import_local_data.py SOURCE DESTINATION_WITHIN_DATA
The originals are kept. A local provenance manifest is written under data/local.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('destination', type=Path)
    args = ap.parse_args()
    src, dst = args.source.resolve(), args.destination.resolve()
    dst.relative_to((ROOT/'data').resolve())
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError('Source and destination must not overlap')
    manifest = ROOT/'data/local'/('import_'+dst.name+'.json')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for p in sorted(src.rglob('*')):
        if not p.is_file():
            continue
        q = dst/p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        before = digest(p)
        if q.exists():
            if digest(q) != before:
                raise FileExistsError(f'Destination differs: {q}')
        else:
            partial = q.with_name(q.name+'.import-partial')
            shutil.copy2(p, partial)
            if digest(partial) != before:
                raise IOError(f'Copy verification failed: {p}')
            partial.replace(q)
        records.append(dict(source=str(p), destination=str(q.relative_to(ROOT)), bytes=q.stat().st_size, sha256=before))
        manifest.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Verified {len(records)} files, {sum(r["bytes"] for r in records)/1e9:.2f} GB: {dst}', flush=True)

if __name__ == '__main__':
    main()
