"""Copy local data with SHA-256 verification; preserve source names and sidecars.

Usage: python scripts/import_local_data.py SOURCE DESTINATION_WITHIN_DATA
The originals are kept. Provenance is added to the local data/index.csv.
"""
from pathlib import Path
import argparse
import csv
import hashlib
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
    manifest = ROOT/'data/index.csv'
    rows = {}
    fields = ['path', 'bytes', 'sha256', 'source_path']
    if manifest.exists():
        with manifest.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fields = list(dict.fromkeys([*(reader.fieldnames or []), *fields]))
            rows = {row['path']: row for row in reader}
    records = []
    for p in sorted(src.rglob('*')):
        if not p.is_file():
            continue
        q = dst/p.relative_to(src)
        if q.resolve() == manifest.resolve():
            raise ValueError('The data index is reserved; choose a dataset subfolder')
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
        key = q.relative_to(ROOT/'data').as_posix()
        row = rows.get(key, {})
        row.update(path=key, bytes=q.stat().st_size, sha256=before, source_path=str(p))
        rows[key] = row
        partial_manifest = manifest.with_suffix('.csv.import-partial')
        with partial_manifest.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows.values())
        partial_manifest.replace(manifest)
    print(f'Verified {len(records)} files, {sum(r["bytes"] for r in records)/1e9:.2f} GB: {dst}', flush=True)

if __name__ == '__main__':
    main()
