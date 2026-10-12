#!/usr/bin/env python3
"""Controlled extraction audit of eight Polarcam five-frame recordings.

Candidate coordinates are held fixed at the previous intensity detector's outputs.
No detector is tuned, no rod identity is assigned, and no r or theta clipping is used.
The two principal curves use the SAME frames (legacy pair sums > 50 camera codes).

Author: ClawsGO Science Agent
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np

PARITIES = ((0, 0), (0, 1), (1, 0), (1, 1))  # I90, I45, I135, I0
RMAX = 0.9268984950974504  # Full pupil, NA=1.3, n=1.33; model reference only.


def anisotropy(channels):
    i90, i45, i135, i0 = np.moveaxis(channels, -1, 0)
    sx, sy = i0 + i90, i45 + i135
    good = np.isfinite(sx) & np.isfinite(sy) & (sx > 0) & (sy > 0)
    x = np.full(sx.shape, np.nan)
    y = np.full(sy.shape, np.nan)
    np.divide(i0 - i90, sx, out=x, where=good)
    np.divide(i45 - i135, sy, out=y, where=good)
    d = np.full(sx.shape, np.nan)
    np.divide(sx - sy, sx + sy, out=d, where=good)
    return np.hypot(x, y), d, sx, sy


def median_or_nan(values):
    return float(np.median(values)) if len(values) else np.nan


def audit_file(path, candidate_path, out):
    data = np.load(path, mmap_mode='r', allow_pickle=False)
    assert data.ndim == 3 and data.shape[0] == 5 and data.dtype == np.uint16
    tag = path.stem.removeprefix('first5_frame_stack_')
    with np.load(candidate_path, allow_pickle=False) as cache:
        spots = cache['spots'].copy().astype(int)
        historical_r = cache['R'].copy()
    t, height, width = data.shape
    n = len(spots)
    # Match the previous float32 arithmetic so threshold-boundary frames match.
    saved = np.empty((n, t, 4), np.float32)
    legacy = np.empty_like(saved)
    corner_bg = np.empty(n, np.float32)
    counts = np.empty((n, 4), int)
    corner_counts = np.empty((n, 4), int)
    near_upper = np.empty(n, bool)
    # Preserve the previous 7x7 channel means to isolate the subtraction itself.
    for i, (y, x) in enumerate(spots):
        assert y >= 3 and x >= 3 and y + 4 <= height and x + 4 <= width
        crop = np.asarray(data[:, y-3:y+4, x-3:x+4], dtype=np.float32)
        yy, xx = np.ogrid[y-3:y+4, x-3:x+4]
        dist2 = (yy-y)**2 + (xx-x)**2
        corners = (dist2 >= 16) & (dist2 <= 64)
        assert int(corners.sum()) == 4
        corner_bg[i] = np.median(crop.mean(axis=0)[corners])
        for k, (pr, pc) in enumerate(PARITIES):
            mask = (yy % 2 == pr) & (xx % 2 == pc)
            counts[i, k] = mask.sum()
            corner_counts[i, k] = (mask & corners).sum()
            saved[i, :, k] = crop[:, mask].mean(axis=1)
        legacy[i] = saved[i] - corner_bg[i]
        near_upper[i] = bool((crop >= 4080).any())  # Screen only, not saturation proof.
    assert np.all(np.sort(counts, axis=1) == [9, 12, 12, 16])
    assert np.all(np.sort(corner_counts, axis=1) == [0, 0, 0, 4])
    rs, ds, ssx, ssy = anisotropy(saved)
    rl, dl, slx, sly = anisotropy(legacy)
    common = (slx > 50) & (sly > 50)  # Exactly the legacy per-file validity gate.
    r_saved = np.array([median_or_nan(rs[i, common[i]]) for i in range(n)])
    r_legacy = np.array([median_or_nan(rl[i, common[i]]) for i in range(n)])
    d_saved = np.array([median_or_nan(ds[i, common[i]]) for i in range(n)])
    d_legacy = np.array([median_or_nan(dl[i, common[i]]) for i in range(n)])
    r_saved_all = np.median(rs, axis=1)
    valid = np.isfinite(r_legacy)
    assert np.array_equal(valid, np.isfinite(historical_r))
    np.testing.assert_allclose(r_legacy[valid], historical_r[valid], rtol=3e-5, atol=3e-6)
    # Algebraic sanity checks; float tolerances only.
    assert np.nanmax(rs) <= np.sqrt(2) + 1e-6
    assert np.all(rs[common] <= rl[common] + 1e-6)
    # No claim that a passing result is a rod. Save the raw comparison for review.
    np.savez_compressed(out / f'audit_{tag}.npz', spots=spots, channels_saved=saved,
                        corner_background=corner_bg, r_saved=r_saved, r_legacy=r_legacy,
                        d_saved=d_saved, d_legacy=d_legacy, common_frame_mask=common,
                        near_upper_codes=near_upper, r_saved_all=r_saved_all)
    summary = {
        'file': tag, 'shape': list(data.shape), 'dtype': str(data.dtype),
        'n_candidates': n, 'n_same_spots': int(valid.sum()),
        'n_common_spot_frames': int(common.sum()),
        'median_r_saved_same_frames': float(np.median(r_saved[valid])),
        'median_r_legacy_same_frames': float(np.median(r_legacy[valid])),
        'median_r_saved_all_candidates': float(np.median(r_saved_all)),
        'fraction_r_saved_gt1': float(np.mean(r_saved[valid] > 1)),
        'fraction_r_legacy_gt1': float(np.mean(r_legacy[valid] > 1)),
        'fraction_r_saved_gt_model': float(np.mean(r_saved[valid] > RMAX)),
        'fraction_r_legacy_gt_model': float(np.mean(r_legacy[valid] > RMAX)),
        'median_abs_d_saved': float(np.median(np.abs(d_saved[valid]))),
        'median_abs_d_legacy': float(np.median(np.abs(d_legacy[valid]))),
        'fraction_negative_legacy_common_frames': float(np.mean(np.any(legacy < 0, axis=2)[common])),
        'median_corner_background': float(np.median(corner_bg)),
        'near_upper_candidate_count': int(near_upper.sum()),
        'zero_fraction_all_5_frames': float(sum(np.count_nonzero(frame == 0) for frame in data) / data.size),
        'maximum_saved': int(data.max()),
        'sample_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    print(f"{tag}: N={summary['n_same_spots']}; median r "
          f"{summary['median_r_legacy_same_frames']:.3f} -> {summary['median_r_saved_same_frames']:.3f}; "
          f"r>model {summary['fraction_r_legacy_gt_model']:.1%} -> {summary['fraction_r_saved_gt_model']:.1%}", flush=True)
    return summary


def unit_tests():
    # Positive channels and equal pair sums can still be inconsistent Stokes data.
    # Order = I90, I45, I135, I0.
    r, d, *_ = anisotropy(np.array([[10., 190., 10., 190.]]))
    assert abs(float(r[0])-np.sqrt(2)*0.9) < 1e-12 and d[0] == 0
    # A physical input with D=0 exceeds r=1 after subtracting too much scalar bg.
    r0, d0, *_ = anisotropy(np.array([[50., 150., 50., 150.]]))
    r1, d1, *_ = anisotropy(np.array([[50., 150., 50., 150.]])-40)
    assert r0[0] < 1 < r1[0] < np.sqrt(2) and d0[0] == d1[0] == 0
    yy, xx = np.mgrid[-3:4, -3:4]
    m=(yy*yy+xx*xx >= 16)&(yy*yy+xx*xx <=64)
    assert m.sum() == 4 and np.all(yy[m]%2 == 1) and np.all(xx[m]%2 == 1)
    print('Algebra and geometry unit tests: PASS', flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--candidate-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    unit_tests()
    summaries=[]
    for path in sorted(args.data_dir.glob('first5_frame_stack_*.npy')):
        tag=path.stem.removeprefix('first5_frame_stack_')
        summaries.append(audit_file(path,args.candidate_dir/f'perfile_{tag}.npz',args.out))
        (args.out/'extraction_ablation_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    assert len(summaries)==8, f'Expected 8 stacks; audited {len(summaries)}'
    print('All eight matched-coordinate, matched-frame comparisons: PASS',flush=True)


if __name__=='__main__':
    main()
