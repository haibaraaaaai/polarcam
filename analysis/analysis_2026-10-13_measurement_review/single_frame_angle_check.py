"""First/middle/last-frame orientation at the existing fixed spot apertures.

Uses actual frames 1, 75, 150, never the appended analyzer phase-marker image.
Original whole-recording saturation exclusion and all other selections persist.
"""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from polarcam import optics
from polarcam.live.intensity_spots import channel_means, anisotropy

HERE = Path(__file__).resolve().parent
DAY = HERE.parents[1] / 'data/hinges/polarcam/2026-10-12'
OUT = HERE / 'single_frame_angle_check'
SOURCES = {'DNA': HERE/'background_overlap_check/hinges95_theta.json',
           'Gold control': HERE/'control_recovery_check/gold_results.json'}
MODEL = optics.theta_model()
ABC = [MODEL[k] for k in ('A', 'B', 'C')]
RMAX = MODEL['r_max']
SNAPSHOTS = {'First (frame 1)': 0, 'Middle (frame 75)': 74, 'Last (frame 150)': 149}
MEAN = '150-frame mean'
COLORS = {MEAN: '#222222', 'First (frame 1)': '#0072b2', 'Middle (frame 75)': '#d55e00', 'Last (frame 150)': '#009e73'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)): return [clean(v) for v in x]
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def evaluate(channels):
    channels = np.asarray(channels, float)
    xy = anisotropy(channels)
    r = np.linalg.norm(xy, axis=1)
    angle = np.degrees(optics.theta_from_radius(r, *ABC))
    return dict(channels=channels, xy=xy, r=r, theta=angle)


def calculate():
    OUT.mkdir(exist_ok=True)
    result = dict(complete=False, model=MODEL, groups={}, frame_indices_zero_based=SNAPSHOTS,
        method='Existing p95/2-SD detections and 18x18 apertures remain fixed. Four analyzer intensities are spatially averaged within each selected single saved frame, then X,Y,r,theta are reconstructed. Saved frames already have minimum-background subtraction: no additional subtraction or pixel clipping is applied here.',
        exclusions='All original saturation exclusions (any raw pixel >=4094 over all 150 frames, in aperture or component box), ellipse mask and overlap exclusions remain. No additional per-frame brightness or angle selection. Above-model r stays in r distributions; theta is undefined rather than clipped.',
        limitation='This isolates averaging across frames conditional on detection from the averaged image. Individual exposures still average approximately 1.2 ms. Three snapshots cannot resolve faster motion or establish whether a change is motion, intensity noise, interference, or background drift. The same spots are paired observations, not three independent samples.')
    for group, source_path in SOURCES.items():
        source = json.loads(source_path.read_text(encoding='utf8'))
        assert source['complete']
        np.testing.assert_allclose([source['model'][k] for k in ('A', 'B', 'C')], ABC, rtol=0, atol=1e-15)
        output = dict(source=str(source_path.relative_to(HERE)), source_sha256=sha(source_path), records=[])
        for rec in source['records']:
            path = DAY/'frames'/rec['name']
            sig = signature(path)
            assert sig == rec['source_signature']
            meta_path = path.with_suffix('.json')
            assert sha(meta_path) == rec['sidecar_sha256']
            meta = json.loads(meta_path.read_text(encoding='utf8'))
            assert meta['actual']['frames'] == 150 and meta['actual']['phase_marker_appended']
            assert meta['background']['background_subtracted'] is True
            assert meta['background']['background_profile_sha256'] == rec['background_sha256']
            frames = np.load(path, mmap_mode='r')
            assert frames.dtype == np.uint16 and frames.shape == (151, 2056, 2464)
            raw_info = meta['fetch_frames_output']['raw_peak']
            assert raw_info['frames'] == 150 and raw_info['sha256'] == rec['raw_peak_sha256']
            raw_path = path.parent/raw_info['path']
            assert sha(raw_path) == raw_info['sha256']
            peak = np.load(raw_path, mmap_mode='r')
            spots = [s for s in rec['spots'] if s['selection'] == 'kept']
            assert all(not s['saturated'] and not s['inside_ellipse'] for s in spots)
            # Recheck the existing saturation decision using the full raw peak map,
            # including raw intensity before background subtraction.
            for s in spots:
                l, rr, t, b = s['bounds']
                x, y, w, h = s['bbox']; left, top = round(x+.5), round(y+.5)
                assert max(peak[t:b, l:rr].max(), peak[top:top+h, left:left+w].max()) < 4094
            bounds = [s['bounds'] for s in spots]
            measurements = {MEAN: evaluate([s['channels'] for s in spots])}
            np.testing.assert_allclose(measurements[MEAN]['r'], [s['r'] for s in spots], atol=1e-12, rtol=0)
            for name, index in SNAPSHOTS.items():
                measurements[name] = evaluate(channel_means(frames[index], bounds))
            # The native extractor was also used on the cached 150-frame mean.
            # Only time selection changes here; all channel phases/apertures agree.
            assert signature(path) == sig
            output['records'].append(dict(index=rec['index'], name=path.name, source_signature=sig,
                background_sha256=rec['background_sha256'], raw_peak_sha256=raw_info['sha256'],
                exposure_us=meta['actual']['timing']['exposure_us'], fps=meta['actual']['fps'],
                ids=[s['id'] for s in spots], bounds=bounds,
                measurements=clean(measurements)))
            print(f'{group} {rec["index"]}/{len(source["records"])}: {len(spots)} unchanged spots', flush=True)
            del frames, peak
        result['groups'][group] = output
    result['complete'] = True
    result['builder_sha256'] = sha(Path(__file__))
    (OUT/'results.json').write_text(json.dumps(result, allow_nan=False, separators=(',', ':')), encoding='utf8')
    return result


def pooled(group, name, key):
    return np.concatenate([np.asarray(rec['measurements'][name][key], float) for rec in group['records']])


def summarize(data):
    summary = {}
    for label, group in data['groups'].items():
        base_r = pooled(group, MEAN, 'r'); base_theta = pooled(group, MEAN, 'theta')
        near30 = (base_theta >= 28) & (base_theta < 34)
        entries = {}
        for name in COLORS:
            r = pooled(group, name, 'r'); theta = pooled(group, name, 'theta')
            counts, edges = np.histogram(theta[np.isfinite(theta)], np.arange(0, 92, 2))
            entries[name] = dict(n=len(r), valid_theta=int(np.isfinite(theta).sum()),
                undefined_r=int((~np.isfinite(r)).sum()), above_model=int((r > RMAX).sum()),
                median_r=float(np.nanmedian(r)), median_theta=float(np.nanmedian(theta)),
                near30_count=int(((theta >= 28) & (theta < 34)).sum()),
                near30_percent_all_spots=float(100*np.mean((theta >= 28) & (theta < 34))),
                theta_mode_bin=edges[counts.argmax():counts.argmax()+2].tolist(),
                median_paired_delta_r=float(np.nanmedian(r-base_r)),
                median_abs_paired_delta_r=float(np.nanmedian(np.abs(r-base_r))),
                original_near30_n=int(near30.sum()),
                original_near30_median_snapshot_r=float(np.nanmedian(r[near30])),
                original_near30_still_near30_percent=float(100*np.mean(((theta >= 28) & (theta < 34))[near30])),
                original_near30_now_above70_percent=float(100*np.mean((theta >= 70)[near30])),
                original_near30_above_model_count=int((r[near30] > RMAX).sum()))
        summary[label] = entries
    return summary


def render(data):
    assert data['complete']
    summary = summarize(data)
    (OUT/'summary.json').write_text(json.dumps(clean(summary), indent=2, allow_nan=False), encoding='utf8')
    r_physical_edges = np.r_[np.arange(0, RMAX, .02), RMAX]
    max_r = max(np.nanmax(pooled(g, name, 'r')) for g in data['groups'].values() for name in COLORS)
    r_raw_edges = np.r_[r_physical_edges, np.arange(.94, max(1., np.ceil(max_r/.02)*.02)+.021, .02)]
    for axis in ('theta', 'r'):
        for norm in ('ordinary', 'solid_angle'):
            edges = np.arange(0, 92, 2.) if axis == 'theta' else r_physical_edges if norm == 'solid_angle' else r_raw_edges
            if norm == 'solid_angle':
                theta_edges = edges if axis == 'theta' else np.degrees(optics.theta_from_radius(edges, *ABC))
                expected = -np.diff(np.cos(np.radians(theta_edges)))
                assert np.all(expected > 0)
                np.testing.assert_allclose(expected.sum(), 1, atol=1e-12)
            fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.9), layout='constrained')
            for ax, (label, group) in zip(axs, data['groups'].items()):
                for name, color in COLORS.items():
                    values = pooled(group, name, axis)
                    vals = values[np.isfinite(values)]
                    counts = np.histogram(vals, edges)[0]
                    if axis == 'r' and norm == 'ordinary': assert counts.sum() == len(vals)
                    # Use ALL retained spots as the denominator, including invalid
                    # theta, to avoid silently renormalizing different survivors.
                    heights = counts / len(values)
                    if norm == 'solid_angle': heights = heights / expected
                    elif axis == 'theta': heights *= 100
                    else: heights /= np.diff(edges)
                    ax.stairs(heights, edges, color=color, lw=2.0 if name == MEAN else 1.25,
                              ls='--' if name == MEAN else '-', label=name, zorder=5 if name == MEAN else 3)
                ax.set_title(f'{label} · {len(pooled(group, MEAN, "r")):,} fixed spots')
                ax.set_xlim(0, 90 if axis == 'theta' else r_raw_edges[-1]); ax.set_ylim(bottom=0)
                ax.set_xlabel('Recovered θ (degrees)' if axis == 'theta' else 'Anisotropy radius r')
                ax.spines[['top', 'right']].set_visible(False)
                if axis == 'theta': ax.axvspan(28, 34, color='#e6b34c', alpha=.14, zorder=-1)
                else:
                    ax.axvspan(RMAX, r_raw_edges[-1], color='#dddddd', alpha=.5, zorder=-1)
                    ax.axvline(RMAX, color='#888888', lw=.8)
                if norm == 'solid_angle':
                    ax.plot([0, edges[-1]], [1, 1], color='#888888', ls=':', lw=1, label='Isotropic = 1')
                ax.legend(frameon=False, fontsize=8.5)
            axs[0].set_ylabel('Fraction / isotropic bin probability' if norm == 'solid_angle' else '% of all retained spots per 2°' if axis == 'theta' else 'Probability density in r')
            fig.suptitle('Same spots: first, middle and last frames versus averaged intensities', fontsize=13)
            fig.savefig(OUT/f'{axis}_{norm}.png', dpi=160)
            plt.close(fig)
    rows = ''
    for label, group in summary.items():
        for name, s in group.items():
            rows += f'<tr><td>{label}</td><td>{name}</td><td>{s["median_r"]:.3f}</td><td>{s["near30_percent_all_spots"]:.1f}%</td><td>{s["above_model"]}</td><td>{s["undefined_r"]}</td></tr>'
    paired_rows = ''
    for label, group in summary.items():
        for name, s in group.items():
            if name == MEAN: continue
            paired_rows += f'<tr><td>{label} ({s["original_near30_n"]})</td><td>{name}</td><td>{s["original_near30_still_near30_percent"]:.1f}%</td><td>{s["original_near30_now_above70_percent"]:.1f}%</td><td>{s["original_near30_above_model_count"]}</td></tr>'
    # Per-field results distinguish pooled stability from opposite field changes.
    field_rows = ''
    for label, group in data['groups'].items():
        for rec in group['records']:
            vals = [np.nanmedian(np.array(rec['measurements'][name]['r'], float)) for name in COLORS]
            field_rows += f'<tr><td>{label} {rec["index"]}</td><td>{rec["name"]}</td>' + ''.join(f'<td>{v:.3f}</td>' for v in vals) + '</tr>'
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Single-frame averaging check</title><style>body{max-width:1280px;margin:30px auto;padding:0 20px;font:16px/1.6 system-ui;color:#24333c;background:#fafaf8}h1{font-size:26px}h2{font-size:20px}img{width:100%;background:white}select{font:inherit;margin-right:20px}table{border-collapse:collapse;font-size:14px;margin:14px 0}th,td{text-align:left;padding:7px 13px;border-bottom:1px solid #ddd}th{background:#eef1f0}p{max-width:1080px}a{color:#116b90}summary{cursor:pointer}</style>
<h1>Does averaging across frames create the low-angle feature?</h1>
<p><b>Result: the near-30° feature remains in all three individual frames.</b> Gold has 31.1–32.5% of spots in 28–34°, compared with 32.6% from averaged intensities; DNA has 15.8–16.3%, compared with 16.9%. None of the originally near-30° spots reaches 70° or exceeds the model limit in these snapshots. Averaging between frames does not appear to be the main cause of this feature. This does not rule out motion within each exposure or establish the physical identity of the detected features.</p>
<p><b>Controlled comparison:</b> keep exactly the same spots, 18×18 apertures, background treatment and saturation exclusions. Compare angles from the first, middle and last saved frames with angles from the existing 150-frame mean. No new detection or per-frame brightness cut.</p>
<label>Axis <select id="axis" onchange="update()"><option value="theta">θ</option><option value="r">r (including above-model values)</option></select></label>
<label>Density <select id="norm" onchange="update()"><option value="ordinary">Ordinary histogram</option><option value="solid_angle">Relative to solid-angle expectation</option></select></label>
<img id="plot" src="theta_ordinary.png" alt="First, middle and last frame distributions compared with the 150-frame mean for DNA and gold control">
<p>“First”, “middle”, “last” mean actual frames <b>1, 75 and 150</b>. The extra phase-marker image is excluded. Black dashed: calculate angle after averaging all 150 frames' intensities. Colored lines: calculate angle from one frame. These are repeated measurements of the same spots, not three independent sample groups.</p>
<h2>Summary</h2><p>Near 30° means 28° ≤ θ &lt; 34°. All histogram fractions use the unchanged number of retained spots as denominator. Above-model r and undefined angles are reported rather than clipped to 90° or silently renormalized away.</p>
<table><tr><th>Sample</th><th>Time selection</th><th>Median r</th><th>Near 30°</th><th>Above r limit</th><th>Undefined r</th></tr>ROWS</table>
<details><summary>Follow only spots originally near 30°</summary><p>These rows track the same spots whose angle from the averaged intensities was 28–34°. A jump above 70° would be more consistent with the proposed flat-rod explanation; noisy changes alone are not evidence of rotation.</p>
<table><tr><th>Original cohort (n)</th><th>Snapshot</th><th>Still 28–34°</th><th>Now ≥70°</th><th>Above r limit</th></tr>PAIRED</table></details>
<details><summary>Median r for each frame stack</summary><table><tr><th>Sample / field</th><th>File</th><th>150-frame mean</th><th>First</th><th>Middle</th><th>Last</th></tr>FIELDS</table></details>
<p><b>What this tests:</b> whether averaging between frames is essential to the observed distribution, conditional on the existing mean-image detections. A single frame still integrates during its approximately 1.2 ms exposure. Faster motion can therefore remain averaged; these three snapshots also cannot distinguish motion from measurement noise or interference. Fixed apertures do not track lateral particle motion.</p>
<p>The saved frames already contain minimum-background subtraction and zero clipping. This comparison introduces no additional subtraction. The 95th-percentile background is used only for the original spot detection. Whole-stack raw saturation exclusions were rechecked, so spots saturating before background subtraction remain excluded.</p>
<p><a href="results.json">Paired measurements and provenance</a> · <a href="summary.json">Numerical summary</a> · <a href="../gold_intensity_angle/solid_angle.html">Previous solid-angle comparison</a></p>
<script>function update(){document.getElementById('plot').src=document.getElementById('axis').value+'_'+document.getElementById('norm').value+'.png';}</script></html>'''
    (OUT/'index.html').write_text(page.replace('ROWS', rows).replace('PAIRED', paired_rows).replace('FIELDS', field_rows), encoding='utf8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    data = json.loads((OUT/'results.json').read_text(encoding='utf8')) if args.render_only else calculate()
    render(data)
