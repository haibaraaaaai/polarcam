"""Controlled 1x-radius comparisons across detectors and 2x-survival selection.

Read-only acquisitions. All methods use 150 frames and the same 18x18 extraction.
Only candidate positions and the optional 2x positional-match requirement vary.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from polarcam.live import Detection_alg_offline as native
from polarcam.live import intensity_spots
import background_sensitivity as bg

HERE = Path(__file__).resolve().parent
OUT = HERE/'radius_detector_comparison'
PRIOR = HERE/'matched_background_theta/results.json'
OLD = HERE.parent/'analysis_2026-10-10_friday'
sys.path.insert(0, str(OLD))
from local_spot_check import filtering, local_metrics

SUPPLIED = OLD/'pipeline_audit/package/detector_intensity_reference.py'
spec = importlib.util.spec_from_file_location('supplied_intensity_reference', SUPPLIED)
supplied = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supplied)
METHODS = {
    'native': dict(label='GUI DoG', description='Polarizer-cell intensity map; DoG sigma 1/6 px; global mean + 8 SD; component area 10–250 px.'),
    'local': dict(label='GUI DoG + local contrast', description='Same GUI candidates, then local contrast score ≥8 in the 12–30 px ring, excluding 8 px around other candidates. Applied at both 1× and 2×. A positive peak above a zero-MAD reference passes; a flat core does not. Exploratory quality cutoff.'),
    'supplied': dict(label='Supplied September detector', description='Raw polarization mosaic; hot-pixel heuristic; DoG median + 8 robust MAD; area 10–250 px; additional broad-object pass. Run on the 150-frame mean, with no legacy brightness gate.'),
}
GROUPS = ('Hinges', 'Gold')
LABELS = {'Hinges': 'DNA / hinges', 'Gold': 'Gold control'}
COLORS = {'Hinges': '#147a8d', 'Gold': '#c07812'}
EDGES = np.arange(72, dtype=float)*.02


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def save(report):
    path = OUT/'results.json'
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(clean(report), separators=(',', ':'), allow_nan=False), encoding='utf-8')
    temp.replace(path)


def score_only(field, points, j):
    """Same local DoG score, without unrelated raw-image contrast division.

    Zero reference MAD with a strictly higher peak means unbounded spatial
    contrast. This is a threshold heuristic, never an independent-noise sigma.
    """
    x,y = points[j]
    ix,iy = round(x),round(y)
    e = field['edge']
    h,w = field['image'].shape
    yy,xx = np.mgrid[max(e,iy-32):min(h-e,iy+33), max(e,ix-32):min(w-e,ix+33)]
    radial = np.hypot(xx-x,yy-y)
    mask = (radial>=12)&(radial<=30)&(xx%2==0)&(yy%2==0)
    for k,(cx,cy) in enumerate(points):
        if k!=j and np.hypot(cx-x,cy-y)<39:
            mask &= np.hypot(xx-cx,yy-cy)>8
    values = field['dog'][yy-e,xx-e][mask]
    if len(values)<40: raise ValueError('Too few neighbouring samples')
    median = float(np.median(values))
    spread = float(1.4826*np.median(np.abs(values-median)))
    label = field['label_ids'][j]
    bx,by,bw,bh,_ = field['stats'][label]
    patch = field['dog'][by:by+bh,bx:bx+bw]
    peak = float(patch[field['labels'][by:by+bh,bx:bx+bw]==label].max())
    return (peak-median)/spread if spread>0 else (np.inf if peak>median else 0.)


def local_scores(field):
    points = field['centres']
    scores, failures, zero_division_handled = [], [], []
    for i in range(len(points)):
        try:
            value = local_metrics(field, points, i)['local_score']
            np.testing.assert_allclose(value, score_only(field, points, i), atol=1e-12, rtol=1e-12)
            scores.append(value)
        except ZeroDivisionError:
            scores.append(score_only(field, points, i))
            zero_division_handled.append(i+1)
        except ValueError as exc:
            scores.append(np.nan)
            failures.append(dict(id=i+1, reason=str(exc)))
    scores = np.array(scores)
    keep = ~np.isnan(scores) & (scores >= 8)
    return scores, keep, failures, zero_division_handled


def detect(mean):
    field = filtering(native.intensity_detection_map(mean), native)
    points = field['centres']
    scores, keep, failures, handled = local_scores(field)
    old_points, info = supplied.detect_candidates(mean.astype(np.float32)[None])
    return dict(native=points, local=points[keep], supplied=old_points[:, ::-1].astype(float)), dict(
        native_threshold=field['threshold'], local_scores=scores, local_ids=np.flatnonzero(keep)+1,
        local_failures=failures, local_zero_division_handled=handled,
        local_zero_mad_pass_ids=np.flatnonzero(np.isposinf(scores))+1, supplied=info)


def compute():
    OUT.mkdir(exist_ok=True)
    source_bytes = PRIOR.read_bytes()
    previous = json.loads(source_bytes)
    paths = [Path(native.__file__), Path(intensity_spots.__file__), SUPPLIED,
             OLD/'local_spot_check.py', HERE/'background_sensitivity.py']
    signature = dict(version=2, source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                     code_sha256={str(p): bg.digest(p) for p in paths})
    report = dict(signature=signature, methods=METHODS, records=[], r_max=previous['model']['r_max'],
        bins=EDGES, bin_width=.02, data_date='2026-10-12', complete=False,
        settings=dict(measurement_frames=150, extraction_width=18, raw_ceiling_veto=4094,
                      match_radius=6, background='Recorded 1x; extra 2x subtraction clips per frame before the mean.',
                      radius='From mean channels at the original 1x centre; no theta/radius cut or clipping.',
                      matching='Same recipe at 1x and 2x; one-to-one spatial matching independently within each stack.',
                      weighting='Pooled observation counts; no cross-stack deduplication or claims of independent rods.'))
    if (OUT/'results.json').exists():
        existing = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
        if existing['signature'] == signature:
            report['records'] = existing['records']
    by_name = {r['name']: r for r in report['records']}
    backgrounds = {}
    started = time.perf_counter()
    for order, rec in enumerate(previous['records'], 1):
        path = bg.DAY/'frames'/rec['name']
        stat = path.stat()
        assert stat.st_size == rec['source_bytes'] and stat.st_mtime_ns == rec['source_mtime_ns']
        if rec['name'] in by_name:
            print(f'{order}/30 cached {rec["group"]} {rec["group_index"]}', flush=True)
            continue
        meta_path = path.with_suffix('.json')
        assert bg.digest(meta_path) == rec['sidecar_sha256']
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        assert meta['background']['background_subtracted']
        sha = rec['background_sha256']
        assert sha == meta['background']['background_profile_sha256']
        if sha not in backgrounds:
            bpath = bg.DAY/'backgrounds/used'/f'{sha}.npy'
            assert bg.digest(bpath) == sha
            backgrounds[sha] = np.load(bpath, allow_pickle=False).astype(np.float32)
        background = backgrounds[sha]
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        assert len(a)-int(meta['actual']['phase_marker_appended']) == meta['actual']['frames'] == 150
        sums = np.zeros((2, *a.shape[1:]), dtype=np.float64)
        work = np.empty_like(background)
        for frame in a[:150]:
            sums[0] += frame
            np.subtract(frame, background, out=work)
            np.maximum(work, 0, out=work)
            sums[1] += work
        means = sums/150
        del sums, a, work
        points1, info1 = detect(means[0])
        points2, info2 = detect(means[1])
        np.testing.assert_allclose(points1['native'], rec['factors'][0]['native']['points'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(points2['native'], rec['factors'][-1]['native']['points'], atol=1e-12, rtol=0)
        peakpath = path.parent/meta['fetch_frames_output']['raw_peak']['path']
        assert bg.digest(peakpath) == rec['raw_peak_sha256']
        peak = np.load(peakpath, mmap_mode='r', allow_pickle=False)
        origin = (int(meta['roi']['OffsetX']), int(meta['roi']['OffsetY']))
        record = {key: rec[key] for key in ('name','group','group_index','time','stamp','note','background',
                    'background_sha256','source_bytes','source_mtime_ns','sidecar_sha256','raw_peak_sha256')}
        record.update(methods={}, detection_info_1x=info1, detection_info_2x=info2)
        for method in METHODS:
            base, other = points1[method], points2[method]
            paired = bg.compare(base, other)
            matched = np.ones(len(base), dtype=bool)
            matched[np.array(paired['lost_ids'], dtype=int)-1] = False
            bounds = [(round(x)-9, round(x)+9, round(y)-9, round(y)+9) for x,y in base]
            assert all(0<=l<r<=means.shape[2] and 0<=t<b<=means.shape[1] for l,r,t,b in bounds)
            channels = intensity_spots.channel_means(means[0], bounds, origin)
            xy = intensity_spots.anisotropy(channels)
            radius = np.linalg.norm(xy, axis=1)
            raw_peaks = [int(peak[t:b,l:r].max()) for l,r,t,b in bounds]
            spots = [dict(id=i+1, x=float(x), y=float(y), bounds=bounds[i], channels=channels[i],
                          xy=xy[i], r=radius[i], raw_peak=raw_peaks[i],
                          ceiling_veto=raw_peaks[i]>=4094, matched=bool(matched[i]))
                     for i,(x,y) in enumerate(base)]
            if method == 'native':
                for s, old in zip(spots, rec['spots']):
                    np.testing.assert_allclose(s['channels'], old['channels_1x'], atol=1e-10, rtol=1e-12)
                    assert s['matched'] == old['matched_native'] and s['raw_peak'] == old['raw_peak']
                    np.testing.assert_allclose(s['r'], np.nan if old['r_1x'] is None else old['r_1x'], equal_nan=True)
            record['methods'][method] = dict(spots=spots, count_2x=len(other), matched=int(matched.sum()),
                                            points_2x=other, baseline_ids_2x=paired['baseline_ids'])
        assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns
        report['records'].append(clean(record))
        save(report)
        counts = {m: f'{len(record["methods"][m]["spots"])}->{record["methods"][m]["matched"]}' for m in METHODS}
        print(f'{order}/30 {rec["group"]} {rec["group_index"]}: {counts}; elapsed {time.perf_counter()-started:.0f}s', flush=True)
        del means, peak
    assert len(report['records']) == 30 and {r['name'] for r in report['records']} == {r['name'] for r in previous['records']}
    report['complete'] = True
    save(report)
    assert PRIOR.read_bytes() == source_bytes
    return clean(report)



def summarize(report):
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for group in GROUPS:
            records = [r for r in report['records'] if r['group'] == group]
            spots = [s for r in records for s in r['methods'][method]['spots']]
            usable = [s for s in spots if not s['ceiling_veto'] and s['r'] is not None]
            entry = dict(stacks=len(records), detected=len(spots), ceiling=sum(s['ceiling_veto'] for s in spots),
                         undefined=sum(not s['ceiling_veto'] and s['r'] is None for s in spots))
            for mode in ('all','matched','removed'):
                v = np.array([s['r'] for s in usable if mode=='all' or s['matched']==(mode=='matched')])
                counts = np.histogram(v, EDGES)[0]
                assert counts.sum() == len(v)
                entry[mode] = dict(n=len(v), median=float(np.median(v)) if len(v) else None,
                    q25=float(np.percentile(v,25)) if len(v) else None,
                    q75=float(np.percentile(v,75)) if len(v) else None,
                    outside_model=int(np.count_nonzero(v>report['r_max'])),
                    counts=counts, percent=100*counts/len(v) if len(v) else counts)
            assert entry['all']['n'] == entry['matched']['n'] + entry['removed']['n']
            summary[method][group] = entry
    return clean(summary)


def plot_rows(report, methods, stem):
    n = len(methods)
    fig, axes = plt.subplots(n, 2, figsize=(12.6, 4.1*n+.95), squeeze=False, sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.98, bottom=.13 if n==1 else .075,
                        top=.78 if n==1 else .91, hspace=.3, wspace=.12)
    summary = report['summary']
    maximum = max(max(summary[m][g][v]['percent']) for m in METHODS for g in GROUPS for v in ('all','matched'))
    ymax = max(8, np.ceil(maximum+1))
    for row, method in enumerate(methods):
        for col, mode in enumerate(('all','matched')):
            ax = axes[row,col]
            ax.axvline(report['r_max'], color='#667c83', ls='--', lw=1)
            ax.axvspan(report['r_max'], EDGES[-1], color='#8b9ca0', alpha=.06)
            for group in GROUPS:
                s = summary[method][group][mode]
                ax.stairs(s['percent'], EDGES, color=COLORS[group], lw=1.8,
                          label=f'{LABELS[group]} · n={s["n"]:,} · median {s["median"]:.3f}')
                ax.stairs(s['percent'], EDGES, color=COLORS[group], alpha=.055, fill=True)
            status = 'Without 2× survival filter' if mode=='all' else 'With 2× survival filter'
            ax.set(title=METHODS[method]['label']+'\n'+status, xlim=(0,EDGES[-1]), ylim=(0,ymax),
                   xticks=np.arange(0,1.41,.2))
            ax.legend(frameon=False, fontsize=9, loc='upper left')
            ax.grid(axis='y',alpha=.18)
            if col==0: ax.set_ylabel('Observations per 0.02-r bin (%)')
            if row==n-1: ax.set_xlabel('Radius r from original 1× BG intensities')
    fig.suptitle('Does requiring survival at 2× BG change the 1× radius distributions?', fontsize=15, y=.985 if n>1 else .98)
    fig.text(.5,.013,'All finite r included; dashed line is the optical-model limit, not an exclusion. Raw-ceiling veto retained.\n'
             'Same 150-frame input and 18 × 18 extraction for every recipe. Each group normalized separately; observations may repeat rods.',
             ha='center',fontsize=9,color='#526a70')
    for ext in ('png','svg'): fig.savefig(OUT/f'{stem}.{ext}',dpi=170)
    plt.close(fig)


def render(report):
    assert report['complete']
    report['summary'] = summarize(report)
    report['analysis_source_sha256'] = bg.digest(Path(__file__))
    previous = json.loads(PRIOR.read_text(encoding='utf-8'))
    for group in GROUPS:
        old = previous['theta_summary']['native'][group]
        new = report['summary']['native'][group]['matched']
        assert new['n'] == old['unsaturated_matched']
        assert new['outside_model'] == old['1x']['invalid']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','svg.fonttype':'none'})
    plot_rows(report, list(METHODS), 'comparison')
    for method in METHODS: plot_rows(report, [method], method)
    save(report)
    small = {key: report[key] for key in ('methods','summary','r_max','settings','signature')}
    (OUT/'index.html').write_text(HTML.replace('__DATA__',json.dumps(small,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    print('SUMMARY',json.dumps({m:{g:{mode:s[mode]['n'] for mode in ('all','matched','removed')} for g,s in groups.items()}
                              for m,groups in report['summary'].items()}),flush=True)
    print(OUT/'index.html',flush=True)


HTML = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Radius: detectors and BG survival filter</title><style>
body{margin:0;background:#f5f7f3;color:#243b40;font:16px/1.5 system-ui,sans-serif}main{max-width:1450px;padding:24px;margin:auto}h1{font-size:29px}p{max-width:1150px}a{color:#08788c}select{font:inherit;padding:7px;border:1px solid #a0b7b0;border-radius:5px}.plot{display:block;width:100%;height:auto;background:white;margin:18px 0}.box{background:white;border:1px solid #d2dfd7;padding:16px;margin:18px 0;border-radius:7px}.small{font-size:13px;color:#536c70}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #d2dfd7;text-align:left;white-space:nowrap}summary{cursor:pointer}
</style></head><body><main><div class="small">12 OCTOBER · 16 DNA STACKS + 14 GOLD SURVEY STACKS</div><h1>Spot detection and the 2× BG survival filter</h1>
<p><b>Every distribution here uses the original 1× BG intensities and includes all finite radii.</b> The two columns differ only in whether a candidate must also be detected at 2× BG. Raw-ceiling exclusions remain in both. Every detector sees the same 150-frame mean; every radius uses the same balanced 18 × 18 window.</p>
<label>Detector view <select id="method"><option value="comparison">All three recipes</option><option value="native">GUI DoG</option><option value="local">GUI DoG + local contrast</option><option value="supplied">Supplied September detector</option></select></label>
<img class="plot" id="plot" src="comparison.png" alt="DNA and gold radius distributions, comparing detection recipes with and without the 2x survival filter">
<p class="small">“Without” retains all finite, unsaturated 1× detections. “With” additionally requires a one-to-one match within 6 sensor pixels to a 2× candidate from the same recipe. Failure to match can also arise from a moving centre, splitting, merging, or a threshold change; it does not establish that a candidate was background.</p>
<div class="box scroll"><table><thead><tr><th>Recipe</th><th>Group</th><th>Detected</th><th>Raw ceiling</th><th>Undefined r</th><th>Without filter</th><th>With filter</th><th>Removed</th><th>Median r: without → with</th><th>Removed median r</th><th>r &gt; model: without / with</th></tr></thead><tbody id="counts"></tbody></table></div>
<div class="box"><h2>What differs between the recipes?</h2><div id="methods"></div><p class="small">The local cutoff is exploratory. The supplied recipe includes a broad-object pass and may count weak, broad or overlapping features differently. Its historical five-frame extraction and brightness gate are not used: this comparison changes the detector while holding extraction fixed. No method was selected or tuned to produce a preferred radius peak.</p></div>
<details class="box"><summary>Why staying below r_sat is encouraging but not a calibration check</summary><p>The observed 1× radii were never clipped at the model limit. Few observations exceeding that limit is compatible with the model, but does not establish accurate background, channel calibration, or particle selection. Residual background can keep an incorrect estimate inside the valid range.</p><p>For an equal additive background b in the two channels of a pair, X_observed = (I0 − I90)/(I0 + I90 + 2b): positive b compresses the ratio. Equal extra subtraction expands it while the denominator remains positive. The recorded background is spatially and polarization dependent, and zero clipping makes the change nonlinear, so extra subtraction can also move the X–Y direction and does not guarantee a larger radius for every spot.</p></details>
<p><a href="../matched_background_theta/radius_comparison.html">Previous 1× versus 2× radius plots</a> · <a href="../excluded_hinge_spots/index.html">Excluded hinge gallery</a> · <a id="png" href="comparison.png">PNG</a> · <a id="svg" href="comparison.svg">SVG</a> · <a href="results.json">All coordinates, radii, counts and provenance</a></p>
<script>const DATA=__DATA__, $=id=>document.getElementById(id),label={Hinges:'DNA / hinges',Gold:'Gold control'};
function render(){const view=$('method').value;$('plot').src=view+'.png';$('png').href=view+'.png';$('svg').href=view+'.svg';const methods=view==='comparison'?Object.keys(DATA.methods):[view];
$('counts').innerHTML=methods.flatMap(m=>['Hinges','Gold'].map(g=>{const s=DATA.summary[m][g];return `<tr><td>${DATA.methods[m].label}</td><td>${label[g]}</td><td>${s.detected}</td><td>${s.ceiling}</td><td>${s.undefined}</td><td>${s.all.n}</td><td>${s.matched.n}</td><td>${s.removed.n}</td><td>${s.all.median?.toFixed(3)} → ${s.matched.median?.toFixed(3)}</td><td>${s.removed.median?.toFixed(3)??'—'}</td><td>${s.all.outside_model} / ${s.matched.outside_model}</td></tr>`})).join('')}
$('methods').innerHTML=Object.values(DATA.methods).map(m=>`<p><b>${m.label}:</b> ${m.description}</p>`).join('');$('method').onchange=render;render();
</script></main></body></html>'''

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    result=json.loads((OUT/'results.json').read_text(encoding='utf-8')) if args.render_only else compute()
    render(result)
