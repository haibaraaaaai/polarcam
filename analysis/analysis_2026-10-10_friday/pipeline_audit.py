"""Reconcile the supplied five-frame archive with the recent native-detector report.

Offline only. Sources are read-only. Bundled coordinates are held fixed for
extraction ablations; detector comparisons are explicitly separate populations.
Run using the repository Python; --render-only reuses the numerical audit.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import warnings
import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from first_hinges_theta import load_module, clean_json
from local_spot_check import filtering

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
OUT = BASE / 'pipeline_audit'
PACKAGE = OUT / 'package'
OLD_ABC = (0.097840, 0.011915, 0.101852)
GRID = np.arange(0, 90.001, .1)
PARITIES = ((0, 0), (0, 1), (1, 0), (1, 1))


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def theta(r, abc=OLD_ABC):
    r = np.asarray(r, float)
    a, b, c = abc
    good = np.isfinite(r) & (r >= 0) & (r <= c / (a + b))
    result = np.full(r.shape, np.nan)
    result[good] = np.degrees(np.arcsin(np.sqrt(a*r[good]/(c-b*r[good]))))
    return result


def radius(channels):
    # Archive order: I90, I45, I135, I0.
    a, b, c, d = np.moveaxis(channels, -1, 0)
    sx, sy = d+a, b+c
    good = np.isfinite(sx)&np.isfinite(sy)&(sx>0)&(sy>0)
    x = np.full(sx.shape, np.nan)
    y = np.full(sy.shape, np.nan)
    np.divide(d-a, sx, out=x, where=good)
    np.divide(b-c, sy, out=y, where=good)
    return np.hypot(x,y)


def extract_mean(image, points_yx, width):
    result = []
    for y, x in points_yx:
        top, left = int(round(y))-width//2, int(round(x))-width//2
        crop = image[top:top+width, left:left+width]
        assert crop.shape == (width, width)
        result.append([crop[(pr-top)%2::2, (pc-left)%2::2].mean()
                       for pr, pc in PARITIES])
    return np.asarray(result)


def density(t):
    t = np.asarray(t)
    t = t[np.isfinite(t)]
    return np.exp(-.5*((GRID[:, None]-t[None, :])/4)**2).sum(axis=1)/(len(t)*4*np.sqrt(2*np.pi))


def describe(t):
    t = np.asarray(t)
    t = t[np.isfinite(t)]
    d = density(t)
    peaks, _ = find_peaks(d, prominence=.02*d.max(), distance=40)
    return dict(n=len(t), median=float(np.median(t)), mean=float(np.mean(t)),
                modes=GRID[peaks].round(1).tolist(),
                fraction_35_45=float(np.mean((t>=35)&(t<45))))


def paired(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    delta = b[valid]-a[valid]
    return dict(n=int(valid.sum()), median_delta=float(np.median(delta)),
                median_absolute_delta=float(np.median(np.abs(delta))),
                q10=float(np.percentile(delta, 10)), q90=float(np.percentile(delta, 90)))


def compute():
    OUT.mkdir(exist_ok=True)
    state = json.loads((BASE/'theta_comparison/results.json').read_text(encoding='utf-8'))
    new_abc = tuple(state['methods']['ABC'])
    detector, _ = load_module('audit_native_detector', 'src/polarcam/live/Detection_alg_offline.py')
    old_detector = import_file('archive_detector', PACKAGE/'detector_intensity_reference.py')
    summary = []
    for current in [s for s in state['stacks'] if s['group']=='september']:
        stamp = current['stamp']
        source = REPO/'data/hinges/polarcam/2026-09-18'/current['file']
        before = source.stat()
        raw = np.load(source, mmap_mode='r', allow_pickle=False)
        meta = json.loads(source.with_suffix('.json').read_text(encoding='utf-8-sig'))
        assert meta['actual']['frames']==150 and meta['actual']['phase_marker_appended'] and len(raw)==151
        assert meta['roi']['OffsetX']==meta['roi']['OffsetY']==0
        archive_path = PACKAGE/f'audit_20260918-{stamp}.npz'
        with np.load(archive_path, allow_pickle=False) as a:
            archived = {k:a[k].copy() for k in a.files}
        old_pts = archived['spots']
        native_pts = np.array([[row['y'], row['x']] for row in current['rows']])
        keep = np.array([row['included'] for row in current['rows']])
        local_pass = np.array([row['local_pass'] for row in current['rows']])

        # Read every observation frame once. 7x7 extraction uses each absolute
        # sensor parity separately, reproducing the archive's float32 means.
        offsets = np.array([(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)])
        ys = old_pts[:, 0, None] + offsets[None, :, 0]
        xs = old_pts[:, 1, None] + offsets[None, :, 1]
        masks = [((ys%2==pr)&(xs%2==pc)).astype(np.float32) for pr, pc in PARITIES]
        counts = [m.sum(axis=1) for m in masks]
        ch7 = np.empty((len(old_pts), 150, 4), dtype=np.float32)
        mean = np.zeros(raw.shape[1:], dtype=np.float64)
        for frame_idx, frame in enumerate(raw[:150]):
            mean += frame
            crops = frame[ys, xs].astype(np.float32)
            for k, mask in enumerate(masks):
                ch7[:, frame_idx, k] = (crops*mask).sum(axis=1)/counts[k]
        mean /= 150
        np.testing.assert_array_equal(ch7[:, :5], archived['channels_saved'])
        common = archived['common_frame_mask']
        r_frame = radius(ch7[:, :5])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r_reproduced = np.nanmedian(np.where(common, r_frame, np.nan), axis=1)
        np.testing.assert_allclose(r_reproduced, archived['r_saved'], rtol=0, atol=1e-14, equal_nan=True)
        gate = np.isfinite(archived['r_saved'])
        old_t = theta(archived['r_saved'])
        old_t_new_model = theta(archived['r_saved'], new_abc)
        # Preserve the old set of accepted frames first; then remove that gate.
        ch5_gated = np.empty((len(old_pts), 4))
        ch5_gated[:] = np.nan
        for j in np.flatnonzero(gate):
            ch5_gated[j] = ch7[j, :5][common[j]].astype(float).mean(axis=0)
        r_mean5_gated = radius(ch5_gated)
        r_mean5_all = radius(ch7[:, :5].astype(float).mean(axis=1))
        r_mean150_7 = radius(extract_mean(mean, old_pts, 7))
        np.testing.assert_allclose(r_mean150_7, radius(ch7.astype(float).mean(axis=1)), rtol=2e-7, atol=2e-7)
        r_median150_7 = np.median(radius(ch7), axis=1)
        r_mean150_8 = radius(extract_mean(mean, old_pts, 8))
        r_mean150_18 = radius(extract_mean(mean, old_pts, 18))
        native7 = radius(extract_mean(mean, native_pts, 7))
        native18 = radius(extract_mean(mean, native_pts, 18))
        np.testing.assert_allclose(native18, [row['r'] for row in current['rows']], rtol=1e-12, atol=1e-12)
        nt_new = theta(native18, new_abc)
        np.testing.assert_allclose(nt_new[keep], [row['theta_deg'] for row in current['rows'] if row['included']], atol=1e-11)

        # Re-run the supplied reconstructed detector against the actual first 5.
        ref_pts, ref_info = old_detector.detect_candidates(raw[:5])
        exact = np.array_equal(old_pts, ref_pts)
        overlap = len(set(map(tuple, old_pts)) & set(map(tuple, ref_pts)))
        f5 = filtering(detector.intensity_detection_map(np.mean(raw[:5], axis=0)), detector)
        native_distance, nearest_native = cKDTree(native_pts).query(old_pts)
        within6 = native_distance<=6  # Proximity only; not a one-to-one identity claim.
        near_kept_distance = cKDTree(native_pts[keep]).query(old_pts)[0]
        low = (old_t>=35)&(old_t<45)
        high = (old_t>=60)&(old_t<80)
        curve_r = dict(old=archived['r_saved'], old_new_model=archived['r_saved'],
            old_no_gate=archived['r_saved_all'], mean5_common=r_mean5_gated,
            mean5_all=np.where(gate, r_mean5_all, np.nan),
            mean150_7=np.where(gate, r_mean150_7, np.nan),
            median150_7=np.where(gate, r_median150_7, np.nan),
            mean150_8=np.where(gate, r_mean150_8, np.nan),
            mean150_18=np.where(gate, r_mean150_18, np.nan),
            all_old18=r_mean150_18, native7=native7, native18=native18,
            native18_local=native18[local_pass], current=native18[keep])
        curve_t = {k:theta(v, new_abc if k in ('old_new_model', 'current') else OLD_ABC)
                   for k, v in curve_r.items()}
        # Radius-domain conditioning reduces model-dependent inclusion effects.
        def pop(mask):
            return dict(n=int(mask.sum()), native_within6=int(np.sum(mask&within6)),
                kept_native_within6=int(np.sum(mask&(near_kept_distance<=6))),
                median_channel_sum=float(np.median(ch7[mask,:5].mean(axis=1).sum(axis=1))),
                mean150_7_median=float(np.nanmedian(theta(r_mean150_7)[mask])),
                mean150_18_median=float(np.nanmedian(theta(r_mean150_18)[mask])))
        s = dict(stamp=stamp, source=source.name, source_size=before.st_size,
            source_mtime_ns=before.st_mtime_ns, archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            archive_first5_channels_exact=True, archive_r_saved_max_error=float(np.nanmax(np.abs(r_reproduced-archived['r_saved']))),
            old_detector=dict(exact_ordered_match=exact, coordinate_set_overlap=overlap,
                              stored_count=len(old_pts), rerun_count=len(ref_pts), **ref_info),
            native_detection=dict(first5_count=len(f5['centres']), full150_count=len(native_pts),
                first5_threshold=float(f5['threshold']), full150_threshold=current['image_stats']['dog_threshold'],
                local_excluded=int((~local_pass).sum())),
            distributions={k:describe(v) for k, v in curve_t.items()},
            paired=dict(model=paired(old_t, old_t_new_model),
                aggregation=paired(curve_t['old'], curve_t['mean5_common']),
                remove_frame_gate=paired(curve_t['mean5_common'], curve_t['mean5_all']),
                extend_to150=paired(curve_t['mean5_all'], curve_t['mean150_7']),
                aperture7to18=paired(curve_t['mean150_7'], curve_t['mean150_18']),
                aperture7to8=paired(curve_t['mean150_7'], curve_t['mean150_8'])),
            low_population=pop(low), high_population=pop(high),
            invalid_old_model=int(np.sum(gate&~np.isfinite(old_t))),
            gate_failure_count=int((~gate).sum()))
        arrays = {f't_{k}':v for k,v in curve_t.items()}
        arrays.update(old_points=old_pts, native_points=native_pts, native_keep=keep,
            old_gate=gate, old_r=archived['r_saved'], old_r7_150=r_mean150_7,
            old_r18_150=r_mean150_18, old_distance_to_native=native_distance,
            old_distance_to_kept_native=near_kept_distance,
            old_channel_sum=ch7[:,:5].mean(axis=1).sum(axis=1),
            native_local_pass=local_pass)
        np.savez_compressed(OUT/f'comparison_{stamp}.npz', **arrays)
        assert before.st_size==source.stat().st_size and before.st_mtime_ns==source.stat().st_mtime_ns
        summary.append(s)
        (OUT/'results.json').write_text(json.dumps(clean_json(dict(stacks=summary, old_ABC=OLD_ABC,new_ABC=new_abc)),indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(s), flush=True)
        del raw, mean, f5, ch7
    return dict(stacks=summary, old_ABC=OLD_ABC, new_ABC=new_abc)


def figures(state):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig, axs = plt.subplots(2,4,figsize=(14,7),sharex=True,sharey=True,layout='constrained')
    for ax,s in zip(axs.flat,state['stacks']):
        with np.load(OUT/f"comparison_{s['stamp']}.npz") as z:
            for key,color,label in [('old','#0072b2','Supplied pipeline'),('current','#bf552d','Recent pipeline')]:
                ax.plot(GRID,density(z['t_'+key]),color=color,lw=2,label=label)
        ax.axvspan(35,45,color='#666666',alpha=.07,lw=0)
        old=s['distributions']['old'];new=s['distributions']['current']
        ax.set(title=f"{s['stamp']} · n = {old['n']} / {new['n']}",xlim=(0,90),xticks=[0,30,60,90])
    axs.flat[0].legend(frameon=False,fontsize=9)
    for ax in axs[1]:ax.set_xlabel('Effective optical θ (degrees)')
    for ax in axs[:,0]:ax.set_ylabel('Density (per degree)')
    fig.suptitle('The discrepancy is reproducible · identical 4° smoothing on both pipelines',fontsize=14)
    fig.savefig(OUT/'pipeline_comparison.png',dpi=180);plt.close(fig)

    selected = ('151106','151218','154622','155126')
    fig,axs=plt.subplots(3,4,figsize=(14,10),sharex=True,sharey=True,layout='constrained')
    for col,stamp in enumerate(selected):
        with np.load(OUT/f'comparison_{stamp}.npz') as z:
            panels=[ [('old','#0072b2','Archive: median r, 5 frames'),('mean5_common','#999999','Mean channels, same frames'),('mean150_7','#bf552d','Mean channels, 150 frames')],
                     [('mean150_7','#0072b2','7×7 window'),('mean150_18','#bf552d','18×18 window')],
                     [('all_old18','#0072b2','All archived centres'),('native18','#bf552d','Native centres'),('native18_local','#313131','Native + local cutoff')] ]
            for row,lines in enumerate(panels):
                ax=axs[row,col]
                # First two panels compare the identical candidates, including
                # an identical validity mask, to isolate the processing change.
                common_valid = np.logical_and.reduce([np.isfinite(z['t_'+key]) for key,_,_ in lines]) if row<2 else None
                for key,color,label in lines:
                    vals=z['t_'+key]
                    if common_valid is not None:vals=vals[common_valid]
                    ax.plot(GRID,density(vals),color=color,lw=1.6,label=label)
                ax.axvspan(35,45,color='#666666',alpha=.07,lw=0)
                ax.set(xlim=(0,90),xticks=[0,30,60,90])
                if col==0:ax.legend(frameon=False,fontsize=7.5)
            axs[0,col].set_title(stamp)
    for row,label in enumerate(('A · Averaging','B · Window size','C · Candidate selection')):
        axs[row,0].set_ylabel(label+'\nDensity (per degree)')
    for ax in axs[-1]:ax.set_xlabel('Effective optical θ (degrees)')
    fig.suptitle('Controlled comparisons · one common optical model and 4° smoothing throughout',fontsize=13)
    fig.savefig(OUT/'controlled_changes.png',dpi=170);plt.close(fig)


def page(state):
    rows=[]
    for s in state['stacks']:
        d=s['distributions'];o=s['old_detector'];n=s['native_detection']
        time=f"{s['stamp'][:2]}:{s['stamp'][2:4]}:{s['stamp'][4:]}"
        cells=(time,', '.join(f'{p:.1f}°' for p in d['old']['modes']),
            ', '.join(f'{p:.1f}°' for p in d['current']['modes']),
            o['stored_count'],n['full150_count'],n['local_excluded'],d['current']['n'])
        rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in cells)+'</tr>')
    deltas=[]
    for s in state['stacks']:
        p=s['paired']
        cells=(s['stamp'],*[f"{p[k]['median_delta']:+.2f}°" for k in ('model','aggregation','extend_to150','aperture7to18')])
        deltas.append('<tr>'+''.join(f'<td>{v}</td>' for v in cells)+'</tr>')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Why the September theta peaks differ</title><style>
body{margin:0;background:#f6f7f5;color:#23313c;font:16px/1.6 system-ui,sans-serif}main{max-width:1260px;margin:auto;padding:28px}
h1{font-size:30px;line-height:1.25}h2{font-size:22px;margin-top:32px}p{max-width:1050px}a{color:#0072b2}
img{display:block;width:100%;height:auto;background:white}table{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #cdd3d4;vertical-align:top}th{background:#e9eeed}
.scroll{overflow-x:auto}.scroll td{white-space:nowrap}.note{font-size:14px;color:#52636e}.lead{font-size:19px}
.box{border-left:4px solid #0072b2;padding:8px 18px;background:#eaf1f4}code{font-size:13px;overflow-wrap:anywhere}
@media(max-width:700px){main{padding:18px}.methods{display:block;overflow-x:auto}.methods td{min-width:160px}}
</style><main>
<h1>The ~40° peaks are reproducible. The selection and extraction changed.</h1>
<p class="lead">I reproduced all eight archived candidate lists exactly from the first five observation frames of the original <b>18 September 2026</b> stacks. The saved four-channel intensities match exactly; reconstructed radii match within 2.3 × 10<sup>−16</sup>. These are the same underlying recordings.</p>
<p>The supplied analysis has lower-angle modes at <b>40.1°, 43.3°, 42.3° and 40.2°</b> in four stacks, alongside larger-angle modes. The other four have a single detected mode around 67–70°. My recent analysis used substantially different candidate selection and a larger extraction window. Its results do not establish that the earlier lower-angle population was absent.</p>
<img src="pipeline_comparison.png" alt="Eight September distributions: supplied and recent pipelines, plotted with identical smoothing">
<p class="note">Both curves use the supplied plotting method: Gaussian density smoothing with a 4° bandwidth and the same 2% prominence rule for modes. Each distribution is normalized to its own number of model-valid candidates. Counts above the panels refer to those candidates, not the larger radius-space counts in supplied Figure A. These are modes, distinct from the medians in my previous report.</p>
<div class="scroll"><table><thead><tr><th>18 September</th><th>Supplied modes</th><th>Recent modes</th><th>Old candidates</th><th>Native candidates</th><th>Local rejections</th><th>Recent θ count</th></tr></thead><tbody>__TABLE__</tbody></table></div>
<h2>What changed</h2>
<table class="methods"><thead><tr><th>Step</th><th>Supplied pipeline</th><th>Recent comparison</th><th>Effect observed</th></tr></thead><tbody>
<tr><td>Spot detection</td><td>First 5 frames; raw polarization mosaic; despiking; DoG median + 8 × robust MAD; additional broad-object pass</td><td>150 frames; 2 × 2 polarization-cell intensity map; native DoG mean + 8 × SD; one pass</td><td>Very different candidate populations: 843–2,642 versus 123–478 per stack, before later exclusions. “8” does not make these equivalent thresholds.</td></tr>
<tr><td>Local quality cutoff</td><td>None</td><td>Neighbourhood DoG contrast ≥8</td><td>In 15:11:06 it rejects 24 of the 32 native candidates at 35–45°, removing the remaining lower-angle mode. In 15:46:22 and 15:51:26 it rejects none: their loss occurs at the earlier detector selection.</td></tr>
<tr><td>Intensity window</td><td>7 × 7 sensor pixels</td><td>18 × 18 sensor pixels</td><td>At fixed archived centres, fixed 150-frame means and fixed model, the wider window lowers paired θ by a median 8.6–15.0° across stacks.</td></tr>
<tr><td>Time summary</td><td>Median of per-frame radii over the accepted first 5 frames</td><td>Average channel intensities over 150 frames, then compute radius</td><td>Much smaller effect than selection/window size. Lower-angle features remain when using the old centres and a small window with all 150 frames.</td></tr>
<tr><td>Optical model</td><td>Notebook hole + Fresnel coefficients; r<sub>max</sub> ≈0.9280</td><td>Manuscript annular coefficients; r<sub>max</sub> ≈0.9208</td><td>At identical radii valid under both models, median shifts are only +0.18 to +0.47°. Not the cause of losing the lower-angle peak.</td></tr>
<tr><td>Display</td><td>4° smoothed density in θ; separate density in radius</td><td>5° θ histogram; summaries emphasized medians</td><td>Mode and median are different summaries. The discrepancy remains when both pipelines use identical smoothing, as above.</td></tr>
</tbody></table>
<h2>Changing one part at a time</h2>
<p>All curves below use the <b>same supplied optical model</b>. Row A holds the archived coordinates and 7 × 7 window fixed while changing the time summary. Row B holds those coordinates and the 150-frame mean fixed while widening the window. These two rows use the common set valid for all curves within each panel. Row C holds the 150-frame mean and 18 × 18 window fixed while changing candidate selection; each selected population is normalized separately.</p>
<img src="controlled_changes.png" alt="Controlled comparisons isolating averaging, window size and candidate selection">
<p>The enlarged window samples more surrounding intensity. The measured radius decreases at these same centres; that produces lower reconstructed θ. Contributions from residual background, neighbours and the spot's spatial profile can all matter. This audit does not identify which contribution dominates each object.</p>
<div class="box"><p><b>A concrete example: 15:11:06.</b> The archive finds 1,998 candidates; 1,681 pass its legacy frame gate, and 1,592 are within its θ model. The native method finds 123; the local cutoff leaves 93. Under one common model, those native candidates have modes near <b>36.7° and 64.0° before</b> the local cutoff, and only <b>64.3° after</b>. Applying the recent model moves that remaining mode to 64.8°.</p></div>
<p>Of the archive's 378 candidates originally at 35–45° in this stack, only 13 lie within 6 sensor pixels of any native centre, and only 4 lie within 6 pixels of a retained native centre. These are proximity counts, not proof of one-to-one object identity. At 15:51:26, just 1 of 167 old candidates in that angle band is near a native centre.</p>
<h2>What this does and does not establish</h2>
<p><b>The lower-angle feature is a reproducible feature of the supplied extraction and selection.</b> It is not erased simply by including all 150 frames or changing the optical coefficients. A more permissive detector also does not establish that every candidate is a separate rod; the native detector and local cutoff are not validated as angle-neutral selection rules either. Neither distribution should yet be treated as the definitive physical hinge-angle distribution.</p>
<p>I should have made the magnitude of these method changes clearer when presenting the recent comparison. The useful next check is to inspect representative candidates contributing to the old lower-angle peak, including those omitted by the native detector, and assess extraction-window sensitivity before choosing one distribution for interpretation.</p>
<details><summary>Additional audit details and numerical shifts</summary>
<p>All rows below are median changes for the same candidates valid before and after the specified change; they are not shifts of the plotted modes. Columns isolate the optical model, radius-median versus intensity-mean over the same accepted five frames, extending the five-frame channel mean to 150 frames, and widening 7 × 7 to 18 × 18 at fixed 150-frame means.</p>
<div class="scroll"><table><thead><tr><th>Stack</th><th>Model only</th><th>Aggregation only</th><th>5 → 150 frames</th><th>7 → 18 pixels</th></tr></thead><tbody>__DELTAS__</tbody></table></div>
<p>The supplied figures use <code>r_saved</code>, calculated without another channel-background subtraction. The “corner background” in <code>audit_extraction.py</code> is retained in an old validity gate: both channel pair sums must exceed 50 after subtracting that scalar. The four reference corners all belong to one polarization parity; they are not a complete background annulus. Removing this gate still leaves the lower-angle modes. It is therefore not the principal cause here.</p>
<p>A 7 × 7 window samples 9, 12, 12 and 16 pixels in the four channels. Means correct unequal counts, but the spatial supports still differ. An 8 × 8 probe at fixed centres retains lower-angle modes, though their positions shift (e.g. 15:11:06: 37.3°; 15:51:26: 33.6°). Small-window sensitivity is another reason not to claim a precisely validated physical 40° angle yet.</p>
<p>Although the archive's θ conversion function contains a clamp, Figure B excludes radii above its model limit before conversion; its lower peak is not caused by clamping. The radius-space peak annotations are stored constants in <code>PEAKS_R</code>; the supplied package does not itself rerun their claimed bandwidth-stability assessment. This audit independently regenerates the θ-space modes. “Stationary” appears in the plot script description, but no stationarity selection is implemented in the supplied detector/extractor.</p>
<p>Using the native detector on just the first five frames yields 139 candidates for 15:11:06, compared with 123 on the 150-frame mean and 1,998 in the archive. Frame count alone therefore does not explain the large detection-count difference.</p>
<p>θ remains an effective optical polar angle under the specified reconstruction, not a direct measurement of hinge opening. No cross-stack exclusion or September pooling was added. Raw data, camera code and the prior reports are unchanged; this is an offline audit.</p></details>
<p class="note"><a href="results.json">Numerical audit</a> · <a href="../pipeline_audit.py">Reproducible audit script</a> · <a href="package/detector_intensity_reference.py">Supplied detector</a> · <a href="package/audit_extraction.py">Supplied extraction</a> · <a href="package/plot.py">Supplied plotting code</a> · <a href="../theta_comparison/index.html">Previous comparison</a></p>
</main></html>'''
    (OUT/'index.html').write_text(html.replace('__TABLE__',''.join(rows)).replace('__DELTAS__',''.join(deltas)),encoding='utf-8')


if __name__=='__main__':
    state=json.loads((OUT/'results.json').read_text(encoding='utf-8')) if '--render-only' in sys.argv else compute()
    figures(state)
    page(state)
