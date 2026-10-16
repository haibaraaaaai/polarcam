"""Theta/r of masked p95 detections, including oversized unsaturated components."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detector_grid_trial import DAY, NAME, sha
from background_percentile_trial import OUT, signature
from polarcam import optics
from polarcam.live import intensity_spots


def summary(r, theta):
    finite = np.isfinite(theta)
    return dict(n=len(r), valid_theta=int(finite.sum()), above_limit=int((r > optics.theta_model()['r_max']).sum()),
                undefined_r=int((~np.isfinite(r)).sum()), median_r=float(np.nanmedian(r)),
                median_theta=float(np.nanmedian(theta)) if finite.any() else None)


def build():
    selection_path = OUT/'percentile95_region.json'
    selection = json.loads(selection_path.read_text())
    source = DAY/'frames'/NAME
    assert signature(source) == selection['source_signature']
    meta = json.loads(source.with_suffix('.json').read_text())
    peak_info = meta['fetch_frames_output']['raw_peak']
    assert peak_info['frames'] == 150
    assert sha(source.parent/peak_info['path']) == selection['raw_peak_sha256']
    raw_peak = np.load(source.parent/peak_info['path'])
    source_frames = np.load(source, mmap_mode='r')
    assert meta['actual']['frames'] == 150 and len(source_frames) == 151
    cache_meta = json.loads((OUT/'percentile_results.json').read_text())
    assert sha(source.with_suffix('.json')) == cache_meta['sidecar_sha256']
    with np.load(OUT/'percentile_arrays.npz') as cache:
        indices = [list(cache['percentiles']).index(p) for p in [0, 95]]
        bg = cache['background_profiles'][indices]
        means = cache['sum_frames'][indices].astype(np.float64)/150
    first = [source_frames[0], np.rint(np.maximum(source_frames[0].astype(np.float32)-(bg[1]-bg[0]), 0)).astype(np.uint16)]
    model = optics.theta_model()
    a, b, c = [model[k] for k in ['A', 'B', 'C']]
    modes = {}
    for mode, inputs in [('first', first), ('mean', means)]:
        previous = selection['modes'][mode]['methods']['p95']
        policy = previous['policies']['m1_a1_o1']
        ids = set(policy['kept'])
        spots = [s for s in previous['spots'] if s['id'] in ids]
        assert len(spots) == len(ids) and not policy['inside']
        assert all(not s['saturated'] and not s['geometric_inside'] for s in spots)
        bounds = [s['bounds'] for s in spots]
        ch_min, ch_p95 = [intensity_spots.channel_means(frame, bounds) for frame in inputs]
        xy_min, xy_p95 = [intensity_spots.anisotropy(ch) for ch in [ch_min, ch_p95]]
        r_min, r_p95 = [np.linalg.norm(xy, axis=1) for xy in [xy_min, xy_p95]]
        t_min, t_p95 = [np.degrees(optics.theta_from_radius(r, a, b, c)) for r in [r_min, r_p95]]
        # Same p95 channels, apertures and radii as the inspected gallery.
        np.testing.assert_allclose(r_p95, [s['r'] for s in spots], rtol=0, atol=1e-12)
        for radius, theta in [(r_min, t_min), (r_p95, t_p95)]:
            valid = np.isfinite(theta)
            assert np.array_equal(valid, np.isfinite(radius) & (radius >= 0) & (radius <= model['r_max']))
            np.testing.assert_allclose(optics.radius_from_theta(np.radians(theta[valid]), a, b, c), radius[valid], atol=1e-14)
        rows = []
        for i, s in enumerate(spots):
            l, r, t, bottom = s['bounds']; x, y, w, h = s['bbox']; bx, by = round(x+.5), round(y+.5)
            peak = max(int(raw_peak[t:bottom, l:r].max()), int(raw_peak[by:by+h, bx:bx+w].max()))
            assert peak == s['raw_peak'] and peak < 4094
            row = {k: s[k] for k in ['id', 'x', 'y', 'area', 'bbox', 'bounds', 'raw_peak']}
            for key, ch, xy, radius, theta in [('minimum', ch_min, xy_min, r_min, t_min), ('p95', ch_p95, xy_p95, r_p95, t_p95)]:
                row[key] = dict(channels=ch[i].tolist(), xy=xy[i].tolist(), r=float(radius[i]),
                                theta_deg=float(theta[i]) if np.isfinite(theta[i]) else None,
                                theta_status='valid' if np.isfinite(theta[i]) else 'above_model_limit')
            rows.append(row)
        modes[mode] = dict(spots=rows, summary={key: summary(r, t) for key, r, t in
                          [('minimum', r_min, t_min), ('p95', r_p95, t_p95)]},
                          retained=len(spots), oversized=len(policy['oversized']), exclusions=policy['counts'],
                          detector_threshold=previous['threshold'])
    assert signature(source) == selection['source_signature']
    result = dict(source=NAME, modes=modes, model=model, ellipse=selection['ellipse'],
                  selection_policy='m1_a1_o1: top-right ellipse excluded; component area >=10 with no maximum; '
                                   'raw saturation >=4094 excluded over all 150 frames in aperture or full '
                                   'component bounding rectangle; previous strict overlap suppression retained.',
                  extraction='One value per candidate: four analyzer means in the same 18x18 centroid-centred '
                             'sensor aperture, including for oversized components. First-frame mode uses '
                             'frame 1; mean mode averages corrected frames first. X=(I0-I90)/(I0+I90); '
                             'Y=(I45-I135)/(I45+I135); r=hypot(X,Y). Theta=asin(sqrt(A*r/(C-B*r))). '
                             'No APD correction matrix or camera channel gain fit applied. No temporal '
                             'angle averaging or solid-angle-density correction.',
                  reference='Minimum-background reference uses exactly the same p95-selected coordinates '
                            'and apertures. Only extraction background changes, not the selected population.',
                  invalid_theta='r above C/(A+B) remains in per-spot data and full-radius plot but has null '
                                'theta. Never clipped to 90 degrees or removed silently.',
                  selection_sha256=sha(selection_path), percentile_metadata_sha256=sha(OUT/'percentile_results.json'),
                  source_signature=selection['source_signature'], raw_peak_sha256=selection['raw_peak_sha256'],
                  optics_sha256=sha(Path(optics.__file__)), extraction_sha256=sha(Path(intensity_spots.__file__)),
                  builder_sha256=sha(Path(__file__)))
    (OUT/'percentile95_theta.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    (OUT/'percentile95_theta.html').write_text(HTML.replace('__DATA__', json.dumps(result).replace('</', '<\\/')), encoding='utf-8')
    for mode, entry in modes.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
        for key, color, label in [('minimum', '#858e95', 'Minimum BG at same positions'), ('p95', '#087e8b', '95th-percentile BG')]:
            for ax, field, edges in [(axes[0], 'theta_deg', np.arange(0, 92, 2)), (axes[1], 'r', np.arange(0, 1.441, .02))]:
                values = [s[key][field] for s in entry['spots'] if s[key][field] is not None]
                counts, _ = np.histogram(values, edges)
                assert counts.sum() == len(values)
                if key == 'p95':
                    ax.stairs(counts, edges, fill=True, color=color, alpha=.35)
                ax.stairs(counts, edges, color=color, linewidth=1.5, label=label,
                          linestyle='--' if key == 'minimum' else '-')
        s = entry['summary']['p95']; ref = entry['summary']['minimum']
        axes[0].set(xlabel='Polar angle theta from optical axis (degrees)', ylabel='Spots per 2-degree bin',
                    title=f"Valid theta: {s['valid_theta']} / {entry['retained']} spots", xlim=(0, 90))
        axes[1].set(xlabel='Anisotropy radius r', ylabel='Spots per 0.02 bin', title=f"All {entry['retained']} selected spots", xlim=(0, 1.44))
        axes[1].axvspan(model['r_max'], 1.44, color='#c3574e', alpha=.08)
        axes[1].axvline(model['r_max'], color='#b14d45', linestyle=':', linewidth=1.4,
                       label=f"Model limit = {model['r_max']:.4f}")
        axes[1].text(.97, .92, f"{s['above_limit']} above model limit\n({100*s['above_limit']/entry['retained']:.1f}% of selected)",
                     transform=axes[1].transAxes, ha='right', va='top', fontsize=10, color='#943d35')
        for ax in axes:
            ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.15)
        axes[1].legend(loc='upper left', fontsize=8)
        label = '150-frame mean' if mode == 'mean' else 'First saved frame'
        fig.suptitle(f"First hinge stack | {label} | 95th-percentile detection, 2 SD\nTop-right mask on; oversized allowed; raw saturation excluded; strict overlap on", fontsize=12)
        fig.savefig(OUT/f'percentile95_theta_{mode}.png', dpi=170)
        fig.savefig(OUT/f'percentile95_theta_{mode}.svg')
        plt.close(fig)
    print(json.dumps({mode: dict(retained=e['retained'], oversized=e['oversized'], summary=e['summary']) for mode, e in modes.items()}), flush=True)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>95th background · theta distribution</title>
<style>body{font:16px/1.5 system-ui;color:#263637;background:#f7f8f5;margin:0}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:27px}h2{font-size:21px}p{max-width:1200px}a{color:#087985}.small{font-size:14px;color:#526369}select{font:inherit;padding:6px}label{display:inline-block;margin:5px 16px 5px 0}.plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.plot{background:white;border:1px solid #d6ded8;padding:12px}.plot svg{width:100%;height:auto}#limit{padding:12px 16px;border-left:4px solid #b14d45;background:#fcf1ec}table{border-collapse:collapse}td,th{padding:7px 14px 7px 0;text-align:left;border-bottom:1px solid #d3ddd7}.scroll{overflow:auto}@media(max-width:850px){.plots{grid-template-columns:1fr}}</style><main>
<h1>Theta distribution with 95th-percentile background subtraction</h1>
<p><b>First hinge stack: 18:38:50 on 12 October.</b> Detection at <b>2 SD</b>; the top-right mask is <b>retained</b>; oversized components are <b>included</b>; raw saturation is <b>excluded</b>. Previous strict overlap suppression stays on. The default is one value per spot from the <b>150-frame mean</b>; the first saved frame is also available.</p>
<label>Input <select id="mode"><option value="mean">150-frame mean</option><option value="first">First saved frame</option></select></label><label>Theta bin width <select id="bin"><option value="1">1°</option><option value="2" selected>2°</option><option value="5">5°</option></select></label><label><input type="checkbox" id="reference" checked>Minimum-BG extraction at the same selected positions</label>
<p id="summary"></p><p id="limit"></p>
<div class="plots"><section class="plot"><h2 id="thetaTitle"></h2><svg id="theta" viewBox="0 0 650 370" role="img" aria-label="Theta distribution"></svg></section><section class="plot"><h2 id="radiusTitle"></h2><svg id="radius" viewBox="0 0 650 370" role="img" aria-label="Full anisotropy radius distribution"></svg></section></div>
<p class="small"><span style="color:#087e8b">Teal: 95th-percentile subtraction.</span> <span style="color:#707b85">Grey dashed: minimum subtraction, using exactly the same p95-selected spots and extraction windows.</span> These are counts, not densities renormalized after discarding invalid angles. The r plot includes every selected spot, including values above the optical model’s limit.</p>
<div id="table" class="scroll"></div><p id="referenceNote"></p>
<details><summary>Angle model and averaging</summary><p>Water model with Fresnel transmission, back-focal-plane field mapping and a centred annular hole: NA<sub>out</sub>=1.3, NA<sub>in</sub>=0.39, n<sub>water</sub>=1.33, n<sub>oil</sub>=1.515. Theta is measured from the optical axis, 0–90°. I(ψ)=scale×[A+B sin²θ+C sin²θ cos2(ψ−φ)]. The same current GUI optical coefficients are used.</p><p id="coefficients"></p><p>Four channel intensities are averaged in each unchanged 18×18 sensor-pixel aperture. X=(I0−I90)/(I0+I90), Y=(I45−I135)/(I45+I135), r=√(X²+Y²), θ=asin√[Ar/(C−Br)]. In mean mode, corrected frames are averaged before ratios and angle inversion. No APD correction matrix, extra gain fit, temporal angle average or solid-angle correction is applied.</p><p>There is no maximum detection-component area, but the extraction aperture remains 18×18 even for oversized components. Oversized detections can represent merged objects; inclusion does not establish that each is one rod. Saturation is checked before subtraction over all 150 frames, in both the extraction aperture and the complete component bounding rectangle, at ≥4094. Display clipping at 20× gain is unrelated.</p><p>Radii above r<sub>sat</sub>=C/(A+B) have no real theta under this model. They are recorded with null theta, retained in the r plot, and never forced to 90°. Consequently the theta plot is only the convertible subset, not the complete selected population.</p></details>
<p><a id="png" href="percentile95_theta_mean.png">Download 2° plot (PNG)</a> · <a id="svg" href="percentile95_theta_mean.svg">SVG</a> · <a href="percentile95_theta.json">Per-spot values and provenance</a> · <a href="percentile95_region.html">Region inspection</a> · <a href="percentile_spots.html">Detection comparison</a></p>
</main><script>
const DATA=__DATA__,$=id=>document.getElementById(id);
function histogram(values,width,max){const n=Math.round(max/width),a=Array(n).fill(0);for(const v of values){if(v===null||!Number.isFinite(v)||v<0||v>max)continue;a[Math.min(n-1,Math.floor(v/width))]++}return a}
function plot(id,field,width,max){const e=DATA.modes[$('mode').value],W=650,H=370,L=57,R=18,T=20,B=58,drawRef=$('reference').checked,series=[{key:'p95',color:'#087e8b',counts:histogram(e.spots.map(s=>s.p95[field]),width,max)}];if(drawRef)series.unshift({key:'minimum',color:'#858e95',counts:histogram(e.spots.map(s=>s.minimum[field]),width,max)});
const ymax=Math.max(1,Math.ceil(Math.max(...series.flatMap(s=>s.counts))*1.12)),x=v=>L+v/max*(W-L-R),y=v=>H-B-v/ymax*(H-T-B);let svg='';
if(field==='r')svg+=`<rect x="${x(DATA.model.r_max)}" y="${T}" width="${x(max)-x(DATA.model.r_max)}" height="${H-T-B}" fill="#fcf1ec"/>`;
for(let i=0;i<=4;i++){const v=ymax*i/4;svg+=`<path d="M${L},${y(v)}H${W-R}" stroke="#e4e9e6"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end" font-size="12">${v.toFixed(1)}</text>`}
const step=field==='r'?.2:15;for(let v=0;v<=max+1e-9;v+=step)svg+=`<text x="${x(v)}" y="${H-B+22}" text-anchor="middle" font-size="12">${field==='r'?v.toFixed(1):v}</text>`;
for(const s of series){const a=s.counts;if(s.key==='p95')for(let j=0;j<a.length;j++)svg+=`<rect x="${x(j*width)}" y="${y(a[j])}" width="${x((j+1)*width)-x(j*width)}" height="${y(0)-y(a[j])}" fill="${s.color}" opacity=".3"/>`;let d=`M${x(0)},${y(a[0])}`;for(let j=0;j<a.length;j++)d+=`L${x(j*width)},${y(a[j])}L${x((j+1)*width)},${y(a[j])}`;svg+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.8" ${s.key==='minimum'?'stroke-dasharray="6,4"':''}/>`}
if(field==='r')svg+=`<path d="M${x(DATA.model.r_max)},${T}V${H-B}" stroke="#b14d45" stroke-dasharray="3,4"/><text x="${x(DATA.model.r_max)+7}" y="${T+14}" fill="#943d35" font-size="12">r_sat=${DATA.model.r_max.toFixed(4)}</text>`;
svg+=`<path d="M${L},${T}V${H-B}H${W-R}" stroke="#526369" fill="none"/><text x="${W/2}" y="${H-10}" text-anchor="middle" font-size="14">${field==='r'?'Anisotropy radius r':'Polar angle θ from optical axis (degrees)'}</text><text transform="translate(16,${H/2}) rotate(-90)" text-anchor="middle" font-size="13">Spots per ${width}${field==='r'?'':'°'} bin</text>`;$(id).innerHTML=svg}
function refresh(){const mode=$('mode').value,e=DATA.modes[mode],s=e.summary.p95,ref=e.summary.minimum;const pct=100*s.above_limit/e.retained;
$('summary').textContent=`${e.retained} selected spots (${e.oversized} oversized); ${e.exclusions.saturated} rejected for raw saturation. Mask on; previous strict overlap on.`;
$('limit').textContent=`${s.above_limit} / ${e.retained} spots (${pct.toFixed(1)}%) exceed r_sat=${DATA.model.r_max.toFixed(4)} after 95th-percentile subtraction. Only ${s.valid_theta} have a valid theta; the others stay in the r plot. No clipping to 90°.`;
$('thetaTitle').textContent=`Theta: ${s.valid_theta} valid of ${e.retained}`;$('radiusTitle').textContent=`Radius: all ${e.retained} selected spots`;
plot('theta','theta_deg',Number($('bin').value),90);plot('radius','r',.02,1.44);
$('table').innerHTML='<table><tr><th>Extraction background</th><th>Selected spots</th><th>Valid theta</th><th>Above r_sat</th><th>Median r (all)</th><th>Median theta (valid only)</th></tr>'+['p95','minimum'].map(key=>{const q=e.summary[key];return `<tr><td>${key==='p95'?'95th percentile':'Minimum, same positions'}</td><td>${q.n}</td><td>${q.valid_theta}</td><td>${q.above_limit}</td><td>${q.median_r.toFixed(3)}</td><td>${q.median_theta.toFixed(1)}°</td></tr>`}).join('')+'</table>';
$('referenceNote').textContent=`At these same positions, minimum-background extraction gives ${ref.above_limit} above-limit radii; 95th-percentile extraction gives ${s.above_limit}. The large shift is caused by changing the extraction background on the fixed spot list, and is consistent with oversubtraction. A cleaner-looking image does not by itself validate these reconstructed angles.`;
$('png').href=`percentile95_theta_${mode}.png`;$('svg').href=`percentile95_theta_${mode}.svg`}
$('coefficients').textContent=`A=${DATA.model.A.toPrecision(9)}, B=${DATA.model.B.toPrecision(9)}, C=${DATA.model.C.toPrecision(9)}; r_sat=${DATA.model.r_max.toPrecision(9)}.`;
for(const id of ['mode','bin','reference'])$(id).onchange=refresh;refresh();
</script></html>'''

if __name__ == '__main__':
    build()
