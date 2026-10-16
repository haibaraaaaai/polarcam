"""16 wide-field hinge stacks: p95 detection, minimum-background angle extraction."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detector_grid_trial import HERE, DAY, sha
from background_percentile_trial import OUT, signature
from background_percentile_spots import detect
from detector_threshold_trial import select_spots
from background_region_trial import ELLIPSE
from polarcam import optics
from polarcam.live import intensity_spots

RESULT = OUT/'hinges95_theta.json'


def finite(x):
    return float(x) if np.isfinite(x) else None


def stats(spots):
    keep = [s for s in spots if s['selection'] == 'kept']
    theta = np.array([s['theta_deg'] for s in keep if s['theta_deg'] is not None])
    r = np.array([s['r'] for s in keep])
    counts, edges = np.histogram(theta, np.arange(0, 92, 2))
    assert counts.sum() == len(theta)
    return dict(detected=len(spots), selected=len(keep), valid=len(theta),
                above_limit=sum(s['theta_status'] == 'above_model_limit' for s in keep),
                oversized=sum(s['area'] > 250 for s in keep),
                raw_saturated=sum(s['saturated'] for s in spots),
                median_theta=finite(np.median(theta)) if len(theta) else None,
                median_r=finite(np.median(r)) if len(r) else None,
                theta_quartiles=np.quantile(theta, [.25, .75]).tolist() if len(theta) else [],
                peak_2deg=[float(edges[np.argmax(counts)]), float(edges[np.argmax(counts)+1])] if len(theta) else [],
                counts_2deg=counts.tolist(),
                exclusions={k: sum(s['selection'] == k for s in spots)
                            for k in ['region', 'saturated', 'invalid', 'overlap', 'kept']})


def background(bg_hash):
    bgpath = next(p for p in (DAY/'backgrounds').glob('background_min_*.npy') if sha(p) == bg_hash)
    meta = json.loads(bgpath.with_suffix('.json').read_text(encoding='utf-8'))
    old = np.load(bgpath)
    assert np.array_equal(old, np.rint(old))
    movie = DAY/'backgrounds'/meta['source_stack_file']
    sig = signature(movie); raw = np.load(movie, mmap_mode='r')
    n = meta['actual']['frames']
    assert meta['actual']['phase_marker_appended'] and len(raw) == n+1
    assert meta['background']['background_subtracted'] is False
    p95 = np.empty(old.shape, dtype=np.float32)
    for y in range(0, old.shape[0], 64):
        block = raw[:n, y:y+64]
        np.testing.assert_array_equal(np.min(block, axis=0), old[y:y+64])
        p95[y:y+64] = np.percentile(block, 95, axis=0, method='linear')
    assert np.all(p95 >= old) and signature(movie) == sig
    provenance = dict(minimum_file=bgpath.name, minimum_sha256=bg_hash, capture=movie.name,
                      capture_sha256=sha(movie), capture_signature=sig, frames=n,
                      percentile=95, interpolation='linear', profile_dtype='float32',
                      p95_array_sha256=hashlib.sha256(p95.tobytes()).hexdigest())
    print(f'Background ready: {bgpath.name}, {n} frames', flush=True)
    return p95-old, provenance


def calculate():
    manifest = json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    inputs = [r for r in manifest['records'] if r['type'] == 'Frame stack' and r['sample'] == 'Hinge']
    assert len(inputs) == 16
    inputs.sort(key=lambda r: r['name'])
    model = optics.theta_model(); a, b, c = [model[k] for k in ['A', 'B', 'C']]
    hashes = list(dict.fromkeys(r['bg_hash'] for r in inputs)); assert len(hashes) == 2
    bg = {h: background(h) for h in hashes}
    records = []
    result = dict(complete=False, model=model, ellipse=ELLIPSE, backgrounds=[bg[h][1] for h in hashes], records=records,
                  scope='16 wide-field hinge stacks numbered 1-16, 12 October; excludes setup and later targeted wide/narrow recordings.',
                  detection='150-frame mean after p95 subtraction; each frame corrected, zero-clipped and rounded before averaging. '
                            'Same recorded BG capture for its group. GUI intensity map, repeated 2x2 cell mean; 10-pixel edge; '
                            'DoG sigma 1/6 sensor pixels; whole filtered mean+2SD; 8-connectivity; area >=10 without upper limit.',
                  extraction='Minimum-background-subtracted 150-frame mean at p95-selected centroids, fixed 18x18 aperture, '
                             'including oversized detections. Four intensities averaged before X/Y ratios and direct theta inversion. '
                             'No APD correction matrix or additional camera gain calibration.',
                  selection='Fixed top-right ellipse excluded; any raw pixel >=4094 over 150 frames in aperture OR component '
                            'bounding rectangle excluded. Undefined MINIMUM-background r excluded. Highest p95 DoG peak wins '
                            'if 18x18 apertures overlap by any pixel. No survival, low-background or finite-r cut before inversion.',
                  pooling='Pooled histogram weights each retained detection equally. No cross-field deduplication or independent '
                          'biological-replicate claim. Stack 16 is an extra angle check and is included by default but can be hidden.',
                  code_sha256={p.name: sha(p) for p in [Path(__file__), Path(optics.__file__), Path(intensity_spots.__file__),
                       HERE/'background_percentile_spots.py', HERE/'detector_threshold_trial.py']},
                  manifest_sha256=sha(HERE/'manifest.json'))
    for index, record in enumerate(inputs, 1):
        source = DAY/'frames'/record['name']; sig = signature(source)
        meta = json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
        bg_hash = meta['background']['background_profile_sha256']
        assert meta['background']['background_subtracted'] is True and bg_hash == record['bg_hash']
        extra, bgmeta = bg[bg_hash]
        assert sha(DAY/'backgrounds/used'/f'{bg_hash}.npy') == bg_hash
        frames = np.load(source, mmap_mode='r'); n = meta['actual']['frames']
        assert n == 150 and meta['actual']['phase_marker_appended'] and len(frames) == n+1
        assert frames.dtype == np.uint16 and frames.shape[1:] == extra.shape
        sums_min = np.zeros(extra.shape, dtype=np.uint32); sums95 = np.zeros_like(sums_min)
        work = np.empty(extra.shape, dtype=np.float32)
        for frame in frames[:n]:
            sums_min += frame
            np.subtract(frame, extra, out=work); np.maximum(work, 0, out=work); np.rint(work, out=work)
            sums95 += work.astype(np.uint16)
        mean_min = sums_min.astype(np.float64)/n; mean95 = sums95.astype(np.float64)/n
        assert np.all(mean95 <= mean_min)
        info = meta['fetch_frames_output']['raw_peak']; peakpath = source.parent/info['path']
        assert info['frames'] == n and sha(peakpath) == info['sha256']
        raw_peak = np.load(peakpath); assert raw_peak.shape == extra.shape
        detection, _ = detect(mean95, raw_peak, f'hinge{index}')
        rows = detection['spots']
        channels = intensity_spots.channel_means(mean_min, [s['bounds'] for s in rows])
        xy = intensity_spots.anisotropy(channels); radius = np.linalg.norm(xy, axis=1)
        theta = np.degrees(optics.theta_from_radius(radius, a, b, c))
        for j, s in enumerate(rows):
            x, y, w, h = s['bbox']; left, top = round(x+.5), round(y+.5)
            s['raw_peak'] = max(s['raw_peak'], int(raw_peak[top:top+h, left:left+w].max()))
            s['saturated'] = s['raw_peak'] >= 4094
            s['r_p95_diagnostic'] = s['r']
            s['accepted'] = True  # Keep all component areas >=10, including oversized.
            s['channels'] = channels[j].tolist()
            s['xy'] = xy[j].tolist() if np.isfinite(xy[j]).all() else None
            s['r'] = finite(radius[j]); s['theta_deg'] = finite(theta[j])
            s['theta_status'] = 'valid' if np.isfinite(theta[j]) else 'above_model_limit' if radius[j] > model['r_max'] else 'undefined_r'
        kept = select_spots(rows)
        assert all(not s['inside_ellipse'] and not s['saturated'] for s in kept)
        valid = np.isfinite(theta)
        np.testing.assert_allclose(optics.radius_from_theta(np.radians(theta[valid]), a, b, c), radius[valid], atol=1e-14)
        s = stats(rows); assert sum(s['exclusions'].values()) == len(rows)
        assert s['selected'] == s['valid'] + s['above_limit']
        if index == 1:
            old = json.loads((OUT/'percentile95_theta.json').read_text(encoding='utf-8'))['modes']['mean']
            assert len(kept) == old['retained'] == 365
            np.testing.assert_allclose([[r['x'], r['y'], r['r'], r['theta_deg']] for r in rows if r['selection'] == 'kept'],
                                       [[r['x'], r['y'], r['minimum']['r'], r['minimum']['theta_deg']] for r in old['spots']], rtol=0, atol=1e-12)
        assert signature(source) == sig
        records.append(dict(index=index, name=source.name, time=record['time'], note=record['note'],
                            background_group=hashes.index(bg_hash)+1, background_sha256=bg_hash, frames=n,
                            threshold=detection['threshold'], source_signature=sig, sidecar_sha256=sha(source.with_suffix('.json')),
                            raw_peak_sha256=info['sha256'], raw_peak_max=int(raw_peak.max()),
                            extra_check=index == 16, summary=s, spots=rows))
        RESULT.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
        print(f"{index}/16 {source.stem[-6:]}: {s['selected']} selected, {s['valid']} valid theta, {s['raw_saturated']} raw-saturated candidates", flush=True)
    result['complete'] = True
    result['pooled'] = stats([s for r in records for s in r['spots']])
    result['groups'] = {str(group): stats([s for r in records if r['background_group'] == group for s in r['spots']]) for group in [1, 2]}
    RESULT.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    return result


def values(records):
    return np.array([s['theta_deg'] for r in records for s in r['spots'] if s['selection'] == 'kept' and s['theta_deg'] is not None])


def render(data):
    assert data['complete'] and len(data['records']) == 16
    # Embed only selected spots in the HTML; full rejection records stay in JSON.
    public = {k: v for k, v in data.items() if k != 'records'}
    public['records'] = [{**r, 'spots': [s for s in r['spots'] if s['selection'] == 'kept']} for r in data['records']]
    (OUT/'hinges95_theta.html').write_text(HTML.replace('__DATA__', json.dumps(public).replace('</', '<\\/')), encoding='utf-8')
    edges = np.arange(0, 92, 2); records = data['records']; pooled = values(records)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    counts, _ = np.histogram(pooled, edges); assert counts.sum() == data['pooled']['valid']
    np.testing.assert_array_equal(counts, sum(np.array(r['summary']['counts_2deg']) for r in records))
    axes[0].stairs(counts, edges, fill=True, color='#087e8b', alpha=.5)
    axes[0].axvline(np.median(pooled), color='#285258', linestyle=':', label=f'Median {np.median(pooled):.1f} degrees')
    axes[0].set(title=f"All 16 stacks pooled: {len(pooled):,} valid angles", ylabel='Spots per 2-degree bin')
    axes[0].legend(fontsize=9)
    for group, color, label in [(1, '#087e8b', 'Stacks 1-9 / background 1'), (2, '#b76e1b', 'Stacks 10-16 / background 2')]:
        a = values([r for r in records if r['background_group'] == group]); c, _ = np.histogram(a, edges)
        axes[1].stairs(c/len(a)*100, edges, color=color, linewidth=1.7, label=f'{label} (n={len(a):,})')
    axes[1].set(title='Background groups, normalized to compare shape', ylabel='Percent of valid angles per 2-degree bin')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set(xlabel='Theta from optical axis (degrees)', xlim=(0, 90)); ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.15)
    fig.suptitle('95th-percentile background for detection | minimum background for theta\n2 SD; 150-frame means; ellipse excluded; oversized allowed; raw saturation excluded; strict overlap', fontsize=11)
    for ext in ['png', 'svg']: fig.savefig(OUT/f'hinges95_theta_pooled.{ext}', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True, layout='constrained')
    for ax, r in zip(axes.flat, records):
        a = values([r]); c, _ = np.histogram(a, edges)
        color = '#087e8b' if r['background_group'] == 1 else '#b76e1b'
        ax.stairs(c/len(a)*100, edges, fill=True, alpha=.45, color=color)
        ax.axvline(np.median(a), color=color, linestyle=':', linewidth=1)
        ax.set_title(f"#{r['index']} {r['time']} | n={len(a)}"+(' | extra check' if r['extra_check'] else ''), fontsize=10)
        ax.set_xlim(0, 90); ax.set_xticks([0, 30, 60, 90]); ax.grid(axis='y', alpha=.15); ax.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('Theta from optical axis (degrees)'); fig.supylabel('Percent of valid angles per 2-degree bin')
    fig.suptitle('16 hinge stacks separately | detect on p95, extract angles from minimum BG\nTeal: background 1; amber: background 2 | dotted lines: medians', fontsize=13)
    for ext in ['png', 'svg']: fig.savefig(OUT/f'hinges95_theta_separate.{ext}', dpi=160)
    plt.close(fig)
    print(json.dumps({'pooled': data['pooled'], 'groups': data['groups']}), flush=True)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>16 hinge stacks · p95 detection, minimum theta</title>
<style>body{font:16px/1.5 system-ui;color:#263637;background:#f7f8f5;margin:0}main{max-width:1480px;margin:auto;padding:24px}h1{font-size:27px}h2{font-size:21px}p{max-width:1200px}a{color:#087985}.small{font-size:14px;color:#526369}select,button{font:inherit;padding:6px}label{display:inline-block;margin:5px 16px 5px 0}.plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.plot{background:white;border:1px solid #d6ded8;padding:12px}.plot svg{width:100%;height:auto}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.cards .plot{padding:8px}.cards h3{font-size:15px;margin:4px}table{border-collapse:collapse}td,th{padding:7px 14px 7px 0;text-align:left;border-bottom:1px solid #d3ddd7}.scroll{overflow:auto}.controls{position:sticky;top:0;background:#f7f8f5f5;z-index:2}#selectedNote{white-space:pre-wrap}@media(max-width:950px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.plots,.cards{grid-template-columns:1fr}}</style><main>
<h1>16 hinge stacks: p95 detection, minimum-background theta</h1>
<p><b>12 October, wide-field hinges #1–16.</b> Spots are detected on the <b>95th-percentile-subtracted 150-frame mean at 2 SD</b>. Theta uses the four <b>minimum-background-subtracted</b> channel intensities at those same coordinates. The top-right mask stays on; oversized components are allowed; raw-saturated candidates are excluded; previous strict overlap suppression stays on.</p>
<div class="controls"><label>Stacks <select id="group"><option value="all">All 16</option><option value="1">1–9: background 1</option><option value="2">10–16: background 2</option></select></label><label><input type="checkbox" id="extra" checked>Include #16 (extra angle check)</label><label>Theta bins <select id="bin"><option value="1">1°</option><option value="2" selected>2°</option><option value="5">5°</option></select></label><label>Y axis <select id="units"><option value="percent">Percent per bin</option><option value="counts">Spot counts</option></select></label></div>
<p id="summary"></p>
<div class="plots"><section class="plot"><h2 id="pooledTitle"></h2><svg id="pooled" viewBox="0 0 650 360" role="img" aria-label="Pooled theta histogram"></svg></section><section class="plot"><h2>Background groups</h2><svg id="groups" viewBox="0 0 650 360" role="img" aria-label="Background groups compared"></svg><div id="legend" class="small"></div></section></div>
<p class="small">Pooling gives each valid detected spot equal weight; it is not an average giving each stack equal weight. Percentages use only valid theta values, with above-limit counts reported explicitly. Recordings may overlap in sample position; pooled counts are detections, not verified unique rods or independent biological replicates.</p>
<h2>Each stack separately</h2><p class="small">All small plots share the same y-axis scale. Teal: background capture 18:36:50 (45 frames); amber: 18:50:32 (44 frames). Each group uses its own background’s 95th percentile. Click a stack title to inspect its values below.</p><div id="cards" class="cards"></div>
<h2>Selected stack</h2><label>Stack <select id="stack"></select></label><p id="selectedNote" class="small"></p><div class="plots"><section class="plot"><svg id="single" viewBox="0 0 650 360" role="img" aria-label="Selected stack theta"></svg></section><section class="plot"><h2>Selection counts</h2><div id="singleStats"></div></section></div>
<h2>Counts and medians</h2><div id="table" class="scroll"></div>
<details><summary>Method and limits</summary><p>Each saved stack has already had its recorded minimum background subtracted and clipped at zero. Detection uses additional subtraction of B95−Bmin, applied per frame with float32 subtraction, zero clipping and integer rounding, then averaged over the 150 real frames. Synthetic marker frames are excluded. P95 is calculated per sensor pixel from the original raw background movie with linear interpolation. The minimum is verified against the recorded profile.</p><p>GUI intensity detector: 2×2 polarization-cell means repeated to the sensor grid; 10-pixel edge crop; DoG σ=1 and 6 sensor pixels; threshold=max(0, global filtered mean+2SD), recalculated for every stack; 8-connected components of area ≥10 sensor pixels with no maximum. Exclude centres inside the fixed ellipse (2060,500), radii (320,220). Raw saturation is any pixel ≥4094 across all 150 frames, in the union of the 18×18 extraction aperture and complete component bounding rectangle. The raw-peak file is checked by hash.</p><p>For overlapping 18×18 windows, the component with the strongest p95 DoG peak is kept first; any shared pixel suppresses a weaker candidate. Theta extraction retains the fixed 18×18 aperture even for oversized components. A merged component is not automatically resolved into individual rods. Undefined minimum-background channel ratios are excluded; finite r above the model limit is reported and has no theta, rather than being clipped.</p><p>Water Fresnel+BFP+annular-hole model, NA_out=1.3, NA_in=0.39, n_water=1.33, n_oil=1.515. X=(I0−I90)/(I0+I90), Y=(I45−I135)/(I45+I135), r=√(X²+Y²), θ=asin√[Ar/(C−Br)]. Theta is measured from the optical axis. No APD correction matrix, extra gain fit, or solid-angle-density correction is applied. The selected distribution still reflects the p95 detection threshold.</p><p>The first-stack coordinates and minimum-background r/theta values are verified against the previous single-stack analysis. Group comparisons are descriptive; changing the background coincided with changing fields and time, so their differences cannot be attributed to background choice alone.</p></details>
<p><a href="hinges95_theta_pooled.png">Pooled + background groups (PNG)</a> · <a href="hinges95_theta_separate.png">All 16 separately (PNG)</a> · <a href="hinges95_theta_pooled.svg">Pooled SVG</a> · <a href="hinges95_theta_separate.svg">Separate SVG</a> · <a href="hinges95_theta.json">Per-spot values and provenance</a></p><p class="small">Downloads use all 16 stacks and 2° bins; interactive controls affect the on-page plots.</p>
</main><script>
const DATA=__DATA__,$=id=>document.getElementById(id),COLORS={'1':'#087e8b','2':'#b76e1b'};
function records(){return DATA.records.filter(r=>($('group').value==='all'||String(r.background_group)===$('group').value)&&($('extra').checked||!r.extra_check))}
function theta(rs){return rs.flatMap(r=>r.spots.map(s=>s.theta_deg).filter(v=>v!==null))}
function med(a){const b=a.slice().sort((x,y)=>x-y);return b.length?(b[Math.floor((b.length-1)/2)]+b[Math.floor(b.length/2)])/2:null}
function hist(rs){const width=Number($('bin').value),n=90/width,a=Array(n).fill(0),v=theta(rs);for(const x of v)a[Math.min(n-1,Math.floor(x/width))]++;return $('units').value==='percent'?a.map(c=>100*c/Math.max(1,v.length)):a}
function svg(series,small=false,sharedMax=null){const W=650,H=360,L=58,R=18,T=20,B=57,width=Number($('bin').value),ys=series.map(s=>({...s,counts:hist(s.records)})),maximum=sharedMax??Math.max(1,...ys.flatMap(s=>s.counts)),ymax=maximum*1.1,x=v=>L+v/90*(W-L-R),y=v=>H-B-v/ymax*(H-T-B);let out='';
for(let i=0;i<=4;i++){const v=ymax*i/4;out+=`<path d="M${L},${y(v)}H${W-R}" stroke="#e1e7e4"/><text x="${L-8}" y="${y(v)+5}" text-anchor="end" font-size="${small?18:12}">${v.toFixed($('units').value==='percent'?1:0)}</text>`}
for(let v=0;v<=90;v+=30)out+=`<text x="${x(v)}" y="${H-B+24}" text-anchor="middle" font-size="${small?18:12}">${v}</text>`;
for(const s of ys){const a=s.counts;if(ys.length===1)for(let j=0;j<a.length;j++)out+=`<rect x="${x(j*width)}" y="${y(a[j])}" width="${x((j+1)*width)-x(j*width)}" height="${y(0)-y(a[j])}" fill="${s.color}" opacity=".35"/>`;let d=`M${x(0)},${y(a[0])}`;for(let j=0;j<a.length;j++)d+=`L${x(j*width)},${y(a[j])}L${x((j+1)*width)},${y(a[j])}`;out+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2"/>`}
out+=`<path d="M${L},${T}V${H-B}H${W-R}" fill="none" stroke="#526369"/><text x="${W/2}" y="${H-10}" text-anchor="middle" font-size="${small?18:14}">Theta (degrees)</text>`;if(!small)out+=`<text transform="translate(17,${H/2}) rotate(-90)" text-anchor="middle" font-size="13">${$('units').value==='percent'?'Percent':'Spots'} per ${width}° bin</text>`;return out}
function pick(index){$('stack').value=String(index);selected()}
function selected(){const r=records().find(r=>String(r.index)===$('stack').value);if(!r)return;const s=r.summary;$('single').innerHTML=svg([{records:[r],color:COLORS[r.background_group]}]);$('selectedNote').textContent=`#${r.index} · ${r.time} · ${r.name}\nNote: ${r.note}`;$('singleStats').innerHTML=`<p>${s.detected} candidates (area ≥10).<br>${s.selected} retained; ${s.valid} valid theta; ${s.above_limit} above r_sat.<br>${s.oversized} oversized retained.<br>${s.exclusions.region} excluded by ellipse; ${s.exclusions.saturated} then excluded by saturation; ${s.exclusions.invalid} undefined r; ${s.exclusions.overlap} suppressed by overlap.<br>${s.raw_saturated} raw-saturated candidates in total, including any already masked.<br>Median theta ${s.median_theta.toFixed(1)}°; median r ${s.median_r.toFixed(3)}.</p>`}
function refresh(){const rs=records(),v=theta(rs),selectedCount=rs.reduce((a,r)=>a+r.summary.selected,0),invalid=rs.reduce((a,r)=>a+r.summary.above_limit,0),sat=rs.reduce((a,r)=>a+r.summary.raw_saturated,0);$('summary').textContent=`${rs.length} stacks · ${selectedCount.toLocaleString()} selected detections · ${v.length.toLocaleString()} valid angles · ${invalid} above r_sat · ${sat} raw-saturated candidates excluded. Pooled median theta ${med(v).toFixed(1)}°.`;$('pooledTitle').textContent=`Together: ${v.length.toLocaleString()} valid angles`;$('pooled').innerHTML=svg([{records:rs,color:'#087e8b'}]);const groups=[1,2].map(g=>({records:rs.filter(r=>r.background_group===g),color:COLORS[g],g})).filter(g=>g.records.length);$('groups').innerHTML=svg(groups);$('legend').innerHTML=groups.map(g=>`<span style="color:${g.color};margin-right:16px">BG ${g.g}: ${g.records.length} stacks, n=${theta(g.records).length}, median ${med(theta(g.records)).toFixed(1)}°</span>`).join('');
const ymax=Math.max(1,...rs.flatMap(r=>hist([r])));$('cards').innerHTML=rs.map(r=>`<section class="plot"><h3><a href="#selectedNote" onclick="pick(${r.index})">#${r.index} · ${r.time}${r.extra_check?' · extra check':''}</a></h3><svg viewBox="0 0 650 360">${svg([{records:[r],color:COLORS[r.background_group]}],true,ymax)}</svg><div class="small">n=${r.summary.valid}; median ${r.summary.median_theta.toFixed(1)}°</div></section>`).join('');
const old=$('stack').value;$('stack').innerHTML=rs.map(r=>`<option value="${r.index}">#${r.index} · ${r.time}</option>`).join('');$('stack').value=rs.some(r=>String(r.index)===old)?old:String(rs[0].index);selected();
$('table').innerHTML='<table><tr><th>Stack</th><th>Time</th><th>BG</th><th>Selected</th><th>Valid theta</th><th>Above r_sat</th><th>Oversized retained</th><th>Raw saturated</th><th>Median theta</th></tr>'+rs.map(r=>{const s=r.summary;return `<tr><td>${r.index}${r.extra_check?' (check)':''}</td><td>${r.time}</td><td>${r.background_group}</td><td>${s.selected}</td><td>${s.valid}</td><td>${s.above_limit}</td><td>${s.oversized}</td><td>${s.raw_saturated}</td><td>${s.median_theta.toFixed(1)}°</td></tr>`}).join('')+'</table>'}
for(const id of ['group','extra','bin','units'])$(id).onchange=refresh;$('stack').onchange=selected;refresh();
</script></html>'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--render-only', action='store_true'); args = parser.parse_args()
    render(json.loads(RESULT.read_text(encoding='utf-8')) if args.render_only else calculate())
