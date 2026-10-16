"""Compare 2-SD detections under minimum, median and p95 backgrounds."""
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detector_grid_trial import HERE, DAY, NAME, match, sha
from background_region_trial import ELLIPSE, inside
from background_percentile_trial import OUT, signature
from detector_threshold_trial import select_spots
from polarcam.live import Detection_alg_offline as gui
from polarcam.live.intensity_spots import channel_means, anisotropy

METHODS = [('minimum', 0, 'Minimum'), ('median', 50, 'Median'), ('p95', 95, '95th percentile')]
PAIRS = [('minimum', 'median'), ('minimum', 'p95'), ('median', 'p95')]


def detect(frame, raw_peak, prefix):
    image = gui.intensity_detection_map(frame)
    work = image[10:-10, 10:-10]
    dog = (cv2.GaussianBlur(work, (0, 0), sigmaX=1, borderType=cv2.BORDER_REPLICATE)
           - cv2.GaussianBlur(work, (0, 0), sigmaX=6, borderType=cv2.BORDER_REPLICATE))
    mu, sd = gui._finite_stats(dog)
    threshold = max(0, mu + 2*sd)
    _, labels, stats, centers = cv2.connectedComponentsWithStats((dog > threshold).astype(np.uint8), connectivity=8)
    rows = []
    for component in range(1, len(stats)):
        bx, by, w, h, area = map(int, stats[component])
        if area < 10:
            continue
        x, y = map(float, centers[component] + 10)
        l, t = round(x)-9, round(y)-9
        bounds = [l, l+18, t, t+18]
        assert 0 <= l and 0 <= t and l+18 <= frame.shape[1] and t+18 <= frame.shape[0]
        peak = int(raw_peak[t:t+18, l:l+18].max())
        xy = anisotropy(channel_means(frame, [bounds])[0])
        valid = bool(np.isfinite(xy).all())
        value = float(dog[by:by+h, bx:bx+w][labels[by:by+h, bx:bx+w] == component].max())
        rows.append(dict(id=f'{prefix}-{component}', component=component, x=x, y=y,
                         area=area, bbox=[bx+9.5, by+9.5, w, h], bounds=bounds,
                         dog_peak=value, raw_peak=peak, saturated=peak >= 4094,
                         accepted=area <= 250, inside_ellipse=bool(inside(x, y)),
                         r=float(np.linalg.norm(xy)) if valid else None))
    candidates = [s for s in rows if s['accepted']]
    direct = np.asarray(gui.find_spot_centers_dog(work, k_std=2)) + 10
    np.testing.assert_allclose([[s['x'], s['y']] for s in candidates], direct, rtol=0, atol=1e-12)
    select_spots(rows)
    counts = {reason: sum(s['selection'] == reason for s in rows)
              for reason in ['kept', 'overlap', 'oversize', 'region', 'saturated', 'invalid']}
    return dict(spots=rows, threshold=threshold, mean=mu, sd=sd, counts=counts,
                detected=len(candidates)), image


def selected(result, policy):
    allowed = ['kept'] if policy == 'strict' else ['kept', 'overlap']
    return [s for s in result['spots'] if s['selection'] in allowed]


def comparison(a, b):
    pairs, lost, added = match(a, b)
    shared = [dict(before=a[i]['id'], after=b[j]['id'],
                   distance=float(np.hypot(a[i]['x']-b[j]['x'], a[i]['y']-b[j]['y'])))
              for i, j in pairs.items()]
    assert len(a) == len(shared)+len(lost) and len(b) == len(shared)+len(added)
    return dict(shared=shared, removed=[a[i]['id'] for i in lost], added=[b[j]['id'] for j in added],
                counts=dict(before=len(a), after=len(b), shared=len(shared), removed=len(lost), added=len(added)))


def build():
    prior = json.loads((OUT/'percentile_results.json').read_text())
    source = DAY/'frames'/NAME
    meta = json.loads(source.with_suffix('.json').read_text())
    assert signature(source) == prior['source_signature']
    assert sha(source.with_suffix('.json')) == prior['sidecar_sha256']
    peakmeta = meta['fetch_frames_output']['raw_peak']
    peakpath = source.parent/peakmeta['path']
    assert peakmeta['frames'] == 150 and sha(peakpath) == peakmeta['sha256']
    peak = np.load(peakpath)
    frames = np.load(source, mmap_mode='r')
    assert len(frames) == 151 and meta['actual']['frames'] == 150
    with np.load(OUT/'percentile_arrays.npz') as cache:
        ix = [list(cache['percentiles']).index(p) for _, p, _ in METHODS]
        profiles = cache['background_profiles'][ix]
        sums = cache['sum_frames'][ix]
    additions = profiles-profiles[0]
    assert np.all(additions >= 0)
    first = [np.rint(np.maximum(frames[0].astype(np.float32)-d, 0)).astype(np.uint16) for d in additions]
    np.testing.assert_array_equal(first[0], frames[0])
    results = {}
    for mode, inputs in [('first', first), ('mean', sums.astype(np.float64)/150)]:
        methods = {}
        for (key, percentile, label), frame in zip(METHODS, inputs):
            entry, image = detect(frame, peak, f'{mode}-{key}')
            imagefile = f'percentile_spots_{mode}_{key}.webp'
            Image.fromarray(np.rint(np.clip(image*20/4095, 0, 1)*255).astype(np.uint8)).save(OUT/imagefile, lossless=True)
            entry.update(label=label, percentile=percentile, image=imagefile)
            methods[key] = entry
        comparisons = {}
        for policy in ['strict', 'no_overlap']:
            comparisons[policy] = {f'{a}_{b}': comparison(selected(methods[a], policy), selected(methods[b], policy)) for a, b in PAIRS}
        results[mode] = dict(methods=methods, comparisons=comparisons)
        print(mode, json.dumps({k: v['counts'] for k, v in methods.items()}), flush=True)
    # The mean/minimum branch must reproduce the earlier threshold gallery.
    old = json.loads((HERE/'detector_grid_trial/threshold_results.json').read_text())['levels']['2']
    current = results['mean']['methods']['minimum']
    assert current['detected'] == old['counts']['accepted'] == 619
    assert current['counts']['kept'] == old['cuts']['kept'] == 449
    np.testing.assert_allclose([[s['x'], s['y']] for s in current['spots']], [[s['x'], s['y']] for s in old['spots']], rtol=0, atol=1e-12)
    assert signature(source) == prior['source_signature']
    data = dict(source=NAME, shape=list(peak.shape), ellipse=ELLIPSE, modes=results,
                source_signature=prior['source_signature'], sidecar_sha256=prior['sidecar_sha256'],
                background_capture_sha256=prior['background_capture_sha256'],
                percentile_results_sha256=sha(OUT/'percentile_results.json'),
                raw_peak_sha256=peakmeta['sha256'], detector_sha256=sha(Path(gui.__file__)),
                builder_sha256=sha(Path(__file__)),
                method='GUI intensity detector at k=2: 2x2 cell means repeated to sensor grid; '
                       '10-pixel edge crop; DoG sigma 1 and 6; threshold max(0, mean+2*SD) '
                       'recalculated separately on each complete filtered image; 8-connectivity; '
                       'component area 10-250 sensor pixels. Display gain is irrelevant to detection.',
                selection='Previous large ellipse removed, any raw pixel >=4094 over all 150 frames '
                          'in 18x18 aperture removed, undefined r removed but no finite r upper cutoff. '
                          'Strict overlap option: strongest DoG component peak wins if apertures share '
                          'even one pixel. Alternative disables only this last overlap cut. '
                          'No low-background/survival/left-half cuts.',
                matching='One-to-one minimum-distance centroid assignment within 6 sensor pixels, '
                         'independently for each pair and selection. Added/removed are unmatched '
                         'detections, not confirmed appearance/disappearance of physical rods. '
                         'Threshold changes, centroid shifts, component splitting/merging and size '
                         'or overlap filters can all change these lists.')
    (OUT/'percentile_spots.json').write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    (OUT/'percentile_spots.html').write_text(HTML.replace('__DATA__', json.dumps(data).replace('</', '<\\/')), encoding='utf-8')
    # Downloadable maps use the default strict selection and first saved frame.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout='constrained')
    for ax, (a, b) in zip(axes, PAIRS):
        methods = results['first']['methods']; comp = results['first']['comparisons']['strict'][f'{a}_{b}']
        lookup = {s['id']: s for v in methods.values() for s in v['spots']}
        ax.imshow(Image.open(OUT/methods[b]['image']), cmap='gray', vmin=0, vmax=255)
        for ids, color, marker in [([p['after'] for p in comp['shared']], '#65e2e6', 'o'),
                                   (comp['added'], '#ffb331', 'o'), (comp['removed'], '#ff6262', 'x')]:
            points = [lookup[i] for i in ids]
            if points:
                kwargs = dict(c=color) if marker == 'x' else dict(facecolors='none', edgecolors=color)
                ax.scatter([s['x'] for s in points], [s['y'] for s in points], s=10, marker=marker, linewidths=.45, **kwargs)
        c = comp['counts']
        ax.set_title(f"{methods[a]['label']} to {methods[b]['label']}\n{c['shared']} shared / +{c['added']} / -{c['removed']}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('First saved frame | DoG 2 SD | fixed 20x display\nCyan: shared; amber: added; red crosses: removed | previous ellipse and strict overlap cuts', fontsize=12)
    fig.savefig(OUT/'percentile_spots_overview.png', dpi=160); plt.close(fig)
    print(json.dumps({mode: {p: v['counts'] for p, v in r['comparisons']['strict'].items()} for mode, r in results.items()}), flush=True)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2-SD spots · background comparison</title>
<style>body{font:16px/1.5 system-ui;color:#263637;background:#f7f8f5;margin:0}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:27px}h2{font-size:21px}a{color:#087985}p{max-width:1200px}.small{font-size:14px;color:#526369}select,button{font:inherit;padding:6px}label{display:inline-block;margin:5px 16px 5px 0}.controls{position:sticky;top:0;background:#f7f8f5f5;z-index:2;padding:8px 0}#view{max-height:74vh;overflow:auto;background:#111}#map{width:100%;height:auto;display:block;cursor:crosshair}.crops{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.crops canvas{width:100%;height:auto;image-rendering:pixelated;background:#111}figure{margin:0}figcaption{font-weight:600}table{border-collapse:collapse}td,th{padding:7px 14px 7px 0;text-align:left;border-bottom:1px solid #d3ddd7}.scroll{overflow:auto}#error{color:#a22}@media(max-width:750px){.crops{grid-template-columns:1fr}}</style><main>
<h1>2-SD spots: minimum, median and 95th-percentile background</h1>
<p><b>First hinge stack, 18:38:50 on 12 October.</b> The default uses its <b>first saved frame</b>. Switch to the <b>150-frame mean</b> to use the same input as the preceding image comparison. Each background comes from the same 45-frame capture; subtraction, zero clipping and integer rounding precede averaging. All views use fixed 20× display.</p>
<p><a href="percentile95_region.html"><b>Inspect 95th-percentile spots with the top-right mask off and oversized components allowed</b></a></p>
<div class="controls"><label>Detection input <select id="mode"><option value="first">First saved frame (frame 1)</option><option value="mean">Mean of 150 frames (previous page)</option></select></label><label>Compare <select id="pair"><option value="minimum_median">Minimum → median</option><option value="minimum_p95">Minimum → 95th</option><option value="median_p95">Median → 95th</option></select></label><label><input type="checkbox" id="strict" checked>Previous strict overlap filter</label><br>
<label>Image <select id="layer"><option value="minimum">Minimum</option><option value="median" selected>Median</option><option value="p95">95th percentile</option></select></label><label><input type="checkbox" id="shared" checked>Shared · cyan circles</label><label><input type="checkbox" id="added" checked>Added · amber circles</label><label><input type="checkbox" id="removed" checked>Removed · red crosses</label><label>Zoom <select id="zoom"><option value="fit">Fit width</option><option value="native">Native pixels</option></select></label></div>
<p id="counts"></p><p class="small">Added/removed are relative to the selected arrow, using one-to-one centroid matches within 6 sensor pixels. Markers stay fixed when switching the displayed background. A changed detection can be a shifted, split or merged component; it does not necessarily mean a physical rod appeared or disappeared.</p>
<div id="error"></div><div id="view"><canvas id="map" width="2464" height="2056"></canvas></div>
<h2>Inspect the same position under all three backgrounds</h2>
<p><label>Step through <select id="browse"><option value="changes">Added and removed</option><option value="added">Added only</option><option value="removed">Removed only</option></select></label><button id="prev">Previous spot</button> <button id="next">Next spot</button> <span id="position" class="small"></span></p>
<p class="small">Or click anywhere in the large map. Each crop is 88×88 sensor pixels. All three show the same selected comparison markers; turn the marker checkboxes off for an unobstructed view.</p>
<div class="crops"><figure><figcaption>Minimum</figcaption><canvas id="crop_minimum" width="352" height="352"></canvas></figure><figure><figcaption>Median</figcaption><canvas id="crop_median" width="352" height="352"></canvas></figure><figure><figcaption>95th percentile</figcaption><canvas id="crop_p95" width="352" height="352"></canvas></figure></div>
<p id="detail" class="small"></p><div id="nearby" class="scroll"></div>
<h2>Detection and exclusion counts</h2><div id="table" class="scroll"></div>
<details><summary>Exact detector and matching rules</summary><p>Same GUI intensity detector: average each 2×2 polarization cell and repeat to the sensor grid; crop a 10-pixel border; subtract Gaussian σ=6 from σ=1; keep pixels above the whole filtered image’s mean + 2 SD (at least zero); 8-connected components with area 10–250 sensor pixels. The numerical cutoff is recalculated separately for each background/input. This can create new detections even when every pixel becomes dimmer.</p>
<p>The previous large red ellipse is excluded, as are candidates whose 18×18 extraction aperture contains a raw pixel ≥4094 at any time in the 150-frame stack. The raw-peak companion records intensity before background subtraction. Undefined-r candidates are excluded as before; no finite-r or theta limit is used. No survival, low-background or left-half filter is applied.</p>
<p><b>Strict overlap on:</b> keep the strongest component DoG peak first; suppress weaker candidates whose 18×18 windows share even one pixel with a retained candidate. This is the earlier rule that rejected the bright pair. Turning this off restores only overlap-suppressed candidates, making its contribution to the added/removed list visible. Other exclusions stay unchanged.</p></details>
<p><a href="percentile_comparison.html">Background images without markers</a> · <a href="percentile_spots_overview.png">First-frame comparison image</a> · <a href="percentile_spots.json">Coordinates, matches and provenance</a></p>
</main><script>
const DATA=__DATA__,$=id=>document.getElementById(id),images={},keys=['minimum','median','p95'];let point={x:541,y:515},cursor=-1;
function state(){return DATA.modes[$('mode').value]}
function policy(){return $('strict').checked?'strict':'no_overlap'}
function comp(){return state().comparisons[policy()][$('pair').value]}
function lookup(){return new Map(Object.values(state().methods).flatMap(m=>m.spots).map(s=>[s.id,s]))}
function load(mode,key){const id=mode+'_'+key;if(images[id])return images[id];const im=new Image();images[id]=im;im.onload=draw;im.onerror=()=>{$('error').textContent='Could not load '+DATA.modes[mode].methods[key].image};im.src=DATA.modes[mode].methods[key].image;return im}
function markers(){const c=comp(),spots=lookup();return [...c.shared.map(p=>({s:spots.get(p.after),status:'shared'})),...c.added.map(id=>({s:spots.get(id),status:'added'})),...c.removed.map(id=>({s:spots.get(id),status:'removed'}))]}
function overlay(ctx,ox=0,oy=0,scale=1,box=true){ctx.save();ctx.translate(ox,oy);ctx.scale(scale,scale);ctx.lineWidth=1.6/scale;for(const m of markers()){if(!$(m.status).checked)continue;const s=m.s;ctx.strokeStyle={shared:'#65e2e6',added:'#ffb331',removed:'#ff6262'}[m.status];ctx.beginPath();if(m.status==='removed'){ctx.moveTo(s.x-7,s.y-7);ctx.lineTo(s.x+7,s.y+7);ctx.moveTo(s.x+7,s.y-7);ctx.lineTo(s.x-7,s.y+7)}else ctx.arc(s.x,s.y,8,0,Math.PI*2);ctx.stroke()}
const e=DATA.ellipse;ctx.strokeStyle='#b65d64';ctx.setLineDash([5/scale,5/scale]);ctx.beginPath();ctx.ellipse(e.cx,e.cy,e.rx,e.ry,0,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);if(box){ctx.strokeStyle='#fff';ctx.strokeRect(point.x-44,point.y-44,88,88)}ctx.restore()}
function draw(){const mode=$('mode').value,ctx=$('map').getContext('2d'),im=load(mode,$('layer').value);ctx.clearRect(0,0,2464,2056);if(im.complete&&im.naturalWidth)ctx.drawImage(im,0,0);overlay(ctx);for(const key of keys){const c=$('crop_'+key).getContext('2d'),source=load(mode,key),l=point.x-44,t=point.y-44;c.clearRect(0,0,352,352);c.imageSmoothingEnabled=false;if(source.complete&&source.naturalWidth)c.drawImage(source,l,t,88,88,0,0,352,352);overlay(c,-l*4,-t*4,4,false)}inspect()}
function browseList(){return markers().filter(m=>$('browse').value==='changes'?m.status!=='shared':m.status===$('browse').value).sort((a,b)=>a.s.y-b.s.y||a.s.x-b.s.x)}
function setPoint(x,y){point={x:Math.max(44,Math.min(2420,x)),y:Math.max(44,Math.min(2012,y))}}
function step(delta){const list=browseList();if(!list.length)return;cursor=cursor<0?(delta>0?0:list.length-1):(cursor+delta+list.length)%list.length;const m=list[cursor];setPoint(m.s.x,m.s.y);$('position').textContent=`${cursor+1} / ${list.length}: ${m.status} · ${m.s.id}`;draw()}
function inspect(){let nearest=markers().sort((a,b)=>Math.hypot(a.s.x-point.x,a.s.y-point.y)-Math.hypot(b.s.x-point.x,b.s.y-point.y))[0];$('detail').textContent=`Crop centre x=${point.x.toFixed(1)}, y=${point.y.toFixed(1)}. `+(nearest&&Math.hypot(nearest.s.x-point.x,nearest.s.y-point.y)<=20?`${nearest.status}: ${nearest.s.id}; area ${nearest.s.area}; DoG peak ${nearest.s.dog_peak.toFixed(2)}; raw peak ${nearest.s.raw_peak}/4094.`:'No comparison marker within 20 pixels.');
$('nearby').innerHTML='<table><tr><th>Background</th><th>Nearest component within 20 px</th><th>Area</th><th>DoG peak</th><th>Previous selection</th></tr>'+keys.map(key=>{const m=state().methods[key],s=m.spots.filter(s=>Math.hypot(s.x-point.x,s.y-point.y)<=20).sort((a,b)=>Math.hypot(a.x-point.x,a.y-point.y)-Math.hypot(b.x-point.x,b.y-point.y))[0];return `<tr><td>${m.label}</td><td>${s?s.id:'None (area ≥10)'}</td><td>${s?s.area:'—'}</td><td>${s?s.dog_peak.toFixed(2):'—'}</td><td>${s?s.selection:'—'}</td></tr>`}).join('')+'</table>'}
function refresh(){cursor=-1;$('position').textContent='';const c=comp().counts;$('counts').textContent=`${c.before} → ${c.after} retained candidates: ${c.shared} shared, +${c.added} added, −${c.removed} removed.`;
$('table').innerHTML='<table><tr><th>Background</th><th>2-SD cutoff</th><th>Area 10–250</th><th>Ellipse</th><th>Raw saturated</th><th>Undefined r</th><th>Overlap</th><th>Retained</th><th>Oversized</th></tr>'+keys.map(key=>{const m=state().methods[key],c=m.counts;return `<tr><td>${m.label}</td><td>${m.threshold.toFixed(3)}</td><td>${m.detected}</td><td>${c.region}</td><td>${c.saturated}</td><td>${c.invalid}</td><td>${$('strict').checked?c.overlap:0}</td><td>${c.kept+($('strict').checked?0:c.overlap)}</td><td>${c.oversize}</td></tr>`}).join('')+'</table>';draw()}
for(const id of ['mode','strict'])$(id).onchange=refresh;$('pair').onchange=()=>{$('layer').value=$('pair').value.split('_')[1];refresh()};
for(const id of ['layer','shared','added','removed'])$(id).onchange=draw;
$('browse').onchange=()=>{cursor=-1;$('position').textContent=''};$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('zoom').onchange=()=>{$('map').style.width=$('zoom').value==='native'?'2464px':'100%'};
$('map').onclick=e=>{const r=$('map').getBoundingClientRect();setPoint((e.clientX-r.left)*2464/r.width,(e.clientY-r.top)*2056/r.height);cursor=-1;$('position').textContent='';draw()};refresh();
</script></html>'''

if __name__ == '__main__':
    build()
