"""Inspect p95 detections with the former ellipse restored; no theta statistics."""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from detector_grid_trial import DAY, NAME, sha
from detector_threshold_trial import select_spots
from background_percentile_trial import OUT, signature


def policy_key(mask, large, strict):
    return f'm{int(mask)}_a{int(large)}_o{int(strict)}'


def select(rows, mask, large, strict):
    trial = deepcopy(rows)
    for s in trial:
        s['accepted'] = large or s['area'] <= 250
        s['inside_ellipse'] = mask and s['geometric_inside']
    # Re-run greedy suppression after mask/size changes, rather than just
    # restoring old hidden candidates that may conflict with retained ones.
    select_spots(trial)
    if not strict:
        for s in trial:
            if s['selection'] == 'overlap':
                s['selection'] = 'kept'
                s['suppressed_by'] = None
    kept = [s for s in trial if s['selection'] == 'kept']
    assert all(not s['saturated'] for s in kept)
    assert not mask or all(not s['geometric_inside'] for s in kept)
    assert large or all(s['area'] <= 250 for s in kept)
    return dict(kept=[s['id'] for s in kept], inside=[s['id'] for s in kept if s['geometric_inside']],
                oversized=[s['id'] for s in kept if s['area'] > 250],
                selection={s['id']: s['selection'] for s in trial},
                suppressed_by={s['id']: s['suppressed_by'] for s in trial if s['suppressed_by']},
                counts={k: sum(s['selection'] == k for s in trial) for k in
                        ['kept', 'overlap', 'oversize', 'region', 'saturated', 'invalid']})


def build():
    path = OUT/'percentile_spots.json'
    data = json.loads(path.read_text())
    source = DAY/'frames'/NAME
    assert signature(source) == data['source_signature']
    assert sha(source.with_suffix('.json')) == data['sidecar_sha256']
    meta = json.loads(source.with_suffix('.json').read_text())
    info = meta['fetch_frames_output']['raw_peak']
    peakpath = source.parent/info['path']
    assert info['frames'] == 150 and sha(peakpath) == info['sha256'] == data['raw_peak_sha256']
    peak = np.load(peakpath)
    modes = {}
    for mode, original in data['modes'].items():
        methods = {}
        for key, method in original['methods'].items():
            rows = deepcopy(method['spots'])
            for s in rows:
                s['geometric_inside'] = s['inside_ellipse']
                s['aperture_raw_peak'] = s['raw_peak']
                x, y, w, h = s['bbox']
                left, top = round(x+.5), round(y+.5)
                assert left >= 0 and top >= 0 and left+w <= peak.shape[1] and top+h <= peak.shape[0]
                # Oversized detections can extend beyond the 18x18 aperture.
                # Conservatively check their entire component bounding rectangle.
                s['component_box_raw_peak'] = int(peak[top:top+h, left:left+w].max())
                s['raw_peak'] = max(s['aperture_raw_peak'], s['component_box_raw_peak'])
                s['saturated'] = s['raw_peak'] >= 4094
            policies = {policy_key(mask, large, strict): select(rows, mask, large, strict)
                        for mask in [False, True] for large in [False, True] for strict in [False, True]}
            # Unchanged earlier settings reproduce the existing retained list.
            previous = set(s['id'] for s in method['spots'] if s['selection'] == 'kept')
            assert set(policies['m1_a0_o1']['kept']) == previous
            methods[key] = dict(label=method['label'], image=method['image'], threshold=method['threshold'],
                                spots=rows, policies=policies)
        modes[mode] = dict(methods=methods)
    result = dict(source=NAME, modes=modes, ellipse=data['ellipse'],
                  input_sha256=sha(path), source_signature=data['source_signature'],
                  raw_peak_sha256=data['raw_peak_sha256'], raw_peak_max=int(peak.max()),
                  builder_sha256=sha(Path(__file__)),
                  selection='DoG mean+2SD, sigma 1/6, component area >=10 sensor pixels. '
                            'Default: p95 subtraction; no ellipse exclusion; no maximum component area; '
                            'previous strict overlap suppression. Mask, area maximum and overlap are '
                            'independently switchable. Every setting excludes raw saturation >=4094 '
                            'over all 150 frames in the union of the 18x18 extraction aperture and '
                            'full component bounding rectangle, before background subtraction. '
                            'Undefined r excluded as previously; no finite-r/theta cut. '
                            'Each policy re-runs strongest-first overlap suppression.',
                  scope='Inspection of the first hinge stack only. No theta distributions or raw-data changes.')
    (OUT/'percentile95_region.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    (OUT/'percentile95_region.html').write_text(HTML.replace('__DATA__', json.dumps(result).replace('</', '<\\/')), encoding='utf-8')
    e = result['ellipse']; view = (1680, 230, 2430, 780)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout='constrained')
    methods = modes['first']['methods']; p95 = methods['p95']; p = p95['policies']['m0_a1_o1']
    candidates = [s for s in p95['spots'] if s['id'] in p['inside']]
    for ax, key in zip(axes, ['minimum', 'median', 'p95']):
        ax.imshow(np.asarray(Image.open(OUT/methods[key]['image'])), cmap='gray', vmin=0, vmax=255)
        ax.scatter([s['x'] for s in candidates], [s['y'] for s in candidates], s=75,
                   facecolors='none', edgecolors='#ffb331', linewidths=.9)
        for s in candidates:
            if s['area'] > 250:
                ax.add_patch(Rectangle(s['bbox'][:2], s['bbox'][2], s['bbox'][3], fill=False, edgecolor='#ec79dc'))
        ax.add_patch(Ellipse((e['cx'], e['cy']), 2*e['rx'], 2*e['ry'], fill=False,
                            edgecolor='#ff6262', linestyle='--', linewidth=.7))
        ax.set_xlim(view[0], view[2]); ax.set_ylim(view[3], view[1]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(methods[key]['label']+' background · 20x')
    fig.suptitle(f"Former top-right exclusion | first saved frame\nSame {len(candidates)} p95 detections shown on all three images; oversized allowed; raw saturation excluded", fontsize=12)
    fig.savefig(OUT/'percentile95_region_overview.png', dpi=160); plt.close(fig)
    print(json.dumps({mode: {key: dict(total=len(m['policies']['m0_a1_o1']['kept']),
                        inside=len(m['policies']['m0_a1_o1']['inside']),
                        oversized=len(m['policies']['m0_a1_o1']['oversized']),
                        saturated=m['policies']['m0_a1_o1']['counts']['saturated'])
                    for key, m in v['methods'].items()} for mode, v in modes.items()}), flush=True)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>95th background · restore top-right region</title>
<style>body{font:16px/1.5 system-ui;color:#263637;background:#f7f8f5;margin:0}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:27px}h2{font-size:21px}a{color:#087985}p{max-width:1200px}.small{font-size:14px;color:#526369}select,button{font:inherit;padding:6px}label{display:inline-block;margin:5px 16px 5px 0}.controls{position:sticky;top:0;background:#f7f8f5f5;z-index:2;padding:8px 0}#view{max-height:74vh;overflow:auto;background:#111}#map{width:100%;height:auto;display:block;cursor:crosshair}.crops{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.crops canvas{width:100%;height:auto;image-rendering:pixelated;background:#111}figure{margin:0}figcaption{font-weight:600}table{border-collapse:collapse}td,th{padding:7px 14px 7px 0;text-align:left;border-bottom:1px solid #d3ddd7}.scroll{overflow:auto}.overview{width:100%}#error{color:#a22}@media(max-width:750px){.crops{grid-template-columns:1fr}}</style><main>
<h1>Can 95th-percentile subtraction replace the top-right mask?</h1>
<p><b>First hinge stack, 18:38:50 on 12 October.</b> Detection stays at <b>2 SD on the 95th-percentile-subtracted image</b>. The top-right region is restored by default, oversized components are allowed, and raw saturation remains excluded. This is a visual inspection before producing theta distributions.</p>
<img class="overview" src="percentile95_region_overview.png" alt="Formerly masked region under three background subtractions, with the same first-frame p95 detections">
<p class="small">The overview above always shows the first saved frame and default settings. The controls below change the interactive map and crops.</p>
<div class="controls"><label>Detection input <select id="mode"><option value="first">First saved frame</option><option value="mean">150-frame mean</option></select></label><label><input type="checkbox" id="mask">Exclude previous top-right ellipse</label><label><input type="checkbox" id="large" checked>Allow oversized components (&gt;250 px)</label><label><input type="checkbox" id="strict" checked>Previous strict overlap filter</label><br>
<label>Display background <select id="layer"><option value="minimum">Minimum</option><option value="median">Median</option><option value="p95" selected>95th percentile</option></select></label><label><input type="checkbox" id="marks" checked>Show detections</label><label><input type="checkbox" id="outline" checked>Show old mask outline</label><label>Zoom <select id="zoom"><option value="fit">Fit width</option><option value="native">Native pixels</option></select></label></div>
<p id="counts"></p><p class="small"><b>Amber circles:</b> retained inside the former mask. <b>Cyan circles:</b> retained outside. <b>Magenta rectangles:</b> retained oversized components. The dashed ellipse is a reference outline; it excludes spots only when its checkbox above is ticked. Changing the displayed background keeps p95 detections fixed.</p>
<div id="error"></div><div id="view"><canvas id="map" width="2464" height="2056"></canvas></div>
<h2>Inspect restored detections</h2><p><label>Browse <select id="browse"><option value="inside">Retained inside former mask</option><option value="oversized">Retained oversized components</option><option value="all">All retained detections</option></select></label><button id="prev">Previous</button> <button id="next">Next</button> <button id="feature">Centre of old mask</button> <span id="position" class="small"></span></p>
<div class="crops"><figure><figcaption>Minimum · 20×</figcaption><canvas id="crop_minimum" width="352" height="352"></canvas></figure><figure><figcaption>Median · 20×</figcaption><canvas id="crop_median" width="352" height="352"></canvas></figure><figure><figcaption>95th · 20×</figcaption><canvas id="crop_p95" width="352" height="352"></canvas></figure></div>
<p id="detail" class="small"></p>
<h2>How many detections remain in this region?</h2><p class="small">Here each background has its own 2-SD detections, using the same current size, mask and overlap settings. The interactive map above always uses p95 detections.</p><div id="table" class="scroll"></div>
<details><summary>Selection details</summary><p>GUI intensity detector: repeated 2×2 cell means; 10-pixel edge crop; DoG widths 1 and 6 sensor pixels; threshold recalculated as whole filtered image mean + 2 SD; 8-connected component area ≥10 pixels. With oversized components allowed there is no upper area limit; a large component is still one detection at its centroid, without deblending it into rods.</p>
<p>Saturation is always checked <b>before background subtraction</b>, using the raw-peak companion over all 150 frames. Any pixel ≥4094 in either the 18×18 aperture or the full component bounding rectangle rejects the candidate. Checking the full rectangle also covers oversized components extending beyond the usual aperture. There is no saturation toggle. Undefined-r candidates remain excluded as previously; no finite-r or theta limit is applied.</p>
<p>When enabled, the previous strict overlap rule keeps the strongest component DoG peak first and suppresses a weaker candidate if its 18×18 aperture shares any pixel with an already retained one. Selection is recomputed after changing the mask or area limit. This is the earlier strict rule, not a new deblending algorithm. No survival, low-background or left-half filter is applied.</p></details>
<p><a href="percentile_spots.html">Added/removed comparison</a> · <a href="percentile_comparison.html">All background percentiles</a> · <a href="percentile95_region_overview.png">Region comparison image</a> · <a href="percentile95_region.json">Coordinates and selection records</a></p>
</main><script>
const DATA=__DATA__,$=id=>document.getElementById(id),images={},keys=['minimum','median','p95'];let point={x:2060,y:500},cursor=-1;
function methods(){return DATA.modes[$('mode').value].methods}
function policy(){return `m${+$('mask').checked}_a${+$('large').checked}_o${+$('strict').checked}`}
function choice(key='p95'){return methods()[key].policies[policy()]}
function kept(){const ids=new Set(choice().kept);return methods().p95.spots.filter(s=>ids.has(s.id))}
function load(key){const mode=$('mode').value,id=mode+'_'+key;if(images[id])return images[id];const im=new Image();images[id]=im;im.onload=draw;im.onerror=()=>{$('error').textContent='Could not load '+methods()[key].image};im.src=methods()[key].image;return im}
function overlay(ctx,ox=0,oy=0,scale=1,box=true){ctx.save();ctx.translate(ox,oy);ctx.scale(scale,scale);ctx.lineWidth=1.5/scale;if($('marks').checked)for(const s of kept()){ctx.strokeStyle=s.geometric_inside?'#ffb331':'#65e2e6';ctx.beginPath();ctx.arc(s.x,s.y,8,0,Math.PI*2);ctx.stroke();if(s.area>250){ctx.strokeStyle='#ec79dc';ctx.strokeRect(...s.bbox)}}if($('outline').checked){const e=DATA.ellipse;ctx.strokeStyle='#ff6262';ctx.setLineDash([5/scale,5/scale]);ctx.beginPath();ctx.ellipse(e.cx,e.cy,e.rx,e.ry,0,0,Math.PI*2);ctx.stroke();ctx.setLineDash([])}if(box){ctx.strokeStyle='#fff';ctx.strokeRect(point.x-44,point.y-44,88,88)}ctx.restore()}
function draw(){const im=load($('layer').value),ctx=$('map').getContext('2d');ctx.clearRect(0,0,2464,2056);if(im.complete&&im.naturalWidth)ctx.drawImage(im,0,0);overlay(ctx);for(const key of keys){const c=$('crop_'+key).getContext('2d'),source=load(key),l=point.x-44,t=point.y-44;c.clearRect(0,0,352,352);c.imageSmoothingEnabled=false;if(source.complete&&source.naturalWidth)c.drawImage(source,l,t,88,88,0,0,352,352);overlay(c,-l*4,-t*4,4,false)}inspect()}
function list(){return kept().filter(s=>$('browse').value==='inside'?s.geometric_inside:$('browse').value==='oversized'?s.area>250:true).sort((a,b)=>a.y-b.y||a.x-b.x)}
function setPoint(x,y){point={x:Math.max(44,Math.min(2420,x)),y:Math.max(44,Math.min(2012,y))}}
function step(delta){const a=list();if(!a.length){$('position').textContent='No retained detections in this category';return}cursor=cursor<0?(delta>0?0:a.length-1):(cursor+delta+a.length)%a.length;const s=a[cursor];setPoint(s.x,s.y);$('position').textContent=`${cursor+1} / ${a.length}: ${s.id}`;draw()}
function inspect(){const s=methods().p95.spots.slice().sort((a,b)=>Math.hypot(a.x-point.x,a.y-point.y)-Math.hypot(b.x-point.x,b.y-point.y))[0];let text=`Crop centre x=${point.x.toFixed(1)}, y=${point.y.toFixed(1)}. `;if(s&&Math.hypot(s.x-point.x,s.y-point.y)<=20){text+=`${s.id}: ${choice().selection[s.id]}; area ${s.area} pixels; component box ${s.bbox[2]}×${s.bbox[3]}; DoG peak ${s.dog_peak.toFixed(2)}; raw peak ${s.raw_peak}/4094. `;const by=choice().suppressed_by[s.id];if(by)text+=`Suppressed by ${by}.`}else text+='No p95 component (area ≥10) within 20 pixels.';$('detail').textContent=text}
function refresh(){cursor=-1;$('position').textContent='';const p=choice();$('counts').textContent=`95th percentile: ${p.kept.length} retained, including ${p.inside.length} inside the former mask and ${p.oversized.length} oversized. ${p.counts.saturated} rejected for raw saturation.`;
$('table').innerHTML='<table><tr><th>Background</th><th>2-SD cutoff</th><th>Total retained</th><th>Inside former mask</th><th>Oversized retained</th><th>Raw saturated rejected</th></tr>'+keys.map(key=>{const m=methods()[key],p=choice(key);return `<tr><td>${m.label}</td><td>${m.threshold.toFixed(3)}</td><td>${p.kept.length}</td><td>${p.inside.length}</td><td>${p.oversized.length}</td><td>${p.counts.saturated}</td></tr>`}).join('')+'</table>';draw()}
for(const id of ['mode','mask','large','strict'])$(id).onchange=refresh;for(const id of ['layer','marks','outline'])$(id).onchange=draw;
$('browse').onchange=()=>{cursor=-1;$('position').textContent=''};$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('feature').onclick=()=>{setPoint(DATA.ellipse.cx,DATA.ellipse.cy);cursor=-1;$('position').textContent='';draw()};
$('zoom').onchange=()=>{$('map').style.width=$('zoom').value==='native'?'2464px':'100%'};
$('map').onclick=e=>{const r=$('map').getBoundingClientRect();setPoint((e.clientX-r.left)*2464/r.width,(e.clientY-r.top)*2056/r.height);cursor=-1;$('position').textContent='';draw()};refresh();
</script></html>'''

if __name__ == '__main__':
    build()
