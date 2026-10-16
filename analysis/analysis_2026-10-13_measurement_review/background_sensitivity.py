"""Controlled extra-background subtraction for the 14 gold survey stacks.

Source arrays/metadata are read-only. Clip each measurement frame BEFORE temporal
averaging; exclude the phase marker. The GUI intensity DoG implementation is used
unchanged, with both native adaptive k=8 and a fixed 1x absolute threshold.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import time
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from polarcam.live import Detection_alg_offline as detector

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DAY = REPO/'data/hinges/polarcam/2026-10-12'
OUT = HERE/'background_sensitivity'
FACTORS = (1., 1.2, 1.5, 2.)
MATCH_RADIUS = 6.0


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(image):
    return np.rint(np.clip(image*20/4095, 0, 1)*255).astype(np.uint8)


def dog_stats(image):
    edge = detector.EDGE_EXCLUDE_PX
    work = image[edge:-edge, edge:-edge]
    dog = (cv2.GaussianBlur(work, (0,0), sigmaX=detector.DOG_SIGMA_SMALL, borderType=cv2.BORDER_REPLICATE)
           - cv2.GaussianBlur(work, (0,0), sigmaX=detector.DOG_SIGMA_LARGE, borderType=cv2.BORDER_REPLICATE))
    mean, sd = detector._finite_stats(dog)
    return work, mean, sd


def find(work, k):
    points = detector.find_spot_centers_dog(work, k_std=k)
    return np.asarray(points, dtype=float).reshape(-1, 2)+detector.EDGE_EXCLUDE_PX


def compare(base, points):
    labels = np.zeros(len(points), dtype=int)
    matched = np.zeros(len(base), dtype=bool)
    if len(base) and len(points):
        distances = cdist(base, points)
        # Maximise permitted matches, then minimise displacement. Dummy columns
        # allow disappearance; this is an image-coordinate match, not rod identity.
        costs = np.concatenate([np.where(distances <= MATCH_RADIUS, distances, 1e8),
                                np.full((len(base), len(base)), 1e5)], axis=1)
        rows, cols = linear_sum_assignment(costs)
        for row, col in zip(rows, cols):
            if col < len(points) and distances[row, col] <= MATCH_RADIUS:
                labels[col] = row+1
                matched[row] = True
    return {'points': points.tolist(), 'baseline_ids': labels.tolist(),
            'lost_ids': (np.flatnonzero(~matched)+1).tolist(), 'count': len(points),
            'retained': int(matched.sum()), 'lost': int((~matched).sum()), 'new': int((labels == 0).sum())}


def calculate():
    OUT.mkdir(exist_ok=True); (OUT/'images').mkdir(exist_ok=True)
    bgpath = DAY/'backgrounds/background_min_20261012-192157_exp1.2ms_gain1.npy'
    bghash = digest(bgpath)
    assert digest(DAY/'backgrounds/used'/f'{bghash}.npy') == bghash
    background = np.load(bgpath, allow_pickle=False).astype(np.float32)
    assert np.isfinite(background).all() and (background >= 0).all()
    assert np.array_equal(background, np.rint(background))
    inputs = [p for p in sorted((DAY/'frames').glob('frame_stack_*.npy'))
              if 'frame_stack_20261012-192231' <= p.stem <= 'frame_stack_20261012-193127']
    assert len(inputs) == 14
    records, csv_rows = [], []
    additions = [(factor-1)*background for factor in FACTORS[1:]]
    timer = time.perf_counter()
    for index, path in enumerate(inputs, 1):
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        assert meta['background']['background_subtracted'] is True
        assert meta['background']['background_profile_sha256'] == bghash
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        assert a.dtype == np.uint16 and a.shape[1:] == background.shape
        n = a.shape[0]-int(meta['actual']['phase_marker_appended'])
        assert n == meta['actual']['frames'] == 150
        sums = np.zeros((len(FACTORS), *background.shape), dtype=np.float64)
        clipped = np.zeros(len(FACTORS), dtype=np.int64)
        work = np.empty_like(background)
        for frame in a[:n]:
            sums[0] += frame
            clipped[0] += np.count_nonzero(frame == 0)
            for j, extra in enumerate(additions, 1):
                np.subtract(frame, extra, out=work)
                np.maximum(work, 0, out=work)
                sums[j] += work
                clipped[j] += np.count_nonzero(work == 0)
        means = (sums/n).astype(np.float32)
        assert np.all(np.diff(means, axis=0) <= 1e-5)
        # Native GUI uses float64 temporal means before its polarization-cell map.
        maps = [detector.intensity_detection_map(sums[j]/n) for j in range(len(FACTORS))]
        del sums, a
        stamp = path.stem[-6:]
        result = {'name': path.name, 'time': ':'.join(stamp[i:i+2] for i in (0,2,4)),
                  'stamp': stamp, 'url': path.as_uri(), 'shape': list(background.shape), 'frames': n,
                  'note': meta['conditions'].get('condition_note',''), 'factors': [],
                  'sidecar_sha256': digest(path.with_suffix('.json')), 'source_bytes': path.stat().st_size,
                  'source_mtime_ns': path.stat().st_mtime_ns, 'source_max_raw': meta['actual'].get('max_raw_value')}
        base = None; base_threshold = None
        for j, (factor, image) in enumerate(zip(FACTORS, maps)):
            work_map, mu, sd = dog_stats(image)
            threshold = max(0., mu+detector.DOG_K_STD*sd)
            native = find(work_map, detector.DOG_K_STD)
            if base is None:
                base, base_threshold = native, threshold
            fixed_k = (base_threshold-mu)/sd if sd > 0 else detector.DOG_K_STD
            fixed = find(work_map, fixed_k)
            if j == 0:
                np.testing.assert_allclose(native, fixed, rtol=0, atol=0)
            image_path = OUT/'images'/f'{stamp}_{factor:g}x.png'
            Image.fromarray(display(image)).save(image_path, compress_level=3)
            variants = {'native': compare(base, native), 'fixed': compare(base, fixed)}
            for mode, variant in variants.items():
                variant['threshold'] = threshold if mode == 'native' else base_threshold
                for (x,y), baseline_id in zip(variant['points'], variant['baseline_ids']):
                    csv_rows.append([path.name, factor, mode, x, y, baseline_id or '', variant['threshold']])
            result['factors'].append({'factor': factor, 'image': image_path.relative_to(OUT).as_posix(),
                                      'zero_fraction': float(clipped[j]/(n*background.size)),
                                      'mean_signal': float(means[j].mean()), 'dog_mean': mu, 'dog_sd': sd,
                                      **variants})
        records.append(result)
        print(f'{index}/14 {stamp}: native {[v["native"]["count"] for v in result["factors"]]}; '
              f'fixed {[v["fixed"]["count"] for v in result["factors"]]}', flush=True)
    detector_path = Path(detector.__file__)
    report = {'records': records, 'factors': list(FACTORS), 'background': bgpath.name, 'background_sha256': bghash,
              'background_url': bgpath.as_uri(), 'background_image': '../previews/325f9a59b940d8_20x.webp',
              'formula': 'S_alpha[t] = max(S_saved[t] - (alpha-1)*B, 0); then mean over 150 frames',
              'detector_source': str(detector_path.relative_to(REPO)), 'detector_sha256': digest(detector_path),
              'git_revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,encoding='utf-8').strip(),
              'parameters': {'k': detector.DOG_K_STD, 'sigma_small': detector.DOG_SIGMA_SMALL,
                             'sigma_large': detector.DOG_SIGMA_LARGE, 'area_min': detector.DOG_MIN_AREA,
                             'area_max': detector.DOG_MAX_AREA, 'edge': detector.EDGE_EXCLUDE_PX,
                             'match_radius_px': MATCH_RADIUS, 'display_gain':20},
              'elapsed_seconds': time.perf_counter()-timer,
              'selection': '14 actual saved stacks; note numbering skips 13. No local-contrast, saturation, angle or persistence exclusion.'}
    with (OUT/'candidates.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['stack','total_bg_factor','threshold_mode','x_sensor_px','y_sensor_px','matched_1x_id','threshold']);writer.writerows(csv_rows)
    (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def render(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    records = report['records']
    totals = {}
    for mode in ('native','fixed'):
        totals[mode] = [{key: sum(r['factors'][i][mode][key] for r in records)
                        for key in ('count','retained','lost','new')} for i in range(4)]
    report['totals'] = totals
    (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':10, 'figure.facecolor':'white'})
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,mode,title in zip(axes,('native','fixed'),('GUI threshold: mean + 8 SD','Original 1× threshold held fixed')):
        for r in records:
            ax.plot(FACTORS,[v[mode]['count'] for v in r['factors']],color='#507f8c',alpha=.5,lw=1,marker='.',ms=5)
        ax.set(xlabel='Total background multiplier',ylabel='Detected candidates per stack',title=title,xticks=FACTORS)
        ax.grid(alpha=.2)
    fig.suptitle('14 gold survey stacks · identical detector parameters · one line per stack')
    fig.savefig(OUT/'counts.png',dpi=170);plt.close(fig)
    # The stack with the most baseline candidates lost at fixed 2x is a disclosed
    # illustrative example; the viewer/table expose every stack and candidate.
    chosen = max(records,key=lambda r:r['factors'][-1]['fixed']['lost'])
    baseline = chosen['factors'][0]['fixed']['points']
    fig,axes = plt.subplots(1,4,figsize=(16,4.6),layout='constrained')
    for ax,v in zip(axes,chosen['factors']):
        image = np.asarray(Image.open(OUT/v['image']))
        ax.imshow(image,cmap='gray',vmin=0,vmax=255)
        info=v['fixed']
        for (x,y),bid in zip(info['points'],info['baseline_ids']):
            ax.add_patch(Circle((x,y),18,fill=False,lw=.7,color='#00dfec' if bid else '#ffc447'))
        for bid in info['lost_ids']:
            ax.add_patch(Circle(baseline[bid-1],18,fill=False,lw=.7,ls='--',color='#ff535d'))
        ax.set_title(f'{v["factor"]:g}× BG · {info["count"]} candidates\n{info["lost"]} baseline candidates lost')
        ax.set(xticks=[],yticks=[])
    fig.suptitle(f'{chosen["time"]} · fixed 1× threshold · 150-frame mean · 20× display\n'
                 'Illustrative stack with most losses at 2×; cyan retained, red dashed lost, yellow new')
    fig.savefig(OUT/'example.png',dpi=160);plt.close(fig)
    html = HTML.replace('__DATA__',json.dumps(report,ensure_ascii=False).replace('</','<\\/'))
    (OUT/'index.html').write_text(html,encoding='utf-8')
    print('TOTALS',json.dumps(totals));print('Example:',chosen['time']);print(OUT/'index.html')


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold survey — background subtraction sensitivity</title>
<style>
:root{font:15px/1.5 system-ui,sans-serif;color:#24383c;background:#f5f6f3}*{box-sizing:border-box}body{margin:0}main{max-width:1600px;padding:24px;margin:auto}h1{font-size:31px;margin:7px 0}h2{font-size:21px}p{max-width:1050px}.muted{color:#657775}a{color:#126978}button,select{font:inherit;border:1px solid #abbdb5;background:#fff;color:inherit;padding:7px 10px;border-radius:5px}button{cursor:pointer}button:hover{background:#e2ede7}label{display:flex;align-items:center;gap:6px}.box{border:1px solid #d0dcd5;background:white;padding:16px 20px;border-radius:7px;margin:18px 0}.controls{position:sticky;top:0;z-index:5;background:#f5f6f3f5;border-block:1px solid #c7d6cd;padding:12px 0;display:flex;gap:10px;flex-wrap:wrap;align-items:center}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:14px 0}.pane{background:white;border:1px solid #d1dcd7;padding:10px;border-radius:6px}.pane h3{font-size:18px;margin:0}.count{font-size:13px;min-height:60px}.overview{display:block;width:100%;aspect-ratio:2464/2056;background:#111;cursor:crosshair}.crop{display:block;width:100%;aspect-ratio:1;background:#111;image-rendering:pixelated}.legend{font-size:13px}.cyan{color:#008c98}.red{color:#c03243}.yellow{color:#a36900}.note{white-space:pre-wrap;border-left:3px solid #a0b9a8;padding:8px 12px;background:#edf2ec}.flex{display:flex;gap:12px;align-items:center;flex-wrap:wrap}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:8px;border-bottom:1px solid #d9e2dd}tr.active{background:#e2eee6}td button{font-size:13px;padding:3px 8px}.plot{max-width:100%;height:auto}.tablebox{overflow:auto}details summary{cursor:pointer}.small{font-size:13px}.bgcrop{width:176px;height:176px;image-rendering:pixelated;background:#111}.foot{margin-top:30px;font-size:13px}@media(max-width:950px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.controls{position:static}}@media(max-width:550px){main{padding:12px}.grid{gap:5px}.pane{padding:5px}.count{min-height:80px}}
</style>
<main><div class="muted">12 OCTOBER 2026 · GOLD SURVEY · READ-ONLY COMPARISON</div><h1>Do candidates survive stronger background subtraction?</h1>
<p>All <b>14 saved stacks from 19:22:31–19:31:27</b>, using the same verified 19:21:57 background. The note numbering skips 13. Compare total subtraction of <b>1×, 1.2×, 1.5× and 2×</b>; the original recordings remain unchanged.</p>
<div class="box"><b>What is being changed</b><p>The saved frames already have 1× background removed. For each measurement frame, this trial computes <code>max(saved − (multiplier−1) × background, 0)</code>, then averages all 150 frames. It does <b>not</b> clip an already-averaged frame. The synthetic phase marker is excluded.</p><p><b>Detector:</b> the current GUI's <b>Intensity</b> mode: mean four-polarizer-cell intensity → Difference of Gaussians (σ=1 and 6 sensor pixels) → threshold → components of 10–250 pixels; 10-pixel border excluded. Default threshold is mean + 8 SD. No local-contrast, angle, saturation or cross-stack position exclusions are added in this candidate comparison.</p><p class="small">The GUI recalculates the threshold after subtraction, which can introduce new detections as the threshold falls. Switch to <b>Hold original 1× threshold</b> to keep the absolute threshold fixed within each stack. Disappearance is sensitivity to this processing, not proof that a spot was background; genuine dim rods can also disappear.</p></div>
<div class="controls"><button id="prev">← Previous stack</button><select id="stack" aria-label="Frame stack"></select><button id="next">Next stack →</button><label>Threshold <select id="mode"><option value="native">GUI: mean + 8 SD</option><option value="fixed">Hold original 1× threshold</option></select></label><label><input type="checkbox" id="rings" checked>Overlays</label><label><input type="checkbox" id="reference">Show 1× image in every panel</label></div>
<div id="note" class="note"></div><p class="legend"><span class="cyan">● Cyan: matched to a 1× candidate</span> · <span class="red">◌ Red dashed: 1× candidate no longer detected</span> · <span class="yellow">● Yellow: no 1× match</span>. One-to-one matching within 6 sensor pixels. Circles mark centres, not extraction windows.</p>
<div class="grid" id="overviews"></div>
<h2>Inspect the same location at all four settings</h2><p>Click any overview above, or step through the original candidates. Crops show <b>88 × 88 sensor pixels</b>; all images use a fixed <b>20× display gain</b> (white = 204.75 saved units). Detections use the numeric mean before display gain.</p>
<div class="flex"><button id="spotPrev">← Previous candidate</button><select id="spot" aria-label="Baseline candidate"></select><button id="spotNext">Next candidate →</button><span id="coord"></span></div>
<div class="grid" id="crops"></div><details class="box"><summary>Show the matching background patch</summary><p class="small">This is the original background profile at the same coordinates, at 20× display.</p><canvas class="bgcrop" id="bgcrop" width="352" height="352"></canvas></details>
<div class="box"><h2>Counts across all 14 stacks</h2><p id="totals"></p><p class="small">Counts below follow the selected threshold mode. They are candidate counts, not confirmed rods or independent sample identities. “Retained” requires a positional match to a 1× detection; component splits/merges or centroid movement can affect matching.</p><div class="tablebox"><table><thead><tr><th>Stack</th><th>1×</th><th>1.2×</th><th>1.5×</th><th>2×</th><th>1× retained at 2×</th><th>Lost / new at 2×</th></tr></thead><tbody id="table"></tbody></table></div></div>
<details class="box"><summary>Plot counts and read reproducibility details</summary><img class="plot" src="counts.png" alt="Candidate count across subtraction factors for each stack"><p class="small">Background: <a id="bgfile"></a><br>SHA-256 <code id="bghash"></code><br>Camera detector source SHA-256 <code id="dethash"></code></p><p class="small">Only this gold batch is analysed here. Matching never combines candidates across different frame stacks. Extra subtraction does not recover saturated raw pixels. The background contains nonnegative integer values, so further subtraction with clipping is compatible with the saved once-subtracted frames; fractional results are retained as floating-point values.</p></details>
<p class="foot"><a href="../index.html">Complete recording gallery</a> · <a href="results.json">Results and parameters (JSON)</a> · <a href="candidates.csv">Candidate coordinates (CSV)</a> · <a href="example.png">Static example</a></p>
</main><script>
const DATA=__DATA__;
const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let stackIndex=0,cx=1232,cy=1028,images=[],bgImage=null,token=0;
const records=DATA.records;
for(const [i,r] of records.entries())$('stack').add(new Option(`${i+1}/14 · ${r.time}`,i));
$('overviews').innerHTML=DATA.factors.map((v,i)=>`<section class="pane"><h3>${v}× background</h3><div class="count" id="count${i}"></div><canvas class="overview" id="overview${i}" width="616" height="514" data-index="${i}"></canvas></section>`).join('');
$('crops').innerHTML=DATA.factors.map((v,i)=>`<section class="pane"><h3>${v}× background</h3><canvas class="crop" id="crop${i}" width="352" height="352"></canvas></section>`).join('');
function loadImage(path){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('Image unavailable: '+path));im.src=path})}
loadImage(DATA.background_image).then(im=>{bgImage=im;draw()});
function marks(ctx,info,base,transform,overview){if(!$('rings').checked)return;ctx.lineWidth=overview?1.1:1.4;const radius=overview?4:24;function circle(p,color,dashed,label){const xy=transform(p);ctx.strokeStyle=color;ctx.setLineDash(dashed?[4,3]:[]);ctx.beginPath();ctx.arc(xy[0],xy[1],radius,0,Math.PI*2);ctx.stroke();if(!overview&&xy[0]>=0&&xy[0]<352&&xy[1]>=0&&xy[1]<352){ctx.fillStyle=color;ctx.font='12px system-ui';ctx.fillText(label,xy[0]+radius+2,xy[1]-3)}}info.points.forEach((p,i)=>circle(p,info.baseline_ids[i]?'#00e6f0':'#ffc24b',false,info.baseline_ids[i]||'new'));info.lost_ids.forEach(id=>circle(base[id-1],'#ff4c66',true,id));ctx.setLineDash([])}
function draw(){if(images.length!==4)return;const r=records[stackIndex],mode=$('mode').value,base=r.factors[0][mode].points,w=r.shape[1],h=r.shape[0];cx=Math.max(44,Math.min(w-44,cx));cy=Math.max(44,Math.min(h-44,cy));for(let i=0;i<4;i++){const v=r.factors[i],info=v[mode],im=images[$('reference').checked?0:i];const c=$('overview'+i),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(im,0,0,c.width,c.height);marks(ctx,info,base,p=>[p[0]/w*c.width,p[1]/h*c.height],true);ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.strokeRect((cx-44)/w*c.width,(cy-44)/h*c.height,88/w*c.width,88/h*c.height);const crop=$('crop'+i),cr=crop.getContext('2d');cr.imageSmoothingEnabled=false;cr.clearRect(0,0,352,352);cr.drawImage(im,cx-44,cy-44,88,88,0,0,352,352);marks(cr,info,base,p=>[(p[0]-cx+44)*4,(p[1]-cy+44)*4],false);$('count'+i).innerHTML=`<b>${info.count} candidates</b> · ${info.retained} retained<br>${info.lost} lost · ${info.new} new · threshold ${info.threshold.toFixed(2)}<br><span class="muted">${(100*v.zero_fraction).toFixed(1)}% of frame-pixels zero</span>`}if(bgImage){const c=$('bgcrop').getContext('2d');c.imageSmoothingEnabled=false;c.drawImage(bgImage,cx-44,cy-44,88,88,0,0,352,352)}$('coord').textContent=`Centre x=${cx.toFixed(1)}, y=${cy.toFixed(1)} sensor pixels${$('reference').checked?' · images held at 1×; detections still use each processed image':''}`;}
function table(){const mode=$('mode').value,t=DATA.totals[mode];$('totals').innerHTML=`Sum of candidates: <b>${t.map(v=>v.count).join(' → ')}</b> at 1× → 1.2× → 1.5× → 2×. At 2×, <b>${t[3].retained}/${t[0].count}</b> original candidates have a positional match; <b>${t[3].lost}</b> no longer match and <b>${t[3].new}</b> are newly detected.`;$('table').innerHTML=records.map((r,i)=>`<tr class="${i===stackIndex?'active':''}"><td><button data-stack="${i}">${r.time}</button></td>${r.factors.map(v=>`<td>${v[mode].count}</td>`).join('')}<td>${r.factors[3][mode].retained}/${r.factors[0][mode].count}</td><td>${r.factors[3][mode].lost} / ${r.factors[3][mode].new}</td></tr>`).join('')}
async function choose(index){stackIndex=Math.max(0,Math.min(records.length-1,Number(index)));const current=++token,r=records[stackIndex];$('stack').value=stackIndex;$('prev').disabled=stackIndex===0;$('next').disabled=stackIndex===records.length-1;$('note').innerHTML=`<b>${esc(r.name)}</b> · ${r.frames} frames · <a href="${esc(r.url)}">Source</a><br>Recorded note: ${esc(r.note||'None')}`;images=[];$('spot').innerHTML='<option value="-1">Click image to choose a location</option>';const base=r.factors[0].native.points;for(let i=0;i<base.length;i++)$('spot').add(new Option(`Candidate ${i+1} · x=${base[i][0].toFixed(0)}, y=${base[i][1].toFixed(0)}`,i));$('spotPrev').disabled=!base.length;$('spotNext').disabled=!base.length;table();try{const loaded=await Promise.all(r.factors.map(v=>loadImage(v.image)));if(current!==token)return;images=loaded;if(base.length){[cx,cy]=base[0];$('spot').value='0'}draw()}catch(e){$('note').textContent=e.message}}
function spot(index){const points=records[stackIndex].factors[0].native.points;if(!points.length)return;index=(index+points.length)%points.length;[cx,cy]=points[index];$('spot').value=index;draw()}
$('stack').onchange=()=>choose($('stack').value);$('prev').onclick=()=>choose(stackIndex-1);$('next').onclick=()=>choose(stackIndex+1);$('mode').onchange=()=>{table();draw()};$('reference').onchange=draw;$('rings').onchange=draw;$('spot').onchange=()=>{if(Number($('spot').value)>=0)spot(Number($('spot').value))};$('spotPrev').onclick=()=>spot(Math.max(0,Number($('spot').value))-1);$('spotNext').onclick=()=>spot(Math.max(-1,Number($('spot').value))+1);$('table').onclick=e=>{if(e.target.dataset.stack!==undefined)choose(e.target.dataset.stack)};
for(let i=0;i<4;i++)$('overview'+i).onclick=e=>{const rect=e.currentTarget.getBoundingClientRect(),r=records[stackIndex];cx=(e.clientX-rect.left)/rect.width*r.shape[1];cy=(e.clientY-rect.top)/rect.height*r.shape[0];$('spot').value='-1';draw()};
$('bgfile').textContent=DATA.background;$('bgfile').href=DATA.background_url;$('bghash').textContent=DATA.background_sha256;$('dethash').textContent=DATA.detector_sha256;choose(0);
</script></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--render-only',action='store_true')
    args = parser.parse_args()
    report = json.loads((OUT/'results.json').read_text(encoding='utf-8')) if args.render_only else calculate()
    render(report)
