"""Trial a fixed background-feature mask without changing upstream selections.

Uses cached 1x radii and pre-subtraction saturation flags. No acquisitions are
modified. The ellipse was chosen from the marked background image, before
examining its effect on the radius distributions; it is shared by all groups.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / 'background_region_trial'
SOURCE = HERE / 'radius_detector_comparison/results.json'
MAP_SOURCE = HERE / 'survival_background_map/data.json'
ELLIPSE = dict(cx=2060, cy=500, rx=320, ry=220)
POLICIES = {
    'all': 'All unsaturated 1x spots',
    'targeted': 'Remove failures inside ellipse',
    'region': 'Remove every spot inside ellipse',
    'survival': 'Previous: require 2x survival everywhere',
}
COLORS = {'Hinges': '#087e8b', 'Gold': '#c07a11'}
LABELS = {'Hinges': 'DNA / hinges', 'Gold': 'Gold control'}


def inside(x, y):
    return ((x-ELLIPSE['cx'])/ELLIPSE['rx'])**2 + ((y-ELLIPSE['cy'])/ELLIPSE['ry'])**2 <= 1


def keep(s, policy):
    return {'all': True, 'targeted': s['matched'] or not s['inside'],
            'region': not s['inside'], 'survival': s['matched']}[policy]


def describe(rows, edges):
    a = np.array([s['r'] for s in rows], dtype=float)
    counts, _ = np.histogram(a, edges)
    assert int(counts.sum()) == len(a)
    return dict(n=len(a), median=float(np.median(a)) if len(a) else None,
                counts=counts.tolist(), percent=(counts/max(1, len(a))*100).tolist())


def build():
    OUT.mkdir(exist_ok=True)
    raw, map_raw = SOURCE.read_bytes(), MAP_SOURCE.read_bytes()
    source, oldmap = json.loads(raw), json.loads(map_raw)
    assert source['complete'] and oldmap['source_sha256'] == hashlib.sha256(raw).hexdigest()
    bg_by_hash = {g['sha256']: g['id'] for g in oldmap['groups']}
    map_records = {r['name']: r for r in oldmap['records']}
    records = []
    for r in source['records']:
        mr = map_records[r['name']]
        item = dict(name=r['name'], group=r['group'], index=r['group_index'], time=r['time'],
                    note=r['note'], background_group=bg_by_hash[r['background_sha256']],
                    images=mr['images'], methods={})
        for method, entry in r['methods'].items():
            points = []
            for s in entry['spots']:
                if s['ceiling_veto'] or s['r'] is None or not np.isfinite(s['r']):
                    continue
                points.append(dict(id=s['id'], key=f'{"H" if r["group"]=="Hinges" else "G"}{r["group_index"]:02d}-S{s["id"]:04d}',
                                   record=r['name'], x=s['x'], y=s['y'], bounds=s['bounds'],
                                   r=s['r'], raw_peak=s['raw_peak'], matched=s['matched'],
                                   inside=bool(inside(s['x'], s['y']))))
            item['methods'][method] = points
        records.append(item)
    max_r = max(s['r'] for r in records for ss in r['methods'].values() for s in ss)
    edges = np.arange(0, max(1.02, np.ceil(max_r/.02)*.02+.02)+.001, .02)
    groups = [dict(g, image='../survival_background_map/'+g['image']) for g in oldmap['groups']]
    result = dict(source_sha256=hashlib.sha256(raw).hexdigest(),
                  map_source_sha256=hashlib.sha256(map_raw).hexdigest(),
                  builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  ellipse=ELLIPSE, methods=source['methods'], policies=POLICIES,
                  r_max=source['r_max'], bins=edges.tolist(), records=records,
                  groups=groups, summary={}, background_summary={},
                  selection='Unsaturated 1x detections; retain every finite radius. No theta or radius cut.',
                  saturation='Any pixel >=4094 in the 18x18 aperture over 150 frames vetoes the spot; cached SHA-verified pre-subtraction raw peak.',
                  region_rule='Candidate centre inside ((x-2060)/320)^2+((y-500)/220)^2 <= 1; x right, y down, in sensor pixels.',
                  targeted_rule='Exclude only candidates inside the ellipse that have no same-detector 2x match within 6 px. Restore survival failures elsewhere.')
    for method in source['methods']:
        summary = result['summary'][method] = {}
        for group in ('Hinges', 'Gold'):
            rows = [s for r in records if r['group']==group for s in r['methods'][method]]
            summary[group] = {p: describe([s for s in rows if keep(s,p)], edges) for p in POLICIES}
        by_bg = result['background_summary'][method] = {}
        for g in groups:
            rows = [s for r in records if r['background_group']==g['id'] for s in r['methods'][method]]
            by_bg[str(g['id'])] = {p: describe([s for s in rows if keep(s,p)], edges) for p in POLICIES}
    (OUT/'results.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    figures(result)
    page = HTML.replace('__DATA__', json.dumps(result, separators=(',', ':'), allow_nan=False).replace('</', '<\\/'))
    (OUT/'index.html').write_text(page, encoding='utf-8')
    # All source selections remain byte-for-byte unchanged.
    assert SOURCE.read_bytes() == raw and MAP_SOURCE.read_bytes() == map_raw
    print(json.dumps(result['summary'], ensure_ascii=True))


def style_axis(ax, rmax, xmax):
    ax.axvline(rmax, color='#52616b', ls=':', lw=1.2, label=f'Model limit {rmax:.3f}')
    ax.set_xlim(0, xmax)
    ax.set_xlabel('Anisotropy radius r at 1x background subtraction')
    ax.set_ylabel('Spots per 0.02 bin (%)')
    ax.grid(axis='y', alpha=.16)
    ax.spines[['top', 'right']].set_visible(False)


def figures(d):
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'font.family': 'DejaVu Sans'})
    edges = np.array(d['bins'])
    xmax = max(1., edges[-1])
    for method, meta in d['methods'].items():
        fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), sharey=True)
        ymax = max(max(d['summary'][method][g][p]['percent'])
                   for g in ('Hinges', 'Gold') for p in ('all', 'targeted', 'region')) * 1.42
        for ax, policy, title in zip(axes, ('all','targeted','region'),
                                    ('Before: no survival filter', 'Trial: remove failures in ellipse', 'Sensitivity: remove entire ellipse')):
            for group in ('Hinges', 'Gold'):
                q = d['summary'][method][group][policy]
                ax.stairs(q['percent'], edges, color=COLORS[group], lw=1.6,
                          label=f'{LABELS[group]}: n={q["n"]:,}')
            style_axis(ax, d['r_max'], xmax)
            ax.set_ylim(0, ymax)
            ax.set_title(title)
            ax.legend(fontsize=8.5, loc='upper left', frameon=False)
        fig.suptitle(meta['label']+' | DNA 16 stacks and gold 14 stacks', fontsize=14, x=.055, ha='left')
        fig.text(.055,.015,'Same 1x values and 18x18 apertures in all panels; raw saturation excluded; all finite radii retained. Each group normalised separately.',fontsize=9,color='#46545b')
        fig.tight_layout(rect=(0,.04,1,.94))
        fig.savefig(OUT/f'{method}.png', dpi=150)
        fig.savefig(OUT/f'{method}.svg')
        plt.close(fig)
        fig, axes = plt.subplots(1,3,figsize=(15.2,4.6),sharey=True)
        ymax = max(max(v[p]['percent']) for v in d['background_summary'][method].values()
                   for p in ('all', 'targeted')) * 1.35
        for ax, g in zip(axes, d['groups']):
            q = d['background_summary'][method][str(g['id'])]
            group='Hinges' if g['id']<2 else 'Gold'
            ax.stairs(q['all']['percent'],edges,color='#7a838a',lw=1.2,label=f'Before: n={q["all"]["n"]:,}')
            ax.stairs(q['targeted']['percent'],edges,color=COLORS[group],lw=1.8,label=f'Trial: n={q["targeted"]["n"]:,}')
            style_axis(ax,d['r_max'],xmax)
            ax.set_ylim(0,ymax)
            ax.set_title(g['label'])
            ax.legend(frameon=False, fontsize=8.5)
        fig.suptitle(meta['label']+' | separate background groups',fontsize=14,x=.055,ha='left')
        fig.text(.055,.015,'The same fixed ellipse is used for all three backgrounds. Only survival failures inside it are removed in the trial.',fontsize=9,color='#46545b')
        fig.tight_layout(rect=(0,.04,1,.94))
        fig.savefig(OUT/f'{method}_groups.png',dpi=150)
        plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(15,5.2))
    for ax,g in zip(axes,d['groups']):
        im=Image.open(OUT/g['image'])
        ax.imshow(im,extent=(0,g['width'],g['height'],0),cmap='gray')
        rows=[s for r in d['records'] if r['background_group']==g['id'] for s in r['methods']['native'] if not s['matched']]
        for flag,color in ((False,'#62dbed'),(True,'#ff687e')):
            pts=[s for s in rows if s['inside']==flag]
            ax.scatter([s['x'] for s in pts],[s['y'] for s in pts],s=11,facecolors='none',edgecolors=color,lw=.7)
        ax.add_patch(Ellipse((ELLIPSE['cx'],ELLIPSE['cy']),ELLIPSE['rx']*2,ELLIPSE['ry']*2,
                             facecolor='none',edgecolor='#ffca55',lw=1.5))
        ax.set_title(g['label'],fontsize=10)
        ax.set_xlim(0,g['width']);ax.set_ylim(g['height'],0)
        ax.set_xlabel('Sensor x (px)');ax.set_ylabel('Sensor y (px)')
    fig.suptitle('Fixed background region | GUI survival failures: pink removed, cyan retained',fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT/'mask_overview.png',dpi=160)
    plt.close(fig)


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Background region trial</title>
<style>
*{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;background:#f7f8f5;color:#263337;margin:0;padding:28px}main{max-width:1500px;margin:auto}h1{font-size:28px;margin:0 0 8px}h2{font-size:21px;margin:24px 0 10px}p{max-width:1100px}a{color:#08798b}select,button{font:inherit;padding:6px}label{display:inline-flex;gap:6px;align-items:center;margin:5px 14px 5px 0}.box{padding:16px;background:white;border:1px solid #d9dfdb;border-radius:7px;margin:18px 0}.small{font-size:13px;color:#536065}.plots{width:100%;display:block}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #dee3e0;text-align:left;padding:7px}th{font-weight:600}.scroll{overflow:auto}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px}.controls label{white-space:nowrap}#viewport{overflow:auto;max-height:78vh;background:#111}#map{display:block;cursor:crosshair;width:100%;height:auto}.detail{display:flex;flex-wrap:wrap;gap:16px}.detail canvas{width:264px;height:264px;max-width:100%;background:#111;image-rendering:pixelated}figure{margin:0}figcaption{font-size:13px}.removed{color:#bc3349}.retained{color:#067d91}#error{color:#b32e32}.row{margin:7px 0}.active{background:#edf7f5}code{font-size:14px}
</style></head><body><main>
<h1>Remove only the large background feature</h1>
<p>The main trial removes <b>1× detections that fail the 2× survival check and fall inside the yellow ellipse</b>. Failures elsewhere return to the distribution. The small central feature is left alone. A second trial removes every detection inside the ellipse, including survivors.</p>
<p class="small">All plots use the original 1× channel intensities and 18×18 extraction. Raw saturation stays excluded, including saturation hidden by background subtraction. Every finite r is retained, even above the model limit. This is a spatial sensitivity check, not a claim that each removed spot is an artefact.</p>
<div class="controls"><label>Detector <select id="method"></select></label><a href="../survival_background_map/index.html">Original survival-failure viewer</a><a href="../radius_detector_comparison/index.html">Previous comparison</a></div>
<div class="box"><div id="headline"></div><div class="scroll"><table><thead><tr><th>Sample</th><th>Before</th><th>Removed in trial</th><th>Trial remaining</th><th>Median r: before → trial</th><th>Whole-region remaining</th></tr></thead><tbody id="summary"></tbody></table></div></div>
<img id="distribution" class="plots" alt="Radius distributions before and after regional exclusion">
<details><summary>Show DNA stacks 1–9, DNA stacks 10–16, and gold separately</summary><img id="groupPlot" class="plots" alt="Separate radius distributions for the three backgrounds"></details>
<h2>Inspect the fixed region</h2>
<p class="small">Ellipse centre (2060, 500), horizontal radius 320 px, vertical radius 220 px. A spot is inside if ((x−2060)/320)² + ((y−500)/220)² ≤ 1. Coordinates refer to the candidate centre, x right and y down. The region was drawn around the marked large feature before examining the resulting distributions; it is identical for all backgrounds and detectors.</p>
<div class="controls"><label>Background <select id="group"></select></label><label>Trial <select id="policy"><option value="targeted">Only failed survival inside ellipse</option><option value="region">All spots inside ellipse</option></select></label><label>Zoom <select id="zoom"><option value="1">Fit</option><option value="1.5">150%</option><option value="2">200%</option></select></label></div>
<div class="controls"><label><input type="checkbox" id="spots" checked>Show spots</label><label><input type="checkbox" id="survivors">Also show 2× survivors</label><label><input type="checkbox" id="mask" checked>Show ellipse</label><button id="download">Export excluded spot list</button></div>
<p id="mapCount" class="small"></p><div id="error"></div><div id="viewport"><canvas id="map" width="2464" height="2056"></canvas></div>
<p class="small"><span class="removed">Pink = removed by selected trial</span> · <span class="retained">cyan = retained survival failure</span> · grey = retained survivor (optional). Rings pool recordings sharing this background. Click a spot to inspect; repeated clicks cycle overlapping rings.</p>
<div class="box"><b id="selectedTitle">Click a ring in the map</b><p id="selectedInfo" class="small"></p><div class="detail"><figure><canvas id="cropBG" width="352" height="352"></canvas><figcaption>Recorded background · 20×</figcaption></figure><figure><canvas id="crop1" width="352" height="352"></canvas><figcaption>1× subtracted mean · 20×</figcaption></figure><figure><canvas id="crop2" width="352" height="352"></canvas><figcaption>2× subtracted mean · 20×</figcaption></figure></div><p id="note" class="small"></p></div>
<details><summary>Selection and interpretation</summary><p>This trial reuses the earlier per-stack detections and one-to-one matches within 6 sensor pixels. It does not rerun detection or subtract more background when calculating r. The main comparison pools spot observations, not independent preparations: 16 DNA stacks and 14 gold stacks. Counts need not represent unique rods.</p><p>No clustering threshold is introduced. Close spacing alone does not distinguish a background feature from nearby genuine rods. The chosen region is an explicit image-based mask; the whole-region trial shows sensitivity to whether 2× survival is still required for exclusion there.</p><p>The saturation veto uses the verified pre-subtraction raw-peak companion, with any pixel ≥4094 in the extraction aperture across 150 measured frames excluded. r is neither clipped nor filtered by the theta model. Histogram width 0.02; each sample is separately normalised. The dotted line is the model limit, not an exclusion threshold.</p><p id="description"></p><p><a href="results.json">Numerical results and exact spot selections</a> · <a href="mask_overview.png">Three-background mask overview</a></p></details>
</main><script>
const DATA=__DATA__;
const $=id=>document.getElementById(id), records=new Map(DATA.records.map(r=>[r.name,r]));
let bgImage=null, points=[], activeKey=null, groupToken=0, detailToken=0;
const cache=new Map();
function image(url){if(!cache.has(url))cache.set(url,new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('Cannot load '+url));im.src=url}));return cache.get(url)}
function excluded(s){return s.inside&&($('policy').value==='region'||!s.matched)}
function currentRecords(){return DATA.records.filter(r=>r.background_group===Number($('group').value))}
function currentPoints(){return currentRecords().flatMap(r=>r.methods[$('method').value])}
function displayPoints(){return currentPoints().filter(s=>!s.matched||$('survivors').checked)}
function draw(){const c=$('map'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);if(bgImage)ctx.drawImage(bgImage,0,0,c.width,c.height);
 if($('mask').checked){const e=DATA.ellipse;ctx.strokeStyle='#ffca55';ctx.lineWidth=3;ctx.beginPath();ctx.ellipse(e.cx,e.cy,e.rx,e.ry,0,0,Math.PI*2);ctx.stroke()}
 if(!$('spots').checked)return;for(const s of points){ctx.strokeStyle=excluded(s)?'#ff687e':s.matched?'#a3b4b4':'#62dbed';ctx.lineWidth=2;ctx.beginPath();ctx.arc(s.x,s.y,8,0,Math.PI*2);ctx.stroke();if(s.key===activeKey){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(Math.round(s.x)-44,Math.round(s.y)-44,88,88)}}}
function clearDetail(){++detailToken;activeKey=null;$('selectedTitle').textContent='Click a ring in the map';$('selectedInfo').textContent='';$('note').textContent='';for(const id of ['cropBG','crop1','crop2'])$(id).getContext('2d').clearRect(0,0,352,352)}
function refreshMap(){points=displayPoints();clearDetail();const a=currentPoints(),n=a.filter(excluded).length,fail=a.filter(s=>!s.matched).length;$('mapCount').textContent=`${n} excluded / ${a.length} unsaturated spots in this background group. ${fail} total survival failures; ${points.length} rings currently shown (overlaps possible).`;draw()}
function render(){const m=$('method').value,q=DATA.summary[m];$('distribution').src=m+'.png';$('groupPlot').src=m+'_groups.png';$('description').textContent=DATA.methods[m].description;
 $('summary').innerHTML=['Hinges','Gold'].map(group=>{const g=q[group],removed=g.all.n-g.targeted.n;return `<tr><td>${group==='Hinges'?'DNA / hinges':'Gold control'}</td><td>${g.all.n.toLocaleString()}</td><td>${removed} (${(100*removed/g.all.n).toFixed(1)}%)</td><td>${g.targeted.n.toLocaleString()}</td><td>${g.all.median.toFixed(3)} → ${g.targeted.median.toFixed(3)}</td><td>${g.region.n.toLocaleString()}</td></tr>`}).join('');
 const h=q.Hinges,g=q.Gold;$('headline').textContent=`The regional trial restores ${h.targeted.n-h.survival.n} DNA and ${g.targeted.n-g.survival.n} gold observations that the blanket 2× survival filter removed.`;refreshMap()}
async function chooseGroup(){const token=++groupToken;bgImage=null;refreshMap();$('error').textContent='';try{const g=DATA.groups[Number($('group').value)],im=await image(g.image);if(token!==groupToken)return;bgImage=im;draw()}catch(e){if(token===groupToken)$('error').textContent=e.message}}
function crop(id,im,s){const ctx=$(id).getContext('2d'),left=Math.round(s.x)-44,top=Math.round(s.y)-44;ctx.fillStyle='#080808';ctx.fillRect(0,0,352,352);ctx.imageSmoothingEnabled=false;const x0=Math.max(0,left),y0=Math.max(0,top),x1=Math.min(im.width,left+88),y1=Math.min(im.height,top+88);if(x1>x0&&y1>y0)ctx.drawImage(im,x0,y0,x1-x0,y1-y0,(x0-left)*4,(y0-top)*4,(x1-x0)*4,(y1-y0)*4);if($('spots').checked){const [l,r,t,b]=s.bounds;ctx.strokeStyle=excluded(s)?'#ff687e':'#62dbed';ctx.lineWidth=1.5;ctx.strokeRect((l-left)*4,(t-top)*4,(r-l)*4,(b-t)*4)}}
async function inspect(s){const token=++detailToken;activeKey=s.key;draw();const r=records.get(s.record);$('selectedTitle').textContent=`${s.key} · ${excluded(s)?'Excluded':'Retained'} · ${r.time}`;$('selectedInfo').textContent=`x=${s.x.toFixed(1)}, y=${s.y.toFixed(1)} · r=${s.r.toFixed(3)} · ${s.inside?'inside':'outside'} ellipse · ${s.matched?'survives':'fails'} 2× matching · raw peak ${s.raw_peak}/4095`;$('note').textContent=`${r.name}. Note: ${r.note||'None'}`;for(const id of ['cropBG','crop1','crop2'])$(id).getContext('2d').clearRect(0,0,352,352);try{const g=DATA.groups[r.background_group],ims=await Promise.all([image(g.image),image(r.images['1x']),image(r.images['2x'])]);if(token!==detailToken)return;['cropBG','crop1','crop2'].forEach((id,i)=>crop(id,ims[i],s))}catch(e){if(token===detailToken)$('error').textContent=e.message}}
$('method').innerHTML=Object.entries(DATA.methods).map(([m,v])=>`<option value="${m}">${v.label}</option>`).join('');$('group').innerHTML=DATA.groups.map(g=>`<option value="${g.id}">${g.label}</option>`).join('');
$('method').value='native';$('group').value='0';$('method').onchange=render;$('group').onchange=chooseGroup;$('policy').onchange=refreshMap;$('survivors').onchange=refreshMap;$('mask').onchange=draw;$('spots').onchange=()=>{draw();const s=points.find(s=>s.key===activeKey);if(s)inspect(s)};$('zoom').onchange=()=>{$('map').style.width=(100*Number($('zoom').value))+'%'};
$('map').onclick=e=>{if(!$('spots').checked)return;const c=$('map'),box=c.getBoundingClientRect(),x=(e.clientX-box.left)*c.width/box.width,y=(e.clientY-box.top)*c.height/box.height;const hits=points.map(s=>({s,d:Math.hypot(s.x-x,s.y-y)})).filter(v=>v.d<=24).sort((a,b)=>a.d-b.d);if(!hits.length)return;const at=hits.findIndex(v=>v.s.key===activeKey);inspect(hits[(at+1)%hits.length].s)};
$('download').onclick=()=>{const rows=currentPoints().filter(excluded),fields=['key','record','x','y','r','raw_peak','matched','inside'];const csv=[fields.join(','),...rows.map(s=>fields.map(k=>s[k]).join(','))].join('\r\n');const u=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=u;a.download=`excluded_${$('method').value}_bg${Number($('group').value)+1}_${$('policy').value}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
render();chooseGroup();
</script></body></html>'''

if __name__ == '__main__':
    build()
