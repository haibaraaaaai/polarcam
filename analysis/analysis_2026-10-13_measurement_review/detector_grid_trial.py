"""One-stack sensor-grid and size-cutoff comparison; no GUI changes."""
from pathlib import Path
import json, hashlib
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from polarcam.live import Detection_alg_offline as gui
from polarcam.live.intensity_spots import channel_means,anisotropy

HERE=Path(__file__).resolve().parent
OUT=HERE/'detector_grid_trial'
DAY=HERE.parents[1]/'data/hinges/polarcam/2026-10-12'
NAME='frame_stack_20261012-183850.npy'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def display(a):return np.rint(np.clip(a*20/4095,0,1)*255).astype(np.uint8)
def detect(image,step=1,origin=0,max_area=250):
    edge=round(10/step);work=image[edge:-edge,edge:-edge].astype(np.float32)
    dog=cv2.GaussianBlur(work,(0,0),sigmaX=1/step,borderType=cv2.BORDER_REPLICATE)-cv2.GaussianBlur(work,(0,0),sigmaX=6/step,borderType=cv2.BORDER_REPLICATE)
    mu,sd=gui._finite_stats(dog);threshold=max(0,mu+8*sd)
    _,labels,stats,centers=cv2.connectedComponentsWithStats((dog>threshold).astype(np.uint8),connectivity=8)
    rows=[]
    for idx in range(1,len(stats)):
        x,y,w,h,n=stats[idx];area=int(n*step*step)
        if area<10:continue
        cx,cy=(centers[idx]+edge)*step+origin
        rows.append(dict(component=int(idx),x=float(cx),y=float(cy),area=area,width=int(w*step),height=int(h*step),
            bbox=[float((x+edge)*step+origin-step/2),float((y+edge)*step+origin-step/2),int(w*step),int(h*step)],
            pass100=area<=100,pass250=area<=250,pass400=area<=400,passbox20=bool(area<=250 and w*step<=20 and h*step<=20)))
    return rows,dict(threshold=threshold,mean=mu,sd=sd,step=step,origin=origin,shape=list(image.shape))

def match(a,b,radius=6):
    if not a or not b:return {},list(range(len(a))),list(range(len(b)))
    dist=cdist([[s['x'],s['y']] for s in a],[[s['x'],s['y']] for s in b])
    costs=np.c_[np.where(dist<=radius,dist,1e8),np.full((len(a),len(a)),1e5)]
    ii,jj=linear_sum_assignment(costs);mapping={int(i):int(j) for i,j in zip(ii,jj) if j<len(b) and dist[i,j]<=radius}
    return mapping,[i for i in range(len(a)) if i not in mapping],[j for j in range(len(b)) if j not in mapping.values()]

def build():
    OUT.mkdir(exist_ok=True);(OUT/'crops').mkdir(exist_ok=True)
    p=DAY/'frames'/NAME;m=json.loads(p.with_suffix('.json').read_text());src=json.loads((HERE/'radius_detector_comparison/results.json').read_text())
    ref=next(r for r in src['records'] if r['name']==NAME);stat=p.stat()
    signature={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sidecar':sha(p.with_suffix('.json'))}
    cache=OUT/'mean.npy';sig=OUT/'mean_signature.json'
    if cache.exists() and sig.exists() and json.loads(sig.read_text())==signature:mean=np.load(cache)
    else:
        a=np.load(p,mmap_mode='r');assert a.shape[0]==151 and m['actual']['frames']==150
        mean=np.zeros(a.shape[1:],float)
        for frame in a[:150]:mean+=frame
        mean/=150;np.save(cache,mean);sig.write_text(json.dumps(signature));del a
    assert (stat.st_size,stat.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
    bgpath=DAY/'backgrounds/used'/f'{ref["background_sha256"]}.npy';assert sha(bgpath)==ref['background_sha256'];bg=np.load(bgpath)
    peakmeta=m['fetch_frames_output']['raw_peak'];peakpath=p.parent/peakmeta['path'];assert sha(peakpath)==peakmeta['sha256'];peak=np.load(peakpath)
    cells=(mean[0::2,0::2]+mean[0::2,1::2]+mean[1::2,0::2]+mean[1::2,1::2])/4
    repeated=gui.intensity_detection_map(mean)
    sliding=(mean[:-1,:-1]+mean[1:,:-1]+mean[:-1,1:]+mean[1:,1:])/4
    configs={'current':(repeated,1,0,'Current GUI: cell means repeated to sensor grid'),
             'direct':(cells,2,.5,'Direct half-size cell map; widths/areas rescaled'),
             'sliding':(sliding,1,.5,'Sliding 2x2 intensity; one sensor-pixel step')}
    methods={}
    for method,(im,step,origin,label) in configs.items():
        rows,info=detect(im,step,origin)
        for i,s in enumerate(rows):
            l,t=round(s['x'])-9,round(s['y'])-9;s['id']=f'{method}-{i+1}'
            s['raw_peak']=int(peak[t:t+18,l:l+18].max());s['saturated']=s['raw_peak']>=4094
            v=anisotropy(channel_means(mean,[(l,l+18,t,t+18)]))[0];s['xy']=v.tolist();s['r']=float(np.linalg.norm(v))
        methods[method]=dict(label=label,info=info,spots=rows,counts={k:sum(s['pass'+k] for s in rows) for k in ['100','250','400','box20']})
    baseline=[s for s in methods['current']['spots'] if s['pass250']]
    np.testing.assert_allclose([[s['x'],s['y']] for s in baseline],[[s['x'],s['y']] for s in ref['methods']['native']['spots']],atol=1e-12)
    comparison={}
    for name in ['direct','sliding']:
        other=[s for s in methods[name]['spots'] if s['pass250']];pairs,lost,new=match(baseline,other)
        comparison[name]=dict(matched=len(pairs),lost=len(lost),new=len(new),
            lost_ids=[baseline[i]['id'] for i in lost],new_ids=[other[j]['id'] for j in new],
            median_displacement=float(np.median([np.hypot(baseline[i]['x']-other[j]['x'],baseline[i]['y']-other[j]['y']) for i,j in pairs.items()])))
    cards=[]
    for s in methods['current']['spots']:
        reasons=[]
        if s['pass250'] and not s['pass100']:reasons.append('Removed by area <=100')
        if s['pass250'] and not s['passbox20']:reasons.append('Removed by bounding box <=20x20')
        if s['pass400'] and not s['pass250']:reasons.append('Added if area limit rises to 400')
        if not reasons:continue
        x,y=round(s['x']),round(s['y']);size=44
        crop_paths={}
        for gain in [5,20]:
            fig,axes=plt.subplots(1,2,figsize=(5.2,2.7))
            for ax,image,title,scale in zip(axes,[mean,bg],[f'Saved 1x mean / {gain}x display','Background / 20x display'],[gain,20]):
                crop=image[y-size:y+size,x-size:x+size]
                ax.imshow(crop,cmap='gray',vmin=0,vmax=4095/scale,interpolation='nearest',extent=(x-size-.5,x+size-.5,y+size-.5,y-size-.5))
                bx,by,bw,bh=s['bbox'];ax.add_patch(plt.Rectangle((bx,by),bw,bh,fill=False,color='#32d8e6',lw=1))
                ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(f'{s["id"]}: area {s["area"]}, box {s["width"]}x{s["height"]} sensor px',fontsize=10)
            fig.tight_layout(rect=(0,0,1,.92));path=f'crops/{s["id"]}'+('_gain5' if gain==5 else '')+'.png';fig.savefig(OUT/path,dpi=140);plt.close(fig)
            crop_paths[str(gain)]=path
        cards.append(dict(spot=s,reasons=reasons,image=crop_paths['20'],images=crop_paths))
    Image.fromarray(display(repeated)).save(OUT/'mean.png');Image.fromarray(display(gui.intensity_detection_map(bg))).save(OUT/'background.png')
    d=dict(source=NAME,source_signature=signature,methods=methods,comparison=comparison,cards=cards,
        area_units='Thresholded connected-component area in sensor pixels, not extraction-window area. Direct half-grid areas multiplied by 4. A 20x20 bounding box and area<=400 are different conditions.',
        matching='One-to-one minimum-distance assignment within 6 sensor pixels; positional matches are approximate candidate correspondence.',
        selection='Detection comparison before saturation, low-BG, ellipse or survival filtering. Saturated candidates flagged; no physical-rod claims.',
        detector_sha256=sha(Path(gui.__file__)),builder_sha256=sha(Path(__file__)))
    (OUT/'results.json').write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'index.html').write_text(HTML.replace('__DATA__',json.dumps(d).replace('</','<\\/')),encoding='utf-8')
    print(json.dumps({'counts':{k:v['counts'] for k,v in methods.items()},'comparison':comparison,'cards':len(cards)}))


HTML=r'''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>One-stack grid and size trial</title><style>body{font:16px/1.5 system-ui;color:#253236;background:#f7f8f5;margin:0}main{max-width:1400px;margin:auto;padding:24px}h1{font-size:27px}select,button{font:inherit;padding:6px}label{margin-right:15px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccd5d0;text-align:left}.scroll{overflow:auto}canvas{width:100%;height:auto;background:#111}#view{max-height:75vh;overflow:auto}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}.card{background:white;padding:12px;border:1px solid #d8ded9}.card img{width:100%}.small{font-size:14px;color:#526369}a{color:#087c89}</style><main>
<h1>Does grid choice change spot detection?</h1><p>First DNA survey stack: <b>18:38:50 on 12 October</b>, mean of 150 saved 1×-subtracted frames. Every method uses DoG widths of 1/6 <b>sensor pixels</b>, mean + 8 SD, and a minimum component area of 10 sensor pixels. The threshold is recalculated on each map. No low-background, survival or ellipse filter is applied.</p>
<p><a href="threshold_trial.html"><b>New: relax the threshold on this stack (k = 8, 6, 4, 3, 2)</b></a></p>
<div id="table" class="scroll"></div><p id="comparison"></p>
<p><b>250 currently means sensor pixels.</b> The GUI repeats the cell map back to full size before detection. It is about 62.5 cell areas, not 250 cells. A 20×20 sensor box covers 400 pixels (100 cell areas). The component area counts only pixels above the DoG threshold; it is not the rod's complete footprint or the 18×18 extraction aperture.</p>
<label>Map <select id="layer"><option value="mean">Saved mean · 20×</option><option value="background">Recorded background · 20×</option></select></label><label>Size rule <select id="rule"><option value="250">Current area ≤250</option><option value="100">Trial area ≤100</option><option value="400">Area ≤400 (100 cell areas)</option><option value="box20">Current area ≤250 AND box ≤20×20</option></select></label><label><input type="checkbox" id="excluded" checked>Show current spots removed by size rule</label>
<div><label><input type="checkbox" id="current" checked>Current GUI · cyan</label><label><input type="checkbox" id="direct">Direct half-grid · amber</label><label><input type="checkbox" id="sliding">Sliding grid · pink</label></div><p id="count" class="small"></p><div id="view"><canvas id="map" width="2464" height="2056"></canvas></div>
<p class="small">Dashed red boxes: current detections removed by the selected stricter size rule. Bright outlines show thresholded component bounding boxes. The comparison includes saturation flags but does not discard candidates, to isolate the detector changes.</p>
<h2>Inspect candidates affected by the size limits</h2><p>Cyan boxes outline the thresholded component, not a fitted rod boundary. The spot display can be reduced to inspect bright cores; the background remains at 20×. Display gain does not affect detection. These images do not automatically classify an object as one rod or a cluster.</p><label>Spot display <select id="gain"><option value="5">5×</option><option value="20">20×</option></select></label><div class="cards" id="cards"></div>
<p><a href="../background_capture_simulation/index.html">Moving-rod background-capture simulation</a> · <a href="results.json">Coordinates, sizes and provenance</a></p></main><script>
const DATA=__DATA__,$=id=>document.getElementById(id),images={};let version=0;
function update(){const layer=$('layer').value,im=images[layer]||new Image();if(!images[layer]){images[layer]=im;im.onload=()=>{if(images[$('layer').value]===im)draw()};im.src=layer+'.png'}draw()}
function draw(){const canvas=$('map'),ctx=canvas.getContext('2d'),im=images[$('layer').value],rule=$('rule').value;ctx.clearRect(0,0,canvas.width,canvas.height);if(im?.complete)ctx.drawImage(im,0,0);let labels=[];for(const [key,color] of [['current','#65e2e6'],['direct','#ffb331'],['sliding','#f877c1']]){if(!$(key).checked)continue;const spots=DATA.methods[key].spots.filter(s=>s['pass'+rule]);labels.push(key+': '+spots.length);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash([]);for(const s of spots){ctx.beginPath();ctx.arc(s.x,s.y,10,0,2*Math.PI);ctx.stroke()}}if($('excluded').checked){ctx.strokeStyle='#ff5353';ctx.setLineDash([5,4]);for(const s of DATA.methods.current.spots.filter(s=>s.pass250&&!s['pass'+rule])){const [x,y,w,h]=s.bbox;ctx.strokeRect(x-3,y-3,w+6,h+6)}ctx.setLineDash([])}$('count').textContent=labels.join(' · ')}
$('table').innerHTML='<table><tr><th>Grid</th><th>Area ≤100</th><th>Current ≤250</th><th>Area ≤400</th><th>≤250 + box ≤20×20</th></tr>'+Object.values(DATA.methods).map(m=>'<tr><td>'+m.label+'</td>'+['100','250','400','box20'].map(k=>'<td>'+m.counts[k]+'</td>').join('')+'</tr>').join('')+'</table>';
$('comparison').textContent=Object.entries(DATA.comparison).map(([k,v])=>`${k} vs current, area ≤250: ${v.matched} matched, ${v.lost} current-only, ${v.new} alternative-only (6 px match radius).`).join(' ');
function renderCards(){$('cards').innerHTML=DATA.cards.map(c=>`<article class="card"><b>${c.reasons.join(' · ')}</b><img src="${c.images[$('gain').value]}" loading="lazy"><div>${c.spot.id} · x=${c.spot.x.toFixed(1)}, y=${c.spot.y.toFixed(1)} · r=${c.spot.r.toFixed(3)} · raw peak ${c.spot.raw_peak}${c.spot.saturated?' · SATURATED':''}</div></article>`).join('')}
$('gain').onchange=renderCards;$('layer').onchange=update;for(const id of ['rule','excluded','current','direct','sliding'])$(id).onchange=draw;renderCards();update();
</script></html>'''

if __name__=='__main__':build()
