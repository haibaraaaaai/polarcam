"""Low-background spatial selection with area- and spot-retention cutoffs.

The spot target uses pooled GUI candidate positions to choose one shared BG
threshold, without tuning to angles. Acquisitions and upstream selections stay intact.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.ndimage import uniform_filter
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'low_background_mask'
SOURCE=HERE/'radius_detector_comparison/results.json'
MAP_SOURCE=HERE/'survival_background_map/data.json'
DAY=HERE.parents[1]/'data/hinges/polarcam/2026-10-12'
QUANTILES=(25,75)
CUTOFF_LABELS={'spots50':'50% GUI spots, pooled (default)', '50':'50% common area',
               '25':'Pooled background P25','75':'Pooled background P75'}
EDGE=10


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def make_masks(groups,source):
    """BG scores define masks; GUI positions set the additional spot-count target."""
    (OUT/'masks').mkdir(parents=True,exist_ok=True)
    arrays=[];scores=[]
    for g in groups:
        path=DAY/'backgrounds/used'/f'{g["sha256"]}.npy'
        assert digest(path)==g['sha256']
        a=np.load(path,allow_pickle=False).astype(np.float64)
        arrays.append(a)
        scores.append(uniform_filter(a,size=18,mode='nearest'))
    assert len(scores)==3 and len({s.shape for s in scores})==1
    h,w=scores[0].shape
    valid=np.zeros((h,w),dtype=bool);valid[EDGE:-EDGE,EDGE:-EDGE]=True
    # Equal weighting of the three backgrounds, sampled on a fixed 2-pixel grid.
    population=np.concatenate([s[EDGE:-EDGE:2,EDGE:-EDGE:2].ravel() for s in scores])
    thresholds={str(q):float(np.quantile(population,q/100)) for q in QUANTILES}
    # A common-mask pixel passes only if its largest BG mean passes. Choosing
    # the median of that worst-background map retains half the centre domain.
    # No spot counts, angles, signal, or detector outputs enter this choice.
    worst=np.maximum.reduce(scores)
    thresholds['50']=float(np.median(worst[valid]))
    # One reference population and one threshold for both sample groups.
    # Existing eligibility is unchanged; no angle values or survival flags tune it.
    reference=[]
    for r in source['records']:
        for s in r['methods']['native']['spots']:
            if s['ceiling_veto'] or s['r'] is None or not np.isfinite(s['r']):continue
            x,y=round(s['x']),round(s['y'])
            if valid[y,x]:reference.append(worst[y,x])
    reference=np.asarray(reference)
    assert len(reference)>0
    thresholds['spots50']=float(np.median(reference))
    target=dict(method='native',scope='common',pooled=True,eligible=len(reference),
                retained=int(np.sum(reference<=thresholds['spots50'])),target_fraction=.5,
                rule='Median worst-background aperture mean at pooled eligible GUI candidate centres; ties retained.')
    masks={'common':{},'own':{}}
    meta={'common':{},'own':{}}
    for q,threshold in thresholds.items():
        own=[valid & (s<=threshold) for s in scores]
        common=np.logical_and.reduce(own)
        for scope,items in [('common',[common]),('own',own)]:
            masks[scope][q]=items
            meta[scope][q]=[]
            for j,mask in enumerate(items):
                path=OUT/'masks'/f'{scope}_{q}_{j}.png'
                Image.fromarray(mask.astype(np.uint8)*255).save(path)
                # Retained area is lightly green; excluded area is dimmed.
                tint=np.zeros((h,w,4),dtype=np.uint8)
                tint[mask]=[25,185,140,42];tint[~mask]=[0,0,0,175]
                overlay=path.with_name(path.stem+'_overlay.png')
                Image.fromarray(tint).save(overlay)
                meta[scope][q].append(dict(image=f'masks/{path.name}',overlay=f'masks/{overlay.name}',
                    area=int(mask.sum()),fraction=float(mask.sum()/valid.sum()),sha256=digest(path)))
    return arrays,scores,valid,masks,meta,thresholds,target


def summary(rows,edges,area_equivalents):
    values=np.array([s['r'] for s in rows],dtype=float)
    counts=np.histogram(values,edges)[0]
    assert counts.sum()==len(values)
    return dict(n=len(rows),median=float(np.median(values)) if len(rows) else None,
                failures=sum(not s['matched'] for s in rows),
                below_02=int(np.sum(values<.2)),above_model=int(np.sum(values>RMAX)),
                area_equivalents=float(area_equivalents),density=float(len(rows)/area_equivalents) if area_equivalents else None,
                counts=counts.tolist(),percent=(100*counts/max(1,len(rows))).tolist())


def build():
    global RMAX
    OUT.mkdir(exist_ok=True)
    map_raw=MAP_SOURCE.read_bytes();old=json.loads(map_raw)
    raw=SOURCE.read_bytes();src=json.loads(raw)
    assert src['complete'] and old['source_sha256']==hashlib.sha256(raw).hexdigest()
    arrays,scores,valid,masks,mask_meta,thresholds,target=make_masks(old['groups'],src)
    RMAX=src['r_max']
    by_name={r['name']:r for r in old['records']}
    groups=[dict(g,image='../survival_background_map/'+g['image']) for g in old['groups']]
    records=[]
    for r in src['records']:
        previous=by_name[r['name']];bgid=previous['background_group']
        rr=dict(name=r['name'],group=r['group'],index=r['group_index'],time=r['time'],note=r['note'],
                background_group=bgid,images=previous['images'],methods={},excluded={})
        for method,entry in r['methods'].items():
            ss=[];veto={'saturated':0,'nonfinite':0,'border':0}
            for s in entry['spots']:
                if s['ceiling_veto']:veto['saturated']+=1;continue
                if s['r'] is None or not np.isfinite(s['r']):veto['nonfinite']+=1;continue
                x,y=round(s['x']),round(s['y'])
                if not valid[y,x]:veto['border']+=1;continue
                l,right,t,bottom=s['bounds']
                # Same aperture average and rounding convention as extraction.
                assert right-l==bottom-t==18
                np.testing.assert_allclose(scores[bgid][y,x],arrays[bgid][t:bottom,l:right].mean(),rtol=0,atol=1e-9)
                flags={scope:{q:bool(masks[scope][q][0 if scope=='common' else bgid][y,x])
                              for q in thresholds} for scope in masks}
                ss.append(dict(id=s['id'],key=f'{"H" if r["group"]=="Hinges" else "G"}{r["group_index"]:02d}-S{s["id"]:04d}',
                    record=r['name'],x=s['x'],y=s['y'],bounds=s['bounds'],r=s['r'],raw_peak=s['raw_peak'],
                    matched=s['matched'],background_mean=float(scores[bgid][y,x]),
                    worst_background_mean=float(max(a[y,x] for a in scores)),keep=flags))
            rr['methods'][method]=ss;rr['excluded'][method]=veto
        records.append(rr)
    max_r=max(s['r'] for r in records for ss in r['methods'].values() for s in ss)
    edges=np.arange(0,max(1.02,np.ceil(max_r/.02)*.02+.02)+.001,.02)
    result=dict(groups=groups,methods=src['methods'],records=records,masks=mask_meta,thresholds=thresholds,cutoff_labels=CUTOFF_LABELS,
        r_max=RMAX,bins=edges.tolist(),valid_area=int(valid.sum()),shape=list(valid.shape),
        source_sha256=hashlib.sha256(raw).hexdigest(),map_source_sha256=hashlib.sha256(map_raw).hexdigest(),
        builder_sha256=digest(Path(__file__)),summary={},spot_target=target,default_cutoff='spots50',
        rule='18x18 mean of recorded background <= one shared DN threshold. Default spots50 is the median worst-background mean at pooled eligible native GUI candidate centres, targeting 50% of those observations in the common mask. Key 50 is the median worst-background mean over all valid centres, targeting 50% common area. Other cutoffs are pooled BG P25/P75 on a fixed 2-pixel grid, equally weighting three profiles.',
        common_rule='Intersection of the three background masks. Same sensor coordinates and same accepted area for every stack.',
        own_rule='Apply the mask of each recording background; same DN threshold but potentially different areas. Correct counts for retained area.',
        selection='Start from all finite unsaturated 1x detections. Do not apply the 2x survival filter or previous ellipse. Raw ceiling veto >=4094 over any aperture pixel in the 150 pre-subtraction frames remains.',
        edge='10-pixel border excluded from the possible centre domain. The 18x18 mean is evaluated at the rounded original candidate centre.',
        caution='The spot-target threshold uses GUI candidate positions/counts, pooled across DNA and gold; the area/percentile thresholds use background alone. No cutoff is tuned to r or theta values. This is not proof of angle-unbiased detection or of a real rod; spatial correlations, threshold completeness, dynamics, and residual errors can remain.')
    for method in src['methods']:
        result['summary'][method]={}
        for scope in masks:
            result['summary'][method][scope]={}
            for selection in ['all','spots50','50','75','25']:
                result['summary'][method][scope][selection]={}
                for group in ['Hinges','Gold']:
                    rr=[r for r in records if r['group']==group]
                    spots=[s for r in rr for s in r['methods'][method]
                           if selection=='all' or s['keep'][scope][selection]]
                    area_equiv=sum(1. if selection=='all' else mask_meta[scope][selection][0 if scope=='common' else r['background_group']]['fraction'] for r in rr)
                    result['summary'][method][scope][selection][group]=summary(spots,edges,area_equiv)
    (OUT/'results.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False),encoding='utf-8')
    figures(result)
    (OUT/'index.html').write_text(HTML.replace('__DATA__',json.dumps(result,separators=(',',':'),allow_nan=False).replace('</','<\\/')),encoding='utf-8')
    assert SOURCE.read_bytes()==raw and MAP_SOURCE.read_bytes()==map_raw
    assert sum(result['summary']['native']['common']['spots50'][g]['n'] for g in ['Hinges','Gold'])==target['retained']
    print(json.dumps(dict(thresholds=thresholds,spot_target=target,
        counts={m:{q:{g:result['summary'][m]['common'][q][g]['n'] for g in ['Hinges','Gold']}
                   for q in ['all','spots50','50']} for m in src['methods']})))


def figures(d):
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'font.family':'DejaVu Sans'})
    colors={'Hinges':'#087e8b','Gold':'#c07a11'};labels={'Hinges':'DNA / hinges','Gold':'Gold control'}
    edges=np.array(d['bins'])
    for method,meta in d['methods'].items():
        for scope in ['common','own']:
            fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True,sharey=True)
            entries=d['summary'][method][scope]
            ymax=max(max(entries[p][g]['percent']) for p in entries for g in colors)*1.4
            for ax,p in zip(axes.flat,['all','50','spots50','25','75']):
                for g in colors:
                    q=entries[p][g];ax.stairs(q['percent'],edges,color=colors[g],lw=1.6,label=f'{labels[g]}: n={q["n"]:,}')
                title='Baseline: all eligible 1x detections' if p=='all' else CUTOFF_LABELS[p].replace(' (default)','')+f'\nBG <= {d["thresholds"][p]:.2f} DN'
                if p!='all' and scope=='common':title+=f' | {100*d["masks"][scope][p][0]["fraction"]:.1f}% of field'
                ax.set_title(title);ax.set_xlim(0,max(1.,edges[-1]));ax.set_ylim(0,ymax)
                ax.axvline(d['r_max'],ls=':',lw=1,color='#75838a')
                ax.set_xlabel('r at original 1x background subtraction');ax.set_ylabel('Spots per 0.02 bin (%)')
                ax.legend(frameon=False,fontsize=9,loc='upper left');ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
            axes.flat[-1].axis('off')
            axes.flat[-1].text(.02,.85,'50% spots: one shared cutoff\nchosen from pooled GUI candidate positions.\n\nThe target applies to GUI + common mask.\nOther detectors and the own-BG mask\ncan retain different fractions.\n\n50% area: independent of spot positions.\nNo cutoff is tuned to r or theta values.',
                               transform=axes.flat[-1].transAxes,va='top',fontsize=11,linespacing=1.5)
            scope_label='same low-BG coordinates for all stacks' if scope=='common' else 'each recording uses its own background mask'
            fig.suptitle(meta['label']+' | '+scope_label,fontsize=14)
            fig.text(.025,.014,'Low-BG masks; 2x survival NOT required. Raw saturation excluded; all finite r retained. DNA: 16 stacks; gold: 14.',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(OUT/f'{method}_{scope}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5.2))
    tint=np.asarray(Image.open(OUT/d['masks']['common']['spots50'][0]['overlay']))
    for ax,g in zip(axes,d['groups']):
        ax.imshow(Image.open(OUT/g['image']),cmap='gray');ax.imshow(tint)
        ax.set_title(g['label'],fontsize=10);ax.set_xlabel('Sensor x (px)');ax.set_ylabel('Sensor y (px)')
    fig.suptitle(f'50% GUI spots: same low-BG mask in all three images | <= {d["thresholds"]["spots50"]:.2f} DN in every background | green = retained',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'mask_overview.png',dpi=160);plt.close(fig)


HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Low-background area selection</title>
<style>*{box-sizing:border-box}body{font:16px/1.5 system-ui;background:#f7f8f5;color:#263337;margin:0;padding:28px}main{max-width:1500px;margin:auto}h1{font-size:28px;margin:0 0 8px}h2{font-size:21px;margin-top:24px}p{max-width:1150px}a{color:#08798b}.small{font-size:13px;color:#536065}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px}label{display:inline-flex;align-items:center;gap:6px;margin:5px 0}select,button{font:inherit;padding:6px}.box{padding:16px;background:white;border:1px solid #d9dfdb;border-radius:7px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:7px;border-bottom:1px solid #dde3df}.plots{width:100%;display:block}#viewport{overflow:auto;max-height:78vh;background:#111}#map{width:100%;height:auto;display:block;cursor:crosshair}.detail{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0}.detail canvas{width:264px;height:264px;image-rendering:pixelated;background:#111}figcaption{font-size:13px}#error{color:#b43248}</style></head><body><main>
<h1>Count only detections in low-background areas</h1>
<p>The default cutoff keeps <b>about 50% of eligible GUI-detected spots, pooled across DNA and gold</b>. It uses the recorded background mean in the same 18×18 window used for spot extraction, choosing one shared threshold from those candidate positions. DNA and gold use exactly the same spatial mask; each group need not retain 50%. The <b>50% common area</b> option is also available.</p>
<p><b>2× survival is not required.</b> This trial starts from all unsaturated 1× detections and replaces both the survival filter and the previous ellipse. All finite r values remain eligible. A spot surviving 2× subtraction can still be excluded by a bright background, and a failed-survival spot can remain if its background is low.</p>
<div class="controls"><label>Detector <select id="method"></select></label><label>Mask <select id="scope"><option value="common">Common region: low in all 3 backgrounds</option><option value="own">Each recording's own background</option></select></label><label>BG cutoff <select id="cutoff"></select></label></div>
<div class="box"><b id="headline"></b><div class="scroll"><table><thead><tr><th>Sample</th><th>Eligible before mask</th><th>Kept</th><th>Spots retained</th><th>Kept despite failing 2×</th><th>Area retained</th><th>Spots / full-field equivalent</th><th>Median r</th></tr></thead><tbody id="summary"></tbody></table></div><p class="small">The 50%-spot target applies to the GUI detector with the common mask. Other detectors or each recording's own mask may retain different fractions at that same cutoff. Area is summed over recordings; a full-field equivalent is the area inside the 10-pixel border. Counts are observations, not guaranteed distinct rods.</p></div>
<p class="small">The five panels compare the baseline and all four cutoffs; the table and map use the selected cutoff. Each sample's histogram is normalised to its own kept count. Cutoffs use background levels, without tuning to r or theta values.</p>
<img id="plot" class="plots" alt="Radius distributions comparing the 50%-spot and 50%-area masks plus two other background cutoffs">
<h2>Inspect the mask and candidate spots</h2>
<div class="controls"><label>Background <select id="group"></select></label><label>Recording <select id="record"><option value="all">All sharing this background</option></select></label><label>Image <select id="layer"><option value="background">Background</option><option value="mask">Binary mask only</option></select></label><label>Zoom <select id="zoom"><option value="1">Fit</option><option value="1.5">150%</option><option value="2">200%</option></select></label></div>
<div class="controls"><label><input id="overlay" type="checkbox" checked>Show mask overlay</label><label><input id="spots" type="checkbox" checked>Show spots</label><label><input id="outside" type="checkbox">Also show excluded spots</label><button id="download">Export kept spots</button></div>
<p id="mapCount" class="small"></p><div id="error"></div><div id="viewport"><canvas id="map" width="2464" height="2056"></canvas></div>
<p class="small">Green shading / white binary pixels = allowed centres. Cyan rings = kept and survive 2×; pink rings = kept but fail 2×; grey = outside mask. All saturated detections are excluded. Click a ring; repeat clicks to cycle overlaps.</p>
<div class="box"><b id="selectedTitle">Click a spot to inspect it</b><p id="selectedInfo" class="small"></p><div class="detail"><figure><canvas id="cropBG" width="352" height="352"></canvas><figcaption>Actual recorded background · 20×</figcaption></figure><figure><canvas id="crop1" width="352" height="352"></canvas><figcaption>1× mean · 20×</figcaption></figure><figure><canvas id="crop2" width="352" height="352"></canvas><figcaption>2× mean · 20× (inspection only)</figcaption></figure></div><p id="note" class="small"></p></div>
<details><summary>Exact rule and limits</summary><p>At every possible centre, take the largest of the three 18×18 mean-background values. The default 50%-spot cutoff is the median of those values at all eligible GUI candidate centres, pooling DNA and gold observations; ties are retained. This targets half the pooled GUI spots in the common mask, without choosing separate cutoffs for either sample or using the r distribution. The 50%-area cutoff is the median over all valid image centres, independent of candidate positions. The two other cutoffs are the pooled background 25th and 75th percentiles. Every cutoff uses one identical DN value for DNA and control; choosing each recording's own mask can retain more area and spots than the common intersection.</p><p>All maps use sensor coordinates, x right and y down. A candidate's rounded centre is tested against the mask. The average is of the original background mosaic across the whole 18×18 aperture; it does not use a rod/background ratio. The field border excludes possible centres within 10 pixels of an edge. The background files are verified against their recorded SHA-256 identifiers.</p><p>This selects low average background, not necessarily smooth background, and does not prove that each accepted candidate is a rod. Noise or local residual features can remain. The spot-target threshold uses detected positions, but no cutoff is tuned to rod angles. Spatial correlations and the existing detection thresholds can still affect the observed distribution. We have not rerun detection inside the mask; the existing detector choices are held fixed.</p><p>All r values use original 1× channels. The raw saturation veto is unchanged: any pixel ≥4094 in an 18×18 aperture over the 150 measured pre-subtraction frames excludes the candidate. No r or theta threshold is applied.</p><p><a href="results.json">Numbers, masks, exact selections and provenance</a> · <a href="mask_overview.png">Three-background mask overview (50% GUI spots)</a> · <a href="../background_region_trial/index.html">Previous ellipse trial</a> · <a href="../survival_background_map/index.html">Original survival-failure map</a></p></details>
</main><script>
const DATA=__DATA__, $=id=>document.getElementById(id),records=new Map(DATA.records.map(r=>[r.name,r]));
let bgImage=null,maskImage=null,tintImage=null,points=[],activeKey=null,mapToken=0,detailToken=0;
const cache=new Map();function image(url){if(!cache.has(url))cache.set(url,new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('Cannot load '+url));im.src=url}));return cache.get(url)}
function kept(s){return s.keep[$('scope').value][$('cutoff').value]}
function maskMeta(){return DATA.masks[$('scope').value][$('cutoff').value][$('scope').value==='common'?0:Number($('group').value)]}
function currentRecords(){return DATA.records.filter(r=>r.background_group===Number($('group').value)&&($('record').value==='all'||r.name===$('record').value))}
function allPoints(){return currentRecords().flatMap(r=>r.methods[$('method').value])}
function clearDetail(){++detailToken;activeKey=null;$('selectedTitle').textContent='Click a spot to inspect it';$('selectedInfo').textContent='';$('note').textContent='';for(const id of ['cropBG','crop1','crop2'])$(id).getContext('2d').clearRect(0,0,352,352)}
function draw(){const c=$('map'),ctx=c.getContext('2d');ctx.fillStyle='#090909';ctx.fillRect(0,0,c.width,c.height);if($('layer').value==='mask'){if(maskImage)ctx.drawImage(maskImage,0,0)}else{if(bgImage)ctx.drawImage(bgImage,0,0);if(tintImage&&$('overlay').checked)ctx.drawImage(tintImage,0,0)}if(!$('spots').checked)return;
 for(const s of points){ctx.strokeStyle=!kept(s)?'#8c9399':s.matched?'#69e1eb':'#ff7aac';ctx.lineWidth=2;ctx.beginPath();ctx.arc(s.x,s.y,8,0,2*Math.PI);ctx.stroke();if(s.key===activeKey){ctx.strokeStyle='#fff';ctx.strokeRect(Math.round(s.x)-44,Math.round(s.y)-44,88,88)}}}
function refreshPoints(){clearDetail();const a=allPoints();points=a.filter(s=>$('outside').checked||kept(s));const k=a.filter(kept);$('mapCount').textContent=`${k.length} kept / ${a.length} eligible candidates from ${currentRecords().length} recording(s); ${k.filter(s=>!s.matched).length} kept candidates fail 2× survival. Mask covers ${(100*maskMeta().fraction).toFixed(1)}% of this field.`;draw()}
function updateSummary(){const m=$('method').value,scope=$('scope').value,q=$('cutoff').value,all=DATA.summary[m][scope].all,now=DATA.summary[m][scope][q],n=now.Hinges.n+now.Gold.n,total=all.Hinges.n+all.Gold.n;$('plot').src=m+'_'+scope+'.png';$('headline').textContent=`${DATA.cutoff_labels[q]} · cutoff ${DATA.thresholds[q].toFixed(2)} DN · ${n} / ${total} spots kept (${(100*n/total).toFixed(1)}% pooled); ${scope==='common'?'identical coordinates for all 30 recordings':'each recording uses its own background mask'}.`;
 $('summary').innerHTML=['Hinges','Gold'].map(g=>{const a=all[g],s=now[g];return `<tr><td>${g==='Hinges'?'DNA / hinges':'Gold control'}</td><td>${a.n}</td><td>${s.n}</td><td>${(100*s.n/a.n).toFixed(1)}%</td><td>${s.failures}</td><td>${(100*s.area_equivalents/a.area_equivalents).toFixed(1)}%</td><td>${s.density?.toFixed(1)??'—'}</td><td>${s.median?.toFixed(3)??'—'}</td></tr>`}).join('')}
async function loadMap(){const token=++mapToken;bgImage=null;maskImage=null;tintImage=null;$('error').textContent='';refreshPoints();const g=DATA.groups[Number($('group').value)],m=maskMeta();try{const ims=await Promise.all([image(g.image),image(m.image),image(m.overlay)]);if(token!==mapToken)return;[bgImage,maskImage,tintImage]=ims;draw()}catch(e){if(token===mapToken)$('error').textContent=e.message}}
function rebuildRecords(){$('record').innerHTML='<option value="all">All sharing this background</option>'+DATA.records.filter(r=>r.background_group===Number($('group').value)).map(r=>`<option value="${r.name}">${r.group==='Hinges'?'DNA':'Gold'} ${r.index} · ${r.time}</option>`).join('');$('record').value='all'}
function crop(id,im,s){const ctx=$(id).getContext('2d'),left=Math.round(s.x)-44,top=Math.round(s.y)-44;ctx.fillStyle='#080808';ctx.fillRect(0,0,352,352);ctx.imageSmoothingEnabled=false;const x0=Math.max(0,left),y0=Math.max(0,top),x1=Math.min(im.width,left+88),y1=Math.min(im.height,top+88);if(x1>x0&&y1>y0)ctx.drawImage(im,x0,y0,x1-x0,y1-y0,(x0-left)*4,(y0-top)*4,(x1-x0)*4,(y1-y0)*4);if($('spots').checked){const [l,r,t,b]=s.bounds;ctx.strokeStyle=kept(s)?'#69e1eb':'#8c9399';ctx.lineWidth=1.5;ctx.strokeRect((l-left)*4,(t-top)*4,(r-l)*4,(b-t)*4)}}
async function inspect(s){const token=++detailToken;activeKey=s.key;draw();const r=records.get(s.record);$('selectedTitle').textContent=`${s.key} · ${kept(s)?'Kept':'Outside mask'} · ${r.time}`;$('selectedInfo').textContent=`Actual BG mean ${s.background_mean.toFixed(2)} DN; maximum across 3 BGs ${s.worst_background_mean.toFixed(2)} DN. r=${s.r.toFixed(3)} · x=${s.x.toFixed(1)}, y=${s.y.toFixed(1)} · ${s.matched?'survives':'fails'} 2× · raw peak ${s.raw_peak}/4095`;$('note').textContent=`${r.name}. Note: ${r.note||'None'}`;for(const id of ['cropBG','crop1','crop2'])$(id).getContext('2d').clearRect(0,0,352,352);try{const ims=await Promise.all([image(DATA.groups[r.background_group].image),image(r.images['1x']),image(r.images['2x'])]);if(token!==detailToken)return;['cropBG','crop1','crop2'].forEach((id,i)=>crop(id,ims[i],s))}catch(e){if(token===detailToken)$('error').textContent=e.message}}
$('method').innerHTML=Object.entries(DATA.methods).map(([m,v])=>`<option value="${m}">${v.label}</option>`).join('');$('group').innerHTML=DATA.groups.map(g=>`<option value="${g.id}">${g.label}</option>`).join('');$('cutoff').innerHTML=['spots50','50','25','75'].map(q=>`<option value="${q}">${DATA.thresholds[q].toFixed(2)} DN · ${DATA.cutoff_labels[q]}</option>`).join('');$('method').value='native';$('group').value='0';$('cutoff').value=DATA.default_cutoff;
$('method').onchange=()=>{updateSummary();refreshPoints()};$('scope').onchange=$('cutoff').onchange=()=>{updateSummary();loadMap()};$('group').onchange=()=>{rebuildRecords();loadMap()};$('record').onchange=$('outside').onchange=refreshPoints;$('overlay').onchange=$('layer').onchange=draw;$('spots').onchange=()=>{draw();const s=points.find(s=>s.key===activeKey);if(s)inspect(s)};$('zoom').onchange=()=>{$('map').style.width=(100*Number($('zoom').value))+'%'};
$('map').onclick=e=>{if(!$('spots').checked)return;const c=$('map'),b=c.getBoundingClientRect(),x=(e.clientX-b.left)*c.width/b.width,y=(e.clientY-b.top)*c.height/b.height;const hits=points.map(s=>({s,d:Math.hypot(s.x-x,s.y-y)})).filter(v=>v.d<=24).sort((a,b)=>a.d-b.d);if(!hits.length)return;const at=hits.findIndex(v=>v.s.key===activeKey);inspect(hits[(at+1)%hits.length].s)};
$('download').onclick=()=>{const fields=['key','record','x','y','r','background_mean','worst_background_mean','raw_peak','matched'];const csv=[fields.join(','),...allPoints().filter(kept).map(s=>fields.map(k=>s[k]).join(','))].join('\r\n');const u=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=u;a.download=`low_bg_${$('method').value}_${$('scope').value}_cut${$('cutoff').value}_bg${Number($('group').value)+1}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
rebuildRecords();updateSummary();loadMap();
</script></body></html>'''

if __name__=='__main__':build()
