"""Map 1x-only candidates onto their actual recorded backgrounds; no reselection."""
from pathlib import Path
import hashlib
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from polarcam.live.Detection_alg_offline import intensity_detection_map

HERE = Path(__file__).resolve().parent
OUT = HERE/'survival_background_map'
SOURCE = HERE/'radius_detector_comparison/results.json'
PRIOR = HERE/'matched_background_theta/results.json'
DAY = HERE.parents[1]/'data/hinges/polarcam/2026-10-12'
GROUP_NAMES = ('DNA · stacks 1–9 · BG 18:36:50', 'DNA · stacks 10–16 · BG 18:50:32',
               'Gold · stacks 1–14 · BG 19:21:57')


def url(path):
    return os.path.relpath(path,OUT).replace('\\','/')


def build():
    OUT.mkdir(exist_ok=True)
    (OUT/'images').mkdir(exist_ok=True)
    source_bytes, previous_bytes = SOURCE.read_bytes(), PRIOR.read_bytes()
    source, previous = json.loads(source_bytes), json.loads(previous_bytes)
    assert source['complete']
    assert source['signature']['source_sha256'] == hashlib.sha256(previous_bytes).hexdigest()
    by_name = {r['name']:r for r in previous['records']}
    backgrounds = {}
    for r in source['records']:
        sha=r['background_sha256']
        if sha not in backgrounds:
            path=DAY/'backgrounds/used'/f'{sha}.npy'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
            a=np.load(path,allow_pickle=False).astype(np.float64)
            mapped=intensity_detection_map(a)
            image=f'images/background_{sha[:12]}.webp'
            Image.fromarray(np.rint(np.clip(mapped*20/4095,0,1)*255).astype(np.uint8)).save(OUT/image,lossless=True,method=4)
            # Distribution of the same 18x18 aperture means, every two sensor pixels.
            average=uniform_filter(a,size=18,mode='nearest')
            population=np.sort(average[9:a.shape[0]-8:2,9:a.shape[1]-8:2],axis=None)
            backgrounds[sha]=dict(array=a,mapped=mapped,population=population,
                                  index=len(backgrounds),image=image)
    assert len(backgrounds)==3
    groups=[]
    for sha,b in backgrounds.items():
        rr=[r for r in source['records'] if r['background_sha256']==sha]
        groups.append(dict(id=b['index'],label=GROUP_NAMES[b['index']],image=b['image'],
            sha256=sha,background=rr[0]['background'],width=b['array'].shape[1],height=b['array'].shape[0],
            recordings=[r['name'] for r in rr]))
    records=[]
    for r in source['records']:
        old=by_name[r['name']]
        b=backgrounds[r['background_sha256']]
        a=b['array']
        record=dict(name=r['name'],group=r['group'],index=r['group_index'],time=r['time'],
            note=r['note'],background_group=b['index'],background=r['background'],
            images={f: url((HERE/'matched_background_theta'/old['factors'][j]['image']).resolve())
                    for f,j in (('1x',0),('2x',-1))},methods={})
        for method,entry in r['methods'].items():
            rows=[]
            other=np.asarray(entry['points_2x'],dtype=float).reshape(-1,2)
            tree=cKDTree(other) if len(other) else None
            native_pts=np.asarray(r['methods']['native']['points_2x'],dtype=float).reshape(-1,2)
            native_tree=cKDTree(native_pts) if len(native_pts) else None
            eligible=[s for s in entry['spots'] if not s['matched'] and not s['ceiling_veto']]
            for s in eligible:
                l,rr,t,bb=s['bounds']
                x,y=s['x'],s['y']
                bg_mean=float(a[t:bb,l:rr].mean())
                # Validate the aperture-centre convention used for its percentile.
                ix,iy=round(x),round(y)
                test=uniform_filter(a[max(0,iy-20):iy+21,max(0,ix-20):ix+21],size=18,mode='nearest')
                np.testing.assert_allclose(test[min(iy,20),min(ix,20)],bg_mean,atol=1e-9,rtol=0)
                signal=float(np.mean(s['channels']))
                yy,xx=np.mgrid[max(0,iy-32):min(a.shape[0],iy+33),max(0,ix-32):min(a.shape[1],ix+33)]
                distance=np.hypot(xx-x,yy-y)
                ring=(distance>=12)&(distance<=30)
                ring_median=float(np.median(b['mapped'][yy,xx][ring]))
                nearest=float(tree.query([x,y])[0]) if tree else None
                native_nearest=float(native_tree.query([x,y])[0]) if native_tree else None
                rows.append(dict(id=s['id'],key=f'{"H" if r["group"]=="Hinges" else "G"}{r["group_index"]:02d}-S{s["id"]:04d}',
                    record=r['name'],x=x,y=y,bounds=s['bounds'],r=s['r'],raw_peak=s['raw_peak'],
                    bg_mean=bg_mean,signal_1x_mean=signal,bg_to_1x=bg_mean/signal if signal>0 else None,
                    bg_percentile=float(100*np.searchsorted(b['population'],bg_mean,side='right')/len(b['population'])),
                    bg_ring_median=ring_median,nearest_2x=nearest,nearest_native_2x=native_nearest))
            record['methods'][method]=dict(spots=rows,
                lost_with_raw_ceiling=sum(not s['matched'] and s['ceiling_veto'] for s in entry['spots']))
        records.append(record)
    counts={method:{group:sum(len(r['methods'][method]['spots']) for r in records if r['group']==group)
                    for group in ('Hinges','Gold')} for method in source['methods']}
    for m,groups_count in counts.items():
        for group,count in groups_count.items(): assert count==source['summary'][m][group]['removed']['n']
    data=dict(groups=groups,records=records,methods=source['methods'],counts=counts,
        selection='1x detections with no one-to-one 2x match; raw-ceiling veto retained; no theta/radius cut. 2x-only candidates are not listed or drawn.',
        display='20x, white=204.75 DN. Background and saved mean previews use four-polarizer-cell intensity maps.',
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),prior_sha256=hashlib.sha256(previous_bytes).hexdigest(),
        bg_percentile='Percentile among 18x18 means across the same BG, sampled every 2 sensor pixels. Descriptive, not a rejection rule.')
    (OUT/'data.json').write_text(json.dumps(data,separators=(',',':'),ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (OUT/'index.html').write_text(HTML.replace('__DATA__',json.dumps(data,separators=(',',':'),ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    preview(data)
    assert SOURCE.read_bytes()==source_bytes and PRIOR.read_bytes()==previous_bytes
    print('Counts:',counts)
    print(OUT/'index.html')


def preview(data):
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',20)
    title=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',29)
    cw,ch=1232,1028
    sheet=Image.new('RGB',(cw,3*(ch+86)),'#f5f7f3')
    d=ImageDraw.Draw(sheet)
    for g in data['groups']:
        points=[s for r in data['records'] if r['background_group']==g['id'] for s in r['methods']['native']['spots']]
        im=Image.open(OUT/g['image']).convert('RGB').resize((cw,ch),Image.Resampling.LANCZOS)
        draw=ImageDraw.Draw(im)
        for s in points:
            x,y=s['x']/2,s['y']/2
            draw.ellipse((x-4,y-4,x+4,y+4),outline='#ffd44b',width=1)
        offset=g['id']*(ch+86)
        d.text((16,offset+6),g['label'],fill='#243b40',font=title)
        d.text((16,offset+45),f'{len(points)} unsaturated 1x-only candidates · GUI DoG · background at 20x',fill='#526a70',font=font)
        sheet.paste(im,(0,offset+86))
        im.save(OUT/f'background_map_{g["id"]+1}.png')
    sheet.save(OUT/'overview.png')


HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Survival failures on the background</title><style>
:root{font:15px/1.5 system-ui,sans-serif;color:#243b40;background:#f5f7f3}*{box-sizing:border-box}body{margin:0}main{max-width:1540px;margin:auto;padding:22px}h1{font-size:29px;margin:6px 0}h2{font-size:21px}p{max-width:1170px}.small{font-size:13px;color:#526a70}a{color:#08788c}.box,.controls{background:white;border:1px solid #d0ded7;border-radius:7px;padding:12px;margin:14px 0}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}select,button,input{font:inherit;color:inherit}select,button{padding:6px 9px;background:white;border:1px solid #a2b9b0;border-radius:5px}button{cursor:pointer}button:disabled{opacity:.4}button:hover:not(:disabled){background:#e9f3ed}.toggle{font-weight:650;background:#fff5ce;padding:7px;border-radius:5px}.mapViewport{width:100%;max-height:76vh;overflow:auto;background:#080808;border:1px solid #334c4b}.map{display:block;width:100%;height:auto;cursor:crosshair}.legend{color:#866805}.inspector{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.inspector canvas{width:100%;height:auto;background:#080808;display:block;image-rendering:pixelated}.inspector h3{font-size:15px;margin:0 0 5px}.nav{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin:14px 0}.nav select{max-width:540px}.stats{font-size:14px}.stats b{font-variant-numeric:tabular-nums}.tableWrap{overflow-x:auto;max-height:350px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:6px 8px;border-bottom:1px solid #d4dfd9}thead{position:sticky;top:0;background:#f7faf6}tbody tr{cursor:pointer}tbody tr:hover,tbody tr.selected{background:#e6f1ec}summary{cursor:pointer}.notice{border-left:3px solid #b98234;padding:8px 12px;background:#fff7e8}#error{color:#a33443}#loaded{min-height:1.5em}@media(max-width:740px){main{padding:12px}.inspector{grid-template-columns:1fr}.mapViewport{max-height:64vh}.nav select{max-width:100%}}
</style></head><body><main><div class="small">12 OCTOBER · FAILED 1× → 2× SURVIVAL ONLY</div><h1>Where do the lost candidates sit on the background?</h1>
<p><a href="../background_region_trial/index.html"><b>New trial: exclude only the large background feature and compare the remaining distributions</b></a></p>
<p><b>Yellow rings: candidates detected at 1× BG with no matching 2× detection.</b> Saturated candidates remain excluded. No theta/radius filter is applied. Candidates found only at 2× are not shown.</p>
<div class="controls"><label>Background <select id="group"></select></label><label>Detector <select id="method"></select></label><label>Recording <select id="record"><option value="all">All sharing this background</option></select></label><label class="toggle"><input id="overlay" type="checkbox" checked> Show spots</label><label>Zoom <select id="zoom"><option value="1">Fit whole map</option><option value="1.5">150%</option><option value="2">200%</option><option value="3">300%</option></select></label><button id="centre">Centre selected spot</button></div>
<p id="count"></p><p id="loaded" class="small"></p><p id="error" role="alert"></p>
<div id="viewport" class="mapViewport"><canvas id="map" class="map" width="2464" height="2056" aria-label="Recorded background with togglable failed-candidate markers"></canvas></div>
<p class="small legend">Click a ring to inspect it; repeated clicks cycle through nearby rings. Pink square = selected 88 × 88 crop. Rings from all selected recordings are pooled in sensor coordinates; overlapping rings may represent different recordings. Toggle “Show spots” to see the background unobstructed.</p>
<div class="nav"><button id="previous">← Previous spot</button><select id="spot"></select><button id="next">Next spot →</button><span id="position" class="small"></span><label>Order <select id="sort"><option value="record">Recording, then spot ID</option><option value="bright">Brightest BG first</option><option value="ratio">Highest BG / 1× signal first</option></select></label></div>
<h2 id="selectedTitle">Select a candidate</h2><p id="selectedInfo" class="small"></p>
<div class="inspector"><section><h3>Recorded BG · 20×</h3><canvas id="cropBG" width="352" height="352"></canvas></section><section><h3>1× BG mean · 20×</h3><canvas id="crop1" width="352" height="352"></canvas></section><section><h3>2× BG mean · 20×</h3><canvas id="crop2" width="352" height="352"></canvas></section></div>
<p class="small">Crops show the same 88 × 88 sensor pixels at fixed gain. Red square: the 18 × 18 extraction window. All panels use four-polarizer-cell intensity maps; white = 204.75 saved intensity units. Background is the recorded profile, not an inferred residual.</p>
<div id="metrics" class="box stats"></div><p id="recordNote" class="small"></p>
<p class="notice"><b>Inspection only:</b> a bright background under a candidate does not prove it is a false spot. This view helps compare bright-background regions with dim-signal candidates; it does not alter the survival filter or label causes automatically. Failure can also reflect a shifted centre, merging, or the local-contrast criterion.</p>
<details class="box"><summary>List the displayed candidates and their background brightness</summary><div class="tableWrap"><table><thead><tr><th>Candidate</th><th>Time</th><th>x, y</th><th>BG mean DN</th><th>BG percentile</th><th>1× mean DN</th><th>BG / 1×</th><th>r at 1×</th></tr></thead><tbody id="table"></tbody></table></div><p class="small">BG percentile compares the 18 × 18 mean with windows across this same background. BG / 1× compares the original BG magnitude with the already-subtracted signal; neither is a calibrated false-positive score.</p><button id="download">Download displayed candidates CSV</button></details>
<p><a href="../radius_detector_comparison/index.html">Detector and survival-filter distributions</a> · <a href="../excluded_hinge_spots/index.html">Earlier excluded hinge gallery</a> · <a href="overview.png">Three background maps, GUI detector (PNG)</a> · <a href="data.json">Map data and source hashes</a></p>
</main><script>
const DATA=__DATA__, $=id=>document.getElementById(id), esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const records=new Map(DATA.records.map(r=>[r.name,r])),images=new Map();let candidates=[],active=0,bgImage=null,groupToken=0,detailToken=0;
const number=(v,d=2)=>Number.isFinite(v)?v.toFixed(d):'undefined';
function sizeMap(){const c=$('map'),v=$('viewport'),fit=Math.min(v.clientWidth,window.innerHeight*.70*c.width/c.height);c.style.width=(fit*Number($('zoom').value))+'px'}
function loadImage(path){if(!images.has(path))images.set(path,new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>{images.delete(path);reject(Error('Could not load '+path))};im.src=path}));return images.get(path)}
for(const g of DATA.groups)$('group').add(new Option(g.label,g.id));for(const [key,m] of Object.entries(DATA.methods))$('method').add(new Option(m.label,key));
function rebuildRecords(){const group=Number($('group').value);$('record').innerHTML='<option value="all">All sharing this background</option>';for(const r of DATA.records.filter(r=>r.background_group===group))$('record').add(new Option(`${r.group==='Hinges'?'DNA':'Gold'} ${r.index} · ${r.time}`,r.name))}
function sortedCandidates(){const group=Number($('group').value),method=$('method').value,rec=$('record').value;let points=DATA.records.filter(r=>r.background_group===group&&(rec==='all'||r.name===rec)).flatMap(r=>r.methods[method].spots);
 if($('sort').value==='bright')points.sort((a,b)=>b.bg_mean-a.bg_mean);if($('sort').value==='ratio')points.sort((a,b)=>(b.bg_to_1x??-1)-(a.bg_to_1x??-1));return points}
function drawMap(){const c=$('map'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);if(!bgImage)return;ctx.drawImage(bgImage,0,0,c.width,c.height);if(!$('overlay').checked)return;
 ctx.strokeStyle='#ffd44b';ctx.lineWidth=2;ctx.beginPath();for(const s of candidates){ctx.moveTo(s.x+8,s.y);ctx.arc(s.x,s.y,8,0,Math.PI*2)}ctx.stroke();const s=candidates[active];if(s){ctx.strokeStyle='#ff6984';ctx.lineWidth=3;ctx.strokeRect(Math.round(s.x)-44,Math.round(s.y)-44,88,88)}}
function crop(canvas,im,s){const ctx=canvas.getContext('2d');ctx.fillStyle='#080808';ctx.fillRect(0,0,352,352);ctx.imageSmoothingEnabled=false;const left=Math.round(s.x)-44,top=Math.round(s.y)-44;const x0=Math.max(0,left),y0=Math.max(0,top),x1=Math.min(im.width,left+88),y1=Math.min(im.height,top+88);
 if(x1>x0&&y1>y0)ctx.drawImage(im,x0,y0,x1-x0,y1-y0,(x0-left)*4,(y0-top)*4,(x1-x0)*4,(y1-y0)*4);
 if($('overlay').checked){const [l,r,t,b]=s.bounds;ctx.strokeStyle='#ff6984';ctx.lineWidth=1.5;ctx.strokeRect((l-left)*4,(t-top)*4,(r-l)*4,(b-t)*4)}}
function clearCrops(){for(const id of ['cropBG','crop1','crop2'])$(id).getContext('2d').clearRect(0,0,352,352)}
async function inspect(index){const token=++detailToken;active=Math.max(0,Math.min(candidates.length-1,index));clearCrops();drawMap();const s=candidates[active];$('spot').value=active;$('previous').disabled=active<=0;$('next').disabled=active>=candidates.length-1;$('centre').disabled=!s;
 if(!s){$('selectedTitle').textContent='No candidates in this selection';$('selectedInfo').textContent='';$('metrics').textContent='';$('recordNote').textContent='';$('position').textContent='0 candidates';return}
 const r=records.get(s.record);$('position').textContent=`${active+1} / ${candidates.length}`;$('selectedTitle').textContent=`${s.key} · ${r.time}`;$('selectedInfo').textContent=`${r.name} · x=${number(s.x,1)}, y=${number(s.y,1)} · r at 1× = ${number(s.r,3)}`;
 let match=s.nearest_2x===null?'No 2× candidates in this recipe.':`Nearest 2× candidate: ${number(s.nearest_2x)} sensor pixels away.`;
 if(s.nearest_2x!==null&&s.nearest_2x<=6)match+=' A candidate is nearby, but the one-to-one assignment did not match this 1× candidate.';
 if($('method').value==='local'&&s.nearest_native_2x!==null&&s.nearest_native_2x<=6&&(s.nearest_2x===null||s.nearest_2x>6))match+=' A native GUI detection is within 6 px at 2×; the local-quality recipe did not retain a nearby detection.';
 $('metrics').innerHTML=`<b>BG mean: ${number(s.bg_mean)} DN</b> · ${number(s.bg_percentile,1)}th percentile among BG windows<br><b>1× saved mean: ${number(s.signal_1x_mean)} DN</b> · BG / 1× = ${number(s.bg_to_1x)}<br>BG neighbourhood median (12–30 px ring): ${number(s.bg_ring_median)} DN · Raw peak: ${s.raw_peak}/4095<br><span class="small">${esc(match)} These brightness values use the original numerical data, before display gain.</span>`;
 $('recordNote').textContent=`Recorded note: ${r.note||'None'}. Background: ${r.background}.`;
 const rows=$('table').querySelectorAll('tr');rows.forEach(tr=>tr.classList.toggle('selected',Number(tr.dataset.index)===active));
 try{const g=DATA.groups[Number($('group').value)],ims=await Promise.all([loadImage(g.image),loadImage(r.images['1x']),loadImage(r.images['2x'])]);if(token!==detailToken)return;for(const [i,id] of ['cropBG','crop1','crop2'].entries())crop($(id),ims[i],s)}catch(e){if(token===detailToken)$('error').textContent=e.message}}
function renderList(keepKey){candidates=sortedCandidates();$('spot').innerHTML=candidates.map((s,i)=>`<option value="${i}">${s.key} · ${records.get(s.record).time} · BG ${number(s.bg_mean,1)} DN</option>`).join('');$('spot').disabled=!candidates.length;$('download').disabled=!candidates.length;
 const g=DATA.groups[Number($('group').value)],rr=DATA.records.filter(r=>r.background_group===g.id&&($('record').value==='all'||r.name===$('record').value));const saturated=rr.reduce((sum,r)=>sum+r.methods[$('method').value].lost_with_raw_ceiling,0);
 $('count').textContent=`${candidates.length} unsaturated 1×-only candidates from ${rr.length} recording${rr.length===1?'':'s'}. ${saturated} additional survival failures also hit the raw ceiling and are excluded.`;
 $('table').innerHTML=candidates.map((s,i)=>`<tr data-index="${i}"><td>${s.key}</td><td>${records.get(s.record).time}</td><td>${number(s.x,1)}, ${number(s.y,1)}</td><td>${number(s.bg_mean)}</td><td>${number(s.bg_percentile,1)}%</td><td>${number(s.signal_1x_mean)}</td><td>${number(s.bg_to_1x)}</td><td>${number(s.r,3)}</td></tr>`).join('');
 const idx=keepKey?candidates.findIndex(s=>s.key===keepKey):-1;inspect(idx>=0?idx:0)}
async function chooseGroup(){const token=++groupToken;++detailToken;bgImage=null;rebuildRecords();$('error').textContent='';$('loaded').textContent='Loading background…';renderList();const g=DATA.groups[Number($('group').value)];try{const im=await loadImage(g.image);if(token!==groupToken)return;bgImage=im;$('map').width=g.width;$('map').height=g.height;sizeMap();drawMap();$('loaded').textContent=`${g.background} · recorded profile at 20× · ${g.width} × ${g.height} sensor pixels`;}catch(e){if(token===groupToken)$('error').textContent=e.message}}
$('group').onchange=chooseGroup;$('method').onchange=()=>renderList();$('record').onchange=()=>renderList();$('sort').onchange=()=>renderList(candidates[active]?.key);$('overlay').onchange=()=>{drawMap();inspect(active)};
$('zoom').onchange=sizeMap;window.addEventListener('resize',sizeMap);$('centre').onclick=()=>{const s=candidates[active];if(!s)return;const v=$('viewport'),c=$('map');v.scrollLeft=s.x/c.width*c.clientWidth-v.clientWidth/2;v.scrollTop=s.y/c.height*c.clientHeight-v.clientHeight/2};
$('previous').onclick=()=>inspect(active-1);$('next').onclick=()=>inspect(active+1);$('spot').onchange=()=>inspect(Number($('spot').value));
$('map').onclick=e=>{if(!candidates.length||!$('overlay').checked)return;const box=e.currentTarget.getBoundingClientRect(),c=$('map'),x=(e.clientX-box.left)*c.width/box.width,y=(e.clientY-box.top)*c.height/box.height;const hits=candidates.map((s,i)=>({i,d:Math.hypot(s.x-x,s.y-y)})).filter(v=>v.d<=24).sort((a,b)=>a.d-b.d);if(!hits.length)return;let next=hits[0].i;if(hits.length>1&&hits.some(h=>h.i===active)){const at=hits.findIndex(h=>h.i===active);next=hits[(at+1)%hits.length].i}inspect(next)};
$('table').onclick=e=>{const tr=e.target.closest('tr[data-index]');if(tr)inspect(Number(tr.dataset.index))};
document.addEventListener('keydown',e=>{if(['SELECT','INPUT','TEXTAREA'].includes(e.target.tagName)||!candidates.length)return;if(e.key==='ArrowRight'){e.preventDefault();inspect(active+1)}if(e.key==='ArrowLeft'){e.preventDefault();inspect(active-1)}});
$('download').onclick=()=>{const fields=['key','record','x','y','bg_mean','bg_percentile','signal_1x_mean','bg_to_1x','bg_ring_median','r','raw_peak','nearest_2x'];const rows=[fields,...candidates.map(s=>fields.map(k=>s[k]??''))];const csv=rows.map(r=>r.join(',')).join('\r\n');const u=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=u;a.download=`survival_failures_bg${Number($('group').value)+1}_${$('method').value}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
chooseGroup();
</script></body></html>'''

if __name__=='__main__': build()
