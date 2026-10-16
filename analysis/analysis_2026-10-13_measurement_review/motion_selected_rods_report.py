"""Local, self-contained inspector for dense motion-selected background features."""
import json,html,csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from motion_selected_rods import OUT,PREV,STAMPS,LABELS,read,write,MODEL

MAIN=['183650','185032','192157','210358']
CHOICES=['first','middle','dimmest'];COLORS=['#176d91','#7b8893','#c36225']

def finite(x):return x is not None and np.isfinite(x)
def selected(s,bg,pick,key):
    p=s['series'][bg]['selected'][pick]
    return p[key] if p else None

def stats(d):
    out={}
    for bg in ['minimum','masked_median']:
        out[bg]={}
        for pick in CHOICES:
            kept=[s for s in d['spots'] if s['statistics_kept']]
            th=np.array([selected(s,bg,pick,'theta') for s in kept],float)
            r=np.array([selected(s,bg,pick,'r') for s in kept],float)
            t=th[np.isfinite(th)]
            out[bg][pick]=dict(valid_theta=len(t),valid_r=int(np.isfinite(r).sum()),median_theta=float(np.median(t)) if len(t) else None,
                near30_percent=float(100*np.mean((t>=28)&(t<34))) if len(t) else None,above_r_limit=int(np.sum(r>MODEL['r_max'])))
    return out

def compact(v):
    if isinstance(v,float):return round(v,5) if np.isfinite(v) else None
    if isinstance(v,dict):return {k:compact(x) for k,x in v.items()}
    if isinstance(v,list):return [compact(x) for x in v]
    return v

def figures(all_data):
    for bg in ['minimum','masked_median']:
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,layout='constrained')
        for ax,stamp in zip(axes.ravel(),MAIN):
            d=all_data[stamp]
            # Paired available theta population, to avoid changed denominators between curves.
            rows=[s for s in d['spots'] if s['statistics_kept'] and all(finite(selected(s,bg,p,'theta')) for p in CHOICES)]
            for pick,col in zip(CHOICES,COLORS):
                t=np.array([selected(s,bg,pick,'theta') for s in rows]);h,e=np.histogram(t,np.arange(0,92,2))
                ax.stairs(100*h/max(1,len(t)),e,color=col,lw=1.5 if pick!='middle' else 1,ls=':' if pick=='middle' else '-',label=f'{pick.capitalize()} · median {np.median(t):.1f}°')
            ax.axvspan(28,34,color='#e9c678',alpha=.2);ax.set(title=f'{LABELS[stamp]} · same {len(rows):,} candidates',xlim=(0,90),xlabel='Recovered θ (degrees from optical axis)',ylabel='% of paired valid candidates / 2°');ax.legend(fontsize=9)
        fig.suptitle(('Original minimum background' if bg=='minimum' else 'Earlier moving-feature-masked median background')+'\nOne original frame per candidate; motion selects features, not molecular identity',fontsize=13)
        fig.savefig(OUT/f'theta_{bg}.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,layout='constrained')
    for ax,stamp in zip(axes.ravel(),MAIN):
        d=all_data[stamp];rows=[s for s in d['spots'] if s['statistics_kept'] and all(finite(selected(s,'minimum',p,'r')) for p in ['first','dimmest'])]
        for pick,col in [('first',COLORS[0]),('dimmest',COLORS[2])]:
            r=np.array([selected(s,'minimum',pick,'r') for s in rows]);h,e=np.histogram(r,np.arange(0,1.42001,.02));ax.stairs(100*h/max(1,len(r)),e,color=col,label=f'{pick.capitalize()} · {np.sum(r>MODEL["r_max"])} above r limit')
        ax.axvline(MODEL['r_max'],color='.45',ls='--');ax.set(title=f'{LABELS[stamp]} · same {len(rows):,} candidates',xlabel='Anisotropy radius r (no physical-limit clipping)',ylabel='% of paired valid candidates / 0.02',xlim=(0,1.42));ax.legend(fontsize=9)
    fig.suptitle('Original minimum background · r from first vs dimmest real frame',fontsize=13);fig.savefig(OUT/'radius_minimum.png',dpi=145);plt.close(fig)
    # Systematic examples: old-detected and new candidates at evenly spaced score ranks.
    for stamp in STAMPS:
        d=all_data[stamp];pose=np.array(d['pose']);fis=[0,d['frames']//2,d['frames']-1];example_rows=[]
        for new in [False,True]:
            pool=[s for s in d['spots'] if s['statistics_kept'] and s['never_old_detected']==new]
            example_rows.extend(pool[i] for i in np.linspace(0,len(pool)-1,4,dtype=int))
        fig,axes=plt.subplots(8,3,figsize=(8,17),layout='constrained')
        ims=[np.asarray(Image.open(PREV/stamp/f'f{fi+1:02}.jpg')) for fi in fis]
        for i,s in enumerate(example_rows):
            for k,fi in enumerate(fis):
                ax=axes[i,k];x,y=(np.array([s['x'],s['y']])+pose[fi])/2;xx,yy=int(round(x)),int(round(y));im=ims[k]
                # Constant-width crop, including zero border if near the field edge.
                padded=np.pad(im,20);crop=padded[yy+4:yy+36,xx+4:xx+36]
                ax.imshow(crop,cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.add_patch(plt.Rectangle((11.5,11.5),9,9,fill=False,edgecolor='#58e7d0',lw=.6));ax.axis('off')
                ax.set_title(f'#{s["id"]} · {"new" if s["never_old_detected"] else "old + motion"} · score {s["score"]:.2f} · F{fi+1}',fontsize=8)
        fig.suptitle(LABELS[stamp]+' · fixed-path 64×64 sensor-pixel crops\nMinimum BG subtraction; fixed 20× display; 18×18 extraction box\nFour evenly spaced score ranks per group, not hand-picked examples',fontsize=11)
        fig.savefig(OUT/stamp/'examples.png',dpi=120);plt.close(fig)

HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Motion-selected features</title>
<style>
:root{font:16px/1.5 system-ui;color:#243237;background:#f3f5f3}body{max-width:1400px;margin:auto;padding:24px}h1{font-size:29px;margin:8px 0}h2{font-size:22px}p{max-width:1000px}.card{background:white;border:1px solid #d9dfdb;padding:20px;margin:18px 0;border-radius:7px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:10px 0}label{display:inline-flex;gap:6px;align-items:center}select,input,button{font:inherit;padding:6px}button{cursor:pointer}a{color:#176d91}small,.muted{color:#587078}.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}.three{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}canvas{max-width:100%;height:auto}#map{background:#111;cursor:crosshair;width:100%}#crop{width:256px;image-rendering:pixelated;background:#111}.plot{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:7px;border-bottom:1px solid #ddd}details{margin:14px 0}summary{cursor:pointer;color:#176d91}.legend span{margin-right:15px}.note{border-left:4px solid #b98336;padding-left:12px}#list{max-height:340px;overflow:auto}.selected{background:#fff3c9}tbody tr{cursor:pointer}#summaryimg{max-width:100%} .tiny{font-size:13px}.warning{color:#a13b2a}@media(max-width:850px){.two,.three{grid-template-columns:1fr}body{padding:12px}}
</style><body>
<h1>Features selected by following the stage</h1>
<p>These are the <b>original moving-sample background captures</b>, not the stationary survey frame stacks. No single-frame brightness or DoG threshold is required. The search asks whether a feature follows the measured stage path in both discovery and validation frames.</p>
<p class="note"><b>Motion is useful evidence, not a rod label.</b> Sample-fixed dust, surface texture, coherent scattering or interference structure that moves with the sample can also pass. The dimmest-frame choice below is a test: it can select lower-angle states, noise dips or tracking errors, rather than the best background correction.</p>
<div class="controls"><a href="#inspect">Inspect every candidate</a> · <a href="#distributions">Distribution summary</a> · <a href="#method">Method and limits</a> · <a href="summary.json">Numerical summary</a></div>
<div class="card" id="inspect"><h2>Inspect the motion candidates</h2>
<div class="controls"><label>Capture <select id="capture">__OPTIONS__</select></label><label>Show <select id="filter"><option value="all">All candidates</option><option value="new">Never found by old k4 detector</option><option value="old">Found by old k4 in ≥1 frame</option><option value="kept">Eligible for statistics</option><option value="excluded">Saturation / overlap exclusions</option></select></label><label>Minimum motion score <input id="score" type="number" value="0" min="0" max="1" step="0.05" style="width:65px"></label></div>
<p id="counts"></p><div class="controls"><label>Image <select id="view"><option value="frame">Individual frame · minimum BG subtracted</option><option value="coherence">Motion score · sample coordinates</option></select></label><label><input id="markers" type="checkbox" checked>Candidate outlines</label><label><input id="oldmarkers" type="checkbox">Old k4 detections</label><label><input id="trail" type="checkbox" checked>Selected path</label><label>Display gain <select id="gain"><option value="1">20×</option><option value="2">40×</option><option value="4">80×</option></select></label></div>
<div class="controls"><button id="play">Play</button><input id="frame" type="range" min="1" max="45" value="1" style="width:min(550px,65vw)"><span id="frameLabel"></span></div>
<p class="tiny legend"><span style="color:#158a77">● Found by motion + old detector</span><span style="color:#b3870c">● New to motion search</span><span style="color:#bf233a">● Raw-saturated</span><span style="color:#8752b9">● Overlapping aperture (statistics excluded)</span>Selected = white; dashed grey = old per-frame detections.</p>
<canvas id="map" width="1232" height="1028"></canvas><p class="tiny">Click a marker, or anywhere to select the nearest candidate. Markers move with the stage; the coherence view stays in sample coordinates. No old red-circle or low-background mask is applied. Display images average the four analyser pixels for visibility; angle extraction uses the original sensor pixels.</p>
<div class="controls"><button id="prev">Previous spot</button><button id="next">Next spot</button><label>Spot ID <input id="spotid" type="number" min="1" value="1" style="width:80px"></label><button id="go">Go</button><button id="jumpfirst">First frame</button><button id="jumpmiddle">Middle frame</button><button id="jumpmin">Dimmest frame</button><label>Angle background <select id="bg"><option value="minimum">Original minimum</option><option value="masked_median">Earlier masked median (sensitivity)</option></select></label></div>
<div class="two"><div><canvas id="crop" width="256" height="256"></canvas><p id="spotinfo"></p></div><div><canvas class="plot" id="trace" width="680" height="300"></canvas><canvas class="plot" id="anisotropy" width="680" height="300"></canvas></div></div>
<p class="tiny">Crop: 64×64 sensor pixels, following the selected path; white box: 18×18 extraction. The photograph always uses <b>original minimum subtraction</b>. The angle-background selector changes numerical extraction only. Higher display gains amplify the existing 20× JPEG preview and cannot recover clipped highlight detail.</p>
<h2>Distribution of the displayed, eligible candidates</h2><div class="two"><canvas class="plot" id="theta" width="680" height="320"></canvas><canvas class="plot" id="radius" width="680" height="320"></canvas></div><p id="histinfo" class="tiny"></p>
<p class="tiny">Blue: first frame; orange: dimmest frame. Each curve uses one original frame per candidate. Both curves use the same candidates with finite values in both choices. Saturated and overlapping apertures are always excluded from statistics. θ uses 2° bins; r uses 0.02 bins and retains values above r<sub>sat</sub>. The filters above also update these histograms.</p>
<details><summary>All displayed candidates · click a row to inspect</summary><div id="list"><table><thead><tr><th>ID</th><th>Motion score</th><th>Old detections</th><th>Status</th><th>First θ</th><th>Dimmest θ</th></tr></thead><tbody id="rows"></tbody></table></div></details>
<details><summary>Wrong-path controls and systematic example crops</summary><p id="controlinfo"></p><img id="detectionimg" class="plot"><img id="examplesimg" class="plot"></details>
<p><a id="jsonlink">All per-frame values and provenance</a> · <a id="csvlink">One-row-per-candidate CSV</a></p></div>
<div class="card" id="distributions"><h2>Compare captures, keeping acquisition groups separate</h2><p>__SUMMARY__</p>__TABLE__
<div class="controls"><label>Plot <select id="summarychoice"><option value="theta_minimum.png">θ · original minimum background</option><option value="theta_masked_median.png">θ · masked median background</option><option value="radius_minimum.png">r · original minimum background</option></select></label></div><img id="summaryimg" src="theta_minimum.png">
<p class="tiny">The initial 18:01 test remains available in the inspector but is not pooled with the later hinge measurements. Captures need not be independent or unique populations; repeat appearances within a capture are counted once. These are observed detection-conditioned histograms, without solid-angle or brightness-selection correction.</p></div>
<div class="card" id="method"><h2>What was done</h2>
<ol><li>Use the previously estimated cumulative translation in each background capture. Only five captures have a supported path between every adjacent real frame. Search the common field visible throughout that path (about 88–95% of the sensor). The synthetic final phase-marker frame is omitted.</li>
<li>For detection only, average aligned 2×2 sensor cells, smooth with σ = 2 sensor pixels, and remove the temporal median at each fixed sensor location. Scale by that location’s temporal MAD, then bound the signed evidence with tanh so one bright frame cannot dominate.</li>
<li>Shift this evidence along the measured stage path. Odd frames locate peaks of the registered signed signal without an intensity cutoff. Both odd-frame and even-frame motion scores at that location must exceed the capture threshold. A score is <b>not a probability</b>. The stage path itself was estimated using the full movie, so the odd/even split is an internal consistency check, not a fully independent test.</li>
<li>Run the same search with five shuffled time orders of the stage positions. The threshold is the larger of 0.30 and the 99.9th percentile of the controls’ minimum odd/even score at candidate peaks. These controls check accidental path matches, not molecular identity or a calibrated false-discovery rate. A zero-motion path is an additional check. Suppress duplicate candidate centres closer than 12 sensor pixels.</li>
<li>For every accepted centre, follow the prescribed path through the <b>original unwarped frames</b>. Extract the four analyser-channel means in an 18×18 aperture with correct sensor parity. Subtract the chosen background, clip at zero and round as in the acquisition calculation. Compute X, Y, r and θ using the existing Fresnel/BFP/NA-in 0.39 annular model (r<sub>sat</sub> = 0.92768).</li>
<li>Exclude a track from statistics if any original raw pixel in its 18×18 aperture reaches 4094 in any real frame. For apertures that overlap, retain the stronger motion score for statistics. All such excluded candidates remain visible. Dimmest means the frame with the smallest mean of all four background-subtracted channels, choosing <b>one frame shared by all channels</b>.</li></ol>
<p class="note">The original minimum is biased low by noise. The earlier masked median is a sensitivity check, not a certified truth: it masked the previous brighter detections, so newly recovered weak moving features may remain in that estimate. Invalid masked-background regions are not filled. Neither subtraction establishes an unbiased physical angle for a weak feature. Selecting the minimum-intensity frame can add another bias.</p>
<p>Five other captures (19:07:29, 19:19:03, 19:41:41, 20:19:02 and 20:52:23) lack a reliable full-capture path and are not assigned invented trajectories. The method can miss objects with little measurable contrast, objects that move independently, or features outside the common field. Shared stage motion cannot distinguish an actual rod from other sample-associated structure.</p>
<p><a href="validation.json">Synthetic warp-direction / motion-discrimination check</a> · <a href="../motion_selected_rods.py">Analysis source</a> · <a href="../motion_selected_rods_report.py">Report builder</a></p></div>
<script>window.CAPTURES={};</script><script src="183650/view.js"></script><script src="viewer.js"></script></body></html>'''

JS=r'''"use strict";
const $=id=>document.getElementById(id),number=(v,d=2)=>Number.isFinite(v)?v.toFixed(d):'undefined';
let D=CAPTURES['183650'],selectedID=1,frame=0,timer=null,loadToken=0;
const images=new Map();
function imageFor(src){if(!images.has(src)){const im=new Image();im.onload=()=>{draw();};im.onerror=()=>{$('frameLabel').textContent='Image could not load: '+src;};im.src=src;images.set(src,im);}return images.get(src);}
function rows(){const f=$('filter').value,k=Number($('score').value)||0;return D.spots.filter(s=>s.score>=k&&(f==='all'||f==='new'&&s.never_old_detected||f==='old'&&!s.never_old_detected||f==='kept'&&s.statistics_kept||f==='excluded'&&!s.statistics_kept));}
function status(s){return s.saturated?'Raw saturated':s.overlap_with?'Overlap with #'+s.overlap_with:'Statistics eligible';}
function color(s){return s.saturated?'#ff4c69':s.overlap_with?'#c28dff':s.never_old_detected?'#ffe18a':'#62e4cd';}
function spot(){return D.spots.find(s=>s.id===selectedID);}
function series(s){return s.series[$('bg').value];}
function sel(s,p,key){return series(s).selected[p]?.[key]??null;}
function selectedLocation(s,fi=frame){return [s.x+D.pose[fi][0],s.y+D.pose[fi][1]];}
function frameImage(fi=frame){return imageFor('../background_motion_check/'+D.stamp+'/f'+String(fi+1).padStart(2,'0')+'.jpg');}
function setFrame(fi){frame=Math.max(0,Math.min(D.frames-1,fi));$('frame').value=frame+1;draw();details();}
function draw(){
 const c=$('map'),g=c.getContext('2d'),co=$('view').value==='coherence',im=co?imageFor(D.stamp+'/coherence.png'):frameImage();g.clearRect(0,0,c.width,c.height);g.fillStyle='#111';g.fillRect(0,0,c.width,c.height);
 if(im.complete&&im.naturalWidth){g.filter=co?'none':'brightness('+$('gain').value+')';g.drawImage(im,0,0,c.width,c.height);g.filter='none';}
 if($('oldmarkers').checked&&!co){g.strokeStyle='#ccc';g.setLineDash([3,2]);for(const p of D.old[frame]){g.beginPath();g.arc(p[0]/2,p[1]/2,5,0,Math.PI*2);g.stroke();}g.setLineDash([]);}
 if($('markers').checked){for(const s of rows()){const p=co?[s.x,s.y]:selectedLocation(s);g.strokeStyle=color(s);g.lineWidth=.9;g.beginPath();g.arc(p[0]/2,p[1]/2,4,0,2*Math.PI);g.stroke();}}
 const s=spot();if(s){const p=co?[s.x,s.y]:selectedLocation(s);if(!co&&$('trail').checked){g.beginPath();D.pose.forEach((z,i)=>{let x=(s.x+z[0])/2,y=(s.y+z[1])/2;i?g.lineTo(x,y):g.moveTo(x,y);});g.strokeStyle='#fff9';g.lineWidth=1;g.stroke();}g.strokeStyle='white';g.lineWidth=1.5;g.strokeRect(p[0]/2-6,p[1]/2-6,12,12);g.fillStyle='white';g.font='14px system-ui';g.fillText('#'+s.id,p[0]/2+9,p[1]/2-9);}
 $('frameLabel').textContent=`Frame ${frame+1} / ${D.frames} · stage Δx ${D.pose[frame][0].toFixed(1)}, Δy ${D.pose[frame][1].toFixed(1)} sensor px`;
 crop();
}
function crop(){const c=$('crop'),g=c.getContext('2d'),s=spot();g.fillStyle='#111';g.fillRect(0,0,256,256);if(!s)return;const p=selectedLocation(s),im=frameImage();g.imageSmoothingEnabled=false;if(im.complete&&im.naturalWidth){g.filter='brightness('+$('gain').value+')';g.drawImage(im,p[0]/2-16,p[1]/2-16,32,32,0,0,256,256);g.filter='none';}g.strokeStyle='white';g.lineWidth=1;g.strokeRect(92,92,72,72);}
function plot(id,curves,xmax,xlabel,ylabel,opts={}){
 const c=$(id),g=c.getContext('2d'),W=c.width,H=c.height,L=58,R=18,T=32,B=48;g.clearRect(0,0,W,H);g.fillStyle='white';g.fillRect(0,0,W,H);
 const vals=curves.flatMap(a=>a.points.map(p=>p[1]).filter(Number.isFinite));let lo=opts.lo??Math.min(0,...vals),hi=opts.hi??Math.max(1,...vals)*1.08;if(hi<=lo)hi=lo+1;
 const xx=x=>L+x/xmax*(W-L-R),yy=y=>H-B-(y-lo)/(hi-lo)*(H-T-B);g.strokeStyle='#c5cece';g.fillStyle='#53666a';g.font='12px system-ui';
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;g.beginPath();g.moveTo(L,yy(v));g.lineTo(W-R,yy(v));g.stroke();g.fillText(number(v,hi-lo<5?2:0),3,yy(v)+4);}
 for(let i=0;i<=5;i++){let v=xmax*i/5;g.fillText(number(v,xmax<5?2:0),xx(v)-10,H-B+20);}
 if(opts.band){g.fillStyle='#efcf8b44';g.fillRect(xx(opts.band[0]),T,xx(opts.band[1])-xx(opts.band[0]),H-T-B);}
 if(opts.vline!==undefined){g.setLineDash([4,4]);g.strokeStyle='#555';g.beginPath();g.moveTo(xx(opts.vline),T);g.lineTo(xx(opts.vline),H-B);g.stroke();g.setLineDash([]);}
 curves.forEach(a=>{g.strokeStyle=a.color;g.lineWidth=1.5;g.beginPath();let started=false;for(const [x,y] of a.points){if(!Number.isFinite(y)){started=false;continue;}started?g.lineTo(xx(x),yy(y)):g.moveTo(xx(x),yy(y));started=true;}g.stroke();});
 g.fillStyle='#243237';g.fillText(xlabel,L,H-8);g.fillText(ylabel,L,17);let lx=L;for(const a of curves){g.fillStyle=a.color;g.fillText(a.name,lx,H-27);lx+=Math.min(260,30+a.name.length*6.4);}
}
function details(){const s=spot();if(!s)return;const q=series(s),v=q.selected.dimmest,xy=q.xy[frame],th=q.theta[frame],rr=q.r[frame];
 $('spotid').value=s.id;$('spotinfo').innerHTML=`<b>#${s.id} · ${status(s)}</b><br>Motion score ${number(s.score,3)} (odd ${number(s.train_score,3)}, even ${number(s.validation_score,3)}).<br>Old k4 detections: ${s.old_detection_frames}/${D.frames} frames.<br>Raw maximum along aperture path: ${s.raw_peak}/4095.<br>Current frame: θ ${number(th,1)}°, r ${number(rr,3)}; X ${number(xy?.[0],3)}, Y ${number(xy?.[1],3)}.<br>Mean intensity: ${number(q.intensity[frame])} /255 equivalent.<br>Dimmest: ${v?'frame '+v.frame+', θ '+number(v.theta,1)+'°, mean '+number(v.intensity)+' /255':'unavailable'}.`;
 plot('trace',[{name:'Aperture mean intensity',color:'#176d91',points:q.intensity.map((v,i)=>[i+1,v])}],D.frames,'Frame (one real exposure each)','Intensity · 0–255 equivalent mean / sensor pixel',{vline:frame+1});
 plot('anisotropy',[{name:'X',color:'#176d91',points:q.xy.map((v,i)=>[i+1,v?.[0]])},{name:'Y',color:'#c36225',points:q.xy.map((v,i)=>[i+1,v?.[1]])}],D.frames,'Frame','Single-frame anisotropy',{lo:-1,hi:1,vline:frame+1});
}
function histData(ss,key,bin,xmax,pick){const n=Math.round(xmax/bin),h=Array(n).fill(0);for(const s of ss){const v=sel(s,pick,key);let b=Math.floor(v/bin);if(b>=0&&b<n)h[b]++;}const points=[];h.forEach((v,i)=>{points.push([i*bin,100*v/Math.max(1,ss.length)],[(i+1)*bin,100*v/Math.max(1,ss.length)]);});return points;}
function histograms(){const kept=rows().filter(s=>s.statistics_kept),bg=$('bg').value;const paired=k=>kept.filter(s=>Number.isFinite(sel(s,'first',k))&&Number.isFinite(sel(s,'dimmest',k)));let th=paired('theta'),rr=paired('r');
 for(const [id,key,bin,xmax,ss] of [['theta','theta',2,90,th],['radius','r',.02,1.42,rr]]){plot(id,[{name:'First',color:'#176d91',points:histData(ss,key,bin,xmax,'first')},{name:'Dimmest',color:'#c36225',points:histData(ss,key,bin,xmax,'dimmest')}],xmax,id==='theta'?'Recovered θ (degrees)':'r · no physical-limit clipping','% of paired finite candidates / bin',id==='theta'?{band:[28,34]}:{vline:D.model.r_max});}
 $('histinfo').textContent=`${bg==='minimum'?'Original minimum':'Earlier masked median'}: ${kept.length} eligible displayed candidates; paired finite θ n=${th.length}; paired finite r n=${rr.length}. Above r limit: first ${kept.filter(s=>sel(s,'first','r')>D.model.r_max).length}, dimmest ${kept.filter(s=>sel(s,'dimmest','r')>D.model.r_max).length}. Dimmest is selected separately for each background, across frames with finite background values.`;
}
function table(){const ss=rows();$('rows').innerHTML=ss.map(s=>`<tr data-id="${s.id}"><td>${s.id}</td><td>${number(s.score,3)}</td><td>${s.old_detection_frames}/${D.frames}</td><td>${status(s)}</td><td>${number(sel(s,'first','theta'),1)}°</td><td>${number(sel(s,'dimmest','theta'),1)}°</td></tr>`).join('');}
function refresh(){const ss=rows();if(!ss.some(s=>s.id===selectedID)&&ss.length)selectedID=ss[0].id;const z=D.summary;
 $('counts').textContent=`${ss.length.toLocaleString()} displayed / ${z.candidates.toLocaleString()} total motion candidates · ${z.never_old_detected.toLocaleString()} never found by old k4 · ${z.statistics_kept.toLocaleString()} eligible for statistics · ${z.saturated} saturated · ${z.overlap_excluded} overlapping. Search area ${z.common_sensor_area_percent.toFixed(1)}%.`;
 $('controlinfo').textContent=`Accepted real-path candidates: ${z.candidates}. Shuffled-time path counts (five trials): ${z.wrong_path_counts.join(', ')}. Stationary-path count: ${z.stationary_path_count}. Score threshold: ${z.threshold.toFixed(3)}. Control counts are before the 12-pixel duplicate suppression, so are conservative comparisons; not a formal false-discovery rate.`;
 draw();details();histograms();table();
}
async function loadCapture(stamp){clearInterval(timer);timer=null;$('play').textContent='Play';const token=++loadToken;if(!CAPTURES[stamp]){await new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=stamp+'/view.js';s.onload=resolve;s.onerror=reject;document.body.appendChild(s);}).catch(()=>{$('counts').textContent='Could not load capture data.';});}if(token!==loadToken||!CAPTURES[stamp])return;D=CAPTURES[stamp];selectedID=1;frame=0;images.clear();$('frame').max=D.frames;$('frame').value=1;$('spotid').max=D.spots.length;$('detectionimg').src=stamp+'/detection.png';$('examplesimg').src=stamp+'/examples.png';$('jsonlink').href=stamp+'/results.json';$('csvlink').href=stamp+'/candidates.csv';refresh();}
$('capture').addEventListener('change',()=>loadCapture($('capture').value));for(const id of ['filter','score'])$(id).addEventListener('change',refresh);for(const id of ['view','markers','oldmarkers','trail','gain'])$(id).addEventListener('change',draw);
$('bg').addEventListener('change',()=>{details();histograms();table();});$('frame').addEventListener('input',()=>setFrame(Number($('frame').value)-1));$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play';}else{timer=setInterval(()=>setFrame((frame+1)%D.frames),240);$('play').textContent='Pause';}};
function advance(delta){const ss=rows();if(!ss.length)return;const i=ss.findIndex(s=>s.id===selectedID);selectedID=ss[(i+delta+ss.length)%ss.length].id;draw();details();}
$('prev').onclick=()=>advance(-1);$('next').onclick=()=>advance(1);$('go').onclick=()=>{const n=Number($('spotid').value);if(D.spots.some(s=>s.id===n)){selectedID=n;draw();details();}};
$('jumpfirst').onclick=()=>setFrame(0);$('jumpmiddle').onclick=()=>setFrame(Math.floor(D.frames/2));$('jumpmin').onclick=()=>{const s=spot(),f=s&&series(s).selected.dimmest?.frame;if(f)setFrame(f-1);};
$('map').addEventListener('click',e=>{const r=$('map').getBoundingClientRect(),x=(e.clientX-r.left)*2464/r.width,y=(e.clientY-r.top)*2056/r.height,co=$('view').value==='coherence';let best=null,bd=Infinity;for(const s of rows()){const p=co?[s.x,s.y]:selectedLocation(s),d=(p[0]-x)**2+(p[1]-y)**2;if(d<bd){bd=d;best=s;}}if(best){selectedID=best.id;draw();details();}});
$('rows').addEventListener('click',e=>{const tr=e.target.closest('tr[data-id]');if(tr){selectedID=Number(tr.dataset.id);draw();details();}});$('summarychoice').addEventListener('change',()=>{$('summaryimg').src=$('summarychoice').value;});
loadCapture('183650');
'''

def main():
    data={stamp:read(OUT/stamp/'results.json') for stamp in STAMPS}
    summaries=[]
    for stamp,d in data.items():
        d['statistics']=stats(d);summaries.append(dict(stamp=stamp,label=LABELS[stamp],counts=d['summary'],statistics=d['statistics']))
        old=read(PREV/stamp/'results.json');payload={k:d[k] for k in ['stamp','label','frames','pose','spots','model','summary']};payload['old']=[[[s['x'],s['y']] for s in f['spots']] for f in old['frames']]
        for s in payload['spots']:
            s.pop('neighbours',None);s.pop('path_evidence',None)
        (OUT/stamp/'view.js').write_text('window.CAPTURES['+json.dumps(stamp)+']='+json.dumps(compact(payload),ensure_ascii=False,allow_nan=False,separators=(',',':'))+';',encoding='utf8')
        with (OUT/stamp/'candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
            fields=['id','x','y','score','old_detection_frames','never_old_detected','raw_peak','saturated','overlap_with','statistics_kept']+[f'{bg}_{pick}_{key}' for bg in ['minimum','masked_median'] for pick in CHOICES for key in ['frame','intensity','r','theta']]
            wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
            for s in d['spots']:
                row={k:s[k] for k in fields if k in s}
                for bg in ['minimum','masked_median']:
                    for pick in CHOICES:
                        vals=s['series'][bg]['selected'][pick]
                        for key in ['frame','intensity','r','theta']:row[f'{bg}_{pick}_{key}']=vals[key] if vals else None
                wr.writerow(row)
    write(OUT/'summary.json',summaries);figures(data)
    options=''.join(f'<option value="{s}"'+(' selected' if s=='183650' else '')+'>'+html.escape(LABELS[s])+f' · {s[:2]}:{s[2:4]}:{s[4:]}</option>' for s in ['183650','185032','192157','210358','180149'])
    table='<table><thead><tr><th>Capture</th><th>Motion candidates</th><th>Never old-detected</th><th>Eligible</th><th>Wrong paths</th><th>First → dimmest median θ<br>(minimum BG)</th></tr></thead><tbody>'
    for stamp in MAIN:
        d=data[stamp];z=d['summary'];st=d['statistics']['minimum'];table+=f'<tr><td>{html.escape(LABELS[stamp])}</td><td>{z["candidates"]:,}</td><td>{z["never_old_detected"]:,}</td><td>{z["statistics_kept"]:,}</td><td>{min(z["wrong_path_counts"])}–{max(z["wrong_path_counts"])}</td><td>{st["first"]["median_theta"]:.1f}° → {st["dimmest"]["median_theta"]:.1f}°</td></tr>'
    table+='</tbody></table>'
    summary='The stage path finds far more candidates than the shuffled-time controls, including many weak features missed by the earlier per-frame threshold. <b>Their angle distribution is still background-dependent.</b> Choosing the dimmest frame lowers the median θ in every capture with the original minimum background; it does not establish that the resulting θ is more accurate.'
    (OUT/'index.html').write_text(HTML.replace('__OPTIONS__',options).replace('__TABLE__',table).replace('__SUMMARY__',summary),encoding='utf8');(OUT/'viewer.js').write_text(JS,encoding='utf8')
    print('Report:',OUT/'index.html',flush=True)

if __name__=='__main__':main()
