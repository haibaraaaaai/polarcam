"""Track translating structure in original raw background captures (read only).

No theta statistics or acquisition/GUI changes. Detection is on separate frames.
The temporal median is a registration reference, not an installed background.
"""
from pathlib import Path
import argparse, hashlib, json, html
import numpy as np
import cv2
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
DAY=HERE.parents[1]/'data/hinges/polarcam/2026-10-12'
OUT=HERE/'background_motion_check'
SCALE=255/4095
cv2.setNumThreads(3)

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def write(p,v):p.write_text(json.dumps(clean(v),ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def cells(a):
    a=np.asarray(a,np.float32)
    return (a[::2,::2]+a[1::2,::2]+a[::2,1::2]+a[1::2,1::2])*.25
def dog(a):
    return cv2.GaussianBlur(a,(0,0),.5,borderType=cv2.BORDER_REPLICATE)-cv2.GaussianBlur(a,(0,0),3,borderType=cv2.BORDER_REPLICATE)
def detect(d):
    threshold=max(0,float(d.mean()+4*d.std()))
    count,labels,st,cent=cv2.connectedComponentsWithStats((d>threshold).astype('uint8'),8)
    out=[]
    for i in range(1,count):
        x,y,w,h,area=st[i];cx,cy=cent[i]
        if area<3 or not(5<cx<d.shape[1]-5 and 5<cy<d.shape[0]-5):continue
        out.append(dict(x=2*float(cx)+.5,y=2*float(cy)+.5,bbox=[int(2*x),int(2*y),int(2*w),int(2*h)],area=int(area*4),
                        peak=float(d[y:y+h,x:x+w].max())))
    return out,threshold
def xy(rows):return np.array([[r['x'],r['y']] for r in rows],float).reshape(-1,2)
def matches(a,b,shift,radius=8):
    if len(a)==0 or len(b)==0:return np.empty((0,2),int)
    dist,j=cKDTree(b).query(a+shift)
    _,ii=cKDTree(a+shift).query(b)
    keep=(dist<=radius)&(ii[j]==np.arange(len(a)))
    return np.column_stack([np.where(keep)[0],j[keep]])
def pair_motion(da,db,pa,pb,window):
    # Clip extreme cores; positive temporal-contrast DoG suppresses fixed structure.
    aa=np.maximum(da,0);bb=np.maximum(db,0)
    aa=np.minimum(aa,max(1,float(np.percentile(aa,99.9))));bb=np.minimum(bb,max(1,float(np.percentile(bb,99.9))))
    shift,response=cv2.phaseCorrelate(aa,bb,window)
    shift=np.asarray(shift)*2
    pairs=matches(pa,pb,shift)
    if len(pairs)>=3:
        delta=pb[pairs[:,1]]-pa[pairs[:,0]];shift=np.median(delta,axis=0)
        pairs=matches(pa,pb,shift)
    delta=pb[pairs[:,1]]-pa[pairs[:,0]] if len(pairs) else np.empty((0,2))
    residual=float(np.median(np.linalg.norm(delta-shift,axis=1))) if len(pairs) else None
    frac=len(pairs)/max(1,min(len(pa),len(pb)))
    supported=len(pairs)>=8 and frac>=.12 and response>=.03 and residual is not None and residual<=4
    return dict(dx=float(shift[0]),dy=float(shift[1]),response=float(response),support=len(pairs),fraction=frac,
                median_error=residual,supported=bool(supported),moving=bool(supported and np.linalg.norm(shift)>=6),pairs=pairs.tolist())

def captures():
    return [p for p in sorted((DAY/'backgrounds').glob('background_min*.json'))
            if read(p).get('review', {}).get('data_status') != 'payload_and_capture_stack_deleted']
def process(path):
    m=read(path);stamp=path.name[24:30];folder=OUT/stamp;folder.mkdir(parents=True,exist_ok=True)
    rawpath=path.parent/m['source_stack_file'];raw=np.load(rawpath,mmap_mode='r');n=m['actual']['frames'];minimum=np.load(path.with_suffix('.npy'))
    assert m['background']['background_subtracted'] is False
    assert len(raw)==n+int(m['actual']['phase_marker_appended'])
    assert np.array_equal(raw[:n].min(0),minimum)
    stack=np.stack([cells(f) for f in raw[:n]])
    median=np.empty(stack.shape[1:],np.float32)
    for y in range(0,median.shape[0],64):median[y:y+64]=np.median(stack[:,y:y+64],axis=0)
    cmin=cells(minimum);dyn=[];frames=[];thresholds=[]
    for i,f in enumerate(stack):
        residual=f-cmin;spots,thr=detect(dog(residual));dynamic=dog(f-median);dr,dt=detect(dynamic)
        frames.append(dict(frame=i+1,spots=spots,dynamic=dr,threshold=thr,dynamic_threshold=dt))
        dyn.append(dynamic)
        Image.fromarray(np.rint(np.clip(residual*20*SCALE,0,255)).astype('uint8')).save(folder/f'f{i+1:02}.jpg',quality=88)
    window=cv2.createHanningWindow((median.shape[1],median.shape[0]),cv2.CV_32F)
    pairs=[]
    for i in range(n-1):
        p=pair_motion(dyn[i],dyn[i+1],xy(frames[i]['dynamic']),xy(frames[i+1]['dynamic']),window)
        p.update(frame_from=i+1,frame_to=i+2)
        # Direct minimum-subtracted registration is a diagnostic for stationary dominance.
        sh,resp=cv2.phaseCorrelate(dog(stack[i]-cmin),dog(stack[i+1]-cmin),window)
        p.update(minimum_response=float(resp),minimum_dx=float(sh[0]*2),minimum_dy=float(sh[1]*2))
        pairs.append(p)
    # Assign dynamic tracks only across independently supported translations.
    nextid=0
    for f in frames:
        for s in f['dynamic']:s['track']=None
    for i,p in enumerate(pairs):
        if not p['supported']:continue
        for a,b in p['pairs']:
            s=frames[i]['dynamic'][a];t=frames[i+1]['dynamic'][b]
            if s['track'] is None:s['track']=nextid;nextid+=1
            t['track']=s['track']
    tracks={}
    for i,f in enumerate(frames):
        for s in f['dynamic']:
            if s['track'] is not None:tracks.setdefault(s['track'],[]).append([i+1,s['x'],s['y']])
    tracks={str(k):v for k,v in tracks.items() if len(v)>=3 and np.linalg.norm(np.ptp(np.array(v)[:,1:],axis=0))>=12}
    trees=[cKDTree(xy(f['spots'])) if f['spots'] else None for f in frames]
    for f in frames:
        pts=xy(f['spots']);occupancy=np.zeros(len(pts),int)
        for tr in trees:
            if tr is not None:occupancy+=tr.query(pts)[0]<=6
        moving=[s for s in f['dynamic'] if str(s['track']) in tracks]
        near=cKDTree(xy(moving)).query(pts)[0]<=10 if moving else np.zeros(len(pts),bool)
        for j,s in enumerate(f['spots']):
            s.update(occupancy=int(occupancy[j]),status='moving' if near[j] else 'persistent' if occupancy[j]>=.8*n else 'unclassified')
    occupied=np.zeros(median.shape,np.uint16)
    for f in frames:
        mask=np.zeros(median.shape,'uint8')
        for s in f['dynamic']:
            if str(s['track']) not in tracks:continue
            x,y,w,h=s['bbox'];cv2.rectangle(mask,(max(0,x//2-6),max(0,y//2-6)),(min(mask.shape[1]-1,(x+w)//2+6),min(mask.shape[0]-1,(y+h)//2+6)),1,-1)
        occupied+=mask
    summary=dict(frames=n,fps=m['actual']['fps'],exposure_ms=m['actual']['timing']['exposure_us']/1000,
       detections_median=float(np.median([len(f['spots']) for f in frames])),dynamic_detections_median=float(np.median([len(f['dynamic']) for f in frames])),
       supported_pairs=sum(p['supported'] for p in pairs),moving_pairs=sum(p['moving'] for p in pairs),track_count=len(tracks),
       minimum_mean_8bit=float(minimum.mean())*SCALE,median_cell_mean_8bit=float(median.mean())*SCALE,
       tracked_mask_area_percent=100*float(np.mean(occupied>0)),tracked_mask_max_frames=int(occupied.max()))
    data=dict(stamp=stamp,profile=path.name,raw_stack=rawpath.name,profile_sha256=sha(path.with_suffix('.npy')),metadata_sha256=sha(path),
       raw_signature=dict(bytes=rawpath.stat().st_size,mtime_ns=rawpath.stat().st_mtime_ns),note=m['conditions'].get('condition_note',''),
       summary=summary,frames=frames,pairs=pairs,tracks=tracks,display='20x, aligned 2x2 cell mean; image pixels cover 2x2 sensor pixels; detection before display gain',
       method='Analysis DoG on aligned 2x2 cell averages: sigma 0.5 and 3 cells (1 and 6 sensor pixels), global mean+4SD, >=3 cells, no upper area cutoff. Minimum-subtracted detections shown. Registration uses positive DoG after temporal cell-median removal. Matched dynamic spots within 8 sensor pixels refine phase-correlation translation; supported pairs require >=8 mutual matches, >=12% of smaller candidate set, phase response>=0.03, median residual<=4 pixels. Moving tracks have >=3 frames and >=12-pixel extent. Persistent means detected within 6 pixels in >=80% of frames, not proven optical background.',
       limits='Tracks are image-feature correspondences, not confirmed individual rods or calibrated stage positions. Weak/fast/blurred features can be missed; constant rods cannot be separated from fixed background. Minimum subtraction alone leaves a positive noise offset. No theta or population statistics changed.')
    write(folder/'results.json',data)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    axes[0].imshow(np.clip((stack[0]-cmin)*20/4095,0,1),cmap='gray',vmin=0,vmax=1,extent=(0,2464,2056,0))
    for tr in tracks.values():
        q=np.array(tr);axes[0].plot(q[:,1],q[:,2],lw=.55,alpha=.6)
    axes[0].set(title=f'{len(tracks)} translating feature tracks',xlim=(0,2464),ylim=(2056,0),xlabel='Sensor x',ylabel='Sensor y')
    for k,color in [('dx','#176d91'),('dy','#b46629')]:
        axes[1].plot([p['frame_to'] for p in pairs],[p[k] if p['supported'] else np.nan for p in pairs],'.-',c=color,label=k)
    axes[1].axhline(0,lw=.5,c='.5');axes[1].set(title='Supported adjacent-frame translations',xlabel='Frame',ylabel='Sensor pixels / frame');axes[1].legend()
    im=axes[2].imshow(occupied,cmap='magma',vmin=0,vmax=n,extent=(0,2464,2056,0));axes[2].set(title='Tracked footprints + 12-pixel margin',xlabel='Sensor x',ylabel='Sensor y');fig.colorbar(im,ax=axes[2],label='Frames marked (lower bound)')
    fig.suptitle(f'Background {stamp} · {summary["supported_pairs"]}/{n-1} supported pairs · {summary["moving_pairs"]} moving',fontsize=13)
    fig.savefig(folder/'overview.png',dpi=145);plt.close(fig)
    print(stamp,summary,flush=True)
    return data

def masked_median(path):
    """Exploratory background from predicted rod paths, preserving raw analyzer pixels."""
    m=read(path);stamp=path.name[24:30];folder=OUT/stamp;d=read(folder/'results.json')
    n=d['summary']['frames'];a=np.load(path.parent/m['source_stack_file'],mmap_mode='r');h,w=a.shape[1:]
    # Only apply this pilot when the entire capture has an independently supported pose chain.
    if not all(p['supported'] for p in d['pairs']):
        d['masked_median']=dict(available=False,reason='At least one adjacent translation is unsupported; no complete predicted-path mask made.')
        write(folder/'results.json',d);return
    pose=np.vstack([np.zeros(2),np.cumsum([[p['dx'],p['dy']] for p in d['pairs']],axis=0)])
    objects=[]
    for key,tr in d['tracks'].items():
        tr=np.array(tr);idx=tr[:,0].astype(int)-1;reference=np.median(tr[:,1:]-pose[idx],axis=0)
        footprint=[s['bbox'][2:] for f in d['frames'] for s in f['dynamic'] if str(s['track'])==key]
        radius=np.max(footprint,axis=0)/2+12
        objects.append(dict(reference=reference,radius=radius,track=key))
    # Merge fragments of the same physical trajectory in stage-compensated coordinates.
    if objects:
        refs=np.array([s['reference'] for s in objects]);parent=list(range(len(refs)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for i,j in cKDTree(refs).query_pairs(8):parent[find(i)]=find(j)
        groups={}
        for i in range(len(refs)):groups.setdefault(find(i),[]).append(i)
        objects=[dict(reference=np.median(refs[ix],axis=0),radius=np.max([objects[j]['radius'] for j in ix],axis=0)) for ix in groups.values()]
    masks=np.zeros((n,h//2,w//2),bool)
    for fi,shift in enumerate(pose):
        for ob in objects:
            centre=ob['reference']+shift;rx,ry=ob['radius'];x,y=centre
            l=max(0,int(np.floor((x-rx)/2)));r=min(w//2,int(np.ceil((x+rx)/2)))
            t=max(0,int(np.floor((y-ry)/2)));b=min(h//2,int(np.ceil((y+ry)/2)))
            if l<r and t<b:masks[fi,t:b,l:r]=True
        # Include unlinked transient candidates too; stationary residuals remain unmasked.
        for s in d['frames'][fi]['dynamic']:
            x,y,ww,hh=s['bbox'];l=max(0,x//2-6);r=min(w//2,(x+ww)//2+7);t=max(0,y//2-6);b=min(h//2,(y+hh)//2+7)
            masks[fi,t:b,l:r]=True
    count=n-masks.sum(0);minimum=np.load(path.with_suffix('.npy'));median=np.empty((h,w),np.float32);estimate=np.empty_like(median)
    min_required=max(10,int(np.ceil(n/4)))
    import warnings
    for y in range(0,h,64):
        data=np.asarray(a[:n,y:y+64,:],np.float32);median[y:y+64]=np.median(data,axis=0)
        mask=np.repeat(np.repeat(masks[:,y//2:(y+64)//2],2,axis=1),2,axis=2)
        data[mask]=np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);estimate[y:y+64]=np.nanmedian(data,axis=0)
    valid=np.repeat(np.repeat(count>=min_required,2,axis=0),2,axis=1);estimate[~valid]=np.nan
    np.save(folder/'masked_median.npy',estimate)
    np.save(folder/'unmasked_frame_count.npy',count.astype('uint8'))
    finite=estimate[np.isfinite(estimate)];change=(estimate-median)[valid]
    info=dict(available=True,trajectory_objects=len(objects),minimum_unmasked_frames=min_required,
        valid_area_percent=100*float(valid.mean()),any_mask_area_percent=100*float(np.mean(count<n)),
        count_quantiles=np.percentile(count,[0,10,25,50,75,90,100]).tolist(),
        minimum_mean_8bit=float(minimum.mean())*SCALE,ordinary_median_mean_8bit=float(median.mean())*SCALE,
        masked_median_mean_8bit=float(finite.mean())*SCALE,
        same_valid_area_minimum_mean_8bit=float(minimum[valid].mean())*SCALE,
        same_valid_area_ordinary_median_mean_8bit=float(median[valid].mean())*SCALE,
        masked_minus_ordinary_median_8bit_quantiles=(np.percentile(change,[1,10,50,90,99])*SCALE).tolist(),
        pose_sensor_pixels=pose.tolist(),
        objects=objects,
        method='Predicted moving-object positions in every frame from the common translation, merged reference centres within 8 pixels. Use maximum observed component half-size plus 12 sensor pixels. Also mask transient dynamic detections. Masks cover complete 2x2 analyzer cells; median taken separately at each original raw pixel. Insufficient unmasked samples become NaN, never silently filled.',
        limits='Exploratory, not rod-free truth. Conditional masking can bias the median downward; dim missed rods and long within-exposure trails can remain. A stationary rod cannot be identified from stationarity alone. 10 fps snapshots do not bound abrupt motion during a 1.194 ms exposure. Mask coverage is not a guarantee of uncontaminated frames.')
    d['masked_median']=info;write(folder/'results.json',d)
    # Spatially equal display scale; NaN regions are pink.
    fig,axes=plt.subplots(1,4,figsize=(14,4.1),layout='constrained')
    cmap=plt.get_cmap('gray').copy();cmap.set_bad('#e7889b')
    vmax=max(1,float(np.percentile(median*SCALE,99.5)))
    for ax,img,title in zip(axes[:3],[minimum,median,estimate],['Stored minimum','Ordinary temporal median','Moving-feature-masked median']):
        im=ax.imshow(img*SCALE,cmap=cmap,vmin=0,vmax=vmax);ax.set_title(title,fontsize=10);ax.set_axis_off()
    fig.colorbar(im,ax=axes[:3],shrink=.7,label='8-bit equivalent / raw pixel')
    im=axes[3].imshow(count,vmin=0,vmax=n,cmap='viridis');axes[3].set_title(f'Unmasked frames; need ≥{min_required}',fontsize=10);axes[3].set_axis_off();fig.colorbar(im,ax=axes[3],shrink=.7)
    fig.suptitle(f'{stamp}: pilot background · {info["valid_area_percent"]:.1f}% of area has enough samples · pink = insufficient',fontsize=12)
    fig.savefig(folder/'median_trial.png',dpi=150);plt.close(fig)
    # Pointwise differences identify where the plain median kept a rod passage.
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained');lim=max(.2,float(np.nanpercentile(np.abs(change*SCALE),99)))
    im=ax.imshow((estimate-median)*SCALE,cmap='coolwarm',vmin=-lim,vmax=lim);ax.set_axis_off();ax.set_title('Masked median − ordinary median');fig.colorbar(im,ax=ax,label='8-bit equivalent / raw pixel');fig.savefig(folder/'median_difference.png',dpi=135);plt.close(fig)
    print('Masked median',stamp,info['valid_area_percent'],info['count_quantiles'],info['masked_minus_ordinary_median_8bit_quantiles'],flush=True)

def render():
    data=[read(p) for p in sorted(OUT.glob('*/results.json'))]
    assert len(data)==10
    labels={'180149':'Initial test background','183650':'Hinges: first wide-field BG','185032':'Hinges: second wide-field BG',
      '190729':'Gold: first wide-field BG','191903':'Gold: wide-field recapture','192157':'Gold: main survey BG',
      '194141':'Gold: narrow-field BG','201902':'Gold: narrow-field, reduced power','205223':'Hinges: narrow-field BG','210358':'Hinges: late wide-field reference'}
    table=''
    for d in data:
        d['label']=labels[d['stamp']];s=d['summary'];q=d.get('masked_median',{});s['label']=d['label']
        area=f'{q["valid_area_percent"]:.1f}%' if q.get('available') else 'Not estimated'
        table+=f'<tr><td><a href="#viewer" onclick="choose(\'{d["stamp"]}\')">{d["stamp"]}</a></td><td>{d["label"]}</td><td>{s["detections_median"]:.0f}</td><td>{s["supported_pairs"]}/{s["frames"]-1}</td><td>{s["track_count"]}</td><td>{area}</td></tr>'
    simple=[dict(stamp=d['stamp'],**d['summary'],masked_median={k:v for k,v in d.get('masked_median',{}).items() if k not in ['objects','pose_sensor_pixels']}) for d in data]
    write(OUT/'summary.json',simple)
    # Avoid duplicating complete component tables in the human-readable report.
    payload=[]
    for d in data:
        payload.append({k:d[k] for k in ['stamp','label','note','summary','masked_median','tracks','frames','pairs']})
    page=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Motion in the original backgrounds</title>
<style>body{margin:0;background:#f6f7f4;color:#263943;font:16px/1.5 system-ui}main{max-width:1280px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:22px}p{max-width:1060px}a{color:#176d91}button,select,input{font:inherit}button,select{padding:5px}table{border-collapse:collapse;font-size:14px}th,td{padding:7px 12px;border-bottom:1px solid #ccd5d5;text-align:left}img{max-width:100%;height:auto}details{margin:14px 0;padding:12px;background:white}summary{cursor:pointer}.note{padding:14px;background:#fff0cb}.small{font-size:14px;color:#50636d}.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:12px 0}#map{width:100%;height:auto;background:black}#crop{width:320px;height:320px;image-rendering:pixelated}#slider{width:min(470px,70vw)}.scroll{overflow:auto}.cropbox{display:flex;gap:24px;align-items:center;flex-wrap:wrap}</style><main>
<h1>Can the original background movies give a better background?</h1>
<p><b>The sample motion is recoverable in the main wide-field background captures.</b> This report follows moving image features and tries a median that excludes their passages. These are candidate background estimates for inspection, not replacements installed in the GUI or used to redraw the theta populations.</p><p>For the first hinge background, all 44 adjacent translations are supported. The recovered path spans approximately 105 × 123 sensor pixels, without imposing a two-direction motion model. Its masked median averages 2.85/255 over supported pixels, compared with 2.19/255 for the minimum and 2.88/255 for the ordinary median. Most of the overall change is minimum → median; tracking mainly changes local regions crossed by rods.</p>
<p class="note">A pixelwise minimum is pulled downward by low noise samples. An ordinary median is less affected by those extremes, but can retain a rod that occupies a location for much of the capture. The proposed mask addresses that occupancy problem. It cannot distinguish a truly stationary rod from stationary optical background, and it cannot guarantee removal of every dim or partially exposed rod.</p>
<div class="scroll"><table><tr><th>Time</th><th>Capture</th><th>Median detections/frame</th><th>Supported frame pairs</th><th>Moving track fragments*</th><th>Area with enough unmasked frames</th></tr>__TABLE__</table></div><p class="small">*Track fragments are not unique rod counts. Supported pairs include pauses with little translation. “Enough” means at least 11 or 12 unmasked samples, depending on capture length. Passing this count criterion is not proof of an uncontaminated estimate. Narrow-field captures can have too few translating features for whole-field tracking.</p>
<h2 id="viewer">Inspect the actual background movie</h2><div class="controls"><select id="capture"></select><button id="prev">Previous</button><button id="play">Play</button><button id="next">Next</button><input id="slider" type="range" min="1" value="1"><span id="frameLabel"></span></div>
<div class="controls"><label><input type="checkbox" id="spots" checked>Minimum-subtracted detections</label><label><input type="checkbox" id="trails" checked>Moving tracks</label><label><input type="checkbox" id="dynamic">Transient detections used for tracking</label></div>
<p id="status"></p><p class="small">Green circles: near a translating track. Orange: detected within 6 pixels at the same location in at least 80% of the capture. Cyan: unclassified. Purple, if enabled: transient candidates. Orange does not mean proven background. Fixed 20× display; bright cores can clip in this display. Detection uses original numbers. Playback is slowed to 4 frames/s for inspection. Click the image to inspect a crop.</p>
<canvas id="map" width="1232" height="1028"></canvas><div class="cropbox"><canvas id="crop" width="320" height="320"></canvas><p id="cropLabel">Click the image for an enlarged crop.</p></div>
<h2>Does the motion form a coherent pattern?</h2><img id="overview"><p>Translations are inferred from matching features that move together, after suppressing persistent structure with a temporal median. They describe sample-image motion in sensor pixels, not a calibrated stage encoder measurement. Unsupported steps are left as gaps. Background pixels stay in sensor coordinates; the raw images are not shifted before estimating background.</p>
<h2>A first masked-median estimate</h2><p id="medianText"></p><img id="medianTrial"><details id="differenceBox"><summary>Where the masked median differs from the ordinary median</summary><img id="difference"></details><p>The fixed bright optical structures should remain in the background estimate so that subtraction can remove them. The aim is to remove moving rods from the estimate, not to make the background image itself dark.</p>
<p>The mask predicts a tracked object’s position in <em>every</em> frame, including frames where its detection is missing. Each footprint uses the maximum detected component extent plus a 12-sensor-pixel margin. Additional transient candidates are masked as well. All four analyzer pixels of a 2×2 cell are masked together, but the background median is calculated separately for each original raw pixel, preserving polarization information. Pink regions have insufficient samples and are saved as NaN.</p>
<p><b>Partial exposures:</b> these movies have a recorded camera rate near 10 fps and exposures of 1.194 ms. Their snapshots can reveal motion paths and some streaking, but cannot bound a sudden movement inside an exposure. The margin and predicted paths are a conservative first trial, not a demonstrated complete blur mask. Masking positive excursions can itself pull the retained median downward.</p>
<details><summary>Exact detector, tracking criteria and limitations</summary><p>Analysis detector from the same DoG family as the GUI, not an unchanged GUI call: aligned non-overlapping 2×2 means; Gaussian widths 0.5 and 3 cells, equivalent to 1 and 6 sensor pixels; threshold at the whole DoG image’s mean + 4 SD; components at least 3 cells, no upper size cut. No red-circle exclusion or saturation rejection is used here: saturated/extended features may still help locate motion, but no angles are extracted from them.</p><p>Tracking uses positive DoG contrast after removing the temporal cell median, phase correlation, then mutual centroid matches within 8 sensor pixels. Accepted translations require at least 8 matches, at least 12% of the smaller candidate set, phase response ≥0.03 and median matching error ≤4 pixels. Moving track fragments require at least 3 frames and 12 pixels of total extent. These are heuristic diagnostics, not confidence probabilities. Original synthetic phase-marker frames are excluded, and every stored minimum is checked against the raw movie.</p><p>Weak rods, changing illumination, sample-dependent scattering and interference can violate a fixed-additive-background model. A rod-free mask does not establish that an intensity background removes coherent cross terms. No angle distribution or acquisition file is changed by this report.</p></details>
<p><a id="numbers">Capture measurements and tracking</a> · <a id="estimate">Candidate masked background (NaNs mark insufficient coverage)</a> · <a id="coverage">Unmasked-frame counts</a> · <a href="summary.json">All-capture summary</a></p>
</main><script>const D=__DATA__, $=x=>document.getElementById(x);let img=new Image(),timer=null,point=null,loadedKey='';
$('capture').innerHTML=D.map(d=>`<option value="${d.stamp}">${d.stamp} · ${d.label}</option>`).join('');
function selected(){return D.find(d=>d.stamp===$('capture').value)}
function marks(c,scale,tx=0,ty=0){let d=selected(),f=+$('slider').value,r=d.frames[f-1];c.save();c.translate(tx,ty);c.scale(scale,scale);c.lineWidth=1/scale;
 if($('trails').checked)for(const tr of Object.values(d.tracks)){let q=tr.filter(p=>p[0]<=f);if(q.length<2)continue;c.strokeStyle='rgba(255,218,96,.7)';c.beginPath();q.forEach((p,j)=>j?c.lineTo(p[1]/2,p[2]/2):c.moveTo(p[1]/2,p[2]/2));c.stroke()}
 if($('spots').checked)for(const s of r.spots){c.strokeStyle=s.status==='moving'?'#62ef9a':s.status==='persistent'?'#f2aa42':'#68dce8';c.beginPath();c.arc(s.x/2,s.y/2,5,0,2*Math.PI);c.stroke()}
 if($('dynamic').checked)for(const s of r.dynamic){c.strokeStyle='#df93ff';c.strokeRect(s.bbox[0]/2,s.bbox[1]/2,s.bbox[2]/2,s.bbox[3]/2)}c.restore()}
function draw(){let d=selected(),f=+$('slider').value;if(loadedKey!==`${d.stamp}/${f}`)return;let c=$('map').getContext('2d');c.drawImage(img,0,0);marks(c,1);let r=d.frames[f-1];$('frameLabel').textContent=`Frame ${f}/${d.summary.frames}`;$('status').textContent=`${r.spots.length} minimum-subtracted candidates; ${r.dynamic.length} transient candidates. ${d.note}`;
 if(point){let c=$('crop').getContext('2d'),x=Math.min(img.width-40,Math.max(40,point[0])),y=Math.min(img.height-40,Math.max(40,point[1]));c.imageSmoothingEnabled=false;c.drawImage(img,x-40,y-40,80,80,0,0,320,320);marks(c,4,-(x-40)*4,-(y-40)*4);$('cropLabel').textContent=`Centre x=${Math.round(2*x)}, y=${Math.round(2*y)} sensor pixels; crop covers 160×160 sensor pixels. Frame ${f}.`;}}
function loadFrame(){const d=selected(),f=+$('slider').value,im=new Image();im.onload=()=>{if(selected().stamp===d.stamp&&+$('slider').value===f){img=im;loadedKey=`${d.stamp}/${f}`;draw()}};im.src=`${d.stamp}/f${String(f).padStart(2,'0')}.jpg`}
function stop(){clearInterval(timer);timer=null;$('play').textContent='Play'}
function choose(stamp){stop();$('capture').value=stamp;const d=selected(),q=d.masked_median;$('slider').max=d.summary.frames;$('slider').value=1;point=null;loadedKey='';$('crop').getContext('2d').clearRect(0,0,320,320);$('cropLabel').textContent='Click the image for an enlarged crop.';$('overview').src=`${stamp}/overview.png`;$('numbers').href=`${stamp}/results.json`;
 $('medianTrial').hidden=!q.available;$('differenceBox').hidden=!q.available;$('estimate').hidden=!q.available;$('coverage').hidden=!q.available;
 if(q.available){$('medianTrial').src=`${stamp}/median_trial.png`;$('difference').src=`${stamp}/median_difference.png`;$('estimate').href=`${stamp}/masked_median.npy`;$('coverage').href=`${stamp}/unmasked_frame_count.npy`;$('medianText').textContent=`${q.trajectory_objects} merged candidate paths. ${q.valid_area_percent.toFixed(2)}% of the sensor has at least ${q.minimum_unmasked_frames} unmasked samples. At the median pixel ${q.count_quantiles[3]} of ${d.summary.frames} frames remain. Minimum mean: ${q.minimum_mean_8bit.toFixed(2)}/255; ordinary median mean: ${q.ordinary_median_mean_8bit.toFixed(2)}/255; masked median over supported pixels: ${q.masked_median_mean_8bit.toFixed(2)}/255.`}
 else $('medianText').textContent=q.reason+' The movie and detections remain available above.';loadFrame()}
$('capture').onchange=()=>choose($('capture').value);$('slider').oninput=()=>{stop();loadFrame()};for(const k of ['spots','trails','dynamic'])$(k).onchange=draw;
function step(n){let d=selected();$('slider').value=((+$('slider').value-1+n+d.summary.frames)%d.summary.frames)+1;loadFrame()}
$('prev').onclick=()=>{stop();step(-1)};$('next').onclick=()=>{stop();step(1)};$('play').onclick=()=>{if(timer)stop();else{$('play').textContent='Pause';timer=setInterval(()=>step(1),250)}};
$('map').onclick=e=>{let r=$('map').getBoundingClientRect();point=[(e.clientX-r.left)*1232/r.width,(e.clientY-r.top)*1028/r.height];draw()};choose('183650');</script></html>'''
    page=page.replace('__TABLE__',table).replace('__DATA__',json.dumps(clean(payload),ensure_ascii=False,separators=(',',':')))
    (OUT/'index.html').write_text(page,encoding='utf8')
    print('Report written',OUT/'index.html',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stamp');ap.add_argument('--skip-existing',action='store_true');ap.add_argument('--render',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True)
    if args.render:render();raise SystemExit
    for p in captures():
        if args.stamp and args.stamp not in p.name:continue
        if not args.skip_existing or not (OUT/p.name[24:30]/'results.json').exists():process(p)
        masked_median(p)
