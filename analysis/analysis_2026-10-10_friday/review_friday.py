"""Offline first pass of the 9 October data; does not import/run the camera app."""
from pathlib import Path
import sys,json,csv
import numpy as np
from scipy import ndimage,signal
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from html import escape

B=Path(__file__).resolve().parent; REPO=B.parents[1]
DATA=REPO/'data/hinges/polarcam/2026-10-09'
sys.path.insert(0,str(B.parent/'paper_analysis_20261008/original_scripts'))
import hinge_analysis as ha
MODEL=ha.fourkas_ABC(1.3,1.33,.39)
for name in ('plots','traces'): (B/name).mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def save(name,fig):
    fig.savefig(B/'plots'/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(B/'plots'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def table(name,rows):
    with (B/name).open('w',newline='',encoding='utf-8') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def bounds_channels(a,bounds,origin=(0,0)):
    if a.ndim==2:a=a[None]
    result=np.empty((a.shape[0],len(bounds),4))
    for j,(x,y,size) in enumerate(bounds):
        crop=a[:,y:y+size,x:x+size];px=(x+origin[0])%2;py=(y+origin[1])%2
        result[:,j]=np.stack([crop[:,1-py::2,1-px::2].mean((1,2)),crop[:,py::2,px::2].mean((1,2)),crop[:,py::2,1-px::2].mean((1,2)),crop[:,1-py::2,px::2].mean((1,2))],axis=-1)
    return result

def xytheta(ch):
    with np.errstate(divide='ignore',invalid='ignore'):
        x=(ch[...,0]-ch[...,1])/(ch[...,0]+ch[...,1]);y=(ch[...,2]-ch[...,3])/(ch[...,2]+ch[...,3])
    good=(ch>=0).all(-1)&np.isfinite(ch).all(-1)&((ch[...,0]+ch[...,1])>0)&((ch[...,2]+ch[...,3])>0)
    x=np.where(good,x,np.nan);y=np.where(good,y,np.nan)
    theta,flags=ha.theta_from_r(np.hypot(x,y),*MODEL,return_flags=True)
    theta=np.where(good&flags['valid'],theta,np.nan)  # helper already returns degrees
    return x,y,theta

def detect(mean):
    h,w=mean.shape;cells=(mean[:h//2*2:2,:w//2*2:2]+mean[1:h//2*2:2,:w//2*2:2]+mean[:h//2*2:2,1:w//2*2:2]+mean[1:h//2*2:2,1:w//2*2:2])/4
    dog=ndimage.gaussian_filter(cells,.8)-ndimage.gaussian_filter(cells,3)
    med=np.median(dog);threshold=med+8*1.4826*np.median(np.abs(dog-med))
    yy,xx=np.where((dog==ndimage.maximum_filter(dog,size=7))&(dog>threshold))
    pts=np.c_[2*xx,2*yy];mask=(pts[:,0]>=14)&(pts[:,0]<w-14)&(pts[:,1]>=14)&(pts[:,1]<h-14);pts=pts[mask]
    if len(pts)>1:pts=pts[cKDTree(pts).query(pts,k=2)[0][:,1]>=24]
    return [(int(x)-6,int(y)-6,12) for x,y in pts],float(threshold)

def category(stamp):
    # Explicit notes: sixth stack says 1–6 all hinges. Five gold fields follow oil replacement.
    if stamp in ('154457','154551','154730','154844','154929','155041'):return 'hinges_initial'
    if stamp in ('161931','162234','162337','162429','162516'):return 'gold_only_after_oil'
    if stamp in ('161353','161451'):return 'gold_duplicate_field'
    return 'switching_or_context_only'

def count_frames(a,m):
    marker=bool(m.get('actual',{}).get('phase_marker_appended',False))
    n=a.shape[0]-int(marker)
    actual=m.get('actual',{}).get('frames')
    if actual is not None and int(actual)!=n:raise ValueError(f'Frame count mismatch: actual={actual}, array measurement={n}')
    return n

def run():
    inventory=[];fields=[];spots=[];distributions={};all_spot_traces={}
    paths=sorted(DATA.rglob('*20261009*.json'))
    for p in paths:
        m=json.loads(p.read_text(encoding='utf-8-sig'))
        if not isinstance(m,dict):continue
        kind=m.get('recording_type','');npy=p.with_suffix('.npy')
        if kind not in ('fetch_frames','spotrec_magnified_roi') or not npy.exists():continue
        stamp=p.stem.split('-')[-1];c=m.get('conditions') or {};actual=m.get('actual') or {};roi=m.get('roi') or {}
        declared=c.get('interference_drive','');sound=m.get('sound',{}).get('sound_on')
        conflict=declared in ('On','Off') and isinstance(sound,bool) and ((declared=='On')!=sound)
        row=dict(file=p.stem,folder=str(p.parent.relative_to(DATA)),recording_type=kind,group=category(stamp) if kind=='fetch_frames' else 'spot_context_unconfirmed',note=c.get('condition_note',''),drive_declaration=declared,sound_checkbox=sound,drive_conflict=conflict,fps=actual.get('fps'),exposure_us=(actual.get('timing') or {}).get('exposure_us'),background_subtracted=m.get('background',{}).get('background_subtracted'),max_raw=actual.get('max_raw_value'),cx=roi.get('cx'),cy=roi.get('cy'),roi_requested=roi.get('win_raw'),roi_camera_width=roi.get('w'),roi_camera_height=roi.get('h'))
        a=np.load(npy,mmap_mode='r');n=count_frames(a,m);row['frames']=n;row['duration_s']=n/float(actual['fps']);origin=(int(roi.get('x',0)),int(roi.get('y',0)))
        inventory.append(row)
        if kind=='fetch_frames':
            # Deliberately sample 30 equally spaced frames; don't pretend this is every frame.
            ids=np.unique(np.linspace(0,n-1,min(n,30)).astype(int));mean=np.zeros(a.shape[1:],dtype=np.float64)
            for i in ids:mean+=a[i]
            mean/=len(ids);bounds,thr=detect(mean);ch=bounds_channels(mean,bounds,origin)[0];x,y,theta=xytheta(ch)
            valid=np.isfinite(theta);distributions[p.stem]=theta[valid]
            row.update(detected_isolated=len(bounds),valid_angle=int(valid.sum()),detection_threshold=thr,mean_frames=len(ids),fraction_30_50_conditional=float(np.mean((theta[valid]>=30)&(theta[valid]<50))) if valid.any() else None)
            fields.append(dict(row))
            np.savez_compressed(B/'traces'/f'{p.stem}.npz',channels=ch,theta=theta,xy=np.c_[x,y],bounds=np.array(bounds),sampled_frames=ids)
        else:
            size=min(int(roi.get('win_raw',20)),a.shape[1],a.shape[2]);size-=size%2
            x0=max(0,min(a.shape[2]-size,round(float(roi['cx'])-origin[0])-size//2));y0=max(0,min(a.shape[1]-size,round(float(roi['cy'])-origin[1])-size//2))
            chunks=[];zeros=saturated=0;pixels=0
            for start in range(0,n,4096):
                part=a[start:min(start+4096,n)];chunks.append(bounds_channels(part,[(x0,y0,size)],origin)[:,0])
                crop=part[:,y0:y0+size,x0:x0+size];zeros+=np.count_nonzero(crop==0);saturated+=np.count_nonzero(crop>=4094);pixels+=crop.size
            ch=np.concatenate(chunks);x,y,theta=xytheta(ch);t=np.arange(n)/float(actual['fps'])
            block=max(1,round(float(actual['fps'])/100));use=n//block*block;tb=t[:use].reshape(-1,block).mean(1);cb=ch[:use].reshape(-1,block,4).mean(1);xb,yb,_=xytheta(cb)
            row.update(roi_local_x=x0,roi_local_y=y0,valid_angle_fraction=float(np.isfinite(theta).mean()),saved_zero_fraction=zeros/pixels,saved_saturated_fraction=saturated/pixels,xy_span_100hz=max(float(np.nanpercentile(xb,98)-np.nanpercentile(xb,2)),float(np.nanpercentile(yb,98)-np.nanpercentile(yb,2))),mean_total=float(ch.sum(1).mean()))
            freq,power=signal.welch(np.nan_to_num(np.c_[xb,yb],nan=0).T,fs=float(actual['fps'])/block,nperseg=min(len(tb),2048),detrend='linear',axis=-1)
            band=(freq>=.3)&(freq<=3);row['peak_0p3_3hz']=float(freq[band][np.argmax(power.sum(0)[band])]) if band.any() else None
            spots.append(dict(row));all_spot_traces[p.stem]=(tb,xb,yb,freq,power,ch)
            np.savez_compressed(B/'traces'/f'{p.stem}.npz',time_s=t,channels=ch,theta=theta,xy=np.c_[x,y])
        del a
        print('Reviewed',p.stem,flush=True)
    table('inventory.csv',inventory);table('field_summary.csv',fields);table('spot_summary.csv',spots)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5));edges=np.arange(0,92,2);centres=(edges[:-1]+edges[1:])/2;omega=np.pi*(np.cos(np.radians(edges[:-1]))-np.cos(np.radians(edges[1:])))
    for group,color,label in [('hinges_initial','#0072B2','Initial hinges (6 fields)'),('gold_only_after_oil','#D55E00','Gold-only after oil (5 fields)')]:
        selected=[r for r in fields if r['group']==group];curves=[]
        for r in selected:
            v=distributions[r['file']];hist=np.histogram(v,edges)[0]/max(1,len(v))/omega;curves.append(hist);axs[0].plot(centres,hist,color=color,alpha=.25,lw=.8)
        axs[0].plot(centres,np.mean(curves,axis=0),color=color,lw=2,label=label)
        axs[1].scatter([0 if group=='hinges_initial' else 1]*len(selected),[r['fraction_30_50_conditional'] for r in selected],color=color,s=45,label=label)
    axs[0].axhline(1/np.pi,color='gray',ls=':',label='Uniform folded orientation');axs[0].set(xlabel='Optical θ from vertical (°)',ylabel='Shell-averaged conditional density (sr⁻¹)',xlim=(0,90));axs[0].legend(fontsize=8)
    axs[1].set(xticks=[0,1],xticklabels=['Hinges','Gold only'],ylabel='Fraction of valid detected objects at 30–50°',ylim=(0,1),xlim=(-.4,1.4))
    fig.suptitle('9 October: preliminary field comparison — detection-conditioned, not assembly yield');fig.tight_layout();save('hinge_control_profiles',fig)
    fig,axs=plt.subplots(len(spots),2,figsize=(12,max(6,2.1*len(spots))),squeeze=False)
    for i,r in enumerate(spots):
        t,x,y,f,power,ch=all_spot_traces[r['file']];axs[i,0].plot(t,x,lw=.55,label='X');axs[i,0].plot(t,y,lw=.55,label='Y');axs[i,0].set(title=f"{r['file']} | sound checkbox={r['sound_checkbox']}",ylabel='X, Y',xlabel='time (s)');axs[i,1].semilogy(f,power.sum(0),lw=.8);axs[i,1].set(xlim=(.1,10),xlabel='frequency (Hz)',ylabel='PSD X+Y (Hz⁻¹)');axs[i,0].legend(fontsize=7,loc='upper right')
    fig.suptitle('Spot recordings: ~100 Hz channel means; native traces retained in local caches');fig.tight_layout(rect=(0,0,1,.99));save('spot_trace_gallery',fig)
    # Coordinate proximity identifies candidate repeat-rod comparisons, not confirmed pairs.
    pairs=[]
    for i,a in enumerate(spots):
        for b in spots[i+1:]:
            distance=np.hypot(float(a['cx'])-float(b['cx']),float(a['cy'])-float(b['cy']))
            if distance<4 and a['sound_checkbox']!=b['sound_checkbox']:
                pairs.append(dict(first=a['file'],second=b['file'],distance_px=float(distance),first_sound=a['sound_checkbox'],second_sound=b['sound_checkbox'],first_xy_span=a['xy_span_100hz'],second_xy_span=b['xy_span_100hz'],first_brightness=a['mean_total'],second_brightness=b['mean_total']))
    table('candidate_pairs.csv',pairs)
    stereo_comparison(fields)
    selected=[r for r in spots if r['file'].endswith(('173838','174418','174545'))]
    if selected:
        fig,axs=plt.subplots(len(selected),2,figsize=(12,2.5*len(selected)),squeeze=False)
        for i,r in enumerate(selected):
            t,x,y,f,power,ch=all_spot_traces[r['file']];axs[i,0].plot(t,x,lw=.6,label='X');axs[i,0].plot(t,y,lw=.6,label='Y');axs[i,0].set(title=f"{r['file']} · sound={r['sound_checkbox']}",xlabel='time (s)',ylabel='anisotropy');axs[i,0].legend();axs[i,1].semilogy(f,power.sum(0));axs[i,1].set(xlim=(.1,5),xlabel='frequency (Hz)',ylabel='X+Y PSD (Hz⁻¹)')
        fig.suptitle('Late-session comparison: operator-declared drive state; pairing still provisional');fig.tight_layout();save('late_session_traces',fig)
    (B/'summary.json').write_text(json.dumps({'fields':fields,'spots':spots,'candidate_pairs':pairs},indent=2),encoding='utf-8')
    write_report(fields,spots,pairs)

def stereo_comparison(fields):
    from matplotlib.colors import LogNorm
    from numpy.polynomial.legendre import leggauss
    edges=np.linspace(-2,2,51);width=edges[1]-edges[0];centres=(edges[:-1]+edges[1:])/2
    xx,yy=np.meshgrid(centres,centres,indexing='ij');nodes,weights=leggauss(4);omega=np.zeros_like(xx)
    for i,u in enumerate(nodes):
        for j,v in enumerate(nodes):omega+=weights[i]*weights[j]*(1+((xx+u*width/2)**2+(yy+v*width/2)**2)/4)**-2*(width/2)**2
    grids=[]
    for group in ('hinges_initial','gold_only_after_oil'):
        curves=[]
        for r in fields:
            if r['group']!=group:continue
            z=np.load(B/'traces'/f'{r["file"]}.npz');theta=z['theta'];xy=z['xy'];ok=np.isfinite(theta)
            phi=np.mod(np.arctan2(xy[ok,1],xy[ok,0])/2,np.pi);rho=2*np.tan(np.radians(theta[ok])/2)
            hist=np.histogram2d(rho*np.cos(phi),rho*np.sin(phi),bins=(edges,edges))[0]
            curves.append(hist/max(1,ok.sum())/omega)
        grids.append(np.mean(curves,axis=0))
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');norm=LogNorm(.01,max(g.max() for g in grids))
    for ax,g,title in zip(axs,grids,('Initial hinges: 6 fields','Gold only after oil: 5 fields')):
        image=ax.pcolormesh(edges,edges,np.ma.masked_less_equal(g.T,0),norm=norm,cmap='magma',rasterized=True)
        for angle in (20,40,60,90):
            t=np.linspace(0,np.pi,180);rho=2*np.tan(np.radians(angle)/2);ax.plot(rho*np.cos(t),rho*np.sin(t),color='gray',lw=.6)
        ax.set(title=title,xlabel='stereographic u',ylabel='v',xlim=(-2,2),ylim=(0,2));ax.set_aspect('equal')
    fig.colorbar(image,ax=axs,label='Field-averaged conditional density (sr⁻¹)',shrink=.75)
    fig.suptitle('One mean-channel orientation per detected object; equal field weight');save('hinge_control_stereographic',fig)

def write_report(fields,spots,pairs):
    def rows(items,keys):return '<table><tr>'+''.join('<th>'+escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in items)+'</table>'
    text='''<!doctype html><meta charset="utf-8"><title>9 October hinge data review</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;color:#203448}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}table{font-size:13px;width:100%}h2{margin-top:40px}</style><h1>9 October hinge/control data: first pass</h1>
    <p>Offline analysis only; the camera changes have not been executed or tested. See <a href="../../CAMERA_UPDATE_2026-10-10.md">lab validation checklist</a>. Earlier <a href="../paper_analysis_20261008/review.html">APD/publication figures and review</a> remain available.</p>
    <h2>Selection and identity</h2><p>Six initial stacks are explicitly labelled hinges by the sixth note. Five gold-only fields after re-oiling are the primary control group. 16:13:53 is identified retrospectively as gold-only by the 16:14:51 note; the latter repeats the same field specifically to correct the note, so neither is counted as an independent control field here. 16:10:09 and the later illumination-switching stacks remain contextual. No independent rod counts across overlapping fields are claimed.</p>
    <h2>Population comparison</h2><img src="plots/hinge_control_profiles.png"><p>For each field, detect isolated intensity peaks on the mean of 30 evenly spaced acquired frames, using a 2×2 total-intensity image, DoG scales 0.8/3 cells, median+8 robust-MAD threshold, and 24 sensor-pixel isolation. Use a 12×12 sensor-pixel aperture; no further background subtraction or channel-gain fitting. Convert mean channels using the homogeneous annular model (NA 1.3, inner NA 0.39, water n 1.33). Exclude invalid angles rather than clipping. Curves give equal weight to each field; thin lines show field variability. The 30–50° interval is descriptive, not a validated classifier. Differences in brightness, detection, oil/background changes and field overlap limit comparison. This is not assembly yield or a statistical significance claim.</p>
    <h2>Recording and metadata audit</h2><p>All rows retain original metadata. Drive declaration and sound checkbox sometimes disagree; the code change prevents this duplication going forward. Spot sidecars have no condition notes, so neighbouring frame-stack notes provide context but do not prove sample identity or pairing. Recorded exposure is about 0.598 ms; actual FPS comes from metadata. The appended marker is excluded. Metadata gain is 1; the future requested default of 20 exceeds this camera's recorded analogue maximum of about 16.22, so inspect actual gain readback in the lab.</p>'''
    text+='<img src="plots/hinge_control_stereographic.png">'
    text+='''<h2>Preliminary findings</h2><ul>
    <li>The two groups overlap strongly and share a dominant model-derived peak around 25–30°. The fraction at 30–50° averages 32.5% over six hinge fields versus 24.7% over five gold-only fields. That apparent difference is strongly influenced by the first hinge field (61.0%); the other five range from 19.8% to 30.1%, overlapping controls (20.7–30.2%). This threshold is descriptive and does not estimate attachment yield.</li>
    <li>The same-coordinate candidate pair 17:25:15 (sound on) / 17:28:33 (off) has a larger ~100 Hz X/Y percentile span when off: 0.125 versus 0.242. The later same-coordinate pair 17:44:18 (off) / 17:45:45 (on) has spans 0.600 versus 0.532, but the on recording has a low-frequency PSD peak near 1.64 Hz. These are observations, not proof of cancellation efficacy or vibration causation; absolute levels and brightness differ.</li>
    <li>Only about 0.57% of native samples in 17:14:36 give a valid angle under this optical model. Do not clip the remaining samples to 90° or interpret their inferred θ. Its X/Y traces remain useful for investigating why the radius is outside the model.</li>
    <li>Four frame-stack sidecars disagree between the drive declaration and sound checkbox. Spot recordings lack condition notes. Keep the original evidence and uncertainty, rather than rewriting historical metadata to match an assumed sequence.</li>
    </ul><p>Next analysis decisions: inspect aperture/background sensitivity of the common population peak and the first hinge-field outlier; verify the exact same-rod on/off pairs and acquisition conditions; compare positional drift with polarisation changes before attributing any low-frequency peak. The old APD plots remain illustrations of selected moving hinges, separate from these population-QC measurements.</p>'''
    text+=rows(fields,['file','group','detected_isolated','valid_angle','fraction_30_50_conditional','drive_declaration','sound_checkbox','drive_conflict','note'])
    text+='<h2>Spot motion and candidate pairs</h2><p>Each trace uses the actual small square at the rod centre within the wider hardware ROI, not the mean of the entire 256-pixel camera strip. Native channels are retained; display and spectra use approximately 100 Hz channel means. PSD is Welch with linear detrending. A peak is a descriptive feature, not an assigned mechanical cause. Centroid drift, photon noise and interference have not been separated.</p><img src="plots/late_session_traces.png">'
    text+=rows(spots,['file','cx','cy','sound_checkbox','duration_s','valid_angle_fraction','xy_span_100hz','peak_0p3_3hz','mean_total','saved_zero_fraction'])
    text+='<p>Candidate on/off pairs below share coordinates within four pixels. Coordinate similarity alone cannot establish matched illumination, focus, power, background or identical rod after stage movement.</p>'+rows(pairs,['first','second','distance_px','first_sound','second_sound','first_xy_span','second_xy_span','first_brightness','second_brightness'])
    text+='<p><a href="plots/spot_trace_gallery.png">Complete spot trace gallery</a> · <a href="inventory.csv">All recording metadata</a> · <a href="field_summary.csv">Field table</a> · <a href="spot_summary.csv">Spot table</a></p>'
    (B/'review.html').write_text(text,encoding='utf-8')

if __name__=='__main__':run()
