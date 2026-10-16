"""Dense motion-path search, with no per-frame spot-intensity cutoff.

Detection combines separate frames only to test stage-following evidence.
All angle estimates come from one original, unwarped sensor frame at a time.
"""
from pathlib import Path
import argparse,json
import numpy as np
import cv2
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from background_motion_check import HERE,DAY,cells,read,write,sha
from polarcam.live.intensity_spots import channel_means,anisotropy
from polarcam import optics

OUT=HERE/'motion_selected_rods';PREV=HERE/'background_motion_check'
STAMPS=['180149','183650','185032','192157','210358']
LABELS={'180149':'Initial test (separate diagnostic)','183650':'Early DNA · BG 1','185032':'Early DNA · BG 2','192157':'Gold control','210358':'Later high-power DNA'}
MODEL=optics.theta_model();ABC=[MODEL[k] for k in ['A','B','C']];SCALE=255/4095
cv2.setNumThreads(3)

def theta(ch):
    v=anisotropy(ch);r=np.linalg.norm(v,axis=-1)
    return v,r,np.degrees(optics.theta_from_radius(r,*ABC))

def transform(a,shift):
    # Destination is the sample-coordinate reference; sample frame at ref+pose.
    return cv2.warpAffine(a,np.array([[1,0,shift[0]/2],[0,1,shift[1]/2]],np.float32),
           (a.shape[1],a.shape[0]),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def coherent(z,pose):
    sums=[np.zeros(z.shape[1:],np.float32),np.zeros(z.shape[1:],np.float32)]
    for i in range(len(z)):sums[i%2]+=transform(z[i],pose[i])
    return sums[0]/len(z[::2]),sums[1]/len(z[1::2])

def peak_indices(train,valid):
    return np.where(valid&(train==maximum_filter(train,size=7,mode='nearest')))

def detect(stamp):
    folder=OUT/stamp;folder.mkdir(parents=True,exist_ok=True);prev=read(PREV/stamp/'results.json')
    assert all(p['supported'] for p in prev['pairs'])
    meta=read(DAY/'backgrounds'/prev['profile']);path=DAY/'backgrounds'/prev['raw_stack'];raw=np.load(path,mmap_mode='r');n=meta['actual']['frames']
    assert len(raw)==n+int(meta['actual']['phase_marker_appended'])
    pose=np.vstack([np.zeros(2),np.cumsum([[p['dx'],p['dy']] for p in prev['pairs']],axis=0)])
    stack=np.stack([cells(f) for f in raw[:n]])
    # A small PSF-scale blur tolerates modest coordinate errors. No frame has a spot threshold.
    smooth=np.stack([cv2.GaussianBlur(f,(0,0),1,borderType=cv2.BORDER_REPLICATE) for f in stack])
    med=np.empty(stack.shape[1:],np.float32);scale=np.empty_like(med)
    for y in range(0,med.shape[0],64):
        block=smooth[:,y:y+64];m=np.median(block,axis=0);med[y:y+64]=m
        scale[y:y+64]=np.maximum(1,1.4826*np.median(np.abs(block-m),axis=0))
    residual=smooth-med
    z=np.tanh(residual/scale)
    del smooth
    H,W=stack.shape[1:];yy,xx=np.mgrid[:H,:W];sensorx=xx*2+.5;sensory=yy*2+.5
    # Same field common to every path/control and large enough for an 18x18 aperture.
    valid=(sensorx+pose[:,0].min()>=12)&(sensorx+pose[:,0].max()<2*W-12)&(sensory+pose[:,1].min()>=12)&(sensory+pose[:,1].max()<2*H-12)
    train,test=coherent(z,pose)
    # Bounded scores plateau on strong PSFs: never use their maxima as centres.
    # Locate peaks in the signed, temporal-median-centred moving signal, using
    # discovery frames only. No amplitude threshold is applied to these peaks.
    localisation,_=coherent(residual,pose);iy,ix=peak_indices(localisation,valid)
    rng=np.random.default_rng(20261015+int(stamp));null=[];perms=[];control_maps=[]
    for j in range(5):
        perm=rng.permutation(n);a,b=coherent(z,pose[perm]);loc,_=coherent(residual,pose[perm]);ny,nx=peak_indices(loc,valid)
        null.extend(np.minimum(a[ny,nx],b[ny,nx]).tolist());control_maps.append((a,b,ny,nx));perms.append(perm.tolist())
    threshold=max(.30,float(np.percentile(null,99.9)))
    zero_a,zero_b=coherent(z,np.zeros_like(pose));zero_loc,_=coherent(residual,np.zeros_like(pose));zy,zx=peak_indices(zero_loc,valid)
    null_counts=[int(np.sum((a[y,x]>=threshold)&(b[y,x]>=threshold))) for a,b,y,x in control_maps]
    passed=(train[iy,ix]>=threshold)&(test[iy,ix]>=threshold)
    selected=list(zip(iy[passed],ix[passed]));selected.sort(key=lambda p:-min(train[p],test[p]))
    # Suppress duplicate maxima only, not overlapping extraction windows.
    centres=[]
    for y,x in selected:
        if centres and np.min(np.linalg.norm(np.array(centres)-[x,y],axis=1))<6:continue
        centres.append([x,y])
    oldtrees=[cKDTree([[s['x'],s['y']] for s in f['spots']]) if f['spots'] else None for f in prev['frames']]
    rows=[]
    for j,(x,y) in enumerate(centres):
        pos=np.array([2*x+.5,2*y+.5]);pathxy=pos+pose
        oldfound=sum(tr is not None and tr.query(p)[0]<=8 for tr,p in zip(oldtrees,pathxy))
        # Evidence at the prescribed path; no brightness-guided recentering.
        values=[]
        for f,p in enumerate(pathxy):
            values.append(float(cv2.getRectSubPix(z[f],(1,1),((p[0]-.5)/2,(p[1]-.5)/2))[0,0]))
        rows.append(dict(id=j+1,x=float(pos[0]),y=float(pos[1]),train_score=float(train[y,x]),validation_score=float(test[y,x]),
             score=float(min(train[y,x],test[y,x])),positive_path_fraction=float(np.mean(np.array(values)>0)),
             old_detection_frames=int(oldfound),never_old_detected=bool(oldfound==0),path_evidence=values))
    for r in rows:
        r['neighbours']=[s['id'] for s in rows if s['id']!=r['id'] and abs(s['x']-r['x'])<18 and abs(s['y']-r['y'])<18]
    summary=dict(candidates=len(rows),never_old_detected=sum(r['never_old_detected'] for r in rows),
          wrong_path_counts=null_counts,wrong_path_peak_quantile=float(np.percentile(null,99.9)),threshold=threshold,
          stationary_path_count=int(np.sum((zero_a[zy,zx]>=threshold)&(zero_b[zy,zx]>=threshold))),
          common_sensor_area_percent=100*float(valid.mean()))
    write(folder/'detections.json',dict(stamp=stamp,label=LABELS[stamp],frames=n,pose=pose,spots=rows,summary=summary,
          raw_stack=prev['raw_stack'],profile=prev['profile'],raw_signature=prev['raw_signature'],source_metadata_sha256=sha(DAY/'backgrounds'/prev['profile']),
          stage_path_source_sha256=sha(PREV/stamp/'results.json'),permutation_seed=20261015+int(stamp),permutations=perms,
          method='Dense path-matched search across the common field. Aligned 2x2 intensity cells, Gaussian sigma1 cell; per-sensor-cell temporal median and MAD normalization (floor1 DN), tanh bounded signed residual. Warp only detection maps to sample coordinates. Odd frames locate local maxima of the registered signed temporal-median-centred signal (no amplitude cutoff, avoids flat bounded-score maxima); even frames validate motion at those exact coordinates. Both odd/even bounded scores must exceed max(0.30, 99.9th percentile of minimum discovery/validation score at peaks of five scrambled-timing paths). No per-frame brightness/DoG cutoff or previous spot seed. 12-sensor-pixel duplicate suppression. Empirical wrong-path controls are diagnostic, not calibrated FDR.',
          limitations='Signals still must be measurable above noise. Common motion identifies sample-associated features, not molecular identity. Candidate discovery aggregates temporal evidence but angles are never computed from an averaged/warped polarization image. This is not an independent biological replicate.'))
    # View: stage-registered bounded coherence, not a photograph or angle source.
    score=np.minimum(train,test)
    Image.fromarray(np.rint(np.clip(score,0,1)*255).astype('uint8')).save(folder/'coherence.png')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
    im=axes[0].imshow(np.where(valid,score,np.nan),vmin=0,vmax=1,cmap='magma',extent=(0,2*W,2*H,0));axes[0].set_title('Motion coherence: lower of discovery / validation');fig.colorbar(im,ax=axes[0],label='Bounded path score')
    if rows:axes[0].scatter([s['x'] for s in rows],[s['y'] for s in rows],s=13,facecolors='none',edgecolors='#70f0dd',lw=.5)
    axes[1].hist(null,bins=np.linspace(-.5,1,101),density=True,histtype='step',label='Scrambled paths: discovery peaks',color='.5')
    axes[1].hist(np.minimum(train[iy,ix],test[iy,ix]),bins=np.linspace(-.5,1,101),density=True,histtype='step',label='True path: discovery peaks',color='#176d91');axes[1].axvline(threshold,ls='--',color='#b46629');axes[1].set(yscale='log',xlabel='Minimum discovery / validation score',ylabel='Density (log scale)',title=f'{len(rows)} motion candidates; wrong paths {null_counts}');axes[1].legend(fontsize=8)
    fig.suptitle(LABELS[stamp]+' · '+stamp);fig.savefig(folder/'detection.png',dpi=145);plt.close(fig)
    print('DETECT',stamp,summary,flush=True)

def extract(stamp):
    folder=OUT/stamp;d=read(folder/'detections.json');rows=d['spots'];n=d['frames'];pose=np.array(d['pose']);p=DAY/'backgrounds'/d['raw_stack'];raw=np.load(p,mmap_mode='r')
    minimum=np.load(DAY/'backgrounds'/Path(d['profile']).with_suffix('.npy')).astype('float32');masked=np.load(PREV/stamp/'masked_median.npy')
    for s in rows:s.update(raw_peak=0,series={k:dict(intensity=[],r=[],theta=[],xy=[],bg_valid=[]) for k in ['minimum','masked_median']})
    for fi in range(n):
        frame=np.asarray(raw[fi]);bounds=[]
        for s in rows:
            x,y=np.array([s['x'],s['y']])+pose[fi];l=int(round(x))-9;t=int(round(y))-9
            assert 0<=l<l+18<=frame.shape[1] and 0<=t<t+18<=frame.shape[0];bounds.append((l,l+18,t,t+18));s['raw_peak']=max(s['raw_peak'],int(frame[t:t+18,l:l+18].max()))
        for key,bg in [('minimum',minimum),('masked_median',masked)]:
            corrected=np.rint(np.maximum(frame.astype('float32')-bg,0));ch=channel_means(corrected,bounds);v,r,th=theta(ch)
            for j,s in enumerate(rows):
                sr=s['series'][key];sr['intensity'].append(float(np.mean(ch[j]))*SCALE);sr['r'].append(float(r[j]));sr['theta'].append(float(th[j]));sr['xy'].append(v[j].tolist());sr['bg_valid'].append(bool(np.isfinite(ch[j]).all()))
    # One intensity-independent frame vs the requested minimum-intensity frame, at fixed path positions.
    used=[]
    for s in rows:
        s['saturated']=s['raw_peak']>=4094
        s['overlap_with']=next((t['id'] for t in used if abs(s['x']-t['x'])<18 and abs(s['y']-t['y'])<18),None)
        s['statistics_kept']=not s['saturated'] and s['overlap_with'] is None
        if s['statistics_kept']:used.append(s)
        for key,sr in s['series'].items():
            intensity=np.array(sr['intensity']);valid=np.isfinite(intensity)
            choices={'first':0,'middle':n//2,'dimmest':int(np.nanargmin(intensity)) if valid.any() else None}
            sr['selected']={k:(dict(frame=i+1,intensity=sr['intensity'][i],r=sr['r'][i],theta=sr['theta'][i]) if i is not None else None) for k,i in choices.items()}
    d['spots']=rows;d['model']=MODEL;d['angle_method']='Original unwarped raw frames, tracked 18x18 apertures with sensor parity retained. Clip/rint raw minus selected background, channel means, X/Y/r, existing Fresnel/BFP annular model. Any raw pixel >=4094 anywhere along the 18x18 path in any real frame excludes the track from statistics. Keep all candidates in gallery. For overlapping extraction apertures, retain stronger motion score for statistics. Finite r above model limit stays in r; theta undefined. Dimmest uses minimum background-subtracted aperture mean across valid BG frames, not a pixelwise minimum or separately chosen channel minima. No temporal average for angles.'
    d['summary'].update(saturated=sum(s['saturated'] for s in rows),statistics_kept=sum(s['statistics_kept'] for s in rows),overlap_excluded=sum(not s['saturated'] and not s['statistics_kept'] for s in rows))
    write(folder/'results.json',d);print('ANGLES',stamp,d['summary'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['detect','extract','all']);ap.add_argument('--stamp');args=ap.parse_args();OUT.mkdir(exist_ok=True)
    for stamp in STAMPS:
        if args.stamp and stamp!=args.stamp:continue
        if args.phase in ['detect','all']:detect(stamp)
        if args.phase in ['extract','all']:extract(stamp)
