"""Independent loose single-frame peaks, linked by the measured stage translation.

No old motion-score threshold is required. Old IDs are cross-references only.
"""
from pathlib import Path
import argparse
import numpy as np
import cv2
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib.pyplot as plt
import motion_selected_rods as old
from background_motion_check import cells,dog,read,write,sha

OUT=old.OUT/'repeated';KS=[1,2,3];RADIUS=6.;MIN_FRAMES=5

def sample(a,xy):
    return np.array([cv2.getRectSubPix(a,(1,1),((float(x)-.5)/2,(float(y)-.5)/2))[0,0] for x,y in xy])

def single_frame(d):
    mu=float(d.mean());sd=float(d.std());mx=d==maximum_filter(d,size=7,mode='nearest');rows={}
    for k in KS:
        above=(d>mu+k*sd).astype('float32');count=cv2.boxFilter(above,-1,(3,3),normalize=False)
        y,x=np.where(mx&(above>0)&(count>=3))
        valid=(x>=6)&(x<d.shape[1]-6)&(y>=6)&(y<d.shape[0]-6);y,x=y[valid],x[valid]
        rows[str(k)]=np.column_stack((2*x+.5,2*y+.5,(d[y,x]-mu)/sd)).tolist()
    return rows,mu,sd

def link(frame_lists,pose,H,W):
    # Discovery uses every k=1 per-frame detection. Summed unit votes locate
    # repeated positions, never averaging polarization data for angle estimates.
    ref=[np.asarray(f['1'],float).reshape(-1,3)[:,:2]-p for f,p in zip(frame_lists,pose)]
    votes=np.zeros((H//2,W//2),np.float32)
    for a in ref:
        xx=np.rint((a[:,0]-.5)/2).astype(int);yy=np.rint((a[:,1]-.5)/2).astype(int);valid=(xx>=0)&(xx<W//2)&(yy>=0)&(yy<H//2);np.add.at(votes,(yy[valid],xx[valid]),1)
    votes=cv2.GaussianBlur(votes,(0,0),1.5)
    y,x=np.where((votes==maximum_filter(votes,size=7,mode='nearest'))&(votes>0))
    centres=np.column_stack((x*2+.5,y*2+.5));trees=[cKDTree(a) if len(a) else None for a in ref]
    active=np.ones(len(centres),bool);counts=np.zeros(len(centres),int)
    for _ in range(2):
        ii=np.where(active)[0];matched=np.full((len(ii),len(ref),2),np.nan)
        for fi,(a,tr) in enumerate(zip(ref,trees)):
            if tr is None:continue
            distance,j=tr.query(centres[ii]);ok=distance<=RADIUS;matched[ok,fi]=a[j[ok]]
        counts[ii]=np.isfinite(matched[:,:,0]).sum(axis=1);enough=counts[ii]>=MIN_FRAMES;active[ii[~enough]]=False
        centres[ii[enough]]=np.nanmedian(matched[enough],axis=1)
    candidates=[]
    for c,count in zip(centres[active],counts[active]):
        if not(12<=c[0]+pose[:,0].min() and c[0]+pose[:,0].max()<W-12 and 12<=c[1]+pose[:,1].min() and c[1]+pose[:,1].max()<H-12):continue
        candidates.append((int(count),float(votes[int(round((c[1]-.5)/2)),int(round((c[0]-.5)/2))]),c))
    candidates.sort(key=lambda a:(-a[0],-a[1]));kept=[]
    for count,strength,c in candidates:
        if kept and np.min(np.linalg.norm(np.array(kept)-c,axis=1))<12:continue
        kept.append(c)
    return np.array(kept).reshape(-1,2)

def metrics(centres,frame_lists,pose,dogs):
    n=len(frame_lists);N=len(centres);hits={str(k):np.zeros((N,n),bool) for k in KS};distances={str(k):np.full((N,n),np.nan) for k in KS}
    # One detection belongs to at most one candidate in each frame and threshold.
    if N==0:return [],hits
    tr=cKDTree(centres)
    for fi,(f,p) in enumerate(zip(frame_lists,pose)):
        for k in KS:
            arr=np.array(f[str(k)]).reshape(-1,3);ref=arr[:,:2]-p
            if not len(ref):continue
            dist,ids=tr.query(ref)
            for dd,j in zip(dist,ids):
                if dd<=RADIUS:
                    hits[str(k)][j,fi]=True
                    prev=distances[str(k)][j,fi];distances[str(k)][j,fi]=min(prev,dd) if np.isfinite(prev) else dd
    contrast=np.stack([sample(d,centres+p) for d,p in zip(dogs,pose)],axis=1)
    result=[]
    for i,c in enumerate(centres):
        row={}
        for k in KS:
            matched=hits[str(k)][i];rest=~matched;support=contrast[i]>0
            frac=float(np.mean(support[rest])) if rest.any() else 1.
            row[str(k)]=dict(detected_frames=int(matched.sum()),hit_frames=(np.where(matched)[0]+1).tolist(),rest_frames=int(rest.sum()),rest_positive_frames=int(support[rest].sum()),rest_fraction=frac,
                 pass_strict=bool(matched.sum()>=MIN_FRAMES and frac==1),pass_90=bool(matched.sum()>=MIN_FRAMES and frac>=.9),median_match_error=float(np.nanmedian(distances[str(k)][i])) if matched.any() else None)
        result.append(dict(per_k=row,local_contrast_dn=contrast[i].tolist()))
    return result,hits

def process(stamp):
    folder=OUT/stamp;folder.mkdir(parents=True,exist_ok=True);d=read(old.OUT/stamp/'results.json');pose=np.array(d['pose']);raw=np.load(old.DAY/'backgrounds'/d['raw_stack'],mmap_mode='r');n=d['frames'];H,W=raw.shape[1:]
    minimum=np.load(old.DAY/'backgrounds'/Path(d['profile']).with_suffix('.npy'));dogs=[];frames=[];thresholds=[]
    for fi in range(n):
        corrected=np.maximum(np.asarray(raw[fi],np.float32)-minimum,0);dg=dog(cells(corrected));f,mu,sd=single_frame(dg);frames.append(f);thresholds.append(dict(mean=mu,sd=sd));dogs.append(dg)
    dogs=np.stack(dogs);centres=link(frames,pose,H,W);mm,hits=metrics(centres,frames,pose,dogs)
    # Preserve the familiar old ID when a unique spatial counterpart exists.
    oldxy=np.array([[s['x'],s['y']] for s in d['spots']]);tr=cKDTree(oldxy);dist,idx=tr.query(centres);assigned={};used=set();newid=max(s['id'] for s in d['spots'])+1
    for i in np.argsort(dist):
        if dist[i]<=8 and int(idx[i]) not in used:assigned[int(i)]=d['spots'][idx[i]]['id'];used.add(int(idx[i]))
    rows=[]
    for i,(c,met) in enumerate(zip(centres,mm)):
        sid=assigned.get(i)
        if sid is None:sid=newid;newid+=1
        os=d['spots'][sid-1] if sid<=len(d['spots']) else None
        rows.append(dict(id=sid,x=float(c[0]),y=float(c[1]),repeat_candidate=True,old_id=sid if os else None,old_motion_score=os['score'] if os else None,old_statistics_kept=os['statistics_kept'] if os else False,**met))
    # Old rejected candidates remain inspectable, but do not seed the new search.
    legacy=[s for i,s in enumerate(d['spots']) if i not in used]
    legmet,_=metrics(np.array([[s['x'],s['y']] for s in legacy]).reshape(-1,2),frames,pose,dogs)
    for s,met in zip(legacy,legmet):rows.append(dict(id=s['id'],x=s['x'],y=s['y'],repeat_candidate=False,old_id=s['id'],old_motion_score=s['score'],old_statistics_kept=s['statistics_kept'],**met))
    rows.sort(key=lambda s:s['id'])
    # Angle extraction matches the original analysis, at the new fixed trajectory.
    for s in rows:
        s.update(raw_peak=0,peak_cell_255=[],aperture_mean_255=[],raw_peak_frame=0,series={k:dict(intensity=[],r=[],theta=[],xy=[],bg_valid=[]) for k in ['minimum','masked_median']})
    bgs={'minimum':minimum.astype('float32'),'masked_median':np.load(old.PREV/stamp/'masked_median.npy')}
    for fi in range(n):
        frame=np.asarray(raw[fi]);bounds=[]
        for s in rows:
            x,y=np.array([s['x'],s['y']])+pose[fi];l=int(round(x))-9;t=int(round(y))-9;bounds.append((l,l+18,t,t+18));peak=int(frame[t:t+18,l:l+18].max())
            if peak>s['raw_peak']:s['raw_peak']=peak;s['raw_peak_frame']=fi+1
        for key,bg in bgs.items():
            corrected=np.rint(np.maximum(frame.astype('float32')-bg,0));ch=old.channel_means(corrected,bounds);v,r,theta=old.theta(ch)
            for j,s in enumerate(rows):
                sr=s['series'][key];sr['intensity'].append(float(np.mean(ch[j]))*old.SCALE);sr['r'].append(float(r[j]));sr['theta'].append(float(theta[j]));sr['xy'].append(v[j].tolist());sr['bg_valid'].append(bool(np.isfinite(ch[j]).all()))
                if key=='minimum':
                    l,rr,t,b=bounds[j];le=l+l%2;tt=t+t%2;right=rr-rr%2;bottom=b-b%2;s['peak_cell_255'].append(float(cells(corrected[tt:bottom,le:right]).max())*old.SCALE)
    for s in rows:
        s['saturated']=s['raw_peak']>=4094
        for key,sr in s['series'].items():
            a=np.array(sr['intensity']);picks={'first':0,'middle':n//2,'dimmest':int(np.nanargmin(a)) if np.isfinite(a).any() else None}
            sr['selected']={k:(dict(frame=i+1,intensity=sr['intensity'][i],r=sr['r'][i],theta=sr['theta'][i]) if i is not None else None) for k,i in picks.items()}
        s['peak_cell_max_255']=max(s['peak_cell_255']);s['peak_cell_median_255']=float(np.median(s['peak_cell_255']))
    # Each threshold gets its own overlap exclusion; a rejected stronger neighbour
    # must not suppress an otherwise eligible candidate.
    for k in KS:
        for mode in ['strict','90','five_only']:
            used=[]
            pool=sorted(rows,key=lambda s:(-s['per_k'][str(k)]['detected_frames'],-s['peak_cell_median_255']))
            for s in pool:
                met=s['per_k'][str(k)];accepted=s['repeat_candidate'] and met['detected_frames']>=5 and (mode=='five_only' or met['rest_fraction']>= (1 if mode=='strict' else .9))
                overlap=next((v['id'] for v in used if abs(s['x']-v['x'])<18 and abs(s['y']-v['y'])<18),None) if accepted else None
                keep=bool(accepted and not s['saturated'] and overlap is None)
                if keep:used.append(s)
                met[mode+'_accepted']=bool(accepted);met[mode+'_kept']=keep;met[mode+'_overlap']=overlap
    controls=[]
    for perm in d['permutations'] if 'permutations' in d else read(old.OUT/stamp/'detections.json')['permutations']:
        pp=pose[np.array(perm)];cc=link(frames,pp,H,W);cm,_=metrics(cc,frames,pp,dogs)
        controls.append({str(k):dict(five_only=sum(m['per_k'][str(k)]['detected_frames']>=5 for m in cm),strict=sum(m['per_k'][str(k)]['pass_strict'] for m in cm),ninety=sum(m['per_k'][str(k)]['pass_90'] for m in cm)) for k in KS})
    summary={str(k):{mode:dict(accepted=sum(s['per_k'][str(k)][mode+'_accepted'] for s in rows),statistics=sum(s['per_k'][str(k)][mode+'_kept'] for s in rows),saturated=sum(s['per_k'][str(k)][mode+'_accepted'] and s['saturated'] for s in rows)) for mode in ['strict','90','five_only']} for k in KS}
    result={k:d[k] for k in ['stamp','label','frames','pose','raw_stack','profile','model']};result.update(spots=rows,per_frame=frames,thresholds=thresholds,summary=summary,wrong_time_controls=controls,
       method='Original minimum background subtracted, aligned 2x2 intensity means. DoG sigma 1 and 6 sensor pixels on half-grid. Independent per-frame local maxima in 14x14 sensor neighbourhood; peak > whole DoG mean+k SD with >=3 cells above threshold in its 3x3 cell neighbourhood. No upper component area limit. k=1,2,3. Align per-frame k1 detections by existing stage translations, cluster their sample coordinates, require >=5 distinct frames, refine centres by medians of matched positions. 6-sensor-pixel matching tolerance; 12-pixel duplicate-centre suppression; one per-frame detection cannot support multiple candidate tracks. Report defaults to k1 and positive DoG at the prescribed centre in every unmatched frame. Alternative 90% and five-only rules shown. Old IDs are unique spatial cross-references within8 px; old-only rows are diagnostic and cannot pass. Raw saturation>=4094 in any trajectory aperture is excluded; stronger recurrence then brightness wins overlapping18x18 apertures. All theta uses one real unwarped frame, never temporal averaging.',
       provenance=dict(analysis_source_sha256=sha(Path(__file__)),previous_results_sha256=sha(old.OUT/stamp/'results.json')))
    write(folder/'results.json',result);print(stamp,summary,'controls k2 strict',[v['2']['strict'] for v in controls],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stamp');args=ap.parse_args();OUT.mkdir(exist_ok=True)
    for stamp in old.STAMPS:
        if args.stamp and stamp!=args.stamp:continue
        process(stamp)
