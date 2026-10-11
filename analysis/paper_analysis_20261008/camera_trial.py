"""Camera survey trial: equal weighting by rod, explicit validity and projection."""
from pathlib import Path
import json,csv,ast
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from paper_figures import ROOT,OUT,angles,vectors,stereo,wrap_phi,density_grid,save,MODELS

CAMERA_DATA=(Path(__file__).resolve().parents[2] / 'data/hinges/polarcam/2026-09-18')
FRAMES=150
LOCAL_BACKGROUND=False  # files already background-subtracted; do not subtract a second arbitrary offset
DC_MIN=100.
MIN_SPOT_AMP=100.

def detector_namespace():
    """Reuse only configuration and detector definitions from the preserved source."""
    nb=json.loads((ROOT/'original_scripts/Framestack_theta_survey.ipynb').read_text(encoding='utf-8'))
    env={'np':np,'ndimage':ndimage,'Path':Path}
    config=''.join(nb['cells'][1]['source']);exec(compile(config,'original camera configuration','exec'),env)
    tree=ast.parse(''.join(nb['cells'][4]['source']))
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('build_smap','detect_spots','detect_spots_multiscale','spot_halves')]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'preserved spot detectors','exec'),env)
    return env

def spot_channels(data,y,x,half=3,local_background=False):
    """Absolute sensor parity, real external square annulus; no aperture overlap."""
    nf,H,W=data.shape;outer=half+4
    y0=max(0,y-outer);y1=min(H,y+outer+1);x0=max(0,x-outer);x1=min(W,x+outer+1)
    yy=np.arange(y0,y1)[:,None];xx=np.arange(x0,x1)[None,:]
    inner=(abs(yy-y)<=half)&(abs(xx-x)<=half)
    annulus=(np.maximum(abs(yy-y),abs(xx-x))>=half+2)
    crop=np.asarray(data[:,y0:y1,x0:x1],dtype=float);out=[]
    for py,px in ((1,1),(0,0),(0,1),(1,0)):
        parity=(yy%2==py)&(xx%2==px); signal=inner&parity;bg=annulus&parity
        if not signal.any() or not bg.any(): raise ValueError('Empty extraction region')
        intensity=crop[:,signal].mean(axis=1)
        if local_background:intensity-=np.median(crop[:,bg],axis=1)
        out.append(intensity)
    return np.column_stack(out)

def survey_file(path,detector=None):
    meta=json.loads(path.with_suffix('.json').read_text());raw=np.load(path,mmap_mode='r')
    if tuple(int(meta['roi'][k])%2 for k in ('OffsetY','OffsetX'))!=(0,0):raise ValueError('This trial requires even ROI offsets')
    n=FRAMES
    if meta['actual']['frames']<n:raise ValueError(f'Need {n} acquired frames for equal weighting')
    data=raw[:n] # metadata excludes appended phase marker
    mean=np.zeros(data.shape[1:],dtype=float)
    for start in range(0,n,10):mean+=np.asarray(data[start:start+10],dtype=float).sum(0)
    mean/=n;env=detector or detector_namespace()
    smap,_=env['build_smap'](data,n_frames=40,chunk=5)
    spots,info=env['detect_spots_multiscale'](smap);halves=env['spot_halves'](spots,info)
    base=np.median(mean);rows=[];cloud=[];summaries=[]
    for i,((y,x),h) in enumerate(zip(spots,halves)):
        amp=float(mean[y,x]-base)
        if amp<MIN_SPOT_AMP:continue
        a=spot_channels(data,y,x,h,LOCAL_BACKGROUND)
        th,ph,valid,ax,ay=angles(a);dc=np.minimum(a[:,0]+a[:,1],a[:,2]+a[:,3])
        valid &= dc>DC_MIN
        abar=a.mean(axis=0,keepdims=True);tb,pb,vb,_,_=angles(abar)
        enough=valid.mean()>=.5
        accepted=bool(enough and vb[0] and min(abar[0,0]+abar[0,1],abar[0,2]+abar[0,3])>DC_MIN)
        iqr=float(np.degrees(np.percentile(th[valid],75)-np.percentile(th[valid],25))) if valid.sum()>3 else np.nan
        r=dict(file=path.name,spot_id=i,y=y,x=x,half=h,large_blob=(y,x) in info.get('large',[]),
               amplitude=amp,n_frames=n,valid_frames=int(valid.sum()),theta_deg=float(np.degrees(tb[0])),
               phi_deg=float(np.degrees(pb[0])),theta_iqr_deg=iqr,accepted=accepted)
        rows.append(r)
        if accepted:
            summaries.append((tb[0],pb[0]));cloud.append((th[valid],ph[valid],np.full(valid.sum(),1/valid.sum())))
    if not summaries:raise ValueError(f'No usable objects in {path.name}')
    tp=np.asarray(summaries)
    # For pooled population maps, use one common branch [0,pi), not a per-FOV cut.
    xy=stereo(vectors(tp[:,0],wrap_phi(tp[:,1],0)))
    ex,ey,area=density_grid((-2.05,2.05,-.05,2.05),100)
    counts=np.histogram2d(*xy.T,bins=(ex,ey))[0]
    ct=np.concatenate([x[0] for x in cloud]);cp=np.concatenate([x[1] for x in cloud]);cw=np.concatenate([x[2] for x in cloud])
    cxy=stereo(vectors(ct,wrap_phi(cp,0)));timecounts=np.histogram2d(*cxy.T,bins=(ex,ey),weights=cw)[0]
    result=dict(file=path.name,rows=rows,theta=tp[:,0],phi=tp[:,1],counts=counts,density=counts/len(tp)/area,
                time_density=timecounts/len(tp)/area,ex=ex,ey=ey,mean=mean,spots=spots,metadata=meta)
    print(f'{path.name}: {len(spots)} detected, {len(rows)} bright, {len(tp)} valid objects; vibration={meta["sound"]["sound_on"]}',flush=True)
    return result

def camera_gallery(results):
    fig,axs=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    norm=LogNorm(.03,max(r['density'].max() for r in results))
    for ax,r in zip(axs.flat,results):
        mesh=ax.pcolormesh(r['ex'],r['ey'],np.ma.masked_where(r['density'].T<=0,r['density'].T),norm=norm,cmap='magma',rasterized=True)
        a=np.linspace(0,np.pi,200)
        for deg in (40,60,90):
            rr=2*np.tan(np.radians(deg)/2);ax.plot(rr*np.cos(a),rr*np.sin(a),lw=.7,ls='--' if deg==40 else '-',color='#00a6a6' if deg==40 else '.6')
        ax.set(title=f"{r['file'][21:27]} · n={len(r['theta'])}",aspect='equal',xlim=(-2.05,2.05),ylim=(-.05,2.1),xlabel='u',ylabel='v')
    fig.colorbar(mesh,ax=axs,label='conditional density (sr⁻¹)',shrink=.75)
    fig.suptitle('September camera stacks · one mean-channel orientation per detected object\nVibration off; detection-conditioned exploratory distributions',fontsize=12)
    save(fig,'camera_population_stereographic')
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained');edges=np.radians(np.arange(0,91,3));mid=np.degrees((edges[1:]+edges[:-1])/2)
    for r in results:
        h,_=np.histogram(r['theta'],edges);p=h/h.sum();label=r['file'][21:27]
        axs[0].plot(mid,p/3,label=label,lw=1)
        axs[1].plot(mid,p/(np.pi*(np.cos(edges[:-1])-np.cos(edges[1:]))),label=label,lw=1)
    axs[0].set(xlabel='optical θ (°)',ylabel='probability density (degree⁻¹)',title='Ordinary θ histogram')
    axs[1].axhline(1/np.pi,color='.4',ls=':',label='uniform folded orientation')
    axs[1].set(xlabel='optical θ (°)',ylabel='shell-averaged density (sr⁻¹)',title='Solid-angle corrected θ profile')
    for ax in axs:ax.set_xlim(0,90);ax.axvline(40,color='.6',ls='--',lw=.8)
    axs[1].legend(fontsize=6,ncol=2)
    fig.suptitle('Equal weight per valid object; no fraction of correctly assembled hinges inferred',fontsize=12)
    save(fig,'camera_theta_profiles')

def run_camera():
    results=[];env=detector_namespace()
    for path in sorted(CAMERA_DATA.glob('frame_stack_*.npy')):results.append(survey_file(path,env))
    camera_gallery(results)
    rows=[row for r in results for row in r['rows']]
    with open(ROOT/'camera_spot_summary.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries=[dict(file=r['file'],valid_objects=len(r['theta']),bright_objects=len(r['rows']),
                    frames=FRAMES,sound_on=r['metadata']['sound']['sound_on'],local_background=LOCAL_BACKGROUND,
                    pre_subtracted=r['metadata']['background']['background_subtracted']) for r in results]
    (ROOT/'camera_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    # Save compact arrays, including equal-rod-weight time clouds for later display.
    for r in results:
        np.savez_compressed(ROOT/(Path(r['file']).stem+'_population.npz'),**{k:r[k] for k in ('theta','phi','counts','density','time_density','ex','ey')})
    plt.close('all');return results

if __name__=='__main__':run_camera()
