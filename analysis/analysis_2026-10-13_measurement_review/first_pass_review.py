"""Read-only first-pass synthesis: cached field detections and recorded spot ROIs.

Run with repository Python. Outputs are local; acquisitions are never altered.
Temporal markers are excluded and clipping is checked against verified raw peaks.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import sys
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'first_pass_review'
ROOT=HERE.parents[1]/'data/hinges/polarcam'
A,B,C=.09779825985607313,.011480772916810558,.10137635007725046
RMAX=C/(A+B)
COLORS={'Hinges':'#087e8b','Gold':'#c07a11'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def xy(ch):
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.stack(((ch[...,0]-ch[...,1])/(ch[...,0]+ch[...,1]),
                         (ch[...,2]-ch[...,3])/(ch[...,2]+ch[...,3])),axis=-1)
def theta(r):return float(np.degrees(np.arcsin(np.sqrt(A*r/(C-B*r))))) if 0<=r<=RMAX else None
def blocked(ch,b):return ch[:len(ch)//b*b].reshape(-1,b,4).mean(1)
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def metrics(ch,fps):
    out={}
    for b in (1,8,80):
        z=blocked(ch,b);v=xy(z);r=np.linalg.norm(v,axis=1);s=z.sum(1)
        assert np.isfinite(v).all()
        out[str(b)]={'dt_ms':1000*b/fps,'xy_std':v.std(0).tolist(),
          'xy_p96':np.ptp(np.percentile(v,[2,98],axis=0),axis=0).tolist(),
          'r_median':float(np.median(r)),'r_std':float(r.std()),
          'r_p96':float(np.ptp(np.percentile(r,[2,98]))),
          'intensity_mean':float(s.mean()),'intensity_cv':float(s.std()/s.mean()),
          'intensity_p96_fraction':float(np.ptp(np.percentile(s,[2,98]))/np.median(s)),
          'above_model_fraction':float(np.mean(r>RMAX))}
    v=xy(ch.mean(0));r=float(np.linalg.norm(v))
    out.update(xy_of_mean=v.tolist(),r_of_mean=r,theta_of_mean=theta(r),
      duration_s=len(ch)/fps,frames=len(ch),fps=fps)
    return out


def extract(rec):
    p=ROOT/rec['path'];m=read(p.with_suffix('.json'));stat=p.stat()
    a=np.load(p,mmap_mode='r',allow_pickle=False)
    n=m['actual']['frames'];assert len(a)==n+int(m['actual']['phase_marker_appended'])
    roi=m['roi'];ox,oy=int(roi['x']),int(roi['y']);size=int(roi['win_raw'])
    left=max(0,min(a.shape[2]-size,round(roi['cx']-ox)-size//2))
    top=max(0,min(a.shape[1]-size,round(roi['cy']-oy)-size//2))
    peakmeta=m['fetch_frames_output']['raw_peak'] if 'fetch_frames_output' in m else m['actual']['raw_peak']
    peakpath=p.parent/peakmeta['path'];assert sha(peakpath)==peakmeta['sha256'] and peakmeta['frames']==n
    peak=np.load(peakpath,allow_pickle=False);assert peak.shape==a.shape[1:]
    rawmax=int(peak[top:top+size,left:left+size].max())
    bgpath=ROOT/'2026-10-12/backgrounds/used'/f'{rec["bg_hash"]}.npy'
    assert sha(bgpath)==rec['bg_hash'] and rec['subtracted']
    signature=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,metadata_sha256=sha(p.with_suffix('.json')),
                   raw_peak_sha256=sha(peakpath),background_sha256=rec['bg_hash'],bounds=[ox+left,oy+top,size])
    cache=OUT/f'trace_{rec["stamp"]}.npz';meta=cache.with_suffix('.json')
    if cache.exists() and meta.exists() and read(meta)==signature:
        z=np.load(cache);ch=z['channels'];mean=z['mean']
    else:
        chunks=[];total=np.zeros((size,size),np.float64)
        for start in range(0,n,2048):
            crop=np.asarray(a[start:min(start+2048,n),top:top+size,left:left+size],dtype=float)
            total+=crop.sum(0)
            chunks.append(np.stack([crop[:,(pr-oy-top)%2::2,(pc-ox-left)%2::2].mean((1,2))
                         for pr,pc in ((1,1),(0,0),(0,1),(1,0))],axis=-1))
        ch=np.concatenate(chunks);mean=total/n
        np.savez_compressed(cache,channels=ch,mean=mean)
        meta.write_text(json.dumps(signature),encoding='utf-8')
    assert (stat.st_size,stat.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
    assert len(ch)==n
    # Independent check of extraction parity and synthetic-marker exclusion.
    direct=np.array([mean[(pr-oy-top)%2::2,(pc-ox-left)%2::2].mean() for pr,pc in ((1,1),(0,0),(0,1),(1,0))])
    np.testing.assert_allclose(ch.mean(0),direct,atol=1e-9,rtol=1e-12)
    result=dict(stamp=rec['stamp'],time=rec['time'],type=rec['type'],note=rec['note'],
      sample=rec['sample'],role=rec['role'],path=rec['path'],bounds=signature['bounds'],
      bg_hash=rec['bg_hash'],exposure_ms=rec['exposure_ms'],raw_peak=rawmax,saturated=rawmax>=4094,
      camera_roi_peak=int(peak.max()),source_frame_stack=rec['source_frame_stack'],
      metrics=metrics(ch,rec['fps']),provenance=signature)
    return result,ch,mean


def fields(d):
    summary={};fieldrows=[];bins=np.arange(0,1.4201,.02)
    for method in d['methods']:
        summary[method]={}
        for selection in ['all','spots50']:
            summary[method][selection]={}
            for group in ['Hinges','Gold']:
                values=[];rows=[]
                for rec in d['records']:
                    if rec['group']!=group:continue
                    r=np.array([s['r'] for s in rec['methods'][method] if selection=='all' or s['keep']['common'][selection]])
                    values.extend(r)
                    rows.append(dict(method=method,selection=selection,group=group,index=rec['index'],n=len(r),
                       median=float(np.median(r)),fraction_high=float(np.mean(r>=.6)),fraction_low=float(np.mean(r<.2))))
                fieldrows.extend(rows);v=np.array(values);hist,_=np.histogram(v,bins)
                assert hist.sum()==len(v)
                summary[method][selection][group]=dict(n=len(v),median=float(np.median(v)),
                    fraction_high=float(np.mean(v>=.6)),fraction_low=float(np.mean(v<.2)),
                    median_field_range=[min(r['median'] for r in rows),max(r['median'] for r in rows)],
                    histogram=hist.tolist())
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True,sharey=True)
    for j,(method,meta) in enumerate(d['methods'].items()):
        for i,selection in enumerate(['all','spots50']):
            ax=axes[i,j]
            for g in COLORS:
                s=summary[method][selection][g]
                ax.stairs(100*np.array(s['histogram'])/s['n'],bins,color=COLORS[g],label=f'{"DNA" if g=="Hinges" else "Gold"} n={s["n"]:,}')
            ax.set_title(meta['label']+'\n'+('All eligible spots' if i==0 else 'Same low-BG mask'))
            ax.set(xlim=(0,1.05),xlabel='Anisotropy radius r',ylabel='Spots per 0.02 bin (%)')
            ax.axvline(RMAX,ls=':',c='.5',lw=1);ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    fig.suptitle('12 October: high-r DNA enrichment persists across detectors and the low-background mask',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'population.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),sharey=True)
    for ax,(method,meta) in zip(axes,d['methods'].items()):
        for k,g in enumerate(COLORS):
            rows=[r for r in fieldrows if r['method']==method and r['selection']=='spots50' and r['group']==g]
            ax.scatter(k+np.linspace(-.18,.18,len(rows)),[r['median'] for r in rows],c=COLORS[g],s=35)
        ax.set(xticks=[0,1],xticklabels=['DNA (16 fields)','Gold (14 fields)'],ylim=(0,1),title=meta['label'],ylabel='Field median r')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Each dot is one field, after the common low-BG mask — not an independent preparation',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'fields.png',dpi=160);plt.close(fig)
    return summary,fieldrows


def trace_plot(r,ch,mean,label):
    fps=r['metrics']['fps'];z=blocked(ch,80);v=xy(z);t=(np.arange(len(z))+.5)*80/fps
    fig,ax=plt.subplots(1,4,figsize=(14,3.2),gridspec_kw={'width_ratios':[.65,1,1.5,1.1]})
    ax[0].imshow(mean,cmap='gray',vmin=0);ax[0].set_title('Mean saved ROI');ax[0].set_axis_off()
    ax[1].scatter(v[:,0],v[:,1],s=2,c=t,cmap='viridis',rasterized=True)
    ang=np.linspace(0,2*np.pi,300);ax[1].plot(RMAX*np.cos(ang),RMAX*np.sin(ang),':',c='.6',lw=.8)
    ax[1].set(xlim=(-1,1),ylim=(-1,1),xlabel='X',ylabel='Y',aspect='equal',title='100 ms channel averages')
    ax[2].plot(t,v[:,0],lw=.6,label='X');ax[2].plot(t,v[:,1],lw=.6,label='Y');ax[2].legend(frameon=False,ncol=2,loc='upper right')
    ax[2].set(xlabel='Time (s)',ylim=(-1,1),title='Anisotropy vs time')
    ax[3].plot(t,np.linalg.norm(v,axis=1),lw=.7,c='#7e4084',label='r')
    ax[3].plot(t,z.sum(1)/np.median(z.sum(1)),lw=.6,c='.5',alpha=.6,label='I / median I')
    ax[3].set(xlabel='Time (s)',title='Radius and relative intensity');ax[3].legend(frameon=False,fontsize=8)
    fig.suptitle(f'{label} | {r["time"]} | raw peak {r["raw_peak"]}/4095'+(' | SATURATED: diagnostic only' if r['saturated'] else ''),fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/f'spot_{r["stamp"]}.png',dpi=140);plt.close(fig)


def interference(rows,traces):
    checks=[r for r in rows if r['type']=='Stationary check'];assert len(checks)==3
    assert len({(tuple(r['bounds']),r['bg_hash'],r['exposure_ms'],r['metrics']['fps']) for r in checks})==1
    assert not any(r['saturated'] for r in checks)
    mixed,off,on=checks
    variants=[('Mixed: first 5 s OFF, then ON (notes)',mixed,traces[mixed['stamp']],None),
              ('OFF, z knob moved',off,traces[off['stamp']],False),
              ('ON, z knob moved',on,traces[on['stamp']],True)]
    fig,axes=plt.subplots(3,3,figsize=(13,8),sharex='col')
    for row,(label,r,ch,state) in enumerate(variants):
        fps=r['metrics']['fps'];z=blocked(ch,8);v=xy(z);t=(np.arange(len(z))+.5)*8/fps
        axes[row,0].plot(t,z.sum(1),lw=.8,c='.3');axes[row,0].set(ylabel='Sum of channel means (DN)',title=label)
        axes[row,1].plot(t,v[:,0],lw=.8,label='X');axes[row,1].plot(t,v[:,1],lw=.8,label='Y');axes[row,1].set(ylim=(-1,1),ylabel='Anisotropy');axes[row,1].legend(frameon=False)
        f,p=signal.welch(xy(ch),fs=fps,nperseg=min(len(ch),2048),axis=0)
        axes[row,2].semilogy(f,p[:,0],lw=.7,label='X');axes[row,2].semilogy(f,p[:,1],lw=.7,label='Y');axes[row,2].set(xlim=(0,fps/2),ylabel='PSD (anisotropy²/Hz)')
        if row==0:
            for ax in axes[row,:2]:ax.axvline(5,c='#c64c44',ls='--',lw=1)
    maxI=max(ax.get_ylim()[1] for ax in axes[:,0]);maxP=max(ax.get_ylim()[1] for ax in axes[:,2]);minP=min(ax.get_ylim()[0] for ax in axes[:,2])
    for ax in axes[:,0]:ax.set_ylim(0,maxI)
    for ax in axes[:,2]:ax.set_ylim(minP,maxP)
    for ax in axes[-1,:2]:ax.set_xlabel('Time (s)')
    axes[-1,2].set_xlabel('Frequency (Hz; raw frames)')
    fig.suptitle('Cancellation check: same ROI / background / 1.194 ms exposure / 798.79 fps; manual z sweeps are not matched motion',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'cancellation.png',dpi=160);plt.close(fig)
    om=off['metrics']['8'];nm=on['metrics']['8']
    mixch=traces[mixed['stamp']];fps=mixed['metrics']['fps'];cut=round(5*fps)
    # Leave 0.25 seconds on each side of the approximate manual switch.
    trim=round(.25*fps);pre=metrics(mixch[:cut-trim],fps);post=metrics(mixch[cut+trim:],fps)
    return dict(off_stamp=off['stamp'],on_stamp=on['stamp'],off=om,on=nm,
        on_off_xy_std=np.array(nm['xy_std'])/om['xy_std'],
        on_off_intensity_cv=nm['intensity_cv']/om['intensity_cv'],
        mixed_pre=pre,mixed_post=post,
        mixed_on_off_xy_std=np.array(post['8']['xy_std'])/pre['8']['xy_std'],
        assignment='Off/on from recorded notes. Drive metadata is Unknown. Approximate 5 s mixed-record switch is not a hardware event.',
        limits='Uncontrolled manual z sweeps; cannot estimate intrinsic angular resolution or prove complete cancellation. Actual drive frequency/amplitude not recorded.')


def build():
    OUT.mkdir(exist_ok=True);manifest=read(HERE/'manifest.json');mask=read(HERE/'low_background_mask/results.json')
    rows=[];traces={};means={};counts={'Gold':0,'Hinges':0}
    for rec in manifest['records']:
        if rec['type'] not in ('Spot recording','Stationary check'):continue
        r,ch,mean=extract(rec);traces[r['stamp']]=ch;means[r['stamp']]=mean
        if r['type']=='Spot recording':
            group='Gold' if r['stamp']<'205000' else 'Hinges';counts[group]+=1
            r['group']=group;r['index']=counts[group];r['label']=('DNA' if group=='Hinges' else 'Gold')+f' {counts[group]}'
            trace_plot(r,ch,mean,r['label'])
        rows.append(r)
        print(r['stamp'],r.get('label','Check'),'raw',r['raw_peak'],'r',round(r['metrics']['r_of_mean'],3),'SD100',r['metrics']['80']['xy_std'],flush=True)
    field_summary,fieldrows=fields(mask)
    result=dict(records=rows,fields=field_summary,field_rows=fieldrows,cancellation=interference(rows,traces),
      mask_threshold=mask['thresholds']['spots50'],mask_area=mask['masks']['common']['spots50'][0]['fraction'],
      r_max=RMAX,coefficients=dict(A=A,B=B,C=C),provenance={'manifest_sha256':sha(HERE/'manifest.json'),
      'field_analysis_sha256':sha(HERE/'low_background_mask/results.json'),'builder_sha256':sha(Path(__file__))},
      methods='Field distributions reuse all eligible 1x radii with optional common low-BG mask; no 2x-survival or r-limit rejection. Spot traces use recorded 20x20 selected aperture, original subtracted channels, sensor parity, and verified raw-peak veto at 4094. 100 ms channel averages for display; 10 ms for cancellation traces. Spectra use unaveraged traces. Water Fresnel+BFP+0.39-hole model only for indicative theta.')
    (OUT/'results.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False),encoding='utf-8')
    print('SUMMARY',json.dumps(clean({'fields':field_summary,'cancellation':result['cancellation']})),flush=True)


if __name__=='__main__':build()
