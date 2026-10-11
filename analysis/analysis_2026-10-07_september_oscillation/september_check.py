"""Focused September-only oscillation review. Run with the polarcam .venv Python.
Reads original NPY/JSON files; writes only into this new analysis directory.
"""
from pathlib import Path
import json,csv,hashlib,sys
import numpy as np
from scipy import ndimage as ndi,signal
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[2] / 'data/hinges/polarcam/2026-09-18')
for d in ['traces','plots']:(B/d).mkdir(exist_ok=True)
SEED=20261007
CONFIG=dict(root=str(ROOT),seed=SEED,random_spots_per_stack=6,selection='uniform without replacement among intensity-detected isolated candidates; never selected by temporal spectrum',detection='2x2 cell mean; DoG sigma 0.8 and 3 cells; local maximum 7x7 cells; threshold median+8*1.4826*MAD; minimum 24 native-pixel candidate spacing; 20-pixel edge margin',bands_hz=[[.4,.85],[1.3,1.8]],spectrum='full-length Hann periodogram, linear detrend, density; no line filtering',amplitude='sinusoid semi-amplitude / saved mean, with offset and linear trend fit; not peak-to-peak',sample='September hinge collection as supplied; no October data',timing='sidecar actual fps; no per-frame sensor timestamps')
(B/'config.json').write_text(json.dumps(CONFIG,indent=2))
def csvsave(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (B/name).open('w',newline='',encoding='utf-8-sig') as fp:w=csv.DictWriter(fp,fieldnames=keys);w.writeheader();w.writerows(rows)
def cells(a):return (a[...,::2,::2]+a[...,::2,1::2]+a[...,1::2,::2]+a[...,1::2,1::2])/4

def load(p):
 m=json.loads(p.with_suffix('.json').read_text());a=np.load(p,mmap_mode='r');assert m['actual']['phase_marker_appended'];assert np.count_nonzero(a[-1])==1 and a[-1,:2,:2].sum()==1
 return a[:-1],m

def extract(a,bounds,origin):
 data=[];cent=[]
 for x0,y0,x1,y1 in bounds:
  c=a[:,y0:y1,x0:x1].astype(float);py=(origin[1]+y0)%2;px=(origin[0]+x0)%2
  data.append(np.stack([c[:,1-py::2,1-px::2].sum((1,2)),c[:,py::2,px::2].sum((1,2)),c[:,py::2,1-px::2].sum((1,2)),c[:,1-py::2,px::2].sum((1,2))],-1))
  cm=cells(c);w=np.maximum(cm-np.percentile(cm,20,axis=(1,2))[:,None,None],0);yy,xx=np.indices(cm.shape[1:]);den=w.sum((1,2));cx=(w*(2*xx+.5)).sum((1,2))/np.maximum(den,1e-12);cy=(w*(2*yy+.5)).sum((1,2))/np.maximum(den,1e-12);cent.append(np.stack([cx+x0,cy+y0],-1))
 return np.stack(data,1),np.stack(cent,1)

def save(p,t,ch,cent,rois,meta,mean):
 pairs=np.stack([ch[:,:,0]+ch[:,:,1],ch[:,:,2]+ch[:,:,3]],-1)
 with np.errstate(divide='ignore',invalid='ignore'):xy=np.stack([(ch[:,:,0]-ch[:,:,1])/pairs[:,:,0],(ch[:,:,2]-ch[:,:,3])/pairs[:,:,1]],-1)
 xy[(pairs<=0).any(-1)]=np.nan
 np.savez_compressed(B/'traces'/f'{p.stem}.npz',time=t,channels=ch,total=ch.sum(-1),pairs=pairs,xy=xy,r=np.linalg.norm(xy,axis=-1),centroid=cent,rois=json.dumps(rois),metadata=json.dumps(meta),mean_image=mean)

def spotrecs():
 for p in sorted(ROOT.glob('spotrec*.npy')):
  if (B/'traces'/f'{p.stem}.npz').exists() and 'force' not in sys.argv:continue
  a,m=load(p);r=m['roi'];ox,oy=r['x'],r['y'];x=int(round(r['cx']-ox));h=a.shape[1]//2*2;x=max(5,min(a.shape[2]-5,x));xw=max(7,min(a.shape[2]-7,x));bounds=[(x-5,0,x+5,h),(xw-7,0,xw+7,h),(a.shape[2]-12,0,a.shape[2]-2,h)];rois=[dict(id=n,bounds_xyxy=b) for n,b in zip(['target','wider','background'],bounds)]
  chunks=[];cents=[]
  for start in range(0,len(a),1024):ch,cent=extract(a[start:start+1024],bounds,(ox,oy));chunks.append(ch);cents.append(cent)
  save(p,np.arange(len(a))/m['actual']['fps'],np.concatenate(chunks),np.concatenate(cents),rois,m,np.mean(a[np.linspace(0,len(a)-1,20).astype(int)],axis=0))
  print('spotrec',p.stem,flush=True)

def stacks():
 candidates=[]
 for idx,p in enumerate(sorted(ROOT.glob('frame_stack*.npy'))):
  if (B/'traces'/f'{p.stem}.npz').exists() and 'force' not in sys.argv:continue
  a,m=load(p);mean=np.mean(a[np.linspace(0,len(a)-1,30).astype(int)],axis=0);cm=cells(mean);dog=ndi.gaussian_filter(cm,.8)-ndi.gaussian_filter(cm,3);threshold=np.median(dog)+8*1.4826*np.median(abs(dog-np.median(dog)));yy,xx=np.where((dog==ndi.maximum_filter(dog,7))&(dog>threshold));pts=np.stack([2*xx,2*yy],1);edge=(pts[:,0]>20)&(pts[:,0]<a.shape[2]-20)&(pts[:,1]>20)&(pts[:,1]<a.shape[1]-20);dist=cKDTree(pts).query(pts,k=2)[0][:,1];eligible=np.where(edge&(dist>=24))[0];rng=np.random.default_rng(SEED+idx);chosen=rng.choice(eligible,min(6,len(eligible)),replace=False)
  rois=[];bounds=[]
  for j,i in enumerate(chosen):
   x,y=map(int,pts[i]);rois.append(dict(id=f'random{j+1}',x=x,y=y,bounds_xyxy=[x-6,y-6,x+6,y+6],detector_score=float(dog[y//2,x//2]),nearest_candidate_distance=float(dist[i])));bounds.append((x-6,y-6,x+6,y+6))
  for i in range(len(pts)):candidates.append(dict(file=p.stem,x=int(pts[i,0]),y=int(pts[i,1]),score=float(dog[pts[i,1]//2,pts[i,0]//2]),threshold=float(threshold),eligible=bool(i in eligible),selected=bool(i in chosen)))
  ch,cent=extract(a,bounds,(0,0));m['analysis_selection']=dict(seed=SEED+idx,all_candidates=len(pts),eligible_candidates=len(eligible),selected=len(chosen));save(p,np.arange(len(a))/m['actual']['fps'],ch,cent,rois,m,mean)
  fig,axs=plt.subplots(2,4,figsize=(13,7));axs[0,0].imshow(mean,cmap='gray',vmin=np.percentile(mean,5),vmax=np.percentile(mean,99.8));axs[0,0].set_title(p.stem,fontsize=9)
  for r in rois:axs[0,0].plot(r['x'],r['y'],'o',mfc='none',mec='cyan',ms=5);axs[0,0].text(r['x'],r['y'],r['id'][-1],color='cyan')
  for ax,r in zip(list(axs.flat)[1:],rois):
   x,y=r['x'],r['y'];c=mean[y-20:y+20,x-20:x+20];ax.imshow(c,cmap='gray');ax.add_patch(plt.Rectangle((14,14),12,12,fill=False,color='cyan'));ax.set_title(f"{r['id']}: ({x},{y})");ax.axis('off')
  axs.flat[-1].axis('off');fig.suptitle(f'Uniform random sample of {len(eligible)} intensity-selected isolated candidates; seed {SEED+idx}');fig.tight_layout();fig.savefig(B/'plots'/f'{p.stem}_selection.png',dpi=140);plt.close(fig)
  print('stack',p.stem,'eligible',len(eligible),'selected',len(chosen),flush=True)
 csvsave('detector_candidates.csv',candidates)

def psd(t,y):return signal.periodogram(y,fs=1/np.mean(np.diff(t)),window='hann',detrend='linear',scaling='density')
def fit(t,y,freq):
 good=np.isfinite(y);tt=t[good];yy=y[good];M=np.column_stack([np.ones(len(tt)),tt-tt.mean(),np.cos(2*np.pi*freq*tt),np.sin(2*np.pi*freq*tt)]);coef=np.linalg.lstsq(M,yy,rcond=None)[0];return float(np.hypot(coef[-1],coef[-2]))
def band(t,y,lo,hi):
 ff,pp=psd(t,y/y.mean());sel=(ff>=lo)&(ff<=hi);fp=float(ff[sel][np.argmax(pp[sel])]);amp=fit(t,y,fp)/y.mean();outside=(ff>=.25)&(ff<=3)&(abs(ff-fp)>max(.15,2/(t[-1]+np.mean(np.diff(t)))));contrast=float(pp[sel].max()/np.median(pp[outside])) if outside.sum()>3 else np.nan
 return fp,amp,contrast

def overview_plots():
 rows=[];roilist=[];spotcards=[]
 for p in sorted((B/'traces').glob('*.npz')):
  z=np.load(p);t=z['time'];total=z['total'];rois=json.loads(str(z['rois']));m=json.loads(str(z['metadata']));isstack=p.stem.startswith('frame_stack');dur=len(t)*np.mean(np.diff(t));ids=range(total.shape[1]) if isstack else [0]
  for j in ids:
   y=total[:,j];fp,amp,contrast=band(t,y,.4,.85);f2,a2,c2=band(t,y,1.3,1.8)
   # Descriptive regularity: ACF at period and independently strongest half-record bins.
   detr=signal.detrend(y);acf=signal.correlate(detr,detr,mode='full',method='fft')[len(y)-1:];acf=acf/np.maximum(acf[0],1e-20);lag=int(round(1/fp/np.mean(np.diff(t))));period_acf=float(acf[lag]) if lag<len(acf) else np.nan
   half=[]
   if dur>=10:
    for ii in [slice(0,len(t)//2),slice(len(t)//2,None)]:half.append(band(t[ii],y[ii],.4,.85)[0])
   row=dict(file=p.stem,roi=rois[j]['id'],duration_s=dur,fps=m['actual']['fps'],resolution_hz=1/dur,slow_peak_bin_hz=fp,slow_fractional_semiamplitude=amp,slow_peak_to_neighbor_psd=contrast,acf_at_slow_period=period_acf,half1_peak_hz=half[0] if half else '',half2_peak_hz=half[1] if half else '',original_band_peak_hz=f2,original_band_fractional_semiamplitude=a2,original_band_peak_to_neighbor_psd=c2,fixed_1p55_fractional_semiamplitude=fit(t,y,1.55)/y.mean(),cv=y.std()/y.mean(),r_sd=np.nanstd(z['r'][:,j]),centroid_sd_max_px=np.max(z['centroid'][:,j].std(0)),r_slow_semiamplitude=fit(t,z['r'][:,j],fp),invalid_r_fraction=np.mean(~np.isfinite(z['r'][:,j])))
   row['automatic_description']='too short to identify periodicity' if isstack else 'prominent slow periodic candidate' if contrast>8 and period_acf>.25 and dur*fp>=5 else 'no clear regular 0.4-0.85 Hz component by screening'
   if not isstack and (row['invalid_r_fraction']>.1 or np.mean(y[-max(1,len(y)//10):])<.1*np.mean(y[:max(1,len(y)//10)])):
    row['automatic_description']='signal loss / step: spectral candidate inconclusive'
   if not isstack:
    row['wider_aperture_slow_semiamplitude']=fit(t,total[:,1],fp)/total[:,1].mean();row['background_slow_semiamplitude']=fit(t,total[:,2],fp)/total[:,2].mean()
   rows.append(row);roilist.append(dict(file=p.stem,source=str(ROOT/p.with_suffix('.npy').name),**rois[j]))
  if isstack:
   fig,axs=plt.subplots(6,2,figsize=(12,13))
   for j in range(6):
    y=total[:,j];axs[j,0].plot(t,y/y.mean(),lw=1);axs[j,0].set_ylabel(rois[j]['id']+' I/mean');ff,pp=psd(t,y/y.mean());axs[j,1].semilogy(ff,pp);axs[j,1].set_xlim(0,5)
    for lo,hi,col in [(.4,.85,'teal'),(1.3,1.8,'orange')]:axs[j,1].axvspan(lo,hi,color=col,alpha=.15)
   axs[-1,0].set_xlabel('Seconds');axs[-1,1].set_xlabel('Hz; bin spacing 0.481 Hz');fig.suptitle(p.stem+' — six randomly selected spots\n2.078 seconds: fluctuations can be seen; periodicity cannot be established');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(B/'plots'/f'{p.stem}_analysis.png',dpi=140);plt.close(fig)
  else:
   row=rows[-1];fig,axs=plt.subplots(3,2,figsize=(12,9));step=max(1,int(m['actual']['fps']*.02));n=len(t)//step;tb=t[:n*step].reshape(n,step).mean(1);yb=total[:n*step,0].reshape(n,step).mean(1)
   axs[0,0].plot(tb,yb/total[:,0].mean(),lw=.8);axs[0,0].set_ylabel('I / mean (~20 ms means)');axs[0,0].set_title(f"Duration {dur:.1f}s; CV {row['cv']*100:.1f}%")
   ff,pp=psd(t,total[:,0]/total[:,0].mean());axs[0,1].semilogy(ff,pp,label='target');ff,pwide=psd(t,total[:,1]/total[:,1].mean());axs[0,1].semilogy(ff,pwide,alpha=.7,label='wider aperture');axs[0,1].set_xlim(0,3);axs[0,1].legend(fontsize=8)
   for k,label in enumerate(['X','Y']):axs[1,0].plot(tb,np.nanmean(z['xy'][:n*step,0,k].reshape(n,step),1),lw=.7,label=label)
   axs[1,0].plot(tb,np.nanmean(z['r'][:n*step,0].reshape(n,step),1),lw=.7,label='r');axs[1,0].legend(fontsize=8)
   for key in ['r']:
    yy=z[key][:,0];good=np.isfinite(yy);ff,pp=psd(t,np.interp(t,t[good],yy[good]));axs[1,1].semilogy(ff,pp,label=key);axs[1,1].set_xlim(0,3);axs[1,1].set_ylabel('r PSD / Hz')
   for k,label in enumerate(['cx','cy']):axs[2,0].plot(tb,z['centroid'][:n*step,0,k].reshape(n,step).mean(1),lw=.8,label=label)
   axs[2,0].legend(fontsize=8);axs[2,0].set_ylabel('Centroid (local raw px)');axs[2,1].plot(np.arange(len(acf))*np.mean(np.diff(t)),acf);axs[2,1].set_xlim(0,8);axs[2,1].set_ylabel('Linear-detrended intensity ACF');axs[2,1].set_xlabel('Lag (s)')
   for ax in axs[:2,1]:
    ax.axvspan(.4,.85,color='teal',alpha=.15);ax.axvspan(1.3,1.8,color='orange',alpha=.15);ax.set_xlabel('Hz')
   for ax in axs[:,0]:ax.set_xlabel('Seconds')
   fig.suptitle(p.stem+'\n'+row['automatic_description']+'; motion remains an alternative explanation',fontsize=13);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(B/'plots'/f'{p.stem}_analysis.png',dpi=140);plt.close(fig)
   spotcards.append((p.stem,t,total[:,0],row))
 csvsave('oscillation_metrics.csv',rows);csvsave('selected_rois.csv',roilist)
 for page in range(4):
  subset=spotcards[page*6:(page+1)*6];fig,axs=plt.subplots(len(subset),2,figsize=(12,2.35*len(subset)),squeeze=False)
  for k,(fid,t,y,row) in enumerate(subset):
   step=30;n=len(t)//step;axs[k,0].plot(t[:n*step].reshape(n,step).mean(1),y[:n*step].reshape(n,step).mean(1)/y.mean(),lw=.7);axs[k,0].set_title(fid+f" | {row['duration_s']:.1f}s",fontsize=9);axs[k,0].set_ylabel('I / mean');ff,pp=psd(t,y/y.mean());axs[k,1].semilogy(ff,pp,lw=1);axs[k,1].set_xlim(0,3);axs[k,1].axvspan(.4,.85,color='teal',alpha=.15);axs[k,1].axvspan(1.3,1.8,color='orange',alpha=.15);axs[k,1].set_title(f"Slow peak bin {row['slow_peak_bin_hz']:.3f}Hz | semi-amplitude {100*row['slow_fractional_semiamplitude']:.1f}%",fontsize=9)
  axs[-1,0].set_xlabel('Seconds');axs[-1,1].set_xlabel('Hz');fig.suptitle('September spot recordings — intensity and full-record spectra\nTeal: 0.4–0.85 Hz; orange: original 1.3–1.8 Hz band. No line filtering.');fig.tight_layout(rect=(0,0,1,.95));fig.savefig(B/'plots'/f'spotrec_overview_{page+1}.png',dpi=140);plt.close(fig)
 print('Finished metrics',len(rows),flush=True)
if __name__=='__main__':
 mode=sys.argv[1] if len(sys.argv)>1 else 'all'
 if mode in ['all','spotrecs']:spotrecs()
 if mode in ['all','stacks']:stacks()
 if mode in ['all','plots']:overview_plots()
