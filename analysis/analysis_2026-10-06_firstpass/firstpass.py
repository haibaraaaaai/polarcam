"""Read-only Polarcam first pass. All outputs remain next to this script."""
from pathlib import Path
import json, csv, sys, hashlib
import numpy as np
import cv2
from scipy import ndimage as ndi, signal, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
OLD=(Path(__file__).resolve().parents[2] / 'data/hinges/polarcam/import_2026-10-09/remember to correct save pathing from hugh')
RUNS=(Path(__file__).resolve().parents[2] / 'data/hinges/polarcam/import_2026-10-09/runs')
for sub in ['traces','plots']: (BASE/sub).mkdir(exist_ok=True)
CONFIG={'roots':[str(OLD),str(RUNS)],'channel_order':['I0','I90','I45','I135'],'apertures_raw_px':[12,20],'detection':'2x2 cell mean, DoG sigma 0.8/3 cells; local maxima 7 cells; robust threshold median+8*MAD; reject another candidate within 12 raw px; strongest isolated up to 12','background':'four low-DoG patches, spatially separated; no further subtraction','frequency_search_hz':[1.2,1.9],'spectrum':'full-record Hann periodogram, linear detrend, density units DN^2/Hz; no line removal','amplitude':'least squares constant+linear trend+sin+cos; semi-amplitude / unmodified trace mean','timing':'sidecar actual fps; AVI header fps conditional; diagnostic fits use host receipt timestamps','raw_data_policy':'no original writes; sibling sidecars resolved locally, stale absolute paths ignored'}
(BASE/'config.json').write_text(json.dumps(CONFIG,indent=2))
def dumpcsv(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def cells(a):return (a[...,0::2,0::2]+a[...,0::2,1::2]+a[...,1::2,0::2]+a[...,1::2,1::2])/4

def detect(mean,n=12):
 h,w=mean.shape; m=cells(mean.astype(float));dog=ndi.gaussian_filter(m,.8)-ndi.gaussian_filter(m,3)
 th=np.median(dog)+8*1.4826*np.median(abs(dog-np.median(dog)))
 yy,xx=np.where((dog==ndi.maximum_filter(dog,7))&(dog>th))
 cand=sorted([(float(dog[y,x]),2*y,2*x) for y,x in zip(yy,xx) if 12<2*y<h-12 and 12<2*x<w-12],reverse=True)
 rois=[]
 for v,y,x in cand:
  if any(0<(x-x2)**2+(y-y2)**2<24**2 for _,y2,x2 in cand):continue
  rois.append({'id':f'rod{len(rois)+1:02d}','kind':'rod','x':int(x),'y':int(y),'size':12})
  if len(rois)==n:break
 # background candidates avoid every detected peak, not only selected rods
 bg=[]
 for y in range(12,h-12,12):
  for x in range(12,w-12,12):
   if any((x-x2)**2+(y-y2)**2<24**2 for _,y2,x2 in cand):continue
   bg.append((float(np.mean(mean[y-6:y+6,x-6:x+6])),y,x))
 for v,y,x in sorted(bg):
  if any((x-r['x'])**2+(y-r['y'])**2<32**2 for r in rois if r['kind']=='background'):continue
  rois.append({'id':f'bg{sum(r["kind"]=="background" for r in rois)+1}','kind':'background','x':int(x),'y':int(y),'size':12})
  if sum(r['kind']=='background' for r in rois)==4:break
 return rois,th,len(cand)

def roi_measure(a,rois,origin):
 if a.ndim==2:a=a[None]
 n=len(a);ch=np.empty((n,len(rois),4));cen=np.empty((n,len(rois),3));qual=np.empty((n,len(rois),3))
 for j,r in enumerate(rois):
  size=r['size'];x0=int(r['x']-size//2);y0=int(r['y']-size//2)
  crop=a[:,y0:y0+size,x0:x0+size].astype(float)
  py=(origin[1]+y0)%2;px=(origin[0]+x0)%2
  ch[:,j]=np.stack([crop[:,1-py::2,1-px::2].sum((1,2)),crop[:,py::2,px::2].sum((1,2)),crop[:,py::2,1-px::2].sum((1,2)),crop[:,1-py::2,px::2].sum((1,2))],axis=-1)
  cm=cells(crop);weight=np.maximum(cm-np.percentile(cm,20,axis=(1,2))[:,None,None],0);den=weight.sum((1,2));yy,xx=np.indices(cm.shape[1:]);xx=xx*2+x0+.5;yy=yy*2+y0+.5
  cx=(weight*xx).sum((1,2))/np.maximum(den,1e-12);cy=(weight*yy).sum((1,2))/np.maximum(den,1e-12)
  cen[:,j]=np.stack([cx,cy,np.sqrt((weight*((xx-cx[:,None,None])**2+(yy-cy[:,None,None])**2)).sum((1,2))/np.maximum(den,1e-12))],-1)
  qual[:,j]=np.stack([(crop==0).mean((1,2)),(crop>=4094).mean((1,2)),crop.max((1,2))],-1)
 return ch,cen,qual

def save_trace(fid,path,t,ch,rois,info,mean=None,centroid=None,quality=None,extra=None):
 pair=np.stack([ch[:,:,0]+ch[:,:,1],ch[:,:,2]+ch[:,:,3]],-1)
 with np.errstate(divide='ignore',invalid='ignore'):
  xy=np.stack([(ch[:,:,0]-ch[:,:,1])/pair[:,:,0],(ch[:,:,2]-ch[:,:,3])/pair[:,:,1]],-1)
 xy[(pair<=0).any(-1)]=np.nan
 args=dict(time_s=t,channels=ch,total=ch.sum(-1),pairs=pair,xy=xy,r=np.linalg.norm(xy,axis=-1),rois_json=json.dumps(rois),info_json=json.dumps(dict(info,source_path=str(path))))
 if centroid is not None:args['centroid_width']=centroid
 if quality is not None:args['zero_sat_max']=quality
 if extra:args.update(extra)
 np.savez_compressed(BASE/'traces'/f'{fid}.npz',**args)
 if mean is not None:
  fig,ax=plt.subplots(figsize=(12,6));ax.imshow(mean,cmap='gray',vmin=np.percentile(mean,10),vmax=np.percentile(mean,99.8))
  for r in rois:
   if r['id'].endswith('_large'):continue
   col='cyan' if r['kind']=='rod' else 'orange';ax.add_patch(plt.Rectangle((r['x']-r['size']/2,r['y']-r['size']/2),r['size'],r['size'],fill=False,color=col));ax.text(r['x']+6,r['y'],r['id'],color=col,fontsize=7)
  ax.set_title(fid+' | local raw-pixel coordinates');fig.tight_layout();fig.savefig(BASE/'plots'/f'{fid}_rois.png',dpi=150);plt.close(fig)
 print('extracted',fid,len(t),'frames',len(rois),'ROIs',flush=True)

def avi():
 p=RUNS/'datasets/temp/recording_2026-10-02_2.avi';fid=p.stem
 if (BASE/'traces'/f'{fid}.npz').exists():return
 cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
 for i in np.linspace(0,n-1,80).astype(int):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,f=cap.read()
  if ok:frames.append(f[:,:,0].astype(float))
 mean=np.mean(frames,axis=0);rois,th,nc=detect(mean)
 rois += [dict(r,id=r['id']+'_large',size=20) for r in rois[:3] if r['kind']=='rod']
 cap.release();cap=cv2.VideoCapture(str(p));cc=[];ce=[];qq=[];globalmean=[];duplicates=0;prev=None;hashes=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  a=f[:,:,0];c,e,q=roi_measure(a,rois,(0,0));cc.append(c[0]);ce.append(e[0]);qq.append(q[0]);globalmean.append(a.mean());h=hashlib.blake2b(a.tobytes(),digest_size=8).hexdigest();hashes.append(h);duplicates+=int(h==prev);prev=h
 cap.release();nread=len(cc)
 save_trace(fid,p,np.arange(nread)/fps,np.array(cc),rois,dict(fps=fps,timing='AVI header only; physical Hz conditional',units='8-bit MJPEG encoded DN sum',origin_known=False,origin_assumed=[0,0],background='unknown',duplicate_adjacent=duplicates,unique_frames=len(set(hashes)),candidates=nc,detection_threshold=th,frames_header=n),mean,np.array(ce),np.array(qq),{'global_mean':np.array(globalmean)})

def arrayfile(p):
 fid=p.stem if p.stem!='capture_maxfps_15x15' else p.parent.name
 if (BASE/'traces'/f'{fid}.npz').exists():return
 m=json.loads(p.with_suffix('.json').read_text());a=np.load(p,mmap_mode='r');marker=m['actual'].get('phase_marker_appended',False)
 if marker:
  assert np.count_nonzero(a[-1])==1 and a[-1,:2,:2].sum()==1
  a=a[:-1]
 roi=m.get('fetch_frames_output',{}).get('roi',m.get('roi',{}));origin=(int(roi.get('OffsetX',roi.get('x',0))),int(roi.get('OffsetY',roi.get('y',0))));fps=m['actual']['fps']
 mean=np.mean(a[np.linspace(0,len(a)-1,min(30,len(a))).astype(int)],axis=0,dtype=float)
 if a.shape[1]>100:
  rois,th,nc=detect(mean);rois += [dict(r,id=r['id']+'_large',size=20) for r in rois[:3] if r['kind']=='rod']
 else:
  rm=m['roi'];size=min(a.shape[1]//2*2,10);x=int(round(rm['cx']-origin[0]));y=int(round(rm['cy']-origin[1]));x=max(size//2,min(a.shape[2]-size//2,x));y=max(size//2,min(a.shape[1]-size//2,y));rois=[{'id':'target','kind':'rod','x':x,'y':y,'size':size},{'id':'bg1','kind':'background','x':a.shape[2]-size,'y':size//2,'size':size}];th=None;nc=None
 ch=[];cen=[];qual=[];glob=[]
 for start in range(0,len(a),max(1,min(256,10000000//(a.shape[1]*a.shape[2])))):
  chunk=a[start:start+max(1,min(256,10000000//(a.shape[1]*a.shape[2])))];c,e,q=roi_measure(chunk,rois,origin);ch.append(c);cen.append(e);qual.append(q);glob.extend(chunk[:,::4,::4].mean((1,2)))
 timing=m['actual'].get('timing',m['actual'].get('timing_snapshot',{}))
 save_trace(fid,p,np.arange(len(a))/fps,np.concatenate(ch),rois,dict(fps=fps,timing='sidecar actual fps; no per-frame sensor timestamps',units='saved uint16 DN sum',origin_known=True,origin_xy=origin,background_subtracted=m.get('background',{}).get('background_subtracted'),exposure_us=timing.get('exposure_us'),metadata=m,marker_removed=marker,candidates=nc,detection_threshold=th),mean,np.concatenate(cen),np.concatenate(qual),{'global_mean_sparse':np.array(glob)})

def diagnostic(p):
 fid=p.stem
 if (BASE/'traces'/f'{fid}.npz').exists():return
 z=np.load(p);s=z['samples'];m=json.loads(str(z['metadata_json']));i=m['initial_settings'];t=s[:,0]-s[0,0]
 rois=[dict(id='target',kind='rod',**i['analysis_roi'])]
 save_trace(fid,p,t,s[:,None,5:9],rois,dict(fps=i['timing_readback']['fps'],timing='host SDK receipt timestamps, not exposure timestamps',units='channel mean DN; total is sum of four channel means',metadata=m,background_subtracted=i['background_requested'],exposure_us=i['timing_readback']['exposure_us']),extra={'diagnostic_samples':s,'diagnostic_columns':z['columns']})

def spectrum(t,y):
 fs=(len(t)-1)/(t[-1]-t[0]);tu=np.linspace(t[0],t[-1],len(t));yu=np.interp(tu,t,y);return signal.periodogram(yu,fs=fs,window='hann',detrend='linear',scaling='density')
def fit(t,y,f):
 good=np.isfinite(y);t=t[good];y=y[good];tc=t-t.mean();M=np.column_stack([np.ones(len(t)),tc,np.cos(2*np.pi*f*t),np.sin(2*np.pi*f*t)]);b=np.linalg.lstsq(M,y,rcond=None)[0];res=y-M@b;return float(np.hypot(b[2],b[3])),float(np.arctan2(-b[3],b[2])),float(np.std(res)),b

def peak(t,y,lo=1.2,hi=1.9):
 fs=(len(t)-1)/(t[-1]-t[0]);hi=min(hi,fs/2*.999)
 if hi<lo:lo=.9
 # full-grid 8x padded spectrum only locates candidate; duration remains resolution
 tu=np.linspace(t[0],t[-1],len(t));yu=np.interp(tu,t,y);f,p=signal.periodogram(yu,fs=fs,window='hann',detrend='linear',nfft=8*len(t));sel=(f>=lo)&(f<=hi);return float(f[sel][np.argmax(p[sel])])

def analyse():
 rows=[];roirows=[];files=[]
 for p in sorted((BASE/'traces').glob('*.npz')):
  z=np.load(p);t=z['time_s'];info=json.loads(str(z['info_json']));rois=json.loads(str(z['rois_json']));fs=(len(t)-1)/(t[-1]-t[0]);dur=t[-1]-t[0]+1/fs
  rodidx=[j for j,r in enumerate(rois) if r['kind']=='rod' and not r['id'].endswith('_large')]
  # frequency from median of rod-normalized spectra, independent of background ROIs
  ps=[]
  for j in rodidx:
   ff,pp=spectrum(t,z['total'][:,j]/np.mean(z['total'][:,j]));ps.append(pp)
  aggregate=np.median(ps,axis=0);sel=(ff>=1.2)&(ff<=min(1.9,fs/2));fcommon=float(ff[sel][np.argmax(aggregate[sel])])
  fi=dict(file_id=p.stem,source_path=info['source_path'],duration_s=dur,frames=len(t),inferred_sample_rate_hz=fs,settings_fps=info['fps'],frequency_resolution_hz=1/dur,common_bin_hz=fcommon,timing=info['timing'],units=info['units'],background_subtracted=info.get('background_subtracted','unknown'),exposure_us=info.get('exposure_us','unknown'));files.append(fi)
  for j,r in enumerate(rois):
   total=z['total'][:,j];fpeak=peak(t,total);amp,phase,res,_=fit(t,total,fpeak);a155,ph155,_,_=fit(t,total,1.55)
   freq,psd=spectrum(t,total);noise=(freq>.5)&(freq<min(5,fs/2))&(abs(freq-fpeak)>max(.2,2/dur));contrast=float(np.interp(fpeak,freq,psd)/np.median(psd[noise])) if noise.sum()>3 else np.nan
   row=dict(fi,roi_id=r['id'],kind=r['kind'],peak_hz=fpeak,peak_cycles_per_frame=fpeak/fs,mean_total=float(total.mean()),cv=float(total.std()/total.mean()),semiamplitude=float(amp),fractional_semiamplitude=amp/total.mean(),phase_rad=phase,at_1p55_fractional_semiamplitude=a155/total.mean(),at_1p55_phase_rad=ph155,peak_psd_to_neighbor_median=contrast,evidence='short/aliased: inconclusive' if dur<5 or fs<4 else 'candidate spectral contrast; not a calibrated significance test',mean_r=float(np.nanmean(z['r'][:,j])),std_r=float(np.nanstd(z['r'][:,j])),r_exceeds_water=float(np.mean(z['r'][:,j]>.951/(.913+.120))),invalid_pairs=float(np.mean((z['pairs'][:,j]<=0).any(-1))),negative_channels=float(np.mean((z['channels'][:,j]<0).any(-1))))
   for key,y in [('X',z['xy'][:,j,0]),('Y',z['xy'][:,j,1]),('r',z['r'][:,j])]:
    row[key+'_semiamplitude_at_intensity_peak']=fit(t,y,fpeak)[0] if np.isfinite(y).sum()>10 else np.nan
   cm=z['channels'][:,j].mean(0);rr=np.hypot((cm[0]-cm[1])/(cm[0]+cm[1]),(cm[2]-cm[3])/(cm[2]+cm[3]));row['r_from_mean_channels']=rr;row['mean_r_minus_r_of_means']=row['mean_r']-rr
   if 'centroid_width' in z:
    cent=z['centroid_width'][:,j];row.update(centroid_x_sd_px=float(cent[:,0].std()),centroid_y_sd_px=float(cent[:,1].std()),width_mean_px=float(cent[:,2].mean()),centroid_x_line_px=fit(t,cent[:,0],fpeak)[0],centroid_y_line_px=fit(t,cent[:,1],fpeak)[0])
   if 'zero_sat_max' in z:row.update(zero_pixel_fraction=float(z['zero_sat_max'][:,j,0].mean()),near_12bit_rail_fraction=float(z['zero_sat_max'][:,j,1].mean()),max_saved_pixel=float(z['zero_sat_max'][:,j,2].max()))
   rows.append(row);roirows.append(dict(file_id=p.stem,source_path=info['source_path'],**r,origin_xy=str(info.get('origin_xy','unknown or see metadata'))))
  if p.stem.startswith('recording_') or p.stem in ['frame_stack_20260918-145406','frame_stack_20261002-164530','spotrec_20260918-145944','intensity_diagnostic_20261002-165236_1640','intensity_diagnostic_20261002-165336_1400','intensity_diagnostic_20261002-173339']:
   fig,ax=plt.subplots(3,2,figsize=(13,10));cols=plt.cm.tab10.colors
   for k,j in enumerate(rodidx[:4]):
    c=cols[k];y=z['total'][:,j];lab=rois[j]['id'];ax[0,0].plot(t,y,lw=.5,label=lab,color=c);f,pow=spectrum(t,y/y.mean());ax[0,1].semilogy(f,pow,color=c,label=lab);ax[1,1].loglog(f[1:],pow[1:],color=c)
   for j in [j for j,r in enumerate(rois) if r['kind']=='background'][:2]:
    f,pow=spectrum(t,z['total'][:,j]/z['total'][:,j].mean());ax[0,1].semilogy(f,pow,ls='--',alpha=.7,label=rois[j]['id'])
   j=rodidx[0]
   for k,lab in enumerate(['I0','I90','I45','I135']):ax[1,0].plot(t,z['channels'][:,j,k],lw=.5,label=lab)
   for k,lab in enumerate(['X','Y']):ax[2,0].plot(t,z['xy'][:,j,k],lw=.5,label=lab)
   ax[2,0].plot(t,z['r'][:,j],lw=.5,label='r unclipped')
   for key,y in [('X',z['xy'][:,j,0]),('Y',z['xy'][:,j,1]),('r',z['r'][:,j])]:
    if np.isfinite(y).all():f,pow=spectrum(t,y);ax[2,1].semilogy(f,pow,label=key)
   ax[0,1].set_xlim(0,5);ax[2,1].set_xlim(0,5)
   for a in ax[:,1]:a.axvspan(1.5,1.6,color='orange',alpha=.2);a.set_xlabel('Hz (AVI conditional)');a.set_ylabel('PSD / Hz')
   for a in ax[:,0]:a.set_xlabel('Time (s)');a.legend(loc='upper right',fontsize=8)
   ax[0,0].set_ylabel('Saved total intensity');ax[1,0].set_ylabel('Four channels');ax[2,0].set_ylabel('Anisotropy');ax[0,1].legend(fontsize=8);ax[2,1].legend(fontsize=8);fig.suptitle(p.stem+'\nUnfiltered traces; Hann full-record PSD, linear detrend');fig.tight_layout();fig.savefig(BASE/'plots'/f'{p.stem}_analysis.png',dpi=150);plt.close(fig)
 dumpcsv(BASE/'metrics.csv',rows);dumpcsv(BASE/'selected_rois.csv',roirows);dumpcsv(BASE/'file_metrics.csv',files)
 print('METRICS',len(rows),'rows',flush=True)

if __name__=='__main__':
 mode=sys.argv[1] if len(sys.argv)>1 else 'all'
 if mode in ['all','avi']:avi()
 if mode in ['all','arrays']:
  paths=[OLD/'frame_stack_20260918-145406.npy',RUNS/'datasets/temp/frame_stack_20261002-164530.npy']+sorted(OLD.glob('spotrec*.npy'))+sorted(RUNS.rglob('capture_maxfps_15x15.npy'))+sorted(OLD.glob('frame_stack*.npy'))+sorted(RUNS.rglob('frame_stack*.npy'))
  for p in dict.fromkeys(paths):arrayfile(p)
 if mode in ['all','diagnostics']:
  for p in sorted(RUNS.glob('intensity_diagnostic*.npz')):diagnostic(p)
 if mode in ['all','analyse']:analyse()
