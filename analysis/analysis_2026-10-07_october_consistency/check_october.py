from pathlib import Path
import numpy as np,json,csv
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;SRC=B.parent/'analysis_2026-10-06_firstpass/traces';(B/'plots').mkdir(exist_ok=True)
def spectrum(t,y):
 fs=(len(t)-1)/(t[-1]-t[0]);tt=np.linspace(t[0],t[-1],len(t));return signal.periodogram(np.interp(tt,t,y),fs=fs,window='hann',detrend='linear',scaling='density')
def fit(t,y,fr):
 M=np.column_stack([np.ones(len(t)),t-t.mean(),np.cos(2*np.pi*fr*t),np.sin(2*np.pi*fr*t)]);c=np.linalg.lstsq(M,y,rcond=None)[0];return np.hypot(c[2],c[3])/np.mean(y)
def csvout(name,rows):
 with (B/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
rows=[];files=[];loaded=[]
for p in sorted(SRC.glob('*.npz')):
 if '20260918' in p.stem:continue
 z=np.load(p);t=z['time_s'];info=json.loads(str(z['info_json']));rs=json.loads(str(z['rois_json']));ids=[j for j,r in enumerate(rs) if r['kind']=='rod' and not r['id'].endswith('large')];fs=(len(t)-1)/(t[-1]-t[0]);dur=len(t)/fs;plots=[]
 for j in ids:
  y=z['total'][:,j];ff,pp=spectrum(t,y/y.mean());sel=(ff>=.25)&(ff<=min(5,fs/2));dom=ff[sel][np.argmax(pp[sel])];sb=(ff>=.4)&(ff<=.9);peak=ff[sb][np.argmax(pp[sb])] if sb.any() else np.nan;nb=(ff>=.25)&(ff<=min(3,fs/2))&~sb;contrast=pp[sb].max()/np.median(pp[nb]) if sb.any() and nb.sum()>3 else np.nan
  row=dict(file=p.stem,roi=rs[j]['id'],duration_s=dur,fs=fs,long_enough=dur>=10,timing=info['timing'],dominant_0p25_5_hz=dom,slow_peak_hz=peak,slow_semi_fraction=fit(t,y,peak) if np.isfinite(peak) else np.nan,slow_psd_contrast=contrast,original_1p55_semi_fraction=fit(t,y,1.55) if fs>3.2 else np.nan,cv=y.std()/y.mean(),r_sd=np.nanstd(z['r'][:,j]),source_path=info['source_path'])
  rows.append(row);plots.append((j,y,ff,pp))
 rr=[r for r in rows if r['file']==p.stem];files.append(dict(file=p.stem,duration_s=dur,fs=fs,n_rois=len(rr),long_enough=dur>=10,dominant_slow_count=sum(.4<=r['dominant_0p25_5_hz']<=.9 for r in rr),median_slow_semi_fraction=np.median([r['slow_semi_fraction'] for r in rr]),median_cv=np.median([r['cv'] for r in rr]),median_psd_contrast=np.nanmedian([r['slow_psd_contrast'] for r in rr])))
 loaded.append((p.stem,t,plots,info,dur,fs))
csvout('per_roi.csv',rows);csvout('per_file.csv',files)
# Every long record: intensity and spectra. 5 diagnostic rows per page, 4 other rows per page.
for group in ['diagnostic','other']:
 use=[x for x in loaded if x[4]>=10 and (x[0].startswith('intensity')==(group=='diagnostic'))];pagesize=5 if group=='diagnostic' else 4
 for page,start in enumerate(range(0,len(use),pagesize)):
  subset=use[start:start+pagesize];fig,axs=plt.subplots(len(subset),2,figsize=(13,len(subset)*2.5),squeeze=False)
  for k,(fid,t,plots,info,dur,fs) in enumerate(subset):
   for j,y,ff,pp in plots[:4]:
    step=max(1,int(fs*.04));n=len(t)//step;axs[k,0].plot(t[:n*step].reshape(n,step).mean(1),y[:n*step].reshape(n,step).mean(1)/y.mean(),lw=.65,label=f'ROI {j+1}');axs[k,1].semilogy(ff,pp,lw=.8)
   axs[k,0].set_title(fid.replace('intensity_diagnostic_','diagnostic_'),fontsize=9);axs[k,0].set_ylabel('I / mean');axs[k,1].set_title(f'{dur:.1f} s; {fs:.2f} fps; bin {1/dur:.3f} Hz',fontsize=9);axs[k,1].set_xlim(0,min(3,fs/2));axs[k,1].axvspan(.4,.9,color='teal',alpha=.15);axs[k,1].axvspan(1.5,1.6,color='orange',alpha=.25)
   if len(plots)>1:axs[k,0].legend(fontsize=7,ncol=4)
  axs[-1,0].set_xlabel('Seconds (AVI header conditional)');axs[-1,1].set_xlabel('Hz');fig.suptitle('October consistency: all long recordings\nDisplay means ~40 ms; spectra use full-rate traces, Hann window and linear detrend');fig.tight_layout(rect=(0,0,1,.95));fig.savefig(B/'plots'/f'{group}_{page+1}.png',dpi=135);plt.close(fig)
print('Files',len(files),'long',sum(x['long_enough'] for x in files))
for r in files:print(r)
