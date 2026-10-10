"""Broad-frequency follow-up, matched-duration checks and gallery summary plots."""
import firstpass as f
import numpy as np,json,csv
from scipy import signal
import matplotlib.pyplot as plt
B=f.BASE

def summary():
 rows=[];segrows=[]
 for p in sorted((B/'traces').glob('*.npz')):
  z=np.load(p);t=z['time_s'];dur=t[-1];info=json.loads(str(z['info_json']));rois=json.loads(str(z['rois_json']));fs=(len(t)-1)/dur
  for j,r in enumerate(rois):
   y=z['total'][:,j];freq,ps=f.spectrum(t,y/y.mean());mask=(freq>=.25)&(freq<=min(5,fs/2));dom=float(freq[mask][np.argmax(ps[mask])]);fp=f.peak(t,y,.4,.85);ap,ph,_,_=f.fit(t,y,fp)
   row=dict(file_id=p.stem,roi_id=r['id'],kind=r['kind'],duration_s=dur,dominant_0p25_5Hz_bin=dom,candidate_0p4_0p85Hz=fp,candidate_semi_fraction=ap/y.mean(),candidate_phase=ph,mean_r=float(np.nanmean(z['r'][:,j])),std_r=float(np.nanstd(z['r'][:,j])),background_subtracted=info.get('background_subtracted','unknown'),exposure_us=info.get('exposure_us','unknown'))
   for freq0 in [.625,1.55]:
    aa,ph,_,_=f.fit(t,y,freq0);row[f'semi_fraction_at_{freq0}Hz']=aa/y.mean();row[f'phase_at_{freq0}Hz']=ph
   for key,v in [('X',z['xy'][:,j,0]),('Y',z['xy'][:,j,1]),('r',z['r'][:,j])]:row[key+'_semi_at_candidate']=f.fit(t,v,fp)[0] if np.isfinite(v).sum()>10 else np.nan
   sel=(freq>=.4)&(freq<=.85);row['fractional_rms_0p4_0p85Hz']=np.sqrt(np.sum(ps[sel])*(freq[1]-freq[0])) if dur>=10 else np.nan
   if 'centroid_width' in z:
    cent=z['centroid_width'][:,j];row['centroid_sd_max']=cent[:,:2].std(0).max();row['position_label']='position-stable candidate' if row['centroid_sd_max']<.25 else 'motion/shape/noise ambiguous'
   else:row['position_label']='no image trajectory available'
   row['orientation_label']='orientation variable or noisy' if max(np.nanstd(z['xy'][:,j,0]),np.nanstd(z['xy'][:,j,1]))>.05 else 'low X/Y variability'
   for start in np.arange(0,dur-4+.001,4):
    ix=(t>=start)&(t<start+4);a,phase,_,_=f.fit(t[ix],y[ix],.625);segrows.append(dict(file_id=p.stem,roi_id=r['id'],start_s=start,semi_fraction_0p625=a/y[ix].mean(),phase_rad=phase))
   rows.append(row)
 f.dumpcsv(B/'broad_feature_metrics.csv',rows);f.dumpcsv(B/'four_second_segments.csv',segrows)
 return rows

def matched():
 # Equal 150 samples, same reported 597.5 us exposure, 72.1754 Hz point sampling (NOT exposure averaging).
 dur=150/72.17540574282147;t=np.arange(150)/72.17540574282147;rows=[]
 ids=['frame_stack_20260918-145406','frame_stack_20261002-164530','intensity_diagnostic_20261002-163712','intensity_diagnostic_20261002-164221','intensity_diagnostic_20261002-165236_1640','intensity_diagnostic_20261002-165336_1400']
 for fid in ids:
  z=np.load(B/'traces'/f'{fid}.npz');rs=json.loads(str(z['rois_json']));tt=z['time_s']
  for j,r in enumerate(rs):
   if r['kind']!='rod' or r['id'].endswith('large'):continue
   starts=[0.] if len(tt)==150 else np.arange(0,tt[-1]-dur,dur)
   for start in starts:
    y=np.interp(t+start,tt,z['total'][:,j]);y=y/y.mean();row=dict(file_id=fid,roi_id=r['id'],start_s=start,duration_s=dur,cv=y.std(),r_sd=float(np.nanstd(np.interp(t+start,tt,z['r'][:,j]))))
    for freq in [.625,1.55]:row[f'fractional_semi_{freq}Hz']=f.fit(t,y,freq)[0]
    rows.append(row)
 f.dumpcsv(B/'matched_duration.csv',rows)
 # Noise sensitivity: phase-randomized detrended old empirical residuals preserve periodogram.
 rng=np.random.default_rng(20261006);inj=[];z=np.load(B/'traces/frame_stack_20260918-145406.npz');rs=json.loads(str(z['rois_json']));t=z['time_s']
 for j,r in enumerate(rs):
  if r['kind']!='rod' or r['id'].endswith('large'):continue
  y=z['total'][:,j];res=signal.detrend(y/y.mean());ft=np.fft.rfft(res);nulls=[]
  for k in range(240):
   phase=rng.uniform(0,2*np.pi,len(ft));phase[0]=0;phase[-1]=0;noise=np.fft.irfft(abs(ft)*np.exp(1j*phase),n=len(t));nulls.append(noise)
  for freq in [.625,1.55]:
   nullamp=np.array([f.fit(t,n,freq)[0] for n in nulls]);threshold=np.quantile(nullamp,.95)
   for a in [.02,.05,.10]:
    vals=[f.fit(t,n+a*np.cos(2*np.pi*freq*t+rng.uniform(0,2*np.pi)),freq)[0] for n in nulls]
    inj.append(dict(file_id='frame_stack_20260918-145406',roi_id=r['id'],frequency_hz=freq,injected_fractional_semiamplitude=a,empirical_null_95pct_threshold=threshold,detection_fraction=float(np.mean(np.array(vals)>threshold)),noise_model='phase-randomized linear-detrended observed old trace; colored fluctuations retained; fixed-frequency test'))
 f.dumpcsv(B/'injection_sensitivity.csv',inj)
 fig,ax=plt.subplots(1,2,figsize=(12,4));groups=[ids[0],ids[1],ids[2],ids[4],ids[5]];labels=['Sep18 field','Oct2 field','Oct2 diag\n163712','Oct2 diag\n165236','Oct2 diag\n165336']
 for i,key in enumerate(['cv','r_sd']):
  data=[[r[key] for r in rows if r['file_id']==g] for g in groups];ax[i].boxplot(data,tick_labels=labels,showfliers=True);ax[i].set_ylabel(['Intensity CV','r SD'][i]);ax[i].set_title('Equal 2.078 s windows / 150 samples');ax[i].grid(axis='y',alpha=.2)
 fig.suptitle('Duration/sampling matched; sample, background and rod motion NOT matched');fig.tight_layout();fig.savefig(B/'plots/matched_comparison.png',dpi=160);plt.close(fig)

def meeting_plot():
 files=['recording_2026-10-02_2','intensity_diagnostic_20261002-163712','spotrec_20260918-145944','frame_stack_20260918-145406'];js=[0,0,0,9]
 fig,ax=plt.subplots(4,2,figsize=(12,12))
 for k,(fid,j) in enumerate(zip(files,js)):
  z=np.load(B/'traces'/f'{fid}.npz');t=z['time_s'];y=z['total'][:,j];r=z['r'][:,j];norm=y/y.mean();ax[k,0].plot(t,norm,lw=.6,color='#126985');ax[k,0].set_ylabel('Total / mean');ax[k,0].set_xlabel('Time (s)');ax[k,0].set_title(fid+(' / rod10' if k==3 else ' / rod01 or target'),fontsize=10)
  ff,ps=f.spectrum(t,norm);ax[k,1].semilogy(ff,ps,color='#126985');ax[k,1].axvspan(.4,.85,color='#19a68b',alpha=.15,label='Observed slow band');ax[k,1].axvspan(1.5,1.6,color='#de9b18',alpha=.3,label='Requested band');ax[k,1].set(xlim=(0,5),xlabel='Frequency (Hz; AVI header conditional)',ylabel='Fractional PSD / Hz')
  ax[k,1].text(.97,.93,f'T={len(t)/(len(t)-1)*t[-1]:.2f}s; bin={1/(len(t)/(len(t)-1)*t[-1]):.3f}Hz',transform=ax[k,1].transAxes,ha='right',va='top',fontsize=9)
 ax[0,1].legend(loc='lower right',fontsize=8);fig.suptitle('First-pass finding: about 0.6 Hz (~1.6 s period), not a universal 1.6 Hz line\nOriginal unfiltered traces; full-record Hann spectra with linear detrend',fontsize=13);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(B/'plots/meeting_first_result.png',dpi=160);plt.close(fig)
 # AVI raw traces binned only for readable display, with unfiltered full-record spectra and correlation
 z=np.load(B/'traces/recording_2026-10-02_2.npz');t=z['time_s'];rs=json.loads(str(z['rois_json']));ix=[j for j,r in enumerate(rs) if not r['id'].endswith('large')];n=39;tr=np.array([z['total'][:len(t)//n*n,j].reshape(-1,n).mean(1) for j in ix]);corr=np.corrcoef(signal.detrend(tr,axis=1));f.dumpcsv(B/'avi_correlations.csv',[dict(roi_a=rs[ix[i]]['id'],roi_b=rs[ix[j]]['id'],correlation=corr[i,j],method='linear-detrended 39-frame means (~50 ms), no line filter') for i in range(len(ix)) for j in range(i+1,len(ix))]);fig,ax=plt.subplots(1,2,figsize=(13,5));tb=t[:len(t)//n*n].reshape(-1,n).mean(1)
 for k,j in enumerate(ix):ax[0].plot(tb,(tr[k]/tr[k].mean()-1)+k*.4,label=rs[j]['id'],lw=.8)
 ax[0].set(xlabel='Time (s, AVI header)',ylabel='Fractional intensity + offset',title='Rods and backgrounds: ~50 ms means');ax[0].legend(ncol=3,fontsize=8);im=ax[1].imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax[1].set_xticks(range(len(ix)),[rs[j]['id'] for j in ix],rotation=45);ax[1].set_yticks(range(len(ix)),[rs[j]['id'] for j in ix]);ax[1].set_title('Correlation, detrended binned traces');fig.colorbar(im,ax=ax[1]);fig.tight_layout();fig.savefig(B/'plots/avi_shared_modulation.png',dpi=160);plt.close(fig)

if __name__=='__main__':
 summary();matched();meeting_plot();print('Follow-up outputs complete')
