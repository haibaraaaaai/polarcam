from pathlib import Path
import csv,json,numpy as np
import firstpass as f
B=f.BASE
for name in ['metrics.csv','file_metrics.csv','broad_feature_metrics.csv']:
 rows=list(csv.DictReader((B/name).open(encoding='utf-8-sig')))
 for r in rows:
  r['sample_status']='September hinge collection per user; not independently verified' if '20260918' in r['file_id'] else 'unknown; gallery labelling pending'
  r['amplitude_convention']='sinusoidal semi-amplitude divided by saved mean (not peak-to-peak)' if name!='file_metrics.csv' else ''
 f.dumpcsv(B/name,rows)
out=[]
for p in sorted((B/'traces').glob('intensity*.npz')):
 z=np.load(p);s=z['diagnostic_samples'];i=json.loads(str(z['info_json']))['metadata']['initial_settings'];fps=i['timing_readback']['fps'];dt=np.diff(s[:,0]);out.append(dict(file_id=p.stem,readback_fps=fps,host_rate=(len(s)-1)/(s[-1,0]-s[0,0]),rate_discrepancy_fraction=((len(s)-1)/(s[-1,0]-s[0,0]))/fps-1,max_host_gap_ms=dt.max()*1e3,software_frame_index_gaps=int(np.sum(np.diff(s[:,2])!=1)),zero_pixel_fraction=s[:,11].mean(),saturated_pixel_fraction=s[:,12].mean(),raw_min=s[:,9].min(),raw_max=s[:,10].max(),raw_vs_analyzed_mean_max_difference=np.max(abs(s[:,3]-s[:,4]))))
f.dumpcsv(B/'diagnostic_timing_checks.csv',out)
print('Final audit tables saved')
