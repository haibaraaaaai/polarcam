import firstpass as f
from pathlib import Path
import numpy as np,json,hashlib
B=f.BASE;rows=[];maps=[]
for p in sorted((f.RUNS/'backgrounds').glob('background_min*.npy')):
 m=json.loads(p.with_suffix('.json').read_text());a=np.load(p);sp=p.parent/m['source_stack_file'];s=np.load(sp,mmap_mode='r');marker=np.count_nonzero(s[-1])==1 and s[-1,:2,:2].sum()==1;ss=s[:-1] if marker else s
 # exact minimum compatibility, and sampled temporal/background summaries
 minimum=np.min(ss,axis=0).astype(float);row=dict(file=p.name,source_stack=sp.name,frames=len(ss),fps=m['actual']['fps'],requested_duration=m['requested']['duration_s'],duration_from_frames_fps=len(ss)/m['actual']['fps'],exposure_us=m['actual']['timing']['exposure_us'],sound_checkbox=m['sound']['sound_on'],profile_mean=float(a.mean()),profile_median=float(np.median(a)),profile_p05=float(np.percentile(a,5)),profile_p95=float(np.percentile(a,95)),profile_zero_fraction=float((a==0).mean()),exact_minimum_match=bool(np.array_equal(a,minimum)),max_difference_from_min=float(np.max(abs(a-minimum))),sha256=hashlib.sha256(a.tobytes()).hexdigest());rows.append(row)
for d in json.loads((B/'inventory.json').read_text()):
 if 'metadata' not in d:continue
 p=Path(d['path']);m=d['metadata'];old=m.get('npy_path');row={'file':str(p),'saved_npy_path':old,'resolved_npy_path':str(p),'resolution':'sibling file at user-supplied root, originals unchanged'}
 bg=m.get('background',{}).get('background_profile_path')
 if bg:
  base=Path(bg).name;candidates=list(f.RUNS.rglob(base))
  row.update(saved_background_path=bg,local_candidates=' | '.join(map(str,candidates)),background_reuse='not applied; old generic profile cannot be linked to Sep18; dated October siblings preserved')
 maps.append(row)
f.dumpcsv(B/'background_metrics.csv',rows);f.dumpcsv(B/'path_resolution.csv',maps)
print(json.dumps(rows,indent=2))
