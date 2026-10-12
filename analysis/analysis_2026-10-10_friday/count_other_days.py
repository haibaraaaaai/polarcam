"""Quick, offline candidate census of the 1–2 October observation stacks."""
from pathlib import Path
import hashlib
import json
import numpy as np
from first_hinges_theta import load_module, clean_json
from local_spot_check import filtering, local_metrics
from pipeline_audit import import_file, PACKAGE

B=Path(__file__).resolve().parent
REPO=B.parents[1]


def main():
    detector,source=load_module('other_days_native','src/polarcam/live/Detection_alg_offline.py')
    supplied=import_file('other_days_supplied',PACKAGE/'detector_intensity_reference.py')
    rows=[]
    for date in ('2026-10-01','2026-10-02'):
        for path in sorted((REPO/'data/hinges/polarcam'/date).glob('frame_stack_*.npy')):
            before=path.stat();meta_path=path.with_suffix('.json')
            metadata_bytes=meta_path.read_bytes();meta=json.loads(metadata_bytes.decode('utf-8-sig'))
            raw=np.load(path,mmap_mode='r',allow_pickle=False)
            n=len(raw)-int(meta['actual'].get('phase_marker_appended',False))
            assert n==meta['actual']['frames']==150 and raw.shape[1:]==(2056,2464)
            assert meta['roi']['OffsetX']==meta['roi']['OffsetY']==0
            mean=np.zeros(raw.shape[1:],dtype=np.float64)
            for frame in raw[:n]:mean+=frame
            mean/=n
            f=filtering(detector.intensity_detection_map(mean),detector)
            pts=f['centres'];native=np.asarray(detector.find_spot_centers_dog(f['image'][10:-10,10:-10])).reshape(-1,2)+10
            np.testing.assert_allclose(pts,native,rtol=0,atol=1e-12)
            scores=[]
            for j in range(len(pts)):
                try:score=local_metrics(f,pts,j)['local_score']
                except (ValueError,ZeroDivisionError,FloatingPointError):score=np.nan
                scores.append(score)
            scores=np.asarray(scores)
            old,old_info=supplied.detect_candidates(mean.astype(np.float32)[None])
            row=dict(file=path.name,date=date,time=path.stem[-6:],frames=n,
                native_count=len(pts),native_local_pass=int(np.sum(np.isfinite(scores)&(scores>=8))),
                local_unscorable=int(np.sum(~np.isfinite(scores))),supplied_count=len(old),
                background_subtracted=meta['background']['background_subtracted'],
                exposure_ms=meta['actual']['timing']['exposure_us']/1000,fps=meta['actual']['fps'],
                native_threshold=float(f['threshold']),supplied_threshold=old_info['dog_threshold'],
                saved_gui_detector=meta.get('conditions',{}).get('detector'),
                source_size=before.st_size,source_mtime_ns=before.st_mtime_ns,
                source_json_sha256=hashlib.sha256(metadata_bytes).hexdigest())
            assert before.st_size==path.stat().st_size and before.st_mtime_ns==path.stat().st_mtime_ns
            assert metadata_bytes==meta_path.read_bytes()
            rows.append(row)
            result=clean_json(dict(method='Both intensity detectors on the full 150-frame mean; marker excluded. Local score cutoff 8. No theta, brightness-gate or recurrence exclusions.',
                source_hashes={str(p.relative_to(REPO)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),source,PACKAGE/'detector_intensity_reference.py',B/'local_spot_check.py')},stacks=rows))
            (B/'other_days_spot_counts.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
            print(json.dumps(row),flush=True)
            del raw,mean,f


if __name__=='__main__':main()
