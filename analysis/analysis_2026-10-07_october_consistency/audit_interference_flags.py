from pathlib import Path
import json,csv,numpy as np
root=(Path(__file__).resolve().parents[2] / 'data/hinges/polarcam');out=(Path(__file__).resolve().parents[2] / 'analysis/analysis_2026-10-07_october_consistency');rows=[]
for p in sorted(root.glob('2026-10-*/**/*.json')):
 m=json.loads(p.read_text(encoding='utf-8-sig'))
 if not isinstance(m,dict) or not m.get('recording_type'):continue
 a=m.get('actual',{});fps=a.get('fps');frames=a.get('frames');rows.append(dict(file=str(p.with_suffix('.npy')),metadata=str(p),recording_type=m['recording_type'],sound_on=m.get('sound',{}).get('sound_on','missing'),sound_source=m.get('sound',{}).get('source','missing'),interference_drive=m.get('conditions',{}).get('interference_drive','missing'),background_subtracted=m.get('background',{}).get('background_subtracted','missing'),duration_s=frames/fps if frames and fps else '',fps=fps,note=m.get('conditions',{}).get('condition_note','')))
for p in sorted(root.glob('2026-10-*/intensity_diagnostic*.npz')):
 with np.load(p) as z:m=json.loads(str(z['metadata_json']));i=m['initial_settings'];rows.append(dict(file=str(p),metadata='embedded metadata_json',recording_type='intensity_diagnostic',sound_on=i.get('sound_on_manual','missing'),sound_source='manual diagnostic settings',interference_drive='not separately recorded',background_subtracted=i.get('background_requested','missing'),duration_s=float(z['samples'][-1,0]-z['samples'][0,0]),fps=i['timing_readback']['fps'],note=i.get('drive_note_manual','')))
with (out/'interference_flags.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
for r in rows:
 if r['sound_on'] is True:print(Path(r['file']).name,r['recording_type'],r['duration_s'],r['fps'],r['background_subtracted'],r['interference_drive'])
