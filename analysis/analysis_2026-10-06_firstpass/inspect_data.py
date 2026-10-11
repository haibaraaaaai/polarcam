import json, pathlib, numpy as np, cv2
out=pathlib.Path(__file__).resolve().parent
camera_root=pathlib.Path(__file__).resolve().parents[2] / 'data/hinges/polarcam'
roots=sorted(camera_root.glob('2026-??-??'))
rows=[]
for root in roots:
 for p in root.rglob('*'):
  if not p.is_file():continue
  d={'path':str(p),'bytes':p.stat().st_size,'attributes':p.stat().st_file_attributes}
  try:
   if p.suffix=='.npy':
    a=np.load(p,mmap_mode='r'); d.update(shape=list(a.shape),dtype=str(a.dtype))
    if p.with_suffix('.json').exists():d['metadata']=json.loads(p.with_suffix('.json').read_text())
   if p.suffix=='.npz':
    with np.load(p,allow_pickle=False) as z:
     d['arrays']={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}
   if p.suffix=='.avi':
    c=cv2.VideoCapture(str(p)); ok,f=c.read();d.update(fps=c.get(cv2.CAP_PROP_FPS),frames=c.get(cv2.CAP_PROP_FRAME_COUNT),fourcc=''.join(chr((int(c.get(cv2.CAP_PROP_FOURCC))>>8*i)&255) for i in range(4)),shape=list(f.shape),min=int(f.min()),max=int(f.max()),channels_identical=bool(np.array_equal(f[:,:,0],f[:,:,1]) and np.array_equal(f[:,:,0],f[:,:,2])));cv2.imwrite(str(out/'avi_first_frame.png'),f);c.release()
  except Exception as e:d['error']=str(e)
  rows.append(d)
(out/'inventory.json').write_text(json.dumps(rows,indent=2))
for d in rows:
 if 'metadata' in d:
  m=d['metadata']; print(pathlib.Path(d['path']).name, d['shape'], m.get('actual',{}).get('fps'),m.get('background',{}).get('background_subtracted'),m.get('conditions'), m.get('sound'))
 if d['path'].endswith('.avi') or 'error' in d:print(d)
for p in camera_root.glob('2026-10-*/intensity*.npz'):
 with np.load(p,allow_pickle=False) as z:
  print(p.name,[(k,z[k].shape,str(z[k].dtype)) for k in z.files]);
  for k in z.files:
   if z[k].size<5:print(k,z[k])
