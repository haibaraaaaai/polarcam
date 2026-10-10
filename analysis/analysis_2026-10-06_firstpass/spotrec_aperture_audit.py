import firstpass as f,numpy as np,json
rows=[]
for p in sorted(f.OLD.glob('spotrec*.npy')):
 m=json.loads(p.with_suffix('.json').read_text());a=np.load(p,mmap_mode='r')[:-1];r=m['roi'];cx=r['cx']-r['x'];cy=r['cy']-r['y'];size=r['win_raw'];x0=round(cx)-size//2;y0=round(cy)-size//2;x1=min(a.shape[2],x0+size);y1=min(a.shape[1],y0+size);x0=max(0,x0);y0=max(0,y0);crop=a[:,y0:y1,x0:x1];py=(r['y']+y0)%2;px=(r['x']+x0)%2;chs=np.stack([crop[:,1-py::2,1-px::2].mean((1,2)),crop[:,py::2,px::2].mean((1,2)),crop[:,py::2,1-px::2].mean((1,2)),crop[:,1-py::2,px::2].mean((1,2))],-1)
 with np.errstate(divide='ignore',invalid='ignore'):rr=np.hypot((chs[:,0]-chs[:,1])/(chs[:,0]+chs[:,1]),(chs[:,2]-chs[:,3])/(chs[:,2]+chs[:,3]))
 ours=np.load(f.BASE/'traces'/f'{p.stem}.npz')['r'][:,0];ref=m['xy_series_summary']['r_mean'];rows.append(dict(file_id=p.stem,gui_clip_window_xyxy=str([x0,y0,x1,y1]),gui_like_mean_r=float(np.nanmean(rr)),sidecar_mean_r=ref,delta_gui_like_vs_sidecar=float(np.nanmean(rr)-ref),firstpass_equal_count_window_mean_r=float(np.nanmean(ours)),delta_equal_count_vs_gui_like=float(np.nanmean(ours)-np.nanmean(rr))))
f.dumpcsv(f.BASE/'spotrec_aperture_audit.csv',rows);print('max GUI-like vs sidecar delta',max(abs(r['delta_gui_like_vs_sidecar']) for r in rows));print('max equal aperture vs gui delta',max(abs(r['delta_equal_count_vs_gui_like']) for r in rows))
