from pathlib import Path
import json, csv
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[2] / 'data/hinges/polarcam/2026-10-02')
TIMES=['181519','181706','181829','182853','183613']
A,B,C=.097840,.011915,.101852
LIMIT=C/(A+B)
def cells(a):
 return (a[...,::2,::2]+a[...,::2,1::2]+a[...,1::2,::2]+a[...,1::2,1::2])/4
def anis(ch):
 pairs=np.stack([ch[...,0]+ch[...,1],ch[...,2]+ch[...,3]],-1)
 with np.errstate(invalid='ignore',divide='ignore'):
  xy=np.stack([(ch[...,0]-ch[...,1])/pairs[...,0],(ch[...,2]-ch[...,3])/pairs[...,1]],-1)
 return pairs,xy,np.linalg.norm(xy,axis=-1)
def theta(r):
 r=np.asarray(r); good=np.isfinite(r)&(r>=0)&(r<=LIMIT)
 out=np.full(r.shape,np.nan);out[good]=np.degrees(np.arcsin(np.sqrt(A*r[good]/(C-B*r[good]))));return out
def writecsv(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

config=dict(files=TIMES,source=str(ROOT),detection='30 evenly spaced frame mean; 2x2 total-intensity cell mean; DoG sigma .8/3 cells; local max 7 cells; median+8 robust MAD; 20 pixel edge; nearest candidate >=24 pixels. All eligible spots, no anisotropy selection.',apertures=[12,20],background='Use already-subtracted saved data; no second subtraction.',distribution='One value per candidate: r calculated from time-mean channel sums. This averages polarization vectors, not instantaneous theta. No motion filtering; candidates are not validated single stationary rods.',channel_order=['I0','I90','I45','I135'],phase='[90,45;135,0], absolute sensor parity',theta_model=dict(A=A,B=B,C=C,r_limit=LIMIT,convention='supplied hole+Fresnel script; model-dependent angle from optical axis; out-of-range excluded, never clamped'),caveats='No channel gain calibration. Background clipping and pre-subtraction sensor saturation cannot be reconstructed. Wider-aperture comparison is a sensitivity check, not a correction.')
(OUT/'config.json').write_text(json.dumps(config,indent=2))
rows=[]; summaries=[]; panels=[]
for stamp in TIMES:
 stem='frame_stack_20261002-'+stamp;p=ROOT/(stem+'.npy');m=json.loads(p.with_suffix('.json').read_text());raw=np.load(p,mmap_mode='r')
 assert m['background']['background_subtracted']
 assert m['actual']['phase_marker_appended'] and np.count_nonzero(raw[-1])==1 and raw[-1,:2,:2].sum()==1
 raw=raw[:-1];mean=np.mean(raw[np.linspace(0,len(raw)-1,30).astype(int)],axis=0,dtype=float)
 cm=cells(mean);dog=ndi.gaussian_filter(cm,.8)-ndi.gaussian_filter(cm,3);threshold=np.median(dog)+8*1.4826*np.median(abs(dog-np.median(dog)))
 yy,xx=np.where((dog==ndi.maximum_filter(dog,7))&(dog>threshold));pts=np.column_stack([xx*2,yy*2]);dist=cKDTree(pts).query(pts,k=2)[0][:,1]
 keep=(dist>=24)&(pts[:,0]>20)&(pts[:,0]<raw.shape[2]-20)&(pts[:,1]>20)&(pts[:,1]<raw.shape[1]-20);pts=pts[keep]
 roi=m['roi'];ox=int(roi['OffsetX']);oy=int(roi['OffsetY']);allch=[];allzero=[];allrail=[]
 for size in [12,20]:
  ch=[];zero=[];rail=[]
  for x,y in pts:
   x0=x-size//2;y0=y-size//2;crop=raw[:,y0:y0+size,x0:x0+size].astype(float);px=(x0+ox)%2;py=(y0+oy)%2
   ch.append(np.stack([crop[:,1-py::2,1-px::2].sum((1,2)),crop[:,py::2,px::2].sum((1,2)),crop[:,py::2,1-px::2].sum((1,2)),crop[:,1-py::2,px::2].sum((1,2))],-1));zero.append((crop==0).mean());rail.append((crop>=4094).any())
  allch.append(np.stack(ch,1));allzero.append(np.array(zero));allrail.append(np.array(rail))
 ch=np.stack(allch,2);pairs,xy,r=anis(ch);_,_,rmean=anis(ch.mean(0));th=theta(rmean);rail=np.stack(allrail,1);zero=np.stack(allzero,1)
 np.savez_compressed(OUT/(stem+'_traces.npz'),time_s=np.arange(len(raw))/m['actual']['fps'],channels=ch,total=ch.sum(-1),pairs=pairs,xy=xy,r=r,roi_xy=pts,apertures=[12,20],r_from_mean_channels=rmean,theta_deg=th)
 for j,(x,y) in enumerate(pts):
  rows.append(dict(file=stem,spot_id=j+1,x=int(x),y=int(y),x_sensor=int(x+ox),y_sensor=int(y+oy),r_12=rmean[j,0],theta_12_deg=th[j,0],r_20=rmean[j,1],theta_20_deg=th[j,1],zero_fraction_12=zero[j,0],saved_rail_12=bool(rail[j,0]),saved_rail_20=bool(rail[j,1]),r_temporal_sd_12=np.nanstd(r[:,j,0]),invalid_frame_fraction_12=np.mean(~np.isfinite(r[:,j,0]))))
 valid=np.isfinite(th[:,0])&~rail[:,0];wide=np.isfinite(th[:,1])&~rail[:,1];both=valid&wide
 s=dict(file=stem,duration_s=len(raw)/m['actual']['fps'],fps=m['actual']['fps'],exposure_us=m['actual']['timing']['exposure_us'],candidates=len(pts),valid_theta_12=int(valid.sum()),out_of_model_12=int((rmean[:,0]>LIMIT).sum()),saved_rail_12=int(rail[:,0].sum()),theta_q25=float(np.nanpercentile(th[valid,0],25)),theta_median=float(np.nanmedian(th[valid,0])),theta_q75=float(np.nanpercentile(th[valid,0],75)),median_r=float(np.nanmedian(rmean[:,0])),median_theta_change_20_minus_12=float(np.median(th[both,1]-th[both,0])),valid_theta_20=int(wide.sum()))
 summaries.append(s);panels.append((stamp,raw[0].copy(),mean,pts,th,valid,wide,rmean,s))
 print(json.dumps(s),flush=True)

fig,axes=plt.subplots(5,3,figsize=(16,19))
for i,(stamp,snap,mean,pts,th,valid,wide,rr,s) in enumerate(panels):
 lo,hi=np.percentile(mean,[5,99.8])
 axes[i,0].imshow(cells(snap.astype(float)),cmap='gray',vmin=lo,vmax=hi,extent=[0,snap.shape[1],snap.shape[0],0]);axes[i,0].set_title(f'18:{stamp[2:4]}:{stamp[4:]} | first-frame snapshot')
 axes[i,1].imshow(cells(mean),cmap='gray',vmin=lo,vmax=hi,extent=[0,snap.shape[1],snap.shape[0],0]);axes[i,1].scatter(pts[:,0],pts[:,1],s=13,facecolors='none',edgecolors='cyan',linewidths=.55);axes[i,1].set_title(f'Mean image | {len(pts)} intensity candidates')
 ax=axes[i,2];ax.hist(th[valid,0],bins=np.arange(0,95,5),weights=np.ones(valid.sum())/valid.sum()*100,alpha=.65,label=f'12 px ROI (n={valid.sum()})');ax.hist(th[wide,1],bins=np.arange(0,95,5),weights=np.ones(wide.sum())/wide.sum()*100,histtype='step',lw=1.6,label=f'20 px ROI (n={wide.sum()})');ax.set(xlim=(0,90),xlabel='Model theta (degrees)',ylabel='% of valid candidates / 5 degrees',title=f'{s["duration_s"]:.2f}s | median {s["theta_median"]:.1f} deg\n{s["out_of_model_12"]} outside model; {s["saved_rail_12"]} saved-rail flags');ax.legend(fontsize=8)
 for ax in axes[i,:2]:ax.set(xlabel='Sensor x (pixels)',ylabel='Sensor y (pixels)')
 sf,sa=plt.subplots(1,2,figsize=(15,7));sa[0].imshow(cells(snap.astype(float)),cmap='gray',vmin=lo,vmax=hi,extent=[0,snap.shape[1],snap.shape[0],0]);sa[0].set_title('First frame, 2x2 polarization-cell mean')
 sa[1].imshow(cells(mean),cmap='gray',vmin=lo,vmax=hi,extent=[0,snap.shape[1],snap.shape[0],0]);sa[1].scatter(pts[:,0],pts[:,1],s=28,facecolors='none',edgecolors='cyan',linewidths=.6)
 for j,(x,y) in enumerate(pts):sa[1].text(x+5,y,str(j+1),fontsize=4,color='cyan')
 sa[1].set_title('30-frame mean with candidate IDs');sf.suptitle('frame_stack_20261002-'+stamp);sf.tight_layout();sf.savefig(OUT/(stamp+'_snapshot.png'),dpi=180);plt.close(sf)
fig.suptitle('Five background-subtracted October stacks: snapshots and exploratory theta distributions\nOne value per intensity candidate; angle from time-mean channels; no motion validation or extra background subtraction',fontsize=14);fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'overview.png',dpi=150);plt.close(fig)
writecsv('spot_metrics.csv',rows);writecsv('file_summary.csv',summaries)
html='''<!doctype html><meta charset="utf-8"><title>October background-subtracted stacks: theta gallery</title><style>body{font:17px system-ui;max-width:1400px;margin:35px auto;padding:20px;background:#f7f8fb;color:#202735}img{max-width:100%%}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ccd}p{max-width:1100px;line-height:1.5}</style><h1>Five background-subtracted stacks</h1><p>Snapshots and simple intensity-selected theta distributions. All isolated candidates above the same robust intensity threshold are included; they have not been validated as stationary single rods. Each candidate contributes once, using anisotropy calculated from its time-mean four-channel sums. The first three recordings last 50 seconds; the last two last 2.08 seconds. Different durations can average motion differently.</p><p>Theta uses your supplied hole+Fresnel plotting coefficients (A=0.097840, B=0.011915, C=0.101852), rather than an independently validated calibration. Values above r=%.6f are counted separately, not clamped to 90 degrees. Already-subtracted stored pixels are used without another subtraction. Cyan circles mark intensity candidates on the mean image. Snapshots are 2x2 polarization-cell means with per-file display contrast.</p>'''%LIMIT
html+='<img src="overview.png"><h2>Numerical summary (12-pixel ROI)</h2><table><tr><th>Time</th><th>Candidates</th><th>Valid theta</th><th>Outside model</th><th>Theta median [25th,75th]</th><th>20-px minus 12-px median shift*</th></tr>'
for s in summaries:
 html+=f'<tr><td>{s["file"][-6:]}</td><td>{s["candidates"]}</td><td>{s["valid_theta_12"]}</td><td>{s["out_of_model_12"]}</td><td>{s["theta_median"]:.1f} [{s["theta_q25"]:.1f}, {s["theta_q75"]:.1f}] deg</td><td>{s["median_theta_change_20_minus_12"]:+.1f} deg</td></tr>'
html+='</table><p>*Paired candidates valid at both apertures. The wider ROI comparison tests sensitivity to residual background and spot extent; it is not a corrected answer. Histograms are normalized separately among valid candidates. Counts across files must not be treated as independent rods: the same field may have been recorded repeatedly.</p><p>Limitations: background subtraction may clip low pixels and hide earlier sensor saturation; a saved-value rail check cannot recover this. No per-channel throughput correction, motion selection, or sample identity assignment. These are exploratory apparent-angle distributions, not a hinge-versus-gold result or a measurement of interference suppression. Brightness-based selection itself can bias orientation sampling.</p><h2>Full-size snapshots and selection IDs</h2>'
for t in TIMES:html+=f'<h3>{t}</h3><a href="{t}_snapshot.png"><img src="{t}_snapshot.png"></a>'
html+='<p><a href="spot_metrics.csv">Per-candidate coordinates and metrics</a> | <a href="file_summary.csv">File summary</a> | <a href="config.json">Method configuration</a> | <a href="make_gallery.py">Reusable script</a></p>'
(OUT/'THETA_GALLERY.html').write_text(html,encoding='utf-8')
