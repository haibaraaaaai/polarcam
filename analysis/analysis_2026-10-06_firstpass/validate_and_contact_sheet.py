import firstpass as f
import numpy as np,json,csv
from pathlib import Path
import matplotlib.pyplot as plt
B=f.BASE
cards=json.loads((B/'gallery_manifest.json').read_text());fields=[c for c in cards if c['id'].startswith('frame_stack')]
fig,axs=plt.subplots(5,4,figsize=(16,17))
for ax,c in zip(axs.flat,fields):
 im=plt.imread(B/'gallery_images'/f"{c['id']}.png");ax.imshow(im,cmap='gray');ax.axis('off');p=c['properties'];ax.set_title(c['id'].replace('frame_stack_','').replace('.npy','')+f"\n{p['seconds (fps-based)']:.2f}s | {p['exposure us']:.0f}us | BG {p['background already subtracted']} | sound {p['sound checkbox']}",fontsize=9)
fig.suptitle('Full-frame recording gallery — acquisition labels need your memory\nIndependent display contrast; BG = already background-subtracted; sound = manual checkbox',fontsize=15);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(B/'plots/full_frame_gallery.png',dpi=140);plt.close(fig)
checks=[]
for p in sorted((B/'traces').glob('*.npz')):
 z=np.load(p);info=json.loads(str(z['info_json']));r=json.loads(str(z['rois_json']));ch=z['channels'];checks.append(dict(file=p.name,frames=len(z['time_s']),rois=len(r),channel_total_identity=bool(np.allclose(ch.sum(-1),z['total'])),pair_total_identity=bool(np.allclose(z['pairs'].sum(-1),z['total'])),strictly_increasing_time=bool(np.all(np.diff(z['time_s'])>0))))
(B/'validation.json').write_text(json.dumps(checks,indent=2));print('Validation files',len(checks),'all structural checks',all(c['channel_total_identity'] and c['pair_total_identity'] and c['strictly_increasing_time'] for c in checks))
