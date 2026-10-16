"""User-selected DNA recordings: one row per spot, three panels only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from first_pass_review import xy, ROOT, RMAX, sha
from selected_spots_and_recordings import cells, guide

HERE = Path(__file__).resolve().parent
OUT = HERE/'spot_recording_gallery'
NUMBERS = [1, 2, 3, 5, 7, 10]


def build():
    data = json.loads((OUT/'results.json').read_text(encoding='utf8'))
    records = [next(r for r in data['records'] if r['label'] == f'DNA {n}') for n in NUMBERS]
    fig, axes = plt.subplots(len(records), 3, figsize=(13, 17), gridspec_kw={'width_ratios':[1, 1.35, 3.6]})
    fig.subplots_adjust(left=.075, right=.985, top=.945, bottom=.065, hspace=.54, wspace=.34)
    provenance=[]
    for row, record in enumerate(records):
        source=ROOT/record['path'];m=json.loads(source.with_suffix('.json').read_text(encoding='utf8'))
        sig=record['provenance'];assert source.stat().st_size==sig['bytes'] and source.stat().st_mtime_ns==sig['mtime_ns']
        assert sha(source.with_suffix('.json'))==sig['metadata_sha256']
        cache=HERE/'first_pass_review'/f'trace_{record["stamp"]}.npz'
        assert json.loads(cache.with_suffix('.json').read_text())==sig
        with np.load(cache) as z:channels=z['channels']
        cloud=xy(channels);radius=np.linalg.norm(cloud,axis=1)
        time=np.arange(len(channels))/record['metrics']['fps']
        assert len(channels)==record['metrics']['frames'] and np.isfinite(cloud).all()
        assert (radius<=1.42).all()
        movie=np.load(source,mmap_mode='r');x,y,size=record['bounds']
        l,t=x-m['roi']['x'],y-m['roi']['y'];crop=movie[0,t:t+size,l:l+size]
        a,b,c=axes[row]
        a.imshow(cells(crop),cmap='gray',vmin=0,vmax=4095/20,interpolation='nearest');a.set_axis_off()
        a.set_title(record['label']+'\n'+record['time']+(' · SATURATED' if record['saturated'] else ''),fontsize=10,
                    color='#a44722' if record['saturated'] else '#243d46',pad=8)
        b.scatter(cloud[:,0],cloud[:,1],s=.7,alpha=.15,color='#147f9b',linewidths=0,rasterized=True)
        guide(b);b.set_xticks([-1,0,1]);b.set_yticks([-1,0,1]);b.set_xlabel('X',labelpad=0);b.set_ylabel('Y',labelpad=0)
        b.tick_params(labelsize=9)
        c.plot(time,radius,lw=.32,color='#8255a0',alpha=.85,rasterized=True)
        c.axhline(RMAX,ls=':',color='.55',lw=.9)
        c.set(xlim=(0,time[-1]),ylim=(0,1.04),xlabel='Time (s)',ylabel='r')
        c.set_yticks([0,.5,1]);c.tick_params(labelsize=9);c.grid(axis='y',alpha=.13)
        # These records all fall inside the shared axis limits; no values clipped.
        assert radius.max()<1.04
        provenance.append(dict(label=record['label'],stamp=record['stamp'],frames=len(channels),first_image_frame=1,
                               aperture=record['bounds'],saturated=record['saturated'],r_min=float(radius.min()),
                               r_max=float(radius.max()),cache_sha256=sha(cache),source=record['provenance']))
    fig.suptitle('Selected DNA spot recordings',y=.992,fontsize=17)
    for ax, title in zip(axes[0], ['Frame 1 · 20× display', 'X/Y · all exposures', 'r versus time']):
        pos=ax.get_position();fig.text((pos.x0+pos.x1)/2,.972,title,ha='center',fontsize=11)
    fig.text(.5,.016,'Original recorded minimum-background subtraction · No temporal averaging or smoothing · Same X/Y and r scales across rows\n'
             'Dotted circle/line: model r limit ≈ 0.928.',ha='center',fontsize=9,color='.3')
    fig.savefig(OUT/'dna_selected_rows.png',dpi=220,facecolor='white')
    fig.savefig(OUT/'dna_selected_rows.svg',dpi=180,facecolor='white')
    plt.close(fig)
    (OUT/'dna_selected_rows.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    (OUT/'dna_selected_rows.html').write_text('''<!doctype html><meta charset="utf-8"><title>Selected DNA recordings</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>body{font:16px system-ui;margin:20px;background:#f3f5f5;color:#263d45}main{max-width:1500px;margin:auto}img{width:100%;height:auto}a{color:#087c96}</style><main>
<p><a href="index.html">Full recording gallery</a> · <a href="../hinge_anisotropy_stability/index.html">Hinge stability ranks</a> · <a href="dna_selected_rows.png" download>Download PNG</a> · <a href="dna_selected_rows.svg" download>Download SVG</a></p>
<a href="dna_selected_rows.png"><img src="dna_selected_rows.png" alt="DNA 1, 2, 3, 5, 7 and 10, one row each: first frame at 20x, X/Y from all exposures, and r versus time"></a></main>''',encoding='utf8')
    print('Saved six rows:',[r['label'] for r in records],flush=True)


if __name__=='__main__':build()
