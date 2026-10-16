"""First stack: minimum and 25th-95th percentiles from its original BG movie.

Display-only comparison; no detection changes or population selections.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detector_grid_trial import HERE, DAY, NAME
from background_overlap_check import OUT, HASHES, sha, cell, stats

METHODS = [('minimum', 0, 'Minimum'), ('p25', 25, '25th percentile'), ('median', 50, 'Median (50th)'),
           ('p60', 60, '60th percentile'), ('p70', 70, '70th percentile'), ('p80', 80, '80th percentile'),
           ('p90', 90, '90th percentile'), ('p95', 95, '95th percentile')]


def signature(p):
    s = p.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def build():
    OUT.mkdir(exist_ok=True)
    bgpath=next((DAY/'backgrounds').glob('background_min_20261012-183650*.npy'))
    assert sha(bgpath)==HASHES[0]
    bgmeta=json.loads(bgpath.with_suffix('.json').read_text())
    movie=DAY/'backgrounds'/bgmeta['source_stack_file']
    movie_sig=signature(movie)
    raw=np.load(movie,mmap_mode='r');n=bgmeta['actual']['frames']
    assert n==45 and len(raw)==46 and bgmeta['actual']['phase_marker_appended']
    assert bgmeta['background']['background_subtracted'] is False
    old=np.load(bgpath)
    profiles=np.empty((len(METHODS),*old.shape),dtype=np.float32)
    for y in range(0,old.shape[0],64):
        profiles[:,y:y+64]=np.percentile(raw[:n,y:y+64], [m[1] for m in METHODS],axis=0,method='linear')
    np.testing.assert_array_equal(profiles[0],old)
    assert all(np.all(a<=b) for a,b in zip(profiles[:-1],profiles[1:]))
    # Minimum, p25 and median land on integer samples; p60-p95 interpolate.
    # Match the acquisition code's float32 subtraction, clipping and rounding.
    np.testing.assert_array_equal(profiles[:3],np.rint(profiles[:3]))
    source=DAY/'frames'/NAME;source_sig=signature(source)
    meta=json.loads(source.with_suffix('.json').read_text());frames=np.load(source,mmap_mode='r')
    assert meta['background']['background_subtracted'] is True
    assert meta['background']['background_profile_sha256']==HASHES[0]
    assert meta['actual']['frames']==150 and len(frames)==151 and meta['actual']['phase_marker_appended']
    assert frames.shape[1:]==old.shape and frames.dtype==np.uint16
    sums=np.zeros((len(METHODS),*old.shape),dtype=np.uint32);zero_counts=np.zeros(len(METHODS),dtype=np.int64)
    additions=profiles-profiles[0]
    for i,frame in enumerate(frames[:150]):
        # Higher quantile >= old minimum, so saved clipping does not prevent this
        # additional subtraction: max(max(raw-Bmin,0)-(Bq-Bmin),0)=max(raw-Bq,0).
        sums[0]+=frame;zero_counts[0]+=np.count_nonzero(frame==0)
        for j in range(1,len(METHODS)):
            corrected=np.rint(np.maximum(frame.astype(np.float32)-additions[j],0)).astype(np.uint16)
            sums[j]+=corrected;zero_counts[j]+=np.count_nonzero(corrected==0)
        if (i+1)%50==0:print('Processed',i+1,'measurement frames',flush=True)
    means=sums.astype(np.float64)/150
    np.testing.assert_array_equal(means[0],np.load(HERE/'detector_grid_trial/mean.npy'))
    assert all(np.all(a>=b) for a,b in zip(means[:-1],means[1:]))
    assert signature(source)==source_sig and signature(movie)==movie_sig
    np.savez_compressed(OUT/'percentile_arrays.npz',sum_frames=sums,background_profiles=profiles,
                        percentiles=np.array([m[1] for m in METHODS]))
    rows=[]
    for j,(key,percentile,label) in enumerate(METHODS):
        display=np.rint(np.clip(cell(means[j])*20/4095,0,1)*255).astype(np.uint8)
        filename=f'percentile_{key}.webp'
        Image.fromarray(display).save(OUT/filename,lossless=True)
        rows.append(dict(key=key,percentile=percentile,label=label,image=filename,
            profile=stats(profiles[j]),extra_background=stats(additions[j]),
            saved_mean=stats(means[j]),zero_fraction=float(zero_counts[j]/(150*old.size))))
    fig,axes=plt.subplots(2,4,figsize=(20,10),layout='constrained')
    for ax,row,mean in zip(axes.flat,rows,means):
        ax.imshow(cell(mean),cmap='gray',vmin=0,vmax=4095/20,interpolation='nearest')
        ax.set_title(row['label']+' background',fontsize=12);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('First hinge stack 18:38:50 | mean of 150 frames | identical 20x display\nBackground estimates from the same 45-frame capture at 18:36:50',fontsize=13)
    fig.savefig(OUT/'percentile_overview.png',dpi=160);plt.close(fig)
    result=dict(source=NAME,source_signature=source_sig,sidecar_sha256=sha(source.with_suffix('.json')),
        background_capture=movie.name,background_capture_signature=movie_sig,
        background_capture_sha256=sha(movie),background_minimum_sha256=HASHES[0],
        background_frames=45,measurement_frames=150,shape=list(old.shape),
        gain=20,display_white_dn=4095/20,methods=rows,builder_sha256=sha(Path(__file__)),
        method='Per-pixel temporal percentile of original raw 45-frame BG movie, excluding marker. '
               'Additional nonnegative BG increment subtracted from each of the 150 saved measurement '
               'frames in float32, clipped at zero and rounded to nearest integer before averaging, '
               'matching acquisition arithmetic. Higher profiles permit extra subtraction despite prior clipping. '
               'Percentiles use linear interpolation; p60-p95 may be fractional. '
               'Display averages each 2x2 polarization cell and repeats to sensor grid. '
               'All methods share fixed 20x scaling; no contrast stretching or spot selection.',
        caveat='A higher background can remove faint signal as well as residue. This is an image comparison, '
               'not a claim that a percentile is the correct rod-free background. No second capture or '
               'survey stack is used to estimate these profiles; no added/removed spot analysis yet.')
    (OUT/'percentile_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    options=''.join(f'<option value="{key}">{label} background</option>' for key,_,label in METHODS)
    crops=''.join(f'<figure><figcaption>{label}</figcaption><canvas id="crop_{key}" width="352" height="352"></canvas></figure>' for key,_,label in METHODS)
    page=HTML.replace('__DATA__',json.dumps(result).replace('</','<\\/')).replace('__OPTIONS__',options).replace('__CROPS__',crops)
    (OUT/'percentile_comparison.html').write_text(page,encoding='utf-8')
    print(json.dumps({r['key']:dict(bg_mean=r['profile']['mean'],extra_mean=r['extra_background']['mean'],
                                 saved_mean=r['saved_mean']['mean'],zero_fraction=r['zero_fraction']) for r in rows}),flush=True)


HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>First stack · background percentile comparison</title>
<style>body{font:16px/1.5 system-ui;background:#f7f8f5;color:#263637;margin:0}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:27px}h2{font-size:21px}p{max-width:1150px}.small{font-size:14px;color:#526369}a{color:#087985}select,button{font:inherit;padding:6px}label{margin-right:18px}#view{max-height:75vh;overflow:auto;background:#111}#map{display:block;width:100%;height:auto;cursor:crosshair}.crops{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.crops canvas{width:100%;height:auto;image-rendering:pixelated;background:#111}figure{margin:0}figcaption{font-weight:600;margin-bottom:5px}.controls{padding:12px 0;position:sticky;top:0;background:#f7f8f5f5;z-index:2}.summary{width:100%}#error{color:#b43b32}@media(max-width:700px){.crops{grid-template-columns:1fr}}</style><main>
<h1>Background subtraction: minimum through 95th percentile</h1>
<p><b>First hinge stack: 18:38:50 on 12 October.</b> Compare the minimum, 25th, 50th (median), 60th, 70th, 80th, 90th and 95th percentiles. Every image is the mean of the same 150 measurement frames. All eight backgrounds come from the <b>same original 45-frame background capture at 18:36:50</b>. This isolates the choice of background statistic.</p>
<p class="small">Subtraction and clipping at zero happen on each frame before averaging. Fixed <b>20× display</b> throughout: black = 0, white = 204.75 intensity units. No automatic contrast stretch, spot overlays, or regional exclusions.</p>
<p><a href="percentile_spots.html"><b>Compare added/removed spots at 2 SD: minimum, median and 95th percentile</b></a></p>
<img class="summary" src="percentile_overview.png" alt="Eight full-field images with minimum through 95th-percentile background subtraction">
<div class="controls"><label>Large image <select id="method">__OPTIONS__</select></label><label>Zoom <select id="zoom"><option value="fit">Fit width</option><option value="native">Native pixels</option></select></label><span id="status" class="small"></span></div>
<div id="error"></div><div id="view"><canvas id="map" width="2464" height="2056"></canvas></div>
<h2>Same location under all eight subtractions</h2><p class="small">Click anywhere in the large image. The outlined 88×88-pixel crop appears below in all eight versions. Changing the large image keeps the same location.</p>
<p><button id="pair">Bright pair</button> <button id="residue">Pictured residue patch</button> <button id="feature">Large background feature</button> <span id="coords" class="small"></span></p>
<div class="crops">__CROPS__</div>
<p class="small">A higher percentile can remove faint signal as well as background residue. This comparison shows the images first; added/removed spot analysis is deferred. These are estimates from the first background movie alone, unlike the previous trial that averaged medians from two captures.</p>
<details><summary>How this is reconstructed from the saved stack</summary><p>Every tested percentile is everywhere at least as large as the original minimum, so its extra amount can be subtracted from each saved frame without recovering values lost below the old minimum. Subtraction uses float32, clips at zero and rounds to the nearest integer before averaging, matching the acquisition code. The 60th–95th percentiles interpolate between capture samples and can be fractional. Both synthetic marker frames are excluded.</p><p><a href="percentile_results.json">Parameters and provenance</a> · <a href="percentile_arrays.npz">Numeric background profiles and 150-frame sums</a></p></details>
<p><a href="index.html">Background and overlap review</a> · <a href="../detector_grid_trial/threshold_trial.html">Earlier detector gallery</a> · <a href="percentile_overview.png">Download comparison image</a></p>
</main><script>
const DATA=__DATA__,$=id=>document.getElementById(id),images={},methods=new Map(DATA.methods.map(m=>[m.key,m]));let point={x:541,y:515};
function load(key){const im=new Image();images[key]=im;im.onload=draw;im.onerror=()=>{$('error').textContent='Could not load '+methods.get(key).image};im.src=methods.get(key).image}
function draw(){const key=$('method').value,im=images[key],ctx=$('map').getContext('2d');ctx.clearRect(0,0,2464,2056);if(im?.complete&&im.naturalWidth)ctx.drawImage(im,0,0);ctx.strokeStyle='#68e1e5';ctx.lineWidth=2;ctx.strokeRect(point.x-44-.5,point.y-44-.5,88,88);for(const k of methods.keys()){const c=$('crop_'+k).getContext('2d');c.clearRect(0,0,352,352);c.imageSmoothingEnabled=false;const source=images[k];if(source?.complete&&source.naturalWidth)c.drawImage(source,point.x-44,point.y-44,88,88,0,0,352,352)}$('status').textContent=methods.get(key).label+' · 150-frame mean · 20×';$('coords').textContent=`Centre x=${point.x}, y=${point.y} sensor pixels`}
$('method').onchange=draw;$('zoom').onchange=()=>{$('map').style.width=$('zoom').value==='native'?'2464px':'100%'};
$('map').onclick=e=>{const r=$('map').getBoundingClientRect();point={x:Math.max(44,Math.min(2420,Math.round((e.clientX-r.left)*2464/r.width))),y:Math.max(44,Math.min(2012,Math.round((e.clientY-r.top)*2056/r.height)))};draw()};
for(const [id,x,y] of [['pair',541,515],['residue',1566,851],['feature',2060,500]])$(id).onclick=()=>{point={x,y};draw()};
for(const key of methods.keys())load(key);draw();
</script></html>'''

if __name__=='__main__':build()
