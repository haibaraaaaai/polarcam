"""One-frame comparison using a background estimated outside moving features.

Read-only acquisition inputs; trial profiles stay in analysis. The permissive
motion list defines masks, not a new accepted rod population or theta analysis.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import cv2
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from background_motion_check import HERE, DAY, read, write, sha, cells, SCALE

OUT = HERE/'background_motion_check'
ASSETS = OUT/'183650'
MIN_CLEAN = 12
BASE_RADIUS = 18  # sensor pixels; mask at least 36x36 around each predicted centre
COMPONENT_MARGIN = 12
FRAME = 'frame_stack_20261012-183850.npy'
NAMES = ['minimum', 'median', 'motion_masked']
LABELS = ['Original minimum', 'Ordinary median', 'Moving-feature-masked median']


def rectangle(mask, x, y, rx, ry):
    h, w = mask.shape
    l, r = max(0, int(np.floor((x-rx)/2))), min(w, int(np.ceil((x+rx)/2)))
    t, b = max(0, int(np.floor((y-ry)/2))), min(h, int(np.ceil((y+ry)/2)))
    if l < r and t < b:
        mask[t:b, l:r] = True


def display_image(data):
    # Average only for display. Background estimation/subtraction is per raw pixel.
    cell = cells(data)
    good = np.isfinite(cell)
    grey = np.rint(np.clip(np.nan_to_num(cell)*SCALE*20, 0, 255)).astype('uint8')
    rgb = np.repeat(grey[..., None], 3, axis=2)
    rgb[~good] = [231, 136, 155]
    return rgb


def main():
    motion_path = HERE/'motion_selected_rods/183650/detections.json'
    old_path = OUT/'183650/results.json'
    motion, old = read(motion_path), read(old_path)
    profile_path = DAY/'backgrounds'/motion['profile']
    meta = read(profile_path)
    raw_path = DAY/'backgrounds'/motion['raw_stack']
    signature = dict(bytes=raw_path.stat().st_size, mtime_ns=raw_path.stat().st_mtime_ns)
    assert signature == motion['raw_signature'] == old['raw_signature']
    assert sha(profile_path) == motion['source_metadata_sha256']
    assert sha(old_path) == motion['stage_path_source_sha256']
    n = meta['actual']['frames']
    raw = np.load(raw_path, mmap_mode='r')
    h,w = raw.shape[1:]
    assert n == 45 and len(raw) == n+int(meta['actual']['phase_marker_appended'])
    assert not meta['background']['background_subtracted']
    minimum = np.load(profile_path.with_suffix('.npy')).astype('float32')
    assert sha(profile_path.with_suffix('.npy')) == old['profile_sha256']
    pose = np.asarray(motion['pose'])
    refs = np.array([[s['x'],s['y']] for s in motion['spots']])
    radii = np.full(refs.shape, BASE_RADIUS, dtype=float)
    # Enlarge the footprint when stronger per-frame components are wider than
    # the default. Keep every motion candidate, including saturated/overlapping.
    for fi, f in enumerate(old['frames']):
        tree = cKDTree(refs+pose[fi])
        for s in f['dynamic']:
            dist,j = tree.query([s['x'],s['y']])
            if dist <= 8:
                x,y,ww,hh = s['bbox']
                centre = refs[j]+pose[fi]
                extent = np.maximum(np.abs(centre-[x,y]), np.abs(centre-[x+ww,y+hh]))
                radii[j] = np.maximum(radii[j], extent+COMPONENT_MARGIN)
    masks = np.zeros((n,h//2,w//2),dtype=bool)
    for fi in range(n):
        for centre,radius in zip(refs+pose[fi],radii):
            rectangle(masks[fi], *centre, *radius)
        # Include transient per-frame components, including entrants at image
        # edges that cannot belong to the common-field full-length path list.
        for s in old['frames'][fi]['dynamic']:
            x,y,ww,hh = s['bbox']
            rectangle(masks[fi], x+ww/2,y+hh/2,ww/2+COMPONENT_MARGIN,hh/2+COMPONENT_MARGIN)
    count = n-masks.sum(0)
    valid_cells = count >= MIN_CLEAN
    valid = valid_cells.repeat(2,0).repeat(2,1)
    ordinary = np.empty((h,w),dtype='float32')
    masked = np.empty_like(ordinary)
    for y in range(0,h,64):
        block = np.array(raw[:n,y:y+64],dtype='float32')
        np.testing.assert_array_equal(block.min(0), minimum[y:y+64])
        ordinary[y:y+64] = np.median(block,axis=0)
        exclusion = masks[:,y//2:(y+64)//2].repeat(2,1).repeat(2,2)
        block[exclusion] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            masked[y:y+64] = np.nanmedian(block,axis=0)
    masked[~valid] = np.nan
    assert np.all(masked[valid] >= minimum[valid])
    np.save(ASSETS/'motion_masked_median.npy',masked)
    np.save(ASSETS/'motion_masked_clean_count.npy',count.astype('uint8'))
    # Since the new estimate is never below the stored minimum, this is exact
    # even where acquisition already clipped raw-minus-minimum to zero.
    frame_path = DAY/'frames'/FRAME
    fm = read(frame_path.with_suffix('.json'))
    assert fm['background']['background_subtracted']
    assert fm['background']['background_profile_sha256'] == old['profile_sha256']
    frame = np.array(np.load(frame_path,mmap_mode='r')[0],dtype='float32')
    backgrounds = [minimum, ordinary, masked]
    corrected = [np.rint(np.maximum(frame-(bg-minimum),0)) for bg in backgrounds]
    np.testing.assert_array_equal(corrected[0],frame)
    displays = []
    for key,data in zip(NAMES,corrected):
        rgb = display_image(data)
        Image.fromarray(rgb).save(ASSETS/f'motion_bg_frame1_{key}.webp',lossless=True)
        displays.append(rgb)
    fig,axes = plt.subplots(1,3,figsize=(15,4.9),layout='constrained')
    for ax,rgb,label in zip(axes,displays,LABELS):
        ax.imshow(rgb,extent=(0,w,h,0),interpolation='nearest')
        ax.set_title(label,fontsize=12);ax.set_axis_off()
    fig.suptitle('First DNA stack · Frame 1 · Same 20× display for all three',fontsize=16)
    fig.savefig(ASSETS/'motion_bg_frame1_comparison.png',dpi=180,facecolor='white');plt.close(fig)
    crops=[(541,515,'Bright rod pair'),(1566,851,'Previously inspected background texture'),(2060,500,'Large corner feature')]
    fig,axes=plt.subplots(3,3,figsize=(10,9.7),layout='constrained')
    for j,(x,y,label) in enumerate(crops):
        for i,(ax,rgb) in enumerate(zip(axes[j],displays)):
            ax.imshow(rgb[y//2-32:y//2+32,x//2-32:x//2+32],interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if j==0:ax.set_title(LABELS[i],fontsize=11)
            if i==0:ax.set_ylabel(label,fontsize=10)
    fig.suptitle('Identical 128×128-sensor-pixel crops · 20× display',fontsize=15)
    fig.savefig(ASSETS/'motion_bg_frame1_crops.png',dpi=160,facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5.6),layout='constrained')
    im=ax.imshow(count,cmap='viridis',vmin=0,vmax=n,extent=(0,w,h,0))
    ax.set(title=f'Unmasked observations at each pixel · require at least {MIN_CLEAN}',xlabel='Sensor x',ylabel='Sensor y')
    fig.colorbar(im,ax=ax,label='Frames available')
    fig.savefig(ASSETS/'motion_bg_clean_counts.png',dpi=145,facecolor='white');plt.close(fig)
    difference=(masked-ordinary)[valid]*SCALE
    methods=[]
    for key,label,bg,cor in zip(NAMES,LABELS,backgrounds,corrected):
        methods.append(dict(key=key,label=label,image=f'183650/motion_bg_frame1_{key}.webp',
            background_mean_8bit=float(bg[valid].mean())*SCALE,
            frame1_remaining_mean_8bit=float(cor[valid].mean())*SCALE,
            frame1_zero_percent=float(np.mean(cor[valid]==0))*100))
    summary=dict(background_capture=raw_path.name,measurement_frame=FRAME,frame_number=1,capture_frames=n,
        motion_candidates=len(refs),minimum_clean_frames=MIN_CLEAN,
        valid_area_percent=float(valid.mean())*100,any_mask_area_percent=float(np.mean(count<n))*100,
        per_frame_mask_percent_quantiles=(np.percentile(masks.mean((1,2)),[0,50,100])*100).tolist(),
        unmasked_frame_count_quantiles=np.percentile(count,[0,10,25,50,75,90,100]).tolist(),
        masked_minus_median_8bit_quantiles=np.percentile(difference,[1,10,50,90,99]).tolist(),
        percent_pixels_more_than_one_raw_DN_below_median=float(np.mean(difference < -SCALE))*100,
        percent_pixels_more_than_one_raw_DN_above_median=float(np.mean(difference > SCALE))*100,
        methods=methods,brightness_units='Ungained 0-255 equivalent per original sensor pixel; saved DN * 255/4095. All statistics use the same valid pixel area.',
        method='Use every permissive motion-only candidate, not the later stricter accepted rod list. Predict each position in every real background frame with the cached stage trajectory. Mask at least +/-18 sensor pixels, enlarged to encompass matched dynamic-component bounding boxes plus 12 pixels. Also mask per-frame dynamic components plus 12 pixels, including field-edge entrants. Mask complete 2x2 analyzer cells, but take the temporal median independently at each original analyzer pixel. At least 12 unmasked observations required; otherwise NaN. Exclude the synthetic phase marker. No temporal averaging of the measurement: compare frame 1 only, clipping and rounding the additional subtraction.',
        limitations='Masking does not prove that all remaining signals are background. Missed rods, static sample features and long exposure trails can remain. The permissive list can mask noise and remove useful observations. Mask estimates and values share the same capture, so selection bias remains possible. Stationary optical/interference structure that changes after capture is not repaired by this estimator. Unsupported pixels remain missing, never filled with a fallback.',
        sources=dict(motion_detections_sha256=sha(motion_path),stage_motion_sha256=sha(old_path),
            raw_capture_signature=signature,original_minimum_sha256=old['profile_sha256'],
            measurement_signature=dict(bytes=frame_path.stat().st_size,mtime_ns=frame_path.stat().st_mtime_ns),
            measurement_metadata_sha256=sha(frame_path.with_suffix('.json')),script_sha256=sha(Path(__file__))))
    write(ASSETS/'motion_bg_comparison.json',summary)
    render(summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['method','limitations','sources']},indent=2))


def render(d):
    rows=''.join(f'<tr><td>{m["label"]}</td><td>{m["background_mean_8bit"]:.3f}</td><td>{m["frame1_remaining_mean_8bit"]:.3f}</td><td>{m["frame1_zero_percent"]:.1f}%</td></tr>' for m in d['methods'])
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mask moving features before estimating background</title>
<style>body{margin:0;background:#f5f6f3;color:#263943;font:16px/1.5 system-ui}main{max-width:1300px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:20px}a{color:#176d91}button,select{font:inherit;padding:6px}img{max-width:100%}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.crops{display:flex;gap:16px;flex-wrap:wrap}.crop{width:320px;height:320px;image-rendering:pixelated}#map{width:100%;height:auto;display:block}table{border-collapse:collapse}td,th{padding:8px 12px;border-bottom:1px solid #ccd4d5;text-align:left}.small{font-size:14px}.scroll{overflow:auto;max-height:760px}details{margin:20px 0}summary{cursor:pointer}</style>
<main><h1>Mask moving features, then estimate the background</h1><p>First DNA stack, <b>18:38:50 · frame 1</b>. Background: the 45-frame capture at 18:36:50. Every view uses the same 20× display; brightness is not auto-stretched.</p>
<p>Use all <b>1,434 permissive motion candidates</b> to mask the predicted moving features in each background frame. Take a median of the remaining observations separately at each original sensor pixel. At least 12 observations are required. <b>__VALID__% of the sensor has enough observations; pink means insufficient data.</b></p>
<div class="row"><label>View <select id="method"><option value="minimum">Original minimum subtraction</option><option value="median">Ordinary median subtraction</option><option value="motion_masked" selected>Moving-feature-masked median subtraction</option></select></label><button id="pair">Rod pair</button><button id="texture">Background texture</button><button id="corner">Corner feature</button></div>
<p class="small">Click the image to compare the same 128×128 sensor-pixel crop below. Display pixels average aligned 2×2 cells; the actual background subtraction retains the four analyzer channels.</p>
<div class="scroll"><canvas id="map" width="1232" height="1028"></canvas></div><p id="position"></p>
<div class="crops"><div>Original minimum<br><canvas id="crop_minimum" class="crop" width="320" height="320"></canvas></div><div>Ordinary median<br><canvas id="crop_median" class="crop" width="320" height="320"></canvas></div><div>Moving-feature-masked median<br><canvas id="crop_motion_masked" class="crop" width="320" height="320"></canvas></div></div>
<p>Compared on the <b>same valid pixel area</b>; intensity is ungained 0–255 equivalent per sensor pixel.</p><table><tr><th>Background</th><th>Mean estimated BG</th><th>Mean remaining frame signal</th><th>Pixels clipped to zero</th></tr>__ROWS__</table>
<p>The masked estimate avoids using predicted rod passages; the minimum instead chooses the lowest observation and can leave a positive noise offset. If rods occupy a pixel for only a small part of the capture, the ordinary median already rejects much of their contribution, so masking may make little visible difference. These images do not establish which estimate gives the true rod intensity or angle. A static feature that changes after the background capture can still leave residue.</p>
<details><summary>Mask coverage and exact method</summary><img src="183650/motion_bg_clean_counts.png"><p>Minimum mask half-width: 18 sensor pixels, enlarged to cover observed components plus 12 pixels. Per-frame dynamic detections are also masked to cover transient features and field-edge entrants. Saturated and overlapping moving candidates are included in the masking list. Complete 2×2 cells are masked together; the four analyzer values are never combined for background estimation. No unavailable region is silently filled.</p><p>At roughly 10 fps, the sampled stage positions do not establish the path within each 1.194 ms exposure. The margins help but cannot guarantee that every trail is excluded. Masking can also remove noise fluctuations and bias the remaining sample. The camera background and measurement files are unchanged.</p></details>
<p><a href="183650/motion_bg_frame1_comparison.png">Full-frame comparison</a> · <a href="183650/motion_bg_frame1_crops.png">Fixed crop comparison</a> · <a href="183650/motion_bg_comparison.json">Method and numbers</a> · <a href="index.html">Earlier background-motion review</a></p></main>
<script>const data=__DATA__,images={},$=id=>document.getElementById(id);let point={x:541,y:515};function draw(){const c=$('map').getContext('2d'),im=images[$('method').value];c.clearRect(0,0,1232,1028);if(im?.complete&&im.naturalWidth)c.drawImage(im,0,0);c.strokeStyle='#60dce6';c.lineWidth=1;c.strokeRect(point.x/2-32,point.y/2-32,64,64);for(const m of data.methods){const ctx=$('crop_'+m.key).getContext('2d'),im=images[m.key];ctx.clearRect(0,0,320,320);ctx.imageSmoothingEnabled=false;if(im?.complete&&im.naturalWidth)ctx.drawImage(im,point.x/2-32,point.y/2-32,64,64,0,0,320,320)}$('position').textContent=`Crop centre: x=${point.x}, y=${point.y} sensor pixels.`}for(const m of data.methods){const im=new Image();images[m.key]=im;im.onload=draw;im.src=m.image}$('method').onchange=draw;$('map').onclick=e=>{const b=$('map').getBoundingClientRect();point={x:Math.max(64,Math.min(2400,Math.round((e.clientX-b.left)/b.width*2464))),y:Math.max(64,Math.min(1992,Math.round((e.clientY-b.top)/b.height*2056)))};draw()};for(const [id,x,y] of [['pair',541,515],['texture',1566,851],['corner',2060,500]])$(id).onclick=()=>{point={x,y};draw()};draw();</script></html>'''
    page=page.replace('__VALID__',f'{d["valid_area_percent"]:.2f}').replace('__ROWS__',rows).replace('__DATA__',json.dumps(dict(methods=d['methods'])))
    (OUT/'masked_background_frame1.html').write_text(page,encoding='utf8')


if __name__=='__main__':
    main()
