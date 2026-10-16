"""One-stack SD/MAD threshold trial with explicit spatial and overlap cuts."""
from pathlib import Path
import json, base64, io
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detector_grid_trial import HERE, OUT, DAY, NAME, sha, match
from background_region_trial import ELLIPSE, inside
from polarcam.live import Detection_alg_offline as gui
from polarcam.live.intensity_spots import channel_means, anisotropy


def overlap(a, b):
    """Positive shared area between half-open extraction windows, not display rings."""
    al, ar, at, ab = a['bounds']
    bl, br, bt, bb = b['bounds']
    return max(al, bl) < min(ar, br) and max(at, bt) < min(ab, bb)


def select_spots(rows):
    """Greedy suppression by peak DoG; preserve every exclusion and its reason."""
    eligible = []
    for s in rows:
        s['suppressed_by'] = None
        if not s['accepted']:
            s['selection'] = 'oversize'
        elif s['inside_ellipse']:
            s['selection'] = 'region'
        elif s['saturated']:
            s['selection'] = 'saturated'
        elif s['r'] is None:
            s['selection'] = 'invalid'
        else:
            eligible.append(s)
    kept = []
    for s in sorted(eligible, key=lambda s: (-s['dog_peak'], s['component'])):
        stronger = next((q for q in kept if overlap(s, q)), None)
        if stronger is None:
            s['selection'] = 'kept'
            kept.append(s)
        else:
            s['selection'] = 'overlap'
            s['suppressed_by'] = stronger['id']
            assert stronger['dog_peak'] >= s['dog_peak']
    assert all(not overlap(a, b) for i, a in enumerate(kept) for b in kept[i+1:])
    return kept


def summarize(rows, bins, r_max):
    kept = [s for s in rows if s['selection'] == 'kept']
    radii = np.array([s['r'] for s in kept])
    counts, _ = np.histogram(radii, bins)
    assert int(counts.sum()) == len(kept)
    return dict(n=len(kept), median=float(np.median(radii)) if len(kept) else None,
                counts=counts.tolist(), percent=(counts/max(1, len(kept))*100).tolist(),
                above_model_limit=int(np.count_nonzero(radii > r_max)),
                below_02=int(np.count_nonzero(radii < .2)),
                in_02_04=int(np.count_nonzero((radii >= .2) & (radii < .4))))


def build():
    source = DAY / 'frames' / NAME
    meta = json.loads(source.with_suffix('.json').read_text())
    stat = source.stat()
    signature = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                     sidecar=sha(source.with_suffix('.json')))
    assert signature == json.loads((OUT / 'mean_signature.json').read_text())
    assert meta['actual']['frames'] == 150
    mean = np.load(OUT / 'mean.npy')
    image = gui.intensity_detection_map(mean)
    peakmeta = meta['fetch_frames_output']['raw_peak']
    peakpath = source.parent / peakmeta['path']
    assert sha(peakpath) == peakmeta['sha256'] and peakmeta['frames'] == 150
    peak = np.load(peakpath)
    assert peak.shape == mean.shape
    work = image[10:-10, 10:-10]
    dog = (cv2.GaussianBlur(work, (0, 0), sigmaX=1, borderType=cv2.BORDER_REPLICATE)
           - cv2.GaussianBlur(work, (0, 0), sigmaX=6, borderType=cv2.BORDER_REPLICATE))
    mu, sd = gui._finite_stats(dog)
    median = float(np.median(dog))
    mad = float(np.median(np.abs(dog-median)))
    robust_sd = 1.4826 * mad
    assert robust_sd > 0
    levels = {}
    mad_levels = {}
    baseline_labels = None
    for family, k in [(family, k) for family in ['sd', 'mad'] for k in [8, 6, 4, 3, 2]]:
        threshold = max(0, mu + k * sd) if family == 'sd' else max(0, median + k * robust_sd)
        mask = (dog > threshold).astype(np.uint8)
        _, labels, stats, centers = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        # 0/255 display scaling leaves the connected foreground exactly unchanged.
        white_mask = mask * 255
        _, _, white_stats, white_centers = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
        np.testing.assert_array_equal(stats, white_stats)
        np.testing.assert_array_equal(centers, white_centers)
        full_mask = np.zeros(mean.shape, dtype=np.uint8)
        full_mask[10:-10, 10:-10] = white_mask
        png = io.BytesIO()
        Image.fromarray(full_mask).convert('1', dither=Image.Dither.NONE).save(png, format='PNG')
        binary_mask_png = 'data:image/png;base64,' + base64.b64encode(png.getvalue()).decode('ascii')
        rows = []
        for component in range(1, len(stats)):
            bx, by, width, height, area = map(int, stats[component])
            if area < 10:
                continue
            x, y = map(float, centers[component] + 10)
            l, t = round(x) - 9, round(y) - 9
            assert l >= 0 and t >= 0 and l+18 <= mean.shape[1] and t+18 <= mean.shape[0]
            raw_peak = int(peak[t:t+18, l:l+18].max())
            bounds = [l, l+18, t, t+18]
            channels = channel_means(mean, [bounds])[0]
            xy = anisotropy(channels)
            valid = bool(np.isfinite(xy).all())
            component_pixels = labels[by:by+height, bx:bx+width] == component
            dog_peak = float(dog[by:by+height, bx:bx+width][component_pixels].max())
            rows.append(dict(id=f'{family}{k}-{component}', component=component, x=x, y=y,
                             area=area, bbox=[bx+9.5, by+9.5, width, height],
                             accepted=area <= 250, raw_peak=raw_peak,
                             saturated=raw_peak >= 4094, bounds=bounds, dog_peak=dog_peak,
                             inside_ellipse=bool(inside(x, y)), channels=channels.tolist(),
                             xy=xy.tolist() if valid else None,
                             r=float(np.linalg.norm(xy)) if valid else None))
        accepted = [s for s in rows if s['accepted']]
        # Independently call the actual GUI detector, with only k changed.
        effective_k = (threshold-mu)/sd
        direct = np.asarray(gui.find_spot_centers_dog(work, k_std=effective_k)) + 10
        np.testing.assert_allclose([[s['x'], s['y']] for s in accepted], direct,
                                   rtol=0, atol=1e-12)
        if family == 'sd' and k == 8:
            baseline = accepted
            baseline_labels = labels.copy()
            old = json.loads((OUT / 'results.json').read_text())
            previous = [s for s in old['methods']['current']['spots'] if s['pass250']]
            np.testing.assert_allclose([[s['x'], s['y']] for s in accepted],
                                       [[s['x'], s['y']] for s in previous], rtol=0, atol=1e-12)
            assert len(baseline) == 145
        pairs, lost, new = match(baseline, accepted)
        paired_ids = {accepted[j]['id']: baseline[i]['id'] for i, j in pairs.items()}
        for s in rows:
            s['baseline_id'] = paired_ids.get(s['id'])
            s['status'] = ('matched' if s['baseline_id'] else 'new') if s['accepted'] else 'oversize'
        # Threshold masks are nested, although size-filtered candidate lists need not be.
        lost_rows = []
        for i in lost:
            s = baseline[i]
            bx, by, width, height = s['bbox']
            left, top = round(bx-9.5), round(by-9.5)
            old_region = baseline_labels[top:top+height, left:left+width] == s['component']
            descendants = np.unique(labels[top:top+height, left:left+width][old_region])
            assert len(descendants) == 1 and descendants[0] != 0
            dest = int(descendants[0])
            area = int(stats[dest, cv2.CC_STAT_AREA])
            reason = (f'Component grew/merged to {area} pixels, above the 250-pixel limit'
                      if area > 250 else 'Merged or shifted: no unique centroid match within 6 pixels')
            lost_rows.append(dict(**s, reason=reason, descendant_area=area))
        kept = select_spots(rows)
        cuts = {reason: sum(s['selection'] == reason for s in accepted)
                for reason in ['region', 'saturated', 'invalid', 'overlap', 'kept']}
        assert sum(cuts.values()) == len(accepted)
        target = levels if family == 'sd' else mad_levels
        target[str(k)] = dict(k=k, family=family, threshold=threshold,
            binary_mask_png=binary_mask_png,
            equivalent_sd_k=effective_k, spots=rows, lost=lost_rows, cuts=cuts,
            counts=dict(accepted=len(accepted), matched=len(pairs), new=len(new),
                        lost=len(lost), oversized=sum(not s['accepted'] for s in rows),
                        saturated=sum(s['saturated'] for s in accepted)))
        assert len(accepted) == len(pairs) + len(new)
        assert len(baseline) == len(pairs) + len(lost)
    r_max = json.loads((HERE/'radius_detector_comparison/results.json').read_text())['r_max']
    all_levels = list(levels.values()) + list(mad_levels.values())
    max_r = max(s['r'] for v in all_levels for s in v['spots'] if s['selection'] == 'kept')
    bins = np.arange(0, max(1.02, np.ceil(max_r/.02)*.02+.02)+.001, .02)
    for v in all_levels:
        v['distribution'] = summarize(v['spots'], bins, r_max)
    d = dict(source=NAME, source_signature=signature, mean=mu, sd=sd, frames=150,
             median=median, mad=mad, robust_sd=robust_sd, ellipse=ELLIPSE,
             r_max=r_max, bins=bins.tolist(), levels=levels, mad_levels=mad_levels,
             shape=list(mean.shape), raw_peak_sha256=peakmeta['sha256'],
             detector_sha256=sha(Path(gui.__file__)), builder_sha256=sha(Path(__file__)),
             binary_display='Exact detector foreground before area/region/saturation/overlap cuts: '
                            'DoG > selected threshold is white (255), otherwise black (0). '
                            'Excluded 10-pixel edge is black. Connected-component areas and centroids '
                            'verified identical for 0/1 versus 0/255 masks at every setting. '
                            'DoG peak ranking and original-intensity r extraction are unchanged.',
             method='GUI intensity detector: repeated 2x2 cell mean, 10-pixel edge crop, '
                    'DoG sigma 1/6 sensor pixels, 8-connectivity, area 10-250 sensor pixels. '
                    'SD: mean+k*SD. MAD: median+k*1.4826*median(abs(DoG-median)). '
                    'Both statistics use the full cropped image before regional exclusion.',
             matching='One-to-one centroid assignment within 6 sensor pixels against k=8; '
                      'approximate correspondence, not confirmed rod identity.',
             selection='After size filtering: remove all centres inside the saved ellipse, '
                       'then any raw saturation >=4094 in the 18x18 aperture over 150 frames, '
                       'then nonfinite r; greedily keep the highest component DoG peak and '
                       'suppress weaker candidates sharing any extraction pixels with a kept candidate. '
                       'Ties use component order. No low-background, 2x survival, theta or radius cut.',
             extraction='One r per candidate, from the four channel means in the original '
                        '1x-subtracted 150-frame mean over the unchanged 18x18 centroid-centred aperture. '
                        'X=(I0-I90)/(I0+I90); Y=(I45-I135)/(I45+I135); r=hypot(X,Y). '
                        'Overlap suppression does not deblend signals inside a merged component.')
    Image.fromarray(np.rint(np.clip(image*5/4095, 0, 1)*255).astype(np.uint8)).save(OUT/'mean_gain5.png')
    (OUT/'threshold_results.json').write_text(json.dumps(d, indent=2, allow_nan=False), encoding='utf-8')
    (OUT/'threshold_trial.html').write_text(HTML.replace('__DATA__', json.dumps(d).replace('</', '<\\/')), encoding='utf-8')
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    for ax, level in zip(axes.flat, levels.values()):
        ax.imshow(image, cmap='gray', vmin=0, vmax=4095/20, interpolation='nearest')
        for status, color in [('matched', '#65e2e6'), ('new', '#ffb331')]:
            selected = [s for s in level['spots'] if s['status'] == status and s['selection'] == 'kept']
            if selected:
                ax.scatter([s['x'] for s in selected], [s['y'] for s in selected],
                           s=12, facecolors='none', edgecolors=color, linewidths=.5)
        c = level['counts']
        ax.add_patch(plt.matplotlib.patches.Ellipse((ELLIPSE['cx'], ELLIPSE['cy']),
                     2*ELLIPSE['rx'], 2*ELLIPSE['ry'], fill=False, edgecolor='#ff6262', lw=.8))
        ax.set_title(f"SD k = {level['k']}: {level['cuts']['kept']} retained")
        ax.set_xticks([]); ax.set_yticks([])
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.03, .85, 'Cyan: matched to SD k=8\nAmber: new vs SD k=8\n\nLarge background ellipse removed.\nOverlapping 18x18 windows:\nretain the highest filtered peak.\n\nRaw saturation excluded.\nAll finite r values retained.',
                       transform=axes.flat[-1].transAxes, va='top', fontsize=12, linespacing=1.6)
    fig.suptitle('First DNA stack, 18:38:50 · SD thresholds with the requested exclusions', fontsize=16)
    fig.savefig(OUT/'threshold_overview.png', dpi=160); plt.close(fig)
    radius_figures(d)
    print(json.dumps({'scale': {'sd': sd, 'robust_sd': robust_sd},
        'levels': {f'{v["family"]}{v["k"]}': dict(detected=v['counts']['accepted'],
            cuts=v['cuts'], median_r=v['distribution']['median'],
            above_limit=v['distribution']['above_model_limit'], threshold=v['threshold']) for v in all_levels}}))


def radius_figures(d):
    fig, axes = plt.subplots(5, 2, figsize=(12.5, 15), sharex=True, sharey=True, layout='constrained')
    for col, family in enumerate(['sd', 'mad']):
        levels = d['levels'] if family == 'sd' else d['mad_levels']
        for ax, k in zip(axes[:, col], [2, 3, 4, 6, 8]):
            v = levels[str(k)]; q = v['distribution']
            ax.stairs(q['percent'], d['bins'], fill=True, alpha=.45,
                      color='#087e8b' if family == 'sd' else '#b36a15')
            ax.axvline(d['r_max'], color='#667078', ls=':', lw=1)
            ax.set_title(f'{family.upper()} k={k}: n={q["n"]}, median r={q["median"]:.3f} | cutoff {v["threshold"]:.2f}', fontsize=11)
            ax.set_xlim(0, d['bins'][-1]); ax.grid(axis='y', alpha=.15)
            ax.spines[['top', 'right']].set_visible(False)
            if col == 0: ax.set_ylabel('Spots per 0.02 bin (%)')
    for ax in axes[-1]: ax.set_xlabel('Anisotropy radius r · original 1× background subtraction')
    fig.suptitle('First DNA stack · fixed background ellipse removed, stronger overlapping candidate retained\nSD = mean + k·SD; MAD = median + k·1.4826·MAD · raw saturation excluded; no radius cut', fontsize=13)
    fig.savefig(OUT/'threshold_r_distributions.png', dpi=150)
    plt.close(fig)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spot thresholds and r · first DNA stack</title>
<style>
body{font:16px/1.5 system-ui;color:#263637;background:#f7f8f5;margin:0}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:27px}h2{font-size:21px}a{color:#087985}select,button{font:inherit;padding:6px}label{display:inline-block;margin:6px 18px 6px 0}table{border-collapse:collapse}th,td{padding:7px 16px 7px 0;border-bottom:1px solid #d3ddd7;text-align:left}.small{font-size:14px;color:#526369}.controls{position:sticky;top:0;background:#f7f8f5f5;z-index:2;padding:8px 0;border-block:1px solid #d3ddd7}#view{max-height:77vh;overflow:auto;background:#111}#map{display:block;width:100%;height:auto;cursor:crosshair}.inspect{display:flex;gap:24px;flex-wrap:wrap;margin-top:20px}.inspect canvas{width:352px;height:352px;max-width:90vw;image-rendering:pixelated;background:#111}.table{overflow:auto}#detail{max-width:600px}p{max-width:1200px}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:20px}.plot{background:white;border:1px solid #d6ded8;padding:12px}.plot svg{width:100%;height:auto}.legend{font-size:14px}summary{cursor:pointer}#table tr.active{background:#e0efeb}#export{cursor:pointer}
</style><main>
<h1>Spot thresholds and anisotropy radius</h1>
<p><b>18:38:50, 12 October · first DNA stack only.</b> Same 150-frame mean and original 1× background subtraction. Both methods use the GUI intensity map, DoG widths of 1 and 6 sensor pixels, a 10-pixel edge crop and component area 10–250 sensor pixels.</p>
<p><b>Requested exclusions:</b> remove every candidate centred inside the previous large background ellipse, then raw-saturated or undefined-r candidates. For overlapping <b>18×18 extraction windows</b>, keep the candidate with the highest filtered peak, working from strongest to weakest. The smaller central feature is left alone. No 2× survival or low-background mask is applied.</p>
<details><summary>Exact SD, MAD and overlap definitions</summary><p><b>SD:</b> mean + k × SD. <b>MAD:</b> median + k × 1.4826 × median(|filtered image − median|). Statistics are calculated on the same full cropped filtered image, before the spatial exclusion. The factor 1.4826 expresses MAD on a Gaussian SD scale. This is a change of threshold statistic within the same detector, not the complete supplied September pipeline.</p><p id="scale"></p><p>The fixed ellipse has centre (2060, 500) and radii (320, 220) sensor pixels. “Overlapping” means the rounded 18×18 extraction windows share at least one pixel. Strength is the maximum DoG value inside the connected component, not r or theta. A weaker candidate is suppressed only if its window overlaps an already retained stronger candidate. Ties use component order.</p><p>This rule prevents overlapping extraction windows; it does not unmix the signal of rods already merged into one connected component. Candidate centres and extraction windows remain those of the existing detector. Brightness-based selection can still favour brighter orientations; the resulting histogram is not a completeness-corrected angle distribution.</p></details>
<div class="controls"><label>Threshold model <select id="family"><option value="sd">Mean + k × SD</option><option value="mad">Median + k × scaled MAD</option></select></label><label>k <select id="k"><option>8</option><option>6</option><option>4</option><option>3</option><option selected>2</option></select></label><label>Histogram <select id="units"><option value="percent">Percent per bin</option><option value="counts">Counts per bin</option></select></label><span id="count"></span></div>
<div class="plots"><section class="plot"><h2 id="currentTitle"></h2><svg id="hist" viewBox="0 0 640 340" role="img" aria-label="Selected radius distribution"></svg><div id="histStats" class="small"></div></section><section class="plot"><h2 id="compareTitle"></h2><svg id="compare" viewBox="0 0 640 340" role="img" aria-label="Radius distributions across thresholds"></svg><div id="legend" class="legend"></div></section></div>
<p class="small">Bin width 0.02. One r per retained candidate, from the four channel means in the original 1×-subtracted mean over its 18×18 aperture. X=(I0−I90)/(I0+I90), Y=(I45−I135)/(I45+I135), r=√(X²+Y²). Every finite r is retained, including values above the dotted optical-model limit. Raw saturation (any pixel ≥4094 over 150 frames, before background subtraction) remains excluded.</p>
<details><summary>Counts and exclusions for every threshold</summary><div class="table" id="table"></div><p class="small">Exclusion columns are sequential and mutually exclusive: region, saturation, invalid r, then overlap. Detected = passed the unchanged 10–250-pixel area rule. Oversized components are listed separately.</p></details>
<h2>Inspect retained and excluded candidates</h2>
<label><input type="checkbox" id="binary">Binary threshold view (above cutoff = white; otherwise black)</label>
<label>Image <select id="layer"><option value="mean">Saved mean · 20×</option><option value="mean_gain5">Saved mean · 5×</option><option value="background">Background · 20×</option></select></label><label>Circles <select id="marks"><option value="all">All retained candidates</option><option value="new">Retained, new vs SD k=8</option><option value="none">Off</option></select></label><label><input type="checkbox" id="removed">Show excluded candidates</label><label><input type="checkbox" id="large">Show oversized components</label><label><input type="checkbox" id="ellipse" checked>Show excluded region</label><label><input type="checkbox" id="windows">Show extraction windows</label><label>Zoom <select id="zoom"><option value="fit">Fit width</option><option value="native">1 sensor pixel per screen pixel</option></select></label>
<p class="small" id="binaryNote" hidden>This is the exact threshold mask already used for spot detection, before size and other exclusions. White pixels need not become retained spots. Switching this view does not change detections or r; the excluded 10-pixel border is black.</p>
<p class="small">Cyan: retained, matched to the original SD k=8 detector. Amber: retained, newly matched to no original candidate. Grey: weaker overlapping candidate removed. Red: region or saturation/invalid exclusion. Magenta boxes: above the 250-pixel size limit. Original-candidate correspondence is approximate (one-to-one centroids within 6 pixels). Click to inspect; the crop stays fixed when you change methods.</p>
<div id="view"><canvas id="map" width="2464" height="2056"></canvas></div>
<div class="inspect"><div><b>Selected image</b><br><canvas id="crop" width="352" height="352"></canvas></div><div><b>Recorded background · 20×</b><br><canvas id="bgcrop" width="352" height="352"></canvas></div><div id="detail">Click a candidate or an unmarked feature to inspect an 88×88 crop.</div></div>
<p class="small" id="formula"></p><p><a href="threshold_r_distributions.png">All ten radius histograms</a> · <a href="threshold_overview.png">SD detection overview</a> · <a href="index.html">Grid and size comparison</a> · <a href="threshold_results.json">Coordinates, exclusions and numerical results</a> · <button id="export">Export current retained list</button></p>
</main><script>
const DATA=__DATA__,$=id=>document.getElementById(id),images={};let point=null;
const COLORS={8:'#707b85',6:'#9366ad',4:'#2672ae',3:'#27886a',2:'#d07b12'};
function family(){return $('family').value==='sd'?DATA.levels:DATA.mad_levels}
function level(){return family()[$('k').value]}
function displayName(){return $('binary').checked?'mask_'+level().family+'_'+level().k:$('layer').value}
function load(name){if(images[name])return images[name];const im=new Image();images[name]=im;im.onload=draw;if(name.startsWith('mask_')){const [,f,k]=name.split('_');im.src=(f==='sd'?DATA.levels:DATA.mad_levels)[k].binary_mask_png}else im.src=name+'.png';return im}
function visible(){const mode=$('marks').value;if(mode==='none')return [];return level().spots.filter(s=>s.selection==='kept'&&(mode==='all'||s.status==='new'))}
function overlays(ctx,offsetX=0,offsetY=0,scale=1,selection=true){
 ctx.save();ctx.translate(offsetX,offsetY);ctx.scale(scale,scale);ctx.lineWidth=1.6/scale;ctx.setLineDash([]);
 if($('ellipse').checked){const e=DATA.ellipse;ctx.strokeStyle='#ff6262';ctx.beginPath();ctx.ellipse(e.cx,e.cy,e.rx,e.ry,0,0,Math.PI*2);ctx.stroke()}
 for(const s of visible()){ctx.strokeStyle=s.status==='new'?'#ffb331':'#65e2e6';ctx.beginPath();ctx.arc(s.x,s.y,8,0,2*Math.PI);ctx.stroke();if($('windows').checked){const [l,r,t,b]=s.bounds;ctx.strokeRect(l-.5,t-.5,r-l,b-t)}}
 if($('removed').checked){ctx.setLineDash([3/scale,3/scale]);for(const s of level().spots.filter(s=>s.accepted&&s.selection!=='kept')){ctx.strokeStyle=s.selection==='overlap'?'#b6b6b6':'#ff6262';ctx.beginPath();ctx.arc(s.x,s.y,9,0,2*Math.PI);ctx.stroke()}ctx.setLineDash([])}
 if($('large').checked){ctx.strokeStyle='#ec79dc';for(const s of level().spots.filter(s=>!s.accepted)){const [x,y,w,h]=s.bbox;ctx.strokeRect(x,y,w,h)}}
 if(selection&&point){ctx.strokeStyle='#fff';ctx.lineWidth=1/scale;ctx.strokeRect(point.x-44,point.y-44,88,88)}ctx.restore()
}
function crop(id,im,marks){const ctx=$(id).getContext('2d');ctx.clearRect(0,0,352,352);if(!point)return;ctx.imageSmoothingEnabled=false;const l=point.x-44,t=point.y-44;if(im?.complete&&im.naturalWidth)ctx.drawImage(im,l,t,88,88,0,0,352,352);if(marks)overlays(ctx,-l*4,-t*4,4,false)}
function inspect(){
 if(!point)return;crop('crop',images[displayName()],true);crop('bgcrop',images.background,false);
 let nearest=null,dist=Infinity;for(const s of level().spots){const d=Math.hypot(point.x-s.x,point.y-s.y);if(d<dist){nearest=s;dist=d}}
 let text=`Crop centre x=${point.x.toFixed(1)}, y=${point.y.toFixed(1)} sensor pixels. `;
 if(nearest&&dist<=20){const s=nearest;text+=`${s.id}: ${s.selection}; centroid (${s.x.toFixed(1)}, ${s.y.toFixed(1)}); area ${s.area}, box ${s.bbox[2]}×${s.bbox[3]} pixels. Filtered peak ${s.dog_peak.toFixed(2)}; r=${s.r===null?'undefined':s.r.toFixed(3)}. Raw peak ${s.raw_peak}/4094${s.saturated?' — SATURATED':''}. `;if(s.suppressed_by)text+=`Suppressed by stronger ${s.suppressed_by}.`;if(s.selection==='region')text+='Inside the saved large background ellipse.'}
 else text+='No candidate or oversized-component centre within 20 pixels.';$('detail').textContent=text
}
function draw(){const ctx=$('map').getContext('2d'),im=load(displayName());$('layer').disabled=$('binary').checked;$('binaryNote').hidden=!$('binary').checked;ctx.imageSmoothingEnabled=!$('binary').checked;$('map').style.imageRendering=$('binary').checked?'pixelated':'auto';ctx.clearRect(0,0,2464,2056);if(im?.complete&&im.naturalWidth)ctx.drawImage(im,0,0);overlays(ctx);inspect()}
function plot(id,series,bars=false){
 const W=640,H=340,L=60,R=18,T=18,B=53,edges=DATA.bins,xmax=edges.at(-1),key=$('units').value;
 const maximum=Math.max(1,...series.flatMap(s=>s.q[key])),ymax=Math.ceil(maximum*1.08*10)/10;
 const x=v=>L+v/xmax*(W-L-R),y=v=>H-B-v/ymax*(H-T-B);let svg='';
 for(let i=0;i<=4;i++){const v=ymax*i/4;svg+=`<path d="M${L},${y(v)}H${W-R}" stroke="#e2e7e4"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end" font-size="12">${v.toFixed(key==='counts'?0:1)}</text>`}
 for(let v=0;v<=xmax+1e-8;v+=.2)svg+=`<text x="${x(v)}" y="${H-B+22}" text-anchor="middle" font-size="12">${v.toFixed(1)}</text>`;
 for(const [i,s] of series.entries()){const a=s.q[key];if(bars&&i===0){for(let j=0;j<a.length;j++)svg+=`<rect x="${x(edges[j])}" y="${y(a[j])}" width="${x(edges[j+1])-x(edges[j])}" height="${y(0)-y(a[j])}" fill="${s.color}" opacity=".55"/>`}else{let d=`M${x(edges[0])},${y(a[0])}`;for(let j=0;j<a.length;j++)d+=`L${x(edges[j])},${y(a[j])}L${x(edges[j+1])},${y(a[j])}`;svg+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width||1.6}"/>`}}
 svg+=`<path d="M${x(DATA.r_max)},${T}V${H-B}" stroke="#5c6970" stroke-dasharray="3,4"/><path d="M${L},${T}V${H-B}H${W-R}" stroke="#4a575d" fill="none"/><text x="${W/2}" y="${H-8}" text-anchor="middle" font-size="14">Anisotropy radius r · 1× background subtraction</text><text transform="translate(15,${H/2-15}) rotate(-90)" text-anchor="middle" font-size="13">${key==='counts'?'Spots':'Spots (%)'} per 0.02 bin</text>`;
 $(id).innerHTML=svg
}
function refresh(){
 const v=level(),q=v.distribution,name=v.family.toUpperCase();
 $('count').textContent=`${q.n} retained / ${v.counts.accepted} detected`;
 $('currentTitle').textContent=`${name} k=${v.k} · n=${q.n}`;
 $('histStats').textContent=`Median r=${q.median===null?'undefined':q.median.toFixed(3)} · ${q.below_02} below r=0.2 · ${q.in_02_04} at 0.2≤r<0.4 · ${q.above_model_limit} above model limit ${DATA.r_max.toFixed(3)} (included).`;
 plot('hist',[{q,color:name==='SD'?'#087e8b':'#b36a15'}],true);
 const ks=[8,6,4,3,2];$('compareTitle').textContent=`${name}: sensitivity to threshold`;
 plot('compare',ks.map(k=>({q:family()[k].distribution,color:COLORS[k],width:k===v.k?2.7:1.3})));
 $('legend').innerHTML=ks.map(k=>`<span style="color:${COLORS[k]};margin-right:12px">k=${k} (n=${family()[k].distribution.n})</span>`).join('');
 $('formula').textContent=`${name} k=${v.k}: filtered threshold ${v.threshold.toFixed(3)} intensity units (equivalent SD k=${v.equivalent_sd_k.toFixed(3)}). Display gain does not affect detection. Source: ${DATA.source}.`;
 $('table').innerHTML='<table><tr><th>Model / k</th><th>Cutoff</th><th>Detected</th><th>Region</th><th>Saturation</th><th>Invalid r</th><th>Overlap</th><th>Retained</th><th>Median r</th><th>Oversized</th></tr>'+[DATA.levels,DATA.mad_levels].flatMap(group=>ks.map(k=>{const a=group[k],c=a.cuts;return `<tr class="${a.family===v.family&&a.k===v.k?'active':''}"><td>${a.family.toUpperCase()} ${k}</td><td>${a.threshold.toFixed(2)}</td><td>${a.counts.accepted}</td><td>${c.region}</td><td>${c.saturated}</td><td>${c.invalid}</td><td>${c.overlap}</td><td>${c.kept}</td><td>${a.distribution.median===null?'—':a.distribution.median.toFixed(3)}</td><td>${a.counts.oversized}</td></tr>`})).join('')+'</table>';
 draw()
}
$('scale').textContent=`Here mean=${DATA.mean.toFixed(3)}, SD=${DATA.sd.toFixed(3)}, median=${DATA.median.toFixed(3)}, scaled MAD=${DATA.robust_sd.toFixed(3)}. SD is ${(DATA.sd/DATA.robust_sd).toFixed(2)} times the scaled MAD. Equal k values therefore give different intensity cutoffs; for example MAD k=8 is equivalent to SD k=${DATA.mad_levels['8'].equivalent_sd_k.toFixed(2)}.`;
for(const id of ['family','k','units'])$(id).onchange=refresh;
for(const id of ['binary','marks','removed','large','ellipse','windows'])$(id).onchange=draw;
$('layer').onchange=()=>{load($('layer').value);draw()};$('zoom').onchange=()=>{$('map').style.width=$('zoom').value==='native'?'2464px':'100%'};
$('map').onclick=e=>{const r=$('map').getBoundingClientRect();point={x:Math.max(44,Math.min(2420,(e.clientX-r.left)*2464/r.width)),y:Math.max(44,Math.min(2012,(e.clientY-r.top)*2056/r.height))};draw()};
$('export').onclick=()=>{const rows=level().spots.filter(s=>s.selection==='kept'),fields=['id','x','y','r','dog_peak','raw_peak','area'];const csv=[fields.join(','),...rows.map(s=>fields.map(k=>s[k]).join(','))].join('\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),a=document.createElement('a');a.href=url;a.download=`first_dna_${level().family}_k${level().k}_retained.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
load('mean');load('background');refresh();
</script></html>'''

if __name__ == '__main__':
    build()
