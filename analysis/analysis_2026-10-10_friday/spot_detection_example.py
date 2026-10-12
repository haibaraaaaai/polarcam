"""One-stack detector comparison. Offline only; no GUI/camera imports or batch selection.

Run with the repository's .venv Python. Output: spot_detection_example/index.html
and two PNG figures. Detector source hashes and coordinates are embedded in HTML.
"""
from pathlib import Path
import ast
import base64
import hashlib
import importlib.util
import io
import json

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

B = Path(__file__).resolve().parent
REPO = B.parents[1]
SOURCE = REPO / "data/hinges/polarcam/2026-10-09/frame_stack_20261009-154730.npy"
OUT = B / "spot_detection_example"
NATIVE_SOURCE = REPO / "src/polarcam/live/Detection_alg_offline.py"
EXISTING_SOURCE = B / "review_friday.py"
CYAN, AMBER = "#18e7f0", "#ffac43"


def png_data(a):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def display20(a):
    # Same fixed Mono12 display convention as the gallery; never used for detection.
    return np.rint(np.clip(a.astype(float) * 20 / 4095, 0, 1) * 255).astype(np.uint8)


def run():
    OUT.mkdir(exist_ok=True)
    meta = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8-sig"))
    a = np.load(SOURCE, mmap_mode="r")
    n = len(a) - int(meta["actual"]["phase_marker_appended"])
    if n != meta["actual"]["frames"] or n != 150:
        raise ValueError("Unexpected measurement frame count")
    mean = np.zeros(a.shape[1:], dtype=np.float64)
    for frame in a[:n]:
        mean += frame
    mean /= n
    spec = importlib.util.spec_from_file_location("native_detector", NATIVE_SOURCE)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    detection_map = native.intensity_detection_map(mean)
    e = native.EDGE_EXCLUDE_PX
    work = detection_map[e:-e, e:-e]
    # Match GUI wrapper's crop/offset. No manual spot edits or orientation filters.
    detections = {
        str(k): np.asarray(native.find_spot_centers_dog(work, k_std=k)).reshape(-1, 2) + e
        for k in (4, 6, 8, 10)
    }
    pts = detections["8"]
    # Execute the existing function itself, without the old report's imports/run().
    tree = ast.parse(EXISTING_SOURCE.read_text(encoding="utf-8-sig"))
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "detect")
    ns = {"np": np, "ndimage": ndimage, "cKDTree": cKDTree}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(EXISTING_SOURCE), "exec"), ns)
    bounds, old_threshold = ns["detect"](mean)
    old = np.array([(x + size // 2, y + size // 2) for x, y, size in bounds])
    dog = (cv2.GaussianBlur(work, (0, 0), sigmaX=native.DOG_SIGMA_SMALL, borderType=cv2.BORDER_REPLICATE)
           - cv2.GaussianBlur(work, (0, 0), sigmaX=native.DOG_SIGMA_LARGE, borderType=cv2.BORDER_REPLICATE))
    mu, sd = native._finite_stats(dog)
    native_threshold = max(0, mu + 8 * sd)
    native_dist = cKDTree(old).query(pts)[0]
    old_dist = cKDTree(pts).query(old)[0]
    neighbors = cKDTree(pts).query(pts, k=2)[0][:, 1]
    assert all(np.isfinite(x).all() for x in (mean, pts, old))
    h, w = mean.shape
    for centers in [*detections.values(), old]:
        assert ((centers >= 0) & (centers < [w, h])).all()

    plt.rcParams.update({"font.size": 11, "figure.facecolor": "#f7f7f4"})
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.2), constrained_layout=True)
    for ax, centers, color, title in zip(axs, [pts, old], [CYAN, AMBER],
            [f"Native GUI intensity detector: {len(pts)} candidates",
             f"Existing analysis detector: {len(old)} candidates"]):
        ax.imshow(display20(detection_map), cmap="gray", vmin=0, vmax=255)
        for x, y in centers:
            ax.add_patch(Circle((x, y), 19, fill=False, color=color, linewidth=.7))
        ax.set_title(title)
        ax.set_xlabel("Sensor x (pixels)")
        ax.set_ylabel("Sensor y (pixels)")
    fig.suptitle("15:47:30 hinges | same 150-frame mean | four-polarizer cell average | 20× display\n"
                 "Circles mark candidates, not measurement apertures. No movement/angle filtering.", fontsize=12)
    fig.savefig(OUT / "comparison.png", dpi=170)
    plt.close(fig)

    # Deterministic illustrative crops, explicitly not an accuracy/ground-truth sample.
    values = np.array([detection_map[int(round(y)), int(round(x))] for x, y in pts])
    matched = np.flatnonzero((native_dist <= 6) & (neighbors >= 24))
    close = np.flatnonzero(neighbors < 24)
    old_only = np.flatnonzero(old_dist > 6)
    old_values = np.array([detection_map[int(y), int(x)] for x, y in old[old_only]])
    choices = [
        ("Isolated candidate found by both", pts[matched[np.argsort(values[matched])[len(matched)//2]]]),
        ("Close native candidates (<24 px)", pts[close[np.argmin(neighbors[close])]]),
        ("Brighter analysis-only candidate", old[old_only[np.argsort(old_values)[int(.9*(len(old_only)-1))]]]),
        ("Median-brightness analysis-only candidate", old[old_only[np.argsort(old_values)[len(old_only)//2]]]),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(14, 7.6), constrained_layout=True)
    crop_centers = []
    for col, (title, (cx, cy)) in enumerate(choices):
        cx, cy = int(round(cx)), int(round(cy))
        x0, x1, y0, y1 = max(0, cx-44), min(w, cx+44), max(0, cy-44), min(h, cy+44)
        crop_centers.append({"title": title, "x": cx, "y": cy})
        for row, frame in enumerate([a[0], detection_map]):
            ax = axs[row, col]
            ax.imshow(display20(frame[y0:y1, x0:x1]), cmap="gray", vmin=0, vmax=255,
                      interpolation="nearest", extent=(x0-.5, x1-.5, y1-.5, y0-.5))
            for x, y in pts:
                if x0 <= x < x1 and y0 <= y < y1:
                    ax.add_patch(Circle((x, y), 9, fill=False, color=CYAN, linewidth=1.1))
            for x, y in old:
                if x0 <= x < x1 and y0 <= y < y1:
                    ax.plot(x, y, "+", color=AMBER, ms=8, mew=1.1)
            ax.set_title(title if row == 0 else f"Centre ({cx}, {cy})", fontsize=10)
            if col == 0:
                ax.set_ylabel("First raw frame" if row == 0 else "150-frame detection image")
            ax.tick_params(labelsize=8)
    fig.suptitle("Enlarged examples, 20× display | cyan circle: native k=8 | amber +: existing analysis\n"
                 "Illustrative algorithm differences; candidates are not verified individual rods.", fontsize=12)
    fig.savefig(OUT / "examples.png", dpi=170)
    plt.close(fig)

    payload = {
        "source": str(SOURCE.relative_to(REPO)).replace("\\", "/"),
        "frames": n, "phase_marker_excluded": True,
        "background_already_subtracted": meta["background"]["background_subtracted"],
        "native_source": str(NATIVE_SOURCE.relative_to(REPO)).replace("\\", "/"),
        "native_source_sha256": hashlib.sha256(NATIVE_SOURCE.read_bytes()).hexdigest(),
        "existing_source": str(EXISTING_SOURCE.relative_to(REPO)).replace("\\", "/"),
        "existing_source_sha256": hashlib.sha256(EXISTING_SOURCE.read_bytes()).hexdigest(),
        "native_sigma_pixels": [native.DOG_SIGMA_SMALL, native.DOG_SIGMA_LARGE],
        "native_area_pixels": [native.DOG_MIN_AREA, native.DOG_MAX_AREA],
        "native_edge_crop_pixels": e, "native_k8_threshold": native_threshold,
        "existing_threshold": old_threshold,
        "native": {k: v.tolist() for k, v in detections.items()}, "existing": old.tolist(),
        "near_each_other_within6_pixels": int((native_dist <= 6).sum()),
        "native_candidates_neighbor_under24_pixels": int((neighbors < 24).sum()),
        "illustrative_crops": crop_centers,
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(payload)).replace("__MEAN__", png_data(display20(detection_map)))
    html = html.replace("__FIRST__", png_data(display20(a[0])))
    html = html.replace("__NATIVE_N__", str(len(pts))).replace("__OLD_N__", str(len(old)))
    html = html.replace("__COUNTS__", " · ".join(f"k={k}: {len(v)}" for k,v in detections.items()))
    html = html.replace("__THRESH__", f"{native_threshold:.2f}").replace("__OLD_THRESH__", f"{old_threshold:.2f}")
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "native": len(pts), "existing": len(old),
                      "native_threshold": native_threshold, "existing_threshold": old_threshold,
                      "crowded_native": int((neighbors < 24).sum()), "crops": crop_centers}))


TEMPLATE = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Spot finding · 15:47:30</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f7f7f4;color:#202828;font:16px/1.5 system-ui,sans-serif}
main{max-width:1400px;margin:auto;padding:26px}h1{margin:5px 0;font-size:30px}h2{font-size:21px;margin-bottom:8px}
p{max-width:1050px}a{color:#176a70}.eyebrow{color:#526663;font-size:13px;letter-spacing:.08em}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;padding:12px;background:#e5ece8;border-radius:8px}
select,button{padding:7px;border:1px solid #85938f;border-radius:4px;background:white;font:inherit}
input{accent-color:#176a70}#viewport{overflow:auto;max-height:78vh;background:#050505;margin-top:12px;border-radius:6px}
canvas{display:block;cursor:crosshair;max-width:none}#detail{background:#080b0b;border:1px solid #6c7775;width:352px;max-width:100%;height:auto}
.inspect{display:flex;gap:24px;align-items:flex-start;margin-top:18px;flex-wrap:wrap}.inspect p{max-width:700px;margin-top:0}
.summary{font-weight:600}.note{font-size:14px;color:#526663}.legend{padding:8px 0;color:#344745}
table{border-collapse:collapse;width:100%;max-width:1000px;font-size:14px}td,th{text-align:left;border-bottom:1px solid #c9d1ca;padding:9px;vertical-align:top}
.figure{width:100%;height:auto}details{margin-top:18px}code{overflow-wrap:anywhere}#xy{font-variant-numeric:tabular-nums}
</style><main>
<div class="eyebrow">ONE STACK · ALGORITHM DISCUSSION</div><h1>Where does the detector place the spots?</h1>
<p><b>15:47:30 · hinges · wide field.</b> Native GUI intensity detector: <b>__NATIVE_N__ candidates</b>; existing analysis: <b>__OLD_N__</b>.
Both use the same 150 measurement frames here. The appended phase marker is excluded. Saved background subtraction was on; no second background is subtracted.</p>
<p class="note">The screenshot at 15:47:40 shows Intensity, DoG k=8.00 and 57 spots. The current native code reproduces that count; the screenshot alone does not verify every coordinate.</p>
<div class="controls">
<label>Image <select id="frame"><option value="mean">150-frame mean · 4-polarizer average</option><option value="first">First raw frame</option></select></label>
<label>Detector <select id="detector"><option value="native">Native GUI</option><option value="existing">Existing analysis</option><option value="both">Compare both</option></select></label>
<label>Native k <select id="k"><option>4</option><option>6</option><option selected>8</option><option>10</option></select></label>
<label><input id="overlay" type="checkbox" checked> Marks</label><label><input id="numbers" type="checkbox"> IDs</label>
<label>View <select id="zoom"><option value="fit">Fit width</option><option value="1">1:1 pixels</option><option value="2">2×</option></select></label>
<button id="export">Download coordinates</button></div>
<div class="legend">Cyan circle = native GUI · amber + = existing analysis · circle sizes are visibility markers, not extraction windows.</div>
<div id="viewport"><canvas id="view" aria-label="Image with detected spot overlays"></canvas></div>
<p id="status" class="summary"></p>
<div class="inspect"><canvas id="detail" width="352" height="352" aria-label="Enlarged selected region"></canvas><div><p><b>Click anywhere in the image to inspect it.</b><br>
The crop shows 88 × 88 sensor pixels at 4× enlargement. Changing the image keeps the same detections: all candidates were found from the 150-frame mean.</p>
<p id="xy"></p><p class="note">Fixed 20× display: black = 0, white = 204.75 saved intensity units. Bright cores can clip in this display. Detection uses the original numeric mean, before display gain.</p></div></div>
<h2>How the native detector works</h2>
<ol><li>Average the 150 measurement frames to reduce frame-to-frame noise.</li>
<li>Average each 2 × 2 block of four polarizer pixels and repeat it onto the sensor grid, so polarization channels are combined for finding spots.</li>
<li>Subtract a broadly blurred image (σ=6 pixels) from a narrowly blurred image (σ=1 pixel). This difference-of-Gaussians filter emphasizes compact bright features.</li>
<li>Keep pixels above the filtered-image mean + k × standard deviation. Join touching pixels, retain blobs with 10–250 above-threshold pixels, and mark each blob’s centroid. Crop 10 pixels from the sensor border first.</li></ol>
<p>At k=8, the filtered threshold is __THRESH__. Lower k admits more candidates: <b>__COUNTS__</b>. The blob area is the portion above threshold, not the whole physical spot.</p>
<table><tr><th></th><th>Native GUI intensity detector</th><th>Existing analysis detector</th></tr>
<tr><td>Implementation</td><td>Calls <code>Detection_alg_offline.py</code> directly, with the GUI’s 10-pixel crop.</td><td>Executes <code>detect()</code> from <code>review_friday.py</code> unchanged.</td></tr>
<tr><td>Threshold</td><td>Ordinary mean + 8 × standard deviation.</td><td>Median + 8 × 1.4826 × median absolute deviation; threshold __OLD_THRESH__. The number 8 is not equivalent between methods.</td></tr>
<tr><td>Finding centres</td><td>Centroids of connected thresholded blobs; close spots can merge or split.</td><td>Local peaks; removes both peaks if they are within 24 raw pixels. No blob-area filter.</td></tr>
<tr><td>Spatial smoothing</td><td>σ=1 and 6 raw pixels.</td><td>σ=0.8 and 3 cell pixels (1.6 and 6 raw pixels).</td></tr>
<tr><td>Average used here</td><td>All 150 frames, as in the GUI.</td><td>Same 150-frame mean for comparison. The older full report sampled 30 frames.</td></tr></table>
<p><b>Suggested starting point:</b> native intensity detection at k=8, with crowded or suspicious candidates kept visible for review. Inspection can then establish whether lowering k is useful. Neither algorithm proves that a candidate is one nanorod.</p>
<p class="note">No movement, anisotropy, angle, hollowness, or manual spot filters are applied. The GUI’s “Spot window 19” is an integration setting, not the DoG detection width.</p>
<h2>A few enlarged differences</h2><img class="figure" src="examples.png" alt="First-frame and averaged-image crops comparing isolated, close and analysis-only candidates">
<details><summary>Source and reproducibility</summary><p><code>frame_stack_20261009-154730.npy</code> only. Original data and camera code are unchanged. The rest of the kept gallery has not been processed in this example.</p><p>0-based sensor coordinates, x to the right, y down. Detector IDs are local to each method/threshold; they do not identify the same particle between stacks.</p><button id="metadata">Download source metadata and coordinates</button></details>
<p><a href="comparison.png">Full comparison image</a> · <a href="../framestack_gallery.html?review=20261011">Back to kept gallery</a></p>
</main><script>
const data=__DATA__, sources={mean:'__MEAN__',first:'__FIRST__'},images={};
const $=id=>document.getElementById(id),view=$('view'),ctx=view.getContext('2d'),detail=$('detail'),dc=detail.getContext('2d');
let scale=1,focus=[data.illustrative_crops[1].x,data.illustrative_crops[1].y];
function sets(){return [['native',data.native[$('k').value],'#18e7f0'],['existing',data.existing,'#ffac43']].filter(([name])=>$('detector').value==='both'||$('detector').value===name)}
function marks(context,transform,radius){if(!$('overlay').checked)return;for(const [name,points,color]of sets()){context.strokeStyle=color;context.fillStyle=color;context.lineWidth=1.3;points.forEach(([x,y],i)=>{const [px,py]=transform(x,y);context.beginPath();if(name==='native'){context.arc(px,py,radius,0,2*Math.PI)}else{context.moveTo(px-radius,py);context.lineTo(px+radius,py);context.moveTo(px,py-radius);context.lineTo(px,py+radius)}context.stroke();if($('numbers').checked){context.font='12px system-ui';context.fillText((name==='native'?'N':'A')+(i+1),px+radius+2,py-2)}})}}
function redraw(){const im=images[$('frame').value];if(!im)return;scale=$('zoom').value==='fit'?Math.min(1,$('viewport').clientWidth/im.width):Number($('zoom').value);view.width=Math.round(im.width*scale);view.height=Math.round(im.height*scale);ctx.imageSmoothingEnabled=scale<1;ctx.drawImage(im,0,0,view.width,view.height);marks(ctx,(x,y)=>[(x+.5)*scale,(y+.5)*scale],Math.max(5,9*scale));$('status').textContent=sets().map(([name,p])=>(name==='native'?'Native k='+$('k').value:'Existing analysis')+': '+p.length+' candidates').join(' · ');$('k').disabled=$('detector').value==='existing';crop()}
function crop(){const im=images[$('frame').value];if(!im)return;const x0=Math.round(focus[0])-44,y0=Math.round(focus[1])-44;dc.fillStyle='#080b0b';dc.fillRect(0,0,352,352);dc.imageSmoothingEnabled=false;dc.drawImage(im,x0,y0,88,88,0,0,352,352);marks(dc,(x,y)=>[(x-x0+.5)*4,(y-y0+.5)*4],25);$('xy').textContent='Crop centre: x='+Math.round(focus[0])+', y='+Math.round(focus[1])+' sensor pixels.'}
view.onclick=event=>{const r=view.getBoundingClientRect();focus=[(event.clientX-r.left)/scale-.5,(event.clientY-r.top)/scale-.5];crop()};
for(const id of ['frame','detector','k','overlay','numbers','zoom'])$(id).onchange=redraw;window.onresize=()=>{$('zoom').value==='fit'&&redraw()};
function download(text,type,name){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('export').onclick=()=>{let rows=['detector,k,id,x_sensor_px,y_sensor_px'];for(const[name,points]of sets())points.forEach(([x,y],i)=>rows.push([name,name==='native'?$('k').value:'robust8',i+1,x.toFixed(4),y.toFixed(4)].join(',')));download(rows.join('\n'),'text/csv','154730_spot_coordinates.csv')};
$('metadata').onclick=()=>download(JSON.stringify(data,null,2),'application/json','154730_spot_detection.json');
Promise.all(Object.entries(sources).map(([name,src])=>new Promise(resolve=>{const im=new Image();im.onload=()=>{images[name]=im;resolve()};im.src=src}))).then(redraw);
</script></html>'''

if __name__ == "__main__":
    run()
