"""Native spot detection, all-five position veto, and one mean-intensity theta/spot.

Offline analysis of the five kept first-round hinge stacks. Run with repo .venv.
Original data, review labels, and acquisition code are never modified.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import itertools
import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
from matplotlib.patches import Circle

from spot_detection_example import display20, png_data

B = Path(__file__).resolve().parent
REPO = B.parents[1]
DATA = REPO / "data/hinges/polarcam/2026-10-09"
OUT = B / "first_hinges_theta"
STAMPS = ("154551", "154730", "154844", "154929", "155041")
MATCH_PX = 6.0
WINDOW_PX = 18  # Matches GUI Spot window=19, which is rounded down to even width.
MODEL_PARAMETERS = {"NA": 1.3, "n": 1.33, "NA_in": .39}
COLORS = ("#2b6f96", "#009d9a", "#ca8634", "#a65692", "#655fc1")


def load_module(name, relative):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def cliques(points, stack_indices, tolerance):
    """One detection per requested stack; EVERY pair must be within tolerance.

    No image registration and no transitive chaining of distant neighbours.
    Return all qualifying tuples, retaining ambiguity rather than picking a match.
    """
    first, *others = stack_indices
    result = []
    for i, p in enumerate(points[first]):
        choices = [np.flatnonzero(np.linalg.norm(points[k] - p, axis=1) <= tolerance)
                   for k in others]
        for indices in itertools.product(*choices):
            nodes = [(first, i), *zip(others, indices)]
            q = np.array([points[k][j] for k, j in nodes])
            if cdist(q, q).max() <= tolerance:
                result.append([(int(k), int(j)) for k, j in nodes])
    return result


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def run():
    OUT.mkdir(exist_ok=True)
    detector, detector_path = load_module("native_detector", "src/polarcam/live/Detection_alg_offline.py")
    channels, channels_path = load_module("native_channels", "src/polarcam/live/intensity_spots.py")
    model, model_path = load_module("hinge_model", "analysis/paper_analysis_20261008/original_scripts/hinge_analysis.py")
    A, Bc, C = model.fourkas_ABC(**MODEL_PARAMETERS)
    stacks, points, rows = [], [], []
    for stack_index, stamp in enumerate(STAMPS):
        path = DATA / f"frame_stack_20261009-{stamp}.npy"
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8-sig"))
        raw = np.load(path, mmap_mode="r")
        n = len(raw) - int(metadata["actual"]["phase_marker_appended"])
        if n != metadata["actual"]["frames"] or n != 150:
            raise ValueError(f"Unexpected frame count: {path}")
        if not metadata["background"]["background_subtracted"]:
            raise ValueError(f"Unexpected subtraction state: {path}")
        origin = tuple(int(metadata["roi"][k]) for k in ("OffsetX", "OffsetY"))
        mean = np.zeros(raw.shape[1:], dtype=np.float64)
        for frame in raw[:n]:
            mean += frame
        mean /= n
        image = detector.intensity_detection_map(mean)
        edge = detector.EDGE_EXCLUDE_PX
        centers = np.array(detector.find_spot_centers_dog(image[edge:-edge, edge:-edge], k_std=8)).reshape(-1, 2) + edge
        sensor_centers = centers + origin
        points.append(sensor_centers)
        h, w = mean.shape
        bounds = []
        for cx, cy in centers:
            x, y = max(0, int(round(cx))-WINDOW_PX//2), max(0, int(round(cy))-WINDOW_PX//2)
            bounds.append((x, min(w, x+WINDOW_PX), y, min(h, y+WINDOW_PX)))
        if any(right-left != WINDOW_PX or bottom-top != WINDOW_PX for left,right,top,bottom in bounds):
            raise ValueError("Unexpected truncated measurement window")
        intensities = channels.channel_means(mean, bounds, origin)
        xy = channels.anisotropy(intensities)
        r = np.linalg.norm(xy, axis=1)
        theta, flags = model.theta_from_r(r, A, Bc, C, return_flags=True)
        nearest = cKDTree(centers).query(centers, k=2)[0][:, 1] if len(centers)>1 else np.full(len(centers), np.inf)
        # Independent algebraic round trip verifies angle units and coefficient use.
        s2 = np.sin(np.radians(theta[flags["valid"]]))**2
        np.testing.assert_allclose(C*s2/(A+Bc*s2), r[flags["valid"]], atol=1e-12, rtol=1e-12)
        stack_rows = []
        for j, (p, bound) in enumerate(zip(sensor_centers, bounds)):
            row = dict(stack=stamp, file=path.name, spot_id=j+1, x_sensor_px=float(p[0]),
                       y_sensor_px=float(p[1]), roi_left=bound[0], roi_right=bound[1],
                       roi_top=bound[2], roi_bottom=bound[3], **dict(zip(("I0", "I90", "I45", "I135"), map(float, intensities[j]))),
                       X=float(xy[j, 0]), Y=float(xy[j, 1]), r=float(r[j]), theta_deg=float(theta[j]),
                       theta_valid=bool(flags["valid"][j]), model_over_range=bool(flags["over_range"][j]),
                       nearest_candidate_px=float(nearest[j]), neighbor_under_24px=bool(nearest[j]<24),
                       recurring_all_five=False, included=False)
            rows.append(row)
            stack_rows.append(row)
        stacks.append(dict(stamp=stamp, time=f"{stamp[:2]}:{stamp[2:4]}:{stamp[4:]}", image=image,
                           centers=centers, rows=stack_rows, origin=list(origin), frames=n,
                           source_bytes=path.stat().st_size, source_mtime_ns=path.stat().st_mtime_ns,
                           source_json_sha256=hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest(),
                           note=metadata.get("conditions", {}).get("condition_note", "")))
        print(f"{stamp}: {len(centers)} detected", flush=True)
        del raw, mean

    matching = cliques(points, tuple(range(5)), MATCH_PX)
    remove = set(node for group in matching for node in group)
    sensitivity = []
    for tolerance in (2, 3, 4, 6, 8, 10):
        groups = cliques(points, tuple(range(5)), tolerance)
        sensitivity.append(dict(tolerance_px=tolerance, matching_tuples=len(groups),
                                excluded_detections=len(set(node for group in groups for node in group))))
    partial = []
    for indices in itertools.combinations(range(5), 4):
        partial.extend(cliques(points, indices, MATCH_PX))
    partial_only = [group for group in partial if not all(node in remove for node in group)]
    for k, stack in enumerate(stacks):
        support = sum((cKDTree(q).query(points[k])[0] <= MATCH_PX).astype(int) for q in points)
        for j, row in enumerate(stack["rows"]):
            row["nearby_detection_stack_count"] = int(support[j])
            row["recurring_all_five"] = (k, j) in remove
            row["included"] = row["theta_valid"] and not row["recurring_all_five"]
        kept = [r["theta_deg"] for r in stack["rows"] if r["included"]]
        stack["summary"] = dict(stack=stack["time"], detected=len(stack["rows"]),
             recurring_removed=sum(r["recurring_all_five"] for r in stack["rows"]),
             retained_candidates=sum(not r["recurring_all_five"] for r in stack["rows"]),
             invalid_theta=sum(not r["theta_valid"] and not r["recurring_all_five"] for r in stack["rows"]),
             included=len(kept), median_theta=float(np.median(kept)) if kept else None,
             crowded_candidates=sum(r["neighbor_under_24px"] and r["included"] for r in stack["rows"]))
    kept = np.array([r["theta_deg"] for r in rows if r["included"]])
    assert len(rows) == sum(s["summary"]["included"] + s["summary"]["invalid_theta"] + s["summary"]["recurring_removed"] for s in stacks)
    assert np.all((kept >= 0) & (kept <= 90))
    summary = dict(stacks=[s["summary"] for s in stacks], detected=len(rows), removed=len(remove),
                   included=len(kept), median_theta=float(np.median(kept)) if len(kept) else None,
                   mean_theta=float(np.mean(kept)) if len(kept) else None,
                   quartiles_deg=np.percentile(kept, [25, 75]).tolist() if len(kept) else [],
                   matching_tuples=matching, four_of_five_tuples=partial_only, sensitivity=sensitivity,
                   matching_tolerance_px=MATCH_PX, matching_rule="One detection per stack, every pair <= tolerance; sensor coordinates without registration",
                   same_background_basis="User-specified first-round group; old metadata lacks a profile identity hash",
                   theta_model="Existing paper-analysis manuscript_annular Fourkas model",
                   model_parameters=MODEL_PARAMETERS, ABC=[A, Bc, C], r_max=model.r_max_of(A, Bc, C),
                   theta_estimator="18x18 aperture; average each channel over all 150 frames, then ratios and inversion; one effective polar angle per candidate",
                   detector="Native GUI intensity, k=8, sigma=1/6, area=10..250, 10px crop; no manual or motion filters",
                   sources={str(p.relative_to(REPO)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), detector_path, channels_path, model_path)},
                   inputs=[{key:s[key] for key in ('stamp','frames','origin','source_bytes','source_mtime_ns','source_json_sha256')} for s in stacks])
    with (OUT / "spots.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k: "" if isinstance(v, float) and not np.isfinite(v) else v for k,v in r.items()} for r in rows])
    (OUT / "results.json").write_text(json.dumps(clean_json(summary), indent=2, allow_nan=False), encoding="utf-8")
    make_figures(stacks, summary, partial_only)
    make_html(stacks, summary)
    print(json.dumps(summary, default=str))


def make_figures(stacks, summary, partial):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f7f4", "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained", gridspec_kw={"width_ratios":[1.1,1]})
    rng = np.random.default_rng(20261011)
    populations = []
    for k, s in enumerate(stacks):
        theta = np.array([r["theta_deg"] for r in s["rows"] if r["included"]])
        populations.append(theta)
        axs[0].scatter(k+rng.uniform(-.22, .22, len(theta)), theta, s=18, color=COLORS[k], alpha=.75, linewidths=0)
        if len(theta):
            axs[0].plot([k-.27, k+.27], [np.median(theta)]*2, color="#182b31", lw=2)
    axs[0].set(xticks=range(5), xticklabels=[f"{s['time']}\nn={s['summary']['included']}" for s in stacks],
               ylabel="Effective optical θ (degrees)", ylim=(0,90), title="One value per retained candidate; bars = medians")
    axs[0].tick_params(axis="x", labelsize=9)
    counts, edges, _ = axs[1].hist(populations, bins=np.arange(0,95,5), stacked=True,
                                  color=COLORS, label=[s['time'] for s in stacks], edgecolor="white", linewidth=.5)
    assert int(counts[-1].sum()) == summary["included"]
    axs[1].set(xlabel="Effective optical θ (degrees)", ylabel="Candidate count per 5° bin", xlim=(0,90),
               title=f"Pooled distribution: {summary['included']} candidates")
    axs[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"First-round hinges · {summary['detected']} detections · {summary['removed']} all-five recurring exclusions", fontsize=13)
    fig.get_layout_engine().set(rect=(0,.08,1,.92))
    fig.text(.5,.025,"Mean-channel reconstruction · annular water model (NA 1.3, n 1.33, inner NA 0.39) · equal weight per candidate", ha="center", fontsize=9)
    fig.savefig(OUT / "theta_distribution.png", dpi=180)
    plt.close(fig)

    norm = Normalize(0, 90)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.2), layout="constrained")
    for ax, s in zip(axs.flat, stacks):
        ax.imshow(display20(s["image"]), cmap="gray", vmin=0, vmax=255)
        for (x,y), row in zip(s['centers'], s['rows']):
            if row['included']:
                ax.add_patch(Circle((x,y), 19, color=plt.cm.viridis(norm(row['theta_deg'])), fill=False, lw=.85))
            elif row['recurring_all_five']:
                ax.plot(x,y,'x',color='#ff5f6a',ms=5)
        ax.set_title(f"{s['time']} · {s['summary']['included']} retained")
        ax.set_xticks([]);ax.set_yticks([])
    axs.flat[-1].axis('off')
    axs.flat[-1].text(.05,.7,"20× mean-image display\nCircle colour = θ\nCircle size is a marker, not the aperture\n\nClose or split detections remain candidates.\nOnly the all-five recurrence rule removes spots.", transform=axs.flat[-1].transAxes, va='top')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'), ax=axs.flat[-1], location='bottom', label='Effective optical θ (degrees)', shrink=.8)
    fig.suptitle('Detected positions and one θ per retained candidate',fontsize=14)
    fig.savefig(OUT/'spot_maps.png',dpi=165);plt.close(fig)

    if partial:
        group=partial[0]
        centre=np.mean([stacks[k]['centers'][j] for k,j in group],axis=0)
        cx,cy=map(lambda v:int(round(v)),centre);half=32
        fig,axs=plt.subplots(1,5,figsize=(13,3.2),layout='constrained')
        for k,(ax,s) in enumerate(zip(axs,stacks)):
            x0,x1,y0,y1=cx-half,cx+half,cy-half,cy+half
            ax.imshow(display20(s['image'][y0:y1,x0:x1]),cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(x0-.5,x1-.5,y1-.5,y0-.5))
            hits=[j for kk,j in group if kk==k]
            for j in hits:
                x,y=s['centers'][j];ax.add_patch(Circle((x,y),9,fill=False,color='#ffbd4a',lw=1))
            if not hits:
                ax.add_patch(Circle((cx,cy),9,fill=False,color='#999999',lw=.8,ls=':'))
            ax.set_title(s['time']+('\nDetected; retained' if hits else '\nNot detected at k=8'),fontsize=10)
            ax.tick_params(labelsize=8)
        fig.suptitle(f"Repeat in four stacks near ({cx}, {cy}) · retained under the all-five rule · 20× mean-image display",fontsize=11)
        fig.savefig(OUT/'four_stack_repeat.png',dpi=170);plt.close(fig)


def make_html(stacks, summary):
    rows_html=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in (s['summary']['stack'],s['summary']['detected'],s['summary']['recurring_removed'],s['summary']['included'],f"{s['summary']['median_theta']:.1f}°"))+'</tr>' for s in stacks)
    partial = ''
    if len(summary['four_of_five_tuples']) == 1:
        centre = np.mean([stacks[k]['centers'][j] for k,j in summary['four_of_five_tuples'][0]], axis=0)
        partial = f'<p>One candidate position is detected in four of the five stacks, near <b>({centre[0]:.0f}, {centre[1]:.0f})</b>. A faint feature is also visible there in the first stack, but it does not pass the same k=8 detector. It stays in this distribution under the strict all-five-detections rule.</p><img src="four_stack_repeat.png" class="figure" alt="The four-stack recurring candidate, with the first stack for comparison">'
    payload=[]
    for s in stacks:
        rr=[]
        for row in s['rows']:
            r=dict(row)
            r['color']=to_hex(plt.cm.viridis(row['theta_deg']/90)) if row['included'] else '#ff5f6a'
            rr.append(r)
        payload.append(dict(time=s['time'],stamp=s['stamp'],rows=rr,image=png_data(display20(s['image']))))
    html=HTML.replace('__DATA__',json.dumps(clean_json(payload),allow_nan=False)).replace('__ROWS__',rows_html)
    html=html.replace('__PARTIAL__',partial).replace('__TOTAL__',str(summary['detected'])).replace('__REMOVED__',str(summary['removed']))
    html=html.replace('__VALID__',str(summary['included'])).replace('__MEDIAN__',f"{summary['median_theta']:.1f}")
    (OUT/'index.html').write_text(html,encoding='utf-8')


HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>First-round hinges · θ distribution</title><style>
*{box-sizing:border-box}body{margin:0;background:#f7f7f4;color:#202a2c;font:16px/1.5 system-ui,sans-serif}main{max-width:1380px;margin:auto;padding:25px}
h1{font-size:29px;margin:6px 0}h2{font-size:21px;margin-top:30px}p{max-width:1050px}a{color:#17686b}.figure{width:100%;height:auto}
table{border-collapse:collapse;max-width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 14px;text-align:left;border-bottom:1px solid #d4dcd9}td{white-space:nowrap}
.scroll{overflow:auto}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:15px;background:#e4ece8;padding:12px;border-radius:7px}
select,button{padding:7px;border:1px solid #899994;border-radius:4px;background:white;font:inherit}#viewport{overflow:auto;max-height:75vh;background:#080a0a;margin-top:12px}
canvas{display:block;cursor:crosshair}#crop{width:320px;height:320px;max-width:100%;background:black}.details{display:flex;flex-wrap:wrap;gap:25px;margin-top:15px}.details p{max-width:720px;margin-top:0}.note{font-size:14px;color:#536663}details{margin-top:20px}#spot-table{max-height:340px;overflow:auto;font-size:14px}#spot-table thead{position:sticky;top:0;background:#f7f7f4}
</style><main><div class="note">9 OCTOBER · FIRST-ROUND HINGES · FIVE KEPT STACKS</div>
<h1>One θ per spot, after checking fixed-position repeats</h1>
<p><b>__TOTAL__ detected · __REMOVED__ excluded by the all-five rule · __VALID__ valid θ values.</b> Pooled median: <b>__MEDIAN__°</b>.</p>
<p>Same native GUI intensity detector as the example: <b>k=8</b>, using all 150 measurement frames in each stack. “Same place” means a detection in every stack with every pair of centres within <b>6 sensor pixels</b>; no image alignment is applied. No five-stack match was found at any tested tolerance from 2 to 10 pixels.</p>
<img class="figure" src="theta_distribution.png" alt="One theta per candidate and pooled theta distribution">
<div class="scroll"><table><thead><tr><th>Stack</th><th>Detected</th><th>Removed: all five</th><th>Valid θ retained</th><th>Median θ</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<h2>A position worth inspecting</h2>__PARTIAL__
<h2>Inspect each spot and its θ</h2><div class="controls">
<label>Stack <select id="stack"></select></label><label><input type="checkbox" id="marks" checked> Marks</label>
<label><input type="checkbox" id="labels"> ID + θ labels</label><label>View <select id="zoom"><option value="fit">Fit width</option><option value="1">1:1 pixels</option><option value="2">2×</option></select></label>
<label>Spot <select id="spot"></select></label></div>
<p class="note">Circle colour runs from purple (0°) through green to yellow (90°). Click a spot to see its exact θ, extraction box and enlarged image. Boxes are the 18 × 18 pixel measurement apertures.</p>
<div id="viewport"><canvas id="view" aria-label="Detected spots coloured by theta"></canvas></div>
<div class="details"><canvas id="crop" width="320" height="320" aria-label="Selected spot enlarged with measurement aperture"></canvas><div><p id="selection"></p><p class="note">20× display gain only; detection and θ use the saved numerical intensities. This view is the 150-frame mean with four polarization channels averaged for display. θ uses the four channels separately.</p></div></div>
<div id="spot-table" class="scroll"></div>
<details><summary>Reconstruction and interpretation</summary>
<p>Channel extraction calls the native <code>intensity_spots.channel_means</code> function with sensor parity and an 18 × 18 pixel aperture (the GUI’s Spot window=19 setting). Average each of I0, I90, I45 and I135 over all 150 measurement frames, then calculate X=(I0−I90)/(I0+I90), Y=(I45−I135)/(I45+I135), and r=√(X²+Y²).</p>
<p>θ uses <code>hinge_analysis.fourkas_ABC</code> and <code>theta_from_r</code> from the existing paper-analysis scripts: annular pupil, NA=1.3, water n=1.33, inner NA=0.39; r_max≈0.920825. This is the paper-analysis model, not the GUI’s separate water lookup table. θ=arcsin√[Ar/(C−Br)]. Invalid model values are excluded rather than forced to 90°.</p>
<p>θ is the effective polar angle from the optical axis (0° axial, 90° in the image plane), not the hinge opening angle or the mean of framewise θ. Motion during the stack can reduce averaged anisotropy. This is a simple uncalibrated reconstruction: no fitted channel gains, recentering, additional background subtraction, or solid-angle correction.</p>
<p>Close candidates and possible split detections remain in this first pass and are flagged in the table. Equal weight is given to each retained detection; these have not been confirmed as unique individual rods. A recurring position is an operational exclusion rule, not proof of a background artefact.</p>
<p>The common background is your acquisition grouping. Old JSON files record subtraction on but do not preserve the selected profile’s hash. The appended phase-marker frame is excluded. Original data, labels and camera code are unchanged.</p></details>
<p><a href="spots.csv">All spot values (CSV)</a> · <a href="results.json">Settings and matching audit</a> · <a href="spot_maps.png">Five spot maps</a> · <a href="../spot_detection_example/index.html">Detector comparison</a></p>
</main><script>
const stacks=__DATA__,$=id=>document.getElementById(id),view=$('view'),ctx=view.getContext('2d'),cc=$('crop').getContext('2d');let current=0,chosen=0,scale=1,images=[];
stacks.forEach((s,i)=>$('stack').add(new Option(s.time,i)));
function init(){const s=stacks[current];$('spot').innerHTML='';s.rows.forEach((r,i)=>$('spot').add(new Option('#'+r.spot_id+' · '+(r.theta_deg===null?'invalid':r.theta_deg.toFixed(1)+'°'),i)));$('spot').value=chosen;
$('spot-table').innerHTML='<table><thead><tr><th>ID</th><th>x</th><th>y</th><th>θ</th><th>Stacks nearby</th><th>Neighbour &lt;24px</th><th>Status</th></tr></thead><tbody>'+s.rows.map(r=>'<tr><td>'+r.spot_id+'</td><td>'+r.x_sensor_px.toFixed(1)+'</td><td>'+r.y_sensor_px.toFixed(1)+'</td><td>'+(r.theta_deg===null?'—':r.theta_deg.toFixed(2)+'°')+'</td><td>'+r.nearby_detection_stack_count+'/5</td><td>'+(r.neighbor_under_24px?'Yes':'No')+'</td><td>'+(r.included?'Retained':r.recurring_all_five?'Recurring exclusion':'Invalid θ')+'</td></tr>').join('')+'</tbody></table>';draw()}
function draw(){const im=images[current];if(!im)return;scale=$('zoom').value==='fit'?Math.min(1,$('viewport').clientWidth/im.width):Number($('zoom').value);view.width=Math.round(im.width*scale);view.height=Math.round(im.height*scale);ctx.imageSmoothingEnabled=scale<1;ctx.drawImage(im,0,0,view.width,view.height);
if($('marks').checked)stacks[current].rows.forEach((r,i)=>{const x=(r.x_sensor_px+.5)*scale,y=(r.y_sensor_px+.5)*scale;ctx.strokeStyle=r.color;ctx.lineWidth=i===chosen?2:1.2;ctx.beginPath();ctx.arc(x,y,Math.max(5,10*scale),0,Math.PI*2);ctx.stroke();if($('labels').checked){ctx.fillStyle=r.color;ctx.font='12px system-ui';ctx.fillText('#'+r.spot_id+' '+(r.theta_deg===null?'invalid':r.theta_deg.toFixed(1)+'°'),x+7,y-5)}});crop()}
function crop(){const s=stacks[current],r=s.rows[chosen];if(!r)return;const x0=Math.round(r.x_sensor_px)-40,y0=Math.round(r.y_sensor_px)-40;cc.fillStyle='black';cc.fillRect(0,0,320,320);cc.imageSmoothingEnabled=false;cc.drawImage(images[current],x0,y0,80,80,0,0,320,320);if($('marks').checked){cc.strokeStyle=r.color;cc.lineWidth=1.5;cc.strokeRect((r.roi_left-x0)*4,(r.roi_top-y0)*4,(r.roi_right-r.roi_left)*4,(r.roi_bottom-r.roi_top)*4)}
$('selection').innerHTML='<b>'+s.time+' · spot #'+r.spot_id+' · θ = '+(r.theta_deg===null?'invalid':r.theta_deg.toFixed(2)+'°')+'</b><br>Centre ('+r.x_sensor_px.toFixed(2)+', '+r.y_sensor_px.toFixed(2)+')<br>X='+r.X.toFixed(4)+', Y='+r.Y.toFixed(4)+', r='+r.r.toFixed(4)+'<br>Nearby detections in '+r.nearby_detection_stack_count+'/5 stacks; '+(r.included?'retained.':'excluded.')+'<br>Nearest candidate: '+r.nearest_candidate_px.toFixed(1)+' pixels'+(r.neighbor_under_24px?' — close/split candidate flag.':'.')}
$('stack').onchange=()=>{current=Number($('stack').value);chosen=0;init()};$('spot').onchange=()=>{chosen=Number($('spot').value);draw()};for(const id of ['zoom','marks','labels'])$(id).onchange=draw;
view.onclick=e=>{const box=view.getBoundingClientRect(),x=(e.clientX-box.left)/scale-.5,y=(e.clientY-box.top)/scale-.5;let best=Infinity;stacks[current].rows.forEach((r,i)=>{const d=Math.hypot(x-r.x_sensor_px,y-r.y_sensor_px);if(d<best){best=d;chosen=i}});$('spot').value=chosen;draw()};window.onresize=()=>{$('zoom').value==='fit'&&draw()};
Promise.all(stacks.map((s,i)=>new Promise(resolve=>{const im=new Image();im.onload=()=>{images[i]=im;resolve()};im.src=s.image}))).then(init);
</script></html>'''

if __name__ == "__main__":
    run()
