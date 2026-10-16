"""Seeded examples from two theta bands; read-only crops and time traces.

Run from the repository with .venv/Scripts/python.exe. Source selections are
hinges95_theta.json: p95 detection, minimum-background angle extraction.
"""
from pathlib import Path
import base64
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from polarcam.live.intensity_spots import channel_means, anisotropy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DAY = ROOT/'data/hinges/polarcam/2026-10-12'
SOURCE = HERE/'background_overlap_check/hinges95_theta.json'
OUT = HERE/'background_overlap_check/peak_spot_examples'
BANDS = [(30, 32), (50, 60)]
SEED, SIZE, PER_BAND = 20261014, 88, 3


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def packed(a, dtype):
    return base64.b64encode(np.asarray(a, dtype=dtype).tobytes()).decode('ascii')


def clean(a):
    a = np.asarray(a)
    return np.where(np.isfinite(a), a, None).tolist()


def describe(channels):
    xy = anisotropy(channels)
    mean_channels = channels.mean(axis=0)
    intensity = channels.mean(axis=1)
    return dict(channels=clean(channels), xy=clean(xy),
                mean_channels=mean_channels.tolist(), mean_xy=clean(anisotropy(mean_channels)),
                sd_xy=clean(np.nanstd(xy, axis=0, ddof=1)),
                intensity_mean=float(intensity.mean()),
                intensity_cv=float(intensity.std(ddof=1)/intensity.mean()),
                valid_frames=int(np.isfinite(xy).all(axis=1).sum()))


def build():
    OUT.mkdir(exist_ok=True)
    data = json.loads(SOURCE.read_text(encoding='utf8'))
    assert data['complete'] and len(data['records']) == 16
    bg_cache = {}
    for bg in data['backgrounds']:
        path = DAY/'backgrounds'/bg['minimum_file']
        assert sha(path) == bg['minimum_sha256']
        capture = DAY/'backgrounds'/bg['capture']
        assert signature(capture) == bg['capture_signature']
        bg_cache[bg['minimum_sha256']] = (np.load(path), np.load(capture, mmap_mode='r'), bg)
    manifest = dict(title='Hinge spots in the 30–32° and 50–60° bands',
                    seed=SEED, per_band=PER_BAND, crop_size=SIZE, bands=BANDS,
                    model=data['model'], source_sha256=sha(SOURCE), builder_sha256=sha(Path(__file__)),
                    sampling='Three uniformly sampled without replacement from each stack and band; '
                             'NumPy PCG64 SeedSequence [20261014, stack index, band index]. '
                             'Half-open bands [30,32) and [50,60). No selection by brightness or appearance.',
                    selection=data['selection'], extraction=data['extraction'], detection=data['detection'],
                    reconstruction='Saved minimum-subtracted pixel + recorded minimum BG. At saved zero pixels '
                                   'this is an upper-bound proxy, not recoverable raw data; zero clipping was irreversible.',
                    time_axis='Frame index divided by recorded actual fps, not measured per-frame timestamps.',
                    records=[])
    exported = []
    for rec in data['records']:
        source = DAY/'frames'/rec['name']
        assert signature(source) == rec['source_signature']
        sidecar = source.with_suffix('.json')
        assert sha(sidecar) == rec['sidecar_sha256']
        meta = json.loads(sidecar.read_text(encoding='utf8'))
        assert meta['background']['background_subtracted'] is True
        assert meta['background']['background_profile_sha256'] == rec['background_sha256']
        n = meta['actual']['frames']
        movie = np.load(source, mmap_mode='r')
        assert n == 150 and len(movie) == 151 and meta['actual']['phase_marker_appended']
        bg, bg_movie, bg_meta = bg_cache[rec['background_sha256']]
        raw_info = meta['fetch_frames_output']['raw_peak']
        raw_path = source.parent/raw_info['path']
        assert sha(raw_path) == raw_info['sha256'] == rec['raw_peak_sha256']
        raw_peak = np.load(raw_path, mmap_mode='r')
        out = {k: rec[k] for k in ['index', 'name', 'time', 'note', 'background_group', 'background_sha256',
                                  'source_signature', 'sidecar_sha256', 'raw_peak_sha256', 'extra_check']}
        out.update(fps=meta['actual']['fps'], frames=n, band_counts=[], spots=[])
        for band_index, (lo, hi) in enumerate(BANDS):
            pool = sorted([s for s in rec['spots'] if s['selection'] == 'kept'
                           and s['theta_deg'] is not None and lo <= s['theta_deg'] < hi], key=lambda s: s['component'])
            out['band_counts'].append(len(pool))
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, rec['index'], band_index])))
            indices = sorted(rng.choice(len(pool), min(PER_BAND, len(pool)), replace=False).tolist())
            for i in indices:
                s = pool[i]
                assert not s['saturated'] and not s['inside_ellipse']
                left, right, top, bottom = s['bounds']
                assert right-left == bottom-top == 18
                # An even crop origin preserves the sensor's 2x2 polarization phase.
                x0 = int(np.clip(2*round((s['x']-SIZE/2)/2), 0, movie.shape[2]-SIZE))
                y0 = int(np.clip(2*round((s['y']-SIZE/2)/2), 0, movie.shape[1]-SIZE))
                assert x0 % 2 == y0 % 2 == 0
                local = [left-x0, right-x0, top-y0, bottom-y0]
                assert min(local) >= 0 and max(local) <= SIZE
                pixels = np.array(movie[:n, y0:y0+SIZE, x0:x0+SIZE])
                bg_pixels = bg[y0:y0+SIZE, x0:x0+SIZE]
                bg_capture = np.asarray(bg_movie[:bg_meta['frames'], y0:y0+SIZE, x0:x0+SIZE])
                np.testing.assert_array_equal(bg_capture.min(axis=0), bg_pixels)
                p95 = np.percentile(bg_capture, 95, axis=0, method='linear').astype(np.float32)
                p95_pixels = np.rint(np.maximum(pixels.astype(np.float32)-(p95-bg_pixels), 0)).astype(np.uint16)
                channels = np.array([channel_means(f, [local], origin=(x0, y0))[0] for f in pixels])
                b_channels = channel_means(bg_pixels, [local], origin=(x0, y0))[0]
                saved = describe(channels)
                added = describe(channels+b_channels)
                np.testing.assert_allclose(saved['mean_channels'], s['channels'], rtol=0, atol=1e-12)
                np.testing.assert_allclose(saved['mean_xy'], s['xy'], rtol=0, atol=1e-12)
                np.testing.assert_allclose(np.linalg.norm(saved['mean_xy']), s['r'], rtol=0, atol=1e-12)
                bx, by, bw, bh = s['bbox']
                bx, by = round(bx+.5), round(by+.5)
                peak = max(int(raw_peak[top:bottom, left:right].max()), int(raw_peak[by:by+bh, bx:bx+bw].max()))
                assert peak == s['raw_peak'] and peak < 4094
                ap = pixels[:, local[2]:local[3], local[0]:local[1]]
                row = {k: s[k] for k in ['id', 'component', 'x', 'y', 'area', 'bbox', 'bounds',
                                        'theta_deg', 'r', 'dog_peak', 'raw_peak']}
                row.update(band=band_index, pool_count=len(pool), crop_origin=[x0, y0],
                           minimum=saved, added_back=added,
                           bg_channels=b_channels.tolist(), bg_xy=clean(anisotropy(b_channels)),
                           zero_fraction=float(np.mean(ap == 0)),
                           zero_fraction_first=float(np.mean(ap[0] == 0)),
                           pixels=dict(sum_min=packed(pixels.sum(axis=0), '<u4'),
                                       first_min=packed(pixels[0], '<u2'),
                                       background=packed(bg_pixels, '<u2'),
                                       sum_p95=packed(p95_pixels.sum(axis=0), '<u4')))
                out['spots'].append(row)
        assert len(out['spots']) == 6 and len({s['id'] for s in out['spots']}) == 6
        assert signature(source) == rec['source_signature']
        js = OUT/f"stack_{rec['index']:02}.js"
        js.write_text('window.receiveStack('+json.dumps(out, ensure_ascii=True, allow_nan=False, separators=(',', ':'))+');', encoding='utf8')
        compact = {k: v for k, v in out.items() if k != 'spots'}
        compact['sample_ids'] = [s['id'] for s in out['spots']]
        compact['file'] = js.name
        compact['sha256'] = sha(js)
        manifest['records'].append(compact)
        exported.append({**compact, 'spots': [{k: v for k, v in s.items() if k != 'pixels'} for s in out['spots']]})
        if rec['index'] == 1:
            preview(out, data['model'])
        print(f"{rec['index']}/16: sampled 3/{out['band_counts'][0]} and 3/{out['band_counts'][1]}; original r/theta selection reproduced", flush=True)
    for _, _, b in bg_cache.values():
        assert signature(DAY/'backgrounds'/b['capture']) == b['capture_signature']
    manifest['examples'] = sum(len(r['sample_ids']) for r in manifest['records'])
    (OUT/'selection_and_traces.json').write_text(json.dumps({**manifest, 'records': exported}, ensure_ascii=False, allow_nan=False), encoding='utf8')
    (OUT/'index.html').write_text(HTML.replace('__MANIFEST__', json.dumps(manifest, ensure_ascii=True, allow_nan=False)), encoding='utf8')
    return manifest


def unpack(s, key):
    dtype = '<u4' if key.startswith('sum') else '<u2'
    return np.frombuffer(base64.b64decode(s['pixels'][key]), dtype=dtype).reshape(SIZE, SIZE).astype(float)


def preview(rec, model):
    selected = [next(s for s in rec['spots'] if s['band'] == b) for b in range(2)]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1, 1, 1.2, 1.5]}, layout='constrained')
    for row, s in enumerate(selected):
        corrected = unpack(s, 'sum_min')/rec['frames']; bg = unpack(s, 'background')
        for col, (a, title) in enumerate([(corrected, 'Minimum BG subtracted'), (corrected+bg, 'BG added back (proxy)'), (bg, 'Recorded minimum BG')]):
            ax = axes[row, col]
            ax.imshow(a, cmap='gray', vmin=0, vmax=4095/20, interpolation='nearest')
            l, r, t, b = s['bounds']; x0, y0 = s['crop_origin']
            ax.add_patch(Rectangle((l-x0-.5, t-y0-.5), r-l, b-t, fill=False, color='#51cad5', lw=.8))
            ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{s['id']}\nθ={s['theta_deg']:.2f}° · r={s['r']:.3f}")
        xy = np.array(s['minimum']['xy']); raw = np.array(s['added_back']['xy'])
        ax = axes[row, 3]
        ax.add_patch(Circle((0, 0), model['r_max'], fill=False, ls=':', color='#999'))
        ax.scatter(xy[:, 0], xy[:, 1], s=6, c='#087f8c', alpha=.45, label='Minimum BG')
        ax.scatter(raw[:, 0], raw[:, 1], s=5, c='#bd712f', alpha=.35, label='Added back')
        ax.plot(*s['minimum']['mean_xy'], marker='x', color='#111')
        ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel='X', ylabel='Y', title='Anisotropy, 150 frames', aspect='equal')
        if row == 0: ax.legend(fontsize=7, loc='lower left')
        ax = axes[row, 4]; time = np.arange(rec['frames'])/rec['fps']
        for j, (color, label) in enumerate([('#087f8c', 'X'), ('#8e4aad', 'Y')]):
            ax.plot(time, xy[:, j], color=color, lw=.75, label=label)
            ax.plot(time, raw[:, j], color=color, lw=.65, ls='--', alpha=.6)
        ax.set(xlabel='Time (s)', ylabel='Anisotropy', title='Solid: min BG; dashed: added back')
        ax.legend(fontsize=8)
    fig.suptitle('Stack 1 · one seeded example per angle band · 88 × 88 native sensor pixels at 20× display\n18 × 18 extraction aperture in cyan; full gallery contains 3 per band for each of 16 stacks', fontsize=12)
    fig.savefig(OUT/'preview.png', dpi=150)
    plt.close(fig)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hinge peak spot examples</title><style>
:root{color-scheme:light;font:15px/1.5 system-ui,sans-serif;color:#243438;background:#f4f6f4}*{box-sizing:border-box}body{margin:0 auto;max-width:1560px;padding:25px}h1{font-size:29px;margin:0 0 10px}h2{font-size:21px}h3{margin:0;font-size:18px}p{margin:8px 0}a{color:#087f8c}header{max-width:1120px}.muted{color:#5e6f73;font-size:13px}.controls{position:sticky;top:0;z-index:5;display:flex;gap:12px;align-items:center;flex-wrap:wrap;background:#e8eeeb;padding:12px;border:1px solid #cbd6d1;margin:20px 0 10px;border-radius:8px}select,button{font:inherit;padding:5px;border:1px solid #abbdb7;border-radius:4px;background:white;color:inherit;cursor:pointer}label{font-size:13px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}article{background:white;border:1px solid #d4ded9;padding:15px;border-radius:8px;min-width:0}.band0{border-top:4px solid #087f8c}.band1{border-top:4px solid #b76532}.pixels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;margin-top:12px}.pixels canvas{width:100%;height:auto;image-rendering:pixelated;cursor:zoom-in;background:black}.caption{font-size:12px;line-height:1.3;min-height:36px}.charts{display:grid;grid-template-columns:1fr 1.3fr;gap:8px;align-items:center;margin-top:8px}.charts canvas{width:100%;height:auto}.metrics{display:grid;grid-template-columns:repeat(2,1fr);font-size:13px;gap:4px 12px;margin-top:8px}.tag{background:#edf4ef;border-radius:4px;padding:2px 5px;font-size:12px}details{margin:14px 0}summary{cursor:pointer}#status{padding:8px 0}dialog{max-width:95vw;width:650px;border:1px solid #a3b8b0;border-radius:9px;padding:18px}dialog::backdrop{background:#132124a6}dialog canvas{width:100%;image-rendering:pixelated}#pixelRead{min-height:44px;font:13px monospace}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:4px 8px;border-bottom:1px solid #ddd}.stackhead{grid-column:1/-1;margin:12px 0 0}footer{margin-top:25px}.error{color:#a42b2b}@media(max-width:1050px){.grid{grid-template-columns:1fr}}@media(max-width:620px){body{padding:12px}.pixels{grid-template-columns:repeat(2,1fr)}.charts{grid-template-columns:1fr}.controls{position:static}.metrics{grid-template-columns:1fr}}
</style><header><h1>Pixels behind the two angle bands</h1>
<p>Three randomly selected retained spots per band, per stack: <b>96 examples from the 16 wide-field hinge stacks</b>. The bands are <b>30 ≤ θ &lt; 32°</b> and <b>50 ≤ θ &lt; 60°</b>, using θ from the <b>minimum-background-subtracted mean intensities</b>.</p>
<p>Detection stays on the <b>95th-percentile background-subtracted, 150-frame mean at 2 SD</b>. Oversized components are allowed; the top-right ellipse and raw-saturated spots remain excluded. No selection by brightness or appearance.</p>
<p><b>“BG added back” is an approximation to unsubtracted data.</b> It adds the recorded minimum background to the saved pixels. Where the saved value was clipped to zero, the original raw value is lost; adding back gives an upper bound there. The plots do not undo that clipping.</p>
<p class="muted">Each crop is 88 × 88 sensor pixels. Cyan box: the same 18 × 18 extraction aperture used for θ. All four images use the same display gain, with no per-image contrast stretch. Click an image to enlarge and read pixel values.</p>
</header>
<div class="controls">
<button id="previous">←</button><label>Stack <select id="stack"></select></label><button id="next">→</button>
<label>Band <select id="band"><option value="all">Both bands</option><option value="0">30–32°</option><option value="1">50–60°</option></select></label>
<label>Image <select id="frame"><option value="mean">150-frame mean</option><option value="first">First frame</option></select></label>
<label>Pixels <select id="grid"><option value="native">Native mosaic</option><option value="cell">2 × 2 cell mean</option></select></label>
<label>Gain <select id="gain"><option value="1">1×</option><option value="5">5×</option><option value="20" selected>20×</option><option value="50">50×</option></select></label>
<label><input id="overlay" type="checkbox" checked>Aperture</label>
<label><input id="added" type="checkbox" checked>BG-added-back traces</label>
<label>Trace axes <select id="scale"><option value="local">Local scale</option><option value="full">−1 to 1</option></select></label>
</div><div id="status" role="status">Loading…</div><div id="cards" class="grid"></div>
<details><summary>Method, sampling and available spot counts</summary>
<p>Sampled without replacement with NumPy PCG64, seed [20261014, stack number, band number]. Every eligible spot within a band has equal probability. This is an inspection set, not an estimate of band abundance; the bands have different widths. Stack 16 is the previously included extra angle check.</p>
<p>X = (I₀ − I₉₀)/(I₀ + I₉₀); Y = (I₄₅ − I₁₃₅)/(I₄₅ + I₁₃₅); r = √(X² + Y²). The black × is X/Y from the time-averaged intensities, which defines the reported θ; it is not the arithmetic mean of the per-frame ratios. Dotted circle: model r limit. Grey +: recorded background alone in the same aperture.</p>
<p>Solid teal and purple time traces: minimum-subtracted X and Y; dashed lines: BG added back. All 150 samples are shown with no temporal smoothing. Time uses recorded fps, not individual hardware timestamps. Changing the image from mean to first frame does not change these traces or the original spot selection.</p>
<p>Intensity is the mean of the four channel means in the aperture (saved DN, before display gain). CV and SD describe observed variations, including noise; they do not establish molecular motion. No local-annulus subtraction or camera gain refit is applied. Raw saturation is checked before subtraction over the aperture and the full detected-component bounding rectangle, at ≥4094 DN.</p>
<p>Only display images are optionally averaged into aligned 2 × 2 cells. The quantitative extraction always uses the original polarizer pixels. The 95%-subtracted image is a detection reference and always shows its 150-frame mean.</p><div id="counts"></div></details>
<footer><a href="../hinges95_theta.html">Back to the 16-stack distributions</a> · <a href="selection_and_traces.json" download>Selected coordinates, statistics and all traces (JSON)</a> · <a href="preview.png">Static examples from stack 1</a></footer>
<dialog id="zoom"><div style="display:flex;justify-content:space-between;gap:10px"><b id="zoomTitle"></b><button id="close">Close</button></div><p id="zoomInfo" class="muted"></p><canvas id="zoomCanvas" width="528" height="528"></canvas><div id="pixelRead"></div></dialog>
<script>
const M=__MANIFEST__;
const $=id=>document.getElementById(id), cache={}, pending={}; let token=0, shown=[];
const colors={min:'#087f8c',add:'#c27b32',x:'#087f8c',y:'#8e4aad'};
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(n,d=3)=>Number.isFinite(n)?n.toFixed(d):'undefined';
function unpack(str,bits){const bytes=Uint8Array.from(atob(str),c=>c.charCodeAt(0));const v=new DataView(bytes.buffer), n=bytes.length/(bits/8), out=new Float64Array(n);for(let i=0;i<n;i++)out[i]=bits===32?v.getUint32(i*4,true):v.getUint16(i*2,true);return out;}
window.receiveStack=d=>{for(const s of d.spots){s.arrays={sum_min:unpack(s.pixels.sum_min,32),first_min:unpack(s.pixels.first_min,16),background:unpack(s.pixels.background,16),sum_p95:unpack(s.pixels.sum_p95,32)}; delete s.pixels;}cache[d.index]=d;if(pending[d.index])pending[d.index].resolve(d);};
function loadStack(i){if(cache[i])return Promise.resolve(cache[i]);if(pending[i])return pending[i].promise;let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});pending[i]={promise,resolve,reject};const script=document.createElement('script');script.src=M.records.find(r=>r.index===i).file;script.onerror=()=>reject(new Error('Could not load '+script.src));document.head.appendChild(script);return promise;}
const kinds=['minimum','added_back','background','p95'];
function title(k){return({minimum:$('frame').value==='first'?'First frame · min BG subtracted':'Mean · min BG subtracted',added_back:'BG added back · proxy',background:'Recorded minimum BG',p95:'95% BG subtracted · mean'})[k];}
function pixels(s,r,k){const a=s.arrays,N=M.crop_size,result=new Float64Array(N*N);for(let i=0;i<result.length;i++){const val=$('frame').value==='mean'?a.sum_min[i]/r.frames:a.first_min[i];result[i]=k==='minimum'?val:k==='added_back'?val+a.background[i]:k==='background'?a.background[i]:a.sum_p95[i]/r.frames;}if($('grid').value==='cell'){for(let y=0;y<N;y+=2)for(let x=0;x<N;x+=2){const i=y*N+x,v=(result[i]+result[i+1]+result[i+N]+result[i+N+1])/4;result[i]=result[i+1]=result[i+N]=result[i+N+1]=v;}}return result;}
function drawPixels(c,s,r,k){const N=M.crop_size, a=pixels(s,r,k), tiny=document.createElement('canvas');tiny.width=tiny.height=N;const tc=tiny.getContext('2d'),im=tc.createImageData(N,N),gain=+$('gain').value;for(let i=0;i<a.length;i++){const v=Math.round(Math.min(255,a[i]*gain*255/4095));im.data[i*4]=im.data[i*4+1]=im.data[i*4+2]=v;im.data[i*4+3]=255;}tc.putImageData(im,0,0);const ctx=c.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(tiny,0,0,c.width,c.height);if($('overlay').checked){const scale=c.width/N,[l,rr,t,b]=s.bounds,[x,y]=s.crop_origin;ctx.strokeStyle='#5bd7df';ctx.lineWidth=Math.max(1,scale/4);ctx.strokeRect((l-x)*scale,(t-y)*scale,(rr-l)*scale,(b-t)*scale);}return a;}
function axes(c,xlim,ylim,xlabel,ylabel){const ctx=c.getContext('2d'),p={l:44,r:14,t:20,b:37};ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#fff';ctx.fillRect(0,0,c.width,c.height);const w=c.width-p.l-p.r,h=c.height-p.t-p.b;const X=v=>p.l+(v-xlim[0])/(xlim[1]-xlim[0])*w,Y=v=>p.t+(ylim[1]-v)/(ylim[1]-ylim[0])*h;ctx.font='11px system-ui';ctx.lineWidth=1;for(let j=0;j<5;j++){let x=xlim[0]+(xlim[1]-xlim[0])*j/4,y=ylim[0]+(ylim[1]-ylim[0])*j/4;ctx.strokeStyle='#e6ebe8';ctx.beginPath();ctx.moveTo(X(x),p.t);ctx.lineTo(X(x),p.t+h);ctx.moveTo(p.l,Y(y));ctx.lineTo(p.l+w,Y(y));ctx.stroke();ctx.fillStyle='#57666a';ctx.textAlign='center';ctx.fillText(x.toFixed(1),X(x),p.t+h+15);ctx.textAlign='right';ctx.fillText(y.toFixed(2),p.l-5,Y(y)+4);}ctx.fillStyle='#243438';ctx.textAlign='center';ctx.fillText(xlabel,p.l+w/2,c.height-3);ctx.save();ctx.translate(11,p.t+h/2);ctx.rotate(-Math.PI/2);ctx.fillText(ylabel,0,0);ctx.restore();return{ctx,X,Y};}
function marker(ctx,X,Y,xy,color,type){if(!xy.every(Number.isFinite))return;const x=X(xy[0]),y=Y(xy[1]);ctx.strokeStyle=color;ctx.lineWidth=1.7;ctx.beginPath();if(type==='plus'){ctx.moveTo(x-4,y);ctx.lineTo(x+4,y);ctx.moveTo(x,y-4);ctx.lineTo(x,y+4);}else{ctx.moveTo(x-4,y-4);ctx.lineTo(x+4,y+4);ctx.moveTo(x-4,y+4);ctx.lineTo(x+4,y-4);}ctx.stroke();}
function drawXY(c,s){const {ctx,X,Y}=axes(c,[-1,1],[-1,1],'Anisotropy X','Anisotropy Y');ctx.strokeStyle='#8c9994';ctx.setLineDash([3,4]);ctx.beginPath();ctx.arc(X(0),Y(0),X(M.model.r_max)-X(0),0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);const sets=$('added').checked?[[s.added_back,colors.add],[s.minimum,colors.min]]:[[s.minimum,colors.min]];for(const [d,col]of sets){ctx.fillStyle=col;ctx.globalAlpha=.4;for(const xy of d.xy)if(xy.every(Number.isFinite)){ctx.beginPath();ctx.arc(X(xy[0]),Y(xy[1]),2,0,2*Math.PI);ctx.fill();}}ctx.globalAlpha=1;marker(ctx,X,Y,s.minimum.mean_xy,'#111','cross');if($('added').checked)marker(ctx,X,Y,s.added_back.mean_xy,colors.add,'cross');marker(ctx,X,Y,s.bg_xy,'#7b8582','plus');ctx.fillStyle='#243438';ctx.textAlign='left';ctx.fillText('150 frames · teal: min BG · orange: added back',44,12);}
function drawTime(c,s,r){const sets=$('added').checked?[s.minimum,s.added_back]:[s.minimum];let range=[-1,1];if($('scale').value==='local'){const vals=sets.flatMap(d=>d.xy.flat()).filter(Number.isFinite),lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max(.015,(hi-lo)*.12);range=[Math.max(-1,lo-pad),Math.min(1,hi+pad)];}const {ctx,X,Y}=axes(c,[0,(r.frames-1)/r.fps],range,'Time (s)','X / Y');for(let k=0;k<sets.length;k++)for(let j=0;j<2;j++){ctx.strokeStyle=j?colors.y:colors.x;ctx.lineWidth=1;ctx.globalAlpha=k?.65:1;ctx.setLineDash(k?[4,3]:[]);ctx.beginPath();let connected=false;sets[k].xy.forEach((xy,i)=>{if(Number.isFinite(xy[j])){if(connected)ctx.lineTo(X(i/r.fps),Y(xy[j]));else ctx.moveTo(X(i/r.fps),Y(xy[j]));connected=true;}else connected=false;});ctx.stroke();}ctx.setLineDash([]);ctx.globalAlpha=1;ctx.fillStyle='#243438';ctx.textAlign='left';ctx.fillText('X teal · Y purple; dashed = BG added back',44,12);}
function card(s,r){return `<article class="band${s.band}"><h3>${escape(s.id)} · θ ${num(s.theta_deg,2)}° · r ${num(s.r)}</h3><p class="muted">${s.band===0?'30–32°':'50–60°'} band · x=${num(s.x,1)}, y=${num(s.y,1)} · ${s.area} component pixels ${s.area>250?'<span class="tag">oversized, retained</span>':''}</p><div class="pixels">${kinds.map(k=>`<div><div class="caption">${title(k)}</div><canvas width="176" height="176" data-pixel="${s.id}|${k}" aria-label="${title(k)} for ${s.id}"></canvas></div>`).join('')}</div><div class="charts"><canvas width="330" height="329" data-xy="${s.id}" aria-label="Anisotropy scatter"></canvas><canvas width="420" height="260" data-time="${s.id}" aria-label="Anisotropy versus time"></canvas></div><div class="metrics"><span>Mean intensity: <b>${num(s.minimum.intensity_mean,1)} DN</b> · CV ${num(100*s.minimum.intensity_cv,1)}%</span><span>SD(X), SD(Y): ${num(s.minimum.sd_xy[0])}, ${num(s.minimum.sd_xy[1])}</span><span>Saved zero pixels: ${num(100*s.zero_fraction,1)}% of aperture × frames</span><span>Raw peak: ${s.raw_peak} / 4094 DN saturation cutoff</span><span>Min-BG mean X/Y: ${num(s.minimum.mean_xy[0])}, ${num(s.minimum.mean_xy[1])}</span><span>Added-back mean X/Y: ${num(s.added_back.mean_xy[0])}, ${num(s.added_back.mean_xy[1])}</span></div><details><summary>Channel means and background contribution</summary><table><tr><th>Mean DN in aperture</th><th>0°</th><th>90°</th><th>45°</th><th>135°</th></tr>${[['Minimum BG subtracted',s.minimum.mean_channels],['Recorded minimum BG',s.bg_channels],['BG added back (proxy)',s.added_back.mean_channels]].map(([label,vals])=>`<tr><td>${label}</td>${vals.map(v=>`<td>${num(v,2)}</td>`).join('')}</tr>`).join('')}</table><p class="muted">Sampled 3 of ${s.pool_count} eligible spots in this stack/band. Background-only X/Y: ${num(s.bg_xy[0])}, ${num(s.bg_xy[1])}. ${s.minimum.valid_frames}/150 defined per-frame X/Y pairs. Zero fraction includes genuine zero values as well as clipping; it marks where exact raw reconstruction is unavailable.</p></details></article>`;}
function paint(){for(const c of document.querySelectorAll('[data-pixel]')){const [id,k]=c.dataset.pixel.split('|'),v=shown.find(o=>o.s.id===id);drawPixels(c,v.s,v.r,k);c.onclick=()=>zoom(v.s,v.r,k);}for(const c of document.querySelectorAll('[data-xy]')){const v=shown.find(o=>o.s.id===c.dataset.xy);drawXY(c,v.s);}for(const c of document.querySelectorAll('[data-time]')){const v=shown.find(o=>o.s.id===c.dataset.time);drawTime(c,v.s,v.r);}}
async function render(){const ticket=++token;const ids=$('stack').value==='all'?M.records.map(r=>r.index):[+$('stack').value];$('status').textContent='Loading selected stack pixels…';try{const records=await Promise.all(ids.map(loadStack));if(ticket!==token)return;shown=[];let content='';for(const r of records){const ordered=[0,1,2].flatMap(i=>[r.spots.filter(s=>s.band===0)[i],r.spots.filter(s=>s.band===1)[i]]).filter(Boolean);const chosen=ordered.filter(s=>$('band').value==='all'||s.band===+$('band').value);content+=`<section class="stackhead"><h2>Stack ${r.index} · ${escape(r.time)} · BG ${r.background_group}${r.extra_check?' · extra angle check':''}</h2><p>${escape(r.note)}</p><p class="muted">${escape(r.name)} · 3 of ${r.band_counts[0]} in 30–32°; 3 of ${r.band_counts[1]} in 50–60° · ${num(r.fps,2)} fps</p></section>`;for(const s of chosen){content+=card(s,r);shown.push({s,r});}}$('cards').innerHTML=content;paint();$('status').textContent=`Showing ${shown.length} sampled spots. Fixed ${$('gain').value}× display: black 0, white ${num(4095/+$('gain').value,2)} DN. Image mode: ${$('grid').value==='native'?'native polarization mosaic':'aligned 2×2 cell mean'}; statistics unchanged.`;}catch(e){$('status').textContent=e.message;$('status').className='error';throw e;}}
function zoom(s,r,k){$('zoomTitle').textContent=`${s.id} · ${title(k)}`;$('zoomInfo').textContent=`88 × 88 sensor pixels; origin x=${s.crop_origin[0]}, y=${s.crop_origin[1]}. Same display gain and aperture as the gallery. Hover to read values.`;const c=$('zoomCanvas'),a=drawPixels(c,s,r,k),N=M.crop_size;$('pixelRead').textContent='Move over the image to inspect pixel intensity.';c.onmousemove=e=>{const rect=c.getBoundingClientRect(),x=Math.min(N-1,Math.max(0,Math.floor((e.clientX-rect.left)/rect.width*N))),y=Math.min(N-1,Math.max(0,Math.floor((e.clientY-rect.top)/rect.height*N))),sx=x+s.crop_origin[0],sy=y+s.crop_origin[1],pol=sy%2?(sx%2?'0°':'135°'):(sx%2?'45°':'90°');$('pixelRead').textContent=`Sensor x=${sx}, y=${sy} · ${$('grid').value==='cell'?'2×2 cell mean':pol+' polarizer pixel'} · ${num(a[y*N+x],3)} DN (before display gain)`;};$('zoom').showModal();}
$('stack').innerHTML=M.records.map(r=>`<option value="${r.index}">${r.index} · ${r.time} · BG ${r.background_group}</option>`).join('')+'<option value="all">All 16 stacks</option>';
$('counts').innerHTML='<table><tr><th>Stack</th><th>30–32° eligible</th><th>50–60° eligible</th><th>Examples</th></tr>'+M.records.map(r=>`<tr><td>${r.index} · ${r.time}</td><td>${r.band_counts[0]}</td><td>${r.band_counts[1]}</td><td>3 + 3</td></tr>`).join('')+'</table>';
for(const id of ['stack','band','frame','grid','gain','overlay','added','scale'])$(id).onchange=render;
$('previous').onclick=()=>{$('stack').value=String(Math.max(1,(+$('stack').value||2)-1));render();};
$('next').onclick=()=>{$('stack').value=String(Math.min(16,(+$('stack').value||0)+1));render();};
$('close').onclick=()=>$('zoom').close();
render();
</script></html>'''


if __name__ == '__main__':
    result = build()
    print(f"Complete: {result['examples']} examples; {OUT/'index.html'}")
