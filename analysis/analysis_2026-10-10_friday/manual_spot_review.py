"""Permissive candidates and an autosaving, local manual-review viewer.

Build: .venv/Scripts/python.exe analysis/analysis_2026-10-10_friday/manual_spot_review.py
Serve: append --serve (existing candidates and decisions are preserved).
Only explicit 'keep' decisions will belong to the later manual selection.
"""
from pathlib import Path
import argparse
import copy
from datetime import datetime, timezone
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import itertools
import json
import math
import os
import secrets
import threading
from urllib.parse import urlsplit

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

B = Path(__file__).resolve().parent
REPO = B.parents[1]
SOURCE = REPO/'data/hinges/polarcam/2026-10-09/frame_stack_20261009-154551.npy'
OUT = B/'manual_spot_review_154551'
PORT = 8766
SCHEMA = 'polarcam-manual-spots-v1'


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f:
        f.write(encode(value)); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def build():
    OUT.mkdir(exist_ok=True)
    manifest_path = OUT/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        # Re-rendering must not regenerate IDs or overwrite any manual decisions.
        if manifest['source_size'] != SOURCE.stat().st_size or manifest['source_mtime_ns'] != SOURCE.stat().st_mtime_ns:
            raise RuntimeError('Source changed. Preserve this review and create a new review directory.')
    else:
        before = SOURCE.stat(); meta_bytes = SOURCE.with_suffix('.json').read_bytes()
        meta = json.loads(meta_bytes.decode('utf-8-sig'))
        raw = np.load(SOURCE, mmap_mode='r', allow_pickle=False)
        assert raw.shape == (151, 2056, 2464) and meta['actual']['frames'] == 150
        assert meta['actual']['phase_marker_appended'] and meta['background']['background_subtracted']
        assert meta['roi']['OffsetX'] == meta['roi']['OffsetY'] == 0
        block = []
        for start in (0, 50, 100):
            cell = np.zeros((1028, 1232), dtype=np.float64)
            for frame in raw[start:start+50]:
                cell += frame.reshape(1028, 2, 1232, 2).mean(axis=(1, 3))
            block.append(cell/50)
        mean = sum(block)/3
        first = raw[0].reshape(1028, 2, 1232, 2).mean(axis=(1, 3))
        views = [('mean', mean, '150-frame mean'), ('first', first, 'First frame'),
                 ('early', block[0], 'Frames 1–50 mean'), ('middle', block[1], 'Frames 51–100 mean'),
                 ('late', block[2], 'Frames 101–150 mean')]
        for key, array, _ in views:
            np.asarray(array, dtype='<f4').tofile(OUT/f'{key}.f32')
        dog = ndimage.gaussian_filter(mean, 1.5, mode='nearest') - ndimage.gaussian_filter(mean, 4., mode='nearest')
        median = float(np.median(dog)); scale = float(1.4826*np.median(np.abs(dog-median)))
        threshold = median + 3*scale
        maxima = (dog == ndimage.maximum_filter(dog, size=9, mode='nearest')) & (dog > threshold)
        # Keep candidates near the border as well; edge context is visibly clipped.
        labels, n = ndimage.label(maxima)
        loose = []
        for sl in ndimage.find_objects(labels):
            y0, x0 = sl[0].start, sl[1].start
            yy, xx = np.unravel_index(np.argmax(dog[sl]), dog[sl].shape)
            loose.append((2*(x0+xx)+.5, 2*(y0+yy)+.5))
        cache = B/'well_comparison/stack_154551.npz'
        with np.load(cache, allow_pickle=False) as prior:
            native = prior['native_points'][:, ::-1].copy()
            supplied = prior['supplied_points'][:, ::-1].copy()
        # Preserve earlier candidates even if the permissive scale has no peak there.
        # Only spatial duplicate suppression; there is no brightness/area/local veto.
        points = []; grid = {}; references = []
        def add(x, y, origin):
            gx, gy = int(x//4), int(y//4)
            candidates = itertools.chain.from_iterable(grid.get((gx+dx, gy+dy), []) for dx in (-1,0,1) for dy in (-1,0,1))
            near = [(math.hypot(points[j]['x']-x, points[j]['y']-y), j) for j in candidates]
            best = min(near, default=(math.inf, -1))
            if best[0] <= 4:
                p = points[best[1]]
                if origin not in p['origins']: p['origins'].append(origin)
                return
            iy, ix = min(1027,max(0,round((y-.5)/2))), min(1231,max(0,round((x-.5)/2)))
            score = float((dog[max(0,iy-1):iy+2,max(0,ix-1):ix+2].max()-median)/scale)
            points.append(dict(x=float(x), y=float(y), origins=[origin], score=score,
                               intensity=float(mean[iy,ix]), edge=bool(min(x,y,2463-x,2055-y)<48)))
            grid.setdefault((gx,gy),[]).append(len(points)-1)
        for origin, coords in [('native k=8',native), ('supplied detector',supplied), ('loose DoG',loose)]:
            for x,y in coords: add(x,y,origin)
        points.sort(key=lambda p:(p['y'],p['x']))
        for j,p in enumerate(points): p['id']=f'S{j+1:04d}'
        tree=cKDTree([(p['x'],p['y']) for p in points])
        for origin,coords in [('native',native),('supplied',supplied)]:
            distances=tree.query(coords)[0]
            assert np.all(distances<=4+1e-9)
            references.append(dict(detector=origin,count=len(coords),covered_within4px=int((distances<=4+1e-9).sum())))
        method=dict(name='Custom permissive local maxima + union with earlier candidates',
                    image='150-frame mean, nonoverlapping 2×2 polarizer-cell intensity mean',
                    dog_sigma_sensor_px=[3,8], threshold='Global DoG median + 3 × (1.4826 MAD)',
                    dog_median=median, dog_mad_scale=scale, threshold_value=threshold,
                    maxima_neighbourhood_cells=9, merge_distance_sensor_px=4,
                    rejection='None by component size, local contrast, recurrence, theta or polarization; no candidate-count cap',
                    near_edges='Included, with visibly clipped context', prior_coverage=references)
        identity=hashlib.sha256(encode(dict(source=SOURCE.name,size=before.st_size,mtime=before.st_mtime_ns,method=method,points=points))).hexdigest()
        manifest=dict(schema=SCHEMA,dataset_id=identity,source=str(SOURCE.relative_to(REPO)).replace('\\','/'),
                      source_size=before.st_size,source_mtime_ns=before.st_mtime_ns,
                      metadata_sha256=hashlib.sha256(meta_bytes).hexdigest(),
                      source_code_sha256_at_build=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      frames=150,phase_marker_excluded=True,saved_background_subtracted=True,
                      width=2464,height=2056,cell_width=1232,cell_height=1028,
                      views=[dict(key=k,file=f'{k}.f32',label=label) for k,_,label in views],
                      method=method,loose_peaks=len(loose),candidates=points,created_utc=utc())
        assert SOURCE.stat().st_size==before.st_size and SOURCE.stat().st_mtime_ns==before.st_mtime_ns
        assert SOURCE.with_suffix('.json').read_bytes()==meta_bytes
        atomic_json(manifest_path,manifest)
    (OUT/'index.html').write_text(TEMPLATE,encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),candidates=len(manifest['candidates']),loose=manifest['loose_peaks'],
                          threshold=manifest['method']['threshold_value'],coverage=manifest['method']['prior_coverage'])),flush=True)
    return manifest


class ReviewStore:
    def __init__(self, root, manifest):
        self.root=root;self.manifest=manifest;self.path=root/'manual_review.json';self.lock=threading.RLock()
        self.known={p['id']:p for p in manifest['candidates']}
        if self.path.exists():
            self.state=json.loads(self.path.read_text(encoding='utf-8'))
            if self.state['dataset_id']!=manifest['dataset_id']:raise ValueError('Review belongs to different candidates')
        else:
            self.state=dict(schema=SCHEMA,dataset_id=manifest['dataset_id'],source=manifest['source'],
                            revision=0,created_utc=utc(),updated_utc=utc(),decisions={},added=[])
            atomic_json(self.path,self.state)
    def get(self):
        with self.lock:return copy.deepcopy(self.state)
    def export(self):
        with self.lock:
            spots=[]
            for p in [*self.manifest['candidates'],*self.state['added']]:
                d=self.state['decisions'].get(p['id'],{})
                spots.append(dict(**p,status=d.get('status','unreviewed'),note=d.get('note','')))
            return dict(**self.state,method=self.manifest['method'],spots=spots,
                        selection_rule='Use status == keep only; unreviewed is not accepted',
                        counts={k:sum(p['status']==k for p in spots) for k in ('keep','remove','unreviewed')})
    def update(self,req):
        with self.lock:
            if req.get('revision')!=self.state['revision']:raise RuntimeError('Review changed in another tab; reloaded latest decisions. Please retry.')
            draft=copy.deepcopy(self.state);op=req.get('op')
            all_ids=set(self.known)|{p['id'] for p in draft['added']}
            if op=='label':
                sid=req.get('id');status=req.get('status');note=req.get('note','')
                if sid not in all_ids or status not in ('keep','remove','unreviewed') or not isinstance(note,str) or len(note)>2000:
                    raise ValueError('Invalid decision')
                draft['decisions'][sid]=dict(status=status,note=note,updated_utc=utc())
            elif op=='add':
                x=float(req['x']);y=float(req['y'])
                if not(math.isfinite(x) and math.isfinite(y) and 0<=x<2464 and 0<=y<2056):raise ValueError('Outside image')
                p=dict(id='M'+secrets.token_hex(5),x=round(x,3),y=round(y,3),origins=['manual addition'],
                       score=None,intensity=None,edge=bool(min(x,y,2463-x,2055-y)<48))
                draft['added'].append(p)
            elif op=='import':
                incoming=req.get('data',{})
                if incoming.get('schema')!=SCHEMA or incoming.get('dataset_id')!=draft['dataset_id']:raise ValueError('Backup belongs to another stack or candidate list')
                extras=incoming.get('added',[])
                if not isinstance(extras,list) or len(extras)>10000:raise ValueError('Invalid added candidates')
                existing={p['id']:p for p in draft['added']}
                for p in extras:
                    sid=p.get('id');x=p.get('x');y=p.get('y')
                    if not isinstance(sid,str) or not sid.startswith('M') or len(sid)>60 or not isinstance(x,(int,float)) or not isinstance(y,(int,float)) or not(0<=x<2464 and 0<=y<2056):raise ValueError('Invalid manual coordinate')
                    if sid in existing and (existing[sid]['x']!=x or existing[sid]['y']!=y):raise ValueError('Conflicting manual coordinate')
                    if sid not in existing:
                        existing[sid]=dict(id=sid,x=x,y=y,origins=['manual addition'],score=None,intensity=None,edge=bool(min(x,y,2463-x,2055-y)<48))
                draft['added']=list(existing.values());all_ids=set(self.known)|set(existing)
                incoming_decisions=incoming.get('decisions',{})
                if not isinstance(incoming_decisions,dict):raise ValueError('Invalid backup decisions')
                for sid,d in incoming_decisions.items():
                    if sid not in all_ids or d.get('status') not in ('keep','remove','unreviewed') or not isinstance(d.get('note',''),str) or len(d.get('note',''))>2000:raise ValueError('Invalid backup decision')
                    draft['decisions'][sid]=dict(status=d['status'],note=d.get('note',''),updated_utc=utc())
            else:raise ValueError('Unknown action')
            draft['revision']+=1;draft['updated_utc']=utc()
            atomic_json(self.path,draft);self.state=draft
            return copy.deepcopy(draft)


def serve(manifest,port):
    store=ReviewStore(OUT,manifest);token=secrets.token_hex(24)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(OUT),**kw)
        def log_message(self,fmt,*args):
            if 'POST' in str(args):return
            super().log_message(fmt,*args)
        def send_json(self,value,status=200):
            payload=encode(value);self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            path=urlsplit(self.path).path
            if self.headers.get('Host') not in (f'127.0.0.1:{port}',f'localhost:{port}'):
                self.send_error(403);return
            if path=='/api/review':self.send_json(dict(state=store.get(),token=token));return
            if path=='/api/export':self.send_json(store.export());return
            allowed={'/','/index.html','/manifest.json',*[f'/{v["file"]}' for v in manifest['views']]}
            if path not in allowed:self.send_error(404);return
            return super().do_GET()
        def do_POST(self):
            if self.headers.get('Host') not in (f'127.0.0.1:{port}',f'localhost:{port}') or self.headers.get('X-Review-Token')!=token:
                self.send_error(403);return
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
                self.send_error(403);return
            if urlsplit(self.path).path!='/api/review':self.send_error(404);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=5_000_000:raise ValueError('Invalid request size')
                req=json.loads(self.rfile.read(size))
                self.send_json(dict(state=store.update(req)))
            except RuntimeError as e:self.send_json(dict(error=str(e),state=store.get()),409)
            except (ValueError,TypeError,KeyError) as e:self.send_json(dict(error=str(e)),400)
            except OSError:self.send_json(dict(error='Disk save failed; decision was not committed. Export a backup if needed.'),500)
    http=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Manual review: http://127.0.0.1:{port}/\nDecisions: {store.path}',flush=True)
    http.serve_forever()


TEMPLATE=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Manual spots · hinges 15:45:51</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f6;color:#24323b;font:15px/1.45 system-ui,sans-serif}main{max-width:1530px;margin:auto;padding:20px 24px 45px}h1{font-size:26px;margin:4px 0}h2{font-size:18px;margin:4px 0 10px}p{margin:8px 0}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:#5d6e79}.subtle,small{color:#546775}.topline,.toolbar,.actions,.nav,.counts{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.topline{justify-content:space-between}.counts{margin:12px 0;font-variant-numeric:tabular-nums}.pill{padding:5px 11px;border:1px solid #cbd6dd;border-radius:20px;background:white}.keeptext{color:#11613f}.removetext{color:#a73435}.pendingtext{color:#685519}
button,select,input{font:inherit}button,select,input[type=number],input[type=text]{padding:7px 10px;border:1px solid #adbcc7;border-radius:5px;background:white;color:inherit}button{cursor:pointer}button:hover:not(:disabled){background:#e6edf2}button:disabled{opacity:.45;cursor:default}button:focus-visible,input:focus-visible,select:focus-visible,canvas:focus-visible{outline:3px solid #347fc7;outline-offset:2px}.toolbar{padding:10px 12px;background:#e6edf1;border-radius:6px;margin-bottom:13px}.toolbar label{display:flex;gap:6px;align-items:center}.toolbar input[type=number]{width:76px}.workspace{display:grid;grid-template-columns:minmax(440px,1.15fr) minmax(380px,1fr);gap:20px;align-items:start}.panel{background:white;border:1px solid #d1dce2;padding:14px;border-radius:7px}.imagewrap{background:#030608;position:relative;overflow:auto;max-height:680px}.imagewrap canvas{display:block;cursor:crosshair;touch-action:manipulation}#overview{width:100%;height:auto}#detail{width:100%;height:auto;display:block;background:black;cursor:crosshair}.detailwrap{max-width:480px;margin:auto}#title{font-size:18px;font-weight:650}#position,#meta{font-size:13px;color:#536570}#note{width:100%;margin:8px 0}.actions{margin:12px 0}.actions button{font-size:17px;min-width:106px}.actions .keep{background:#127c51;color:white;border-color:#127c51}.actions .keep:hover{background:#09663f}.actions .remove{background:#a93e43;color:white;border-color:#a93e43}.actions .remove:hover{background:#902d33}.nav{justify-content:space-between}.nav input{width:88px}.status{font-weight:600;color:#41616b;font-size:13px}.status.error{color:#a12525}#message{min-height:22px;color:#934d15;font-size:13px}.kbd{font-size:13px;color:#50616e;line-height:1.65}.legend{font-size:12px;color:#455f6e;margin:8px 0}.legend span{margin-right:12px}.swatch{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}.film{display:grid;grid-template-columns:repeat(8,1fr);gap:8px;margin-top:10px}.thumb{padding:3px;background:white;border:1px solid #c7d2dc;min-width:0}.thumb.selected{outline:3px solid #1677bc}.thumb canvas{width:100%;display:block;background:black}.thumb .caption{font-size:11px;display:block;text-align:center;padding-top:3px}.filmnav{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-top:20px}details{margin-top:20px;border-top:1px solid #cbd6dd;padding-top:12px}summary{cursor:pointer;font-weight:650}code{font-size:12px;overflow-wrap:anywhere}#loading{padding:22px;background:#e6edf1}#app[hidden]{display:none}#addmode:checked+span{color:#a34205;font-weight:bold}a{color:#176b97}.empty{min-height:80px;padding:20px;color:#50616e}@media(max-width:1000px){.workspace{grid-template-columns:1fr 1fr}.panel{padding:10px}.film{grid-template-columns:repeat(6,1fr)}}@media(max-width:780px){main{padding:15px 12px}.workspace{grid-template-columns:1fr}.film{grid-template-columns:repeat(4,1fr)}.imagewrap{max-height:380px}h1{font-size:22px}}@media(prefers-reduced-motion:no-preference){button{transition:background .1s}}
</style></head><body><main>
<div class="topline"><div><div class="eyebrow">9 October · first kept hinge stack · manual selection</div><h1>Review the spots · 15:45:51</h1></div><div><button id="export">Download backup</button> <button id="import">Import backup</button><input id="importfile" type="file" accept=".json,application/json" hidden></div></div>
<p class="subtle">A deliberately loose candidate list. Start with the strongest, inspect the neighbours, then keep or remove. Nothing is accepted automatically.</p>
<div id="loading">Loading image and saved decisions…</div><div id="app" hidden>
<div class="counts"><span class="pill" id="total"></span><span class="pill keeptext" id="kept"></span><span class="pill removetext" id="removed"></span><span class="pill pendingtext" id="pending"></span><span id="save" class="status" role="status" aria-live="polite"></span></div>
<div class="toolbar">
<label>Image <select id="view"></select></label><label>Display gain <input id="gain" type="number" min="1" max="200" step="1" value="20">×</label>
<label><input id="markers" type="checkbox" checked> Show candidates</label>
<label>Order <select id="order"><option value="score">Strongest first</option><option value="raster">Top → bottom</option></select></label>
<label>Show <select id="filter"><option value="all">All candidates</option><option value="unreviewed">Unreviewed</option><option value="keep">Keep</option><option value="remove">Remove</option></select></label>
<label><input id="advance" type="checkbox" checked> Advance after decision</label>
</div>
<div class="workspace"><section class="panel"><h2>Whole field</h2>
<div class="toolbar"><label>Zoom <select id="zoom"><option value="fit">Fit</option><option value="1">1 sensor pixel / screen pixel</option><option value="2">2×</option></select></label><label><input id="addmode" type="checkbox"><span>Add missed candidate by clicking</span></label></div>
<div class="imagewrap" id="overviewwrap"><canvas id="overview" width="1232" height="1028" tabindex="0" aria-label="Whole field. Click a candidate to inspect it."></canvas></div>
<div class="legend"><span><i class="swatch" style="background:#f3d37d"></i>Unreviewed</span><span><i class="swatch" style="background:#39d895"></i>Keep</span><span><i class="swatch" style="background:#fa727c"></i>Remove</span><span><i class="swatch" style="background:#4fccff"></i>Current</span></div>
<p class="subtle" style="font-size:13px">Click a marker in the overview or detail crop to select it. Turn markers off to inspect the image unobstructed. Shift-click also adds a candidate; additions start unreviewed.</p></section>
<section class="panel"><div class="nav"><button id="prev">← Previous</button><label>Go to ID <input id="jump" type="text" placeholder="S0123" aria-label="Candidate ID"></label><button id="next">Next →</button></div>
<div style="margin:12px 0"><div id="title"></div><div id="position"></div></div>
<div class="detailwrap"><canvas id="detail" width="480" height="480" tabindex="0" aria-label="Enlarged candidate and its neighbours"></canvas></div>
<div class="toolbar" style="margin-top:10px"><label>Context <select id="crop"><option value="64">64 × 64 sensor px</option><option value="96" selected>96 × 96 sensor px</option><option value="160">160 × 160 sensor px</option><option value="256">256 × 256 sensor px</option></select></label><button id="locate">Locate in field</button></div>
<div id="meta"></div><label for="note" class="subtle">Optional note</label><input id="note" type="text" maxlength="2000" placeholder="e.g. isolated rod, cluster, background texture">
<div class="actions"><button id="keep" class="keep">Keep · K</button><button id="remove" class="remove">Remove · R</button><button id="unreviewed">Unreviewed · U</button><button id="undo">Undo</button></div>
<div class="kbd">← / → browse · K keep · R remove · U clear decision · Ctrl/Cmd+Z undo · Space next unreviewed<br>Held keys do not repeat decisions. Changing gain or image view never changes the candidate list.</div><div id="message" role="status" aria-live="polite"></div>
</section></div>
<div class="filmnav"><h2>Candidate contact sheet</h2><div><button id="pageprev">← Page</button> <span id="pageinfo"></span> <button id="pagenext">Page →</button></div></div><div id="film" class="film"></div>
<details><summary>What this detector includes, and where decisions are saved</summary><p id="method"></p><p>Earlier native and supplied-detector candidates are included within a 4-pixel merge tolerance. There is no component-area, local-contrast, recurrence, brightness-gate or θ rejection and no maximum-candidate cap. Closely spaced detections within 4 sensor pixels are merged; you can add a missed neighbour manually. Edge candidates remain available with clipped context.</p>
<p>Images average each 2 × 2 polarizer cell, as in the intensity view; the map has 1,232 × 1,028 cells covering 2,464 × 2,056 sensor pixels. The first 150 frames are used and the phase-marker frame is excluded. The saved background subtraction is preserved. Display gain is linear: white = 4,095 / gain, black = 0. A 20× display has white = 204.75 saved intensity units.</p>
<p>Decisions autosave to <code>analysis/analysis_2026-10-10_friday/manual_spot_review_154551/manual_review.json</code>. Coordinates and detector settings are in <code>manifest.json</code> beside it. Download backup exports both coordinates and decisions in one portable JSON. Only <strong>keep</strong> will enter the later manual list; unreviewed candidates remain undecided. θ is intentionally absent from this selection screen.</p>
<p>This viewer runs only on this computer. To restart it from the polarcam folder: <code>.venv/Scripts/python.exe analysis/analysis_2026-10-10_friday/manual_spot_review.py --serve</code>. Re-running preserves your decisions.</p></details>
</div></main><script>
'use strict';
const $=id=>document.getElementById(id);let manifest,state,token,points=[],byId=new Map(),selected=null,order=[],imageData,bufferMap={},imageCanvas=document.createElement('canvas'),page=0,busy=false,history=[],cropBounds=null,lastBackup=null,loadingView=0;
const PAGE_SIZE=32,COLOR={keep:'#39d895',remove:'#fa727c',unreviewed:'#f3d37d'};
const status=p=>state.decisions[p.id]?.status||'unreviewed';
function say(text,error=false){$('save').textContent=text;$('save').classList.toggle('error',error)}
function message(text){$('message').textContent=text}
function localKey(){return 'polarcam-manual-'+manifest.dataset_id}
function backupLocal(){try{localStorage.setItem(localKey(),JSON.stringify({state,selected,view:$('view').value,gain:$('gain').value,order:$('order').value}))}catch(e){/* Disk autosave remains authoritative. */}}
function rebuild(){points=[...manifest.candidates,...state.added];byId=new Map(points.map(p=>[p.id,p]));reorder()}
function reorder(){order=points.filter(p=>$('filter').value==='all'||status(p)===$('filter').value).sort((a,b)=>$('order').value==='raster'?(a.y-b.y||a.x-b.x):((b.score??Infinity)-(a.score??Infinity)||a.y-b.y||a.x-b.x));}
function counters(){let counts={keep:0,remove:0,unreviewed:0};for(const p of points)counts[status(p)]++;$('total').textContent=points.length.toLocaleString()+' candidates';$('kept').textContent=counts.keep+' keep';$('removed').textContent=counts.remove+' remove';$('pending').textContent=counts.unreviewed+' unreviewed';}
async function api(req){
 if(busy)return null;busy=true;setDisabled();say('Saving to disk…');
 try{
  const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify({...req,revision:state.revision})});
  const data=await response.json();if(data.state){state=data.state;rebuild();backupLocal()}
  if(!response.ok)throw Error(data.error||'Save failed');
  say('Saved to disk · revision '+state.revision);return data.state;
 }catch(e){say('Not saved: '+e.message,true);message('This action was not confirmed. Keep this page open and retry; use Download backup if the server has stopped.');return null}
 finally{busy=false;setDisabled();render()}
}
function setDisabled(){for(const id of ['keep','remove','unreviewed','undo','prev','next','import'])$(id).disabled=busy||(!selected&&['keep','remove','unreviewed'].includes(id));}
async function label(newStatus){
 if(busy||!selected)return;const id=selected,prior=state.decisions[id]||{status:'unreviewed',note:''};
 const before=order.map(p=>p.id),i=before.indexOf(id);let next=before[i+1]??null;
 if(!next&&$('filter').value==='unreviewed')next=before.find(s=>s!==id)||null;
 const result=await api({op:'label',id,status:newStatus,note:$('note').value});
 if(result){history.push({id,...prior});message('');if($('advance').checked&&next)select(next);else if($('advance').checked&&!next)message('End of this list. Earlier undecided candidates may remain.');else render()}
}
async function saveNote(id,note){if(!id||busy)return;const old=state.decisions[id];if(note!==(old?.note||''))await api({op:'label',id,status:old?.status||'unreviewed',note});}
async function undo(){if(busy||!history.length){message('No decision to undo in this session. Any candidate can be changed directly.');return}const h=history.pop();const result=await api({op:'label',id:h.id,status:h.status,note:h.note||''});if(result){$('filter').value='all';reorder();select(h.id);message('Last decision restored.')}else history.push(h)}
function select(id){if(!byId.has(id))return;selected=id;const i=order.findIndex(p=>p.id===id);if(i>=0)page=Math.floor(i/PAGE_SIZE);render();backupLocal()}
function browse(step){if(busy||!order.length)return;const i=order.findIndex(p=>p.id===selected),j=Math.max(0,Math.min(order.length-1,i+step));select(order[j].id)}
function nextPending(){if(busy)return;const all=[...points].sort((a,b)=>$('order').value==='raster'?(a.y-b.y||a.x-b.x):((b.score??Infinity)-(a.score??Infinity)||a.y-b.y||a.x-b.x));const i=all.findIndex(p=>p.id===selected);const p=[...all.slice(i+1),...all.slice(0,i+1)].find(p=>status(p)==='unreviewed');if(p){$('filter').value='unreviewed';reorder();select(p.id)}else message('All candidates have a decision.')}
async function changeView(){
 const request=++loadingView,key=$('view').value;message('Loading image…');
 try{if(!bufferMap[key]){const v=manifest.views.find(v=>v.key===key);const r=await fetch('/'+v.file);if(!r.ok)throw Error('Image load failed');bufferMap[key]=new Float32Array(await r.arrayBuffer());if(bufferMap[key].length!==manifest.cell_width*manifest.cell_height)throw Error('Image size mismatch')}
 if(request!==loadingView)return;imageData=bufferMap[key];makeImage();render();message('');backupLocal();}
 catch(e){message(e.message)}
}
function makeImage(){if(!imageData)return;const gain=Math.min(200,Math.max(1,Number($('gain').value)||20));$('gain').value=gain;imageCanvas.width=manifest.cell_width;imageCanvas.height=manifest.cell_height;const ctx=imageCanvas.getContext('2d'),im=ctx.createImageData(imageCanvas.width,imageCanvas.height),factor=255*gain/4095;for(let i=0;i<imageData.length;i++){const v=Math.max(0,Math.min(255,Math.round(imageData[i]*factor)));im.data[4*i]=v;im.data[4*i+1]=v;im.data[4*i+2]=v;im.data[4*i+3]=255}ctx.putImageData(im,0,0)}
function ring(ctx,p,x,y,r,selectedRing=false){ctx.strokeStyle=selectedRing?'#4fccff':COLOR[status(p)];ctx.lineWidth=selectedRing?2:1;ctx.globalAlpha=selectedRing?1:(status(p)==='remove'?.45:.78);ctx.beginPath();ctx.arc(x,y,r,0,2*Math.PI);ctx.stroke();ctx.globalAlpha=1;if(selectedRing){ctx.beginPath();ctx.moveTo(x-r-7,y);ctx.lineTo(x-r-2,y);ctx.moveTo(x+r+2,y);ctx.lineTo(x+r+7,y);ctx.moveTo(x,y-r-7);ctx.lineTo(x,y-r-2);ctx.moveTo(x,y+r+2);ctx.lineTo(x,y+r+7);ctx.stroke()}}
function overview(){const c=$('overview'),ctx=c.getContext('2d');ctx.drawImage(imageCanvas,0,0);if($('markers').checked){for(const p of points)ring(ctx,p,(p.x+.5)/2,(p.y+.5)/2,2.7)}if(selected){const p=byId.get(selected);ring(ctx,p,(p.x+.5)/2,(p.y+.5)/2,8,true);ctx.fillStyle='#4fccff';ctx.font='14px system-ui';const tx=Math.min(c.width-90,Math.max(4,(p.x+.5)/2+11)),ty=Math.max(17,(p.y+.5)/2-9);ctx.fillText(p.id,tx,ty)}const z=$('zoom').value;c.style.width=z==='fit'?'100%':manifest.width*Number(z)+'px';}
function drawCrop(canvas,p,span,markers=true){
 const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;ctx.fillStyle='#030608';ctx.fillRect(0,0,w,h);ctx.imageSmoothingEnabled=false;
 const x0=p.x-span/2,y0=p.y-span/2,scale=w/span;
 // Whole offscreen cell image is mapped into the clipped canvas; padding stays dark.
 ctx.drawImage(imageCanvas,-(x0+.5)*scale,-(y0+.5)*scale,manifest.width*scale,manifest.height*scale);
 if(markers&&$('markers').checked){for(const q of points){if(Math.abs(q.x-p.x)<=span/2+8&&Math.abs(q.y-p.y)<=span/2+8)ring(ctx,q,(q.x-x0)*scale,(q.y-y0)*scale,Math.max(3,4*scale),q.id===p.id)}}
 if(canvas.id==='detail')cropBounds={x0,y0,span};
}
function film(){const area=$('film');area.replaceChildren();page=Math.min(page,Math.max(0,Math.ceil(order.length/PAGE_SIZE)-1));const visible=order.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE);if(!visible.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='No candidates in this filter.';area.append(empty)}for(const p of visible){const b=document.createElement('button');b.className='thumb'+(p.id===selected?' selected':'');b.title=p.id+' · '+status(p);b.setAttribute('aria-label','Inspect '+p.id+' '+status(p));const canvas=document.createElement('canvas');canvas.width=112;canvas.height=112;drawCrop(canvas,p,64,false);const ctx=canvas.getContext('2d');ctx.strokeStyle=COLOR[status(p)];ctx.lineWidth=2;ctx.strokeRect(1,1,110,110);ctx.beginPath();ctx.arc(56,56,7,0,2*Math.PI);ctx.stroke();const caption=document.createElement('span');caption.className='caption';caption.textContent=p.id+' · '+status(p);b.append(canvas,caption);b.onclick=()=>select(p.id);area.append(b)}$('pageinfo').textContent=order.length?`${page+1} / ${Math.ceil(order.length/PAGE_SIZE)}`:'0 / 0';$('pageprev').disabled=page===0;$('pagenext').disabled=(page+1)*PAGE_SIZE>=order.length;}
function render(){if(!manifest||!imageData)return;counters();overview();const p=byId.get(selected);if(p){$('title').textContent=p.id+' · '+status(p);const index=order.findIndex(q=>q.id===selected);$('position').textContent=`${index>=0?'Candidate '+(index+1)+' of '+order.length+' in this order · ':''}x=${p.x.toFixed(1)}, y=${p.y.toFixed(1)} sensor px`;$('meta').textContent=p.origins.join(' + ')+(p.edge?' · edge context clipped':'');if(document.activeElement!==$('note'))$('note').value=state.decisions[p.id]?.note||'';drawCrop($('detail'),p,Number($('crop').value));}else{$('title').textContent='No candidate selected';$('position').textContent='';}film();setDisabled();}
async function imageClick(event,which){
 if(busy)return;const canvas=$(which),rect=canvas.getBoundingClientRect(),sx=(event.clientX-rect.left)/rect.width,sy=(event.clientY-rect.top)/rect.height;
 const x=which==='overview'?sx*manifest.width-.5:cropBounds.x0+sx*cropBounds.span;
 const y=which==='overview'?sy*manifest.height-.5:cropBounds.y0+sy*cropBounds.span;
 if(x<0||y<0||x>=manifest.width||y>=manifest.height)return;
 let closest=null,d=Infinity;for(const p of points){const v=Math.hypot(p.x-x,p.y-y);if(v<d){d=v;closest=p}}
 if($('addmode').checked||event.shiftKey){
  if(d<2){select(closest.id);message('An existing candidate is within 2 pixels; selected it.');return}
  const old=new Set(points.map(p=>p.id));const result=await api({op:'add',x,y});if(result){const p=state.added.find(p=>!old.has(p.id));$('filter').value='all';reorder();select(p.id);message('Candidate added, still unreviewed. Press K to keep it.');$('addmode').checked=false}return;
 }
 const tolerance=which==='overview'?Math.max(12,12*manifest.width/rect.width):Math.max(8,10*cropBounds.span/rect.width);
 if(closest&&d<=tolerance)select(closest.id);else message('No nearby candidate. Shift-click here to add one.');
}
async function download(){try{let data;try{const r=await fetch('/api/export');if(!r.ok)throw Error('export');data=await r.json()}catch(e){data={...state,method:manifest.method,spots:points.map(p=>({...p,status:status(p),note:state.decisions[p.id]?.note||''})),selection_rule:'Use status == keep only; unreviewed is not accepted'}}const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='manual_spots_20261009-154551.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message('Backup downloaded; it includes coordinates and decisions.')}catch(e){message('Export failed: '+e.message)}}
async function restore(file){if(!file)return;try{const data=JSON.parse(await file.text());if(data.dataset_id!==manifest.dataset_id)throw Error('This backup belongs to different candidates.');if(!confirm('Import this backup? Decisions contained in it will replace matching saved decisions. Other decisions stay unchanged.'))return;const result=await api({op:'import',data});if(result){history=[];message('Backup imported and saved to disk.')}}catch(e){message(e.message)}}
async function init(){
 try{const [m,r]=await Promise.all([fetch('/manifest.json').then(r=>r.json()),fetch('/api/review').then(r=>r.json())]);manifest=m;state=r.state;token=r.token;if(state.dataset_id!==manifest.dataset_id)throw Error('Candidate/review mismatch');
 for(const v of manifest.views){const o=document.createElement('option');o.value=v.key;o.textContent=v.label;$('view').append(o)}
 let preferences;try{preferences=JSON.parse(localStorage.getItem(localKey()))}catch(e){}if(preferences){if(manifest.views.some(v=>v.key===preferences.view))$('view').value=preferences.view;$('gain').value=preferences.gain||20;if(['score','raster'].includes(preferences.order))$('order').value=preferences.order}
 rebuild();selected=preferences&&byId.has(preferences.selected)?preferences.selected:order.find(p=>status(p)==='unreviewed')?.id||order[0]?.id;
 $('method').textContent=`${manifest.candidates.length.toLocaleString()} candidates. Custom detector: DoG with σ = 3 and 8 sensor pixels; peaks above the global median + 3 × robust MAD scale (threshold ${manifest.method.threshold_value.toFixed(3)} saved intensity units), local maxima in a 9 × 9 cell neighbourhood. Strongest-first order uses this spatial contrast score; it is not a calibrated SNR. No previous keep/remove classification is inherited.`;
 await changeView();if(!imageData)throw Error('Image data did not load');$('loading').hidden=true;$('app').hidden=false;say('Saved to disk · revision '+state.revision);render();
 }catch(e){$('loading').textContent='Could not load review: '+e.message+'. Start manual_spot_review.py --serve, then refresh this page.'}
}
$('keep').onclick=()=>label('keep');$('remove').onclick=()=>label('remove');$('unreviewed').onclick=()=>label('unreviewed');$('undo').onclick=undo;$('prev').onclick=()=>browse(-1);$('next').onclick=()=>browse(1);
$('view').onchange=changeView;$('gain').onchange=()=>{makeImage();render();backupLocal()};$('markers').onchange=render;$('crop').onchange=render;$('zoom').onchange=render;
for(const id of ['filter','order'])$(id).onchange=()=>{reorder();page=0;if(!order.some(p=>p.id===selected))selected=order[0]?.id||selected;render();backupLocal()};
$('overview').onclick=e=>imageClick(e,'overview');$('detail').onclick=e=>imageClick(e,'detail');
$('pageprev').onclick=()=>{page--;film()};$('pagenext').onclick=()=>{page++;film()};
$('jump').onkeydown=e=>{if(e.key==='Enter'){let id=e.target.value.trim().toUpperCase();if(/^\d+$/.test(id))id='S'+id.padStart(4,'0');if(byId.has(id)){$('filter').value='all';reorder();select(id);e.target.blur()}else message('No candidate with that ID.')}};
$('note').onchange=()=>{const id=selected,note=$('note').value;setTimeout(()=>saveNote(id,note),0)};
$('locate').onclick=()=>{if(!selected)return;const p=byId.get(selected);$('zoom').value='1';overview();const wrap=$('overviewwrap');wrap.scrollLeft=p.x-wrap.clientWidth/2;wrap.scrollTop=p.y-wrap.clientHeight/2;};
$('export').onclick=download;$('import').onclick=()=>$('importfile').click();$('importfile').onchange=e=>{restore(e.target.files[0]);e.target.value=''};
document.addEventListener('keydown',e=>{if(!manifest||e.repeat||['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)||e.altKey)return;if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();undo();return}if(e.ctrlKey||e.metaKey)return;const key=e.key.toLowerCase();if(key==='k'){e.preventDefault();label('keep')}else if(key==='r'){e.preventDefault();label('remove')}else if(key==='u'){e.preventDefault();label('unreviewed')}else if(e.key==='ArrowRight'){e.preventDefault();browse(1)}else if(e.key==='ArrowLeft'){e.preventDefault();browse(-1)}else if(e.key===' '){e.preventDefault();nextPending()}});
window.addEventListener('beforeunload',e=>{if(busy){e.preventDefault();e.returnValue='A decision is still saving.'}});
init();
</script></body></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--serve',action='store_true');parser.add_argument('--port',type=int,default=PORT)
    args=parser.parse_args();manifest=build()
    if args.serve:serve(manifest,args.port)
