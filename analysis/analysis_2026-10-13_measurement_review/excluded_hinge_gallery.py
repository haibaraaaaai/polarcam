"""Browse previously excluded hinge candidates; never rerun or alter selection.

Uses existing detection previews and decisions, with per-frame anisotropy from
the original fixed apertures. Compact lossless sprites keep the gallery small.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'matched_background_theta'
OUT = HERE / 'excluded_hinge_spots'
CROP = 88
TILE_W = CROP * 3
COLS = 8
MODES = ('native', 'fixed')


def reason(spot, mode, factor='1x'):
    if not spot['matched_' + mode]:
        return 'no_match'
    if spot['ceiling_veto']:
        return 'ceiling'
    if spot['theta_' + factor] is None:
        return 'invalid'
    return 'included'


def build():
    OUT.mkdir(exist_ok=True)
    (OUT / 'images').mkdir(exist_ok=True)
    source_file = SOURCE / 'results.json'
    original_bytes = source_file.read_bytes()
    data = json.loads(original_bytes)
    records = [r for r in data['records'] if r['group'] == 'Hinges']
    assert len(records) == 16
    rows, stacks = [], []
    counts = {mode: dict(total=0, no_match=0, ceiling=0, invalid=0, only_2x=0, all_2x=0)
              for mode in MODES}
    for r in records:
        possible = [s for s in r['spots'] if any(reason(s, mode, factor) != 'included'
                    for mode in MODES for factor in ('1x', '2x'))]
        sprite = Image.new('L', (TILE_W * COLS, CROP * math.ceil(len(possible) / COLS)))
        inputs = [Image.open(SOURCE / v).convert('L') for v in
                  (r['factors'][0]['image'], r['factors'][-1]['image'], r['background_image'])]
        assert all(im.size == (r['shape'][1], r['shape'][0]) for im in inputs)
        sprite_path = f'images/hinges_{r["group_index"]:02d}.webp'
        stack = dict(index=r['group_index'], time=r['time'], name=r['name'], note=r['note'],
                     background=r['background'], background_sha256=r['background_sha256'],
                     sprite=sprite_path, baseline_count=len(r['spots']), counts={})
        for mode in MODES:
            all_reasons = [reason(s, mode) for s in r['spots']]
            ss = {key: all_reasons.count(key) for key in ('no_match', 'ceiling', 'invalid')}
            ss['total'] = sum(ss.values())
            ss['only_2x'] = sum(reason(s, mode) == 'included' and reason(s, mode, '2x') == 'invalid'
                                for s in r['spots'])
            ss['all_2x'] = sum(reason(s, mode, '2x') != 'included' for s in r['spots'])
            stack['counts'][mode] = ss
            for key in counts[mode]:
                counts[mode][key] += ss[key]
        for index, s in enumerate(possible):
            left, top = round(s['x']) - CROP // 2, round(s['y']) - CROP // 2
            crop_box = (left, top, left + CROP, top + CROP)
            tx, ty = index % COLS * TILE_W, index // COLS * CROP
            for panel, im in enumerate(inputs):
                # PIL pads out-of-frame areas black without moving the centre.
                sprite.paste(im.crop(crop_box), (tx + panel * CROP, ty))
            row = dict(s, key=f'H{r["group_index"]:02d}-S{s["id"]:03d}', stack=r['group_index'],
                       crop_left=left, crop_top=top, tile=[tx, ty],
                       edge_padding=left < 0 or top < 0 or left + CROP > inputs[0].width
                                    or top + CROP > inputs[0].height,
                       reason={}, reason_2x={}, nearest_2x={}, match_2x={})
            for mode in MODES:
                row['reason'][mode] = reason(s, mode)
                row['reason_2x'][mode] = reason(s, mode, '2x')
                endpoint = r['factors'][-1][mode]
                points = endpoint['points']
                closest = min(points, key=lambda p: math.hypot(p[0]-s['x'], p[1]-s['y'])) if points else None
                row['nearest_2x'][mode] = None if closest is None else dict(
                    x=closest[0], y=closest[1], distance=math.hypot(closest[0]-s['x'], closest[1]-s['y']))
                match = [p for p, bid in zip(points, endpoint['baseline_ids']) if bid == s['id']]
                assert len(match) <= 1
                assert bool(match) == s['matched_' + mode]
                row['match_2x'][mode] = match[0] if match else None
            rows.append(row)
        sprite.save(OUT / sprite_path, lossless=True, method=4)
        stacks.append(stack)
        print(f'Hinge {r["group_index"]:02d}: {stack["counts"]["native"]["total"]} main exclusions, '
              f'{len(possible)} preview tiles covering both policies/panels', flush=True)
    assert counts['native']['total'] == 332
    assert counts['native']['no_match'] == 325
    assert counts['native']['ceiling'] == 1 and counts['native']['invalid'] == 6
    assert counts['native']['only_2x'] == 1062
    for mode in MODES:
        total = sum(r['baseline_count'] for r in stacks)
        previous = data['theta_summary'][mode]['Hinges']
        assert total-counts[mode]['total'] == previous['1x']['n']
        assert total-counts[mode]['all_2x'] == previous['2x']['n']
    report = dict(stacks=stacks, spots=rows, counts=counts, crop=CROP, tile_width=TILE_W,
                  source_sha256=hashlib.sha256(original_bytes).hexdigest(),
                  source='../matched_background_theta/results.json', model=data['model'],
                  parameters=data['parameters'],
                  display='Existing four-polarizer-cell intensity map of 150-frame mean; fixed 20x display; white=204.75 DN. Background is the recorded profile at the same coordinates.',
                  selection='Current saved decisions only. Main exclusions are from 1x-theta histogram after 1x/2x detection matching; optional 2x-only view is separate.')
    add_anisotropy(report, data)
    (OUT / 'gallery.json').write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':')),
                                     encoding='utf-8')
    with (OUT / 'spots.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ('key', 'stack', 'id', 'x', 'y', 'theta_1x', 'theta_2x', 'raw_peak',
                  'X_1x', 'Y_1x', 'r_1x', 'X_2x', 'Y_2x', 'r_2x',
                  'reason_native', 'reason_fixed', 'reason_2x_native', 'reason_2x_fixed')
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in rows:
            row = {k: s[k] for k in fields if k in s}
            row.update(reason_native=s['reason']['native'], reason_fixed=s['reason']['fixed'],
                       reason_2x_native=s['reason_2x']['native'], reason_2x_fixed=s['reason_2x']['fixed'])
            row.update(X_1x=s['xy_1x'][0], Y_1x=s['xy_1x'][1],
                       X_2x=s['xy_2x'][0], Y_2x=s['xy_2x'][1])
            writer.writerow(row)
    html = HTML.replace('__DATA__', json.dumps(report, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/'))
    (OUT / 'index.html').write_text(html, encoding='utf-8')
    make_preview(report)
    assert source_file.read_bytes() == original_bytes
    print('COUNTS', json.dumps(counts), flush=True)
    print(OUT / 'index.html', flush=True)


def add_anisotropy(report, source):
    """Per-frame anisotropy in the original fixed apertures; no new rejection."""
    from polarcam.live.intensity_spots import anisotropy
    day = HERE.parents[1] / 'data/hinges/polarcam/2026-10-12'
    rows = report['spots']
    keys = np.array([s['key'] for s in rows])
    cache_file = OUT / 'anisotropy_traces.npz'
    cache = None
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as z:
            if str(z['source_sha256']) == report['source_sha256'] and np.array_equal(z['keys'], keys):
                cache = z['xy'].copy()
    xy_all = np.full((len(rows), 2, 150, 2), np.nan, dtype=np.float32) if cache is None else cache
    by_name = {r['name']: r for r in source['records'] if r['group'] == 'Hinges'}
    for st in report['stacks']:
        indices = [i for i,s in enumerate(rows) if s['stack'] == st['index']]
        spots = [rows[i] for i in indices]
        rec = by_name[st['name']]
        path = day / 'frames' / st['name']
        stat = path.stat()
        assert stat.st_size == rec['source_bytes'] and stat.st_mtime_ns == rec['source_mtime_ns']
        if cache is None:
            meta_path = path.with_suffix('.json')
            assert hashlib.sha256(meta_path.read_bytes()).hexdigest() == rec['sidecar_sha256']
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            assert len(a)-int(meta['actual']['phase_marker_appended']) == 150
            bp = day / 'backgrounds/used' / (st['background_sha256'] + '.npy')
            assert hashlib.sha256(bp.read_bytes()).hexdigest() == st['background_sha256']
            background = np.load(bp, mmap_mode='r', allow_pickle=False)
            bounds = np.array([s['bounds'] for s in spots])
            ys = bounds[:,2,None,None] + np.arange(18)[None,:,None]
            xs = bounds[:,0,None,None] + np.arange(18)[None,None,:]
            yy = ys + int(meta['roi']['OffsetY'])
            xx = xs + int(meta['roi']['OffsetX'])
            weights = np.stack([((yy%2==py)&(xx%2==px)).astype(float)
                                for py,px in ((1,1),(0,0),(0,1),(1,0))],axis=1)
            assert np.all(weights.sum(axis=(2,3)) == 81)
            b = background[ys,xs].astype(float)
            channels = np.empty((2,150,len(spots),4), dtype=np.float64)
            for t,frame in enumerate(a[:150]):
                patch = frame[ys,xs].astype(float)
                channels[0,t] = np.einsum('nij,nkij->nk',patch,weights)/81
                channels[1,t] = np.einsum('nij,nkij->nk',np.maximum(patch-b,0),weights)/81
            for j,suffix in enumerate(('1x','2x')):
                # Verify against the existing full-frame mean reconstruction.
                expected = np.array([s['channels_'+suffix] for s in spots])
                np.testing.assert_allclose(channels[j].mean(axis=0), expected, rtol=1e-12, atol=1e-10)
                xy_all[indices,j] = np.transpose(anisotropy(channels[j]), (1,0,2))
            assert path.stat().st_mtime_ns == stat.st_mtime_ns
            print(f'Anisotropy H{st["index"]:02d}: {len(spots)} candidates x 150 frames x 2 BG settings',flush=True)
        for i in indices:
            s = rows[i]
            s['anisotropy_frames'] = {}
            for j,suffix in enumerate(('1x','2x')):
                v = xy_all[i,j]
                valid = np.isfinite(v).all(axis=1)
                s['anisotropy_frames'][suffix] = dict(valid=int(valid.sum()),
                    outside_theta_model=int((np.linalg.norm(v[valid],axis=1)>report['model']['r_max']).sum()))
    if cache is None:
        np.savez_compressed(cache_file, keys=keys, xy=xy_all, source_sha256=report['source_sha256'],
                            factors=np.array([1.,2.]), frame_index=np.arange(150),
                            axis_order=np.array(['candidate','background','frame','X_Y']))
    # Two equally scaled panels per candidate; plot all finite X/Y, including
    # radii outside the theta model. The mean-intensity inversion is a cross.
    size, columns = 176, 8
    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
    font = ImageFont.truetype(str(font_path),11) if font_path.exists() else ImageFont.load_default()
    def panel(values, mean_xy, color):
        im = Image.new('RGB',(size,size),'#ffffff'); d = ImageDraw.Draw(im)
        x0,y0,x1,y1 = 27,12,166,151
        def pos(x,y):return (x0+(x+1)*(x1-x0)/2,y1-(y+1)*(y1-y0)/2)
        for k in (-1,-.5,0,.5,1):
            px,py=pos(k,k)
            d.line((px,y0,px,y1),fill='#d2dedc' if k==0 else '#eef2ef')
            d.line((x0,py,x1,py),fill='#d2dedc' if k==0 else '#eef2ef')
        limit=report['model']['r_max']
        for angle in range(0,360,12):
            a=np.radians(np.linspace(angle,angle+6,4))
            points=[pos(limit*np.cos(t),limit*np.sin(t)) for t in a]
            d.line(points,fill='#a5b4b1',width=1)
        for x,y in values:
            if np.isfinite(x) and np.isfinite(y):
                px,py=pos(float(x),float(y));d.ellipse((px-1,py-1,px+1,py+1),fill=color)
        if not np.isfinite(values).all(axis=1).any():
            d.text(((x0+x1)/2,(y0+y1)/2),'No valid X/Y',anchor='mm',font=font,fill='#526a70')
        if all(v is not None and np.isfinite(v) for v in mean_xy):
            px,py=pos(*mean_xy)
            d.line((px-4,py,px+4,py),fill='#101820',width=2)
            d.line((px,py-4,px,py+4),fill='#101820',width=2)
        d.rectangle((x0,y0,x1,y1),outline='#758d86')
        for k in (-1,0,1):
            px,py=pos(k,k);d.text((px-5,y1+1),str(k),font=font,fill='#48625c')
            d.text((12,py-7),str(k),font=font,fill='#48625c')
        d.text((93,163),'X',font=font,fill='#243b40');d.text((0,68),'Y',font=font,fill='#243b40')
        return im
    examples=[]
    for st in report['stacks']:
        indices=[i for i,s in enumerate(rows) if s['stack']==st['index']]
        sheet=Image.new('RGB',(size*2*columns,size*math.ceil(len(indices)/columns)),'white')
        for tile,i in enumerate(indices):
            s=rows[i];tx,ty=tile%columns*size*2,tile//columns*size
            s['anisotropy_tile']=[tx,ty]
            for j,suffix,color in ((0,'1x','#117e94'),(1,'2x','#ca841b')):
                patch=panel(xy_all[i,j],s['xy_'+suffix],color)
                sheet.paste(patch,(tx+j*size,ty))
            if s['reason']['native']!='included' and len(examples)<6:
                examples.append((s['key'],sheet.crop((tx,ty,tx+size*2,ty+size))))
        st['anisotropy_sprite']=f'images/anisotropy_{st["index"]:02d}.webp'
        sheet.save(OUT/st['anisotropy_sprite'],lossless=True,method=4)
    report['anisotropy'] = dict(frames=150, panel_px=size,
        dot='One X/Y point per measured frame; same fixed 18x18 aperture as theta analysis.',
        mean='Black cross and table values: anisotropy of the 150-frame mean channel intensities, not the arithmetic mean of per-frame X/Y.',
        invalid='Only nonfinite X/Y omitted from scatter. Points outside theta model remain visible.',
        axes='X and Y each fixed to [-1,1]. Dashed circle is model r_max.',
        traces='anisotropy_traces.npz')
    example=Image.new('RGB',(size*2*2, (size+38)*3+40),'#f5f7f3');draw=ImageDraw.Draw(example)
    draw.text((12,10),'First six excluded candidates: X/Y per frame · 1x (blue) | 2x (orange)',fill='#243b40',font=font)
    for i,(key,im) in enumerate(examples):
        x,y=i%2*size*2,i//2*(size+38)+40
        draw.text((x+12,y),key,fill='#243b40',font=font);example.paste(im,(x,y+22))
    example.save(OUT/'anisotropy_examples.png')


def make_preview(report):
    """A small, non-selected-by-appearance overview for inline delivery and QA."""
    examples = []
    for stack in report['stacks']:
        candidates = [s for s in report['spots'] if s['stack'] == stack['index'] and s['reason']['native'] != 'included']
        if candidates:
            examples.append(candidates[0])
    scale = 2
    cw, ch = TILE_W * scale + 28, CROP * scale + 75
    canvas = Image.new('RGB', (cw*2, ch*math.ceil(len(examples)/2)+85), '#f6f7f3')
    draw = ImageDraw.Draw(canvas)
    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
    font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    title = ImageFont.truetype(str(font_path), 24) if font_path.exists() else font
    draw.text((15, 12), 'Excluded hinge candidates — first excluded candidate from each stack', fill='#233d42', font=title)
    draw.text((15, 47), '1x | 2x | recorded background · 20x display · red square = 18x18 extraction window', fill='#405d61', font=font)
    cache = {}
    for i, s in enumerate(examples):
        stack = report['stacks'][s['stack']-1]
        if s['stack'] not in cache:
            cache[s['stack']] = Image.open(OUT / stack['sprite'])
        x, y = s['tile']
        im = cache[s['stack']].crop((x,y,x+TILE_W,y+CROP)).convert('RGB').resize((TILE_W*scale,CROP*scale),Image.Resampling.NEAREST)
        mark = ImageDraw.Draw(im)
        l, r, t, b = s['bounds']
        for panel in range(3):
            mark.rectangle(((panel*CROP+l-s['crop_left'])*scale,(t-s['crop_top'])*scale,
                            (panel*CROP+r-s['crop_left'])*scale,(b-s['crop_top'])*scale),outline='#ff6472',width=1)
        ox, oy = i%2*cw+14, i//2*ch+85
        draw.text((ox,oy), f'{s["key"]}  {stack["time"]}  {s["reason"]["native"].replace("_", " ")}',fill='#233d42',font=font)
        canvas.paste(im,(ox,oy+28))
        for panel,label in enumerate(('1x background','2x background','Background profile')):
            draw.text((ox+panel*CROP*scale,oy+32+CROP*scale),label,fill='#405d61',font=font)
    canvas.save(OUT/'overview.png')


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Excluded hinge spots · 12 October</title><style>
:root{font:15px/1.5 system-ui,sans-serif;color:#243b40;background:#f5f7f3}*{box-sizing:border-box}body{margin:0}main{max-width:1600px;margin:auto;padding:24px}h1{font-size:30px;margin:6px 0}h2{font-size:21px}p{max-width:1100px}a{color:#086e84}.muted,.small{color:#526a70}.small{font-size:13px}.controls,.box{background:white;border:1px solid #d3ded9;border-radius:7px;padding:13px;margin:14px 0}.controls{position:sticky;top:0;z-index:3;display:flex;gap:12px;flex-wrap:wrap;align-items:center;box-shadow:0 2px 5px #0001}button,select,input{font:inherit;padding:6px 9px;border:1px solid #a8bbb4;border-radius:5px;background:white;color:inherit}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}button:hover:not(:disabled){background:#eaf3ef}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card{background:white;border:1px solid #d3ded9;border-radius:7px;padding:10px;min-width:0}.card h3{font-size:16px;margin:0 0 4px}.badge{display:inline-block;border-radius:4px;background:#f8e3e6;color:#9b293d;padding:2px 7px;font-size:12px}.badge.ceiling{background:#fff0ce;color:#8a5700}.badge.invalid{background:#ede5f6;color:#694193}.badge.included{background:#e3f0e9;color:#236747}.triplet{width:100%;height:auto;display:block;image-rendering:pixelated;background:#070707}.thumbButton{display:block;width:100%;padding:0;border:0;border-radius:0;margin:8px 0;line-height:0}.captions{display:grid;grid-template-columns:repeat(3,1fr);font-size:12px;color:#526a70;text-align:center}.stats{font-size:12px;margin:7px 0 0}.pager{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:16px 0}.summary{font-size:16px}.empty{padding:30px;background:white;border:1px dashed #aabdb4}.notice{border-left:3px solid #bc8051;padding:8px 12px;background:#fff5e9}dialog{width:min(1180px,96vw);max-height:94vh;border:1px solid #92a8a0;border-radius:9px;padding:20px;color:inherit;background:#f8faf6}dialog::backdrop{background:#172b34bb}dialog .top{display:flex;justify-content:space-between;align-items:center;gap:10px}dialog h2{margin:0}.inspector{margin:12px 0}#detailCanvas{width:100%;height:auto;display:block;image-rendering:pixelated;background:#070707}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:5px 8px;border-bottom:1px solid #d6e0da}.key{font-variant-numeric:tabular-nums;font-weight:650}.footer{margin-top:28px}details summary{cursor:pointer}#overviewImage{width:100%;height:auto;max-height:360px;object-fit:contain;background:#080808;image-rendering:pixelated}#position{min-width:120px}@media(max-width:1050px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.controls{position:static}}@media(max-width:600px){main{padding:12px}.grid{grid-template-columns:1fr}dialog{padding:10px}.controls{gap:7px}.captions{font-size:11px}}
.aniso{width:100%;height:auto;display:block;background:white;image-rendering:pixelated}.pairCaptions{display:grid;grid-template-columns:repeat(2,1fr);font-size:12px;text-align:center;margin-top:12px}.pairCaptions b:first-child{color:#117e94}.pairCaptions b:last-child{color:#ad6b08}.mini{font-size:12px;font-variant-numeric:tabular-nums}.mini th,.mini td{padding:3px 6px}.anisoDetail{max-width:760px;margin:auto}
</style></head><body><main>
<div class="muted">12 OCTOBER · FIRST HINGE ROUND · STACKS 1–16</div><h1>Inspect the excluded hinge spots</h1>
<p><a href="../survival_background_map/index.html"><b>New: map survival-filter failures onto their recorded backgrounds</b></a></p>
<p>Default: <b>332 candidates excluded from the original 1×-theta histogram</b> with the GUI threshold policy. Each tile shows exactly the same coordinates at <b>1× BG, 2× BG, and in the recorded background</b>. Click a tile to enlarge it and step through the current selection.</p>
<p class="small">Images are the existing <b>150-frame mean intensity detection maps</b> at fixed <b>20× display gain</b>; white = 204.75 DN. Each patch is 88 × 88 sensor pixels. The square marks the 18 × 18 extraction window, not the detector's blob boundary. At frame edges, out-of-frame pixels are padded black. Original data and selection decisions are unchanged.</p>
<p class="small"><b>Anisotropy plots:</b> one X–Y dot per measured frame, from the same fixed 18 × 18 window. Blue = 1× BG; orange = 2× BG. Black cross and table values use the 150-frame mean channel intensities. Axes stay at −1 to +1; the dashed circle marks the theta model limit. Points outside this circle are shown, even when theta is invalid. Display gain does not enter the calculation.</p>
<div class="controls"><label>Threshold <select id="mode"><option value="native">GUI: mean + 8 SD</option><option value="fixed">Hold original 1× threshold</option></select></label>
<label>Show <select id="reason"><option value="main">Excluded from main 1× theta</option><option value="no_match">No matching 2× detection</option><option value="ceiling">Raw-ceiling exclusion</option><option value="invalid">Invalid theta at 1×</option><option value="only_2x">Kept at 1×; theta invalid only at 2×</option><option value="all_2x">All exclusions from 2× theta</option></select></label>
<label>Stack <select id="stack"><option value="all">All 16 stacks</option></select></label><label><input type="checkbox" id="overlay" checked> Mark extraction window</label>
<label>Per page <select id="pageSize"><option>24</option><option>60</option><option value="all">All</option></select></label></div>
<div class="box"><div id="counts" class="summary"></div><p id="explanation" class="small"></p><p id="stackNote" class="small"></p></div>
<div class="pager"><button id="prevPage">← Previous page</button><span id="position"></span><button id="nextPage">Next page →</button><span id="showing" class="small"></span></div>
<div id="grid" class="grid"></div><p id="empty" class="empty" hidden>No candidates match these filters.</p>
<div class="pager"><button id="bottomPrev">← Previous page</button><span id="bottomPosition"></span><button id="bottomNext">Next page →</button><button id="download">Download visible selection CSV</button></div>
<details class="box"><summary>What each exclusion means</summary><p><b>No matching 2× detection:</b> no one-to-one match within 6 sensor pixels. This can reflect disappearance, a shifted centre, splitting or merging; it is not proof that the original spot was background.</p>
<p><b>Raw ceiling:</b> any extraction-window pixel reached ≥4094 in any measured frame, from the verified raw-peak file. This is a conservative one-code margin below nominal Mono12 4095.</p>
<p><b>Invalid theta:</b> the channel ratios lie outside the optical model's valid range or have invalid pair sums. These spots may still be visually clear particles.</p>
<p><b>Kept at 1×; invalid only at 2×:</b> these <b>are included in the main 1× histogram</b>. They are separated here so that their rejection from the 2× sensitivity plot is not confused with failed spot detection.</p>
<p class="small">Reasons are assigned in order: matching, raw ceiling, then theta validity. The inspector shows all raw values, including other flags. This gallery uses the saved native Intensity DoG detector results (k=8, σ=1/6, area 10–250 px). It adds no new filtering.</p></details>
<p class="footer"><a href="../matched_background_theta/index.html">Pooled theta and full detection comparison</a>
<a href="../matched_background_theta/radius_comparison.html">Radius distributions (all finite r)</a> · <a href="../index.html">Recording gallery</a> · <a href="spots.csv">All candidates available in this gallery (CSV)</a> · <a href="anisotropy_traces.npz">Per-frame X/Y arrays (NPZ)</a> · <a href="gallery.json">Gallery data</a></p>
</main>
<dialog id="detail"><div class="top"><h2 id="detailTitle"></h2><button id="close">Close ×</button></div><p id="detailReason"></p>
<div class="captions"><b>1× background</b><b>2× background</b><b>Recorded background</b></div><canvas id="detailCanvas" width="1056" height="352"></canvas>
<div class="pager"><button id="prevSpot">← Previous spot</button><span id="detailPosition"></span><button id="nextSpot">Next spot →</button><span class="small">← / → to navigate · Esc to close</span></div>
<div class="anisoDetail"><div class="pairCaptions"><b>1× BG · X–Y per frame</b><b>2× BG · X–Y per frame</b></div><canvas id="detailAnisotropy" class="aniso" width="704" height="352"></canvas><p class="small">Dots: individual frames. Black cross: anisotropy from mean channel intensities. Dashed circle: model radius limit; points beyond it remain visible.</p></div>
<div id="detailStats"></div><p id="detailNote" class="small"></p><details><summary>Location in the full 1× field</summary><canvas id="overviewImage" width="986" height="822"></canvas></details></dialog>
<script>
const DATA=__DATA__;
const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={no_match:'No matching 2× detection',ceiling:'Raw ceiling ≥4094',invalid:'Invalid theta',included:'Included'};
const cache=new Map(),stackMap=new Map(DATA.stacks.map(s=>[s.index,s]));
let page=0,filtered=[],active=-1,renderToken=0,detailToken=0;
for(const s of DATA.stacks)$('stack').add(new Option(`${s.index}/16 · ${s.time}`,s.index));
function imload(path){if(!cache.has(path))cache.set(path,new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>{cache.delete(path);reject(new Error('Image unavailable: '+path))};im.src=path}));return cache.get(path)}
function currentReason(s){const mode=$('mode').value;return ['only_2x','all_2x'].includes($('reason').value)?s.reason_2x[mode]:s.reason[mode]}
function validView(s){const mode=$('mode').value,v=$('reason').value;
 if($('stack').value!=='all'&&s.stack!==Number($('stack').value))return false;
 if(v==='main')return s.reason[mode]!=='included';
 if(v==='only_2x')return s.reason[mode]==='included'&&s.reason_2x[mode]==='invalid';
 if(v==='all_2x')return s.reason_2x[mode]!=='included';return s.reason[mode]===v}
const degree=v=>v===null?'invalid':v.toFixed(1)+'°';
const number=(v,d=3)=>Number.isFinite(v)?v.toFixed(d):'invalid';
function anisotropyRows(s){return ['X','Y','r'].map((label,i)=>`<tr><th>${label}</th><td>${number(i===2?s.r_1x:s.xy_1x[i])}</td><td>${number(i===2?s.r_2x:s.xy_2x[i])}</td></tr>`).join('')}
function mark(ctx,s,scale){if(!$('overlay').checked)return;const [l,r,t,b]=s.bounds;
 for(let p=0;p<3;p++){ctx.strokeStyle='#ff6179';ctx.lineWidth=Math.max(1,scale*.45);ctx.setLineDash([]);ctx.strokeRect((p*88+l-s.crop_left)*scale,(t-s.crop_top)*scale,(r-l)*scale,(b-t)*scale)}
 const match=s.match_2x[$('mode').value];if(match){const x=(88+match[0]-s.crop_left)*scale,y=(match[1]-s.crop_top)*scale;ctx.strokeStyle='#00eeed';ctx.beginPath();ctx.moveTo(x-3*scale,y);ctx.lineTo(x+3*scale,y);ctx.moveTo(x,y-3*scale);ctx.lineTo(x,y+3*scale);ctx.stroke()}}
async function drawTile(canvas,s,token){const st=stackMap.get(s.stack),im=await imload(st.sprite);if(token!==renderToken)return;
 const ctx=canvas.getContext('2d'),scale=canvas.width/DATA.tile_width;ctx.imageSmoothingEnabled=false;ctx.drawImage(im,s.tile[0],s.tile[1],264,88,0,0,canvas.width,canvas.height);mark(ctx,s,scale)}
async function drawAnisotropy(canvas,s,token,detail=false){const st=stackMap.get(s.stack),im=await imload(st.anisotropy_sprite);if(token!==(detail?detailToken:renderToken))return;
 const ctx=canvas.getContext('2d'),p=DATA.anisotropy.panel_px;ctx.imageSmoothingEnabled=false;ctx.drawImage(im,s.anisotropy_tile[0],s.anisotropy_tile[1],p*2,p,0,0,canvas.width,canvas.height)}
function counts(){const mode=$('mode').value,st=$('stack').value==='all'?null:stackMap.get(Number($('stack').value)),c=st?st.counts[mode]:DATA.counts[mode];
 $('counts').innerHTML=`<b>${c.total} main exclusions</b>: ${c.no_match} without a 2× match · ${c.ceiling} raw ceiling · ${c.invalid} invalid theta at 1×. <span class="small">Another ${c.only_2x} are kept at 1× but have invalid theta at 2×.</span>`;
 $('explanation').textContent=$('reason').value==='only_2x'?'This view contains spots included in the main 1× histogram. They fail angle reconstruction only after extra background subtraction.':$('reason').value==='all_2x'?'This view includes every reason for exclusion from the 2×-theta plot, not just failed detection.':'These candidates were excluded from the main 1×-theta histogram. Click any image to inspect the reason and channel values.';
 $('stackNote').textContent=st?`${st.name} · Note: ${st.note||'None'} · Background: ${st.background}`:'Hinge stacks 1–9 used the 18:36:50 background; stacks 10–16 used 18:50:32.';}
function render(reset=false){if(reset)page=0;filtered=DATA.spots.filter(validView);const size=$('pageSize').value==='all'?Math.max(1,filtered.length):Number($('pageSize').value),pages=Math.max(1,Math.ceil(filtered.length/size));page=Math.max(0,Math.min(pages-1,page));const start=page*size,items=filtered.slice(start,start+size),token=++renderToken;
 counts();$('position').textContent=`Page ${page+1} / ${pages}`;$('bottomPosition').textContent=$('position').textContent;
 $('showing').textContent=filtered.length?`${start+1}–${start+items.length} of ${filtered.length} candidates`:'0 candidates';$('empty').hidden=!!items.length;
 for(const id of ['prevPage','bottomPrev'])$(id).disabled=page===0;for(const id of ['nextPage','bottomNext'])$(id).disabled=page===pages-1;
 $('download').disabled=!filtered.length;
 $('grid').innerHTML=items.map((s,i)=>{const st=stackMap.get(s.stack),why=currentReason(s);return `<article class="card"><h3><span class="key">${s.key}</span> · ${st.time}</h3><span class="badge ${why}">${labels[why]}${why==='invalid'?($('reason').value.includes('2x')?' at 2×':' at 1×'):''}</span><button class="thumbButton" data-open="${start+i}" aria-label="Inspect ${s.key}"><canvas class="triplet" width="528" height="176" data-tile="${i}"></canvas></button><div class="captions"><span>1× BG</span><span>2× BG</span><span>Background</span></div><div class="pairCaptions"><b>1× BG · X–Y per frame</b><b>2× BG · X–Y per frame</b></div><button class="thumbButton" data-open="${start+i}" aria-label="Enlarge anisotropy for ${s.key}"><canvas class="aniso" width="704" height="352" data-aniso="${i}"></canvas></button><table class="mini"><thead><tr><th>Mean intensities →</th><th>1× BG</th><th>2× BG</th></tr></thead><tbody>${anisotropyRows(s)}</tbody></table><p class="stats">θ ${degree(s.theta_1x)} → ${degree(s.theta_2x)} · raw peak ${s.raw_peak}<br>x=${s.x.toFixed(1)}, y=${s.y.toFixed(1)}${s.edge_padding?' · edge padding':''}</p></article>`}).join('');
 $('grid').querySelectorAll('[data-tile]').forEach(c=>drawTile(c,items[Number(c.dataset.tile)],token).catch(e=>{c.parentElement.insertAdjacentHTML('afterend',`<p>${esc(e.message)}</p>`)}));
 $('grid').querySelectorAll('[data-aniso]').forEach(c=>drawAnisotropy(c,items[Number(c.dataset.aniso)],token).catch(e=>{c.parentElement.insertAdjacentHTML('afterend',`<p>${esc(e.message)}</p>`)}));}
async function inspect(index){active=Math.max(0,Math.min(filtered.length-1,index));const s=filtered[active];if(!s)return;const st=stackMap.get(s.stack),mode=$('mode').value,token=++detailToken;
 $('detailTitle').textContent=`${s.key} · Hinge stack ${st.index}/16 · ${st.time}`;const why=currentReason(s);
 $('detailReason').innerHTML=`<span class="badge ${why}">${labels[why]}</span> · Main 1× plot: <b>${labels[s.reason[mode]]}</b> · 2× plot: <b>${labels[s.reason_2x[mode]]}</b>`;
 $('detailPosition').textContent=`${active+1} / ${filtered.length}`;$('prevSpot').disabled=active===0;$('nextSpot').disabled=active===filtered.length-1;
 const n=s.nearest_2x[mode];$('detailStats').innerHTML=`<p class="small">Centre: x=${s.x.toFixed(2)}, y=${s.y.toFixed(2)}. Red squares: fixed 18 × 18 extraction windows. Cyan cross: matched 2× detection, if present. ${s.edge_padding?'Black padding extends beyond the recorded field.':''}<br>Nearest detected 2× centre: ${n?n.distance.toFixed(2)+' sensor pixels away':'none'}. One-to-one matching radius: 6 pixels.<br>X = (I0 − I90)/(I0 + I90); Y = (I45 − I135)/(I45 + I135); r = √(X² + Y²). Model maximum r = ${DATA.model.r_max.toFixed(4)}.</p><table><thead><tr><th>Quantity</th><th>1× BG</th><th>2× BG</th></tr></thead><tbody>${anisotropyRows(s)}<tr><td>Theta</td><td>${degree(s.theta_1x)}</td><td>${degree(s.theta_2x)}</td></tr><tr><td>Frames with valid X/Y</td><td>${s.anisotropy_frames['1x'].valid} / 150</td><td>${s.anisotropy_frames['2x'].valid} / 150</td></tr><tr><td>Frames outside theta-model radius</td><td>${s.anisotropy_frames['1x'].outside_theta_model}</td><td>${s.anisotropy_frames['2x'].outside_theta_model}</td></tr>${['I0','I90','I45','I135'].map((v,i)=>`<tr><td>${v} · aperture/frame mean DN</td><td>${s.channels_1x[i].toFixed(2)}</td><td>${s.channels_2x[i].toFixed(2)}</td></tr>`).join('')}<tr><td>Raw peak across all 150 frames</td><td colspan="2">${s.raw_peak} / 4095${s.ceiling_veto?' · conservative ceiling flag':''}</td></tr></tbody></table>`;
 $('detailNote').textContent=`File: ${st.name}. Note: ${st.note||'None'}. Background: ${st.background}.`;
 for(const id of ['detailCanvas','detailAnisotropy','overviewImage']){const c=$(id);c.getContext('2d').clearRect(0,0,c.width,c.height)}
 if(!$('detail').open)$('detail').showModal();
 await drawAnisotropy($('detailAnisotropy'),s,token,true);if(token!==detailToken)return;
 const im=await imload(st.sprite);if(token!==detailToken)return;const c=$('detailCanvas'),ctx=c.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(im,s.tile[0],s.tile[1],264,88,0,0,1056,352);mark(ctx,s,4);
 const stamp=st.name.match(/-(\d{6})/)[1];const full=await imload(`../matched_background_theta/images/${stamp}_1x.png`);if(token!==detailToken)return;
 const oc=$('overviewImage'),ox=oc.getContext('2d');ox.drawImage(full,0,0,oc.width,oc.height);const sx=oc.width/full.width,sy=oc.height/full.height;ox.strokeStyle='#ff536d';ox.lineWidth=2;ox.strokeRect(s.crop_left*sx,s.crop_top*sy,88*sx,88*sy);}
for(const id of ['mode','reason','stack','pageSize'])$(id).onchange=()=>{if($('detail').open)$('detail').close();render(true)};
$('overlay').onchange=()=>{render();if($('detail').open)inspect(active).catch(showError)};
function turnPage(delta){page+=delta;render();$('counts').scrollIntoView({block:'center'})}
for(const id of ['prevPage','bottomPrev'])$(id).onclick=()=>turnPage(-1);for(const id of ['nextPage','bottomNext'])$(id).onclick=()=>turnPage(1);
function showError(e){$('detailNote').textContent=e.message}
$('grid').onclick=e=>{const b=e.target.closest('[data-open]');if(b)inspect(Number(b.dataset.open)).catch(showError)};
$('close').onclick=()=>{$('detail').close();++detailToken};$('prevSpot').onclick=()=>inspect(active-1).catch(showError);$('nextSpot').onclick=()=>inspect(active+1).catch(showError);
$('detail').addEventListener('keydown',e=>{if(e.key==='ArrowLeft'&&active>0){e.preventDefault();inspect(active-1).catch(showError)}if(e.key==='ArrowRight'&&active<filtered.length-1){e.preventDefault();inspect(active+1).catch(showError)}});
$('download').onclick=()=>{const mode=$('mode').value,header=['candidate','stack','x','y','main_1x_reason','2x_reason','theta_1x','theta_2x','raw_peak','X_1x','Y_1x','r_1x','X_2x','Y_2x','r_2x'];const rows=filtered.map(s=>[s.key,s.stack,s.x,s.y,s.reason[mode],s.reason_2x[mode],s.theta_1x??'',s.theta_2x??'',s.raw_peak,...s.xy_1x,s.r_1x,...s.xy_2x,s.r_2x]);const csv=[header,...rows].map(r=>r.join(',')).join('\r\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`hinge_exclusions_${mode}_${$('reason').value}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
render(true);
</script></body></html>'''


if __name__ == '__main__':
    build()
