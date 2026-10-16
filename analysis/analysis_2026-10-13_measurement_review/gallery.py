"""Read-only catalogue of the October 12 run and operator curation history.

Run with the repository Python. Only first measurement frames and small raw-peak
companions are read. Deleted payloads are listed separately. Outputs stay local.
"""
from pathlib import Path, PureWindowsPath
from collections import Counter
import hashlib
import json
import re
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / 'data/hinges/polarcam'
DAY = ROOT / '2026-10-12'
ASSETS = HERE / 'previews'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def basename(path):
    return PureWindowsPath(str(path or '')).name


def clock_name(path):
    match = re.search(r'20261012-(\d{6})', str(path))
    return match.group(1) if match else '999999'


def preview(frame, key):
    result = {}
    for gain in (1, 20):
        output = ASSETS / f'{key}_{gain}x.webp'
        if not output.exists():
            scaled = np.clip(np.nan_to_num(np.asarray(frame, dtype=np.float32))*gain/4095*255, 0, 255)
            Image.fromarray(np.rint(scaled).astype(np.uint8)).save(output, lossless=True, method=3)
        result[str(gain)] = output.relative_to(HERE).as_posix()
    return result


def phase(time, kind):
    if time < '183650':
        return 'Setup / cancellation checks', 'DNA noted; setup', 'Check only'
    if time <= '185850':
        return 'Hinges — wide-field survey', 'Hinge', ('Extra angle check' if time == '185850' else 'Population candidate')
    if time < '191903':
        return 'Gold — first wide-field survey', 'Gold-only control', 'Review: illumination'
    if time < '194141':
        return 'Gold — repeat wide-field survey', 'Gold-only control', 'Review: illumination / oil'
    if time < '205025':
        return 'Gold — spot selection / wide–narrow switching', 'Gold-only control', 'Targeted / repeated fields'
    return 'DNA return — spot selection / wide–narrow switching', 'DNA from notes/sequence', 'Targeted / role conflict'


def build():
    ASSETS.mkdir(parents=True, exist_ok=True)
    previous = read_json(HERE/'manifest.json')
    cleanup = read_json(DAY/'cleanup_log.json')
    catalogue_only = {r['name'] for r in cleanup.get('analysis_usage_review', {}).get('catalogue_only_frame_stacks', [])}
    previous_records = previous.get('records', []) + previous.get('deleted_records', [])
    previous_by_name = {r['name']: r for r in previous_records}
    all_paths = sorted(p for p in DAY.rglob('*') if p.is_file())
    inventory = {p.relative_to(ROOT).as_posix(): {'path': p.relative_to(ROOT).as_posix(),
                 'url': p.as_uri(), 'bytes': p.stat().st_size, 'owners': []} for p in all_paths}
    records, deleted_records, issues = [], [], []
    profiles = sorted((DAY/'backgrounds').glob('background_min*.npy'))
    profile_hashes = {sha(p): p for p in profiles}
    used = sorted((DAY/'backgrounds/used').glob('*.npy'))
    bg_audit = []
    for p in used:
        digest = sha(p)
        bg_audit.append({'hash': digest, 'filename_hash_matches': digest == p.stem,
                         'source': profile_hashes.get(digest).name if digest in profile_hashes else None,
                         'url': p.as_uri(), 'used_by': []})
    bg_by_hash = {row['hash']: row for row in bg_audit}
    primary = [(p.with_suffix('.npy'), 'Frame stack') for p in sorted((DAY/'frames').glob('frame_stack*.json'))]
    primary += [(p, 'Spot recording') for p in sorted((DAY/'spots').glob('*.npy'))]
    primary += [(p, 'Stationary check') for p in sorted((DAY/'stationary').rglob('capture*.npy')) if 'diagnostics' not in p.parts]
    primary += [(p.with_suffix('.npy'), 'Background') for p in sorted((DAY/'backgrounds').glob('background_min*.json'))]
    primary.sort(key=lambda item: (clock_name(item[0]), str(item[0])))
    for index, (path, kind) in enumerate(primary, 1):
        metadata = read_json(path.with_suffix('.json'))
        conditions = metadata.get('conditions', {})
        actual = metadata.get('actual', {})
        output = metadata.get('fetch_frames_output', {})
        background = metadata.get('background', {})
        previous_record = previous_by_name.get(path.name, {})
        legacy_path = 'hinges/polarcam/' + path.relative_to(ROOT).as_posix()
        key = previous_record.get('id') or hashlib.sha256(legacy_path.encode()).hexdigest()[:14]
        if not path.exists():
            review = metadata.get('review', {})
            if review.get('data_status') not in {'payload_deleted', 'payload_and_raw_peak_deleted', 'payload_and_capture_stack_deleted'}:
                raise FileNotFoundError(path)
            if not previous_record:
                raise ValueError(f'Missing original gallery record for deleted payload: {path.name}')
            rec = dict(previous_record)
            relative = path.with_suffix('.json').relative_to(ROOT).as_posix()
            inventory[relative]['owners'].append(key)
            rec.update(data_available=False, review=review,
                       analysis_usage='historical',
                       sample=review.get('sample', rec['sample']),
                       purpose='Deleted payload; metadata and cached preview retained',
                       decision='Deleted at operator request on 2026-10-16', raw_verified=False,
                       files=[{'name': path.with_suffix('.json').name, 'path': relative,
                               'url': path.with_suffix('.json').as_uri()}],
                       flags=review.get('notes', []))
            deleted_records.append(rec)
            continue
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        first = np.asarray(a[0] if a.ndim == 3 else a)
        time = clock_name(path)
        phase_name, sample, purpose = phase(time, kind)
        flags = []
        note = conditions.get('condition_note', '')
        role = conditions.get('recording_role', 'Unknown')
        if time >= '205025' and role == 'Gold-only control':
            flags.append('Role says Gold-only control; DNA in notes / return sequence. Do not group by role alone.')
        if kind == 'Frame stack' and '190833' <= time <= '193127':
            flags.append('Review illumination: later notes report the light left on. No automatic exclusion.')
        if '194141' <= time and kind == 'Frame stack':
            flags.append('Spot selection / wide–narrow switching; may revisit a field. Not automatically an independent population sample.')
        if '201224' <= time <= '201417':
            flags.append('Note raises possible repeat of control 1.')
        digest = background.get('background_profile_sha256') or output.get('background_profile_sha256')
        subtracted = background.get('background_subtracted', output.get('background_subtracted'))
        bg_valid = None
        if subtracted:
            bg_valid = bool(digest in bg_by_hash and bg_by_hash[digest]['filename_hash_matches'])
            if bg_valid:
                bg_by_hash[digest]['used_by'].append(key)
            else:
                flags.append('Missing or unverifiable used-background hash.')
                issues.append(f'{path.name}: background not verified')
        files = [path, path.with_suffix('.json')]
        source_stack = metadata.get('source_stack_file')
        if source_stack and (path.parent/source_stack).exists():
            files.append(path.parent/source_stack)
        if kind == 'Stationary check':
            files.extend([path.with_name(path.stem+'_meta.json'), path.parent/'meta.json'])
        stop = path.with_suffix('.stop')
        if stop.exists():
            files.append(stop)
        raw = output.get('raw_peak') or actual.get('raw_peak')
        raw_verified, peak_info = False, {}
        if raw:
            raw_path = path.parent/raw['path']
            if raw_path.exists():
                files.append(raw_path)
                peak = np.load(raw_path, allow_pickle=False)
                expected_frames = actual.get('frames', output.get('count'))
                raw_verified = (sha(raw_path) == raw.get('sha256') and tuple(peak.shape) == tuple(first.shape)
                                and raw.get('frames') == expected_frames)
                if raw_verified:
                    peak_info = {'max': int(np.max(peak)), 'near_limit_pixels': int(np.sum(peak >= 4095*250/255)),
                                 'pixels_ge_4094': int(np.sum(peak >= 4094)), 'pixels_ge_4095': int(np.sum(peak >= 4095))}
                else:
                    issues.append(f'{path.name}: raw peak failed verification')
            else:
                issues.append(f'{path.name}: raw peak missing')
        if peak_info.get('pixels_ge_4094', 0):
            flags.append('Raw peak reaches 4094: near the observed ceiling. Review spot-level saturation; do not reject the entire field from this alone.')
        num_frames = (int(a.shape[0]) - int(bool(actual.get('phase_marker_appended')))) if a.ndim == 3 else 1
        if a.ndim == 3 and actual.get('frames') is not None and num_frames != actual['frames']:
            flags.append('Frame count disagrees with metadata.')
            issues.append(f'{path.name}: frame count mismatch')
        urls = []
        for source in files:
            if source.exists():
                relative = source.relative_to(ROOT).as_posix()
                inventory[relative]['owners'].append(key)
                urls.append({'name': source.name, 'url': source.as_uri(), 'path': relative})
        timing = actual.get('timing') or actual.get('timing_snapshot') or output.get('timing') or {}
        fps = actual.get('fps') or output.get('actual_fps')
        exp = timing.get('exposure_us')
        gains = actual.get('gains') or actual.get('gains_snapshot') or output.get('gains') or {}
        gain = (gains.get('analog') or {}).get('val')
        rec = {'id': key, 'index': index, 'type': kind, 'name': path.name,
               'path': path.relative_to(ROOT).as_posix(), 'time': ':'.join(time[i:i+2] for i in (0,2,4)),
               'stamp': time, 'phase': phase_name, 'sample': sample, 'purpose': purpose,
               'role': role, 'note': note, 'drive': conditions.get('interference_drive', 'Unknown'),
               'shape': list(a.shape), 'frames': num_frames, 'fps': fps, 'exposure_ms': exp/1000 if exp else None,
               'gain': gain, 'duration': num_frames/fps if fps and a.ndim == 3 else None,
               'subtracted': subtracted, 'bg_source': basename(background.get('background_profile_source_path')),
               'bg_hash': digest, 'bg_verified': bg_valid, 'raw_verified': raw_verified, 'raw_peak': peak_info,
               'raw_max_metadata': actual.get('max_raw_value'), 'preview': preview(first, key),
               'first_frame_mean': float(first.mean()), 'first_frame_max': int(first.max()),
               'flags': flags, 'source_frame_stack': basename(metadata.get('source_frame_stack')),
               'files': urls, 'schema': metadata.get('schema_version'), 'roi': metadata.get('roi', {}),
               'created_local': metadata.get('created_local'), 'marker': actual.get('phase_marker_appended'),
               'decision': 'Kept — not yet selected for statistics'}
        review = metadata.get('review', {})
        rec.update(review=review, data_available=True)
        rec['analysis_usage'] = 'catalogue_only' if path.name in catalogue_only else 'analysed'
        if review:
            rec['sample'] = review.get('sample', rec['sample'])
            rec['flags'] = [f for f in rec['flags'] if not f.startswith(('Review illumination:', 'Role says Gold-only control;'))]
            rec['flags'].extend(review.get('notes', []))
            if review.get('illumination_geometry'):
                rec['flags'].insert(0, 'Acquisition: ' + review['illumination_geometry'].replace('_', ' ') +
                                    '; background: ' + review.get('subtracted_background_geometry', 'not applied').replace('_', ' ') + '.')
        # Show the selected spot as well as the camera's wider minimum-width ROI.
        if kind in ('Spot recording', 'Stationary check'):
            roi = metadata.get('roi', {})
            cx, cy = roi.get('cx'), roi.get('cy')
            if cx is not None and cy is not None:
                x = float(cx)-float(roi.get('x', roi.get('OffsetX', 0)))
                y = float(cy)-float(roi.get('y', roi.get('OffsetY', 0)))
                size = int(roi.get('win_raw', 20)); half = size//2
                left = min(max(0, int(round(x))-half), max(0, first.shape[1]-size))
                top = min(max(0, int(round(y))-half), max(0, first.shape[0]-size))
                rec['spot_preview'] = preview(first[top:top+size, left:left+size], key+'_spot')
        records.append(rec)
        del a
        if index % 20 == 0:
            print(f'Previewed {index}/{len(primary)}', flush=True)
    active = read_json(ROOT/'background_profile.json')
    active_path = ROOT/'background_profile.npy'
    active_digest = sha(active_path) if active_path.exists() else None
    active_info = {'present': active_path.exists(), 'source': basename(active.get('background_profile_source_path')),
                   'hash': active_digest, 'sidecar_matches': bool(active_digest and active_digest == active.get('background_profile_sha256')),
                   'dated_source_matches': active_digest in profile_hashes,
                   'url': active_path.as_uri() if active_path.exists() else None}
    linked_ids = {rec['name']: rec['id'] for rec in records + deleted_records}
    for rec in records + deleted_records:
        rec['source_frame_id'] = linked_ids.get(rec['source_frame_stack'])
    # Runtime files/caches are retained and explicitly listed, not hidden as observations.
    extras = [item for item in inventory.values() if not item['owners']]
    for item in extras:
        name = item['path']
        item['category'] = ('Active background' if name.startswith('background_profile') else
                            'Used-background cache' if '/backgrounds/used/' in name else 'Unassociated file — review')
    summary = {'files': len(inventory), 'bytes': sum(row['bytes'] for row in inventory.values()),
               'types': dict(Counter(rec['type'] for rec in records)), 'phases': dict(Counter(rec['phase'] for rec in records)),
               'subtracted_recordings': sum(rec['subtracted'] is True for rec in records),
               'verified_backgrounds': sum(rec['bg_verified'] is True for rec in records),
               'verified_raw_peaks': sum(rec['raw_verified'] for rec in records),
               'deleted_frame_stacks': sum(r['type']=='Frame stack' for r in deleted_records),
               'deleted_backgrounds': sum(r['type']=='Background' for r in deleted_records),
               'schemas': dict(Counter(str(rec['schema']) for rec in records)), 'issues': issues}
    manifest = {'summary': summary, 'active_background': active_info, 'backgrounds': bg_audit,
                'records': records, 'deleted_records': deleted_records,
                'cleanup': cleanup,
                'files': list(inventory.values()), 'extras': extras,
                'performance': read_json(HERE/'performance.json')}
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    html = HTML.replace('__DATA__', json.dumps(manifest, ensure_ascii=False).replace('</', '<\\/'))
    (HERE/'index.html').write_text(html, encoding='utf-8')
    print(json.dumps(summary, indent=2)); print('ACTIVE', json.dumps(active_info)); print('EXTRAS',len(extras))


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>12 October — complete recording gallery</title>
<style>
:root{font:15px/1.5 system-ui,sans-serif;color:#253438;background:#f5f6f3}*{box-sizing:border-box}body{margin:0}main,header{max-width:1400px;margin:auto;padding:24px}h1{font-size:32px;margin:5px 0}h2{font-size:22px}p{max-width:1000px}a{color:#14687a}button,select,input{font:inherit;border:1px solid #aebcb9;border-radius:5px;padding:7px 11px;background:white;color:inherit}button{cursor:pointer}button:hover{background:#e2f1ed}label{display:flex;gap:7px;align-items:center}.muted{color:#60716f}.box{padding:18px 22px;background:white;border:1px solid #d8e0db;border-radius:9px;margin:15px 0}.stats{display:flex;flex-wrap:wrap;gap:12px}.stat{padding:12px 22px;background:#e0ebe5;border-radius:6px}.stat b{font-size:25px;display:block}.toolbar{position:sticky;top:0;z-index:3;background:#f5f6f3f7;border-block:1px solid #d1dcd6;padding:13px 24px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}.toolbar input{width:280px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{background:white;border:1px solid #d2dcd5;border-radius:8px;padding:18px;scroll-margin-top:120px}.title{display:flex;justify-content:space-between;gap:12px}.title strong{font-size:21px}.filename{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}.tags{display:flex;gap:5px;flex-wrap:wrap;margin:8px 0}.tag{font-size:12px;padding:2px 8px;background:#e3eee9;border-radius:10px}.tag.warn{background:#fff0c9;color:#795212}.preview{display:block;width:100%;height:270px;object-fit:contain;background:#111;cursor:zoom-in}.spot-preview{image-rendering:pixelated;width:110px;height:110px;object-fit:contain;background:#111;float:right;margin:8px}.note{white-space:pre-wrap;background:#f3f5f0;padding:12px;margin:12px 0;border-left:3px solid #829d8e;overflow-wrap:anywhere}.flags{font-size:13px;color:#7a521c;padding-left:20px}.details{font-size:13px}.links{display:flex;flex-direction:column;gap:4px;overflow-wrap:anywhere}details{margin:8px 0}summary{cursor:pointer}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border-bottom:1px solid #dde3df;padding:8px;text-align:left;overflow-wrap:anywhere}.table-scroll{overflow:auto}.phase{margin:30px 0 10px;scroll-margin-top:120px}.empty{padding:60px;text-align:center}dialog{width:min(96vw,1500px);max-height:95vh;border:0;border-radius:8px;padding:15px}dialog::backdrop{background:#142126dd}.dialogbar{position:sticky;top:0;background:white;display:flex;gap:10px;align-items:center;flex-wrap:wrap;z-index:2}.zoomarea{max-height:76vh;overflow:auto;background:#111;text-align:center}.zoomarea img{max-width:100%;image-rendering:pixelated}.zoomarea.native img{max-width:none}#lighttitle{flex:1;font-size:13px;overflow-wrap:anywhere}.foot{margin-top:40px;font-size:13px}.legend{font-size:13px}@media(max-width:850px){.grid{grid-template-columns:1fr}.toolbar{position:static}header,main{padding:16px}.preview{height:240px}.title strong{font-size:18px}}
</style>
<header><div class="muted">POLARCAM · 12 OCTOBER 2026 · READ-ONLY REVIEW</div><h1>Every recording, in order</h1>
<p>First measurement frame at <b>20× display gain</b>, with the original notes and settings. Current files are shown by default. The 33 frame stacks and two backgrounds removed on 16 October remain available as metadata and cached previews under Deleted. Use Analysis → Only previewed to see the 11 retained gold frame stacks without substantive analysis results. Statistical results are not recalculated by this catalogue.</p>
<div id="stats" class="stats"></div>
<details class="box" open><summary><b>What needs attention</b></summary><ul>
<li>The first hinge wide-field series has 15 planned fields plus one extra angle check.</li>
<li>The two initial brightness-test stacks, first 10 gold wide-field stacks and all 21 narrow-field frame stacks were removed at the operator’s request. The 14 repeat gold wide-field stacks, all 16 early DNA survey stacks, and later wide-field fields remain. The operator clarified that the light stayed on throughout.</li>
<li>Later frames accompany selected spot recordings and wide/narrow switching. Their background identity does not prove which illumination was on. Repeated fields must not be counted as independent samples.</li>
<li>From 20:50:25 the notes identify DNA, but the role dropdown remains “Gold-only control”. Both are shown; the DNA label follows notes and sequence.</li>
<li>Several raw maximum images reach 4094. That is near the observed ceiling, while the code's nominal clipping threshold is 4095. Review individual spots before statistics; a field-wide maximum is not a reason to discard every spot.</li></ul></details>
<details class="box"><summary><b>Backgrounds, folder structure and live-view review</b></summary><div id="audit"></div></details>
<p class="legend">Black = 0; white = 204.75 saved units at 20×, or 4095 at 1×. No extra subtraction, per-image normalization or spot filtering. These are first-frame previews, not full-stack saturation analyses. Original notes are quoted verbatim; review notes are shown separately; timestamps come from acquisition filenames. Camera-reported FPS does not measure delivered-frame loss.</p>
</header>
<div class="toolbar"><label>Files <select id="availability"><option value="kept">Retained</option><option value="deleted">Deleted (metadata only)</option><option value="all">All</option></select></label><label>Show <select id="type"><option value="">All recording types</option></select></label><label>Analysis <select id="usage"><option value="">All</option><option value="catalogue_only">Only previewed</option><option value="analysed">Analysed</option></select></label><label>Sequence <select id="phase"><option value="">All phases</option></select></label><input id="search" placeholder="Search notes, filenames, background…" aria-label="Search recordings"><label>Display <select id="gain"><option value="20">20×</option><option value="1">1×</option></select></label><span id="count" class="muted"></span></div>
<main><div id="gallery"></div><details class="box"><summary id="inventoryTitle"></summary><p>Every source file is accounted for. JSON sidecars, raw-peak diagnostics, stop flags and background source stacks are attached to their observation. Active/used profiles are runtime/cache files, not extra measurements.</p><input id="fileSearch" placeholder="Filter source files" aria-label="Filter source files"><div class="table-scroll"><table><thead><tr><th>File</th><th>Size</th><th>Association</th></tr></thead><tbody id="inventory"></tbody></table></div></details><div class="foot">Generated locally from the copied run. Original acquisition fields are preserved; later clarifications are in each sidecar’s review section. <a href="manifest.json">Download catalogue and audit (JSON)</a></div></main>
<dialog id="viewer"><div class="dialogbar"><b id="lighttitle"></b><button id="previous">← Previous</button><button id="next">Next →</button><button id="native">Fit / sensor pixels</button><button id="close">Close</button></div><p id="lightnote"></p><div class="zoomarea" id="zoomarea"><img id="large" alt="Recording preview"></div></dialog>
<script>
const DATA=__DATA__;
const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,n=2)=>v===null||v===undefined?'Unknown':Number(v).toFixed(n),yes=v=>v===true?'on':v===false?'off':'unknown';
const records=[...DATA.records,...(DATA.deleted_records||[])].sort((a,b)=>a.index-b.index);let visible=records,selected=0;
$('stats').innerHTML=Object.entries(DATA.summary.types).map(([k,v])=>`<div class="stat"><b>${v}</b>${esc(k)}${v===1?'':'s'}</div>`).join('');
for(const value of Object.keys(DATA.summary.types))$('type').add(new Option(value,value));
for(const value of new Set(records.map(r=>r.phase)))$('phase').add(new Option(value,value));
const p=DATA.performance.milliseconds||{},b=DATA.active_background;
$('audit').innerHTML=`<p><b>Active profile:</b> the redundant root background_profile.npy/json pair was removed on 16 October. Its exact contents remain in the dated 20:52:23 background and the hash-named used-background copy. Eight background captures and their source stacks remain. The obsolete 19:07:29 and 19:19:03 sets were subsequently removed at the operator’s request; their earlier diagnostic plots are historical results.</p><p><b>Capture provenance:</b> ${DATA.summary.verified_backgrounds}/${DATA.summary.subtracted_recordings} subtracted observations have a matching saved used-background hash; ${DATA.summary.verified_raw_peaks} raw-peak companions verified. All ${records.length} observation sidecars use schema 2. These fields are consistent with the newer saving layout, but no Git revision was recorded, so they cannot prove the exact running version.</p><p><b>Folders:</b> the run now lives directly at <code>data/hinges/polarcam/2026-10-12/</code>. Original lab-PC paths in acquisition metadata are historical provenance; local source links below use the current location. The 33 removed frame stacks retain their JSON sidecars; all 20 spot recordings and three interference checks remain. <a href="../../data/hinges/polarcam/2026-10-12/cleanup_log.json">Cleanup log</a>.</p><p><b>Live view:</b> the lag fix moves subtraction/conversion out of every frame callback and processes only the newest queued preview, uses an exact uint16 lookup table, and reduces temporary arrays. The camera buffer is copied before delivery. The approach is sound; SDK event delivery, tracking and rendering remain possible costs. ${DATA.performance.unit_tests||136} camera-free tests passed. A local ${DATA.performance.iterations||12}-iteration benchmark on a full-size recording gave preview median ${num(p.old_preview?.median)} → ${num(p.new_preview?.median)} ms; background plus preview ${num(p.old_bg_preview?.median)} → ${num(p.new_bg_preview?.median)} ms, with identical pixels. This excludes the camera, event dispatch and drawing; it is not a hardware latency guarantee.</p><p><b>Integrity issues:</b> ${DATA.summary.issues.length?esc(DATA.summary.issues.join('; ')):'None in the checked headers, sidecars and companion hashes.'}</p><div class="table-scroll"><table><thead><tr><th>Background ID</th><th>Matching dated profile</th><th>Recordings</th></tr></thead><tbody>${DATA.backgrounds.map(row=>`<tr><td><a href="${esc(row.url)}">${row.hash.slice(0,12)}</a></td><td>${esc(row.source||'Unmatched')}</td><td>${row.used_by.length}</td></tr>`).join('')}</tbody></table></div>`;
function card(r){return `<article class="card" id="${r.id}"><div class="title"><strong>${r.time}</strong><span>${esc(r.type)} · #${r.index}</span></div><div class="filename">${esc(r.name)}</div><div class="tags"><span class="tag">${esc(r.sample)}</span>${r.analysis_usage==='catalogue_only'?'<span class="tag warn">Only previewed · not analysed</span>':''}${r.data_available===false?'<span class="tag warn">Deleted payload · cached preview</span>':''}<span class="tag ${r.purpose.includes('Review')||r.purpose.includes('conflict')?'warn':''}">${esc(r.purpose)}</span></div><img class="preview" loading="lazy" src="${r.preview[$('gain').value]}" alt="${esc(r.name)} first frame" data-view="${r.id}"><div class="muted details">${r.shape.at(-1)} × ${r.shape.at(-2)} sensor pixels · ${$('gain').value}× display · click to enlarge</div>${r.spot_preview?`<img class="spot-preview" loading="lazy" src="${r.spot_preview[$('gain').value]}" alt="Selected spot crop" title="Selected ROI crop at nearest-pixel enlargement">`:''}<div class="note">${r.note?esc(r.note):'<i>No recorded note.</i>'}</div><div class="details"><b>Recorded role:</b> ${esc(r.role)}<br><b>Drive declaration:</b> ${esc(r.drive)} · <b>Saved subtraction:</b> ${yes(r.subtracted)}<br>${num(r.fps)} fps · ${num(r.exposure_ms,4)} ms · analogue gain ${num(r.gain,1)}<br>${r.frames.toLocaleString()} measurement frame${r.frames===1?'':'s'}${r.type!=='Background'?` · ${num(r.duration)} s at reported FPS`:''}<br><b>Background:</b> ${esc(r.bg_source||(r.subtracted?'Unknown':'not applied'))}${r.bg_hash?`<br>ID ${r.bg_hash.slice(0,12)} · ${r.bg_verified?'hash verified':'UNVERIFIED'}`:''}<br><b>Raw maximum:</b> ${r.raw_peak.max??r.raw_max_metadata??'unknown'}${r.raw_verified?` · raw companion verified; ${r.raw_peak.near_limit_pixels.toLocaleString()} sensor pixels ever ≥250/255`:''}</div>${r.flags.length?`<ul class="flags">${r.flags.map(f=>`<li>${esc(f)}</li>`).join('')}</ul>`:''}${r.source_frame_id?`<p class="details"><a href="#${r.source_frame_id}" data-source="${r.source_frame_id}">Source frame stack: ${esc(r.source_frame_stack)}</a></p>`:''}<details><summary>Source files (${r.files.length}) and details</summary><div class="links">${r.files.map(f=>`<a href="${esc(f.url)}">${esc(f.name)}</a>`).join('')}</div><p class="details">Metadata written ${esc(r.created_local)}. Schema ${r.schema}. Phase marker appended: ${yes(r.marker)}. ROI: ${esc(JSON.stringify(r.roi))}.</p></details></article>`}
function render(){const search=$('search').value.toLowerCase();visible=records.filter(r=>($('availability').value==='all'||(r.data_available!==false)===($('availability').value==='kept'))&&(!$('usage').value||r.analysis_usage===$('usage').value)&&(!$('type').value||r.type===$('type').value)&&(!$('phase').value||r.phase===$('phase').value)&&(!search||JSON.stringify([r.name,r.note,r.role,r.sample,r.bg_source,r.bg_hash,...r.flags]).toLowerCase().includes(search)));$('count').textContent=`${visible.length} / ${records.length} observations`;let last='',html='';for(const r of visible){if(r.phase!==last){if(last)html+='</div>';html+=`<h2 class="phase">${esc(r.phase)}</h2><div class="grid">`;last=r.phase}html+=card(r)}if(last)html+='</div>';$('gallery').innerHTML=html||'<div class="empty">No matching recordings.</div>';}
for(const id of ['type','phase','gain','availability','usage'])$(id).addEventListener('change',render);$('search').addEventListener('input',render);
function show(id){selected=visible.findIndex(r=>r.id===id);if(selected<0)return;const r=visible[selected];$('lighttitle').textContent=`${r.time} · ${r.name} · ${$('gain').value}×`;$('lightnote').textContent=r.note||'No recorded note.';$('large').src=r.preview[$('gain').value];$('previous').disabled=selected===0;$('next').disabled=selected===visible.length-1;if(!$('viewer').open)$('viewer').showModal()}
$('gallery').addEventListener('click',e=>{if(e.target.dataset.view)show(e.target.dataset.view);if(e.target.dataset.source){e.preventDefault();const id=e.target.dataset.source;if(!$(id)){$('availability').value='all';$('usage').value='';$('type').value='';$('phase').value='';$('search').value='';render()}$(id).scrollIntoView({behavior:'smooth'});}});
$('close').onclick=()=>$('viewer').close();$('previous').onclick=()=>show(visible[selected-1]?.id);$('next').onclick=()=>show(visible[selected+1]?.id);$('native').onclick=()=>$('zoomarea').classList.toggle('native');document.addEventListener('keydown',e=>{if(!$('viewer').open)return;if(e.key==='ArrowRight'&&selected<visible.length-1)show(visible[selected+1].id);if(e.key==='ArrowLeft'&&selected>0)show(visible[selected-1].id)});
$('inventoryTitle').textContent=`Complete source inventory — ${DATA.summary.files} files, ${(DATA.summary.bytes/1e9).toFixed(2)} GB`;
function files(){const q=$('fileSearch').value.toLowerCase();$('inventory').innerHTML=DATA.files.filter(f=>f.path.toLowerCase().includes(q)).map(f=>`<tr><td><a href="${esc(f.url)}">${esc(f.path)}</a></td><td>${(f.bytes/1e6).toFixed(2)} MB</td><td>${f.owners.length?f.owners.map(id=>{const r=records.find(r=>r.id===id);return `${r.time} ${esc(r.type)}`}).join(', '):esc(f.category||'Runtime/cache')}</td></tr>`).join('')}$('fileSearch').oninput=files;files();render();
</script></html>'''


if __name__ == '__main__':
    build()
