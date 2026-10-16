"""Pool hinge 1–16 and the repeat gold survey after 1x/2x detection matching.

Read-only acquisition inputs. Reuses the preceding background comparison's
native GUI detector, threshold policies, matching and image inspector.
"""
from pathlib import Path
import argparse
import csv
import json
import subprocess
import time

import numpy as np
from PIL import Image
from polarcam.live import intensity_spots
from polarcam.optics import theta_model, theta_from_radius
import background_sensitivity as bg

HERE = Path(__file__).resolve().parent
OUT = HERE / 'matched_background_theta'
WIDTH = 18
THETA_BIN_WIDTH = 2
THETA_BIN_EDGES = np.arange(0, 90 + THETA_BIN_WIDTH, THETA_BIN_WIDTH)
RAW_CEILING = 4094  # Conservative one-code margin below nominal Mono12 4095.
MODEL = theta_model()
GROUPS = ('Hinges', 'Gold')
MODES = ('native', 'fixed')


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def angles(mean, bounds, origin):
    channels = intensity_spots.channel_means(mean, bounds, origin)
    xy = intensity_spots.anisotropy(channels)
    radius = np.linalg.norm(xy, axis=1)
    theta = np.degrees(theta_from_radius(radius, MODEL['A'], MODEL['B'], MODEL['C']))
    return channels, xy, radius, theta


def calculate():
    OUT.mkdir(exist_ok=True)
    (OUT / 'images').mkdir(exist_ok=True)
    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    old = json.loads((HERE / 'background_sensitivity/results.json').read_text(encoding='utf-8'))
    old_gold = {r['name']: r for r in old['records']}
    hinge = [r for r in manifest['records'] if r['type'] == 'Frame stack'
             and r['phase'] == 'Hinges — wide-field survey']
    gold = [r for r in manifest['records'] if r['name'] in old_gold]
    assert len(hinge) == 16 and len(gold) == 14
    assert hinge[0]['stamp'] == '183850' and hinge[-1]['stamp'] == '185850'
    records, backgrounds = [], {}
    timer = time.perf_counter()
    for group, selected in (('Hinges', hinge), ('Gold', gold)):
        for number, rec in enumerate(selected, 1):
            path = bg.DAY / 'frames' / rec['name']
            meta_path = path.with_suffix('.json')
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            before = path.stat()
            bghash = meta['background']['background_profile_sha256']
            assert meta['background']['background_subtracted']
            if bghash not in backgrounds:
                bp = bg.DAY / 'backgrounds/used' / f'{bghash}.npy'
                assert bg.digest(bp) == bghash
                original = bg.DAY / 'backgrounds' / rec['bg_source']
                assert bg.digest(original) == bghash
                background = np.load(bp, allow_pickle=False).astype(np.float32)
                assert np.isfinite(background).all() and (background >= 0).all()
                assert np.array_equal(background, np.rint(background))
                bgi = f'images/background_{bghash[:12]}.png'
                Image.fromarray(bg.display(background)).save(OUT / bgi, compress_level=3)
                backgrounds[bghash] = (background, bgi, original)
            background, bgimage, bgpath = backgrounds[bghash]
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            n = len(a) - int(meta['actual']['phase_marker_appended'])
            assert n == meta['actual']['frames'] == 150
            assert a.dtype == np.uint16 and a.shape[1:] == background.shape
            # Recompute only the endpoint means for gold; its intermediate
            # detections/previews are already saved in the preceding comparison.
            factors = bg.FACTORS if group == 'Hinges' else (1., 2.)
            additions = [(v-1)*background for v in factors]
            sums = np.zeros((len(factors), *background.shape), dtype=np.float64)
            zeros = np.zeros(len(factors), dtype=np.int64)
            work = np.empty_like(background)
            for frame in a[:n]:
                sums[0] += frame
                zeros[0] += np.count_nonzero(frame == 0)
                for j in range(1, len(factors)):
                    np.subtract(frame, additions[j], out=work)
                    np.maximum(work, 0, out=work)
                    sums[j] += work
                    zeros[j] += np.count_nonzero(work == 0)
            means = sums / n
            del sums, a
            assert np.all(np.diff(means, axis=0) <= 1e-6)
            r = dict(name=path.name, stamp=rec['stamp'], time=rec['time'], group=group,
                     group_index=number, group_size=len(selected), note=rec['note'],
                     url=path.as_uri(), shape=list(background.shape), frames=n,
                     background=bgpath.name, background_url=bgpath.as_uri(),
                     background_sha256=bghash, background_image=bgimage,
                     sidecar_sha256=bg.digest(meta_path), source_bytes=before.st_size,
                     source_mtime_ns=before.st_mtime_ns, source_max_raw=meta['actual']['max_raw_value'],
                     factors=[])
            base = None
            for j, factor in enumerate(factors):
                image = bg.detector.intensity_detection_map(means[j])
                work_map, mu, sd = bg.dog_stats(image)
                native = bg.find(work_map, bg.detector.DOG_K_STD)
                threshold = max(0., mu + bg.detector.DOG_K_STD * sd)
                if base is None:
                    base, base_threshold = native, threshold
                fixed = bg.find(work_map, (base_threshold-mu)/sd if sd else bg.detector.DOG_K_STD)
                if j == 0:
                    np.testing.assert_array_equal(native, fixed)
                variants = {mode: bg.compare(base, pts) for mode, pts in zip(MODES, (native, fixed))}
                for mode in MODES:
                    variants[mode]['threshold'] = threshold if mode == 'native' else base_threshold
                image_file = f'images/{rec["stamp"]}_{factor:g}x.png'
                if group == 'Hinges':
                    Image.fromarray(bg.display(image)).save(OUT / image_file, compress_level=3)
                else:
                    original = next(v for v in old_gold[path.name]['factors'] if v['factor'] == factor)
                    for mode in MODES:
                        np.testing.assert_array_equal(variants[mode]['points'], original[mode]['points'])
                        assert variants[mode]['baseline_ids'] == original[mode]['baseline_ids']
                    image_file = '../background_sensitivity/' + original['image']
                r['factors'].append(dict(factor=factor, image=image_file,
                    zero_fraction=float(zeros[j]/(n*background.size)),
                    mean_signal=float(means[j].mean()), dog_mean=mu, dog_sd=sd, **variants))
            if group == 'Gold':
                intermediate = []
                for v in old_gold[path.name]['factors'][1:3]:
                    intermediate.append({**v, 'image': '../background_sensitivity/' + v['image']})
                r['factors'][1:1] = intermediate
            # Hold extraction centres at the 1x detection positions in both
            # reconstructions, so centroid shifts do not change the aperture.
            bounds = [(int(round(x))-WIDTH//2, int(round(x))+WIDTH//2,
                       int(round(y))-WIDTH//2, int(round(y))+WIDTH//2) for x, y in base]
            for left, right, top, bottom in bounds:
                assert 0 <= left < right <= background.shape[1]
                assert 0 <= top < bottom <= background.shape[0]
            peakmeta = meta['fetch_frames_output']['raw_peak']
            peakpath = path.parent / peakmeta['path']
            assert peakmeta['frames'] == n and peakmeta['full_scale'] == 4095
            assert bg.digest(peakpath) == peakmeta['sha256']
            peak = np.load(peakpath, allow_pickle=False)
            assert peak.shape == background.shape
            rawmax = np.array([peak[t:b, l:rr].max() for l, rr, t, b in bounds])
            roi = meta['roi']
            origin = (int(roi['OffsetX']), int(roi['OffsetY']))
            values = [angles(means[j], bounds, origin) for j in (0, -1)]
            r['spots'] = []
            retained = {mode: {bid for bid in r['factors'][-1][mode]['baseline_ids'] if bid}
                        for mode in MODES}
            for i, (x, y) in enumerate(base):
                spot = dict(id=i+1, x=x, y=y, bounds=bounds[i], raw_peak=int(rawmax[i]),
                            ceiling_veto=bool(rawmax[i] >= RAW_CEILING))
                for mode in MODES:
                    spot['matched_' + mode] = i+1 in retained[mode]
                for suffix, val in zip(('1x', '2x'), values):
                    channels, xy, radius, theta = val
                    spot.update({f'channels_{suffix}': channels[i], f'xy_{suffix}': xy[i],
                                 f'r_{suffix}': radius[i], f'theta_{suffix}': theta[i]})
                r['spots'].append(clean(spot))
            r['raw_peak_sha256'] = peakmeta['sha256']
            assert path.stat().st_size == before.st_size and path.stat().st_mtime_ns == before.st_mtime_ns
            records.append(r)
            print(f'{len(records)}/30 {group} {number}: '
                  f'detections {[v["native"]["count"] for v in r["factors"]]}; '
                  f'matched native/fixed {[len(retained[m]) for m in MODES]}', flush=True)
    report = dict(records=records, factors=list(bg.FACTORS), model=MODEL,
        parameters={**old['parameters'], 'extraction_width_px': WIDTH,
                    'raw_ceiling_veto': RAW_CEILING, 'nominal_raw_full_scale': 4095},
        formula=old['formula'], detector_source=old['detector_source'],
        detector_sha256=bg.digest(Path(bg.detector.__file__)),
        optics_sha256=bg.digest(bg.REPO/'src/polarcam/optics.py'),
        channel_source_sha256=bg.digest(Path(intensity_spots.__file__)),
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=bg.REPO, encoding='utf-8').strip(),
        elapsed_seconds=time.perf_counter()-timer,
        selection='Hinge notes 1–16 (18:38:50–18:58:50); repeat gold survey 19:22:31–19:31:27, 14 saved files.',
        theta_method='Mean of 150 frames and 18x18 aperture per channel; invert mean intensities. Fixed 1x centre in both conditions.',
        weighting='One observation per matched candidate per stack, pooled by group; no cross-stack identity deduplication.',
        exclusions='Unmatched at 2x; any aperture raw peak >=4094; invalid theta separately at each subtraction. No additional local-contrast/angle selection.')
    (OUT/'results.json').write_text(json.dumps(clean(report), indent=2, allow_nan=False), encoding='utf-8')
    return report


def selected(records, group, mode, suffix):
    return np.array([s['theta_'+suffix] for r in records if r['group'] == group for s in r['spots']
                     if s['matched_'+mode] and not s['ceiling_veto'] and s['theta_'+suffix] is not None])


def summarize(records, mode, group):
    rr = [r for r in records if r['group'] == group]
    matched = [s for r in rr for s in r['spots'] if s['matched_'+mode]]
    usable = [s for s in matched if not s['ceiling_veto']]
    summary = dict(stacks=len(rr), detected_1x=sum(len(r['spots']) for r in rr), matched=len(matched),
                   ceiling_excluded=len(matched)-len(usable), unsaturated_matched=len(usable))
    for suffix in ('1x', '2x'):
        v = selected(rr, group, mode, suffix)
        summary[suffix] = dict(n=len(v), invalid=len(usable)-len(v), median=float(np.median(v)) if len(v) else None,
            fraction_35_45=float(np.mean((v >= 35) & (v < 45))) if len(v) else None,
            bin_counts=np.histogram(v, THETA_BIN_EDGES)[0].tolist())
    paired = [(s['theta_1x'], s['theta_2x']) for s in usable
              if s['theta_1x'] is not None and s['theta_2x'] is not None]
    summary['paired_n'] = len(paired)
    summary['paired_median_change'] = float(np.median(np.diff(paired, axis=1))) if paired else None
    return summary


def render(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rr = report['records']
    report['totals'] = {mode: [{key: sum(r['factors'][i][mode][key] for r in rr)
        for key in ('count', 'retained', 'lost', 'new')} for i in range(4)] for mode in MODES}
    report['theta_summary'] = {mode: {g: summarize(rr, mode, g) for g in GROUPS} for mode in MODES}
    report['theta_bin_edges_deg'] = THETA_BIN_EDGES.tolist()
    report['theta_bin_width_deg'] = THETA_BIN_WIDTH
    colors = dict(Hinges='#167989', Gold='#be7b17')
    edges = THETA_BIN_EDGES
    plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white', 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    for mode in MODES:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
        fig.subplots_adjust(left=.075, right=.985, bottom=.21, top=.77, wspace=.10)
        for ax, suffix, title in zip(axes, ('1x', '2x'), ('θ from original 1× BG data', 'θ after 2× subtraction · sensitivity only')):
            for group in GROUPS:
                v = selected(rr, group, mode, suffix)
                counts = np.histogram(v, edges)[0]
                percent = counts / len(v) * 100 if len(v) else counts
                invalid = report['theta_summary'][mode][group][suffix]['invalid']
                label = f'{group} · n={len(v)} ({invalid} invalid)'
                ax.stairs(percent, edges, color=colors[group], linewidth=2, label=label)
                ax.stairs(percent, edges, color=colors[group], alpha=.08, fill=True)
            ax.set(xlabel='θ to optical axis (degrees)', title=title, xlim=(0, 90), xticks=np.arange(0, 91, 15))
            ax.grid(axis='y', alpha=.18)
            ax.legend(frameon=False, fontsize=9)
        axes[0].set_ylabel(f'Selected observations per {THETA_BIN_WIDTH}° bin (%)')
        policy = 'GUI threshold recalculated: mean + 8 SD' if mode == 'native' else 'Original 1× absolute threshold held fixed'
        fig.suptitle('Candidates detected at BOTH 1× and 2× background\n' + policy, fontsize=14, y=.97)
        fig.text(.5, .025, 'Hinges: 16 stacks · Gold: 14 stacks · 18 × 18 pixels · 150-frame mean · raw ceiling veto ≥4094\n'
                 'Each panel excludes its invalid θ values; counts are spot observations, not verified unique rods.',
                 ha='center', fontsize=9, color='#526064')
        fig.savefig(OUT/f'theta_{mode}.png', dpi=180)
        fig.savefig(OUT/f'theta_{mode}.svg')
        plt.close(fig)
    # Per-stack proportions expose field variation without pretending that every
    # detected observation is an independent preparation replicate.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6), sharex=True, layout='constrained')
    for row, group in enumerate(GROUPS):
        for col, suffix in enumerate(('1x', '2x')):
            records = [r for r in rr if r['group'] == group]
            matrix = []
            for r in records:
                v = selected([r], group, 'native', suffix)
                c = np.histogram(v, edges)[0]
                matrix.append(c/len(v)*100 if len(v) else np.full(len(edges)-1, np.nan))
            ax = axes[row, col]
            im = ax.imshow(matrix, aspect='auto', origin='upper', extent=(0, 90, len(records)+.5, .5),
                           cmap='viridis', vmin=0, vmax=20)
            ax.set(title=f'{group} · θ from {suffix} BG', ylabel='Stack order', yticks=range(1,len(records)+1,2))
            if row == 1:
                ax.set_xlabel('θ to optical axis (degrees)')
            if group == 'Hinges':
                ax.axhline(9.5, color='white', ls='--', lw=1)
    fig.colorbar(im, ax=axes, label=f'% of valid selected observations in each stack / {THETA_BIN_WIDTH}° bin', shrink=.85)
    fig.suptitle('Field-to-field variation · GUI threshold · same 1×/2× matching rule\n'
                 'White dashed line: hinge background replaced after stack 9', fontsize=13)
    fig.savefig(OUT/'per_stack.png', dpi=150)
    plt.close(fig)
    rows = []
    for r in rr:
        for s in r['spots']:
            row = {key: r[key] for key in ('name', 'group', 'group_index', 'time', 'background', 'background_sha256')}
            row.update({k: v for k, v in s.items() if k not in ('bounds', 'channels_1x', 'channels_2x', 'xy_1x', 'xy_2x')})
            for suffix in ('1x','2x'):
                for channel, v in zip(('I0','I90','I45','I135'), s['channels_'+suffix]):
                    row[f'{channel}_{suffix}'] = v
            rows.append(row)
    with (OUT/'spots.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT/'results.json').write_text(json.dumps(clean(report), indent=2, allow_nan=False), encoding='utf-8')
    html = bg.HTML
    begin = html.index('<main>')
    end = html.index('<div class="controls">')
    html = html[:begin] + INTRO + html[end:]
    html = html.replace('Gold survey — background subtraction sensitivity', 'Hinges and gold — matched-spot theta')
    html = html.replace('Counts across all 14 stacks', 'Detection counts across the 30 analysed stacks')
    html = html.replace('`${i+1}/14 · ${r.time}`', '`${r.group} ${r.group_index}/${r.group_size} · ${r.time}`')
    html = html.replace("loadImage(DATA.background_image).then(im=>{bgImage=im;draw()});", '')
    html = html.replace("const loaded=await Promise.all(r.factors.map(v=>loadImage(v.image)));", 
        "const loaded=await Promise.all([...r.factors.map(v=>loadImage(v.image)),loadImage(r.background_image)]);")
    html = html.replace("images=loaded;if(base.length)",
        "images=loaded.slice(0,4);bgImage=loaded[4];$('bgfile').textContent=r.background;$('bgfile').href=r.background_url;$('bghash').textContent=r.background_sha256;if(base.length)")
    html = html.replace('${r.time}</button>', '${r.group} ${r.group_index} · ${r.time}</button>')
    html = html.replace('<div class="grid" id="crops"></div>',
                        '<p id="spotTheta" class="small"></p><div class="grid" id="crops"></div>')
    details_start = html.index('<details class="box"><summary>Plot counts')
    details_end = html.index('</details>', details_start) + len('</details>')
    html = html[:details_start] + DETAILS + html[details_end:]
    html = html.replace('<a href="candidates.csv">Candidate coordinates (CSV)</a> · <a href="example.png">Static example</a>',
                        '<a href="spots.csv">All baseline candidates and angles (CSV)</a>')
    html = html.replace("$('bgfile').textContent=DATA.background;$('bgfile').href=DATA.background_url;$('bghash').textContent=DATA.background_sha256;", '')
    html = html.replace('</script>', SUMMARY_JS + '</script>')
    html = html.replace('__DATA__', json.dumps(clean(report), ensure_ascii=False, allow_nan=False).replace('</', '<\\/'))
    (OUT/'index.html').write_text(html, encoding='utf-8')
    print('THETA SUMMARY', json.dumps(report['theta_summary'], indent=2), flush=True)
    print(OUT/'index.html', flush=True)


INTRO = '''<main><div class="muted">12 OCTOBER 2026 · HINGES 1–16 + REPEAT GOLD SURVEY</div>
<h1>Theta after keeping candidates detected at both 1× and 2× BG</h1>
<p><b>Pool the 16 first-round hinge stacks (18:38:50–18:58:50) and, separately, the 14 saved gold survey stacks (19:22:31–19:31:27).</b>
Gold here means the batch in the preceding background comparison, not earlier controls or later targeted wide/narrow recordings.</p>
<div class="box"><b>One controlled selection rule</b><p>Use the current GUI's <b>Intensity DoG detector, k = 8</b> at each background setting.
Keep only 1× candidates with a one-to-one 2× match within 6 sensor pixels, independently within each stack.
The 1× panel uses the original saved intensities; the 2× panel shows what extra subtraction does to those candidates' reconstructed angles.
Changing the threshold selector below updates both theta plots and the image inspector.</p>
<p><b>Reconstruction:</b> one theta from each candidate's four channel means over all 150 frames and a balanced <b>18 × 18 sensor-pixel window</b>.
The window stays at the 1× centre in both conditions. Water Fresnel + BFP + circular-hole model, NA 0.39–1.30; no APD correction matrix.
This is inversion of mean intensities, not the GUI's optional 3D Cartesian direction averaging.</p>
<p class="small">Use each recording's SHA-verified background: hinge stacks 1–9 use 18:36:50, stacks 10–16 use 18:50:32; gold uses 19:21:57.
Every saved frame already has 1× removed; compute <code>max(saved − (factor−1) × background, 0)</code> before averaging. The phase marker is excluded.</p></div>
<h2>Pooled distributions</h2>
<p><a href="radius_comparison.html">Compare radius distributions with invalid-theta observations restored</a></p><label>Threshold policy <select id="thetaMode"><option value="native">GUI: mean + 8 SD</option><option value="fixed">Hold original 1× threshold</option></select></label>
<p id="policy"></p><p id="invalidWarning" class="note"></p><img class="plot" id="thetaPlot" src="theta_native.png" alt="Pooled hinge and gold theta distributions, at 1x and 2x background">
<div class="tablebox"><table><thead><tr><th>Group</th><th>1× candidates</th><th>Matched at 2×</th><th>Raw ceiling excluded</th><th>Valid θ at 1×</th><th>Valid θ at 2×</th><th>Median θ: 1× → 2×</th></tr></thead><tbody id="thetaSummary"></tbody></table></div>
<p class="small">A spot is excluded if any pixel in its extraction window reaches <b>4094</b> in any of the 150 raw frames, using the verified raw-peak companion.
This is a conservative one-code margin below nominal 4095, consistent with the camera's repeated 4094 maxima; it does not prove clipping at 4094.
Invalid radii are excluded separately in each panel, never clipped to 90°. No extra local-contrast or preferred-angle filter is applied.</p>
<p class="small">These are pooled <b>spot observations</b>, not verified unique rods or independent preparations. No cross-stack deduplication is attempted.
Stack 16's note says it was added to check the angle distribution; it is included as requested. Surviving stronger subtraction does not prove a candidate is a rod.</p>
<details class="box"><summary>See field-to-field variation</summary><img class="plot" src="per_stack.png" alt="Per-stack theta distributions"><p class="small">Native GUI threshold, same matching and exclusions. Each row is normalized independently. Background changes after hinge stack 9.</p></details>
<h2>Inspect detections at 1×, 1.2×, 1.5× and 2× BG</h2>
<p>The overlays show all detector candidates, including raw-ceiling or invalid-angle candidates excluded from the histograms.
The detector uses the mean four-polarizer-cell intensity, Gaussian widths 1 and 6 sensor pixels, component area 10–250 pixels, and a 10-pixel excluded border.</p>
'''

DETAILS = '''<details class="box"><summary>Background and reproducibility details for the selected stack</summary>
<p class="small">Background: <a id="bgfile"></a><br>SHA-256 <code id="bghash"></code><br>Camera detector source SHA-256 <code id="dethash"></code></p>
<p class="small">The native threshold is recalculated as mean + 8 SD after each subtraction. The fixed option holds the original 1× absolute DoG threshold within each stack.
Spatial matching is not proof of rod identity; centroid shifts, splits or merges can affect it. Strong subtraction can remove real dim spots and distort surviving spots' channel ratios.</p>
<p class="small">θ = asin(sqrt(A r/(C − B r))), r = sqrt(X² + Y²); X = (I0 − I90)/(I0 + I90), Y = (I45 − I135)/(I45 + I135).
Theta is measured from the optical axis (0°), with 90° parallel to the coverslip; it is not automatically the hinge opening angle.
The JSON preserves optical coefficients, source hashes, extraction bounds, raw maxima, matching flags, channel values and invalid angles for every baseline candidate.</p></details>'''

SUMMARY_JS = '''
const oldTable=table;
table=function(){oldTable();const mode=$('mode').value,s=DATA.theta_summary[mode];
$('thetaMode').value=mode;
$('thetaPlot').src='theta_'+mode+'.png';
$('policy').textContent=(mode==='native'?'GUI threshold recalculated: mean + 8 SD.':'Original 1× absolute threshold held fixed within each stack.')+` Equal weight per valid selected observation; ${DATA.theta_bin_width_deg}° bins.`;
$('invalidWarning').textContent='Extra subtraction strongly affects angle validity: '+['Hinges','Gold'].map(g=>`${g}: ${s[g]['2x'].invalid}/${s[g].unsaturated_matched} (${(100*s[g]['2x'].invalid/s[g].unsaturated_matched).toFixed(1)}%) have invalid θ at 2×`).join('; ')+'. The right panel contains only the remaining valid values. Use the left panel to examine selection without extra subtraction changing the angles.';
$('thetaSummary').innerHTML=['Hinges','Gold'].map(g=>{const v=s[g];return `<tr><td>${g} (${v.stacks} stacks)</td><td>${v.detected_1x}</td><td>${v.matched}</td><td>${v.ceiling_excluded}</td><td>${v['1x'].n} (${v['1x'].invalid} invalid)</td><td>${v['2x'].n} (${v['2x'].invalid} invalid)</td><td>${v['1x'].median?.toFixed(1)??'—'}° → ${v['2x'].median?.toFixed(1)??'—'}°</td></tr>`}).join('');};table();
$('thetaMode').onchange=()=>{$('mode').value=$('thetaMode').value;table();draw()};
const oldDraw=draw;
draw=function(){oldDraw();const r=records[stackIndex],mode=$('mode').value,i=Number($('spot').value),s=r.spots[i];
if(!s){$('spotTheta').textContent='Choose a baseline candidate in the list to see its theta and selection status.';return;}
const status=!s['matched_'+mode]?'Excluded: no 2× match':s.ceiling_veto?'Excluded: raw ceiling veto':'Matched and below raw ceiling';
$('spotTheta').textContent=`Candidate ${s.id}: ${status}. θ at 1×: ${s.theta_1x===null?'invalid':s.theta_1x.toFixed(1)+'°'}; θ at 2×: ${s.theta_2x===null?'invalid':s.theta_2x.toFixed(1)+'°'}. Raw peak: ${s.raw_peak} / 4095.`;};
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    report = json.loads((OUT/'results.json').read_text(encoding='utf-8')) if args.render_only else calculate()
    render(report)
