"""Export selected five-panel examples and a single-exposure spot-recording gallery.

Reads acquisitions and validated channel caches; never modifies source data.
"""
from pathlib import Path
import base64
import gzip
import html
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
from first_pass_review import extract, xy, clean, sha, ROOT, RMAX

HERE = Path(__file__).resolve().parent
SELECTED = HERE / 'sd_spot_gallery/selected_examples'
RECORDINGS = HERE / 'spot_recording_gallery'
SCALE = 255 / 4095
COLORS = {'raw': '#7b838b', 'minimum': '#087f9c', 'median': '#ce782d', 'p95': '#8456a1'}
NAMES = {'raw': 'Raw proxy', 'minimum': 'Minimum BG', 'median': 'Median BG', 'p95': '95th-percentile BG'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'path.simplify': False})


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def unpack(p):
    return np.frombuffer(gzip.decompress(base64.b64decode(p['gzip'])), dtype=p['dtype']).reshape(p['shape'])


def cells(a):
    h, w = a.shape
    assert h % 2 == w % 2 == 0
    return a.reshape(h // 2, 2, w // 2, 2).mean((1, 3))


def guide(ax, limit=1.04):
    angle = np.linspace(0, 2*np.pi, 721)
    ax.plot(RMAX*np.cos(angle), RMAX*np.sin(angle), ':', color='.65', lw=1, zorder=0)
    ax.axhline(0, color='.9', lw=.6, zorder=0)
    ax.axvline(0, color='.9', lw=.6, zorder=0)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal', xlabel='Anisotropy X', ylabel='Anisotropy Y')


def selected_examples():
    SELECTED.mkdir(exist_ok=True)
    entries = []
    for spot_id in ['H12-SD2-C1086', 'H01-SD2-C1159']:
        source = HERE / 'sd_spot_gallery/spots' / (spot_id + '.js')
        text = source.read_text(encoding='utf8')
        row = json.loads(text[len('window.receiveSpot('):-2])
        saved = unpack(row['pixels'])
        profiles = unpack(row['backgrounds'])
        images = {'raw': saved[0].astype('float32') + profiles[0], 'minimum': saved[0]}
        for name, profile in zip(['median', 'p95'], profiles[1:]):
            images[name] = np.rint(np.maximum(saved[0].astype('float32') - (profile-profiles[0]), 0)).astype('uint16')
        fig = plt.figure(figsize=(17.5, 4.8))
        gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.35], left=.015, right=.985,
                              top=.81, bottom=.19, wspace=.11)
        for j, (key, im) in enumerate(images.items()):
            ax = fig.add_subplot(gs[j])
            ax.imshow(cells(im), cmap='gray', vmin=0, vmax=4095/20, interpolation='nearest',
                      extent=(-.5, 63.5, 63.5, -.5))
            l, r, t, b = row['local_bounds']
            ax.add_patch(Rectangle((l-.5, t-.5), r-l, b-t, fill=False, ec='#72d7d8', lw=1))
            ax.set_title(f'{chr(65+j)} · '+('Raw proxy' if key == 'raw' else NAMES[key]+' subtracted'), fontsize=10.5)
            ax.set_axis_off()
        ax = fig.add_subplot(gs[4])
        for key in images:
            points = np.asarray(row['traces'][key]['xy'], float)
            assert points.shape == (150, 2)
            ax.scatter(points[:, 0], points[:, 1], s=8, alpha=.42, linewidths=0, color=COLORS[key], label=NAMES[key])
            ax.scatter(*points[0], s=28, color=COLORS[key], edgecolors='#263238', linewidths=.65)
        for key in ['minimum', 'median', 'p95']:
            ax.scatter(*row['background_stats'][key]['xy'], marker='x', color=COLORS[key], s=30, linewidths=1.25)
        guide(ax)
        ax.tick_params(labelsize=8)
        ax.set_title('E · X/Y · all 150 exposures', fontsize=10.5)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .07), ncol=4, frameon=False,
                   markerscale=1.6, fontsize=9)
        fig.suptitle(f'{spot_id} · Frame 1 · 20× display gain', y=.98, fontsize=15)
        fig.text(.5, .90, f'{row["file"]}  |  Included at 2 SD; no matched component at 4 or 8 SD', ha='center', fontsize=10)
        assert row['pattern'] == '2' and all(row['detectors'][k]['status'] == 'unmatched' for k in ['4', '8'])
        fig.text(.5, .055, '64 × 64 sensor-pixel context; 2 × 2 cell means for display. Cyan: fixed 18 × 18 extraction window. '
                 'Outlined dots: frame 1. ×: background only. Dotted circle: r limit ≈ 0.928.', ha='center', fontsize=8)
        fig.text(.5, .018, 'Raw proxy adds back the recorded minimum background; exact pre-subtraction values at saved zeros are unrecoverable. '
                 'No temporal averaging in the anisotropy plot.', ha='center', fontsize=8, color='.35')
        fig.savefig(SELECTED / (spot_id+'.png'), dpi=220, facecolor='white')
        fig.savefig(SELECTED / (spot_id+'.svg'), facecolor='white')
        plt.close(fig)
        entries.append(dict(id=spot_id, frame=1, display_gain=20, source_sha256=sha(source),
                            file=row['file'], bounds=row['bounds'], background=row['background_file']))
        print('Selected example', spot_id, flush=True)
    (SELECTED / 'selection.json').write_text(json.dumps(entries, indent=2), encoding='utf8')
    cards = ''.join(f'<section><h2>{e["id"]}</h2><a href="{e["id"]}.png"><img src="{e["id"]}.png"></a>'
                    f'<p><a href="{e["id"]}.png" download>PNG</a> · <a href="{e["id"]}.svg" download>SVG</a> · '
                    f'<a href="../index.html#{e["id"]}">Original interactive view</a></p></section>' for e in entries)
    (SELECTED / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Selected spot examples</title>'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><style>body{font:16px system-ui;margin:24px;color:#26383d;background:#f5f6f6}'
        'section{background:white;padding:16px;margin:24px 0;border:1px solid #d7dfe0}img{width:100%;height:auto}a{color:#087f9c}</style>'
        '<h1>Two selected spots</h1><p>Frame 1 for the four pixel views; all 150 exposures for X/Y. Same 20× display gain and fixed extraction window.</p>'
        + cards + '<p><a href="../../spot_recording_gallery/index.html">Spot-recording gallery</a></p>', encoding='utf8')


def recording_figure(record, ch, crop):
    points = xy(ch)
    radius = np.linalg.norm(points, axis=1)
    intensity = ch.mean(1) * SCALE
    time = np.arange(len(ch)) / record['metrics']['fps']
    assert np.isfinite(points).all() and len(ch) == record['metrics']['frames']
    fig = plt.figure(figsize=(13.7, 8.1), layout='constrained')
    gs = fig.add_gridspec(3, 3, height_ratios=[1.7, 1, 1])
    # Both displays are the same single camera exposure; gain is purely visual.
    inner = gs[0, 0].subgridspec(1, 2, wspace=.05)
    for j, gain in enumerate([1, 20]):
        ax = fig.add_subplot(inner[0, j])
        ax.imshow(cells(crop), cmap='gray', vmin=0, vmax=4095/gain, interpolation='nearest')
        ax.set_title(('A · Frame 1\n' if j == 0 else 'Same frame\n')+f'{gain}× display', fontsize=10)
        ax.set_axis_off()
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(points[:, 0], points[:, 1], s=.8, alpha=.13, color='#147f9b', linewidths=0, rasterized=True)
    guide(ax)
    ax.set_title('B · X/Y · all exposures', fontsize=11)
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(points[:, 0], points[:, 1], s=.8, alpha=.15, color='#147f9b', linewidths=0, rasterized=True)
    low, high = points.min(0), points.max(0)
    center = (low+high)/2
    span = max(float((high-low).max())*1.12, .04)
    angle = np.linspace(0, 2*np.pi, 721)
    ax.plot(RMAX*np.cos(angle), RMAX*np.sin(angle), ':', color='.65', lw=1)
    ax.set(xlim=(center[0]-span/2, center[0]+span/2), ylim=(center[1]-span/2, center[1]+span/2),
           aspect='equal', xlabel='Anisotropy X', ylabel='Anisotropy Y', title='C · X/Y · close-up (all points)')
    ax.title.set_fontsize(10)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(time, intensity, color='#3b5964', lw=.35, alpha=.85)
    ax.set(xlim=(0, time[-1]), ylim=(0, max(1, intensity.max()*1.05)), ylabel='Mean intensity\n(0–255 equivalent)',
           title='D · Aperture intensity · individual exposures')
    ax.grid(axis='y', alpha=.15)
    ax = fig.add_subplot(gs[2, :])
    ax.plot(time, radius, color='#8255a0', lw=.35, alpha=.8)
    ax.axhline(RMAX, ls=':', color='.5', lw=1, label='Model r limit ≈ 0.928')
    ax.set(xlim=(0, time[-1]), ylim=(0, max(1.02, radius.max()*1.03)), xlabel='Time (s)', ylabel='Anisotropy radius r',
           title='E · Radius · individual exposures')
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.grid(axis='y', alpha=.15)
    fig.suptitle(f'{record["label"]} · {record["time"]} · {time[-1]:.1f} s · {len(ch):,} exposures'
                 + ('\nSATURATED in the selected aperture — diagnostic view only' if record['saturated'] else ''), fontsize=14)
    file = f'recording_{record["stamp"]}.png'
    fig.savefig(RECORDINGS/file, dpi=160, facecolor='white')
    plt.close(fig)
    return file, dict(r_median=float(np.median(radius)), r_min=float(radius.min()), r_max=float(radius.max()),
                      intensity_mean_8bit=float(intensity.mean()), x_sd=float(points[:, 0].std()), y_sd=float(points[:, 1].std()))


def recordings():
    RECORDINGS.mkdir(exist_ok=True)
    source = read(HERE/'manifest.json')
    previous = {r['stamp']: r for r in read(HERE/'first_pass_review/results.json')['records']}
    chosen = [r for r in source['records'] if r['type'] in ['Spot recording', 'Stationary check']]
    actual_files = set((ROOT/'2026-10-12/spots').glob('spotrec_*.npy'))
    assert actual_files == {ROOT/r['path'] for r in chosen if r['type'] == 'Spot recording'}
    rows = []
    calibration_i = 0
    for rec in chosen:
        result, ch, _ = extract(rec)  # validates source signatures, raw peak, phase and channel cache
        old = previous[rec['stamp']]
        assert result['provenance'] == old['provenance']
        if rec['type'] == 'Stationary check':
            calibration_i += 1
            result.update(group='Calibration', label=f'Calibration {calibration_i}')
        else:
            result.update(group='DNA' if old['group'] == 'Hinges' else 'Gold', label=old['label'])
        path = ROOT/rec['path']
        metadata = read(path.with_suffix('.json'))
        movie = np.load(path, mmap_mode='r')
        x, y, size = result['bounds']
        ox, oy = metadata['roi']['x'], metadata['roi']['y']
        l, t = x-ox, y-oy
        first = np.asarray(movie[0])
        crop = first[t:t+size, l:l+size]
        assert crop.shape == (20, 20)
        direct = np.array([crop[(pr-y)%2::2, (pc-x)%2::2].mean()
                           for pr, pc in [(1, 1), (0, 0), (0, 1), (1, 0)]])
        np.testing.assert_allclose(direct, ch[0], atol=1e-12)
        result['figure'], result['single_exposure_summary'] = recording_figure(result, ch, crop)
        result.update(first_frame=1, image_gain_options=[1, 20], background_subtracted=metadata['background']['background_subtracted'],
                      background_file=Path(metadata['background']['background_profile_source_path'].replace('\\', '/')).name,
                      reported_role=metadata['conditions']['recording_role'],
                      reported_drive=metadata['conditions']['interference_drive'],
                      channel_cache=f'../first_pass_review/trace_{rec["stamp"]}.npz',
                      source_file=path.name, last_sample_s=(len(ch)-1)/result['metrics']['fps'])
        # Full camera ROI context from the same first exposure, with extraction window.
        context = Image.fromarray(np.rint(np.clip(first*20./4095, 0, 1)*255).astype('uint8')).convert('RGB')
        context = context.resize((first.shape[1]*4, first.shape[0]*4), Image.Resampling.NEAREST)
        ImageDraw.Draw(context).rectangle((l*4, t*4, (l+size)*4-1, (t+size)*4-1), outline='#55e5da', width=2)
        result['context_image'] = f'context_{rec["stamp"]}.png'
        context.save(RECORDINGS/result['context_image'])
        rows.append(result)
        print(result['label'], rec['stamp'], len(ch), 'raw peak', result['raw_peak'], flush=True)
    assert len(rows) == 23 and sum(r['group'] == 'DNA' for r in rows) == sum(r['group'] == 'Gold' for r in rows) == 10
    info = dict(records=rows, methods='One selected 20x20 aperture as recorded. Original recorded minimum background subtraction. '
                'Channel means are computed separately for each exposure with the original sensor mosaic phase. '
                'X=(I0-I90)/(I0+I90); Y=(I45-I135)/(I45+I135); r=sqrt(X^2+Y^2). '
                'Intensity is the mean of the four channel means in ungained 0-255 equivalent units (DN*255/4095). '
                'Every exposure is plotted; no temporal averaging, smoothing or r-limit clipping. Appended marker frames are omitted. '
                'Time is frame index / recorded actual FPS, not hardware timestamps. '
                'Any raw pixel >=4094 in the selected aperture over the recording flags that whole record as saturated; retained here for inspection only.',
                sample_identity='The later DNA labels follow the recorded notes and confirmed acquisition sequence; the role dropdown remained Gold-only control. '
                'Control 3 may revisit Control 1 according to its note. Calibration ON/OFF is operator-reported, not hardware readback.',
                manifest_sha256=sha(HERE/'manifest.json'), builder_sha256=sha(Path(__file__)))
    (RECORDINGS/'results.json').write_text(json.dumps(clean(info), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    (RECORDINGS/'data.js').write_text('window.RECORDINGS='+json.dumps(clean(info), ensure_ascii=False, allow_nan=False)+';', encoding='utf8')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv)>1 else 'all'
    if mode in ['selected', 'all']:
        selected_examples()
    if mode in ['recordings', 'all']:
        recordings()
