"""Plot existing calibration channel caches; no redetection or acquisition edits."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
records = {r['stamp']: r for r in json.loads((HERE / 'results.json').read_text(encoding='utf8'))['records']}


def load(stamp):
    with np.load(HERE / f'trace_{stamp}.npz') as cache:
        channels = cache['channels']
    record = records[stamp]
    assert len(channels) == record['metrics']['frames']
    assert not record['saturated']
    xy = np.column_stack(((channels[:, 0] - channels[:, 1]) / (channels[:, 0] + channels[:, 1]),
                          (channels[:, 2] - channels[:, 3]) / (channels[:, 2] + channels[:, 3])))
    assert np.isfinite(xy).all()
    return xy, record['metrics']['fps']


mixed, fps = load('183356')
off, _ = load('183447')
on, _ = load('183516')
cut, trim = round(5 * fps), round(.25 * fps)
pairs = [(mixed[:cut-trim], mixed[cut+trim:]), (off, on)]
titles = ['A · Within 18:33:56 recording\nOFF before 4.75 s; ON after 5.25 s (assumed)',
          'B · Separate recordings\n18:34:47 OFF; 18:35:16 ON (from notes)']
plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.8))
for ax, pair, title in zip(axes, pairs, titles):
    for points, color, label in zip(pair, ['#176b9a', '#db7027'], ['OFF', 'ON']):
        ax.scatter(points[:, 0], points[:, 1], s=4, alpha=.18, linewidths=0,
                   color=color, rasterized=True)
        ax.scatter([], [], s=26, color=color, label=f'{label}: {len(points):,} frames')
    a = np.linspace(0, 2 * np.pi, 600)
    ax.plot(.9276834494677721 * np.cos(a), .9276834494677721 * np.sin(a),
            ':', color='.55', lw=1, label='Model r limit ≈ 0.928')
    ax.axhline(0, color='.85', lw=.7, zorder=0)
    ax.axvline(0, color='.85', lw=.7, zorder=0)
    ax.set(xlim=(-.42, 1), ylim=(-.9, .52), aspect='equal', xlabel='Anisotropy X',
           ylabel='Anisotropy Y', title=title)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
fig.suptitle('Calibration: X–Y anisotropy, one point per camera exposure', fontsize=16)
fig.text(.5, .085, 'Same axis limits and scale • Original minimum-background subtraction • No temporal averaging',
         ha='center', fontsize=10)
fig.text(.5, .045, '1.194 ms exposure; 798.79 fps. OFF/ON comes from notes, not hardware readback.\n'
         'Left excludes ±0.25 s around the assumed switch. Manual z motion was not measured or matched.',
         ha='center', fontsize=10, color='.3')
fig.subplots_adjust(left=.07, right=.98, top=.85, bottom=.17, wspace=.18)
fig.savefig(HERE / 'calibration_xy.png', dpi=180, facecolor='white')
plt.close(fig)
print('Saved', HERE / 'calibration_xy.png')
