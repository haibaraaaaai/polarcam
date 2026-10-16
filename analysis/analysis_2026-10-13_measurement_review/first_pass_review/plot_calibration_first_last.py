"""First versus last three seconds of the existing 18:33:56 channel cache."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
record = next(r for r in json.loads((HERE / 'results.json').read_text(encoding='utf8'))['records']
              if r['stamp'] == '183356')
fps = record['metrics']['fps']
with np.load(HERE / 'trace_183356.npz') as cache:
    channels = cache['channels']
assert len(channels) == record['metrics']['frames'] == 8000
assert not record['saturated']
duration = len(channels) / fps
# Frame centres relative to the nominal recording duration, inferred from FPS.
# The two windows do not depend on an assumed five-second switch.
times = (np.arange(len(channels)) + .5) / fps
first = times < 3
last = times >= duration - 3
assert first.sum() == last.sum() and not np.any(first & last)
xy = np.column_stack(((channels[:, 0] - channels[:, 1]) / (channels[:, 0] + channels[:, 1]),
                      (channels[:, 2] - channels[:, 3]) / (channels[:, 2] + channels[:, 3])))
assert np.isfinite(xy).all()
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(8.4, 8.5))
for mask, color, label in [(first, '#176b9a', 'First 3 s: 0.000–3.000 s'),
                            (last, '#db7027', f'Last 3 s: {duration-3:.3f}–{duration:.3f} s')]:
    points = xy[mask]
    ax.scatter(points[:, 0], points[:, 1], s=6, alpha=.24, linewidths=0, color=color)
    ax.scatter([], [], color=color, s=30, label=f'{label} (n={len(points):,})')
a = np.linspace(0, 2*np.pi, 600)
ax.plot(.9276834494677721*np.cos(a), .9276834494677721*np.sin(a), ':', color='.55',
        lw=1, label='Model r limit ≈ 0.928')
ax.axhline(0, color='.87', lw=.7, zorder=0)
ax.axvline(0, color='.87', lw=.7, zorder=0)
ax.set(xlim=(-.42, 1), ylim=(-.9, .52), aspect='equal', xlabel='Anisotropy X', ylabel='Anisotropy Y')
ax.legend(loc='upper left', frameon=False, fontsize=10)
fig.tight_layout(pad=.6)
out = HERE / 'calibration_xy_first_last_3s.png'
fig.savefig(out, dpi=180, facecolor='white', bbox_inches='tight', pad_inches=.05)
plt.close(fig)
print(json.dumps({'plot': str(out), 'duration_s': duration, 'first_frames': int(first.sum()),
                  'last_frames': int(last.sum()), 'last_window_start_s': duration-3,
                  'first_indices_1based': [int(np.flatnonzero(first)[0]+1), int(np.flatnonzero(first)[-1]+1)],
                  'last_indices_1based': [int(np.flatnonzero(last)[0]+1), int(np.flatnonzero(last)[-1]+1)]}, indent=2))
