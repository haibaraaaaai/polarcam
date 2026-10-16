"""Replot saved matched candidates in radius space, without a theta-validity cut."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'matched_background_theta'
MODES = ('native', 'fixed')
GROUPS = ('Hinges', 'Gold')
LABELS = {'Hinges': 'DNA / hinges', 'Gold': 'Gold control'}
COLORS = {'Hinges': '#147a8d', 'Gold': '#c07812'}
EDGES = np.arange(72, dtype=float) * .02  # 0 to 1.42, covers sqrt(2).


def main():
    path = OUT / 'results.json'
    source_bytes = path.read_bytes()
    source = json.loads(source_bytes)
    limit = source['model']['r_max']
    report = dict(source='results.json', source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  r_max=limit, bin_width=.02, bin_edges=EDGES.tolist(), groups={},
                  selection='Same 1x/2x one-to-one matching and raw-ceiling veto as the theta plots. '
                  'All finite radii retained; no theta-validity rejection and no radius clipping.',
                  weighting='One observation per candidate per stack; pooled by group. '
                  'These are not verified unique rods or independent sample preparations.')
    plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white',
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    for mode in MODES:
        report['groups'][mode] = {}
        for group in GROUPS:
            records = [r for r in source['records'] if r['group'] == group]
            matched = [s for r in records for s in r['spots'] if s['matched_'+mode]]
            eligible = [s for s in matched if not s['ceiling_veto']]
            entry = dict(stacks=len(records), matched=len(matched),
                         raw_ceiling_excluded=len(matched)-len(eligible), eligible=len(eligible))
            for suffix in ('1x', '2x'):
                values = []
                for s in eligible:
                    channels = np.array(s['channels_'+suffix], dtype=float)
                    den = channels[[0, 2]] + channels[[1, 3]]
                    if np.all(den > 0):
                        xy = (channels[[0, 2]]-channels[[1, 3]]) / den
                        radius = float(np.linalg.norm(xy))
                        np.testing.assert_allclose(xy, s['xy_'+suffix], atol=1e-12, rtol=1e-12)
                        np.testing.assert_allclose(radius, s['r_'+suffix], atol=1e-12, rtol=1e-12)
                        assert (s['theta_'+suffix] is None) == (radius > limit)
                        assert 0 <= radius <= np.sqrt(2)+1e-12
                        values.append(s['r_'+suffix])
                    else:
                        assert s['r_'+suffix] is None and s['theta_'+suffix] is None
                values = np.array(values)
                invalid_theta = sum(s['theta_'+suffix] is None for s in eligible)
                counts = np.histogram(values, EDGES)[0]
                assert counts.sum() == len(values)
                previous = source['theta_summary'][mode][group][suffix]
                assert invalid_theta == previous['invalid']
                assert len(eligible)-invalid_theta == previous['n']
                entry[suffix] = dict(n=len(values), undefined_r=len(eligible)-len(values),
                    invalid_theta=invalid_theta, restored=int(np.count_nonzero(values > limit)),
                    above_one=int(np.count_nonzero(values > 1)), median=float(np.median(values)),
                    minimum=float(values.min()), maximum=float(values.max()),
                    counts=counts.tolist(), percent=(counts/len(values)*100).tolist())
            report['groups'][mode][group] = entry
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.7), sharey=True)
        fig.subplots_adjust(left=.065, right=.98, bottom=.22, top=.79, wspace=.12)
        for ax, suffix, title in zip(axes, ('1x', '2x'),
                ('Original 1× background subtraction', '2× background subtraction · sensitivity only')):
            ax.axvspan(limit, EDGES[-1], color='#89999d', alpha=.09, zorder=0)
            ax.axvline(limit, color='#53646b', ls='--', lw=1.25)
            ax.axvline(1, color='#a4afb2', ls=':', lw=1)
            for group in GROUPS:
                s = report['groups'][mode][group][suffix]
                label = f'{LABELS[group]} · n={s["n"]:,}'
                ax.stairs(s['percent'], EDGES, color=COLORS[group], lw=2, label=label)
                ax.stairs(s['percent'], EDGES, color=COLORS[group], alpha=.07, fill=True)
            ax.set(title=title, xlabel='Anisotropy radius r = √(X² + Y²)', xlim=(0, EDGES[-1]),
                   xticks=np.arange(0, 1.41, .2), ylim=(0, 10))
            ax.grid(axis='y', alpha=.17)
            ax.legend(loc='upper left', frameon=False, fontsize=9)
            ax.text(1.14, 9.6, 'Outside theta model\n(values retained)', ha='center', va='top',
                    fontsize=9, color='#53646b')
            lines = ['Restored', f'DNA: {report["groups"][mode]["Hinges"][suffix]["restored"]:,}',
                     f'Gold: {report["groups"][mode]["Gold"][suffix]["restored"]:,}']
            ax.text(1.23, 7.8, '\n'.join(lines), fontsize=9, ha='center', va='top', color='#53646b')
        axes[0].set_ylabel('Selected observations per 0.02-r bin (%)')
        policy = 'GUI detector: threshold recalculated at each BG setting' if mode == 'native' else 'Original 1× absolute detector threshold held fixed'
        fig.suptitle('DNA and gold · radius distributions with invalid-theta observations restored\n'+policy,
                     fontsize=14, y=.97)
        fig.text(.5, .045,
            f'Dashed line: model r limit = {limit:.4f}. Same candidates in both panels; no clipping at the model limit or at r = 1.\n'
            '16 DNA stacks + 14 gold stacks · matched at 1×/2× · raw ceiling veto ≥4094 · one radius from 150-frame channel means.\n'
            'Percentages normalized within each group. Observations may include repeat measurements of the same rods.',
            ha='center', fontsize=9, color='#53646b')
        for ext in ('png', 'svg'):
            fig.savefig(OUT/f'radius_{mode}.{ext}', dpi=180)
        plt.close(fig)
    (OUT/'radius_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    page = HTML.replace('__DATA__', json.dumps(report, ensure_ascii=False, allow_nan=False).replace('</', '<\\/'))
    (OUT/'radius_comparison.html').write_text(page, encoding='utf-8')
    assert path.read_bytes() == source_bytes
    for mode, groups in report['groups'].items():
        print(mode, {g: {f: {k: v for k, v in entry[f].items() if k not in ('counts', 'percent')}
                           for f in ('1x', '2x')} for g, entry in groups.items()})
    print(OUT/'radius_comparison.html')


HTML = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DNA and gold: all finite anisotropy radii</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f7f3;color:#243b40;font:16px/1.5 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:29px}a{color:#0b7285}p{max-width:1120px}.small{font-size:13px;color:#526a70}.box{padding:16px;background:white;border:1px solid #d4dfd9;border-radius:8px;margin:20px 0}.plot{display:block;width:100%;height:auto;background:white}select{padding:7px;font:inherit;border:1px solid #a7bbb4;border-radius:5px}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #d4dfd9;text-align:left}th{font-size:14px}.scroll{overflow-x:auto}.links{margin-top:20px}
</style></head><body><main><div class="small">12 OCTOBER · 16 FIRST-ROUND DNA STACKS + 14 REPEAT GOLD SURVEY STACKS</div>
<h1>Radius distributions, including invalid theta</h1>
<p>Keep the same candidates detected at both 1× and 2× background subtraction, and keep the raw-ceiling exclusion. Remove only the theta-validity filter: <b>every finite radius is plotted, including r &gt; 0.9277 and r &gt; 1</b>.</p>
<label>Threshold policy <select id="mode"><option value="native">GUI: mean + 8 SD</option><option value="fixed">Hold original 1× threshold</option></select></label>
<img id="plot" class="plot" src="radius_native.png" alt="DNA and gold radius distributions with all finite values, at 1x and 2x background subtraction">
<div class="box scroll"><table><thead><tr><th>Group</th><th>BG</th><th>Radius observations</th><th>Theta exclusions restored</th><th>Undefined radius</th><th>r &gt; 1</th><th>Median r</th></tr></thead><tbody id="counts"></tbody></table></div>
<p><a href="../radius_detector_comparison/index.html">Compare detectors and turn the 2× survival filter on/off</a></p>
<p><b>At 1× BG:</b> removing the theta filter restores only six DNA observations and no gold observations. The difference between the groups is therefore not explained by invalid-theta rejection.</p>
<p><b>At 2× BG:</b> much of the DNA distribution extends above the theta-model limit. The radius plot exposes values that the 2× theta histogram could not show. Extra subtraction changes the channel ratios, so this right panel is a sensitivity check, not a correction established as more accurate.</p>
<details class="box"><summary>Calculation and interpretation</summary><p>X = (I0 − I90)/(I0 + I90), Y = (I45 − I135)/(I45 + I135), r = √(X² + Y²). Each I is the mean in the fixed 18 × 18 sensor-pixel window over 150 measured frames. It is one radius per observation, not one per frame.</p>
<p>The finite ratios each lie between −1 and +1 for nonnegative channel intensities, allowing a numerical radius up to √2. The current optical model supports theta inversion only up to r = 0.927683. Plotting a larger measured radius preserves the observation; it does not make it a valid angle under that model.</p>
<p>Both panels use the same candidate identities. A missing/undefined radius cannot be placed on the axis and is counted separately. There are none in these selected groups. There is no cross-stack identity deduplication, and the plotted counts do not establish independent preparation replicates.</p>
<p>Equal-width r bins and equal-width theta bins do not show identical density shapes because the radius-to-angle transformation is nonlinear. Each group is normalized by its own finite-radius count. Background processing and all saved source values are unchanged.</p></details>
<p class="links"><a href="index.html">Theta plots and spot detection</a> · <a href="../excluded_hinge_spots/index.html">Excluded hinge gallery</a> · <a id="png" href="radius_native.png">PNG</a> · <a id="svg" href="radius_native.svg">SVG</a> · <a href="radius_summary.json">Counts and provenance</a></p>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id),labels={Hinges:'DNA / hinges',Gold:'Gold control'};
function render(){const mode=$('mode').value;$('plot').src='radius_'+mode+'.png';$('png').href='radius_'+mode+'.png';$('svg').href='radius_'+mode+'.svg';
$('counts').innerHTML=['Hinges','Gold'].flatMap(g=>['1x','2x'].map(f=>{const s=DATA.groups[mode][g][f];return `<tr><td>${labels[g]}</td><td>${f.replace('x','×')}</td><td>${s.n.toLocaleString()}</td><td>${s.restored.toLocaleString()} (${(100*s.restored/s.n).toFixed(2)}%)</td><td>${s.undefined_r}</td><td>${s.above_one}</td><td>${s.median.toFixed(3)}</td></tr>`})).join('')}
$('mode').onchange=render;render();</script></main></body></html>'''

if __name__ == '__main__':
    main()
