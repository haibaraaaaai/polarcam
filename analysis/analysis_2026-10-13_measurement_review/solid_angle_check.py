"""Re-express existing pooled detections relative to 3-D isotropic orientations.

No spot, background, saturation, or extraction changes. Bin weights are exact
solid-angle integrals, rather than pointwise 1/sin(theta) at a bin centre.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'gold_intensity_angle'
A, B, C = .09779825985607313, .011480772916810558, .10137635007725046
RMAX = C / (A + B)
SOURCES = {
    'DNA': HERE / 'background_overlap_check/hinges95_theta.json',
    'Gold control': HERE / 'control_recovery_check/gold_results.json',
}
COLORS = {'DNA': '#116b90', 'Gold control': '#c26520'}


def theta(r):
    # Called only within the physical interval. Clip floating-point endpoints.
    assert np.all((r >= 0) & (r <= RMAX))
    return np.degrees(np.arcsin(np.sqrt(np.clip(A * r / (C - B * r), 0, 1))))


def mass_from_theta(edges):
    return -np.diff(np.cos(np.radians(edges)))


def main():
    OUT.mkdir(exist_ok=True)
    result = dict(model=dict(A=A, B=B, C=C, r_max=RMAX), groups={}, bins={},
        method='Observed probability in each bin divided by its probability under 3-D isotropy. Exact isotropic bin mass = cos(theta_lower) - cos(theta_upper).',
        scope='Existing p95/2-SD detections; minimum-background 18x18 extraction; original saturation, overlap and ellipse exclusions. Pooled spot-recording observations, not independent preparations.',
        limitation='Geometric normalization only, conditional on the angle model. It does not correct brightness-dependent detection, residual background, averaging, or angle error. Above-model radii remain in the raw r histogram and are not assigned solid-angle weights.')
    groups = {}
    for label, path in SOURCES.items():
        source = json.loads(path.read_text(encoding='utf8'))
        assert source['complete']
        spots = [s for rec in source['records'] for s in rec['spots'] if s['selection'] == 'kept']
        assert not any(s['saturated'] or s['inside_ellipse'] for s in spots)
        radius = np.array([s['r'] for s in spots], float)
        assert np.all(np.isfinite(radius)) and np.all(radius >= 0)
        valid = radius <= RMAX
        angles = theta(radius[valid])
        np.testing.assert_allclose(angles, [s['theta_deg'] for s, v in zip(spots, valid) if v], atol=1e-9)
        groups[label] = (radius, angles)
        result['groups'][label] = dict(source=str(path.relative_to(HERE)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), fields=len(source['records']),
            all_radius=len(radius), valid_theta=int(valid.sum()), above_model=int((~valid).sum()),
            largest_radius=float(radius.max()))

    theta_edges = np.arange(0, 92, 2.)
    physical_r_edges = np.r_[np.arange(0, RMAX, .02), RMAX]
    final_r = max(1., np.ceil(max(v[0].max() for v in groups.values()) / .02) * .02)
    raw_r_edges = np.r_[physical_r_edges, np.arange(.94, final_r + .021, .02)]
    for view, physical_edges in [('theta', theta_edges), ('r', physical_r_edges)]:
        mass = mass_from_theta(physical_edges if view == 'theta' else theta(physical_edges))
        assert np.all(mass > 0)
        np.testing.assert_allclose(mass.sum(), 1, atol=1e-12)
        # Uniform cos(theta), transformed to either axis, must give a flat ratio.
        trial_theta = np.degrees(np.arccos((np.arange(200000) + .5) / 200000))
        trial_s = np.sin(np.radians(trial_theta)) ** 2
        trial = trial_theta if view == 'theta' else C * trial_s / (A + B * trial_s)
        trial_counts = np.histogram(trial, physical_edges)[0]
        assert np.max(np.abs(trial_counts - len(trial) * mass)) < 2
        result['bins'][view] = dict(edges=physical_edges.tolist(), isotropic_mass=mass.tolist(), groups={})
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), layout='constrained')
        for label, (radius, angles) in groups.items():
            values = angles if view == 'theta' else radius[radius <= RMAX]
            counts = np.histogram(values, physical_edges)[0]
            assert counts.sum() == len(angles)
            probability = counts / len(values)
            relative = probability / mass
            np.testing.assert_allclose(np.sum(relative * mass), 1, atol=1e-12)
            raw_edges = theta_edges if view == 'theta' else raw_r_edges
            raw_values = angles if view == 'theta' else radius
            raw_counts = np.histogram(raw_values, raw_edges)[0]
            assert raw_counts.sum() == len(raw_values)
            raw_heights = raw_counts / len(raw_values)
            raw_heights = raw_heights * 100 if view == 'theta' else raw_heights / np.diff(raw_edges)
            axes[0].stairs(raw_heights, raw_edges, color=COLORS[label], lw=1.8,
                           label=f'{label} (n={len(raw_values):,})')
            axes[1].stairs(relative, physical_edges, color=COLORS[label], lw=1.8, label=label)
            result['bins'][view]['groups'][label] = dict(counts=counts.tolist(), relative_density=relative.tolist(),
                raw_edges=raw_edges.tolist(), raw_counts=raw_counts.tolist(),
                modal_bin=[float(physical_edges[relative.argmax()]), float(physical_edges[relative.argmax() + 1])])
        iso_raw = mass * 100 if view == 'theta' else mass / np.diff(physical_edges)
        axes[0].stairs(iso_raw, physical_edges, color='#777777', ls='--', lw=1.4, label='3-D isotropic reference')
        axes[1].plot([0, physical_edges[-1]], [1, 1], color='#777777', ls='--', lw=1.4, label='3-D isotropic = 1')
        axes[0].set_title('Ordinary histogram')
        axes[1].set_title('Same observations / solid-angle expectation')
        axes[0].set_ylabel('% of valid angles per 2°' if view == 'theta' else 'Probability density in r')
        axes[1].set_ylabel('Density relative to 3-D isotropic (1 = flat)')
        for ax in axes:
            ax.set_xlabel('Recovered θ from surface normal (degrees)' if view == 'theta' else 'Anisotropy radius r')
            ax.set_xlim(0, 90 if view == 'theta' else raw_r_edges[-1])
            ax.set_ylim(bottom=0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(frameon=False, fontsize=8.5, loc='upper right')
            if view == 'theta':
                ax.axvspan(28, 34, color='#e6b34c', alpha=.14, zorder=-1)
            else:
                ax.axvspan(RMAX, raw_r_edges[-1], color='#dddddd', alpha=.6, zorder=-1)
                ax.axvline(RMAX, color='#888888', lw=.8)
        if view == 'r':
            axes[0].legend(frameon=False, fontsize=8.5, loc='upper left')
        fig.suptitle('Geometry changes density, not the inferred angle or the spot selection', fontsize=13)
        fig.savefig(OUT / f'solid_angle_{view}.png', dpi=160)
        plt.close(fig)
    (OUT / 'solid_angle_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    (OUT / 'solid_angle.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Solid angle comparison</title>
<style>body{max-width:1280px;margin:30px auto;padding:0 20px;font:17px/1.6 system-ui;color:#24333c;background:#fafaf8}h1{font-size:25px}img{width:100%;background:white}select{font:inherit}p{max-width:1050px}a{color:#116b90}</style>
<h1>What changes when we account for solid angle?</h1>
<p><b>The near-30° feature becomes more prominent relative to larger angles.</b> A 3-D isotropic distribution contains more orientations near 90° than near 0°. Dividing by that geometric expectation raises the relative weight at low angles. This is not evidence that the feature is a hinge.</p>
<label>Horizontal axis <select id="axis" onchange="document.getElementById('plot').src='solid_angle_'+this.value+'.png'"><option value="theta">θ (2° bins)</option><option value="r">r (approximately 0.02 bins)</option></select></label>
<img id="plot" src="solid_angle_theta.png" alt="Ordinary histogram and solid-angle-normalized distribution for DNA and gold control">
<p><b>Same selections:</b> 16 DNA fields (9,558 radii; 9,553 valid angles) and 14 gold fields (2,720 radii; 2,719 valid angles). Detection uses p95 background and 2 SD; angle extraction uses minimum background. Oversized components remain eligible; original saturation, overlap and corner-mask exclusions remain.</p>
<p>The five DNA and one gold radii above r<sub>max</sub> = 0.92768 remain in the raw r histogram. There is no physical θ or solid-angle weight for them under this model, so the normalized view uses valid angles only. The gray r region is outside the model.</p>
<p><b>Exact bin correction:</b> divide each observed bin fraction by cos(θ<sub>lower</sub>) − cos(θ<sub>upper</sub>). This makes isotropy equal to 1, including at the first bin; it avoids dividing by sin(0). For r, transform the bin edges through the existing angle model first. A histogram in cos θ would also be flat for isotropy.</p>
<p><b>Interpretation:</b> this corrects geometry only. It does not restore undetected dim rods or remove background bias. The DNA/gold ratio at the same angle is unchanged by applying the same geometric factor. Rods lying flat on a surface are not 3-D isotropic: they would have θ near 90°, whatever their in-plane direction. The pooled observations do not constitute independent sample preparations.</p>
<p><b>Flat but moving is different:</b> the current angle is inferred from channel intensities averaged across 150 frames. Under the ideal model, a flat rod whose in-plane angle φ varies gives r = r<sub>max</sub> |mean(exp(2iφ))| for constant brightness. Averaging can therefore produce a low apparent θ without a true out-of-plane tilt. Unresolved rods with different directions can have a similar effect. These are possible explanations to test in traces, not diagnoses from this histogram.</p>
<p><a href="index.html">Back to intensity and angle</a> · <a href="solid_angle_summary.json">Counts and provenance</a></p></html>''', encoding='utf8')
    print(json.dumps({'groups':result['groups'], 'corrected_theta_mode_bins':{
        name:row['modal_bin'] for name,row in result['bins']['theta']['groups'].items()}}, indent=2))


if __name__ == '__main__':
    main()
