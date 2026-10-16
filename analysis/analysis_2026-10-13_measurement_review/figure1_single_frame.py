"""Frame-1 survey histograms: percentage of retained detections in fixed bins.

Uses saved independent frame-1 detections. No detector or extraction changes.
The isotropic reference is integrated over the same r or theta bins. It is not
divided out. Radius and angle curves are smoothed and peak-searched separately.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

A, B, C = .09779825985607313, .011480772916810558, .10137635007725046
R_LIMIT = C / (A + B)
MIN_PROMINENCE = .2  # percentage points per displayed bin
GROUPS = [('Early DNA', 'DNA', 16, '#166888'), ('Gold survey', 'Gold', 14, '#c16b2a')]


def theta_from_r(r):
    r = np.asarray(r)
    return np.degrees(np.arcsin(np.sqrt(np.clip(A*r/(C-B*r), 0, 1))))


def radius_from_theta(theta):
    sin2 = np.sin(np.deg2rad(theta))**2
    return C*sin2/(A+B*sin2)


def percentage_curve(values, edges, denominator, separation):
    """Exact percentages plus a Gaussian curve scaled to the display bin width."""
    width = edges[1]-edges[0]
    fine_edges = np.linspace(edges[0], edges[-1], (len(edges)-1)*20+1)
    dx = fine_edges[1]-fine_edges[0]
    centres = (fine_edges[:-1]+fine_edges[1:])/2
    counts = np.histogram(values, edges)[0]
    assert counts.sum() == len(values)
    histogram = counts/denominator*100
    fine = np.histogram(values, fine_edges)[0]/denominator*100*width/dx
    smooth = gaussian_filter1d(fine, width/dx, mode='reflect')
    np.testing.assert_allclose(smooth.sum()*dx/width, len(values)/denominator*100, atol=1e-10)
    indices, props = find_peaks(smooth, prominence=MIN_PROMINENCE, distance=round(separation/dx))
    peaks = [dict(value=float(centres[i]), percent_per_bin=float(smooth[i]), prominence=float(p))
             for i,p in zip(indices, props['prominences'])]
    return dict(bin_edges=edges.tolist(), counts=counts.tolist(), percent=histogram.tolist(),
                curve_x=centres.tolist(), curve_percent_per_bin=smooth.tolist(), peaks=peaks,
                smoothing_sigma=float(width), minimum_peak_separation=separation)


def build(here, out, detection_mode='minimum'):
    if detection_mode not in ('p95', 'minimum'):
        raise ValueError('detection_mode must be p95 or minimum')
    detection_label = 'p95' if detection_mode == 'p95' else 'minimum'
    source = here / 'late_hinges_single_frame/survey_reference.json'
    data = json.loads(source.read_text(encoding='utf8'))
    assert data['complete']
    redges = np.linspace(0, 1, 51)
    thetaedges = np.linspace(0, 90, 46)
    theta_at_redges = theta_from_r(np.clip(redges, 0, R_LIMIT))
    iso_r = np.diff(1-np.cos(np.deg2rad(theta_at_redges)))*100
    iso_theta = np.diff(1-np.cos(np.deg2rad(thetaedges)))*100
    np.testing.assert_allclose([iso_r.sum(), iso_theta.sum()], 100, atol=1e-5)
    summary = dict(source=str(source.relative_to(here)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        r_limit=R_LIMIT, model=dict(A=A, B=B, C=C), groups={},
        detection_background=detection_mode, extraction_background='minimum',
        normalization='100 * detections in bin / all retained detections in that sample group. Same denominator in both panels. Radius includes values above the model limit; these have no theta and leave a small missing percentage in the theta panel. No division by isotropic expectation.',
        display=dict(y_units='Percent of retained detections per bin', r_bin_width=.02, theta_bin_width_deg=2),
        isotropic=dict(r_percent_per_bin=iso_r.tolist(), theta_percent_per_bin=iso_theta.tolist(),
            method='Exact bin integrals of p(theta)=sin(theta) on 0-90 degrees, with r(theta) from the optical model. Curves represent an ideal unbiased isotropic population, not a detection-efficiency prediction.'),
        smoothing=dict(coordinates='r and theta independently', sigma_r=.02, sigma_theta_deg=2, subdivisions_per_display_bin=20, boundary='reflect'),
        peak_search=dict(minimum_prominence_percentage_points=MIN_PROMINENCE, separation_r=.08, separation_theta_deg=8,
            interpretation='Descriptive maxima of independently smoothed ordinary distributions. Peak coordinates generally do not map between panels; no significance test.'),
        detector=f'Frame 1, {detection_label} subtraction for detection, repeated aligned 2x2 intensity means; DoG sigma 1/6 sensor pixels; positive whole filtered mean+2SD; 8-connected components >=10 pixels, no maximum. 10-pixel image edge and fixed top-right ellipse excluded. Strongest DoG peak wins overlapping 18x18 apertures.',
        extraction='Original minimum-subtracted frame 1, four channel means in 18x18 aperture; annular water Fresnel/BFP model NA 0.39-1.30. No APD correction or fitted camera gain correction.',
        saturation='Any original raw pixel >=4094 during all 150 measurement frames in aperture OR component bounding rectangle excludes the candidate. Phase marker excluded. No low-background mask or 2x survival filter.')
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), layout='constrained')
    for group, label, expected, color in GROUPS:
        records = data['groups'][group]
        assert len(records) == expected
        per_field = []
        rows = []
        outside = []
        for rec in records:
            selected = [s for s in rec['modes'][detection_mode]['spots'] if s['selection'] == 'kept']
            assert all(not s['saturated'] and not s['inside_ellipse'] for s in selected)
            per_field.append(dict(file=rec['name'], retained=len(selected), valid=sum(s['theta_deg'] is not None for s in selected)))
            rows.extend(selected)
            outside.extend(dict(file=rec['name'], id=s['id'], r=s['r']) for s in selected if s['r'] > R_LIMIT)
        radii = np.array([s['r'] for s in rows])
        valid = np.isfinite(radii) & (radii >= 0) & (radii <= R_LIMIT)
        assert len(rows) == int(valid.sum()) + len(outside)
        theta = theta_from_r(radii[valid])
        np.testing.assert_allclose(theta, [s['theta_deg'] for s in rows if s['theta_deg'] is not None], atol=1e-11)
        r_curve = percentage_curve(radii, redges, len(rows), .08)
        theta_curve = percentage_curve(theta, thetaedges, len(rows), 8)
        peak_rows = [dict(r=p['value'], theta_deg=float(theta_from_r(p['value'])) if p['value'] <= R_LIMIT else None,
                         percent_per_bin=p['percent_per_bin'], prominence=p['prominence']) for p in r_curve['peaks']]
        theta_peaks = [dict(theta_deg=p['value'], r=float(radius_from_theta(p['value'])),
                           percent_per_bin=p['percent_per_bin'], prominence=p['prominence']) for p in theta_curve['peaks']]
        summary['groups'][label] = dict(stacks=len(records), retained=len(rows), valid=int(valid.sum()),
            above_limit=len(outside), above_limit_records=outside, per_field=per_field,
            near30_percent=float(np.sum((theta>=28)&(theta<34))/len(rows)*100),
            two_largest_field_share=float(sum(sorted(r['retained'] for r in per_field)[-2:])/len(rows)),
            peaks=peak_rows, theta_peaks=theta_peaks, r_histogram=r_curve, theta_histogram=theta_curve)
        for ax, curve in [(axes[0], r_curve), (axes[1], theta_curve)]:
            ax.stairs(curve['percent'], curve['bin_edges'], color=color, lw=.9, alpha=.4)
            ax.plot(curve['curve_x'], curve['curve_percent_per_bin'], color=color, lw=1.8,
                    label=f'{label}: {len(records)} stacks, n={len(rows):,}')
            for peak in curve['peaks']:
                x, y = peak['value'], peak['percent_per_bin']
                annotation = (f'r = {x:.3f}\nθ = {float(theta_from_r(x)):.1f}°' if ax is axes[0] and x <= R_LIMIT
                              else f'r = {x:.3f}' if ax is axes[0] else f'{x:.1f}°')
                ax.scatter(x, y, s=25, color=color, zorder=5)
                ax.annotate(annotation, (x, y), xytext=(0, 11), textcoords='offset points',
                            ha='center', va='bottom', color=color, fontsize=10)
    for ax, key, edges, isotropic in [(axes[0], 'r_histogram', redges, iso_r), (axes[1], 'theta_histogram', thetaedges, iso_theta)]:
        ax.stairs(isotropic, edges, color='.45', ls='--', lw=1.2, label='3-D isotropic')
        ymax = max(float(max(isotropic)), max(max(g[key]['percent']) for g in summary['groups'].values()))*1.35
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='white', framealpha=.96, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set(xlim=(0, 1), xlabel='Anisotropy radius r', ylabel='Retained detections per 0.02 bin (%)', title='A · Radius')
    axes[0].axvspan(R_LIMIT, 1, color='.94', zorder=-3)
    axes[0].axvline(R_LIMIT, color='.45', ls=':', lw=1)
    axes[0].text(R_LIMIT+.012, .48, f'r_lim = {R_LIMIT:.3f}', transform=axes[0].get_xaxis_transform(),
                 rotation=90, va='center', ha='left', fontsize=9, color='.35', bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1))
    axes[1].set(xlim=(0, 90), xlabel='Recovered θ from surface normal (°)', ylabel='Retained detections per 2° bin (%)', title='B · Polar angle')
    title = ('Figure 1 · Frame 1 · Minimum background · 2 SD' if detection_mode == 'minimum'
             else 'Historical comparison · p95 detection / minimum extraction')
    fig.suptitle(title, fontsize=15)
    image_name = 'figure_01_distributions.png' if detection_mode == 'minimum' else 'figure_01_p95_detection.png'
    results_name = 'figure_01_results.json' if detection_mode == 'minimum' else 'figure_01_p95_detection_results.json'
    fig.savefig(out / image_name, dpi=180, facecolor='white')
    plt.close(fig)
    (out / results_name).write_text(json.dumps(summary, indent=2), encoding='utf8')
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detection-bg', choices=['p95', 'minimum'], default='minimum')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    result = build(here, here / 'richard_email_20261014', detection_mode=args.detection_bg)
    print(json.dumps({k:{q:v[q] for q in ['stacks','retained','valid','above_limit','peaks']} for k,v in result['groups'].items()}, indent=2))
