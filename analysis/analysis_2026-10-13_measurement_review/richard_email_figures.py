"""Package an email from saved results and documented first-frame comparisons.

Acquisition files are read only. Figure 2 recomputes its threshold comparison on
frame 1; the other figures reuse their documented saved analyses.
"""
from pathlib import Path
import hashlib
import html
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from figure1_single_frame import build as build_single_frame_figure
from figure2_background import build as build_background_figure

HERE = Path(__file__).resolve().parent
OUT = HERE / 'richard_email_20261014'
OUT.mkdir(exist_ok=True)
SCALE = 255 / 4095
RMAX = .9276834494677721
COLORS = {'DNA': '#166888', 'Gold control': '#c16b2a'}
TH = np.arange(0, 92, 2.)
RR = np.arange(0, 1.021, .02)
INPUTS = {}


def read(name):
    path = HERE / name
    INPUTS[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf8'))


def vals(rows, key):
    return np.array([s.get(key) if s.get(key) is not None else np.nan for s in rows], float)


def pct(v, edges, denominator=None):
    v = np.asarray(v, float)
    good = v[np.isfinite(v)]
    return np.histogram(good, edges)[0] * 100 / (denominator if denominator is not None else len(good))


def finish(fig, filename, title):
    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.savefig(OUT / filename, dpi=180, facecolor='white')
    plt.close(fig)


def angle_ax(ax):
    ax.set(xlim=(0, 90), ylim=(0, None), xlabel='Recovered θ from surface normal (°)')
    ax.axvspan(28, 34, color='#e0b654', alpha=.17, zorder=-2)


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'normal', 'legend.frameon': False, 'legend.fontsize': 8.5})

figure1 = build_single_frame_figure(HERE, OUT)
INPUTS[figure1['source']] = figure1['source_sha256']
INPUTS['figure1_single_frame.py'] = hashlib.sha256((HERE/'figure1_single_frame.py').read_bytes()).hexdigest()

figure2 = build_background_figure(HERE, OUT)
INPUTS.update(figure2['inputs_sha256'])
INPUTS['figure2_background.py'] = figure2['builder_sha256']

intensity = read('gold_intensity_angle/results.json')
intensity_summary = read('gold_intensity_angle/summary.json')
spots = [s for rec in intensity['records'] for s in rec['spots']]
random = [s for rec in intensity['records'] for s in rec['random'] if not s['raw_saturated']]
away = [s for s in random if s['away']]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout='constrained')
for rows, name, color, ls in [(spots, 'Detected gold', '#c16b2a', '-'),
                             (random, 'Uniform windows', '#1e7095', '--'),
                             (away, 'Away from candidates', '#63499b', '-')]:
    vv = vals(rows, 'theta'); n = np.isfinite(vv).sum()
    axes[0].stairs(pct(vv, TH), TH, label=f'{name} (n={n:,})', color=color, ls=ls, lw=1.5)
for key, color in [('minimum', '#425967'), ('median', '#299d92'), ('p95', '#a87c25')]:
    rows = intensity['background_samples'][key]
    axes[1].stairs(pct(vals(rows, 'theta'), TH), TH, label={'minimum': 'Minimum', 'median': 'Median', 'p95': '95th percentile'}[key], color=color, lw=1.5)
for lo, hi, label, color in [(0, 50, '<3.1', '#899aaa'), (50, 200, '3.1–12.5', '#c16b2a'), (200, np.inf, '≥12.5', '#166888')]:
    rows = [s for s in spots if lo <= s['peak_mean'] < hi]
    vv = vals(rows, 'theta'); n = np.isfinite(vv).sum()
    axes[2].stairs(pct(vv, TH), TH, color=color, lw=1.5, label=f'{label} (n={n:,})')
for ax in axes:
    angle_ax(ax)
    ax.set_ylabel('% within each reference population / 2°')
    ax.legend(loc='upper right')
axes[0].set_title('A · Gold images: same 18×18 extraction')
axes[1].set_title('B · Recorded BG: same 5,000 windows')
axes[2].set_title('C · Gold by peak brightness\n8-bit equivalent / sensor pixel')
axes[1].set_xlabel('Background-equivalent θ (°)')
finish(fig, 'figure_03_background_references.png', 'Figure 3 · Low apparent angles also occur in image-window and background references')

snapshots = read('single_frame_angle_check/results.json')
snapshot_summary = read('single_frame_angle_check/summary.json')
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout='constrained')
for ax, (name, group) in zip(axes, snapshots['groups'].items()):
    for key, short, color, ls in [('150-frame mean', '150-frame mean', '#333333', '-'),
                                 ('First (frame 1)', 'Frame 1', '#166888', '-'),
                                 ('Middle (frame 75)', 'Frame 75', '#c16b2a', '--'),
                                 ('Last (frame 150)', 'Frame 150', '#9874ad', ':')]:
        vv = np.concatenate([np.array(r['measurements'][key]['theta'], float) for r in group['records']])
        ax.stairs(pct(vv, TH, len(vv)), TH, color=color, ls=ls, lw=1.6,
                  label=f'{short}: {snapshot_summary[name][key]["near30_percent_all_spots"]:.1f}% at 28–34°')
    ax.set(title=name, ylabel='% of all retained detections / 2°')
    angle_ax(ax)
    ax.legend(loc='upper right')
finish(fig, 'figure_04_single_frames.png', 'Figure 4 · The near-30° feature survives when angles use single frames')

late_summary = read('late_hinges_single_frame/summary.json')
late_background = read('late_hinges_single_frame/background_info.json')
spot_records = read('late_hinges_single_frame/spot_results.json')['records']
quiet_apertures = read('late_hinges_single_frame/quiet_apertures.json')
motion_summary = read('motion_selected_rods/repeated/summary.json')


def cached_image(name):
    path = HERE / name
    INPUTS[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return Image.open(path)


# Reuse the existing motion plot: do not rerun detection or quietly repair its
# known over-strict unmatched-frame gate while preparing the email.
fig, ax = plt.subplots(figsize=(12, 8.9), layout='constrained')
ax.imshow(cached_image('motion_selected_rods/repeated/theta_k1.png'))
ax.set_axis_off()
finish(fig, 'figure_05_motion_check.png',
       'Figure 5 · Moving-sample background captures: an exploratory selection check')

# Display all unsaturated target records alongside one explicitly selected
# quiet example. These intervals are cached frame-wise measurements, not CIs.
fig = plt.figure(figsize=(12.8, 12.2), layout='constrained')
gs = fig.add_gridspec(2, 2, height_ratios=[1.65, 1])
ax = fig.add_subplot(gs[0, :])
ax.imshow(cached_image('late_hinges_single_frame/spot_211300.png'))
ax.set_axis_off()
for ax, group, color in [(fig.add_subplot(gs[1, 0]), 'Gold', '#c16b2a'),
                         (fig.add_subplot(gs[1, 1]), 'DNA', '#166888')]:
    records = [r for r in spot_records if r['label'].startswith(group + ' ') and not r['saturated']]
    assert len(records) == {'Gold': 7, 'DNA': 9}[group], (group, len(records))
    for i, r in enumerate(records):
        low, median, high = r['single_frame']['theta_p2_p50_p98']
        ax.plot([low, high], [i, i], color=color, lw=2)
        ax.scatter([median], [i], color=color, s=22, zorder=3)
    ax.set(yticks=range(len(records)), yticklabels=[r['label'] for r in records],
           xlim=(0, 90), xlabel='Single-frame apparent θ (°)',
           title=f'{group}: all {len(records)} unsaturated target recordings\nMedian and 2nd–98th percentiles of frames')
    ax.invert_yaxis()
    ax.axvspan(28, 34, color='#e0b654', alpha=.17, zorder=-2)
finish(fig, 'figure_06_spot_recordings.png',
       'Figure 6 · Quiet low-angle DNA targets exist; attachment remains unverified')

checks = read('first_pass_review/results.json')['cancellation']
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), layout='constrained')
for ax, label, before, after in [(axes[0], 'A · Within 18:33:56 recording\nApproximate switch at 5 s', checks['mixed_pre']['8'], checks['mixed_post']['8']),
                               (axes[1], 'B · Separate recordings\n18:34:47 OFF → 18:35:16 ON', checks['off'], checks['on'])]:
    ratios = [after['intensity_cv']/before['intensity_cv'], * (np.array(after['xy_std'])/before['xy_std'])]
    bars = ax.bar(np.arange(3), ratios, color=['#5b7e91', '#bc8645', '#6e6997'], width=.6)
    ax.axhline(1, color='.35', lw=1, ls='--')
    for b, v in zip(bars, ratios): ax.text(b.get_x()+b.get_width()/2, v+.04, f'{v:.2f}×', ha='center')
    ax.set(xticks=np.arange(3), xticklabels=['Intensity CV', 'X standard\ndeviation', 'Y standard\ndeviation'],
           ylim=(0, 2.2), ylabel='ON / OFF variation ratio', title=label)
finish(fig, 'figure_07_drive_check.png', 'Figure 7 · Drive ON/OFF: intensity and anisotropy give mixed results')

dna1=figure1['groups']['DNA']; gold1=figure1['groups']['Gold']
bg_comparison=figure2['background_comparison']; sd_comparison=figure2['threshold_comparison']
fig1_numbers='; '.join(f'{g}: {d["retained"]:,} retained, {d["above_limit"]} above r_lim' for g,d in figure1['groups'].items())
fig1_peaks=' '.join(f'{g} r peaks: '+', '.join(f'{p["r"]:.3f} (inverted theta {p["theta_deg"]:.1f} degrees)' for p in d['peaks'])+
    '; theta peaks: '+', '.join(f'{p["theta_deg"]:.1f} degrees' for p in d['theta_peaks'])+'.' for g,d in figure1['groups'].items())
fig2_numbers=' '.join(f'{g}, minimum/median/p95: '+', '.join(f'{d["n"]:,} retained ({d["above_limit"]:,} above r_lim)' for d in bg_comparison[g].values())+
    '. At 2/4/8 SD: '+', '.join(f'{d["n"]:,} retained, median r={d["median_r"]:.3f}' for d in sd_comparison[g].values())+'.' for g in ['DNA','Gold'])

CAPTIONS = [
    ('figure_01_distributions.png', 'Figure 1. Baseline: minimum background for BOTH detection and extraction, using frame 1 of all 16 DNA and 14 gold survey stacks. '+fig1_numbers+'. Each retained detection has equal weight within its group. Each histogram bin is 100 times its count divided by all retained detections in that group; r bins are 0.02 wide and theta bins 2 degrees wide. All finite r values are included, even above r_lim=0.92768. Above-limit radii have no theta, so theta totals are slightly below 100%, without renormalization. Dashed grey steps show exact expected bin percentages for ideal 3-D isotropy, p(theta)=sin(theta); distributions are not divided by this reference. Faint steps are actual percentages. Smooth curves use Gaussian sigma=0.02 in r and 2 degrees in theta independently. Peak search uses prominence >=0.2 percentage points per bin and separation >=0.08r or >=8 degrees; these are descriptive maxima, not significance tests. '+fig1_peaks+' Detection uses repeated aligned 2x2 cell means, DoG widths 1/6 sensor pixels, filtered mean+2SD, and connected components >=10 pixels without an upper size limit. Extraction uses four channel means in 18x18 apertures. The corner ellipse, strongest-first aperture overlap rule, and any-frame raw saturation exclusion in aperture or component bounding box apply. No low-background mask or 2x survival filter is used. The ideal isotropic reference does not correct detection efficiency or establish the expected distribution of surface-bound gold.'),
    ('figure_02_background_selection.png', 'Figure 2. A-C show the first DNA stack, 18:38:50. D/E pool all 16 DNA stacks; F/G pool all 14 gold controls. A: recorded minimum background from the original 45-frame capture at 18:36:50. B: frame 1 after minimum subtraction. C: the same frame after 95th-percentile subtraction. All images use identical 20x display, with white at 12.75 ungained 8-bit-equivalent units per sensor pixel; aligned 2x2 cell averages are for display only. Red ellipses mark the recurring upper-right feature. D/F change ONLY the background choice: minimum, median or 95th percentile, with 2SD throughout. Each corrected frame is used for BOTH detection and extraction, using the matching recorded background capture. Detection is repeated, so positions, counts and r can all change; these curves do not isolate extraction at fixed positions. E/G change ONLY the threshold: 2, 4 or 8SD, using minimum background for BOTH detection and extraction. All four solid baseline curves exactly reproduce their corresponding Figure 1A curve. All variants allow components >=10 pixels without an upper limit and use the same corner, overlap and raw-saturation exclusions. Each curve is normalized to its own retained detections. '+fig2_numbers+' Faint steps show exact percentages per 0.02r bin, and smooth curves use sigma=0.02r. Above-model radii remain included. Grey dashed steps show ideal isotropic bin expectations. Raw saturation is checked before background subtraction over all150 real frames; display gain has no effect on selection.'),
    ('figure_03_background_references.png', 'Figure 3. All angles use four channel means in 18×18 sensor-pixel windows. A compares detected gold with uniform random windows in the gold images and the subset at least 32 pixels (Chebyshev distance) from every p95 candidate. Saturated random windows are excluded. B applies the same angle formula to 5,000 common positions in one recorded gold background capture; these are background-equivalent angles, not rod orientations. C splits gold detections by the brightest aligned 2×2 cell mean inside the original aperture, on the 150-frame mean. Brightness labels are 8-bit equivalents without display gain; exact original cutoffs 50/200 correspond to 3.1136/12.4542. Each curve is normalised separately. Random windows are not verified rod-free blanks; brighter selections can also reject genuine low-angle rods.'),
    ('figure_04_single_frames.png', 'Figure 4. Same detections and 18×18 windows throughout; no redetection. Only the intensities used for angle extraction change: 150-frame mean versus frames 1, 75 and 150. Denominators include all retained detections, so above-model radii have no angle and reduce the plotted total slightly. The near-30° population persists in both sample groups. Mean-based selection is still shared, and a single frame still integrates over the 1.194 ms exposure; this does not exclude faster motion or unresolved mixtures.'),
    ('figure_05_motion_check.png', 'Figure 5. Separate diagnostic data: two early DNA background captures (18:36:50 and 18:50:32), gold (19:21:57), and later DNA (21:03:58), with lateral stage motion. These are not the stationary survey stacks in Figure 1. Grey: earlier permissive motion-only candidates. Blue/orange: independent per-frame 1-SD detections matched to the common stage path in at least five frames, plus positive DoG at the predicted centre in every unmatched frame. Raw-saturated and overlapping candidates are excluded. First and dimmest frames are paired at positions with valid angles in both (n=481, 599, 101 and 475). The dimmest frame is deliberately selected and is not an unbiased angle estimate. The strict rule has a known false exclusion: a convincing track detected in 43/45 frames failed its two remaining frames. These curves illustrate selection dependence, not a validated rod population or corrected hinge-angle distribution. No corner or low-background mask is used here.'),
    ('figure_06_spot_recordings.png', 'Figure 6. Top: DNA 4, selected as a clear quiet example, 21:13:00, 59.5 s. Pixel panels use a common range; recorded narrow-field background is small. X/Y and theta are calculated separately for every exposure in the selected 20×20 aperture, with no temporal smoothing. The anisotropy-cloud axes are local, not the full unit circle. Bottom: all seven gold and nine DNA target recordings with unsaturated selected apertures; lines span frame-wise 2nd–98th percentiles, dots are medians, not confidence intervals. Gold 1–3 and DNA 8 are excluded for raw saturation. Targets were deliberately selected, often by apparent angle, and some may revisit the same rod; these are not random biological replicates. Four quiet low-angle DNA records do not establish that quietness identifies a hinge. Noise, residual background and genuine motion all contribute to the displayed ranges.'),
    ('figure_07_drive_check.png', 'Figure 7. Cached 12 October setup checks, same 20×20 ROI, background, 1.194 ms exposure and 798.79 fps. Statistics use approximately 10 ms channel averages. The mixed recording omits 0.25 s on either side of the approximate 5 s switch; OFF/ON assignments come from notes, not hardware timing. Manual z sweeps were not matched, and drive frequency/amplitude were not recorded. Below 1 means less recorded variation, not a calibrated cancellation efficiency. This is a limited diagnostic, not a resolution measurement or validation of all survey stacks.'),
]

subject = '12 October Polarcam data: current results and questions before the next measurement'
paragraphs = [
    'Dear all,',
    'Sorry for taking so long to finish the analysis. Background handling and deciding which features are rods have been much trickier than I expected. I think my earlier interpretation of the theta peaks was premature. I would like to discuss what we can conclude and what to measure next, rather than keep adjusting the analysis around these peaks.',
    'The main comparison below uses frame 1 for both detection and angle extraction in all 16 wide-field DNA-hinge stacks and 14 gold-only control stacks from 12 October. Exposure was 1.194 ms; the full 150-frame stacks are still checked for raw saturation. The samples were in two wells on the same surface. The light was on throughout; the note about turning it off was an intention I did not carry out. The cancellation drive was recorded as on, but its effectiveness was not calibrated.',
    f'With this working pipeline, the smoothed theta distribution has DNA maxima near 30° and 57°; gold has a stronger maximum near 28°. These are descriptive peaks, not established physical populations. All detections are pooled with equal weight; two gold fields contribute about {100*gold1["two_largest_field_share"]:.0f}% of retained detections. Counts are not verified unique rods or independent preparations. Theta is measured from the surface normal: 0° upright, 90° flat.',
    'Both panels show the percentage of retained detections in each bin: 0.02-wide r bins and 2-degree theta bins. The dashed reference is the expected bin percentage for ideal 3-D isotropy, without correcting for detection bias or implying that surface-bound gold should be isotropic. The distributions are not divided by this reference, so it is not flat, and peaks in r and theta need not correspond.',
    ('figure', 1),
    'Minimum-background subtraction leaves visible structure resembling the recorded background. I compared minimum, median and 95th-percentile backgrounds, using each corrected frame for both detection and angle calculation at the same 2SD setting. Stronger subtraction removes more visible structure but changes both the detected population and its r values. With minimum subtraction, 15/15,761 DNA and 3/9,503 gold detections exceed the model limit; with p95 subtraction this becomes 3,294/9,587 DNA and 158/2,725 gold. Cleaner images therefore do not establish better angles. Likewise, mostly staying below the predicted r limit of about 0.928 does not validate the background correction.',
    'Figure 1 now uses the simplest baseline: minimum background for both detection and angle calculation, at 2SD. Figure 2 changes one setting at a time: background at fixed 2SD, or threshold with minimum background. Oversized components are allowed throughout. Raising the threshold from 2SD to 8SD reduces retained DNA detections from 15,761 to 2,237 and raises median r from 0.33 to 0.76; gold changes from 9,503 to 422 and median r from 0.22 to 0.55. Neither selection is automatically correct. Unpolarized residual background generally reduces anisotropy, polarized background can shift it in either direction, and over-subtraction with zero clipping can also distort the ratios. Both figures use frame 1; the later mean-based diagnostics retain their original selections.',
    ('figure', 2),
    'Intensity alone is difficult: an otherwise identical rod at 20° would have about 11% of its 90° brightness under the current excitation/collection model. Using a linear 0–255 equivalent scale without display gain, gold detections at 28–34° have median peak-cell signal 4.0 per pixel after subtraction, versus recorded background 5.0 at those cells. At 50–60°, these are 11.7 and 2.2. The recorded background is the amount subtracted, not the residual error. Faintness alone does not identify a false spot.',
    'Random 18×18 windows in the gold images, including positions away from detections, peak around 24–26°; recorded backgrounds give apparent angles around 30–34°. These references are generally dimmer than the detected spots. They show that low apparent angles are not specific to identified rods, but do not establish that every near-30° detection is background. Neither reference is a verified rod-free blank.',
    ('figure', 3),
    'Using the same positions but extracting angles from frames 1, 75 and 150 leaves the near-30° feature largely intact: about 16–17% of DNA and 31–33% of gold detections fall at 28–34°. Stack averaging therefore does not explain most of it, although each exposure can still blur faster motion. Simplified known-angle additions also showed bias for dim simulated rods, without providing a calibrated experimental error rate.',
    ('figure', 4),
    'In nine later higher-power DNA fields, independently detecting spots in single frames also leaves a near-30° feature. Its size remains background-dependent: about 58% at 28–34° with the saved narrow-field background, 45% with a minimum built from the other eight fields, and 27% with the recorded wide-field minimum. These are different selections, not a controlled power comparison. More frames also make the constructed minimum more susceptible to downward noise bias.',
    'I also tried identifying sample-associated features by following the common lateral stage motion in the original background movies. Requiring independent detection in at least five frames and positive local contrast in all remaining frames reduces the low-angle population compared with permissive motion-only selection. But the rule rejects at least one convincing feature detected in 43/45 frames, and may preferentially reject genuine dim low-angle rods. I would not use it as a final correction or treat the remaining population as verified hinges.',
    ('figure', 5),
    'The long spot recordings are worth discussing separately. I often selected targets near 30°, so they cannot estimate the population. Four unsaturated DNA targets have quiet per-frame apparent angles around 29–34° for roughly 60–124 s. DNA 4 is a clear localized example, and its low angle persists with smaller apertures. The tiny recorded narrow-field background does not explain its angle by itself, although residual optical contributions remain uncertain. Quietness is compatible with a locked hinge, but also with a stuck rod; it does not establish attachment.',
    ('figure', 6),
    'The interference check remains limited. Switching the drive on within one recording reduced intensity CV from 14.3% to 7.1%, while X variation increased and Y variation decreased. Separate OFF/ON recordings also give mixed results. My manual z motion was not matched, so this does not quantify cancellation efficiency or angular resolution.',
    ('figure', 7),
    'My current conclusion is that the near-30° feature is not hinge-specific, but we cannot dismiss every low-angle target as background either. The higher-angle DNA population is worth following up, without yet assigning it to hinges. A rod angle of 60° implies a 30° hinge opening only if the base lies flat and the rod follows the moving arm; attachment offsets and hinge tilt matter.',
    'Richard, your description was a fully hybridised, hopefully un-nicked test strand. What range of angle and motion should we expect from that preparation, and what orientation/mobility should nonspecifically bound gold have on this surface? Could we agree on a small next measurement set with a matched rod-free surface/background control, a known-orientation reference if feasible, and targets across the apparent-angle range? I think that would be more useful now than further tuning this dataset.',
    'Best,\nDenny',
]

methods = ('Processing details: minimum, median and p95 backgrounds are per-pixel statistics of the original moving-sample background capture, excluding the appended phase marker. '
    'Figure 1 uses the saved minimum-subtracted frame 1 for BOTH detection and angle extraction, at 2SD. '
    'Figure 2 D/F use minimum, median or p95 background for BOTH steps at 2SD; E/G use minimum for BOTH steps at 2/4/8SD. Additional subtraction uses max(saved frame 1 − (chosen percentile−minimum), 0), rounded to saved units. Detection is repeated separately for each variant. '
    'The earlier mean-based diagnostics apply additional subtraction and zero clipping per frame before forming their detection mean. '
    'Detection uses repeated 2×2 polarization-cell means, Gaussian widths 1/6 sensor pixels, global filtered mean + 2 SD, '
    'and connected components of at least 10 pixels, with no maximum size in Figures 1-4. Figure 2 E/G also test 4/8SD. '
    'The fixed top-right ellipse is excluded; the strongest filtered candidate wins if 18×18 extraction windows overlap. '
    'Raw saturation at any time in the aperture or component bounding box excludes a candidate, using the pre-subtraction raw-peak map '
    '(approximately 255/255). No 2× survival or low-background-region filter is used. '
    'Angles use minimum-subtracted channel means in the same 18×18 aperture and the water Fresnel/BFP/annular-hole model '
    '(NA 0.39–1.30), without an APD matrix or additional fitted camera-channel gains. '
    'Finite r above the model limit is kept in radius plots but has no assigned theta; it is not clipped back to 90°. '
    'Intensity conversion is I_8bit-equivalent = I_saved × 255/4095, retaining fractional values. '
    'Figure 1 uses independent frame-1 detection and extraction. Figure 2 uses frame 1 throughout, with one-stack image examples and all 16 DNA stacks pooled in D/E and all 14 gold stacks pooled in F/G; Figure 3 remains the documented mean-based diagnostic. Figure 4 isolates single-frame extraction at the earlier mean-selected positions. '
    'The later nine-field check independently detects and extracts each tested single frame; its background-dependent percentages use valid angles as denominators. '
    'Figures 5–7 have separate data and selections specified in their captions and are not pooled into Figure 1. '
    'The motion filter is still the existing strict version, including its known 43/45-frame false exclusion; it has not been silently revised for this draft. '
    'Figure 1 reuses cached independent frame-1 detections; the other figures retain their documented inputs. No acquisitions or detector settings were changed.')

text = [f'Subject: {subject}', '']
body = []
for p in paragraphs:
    if isinstance(p, tuple):
        fn, caption = CAPTIONS[p[1]-1]
        text += [f'[Insert Figure {p[1]} here: {fn}]', caption, '']
        short = caption.split('. ', 1)[1].split('. ', 1)[0]
        body.append(f'<figure id="figure{p[1]}"><figcaption class="figure-label">Figure {p[1]} · {html.escape(short)}</figcaption><a href="{fn}"><img src="{fn}" alt="{html.escape(caption)}"></a><details class="caption"><summary>Figure {p[1]}: selection, numbers and limitations</summary><p>{html.escape(caption)}</p></details></figure>')
    else:
        text += [p, '']
        body.append('<p>'+html.escape(p).replace('\n', '<br>')+'</p>')
text += ['Processing details (can sit below the email):', methods]
(OUT/'email.txt').write_text('\n'.join(text)+'\n', encoding='utf8')
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Polarcam email draft · updated 15 October</title><style>
body{margin:0;background:#f3f4f2;color:#24333c;font:17px/1.6 system-ui}main{max-width:1120px;margin:24px auto;background:white;padding:28px 38px}
p{max-width:900px}h1{font-size:24px;line-height:1.4;font-weight:600}figure{margin:30px 0}img{width:100%;height:auto;display:block}
figcaption,.methods,.caption{font-size:14px;line-height:1.5;color:#4b5b64}.figure-label{font-weight:600;margin-bottom:10px}nav{padding:10px 0;border-bottom:1px solid #ddd;font-size:14px}a{color:#166888}
details{border-top:1px solid #ddd;padding-top:18px}summary{cursor:pointer}@media(max-width:650px){main{margin:0;padding:18px}body{font-size:16px}}
@media print{nav{display:none}main{margin:0;padding:0}figure{break-inside:avoid}}
</style><main><nav>Updated 15 October 2026 · Local draft · not sent<br><a href="email.txt">Plain text and insertion markers</a> · <a href="email_and_figures.zip" download>Download draft + numbered figures</a><br>Figures '''
page += ' · '.join(f'<a href="#figure{i}">{i}</a>' for i in range(1,len(CAPTIONS)+1))
page += '</nav><h1>'+html.escape(subject)+'</h1>'+''.join(body)
page += '<details><summary>Processing details</summary><p class="methods">'+html.escape(methods)+'</p></details></main></html>'
(OUT/'index.html').write_text(page, encoding='utf8')
(OUT/'sources.json').write_text(json.dumps({'inputs_sha256': INPUTS,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope': 'Figure 1 uses the minimum/minimum 2SD frame-1 baseline. Figure 2 D/F vary background (minimum/median/p95) for both detection and extraction at 2SD; E/G vary threshold (2/4/8SD) with minimum background for both steps. All variants allow oversized components. All16 DNA and14 gold stacks are pooled separately. Acquisition files remain read only. No hypothesis tests.',
    'updated': '2026-10-15',
    'changes': ['Figure 1 shows percentages of all retained detections in fixed r and theta bins, with an ideal isotropic reference and independent descriptive peak searches.',
                'Promoted minimum-background detection and extraction to the single Figure 1 baseline; removed the mixed-background alternative from the draft.',
                'All four Figure 2 baseline curves exactly match Figure 1A; percentages refer to detections per bin, not to the isotropic expectation.',
                'Added existing later-field and stage-motion checks with their limitations.',
                'Added quiet-target example and all unsaturated spot-recording intervals.',
                'Figure 1 uses cached single-frame selection; Figure 2 is entirely frame 1, with one-stack image examples and all 16 DNA stacks pooled in panels D/E and all 14 gold stacks pooled in panels F/G; the residue feature is marked. Other diagnostic figures retain their earlier inputs.',
                'Reframed the conclusion around discussion and a new measurement.'],
    'operator_correction': 'User confirms the light remained on for all recordings; intention to turn it off was not carried out.'}, indent=2), encoding='utf8')
with zipfile.ZipFile(OUT/'email_and_figures.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for fn in ['email.txt', 'index.html', 'sources.json', 'figure_01_results.json', 'figure_02_results.json', *[x[0] for x in CAPTIONS]]: z.write(OUT/fn, arcname=fn)

print('Created', OUT)
print('Body words:', sum(len(p.split()) for p in paragraphs if isinstance(p, str)))
print('Existing random-window results, brightness converted for display:')
for name, rows in [('uniform', random), ('away', away)]:
    vv=vals(rows,'theta'); h=np.histogram(vv[np.isfinite(vv)],TH)[0]
    print(name, 'valid', int(np.isfinite(vv).sum()), 'modal 2-degree bin', TH[h.argmax():h.argmax()+2].tolist(),
          'median peak-cell intensity (8-bit equivalent)', round(float(np.nanmedian(vals(rows,'peak_mean'))*SCALE),3))
