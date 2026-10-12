"""Review all October 9 spot traces without refitting backgrounds or selecting rods.

Reuses the saved native-channel traces from review_friday.py and checks sampled
channel values against each original stack. Both existing theta conversions are
shown deliberately; neither is chosen because it produces a desired peak.
"""
from pathlib import Path
import csv
import json
import sys
from html import escape

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
OUT = BASE / "pre_experiment_check"
sys.path.insert(0, str(REPO / "src/polarcam/live"))
sys.path.insert(0, str(BASE.parent / "paper_analysis_20261008/original_scripts"))
from polarcam.live import intensity_spots
from polarcam.live.Spinners_gui_live import BasicVideoPlayer
import hinge_analysis as ha


def run():
    OUT.mkdir(exist_ok=True)
    player = BasicVideoPlayer.__new__(BasicVideoPlayer)
    player._theta_recon_lut_cache = {}
    lut = player._theta_recon_lut("water")
    abc = ha.fourkas_ABC(1.3, 1.33, .39)
    source_rows = {r["file"]: r for r in csv.DictReader((BASE / "spot_summary.csv").open())}
    summaries, traces = [], {}
    for path in sorted((BASE / "traces").glob("spotrec*.npz")):
        stem, stamp = path.stem, path.stem[-6:]
        source = source_rows[stem]
        raw_path = next((REPO / "data/hinges/polarcam/2026-10-09").rglob(stem + ".npy"))
        meta = json.loads(raw_path.with_suffix(".json").read_text(encoding="utf-8-sig"))
        with np.load(path, allow_pickle=False) as z:
            channels = z["channels"].copy()
        raw = np.load(raw_path, mmap_mode="r")
        assert len(channels) == len(raw) - int(meta["actual"]["phase_marker_appended"])
        left, top = int(source["roi_local_x"]), int(source["roi_local_y"])
        roi = meta["roi"]
        bounds = [(left, left+20, top, top+20)]
        for index in (0, len(channels)//2, len(channels)-1):
            np.testing.assert_allclose(intensity_spots.channel_means(raw[index], bounds, (roi["x"], roi["y"]))[0], channels[index])
        del raw
        fps = float(source["fps"])
        xy = intensity_spots.anisotropy(channels)
        radius = np.linalg.norm(xy, axis=-1)
        native_theta = np.degrees(np.arccos(intensity_spots.unit_vectors(xy, lut)[:, 2]))
        old_theta = ha.theta_from_r(radius, *abc)
        mean_xy = intensity_spots.anisotropy(channels.mean(0))
        mean_r = float(np.linalg.norm(mean_xy))
        gui_mean = np.degrees(np.arccos(intensity_spots.unit_vectors(mean_xy[None], lut)[0, 2]))
        old_mean = float(ha.theta_from_r(np.array([mean_r]), *abc)[0])
        block = 15  # 9.827 ms at the recorded fps; mean channels before ratios.
        used = len(channels)//block*block
        binned = channels[:used].reshape(-1, block, 4).mean(1)
        xb = intensity_spots.anisotropy(binned)
        tb = (np.arange(len(binned))*block + (block-1)/2)/fps
        theta_b = ha.theta_from_r(np.linalg.norm(xb, axis=-1), *abc)
        group = "Gold control, well 2" if stamp == "162044" else "DNA well 1, round 1" if stamp < "162044" else "DNA well 1, round 2"
        row = dict(file=stem, group=group, duration_s=len(channels)/fps,
                   cancellation_on=meta["sound"]["sound_on"], saved_bg_subtracted=meta["background"]["background_subtracted"],
                   recorded_max_raw=int(source["max_raw"]), radius_of_mean_channels=mean_r,
                   theta_of_mean_channels_gui=gui_mean, theta_of_mean_channels_offline=old_mean,
                   valid_fraction_gui=float(np.isfinite(native_theta).mean()),
                   valid_fraction_offline=float(np.isfinite(old_theta).mean()),
                   offline_theta_p05_10ms=float(np.nanpercentile(theta_b, 5)),
                   offline_theta_median_10ms=float(np.nanmedian(theta_b)),
                   offline_theta_p95_10ms=float(np.nanpercentile(theta_b, 95)),
                   sd_x_10ms=float(np.std(xb[:, 0])), sd_y_10ms=float(np.std(xb[:, 1])),
                   mean_sum_channels=float(channels.sum(1).mean()))
        summaries.append(row)
        traces[stamp] = (tb, xb, theta_b)
    with (OUT / "spotrec_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader(); writer.writerows(summaries)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharey=True)
    for axis, row in zip(axes.flat, summaries):
        stamp = row["file"][-6:]
        tb, xy, theta = traces[stamp]
        axis.plot(tb, theta, lw=.45, color="#c08520" if stamp == "162044" else "#257b9e")
        axis.set(title=f"{stamp} | {'gold' if stamp == '162044' else 'DNA well'} | cancel {'on' if row['cancellation_on'] else 'off'}", ylim=(0,90), xlabel="Time (s)", ylabel="Theta (degrees)")
    fig.suptitle("9 October: orientation traces, earlier offline annular model\n15-frame channel averages (9.83 ms); gaps are out of model; these are selected rods, not population statistics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / "all_spotrec_theta.png", dpi=150)
    plt.close(fig)
    selected = ("155855", "160303", "162044", "172515")
    fig, axes = plt.subplots(4, 2, figsize=(12, 11))
    for (ax, scatter), stamp in zip(axes, selected):
        tb, xy, theta = traces[stamp]
        row = next(r for r in summaries if r["file"].endswith(stamp))
        ax.plot(tb, xy[:,0], lw=.6, label="X")
        ax.plot(tb, xy[:,1], lw=.6, label="Y")
        ax.set(title=f"{stamp}: {row['group']}", xlabel="Time (s)", ylabel="Anisotropy", ylim=(-1,1))
        ax.legend(loc="upper right", ncol=2)
        scatter.scatter(xy[:,0], xy[:,1], s=2, alpha=.2, rasterized=True)
        scatter.set(xlim=(-1,1), ylim=(-1,1), xlabel="X", ylabel="Y", aspect="equal",
                    title=f"theta(mean channels): {row['theta_of_mean_channels_offline']:.1f}° offline / {row['theta_of_mean_channels_gui']:.1f}° GUI")
    fig.suptitle("Different shapes of observed variation; 9.83 ms channel averages\nMotion, interference and measurement noise are not separated by these plots", fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(OUT / "selected_spotrec_xy.png", dpi=150)
    plt.close(fig)
    # Equal group sizes, two-sided alpha=.05, power=.80; independent rods only.
    power = []
    for p1, p2 in ((.30,.10), (.20,.10), (.15,.10)):
        pooled = (p1+p2)/2
        n = int(np.ceil(((norm.ppf(.975)*np.sqrt(2*pooled*(1-pooled)) + norm.ppf(.8)*np.sqrt(p1*(1-p1)+p2*(1-p2)))/(p1-p2))**2))
        power.append(dict(dna_band_fraction=p1, control_band_fraction=p2, independent_rods_per_group=n))
    (OUT / "sample_size_examples.json").write_text(json.dumps(power, indent=2))
    table = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in (
        r['file'][-6:], r['group'], round(r['duration_s'],1), r['cancellation_on'],
        f"{r['theta_of_mean_channels_offline']:.1f}", f"{r['theta_of_mean_channels_gui']:.1f}",
        f"{r['valid_fraction_offline']:.1%}", f"{r['valid_fraction_gui']:.1%}")) + "</tr>" for r in summaries)
    (OUT / "index.html").write_text("""<!doctype html><meta charset="utf-8"><title>Pre-experiment spot review</title>
<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#25353d}img{width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}p{line-height:1.5}</style>
<h1>9 October spot recordings: what can we learn?</h1>
<p>All 16 recordings checked. Cached channels were verified at the beginning, middle and end of every original stack. The first round is DNA well 1; 16:20:44 is the sole gold-only recording, well 2; the later round returns to well 1. Repeated positions are not independent rods.</p>
<p><b>There is no demonstrated hinge/control behavioural separation here.</b> The gold rod is fairly constrained; some DNA-well rods also are, while others vary much more. Neither stillness nor motion establishes a DNA attachment. Interference, brightness, averaging, selection and backgrounds confound the comparison. The first five DNA rods were not selected by angle.</p>
<p><b>Absolute theta depends substantially on the existing conversion.</b> Earlier offline analysis: annular Fourkas, NA 1.3, inner NA .39, n 1.33, r_max=0.9208. Current GUI water LUT: existing J coefficients, n 1.333, r_max=0.82166. Identical channels give the gold control 39.2° offline versus 27.2° GUI. The near-40° labels from earlier figures cannot be used directly to select GUI rods. No conversion or calibration was changed in this review.</p>
<p>The table uses theta reconstructed after averaging four channels over the whole recording, <b>not</b> the mean of frame-wise theta, nor the GUI's default Cartesian mean. At 15:58:55 the X/Y trajectory crosses a wide arc: averaging cancels anisotropy, giving 16.4° although instantaneous offline theta is mostly near 39°. Do not interpret that whole-recording average as a dwell angle.</p>
<table><tr><th>Time</th><th>Sample</th><th>Seconds</th><th>Cancel on</th><th>Theta offline</th><th>Theta GUI</th><th>Valid native frames offline</th><th>Valid native frames GUI</th></tr>""" + table + """</table>
<p>NaN means outside the model; invalid frames are never moved to 90°. Many missing frames can bias summaries of the remaining frames. These data use the saved pixels with no further background fitting; absence of saturation is not independently certified from legacy subtracted files.</p>
<h2>Observed dynamics in X and Y</h2><img src="selected_spotrec_xy.png">
<h2>All theta traces, keeping the earlier offline model</h2><img src="all_spotrec_theta.png">
<p>Every trace uses the recorded 1526.4 fps and 0.598 ms exposure, with 15-frame channel averages before conversion. The smoothing reduces noise and also real fast motion. 17:25:15 / 17:28:33 are the same selected position (on/off); so are 17:44:18 / 17:45:45 (off/on). They are not additional independent rods. Background-save settings differ in the first pair.</p>
<p><a href="spotrec_summary.csv">All numerical summaries</a></p>""", encoding="utf-8")
    print(json.dumps({"recordings":len(summaries), "sample_size_examples":power, "output":str(OUT)}, indent=2))


if __name__ == "__main__":
    run()
