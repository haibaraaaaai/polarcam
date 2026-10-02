"""Offline analysis of interference-tuning diagnostic recordings.

Reads the .npz files written by the Interference tuning panel's diagnostic
recorder (IntensityDiagnostic), reconstructs the anisotropy locus from the
four mosaic channel means, and produces:

  - a figure: intensity traces, raw and detrended XY loci, sliding span and
    XY-extent metrics,
  - a JSON summary: per-segment span/extent statistics and, when events were
    marked in the panel (Record reference / Test amplitude / Mark event),
    per-segment reduction factors.

Usage:
    python -m polarcam.live.interference_npz_analysis RECORDING.npz [MORE.npz ...]
        [--out DIR] [--window S] [--step S]

Without --out, figures and summaries are written next to each input file as
<name>_interference_analysis.png / .json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from polarcam.live.intensity_tuning import (  # noqa: E402
    anisotropy_series,
    moving_average_residual,
    xy_residual_stats,
)


def load_recording(path: Path) -> tuple[np.ndarray, list[str], dict]:
    with np.load(path, allow_pickle=False) as data:
        samples = np.asarray(data["samples"], dtype=np.float64)
        columns = [str(item) for item in data["columns"]]
        metadata = json.loads(str(data["metadata_json"]))
    return samples, columns, metadata


def _column(samples: np.ndarray, columns: list[str], name: str) -> np.ndarray:
    return samples[:, columns.index(name)]


def event_segments(metadata: dict, t_end: float) -> list[dict]:
    """Split the recording at marked events; no events means one segment."""
    events = sorted(
        (float(event["host_time_s"]), str(event["label"]))
        for event in metadata.get("events", [])
        if np.isfinite(event.get("host_time_s"))
    )
    boundaries = [0.0] + [time for time, _ in events] + ([t_end] if events else [])
    labels = ["before " + label for _, label in events] + [label for _, label in events]
    segments = []
    for index in range(len(boundaries) - 1):
        start, stop = boundaries[index], boundaries[index + 1]
        if stop - start < 1.0:
            continue
        segments.append({"start": start, "stop": stop, "label": labels[index]})
    if not segments:
        segments = [{"start": 0.0, "stop": t_end, "label": "whole recording"}]
    return segments


def analyse_recording(path: Path, out_dir: Path, *, window_s: float = 5.0, step_s: float = 1.0) -> dict:
    samples, columns, metadata = load_recording(path)
    t = _column(samples, columns, "received_s")
    t = t - t[0]
    t_end = float(t[-1]) if len(t) else 0.0
    mean = _column(samples, columns, "analysed_mean_dn")
    ax, ay, r = anisotropy_series(samples[:, [columns.index(name) for name in
                                               ("i0_mean_dn", "i90_mean_dn", "i45_mean_dn", "i135_mean_dn")]])
    timing = metadata.get("initial_settings", {}).get("timing_readback") or {}
    fps = timing.get("resulting_fps") or timing.get("fps")

    segments = event_segments(metadata, t_end)
    segment_stats = []
    for segment in segments:
        mask = (t >= segment["start"]) & (t < segment["stop"])
        count = int(np.count_nonzero(mask))
        if count < 64:
            continue
        duration = segment["stop"] - segment["start"]
        stats = {
            "label": segment["label"],
            "start_s": segment["start"],
            "stop_s": segment["stop"],
            "samples": count,
        }
        try:
            intensity_stats = {
                "mean": float(np.mean(mean[mask])),
                "p02": float(np.percentile(mean[mask], 2.0)),
                "p98": float(np.percentile(mean[mask], 98.0)),
            }
            intensity_stats["span"] = intensity_stats["p98"] - intensity_stats["p02"]
            stats["intensity"] = intensity_stats
        except IndexError:
            pass
        try:
            xy = xy_residual_stats(t[mask], samples[mask][:, [columns.index(name) for name in
                                                              ("i0_mean_dn", "i90_mean_dn", "i45_mean_dn", "i135_mean_dn")]],
                                   duration_s=duration)
            stats["xy"] = {key: xy[key] for key in ("extent", "extent_x", "extent_y", "detrend_s", "r_mean", "warnings")}
            stats["xy_plot"] = {"ax": xy["plot_ax"], "ay": xy["plot_ay"]}
        except ValueError as exc:
            stats["xy_error"] = str(exc)
        segment_stats.append(stats)

    # sliding metrics over the whole recording
    slide_t, slide_span, slide_extent = [], [], []
    if t_end > window_s:
        for start in np.arange(0.0, t_end - window_s, step_s):
            mask = (t >= start) & (t < start + window_s)
            if np.count_nonzero(mask) < 64:
                continue
            span = float(np.percentile(mean[mask], 98.0) - np.percentile(mean[mask], 2.0))
            finite = np.isfinite(ax) & np.isfinite(ay)
            use = mask & finite
            extent = float("nan")
            if np.count_nonzero(use) >= 300:
                duration = window_s
                try:
                    ax_res = moving_average_residual(t[use], ax[use], max(0.4 * duration, 1.0))
                    ay_res = moving_average_residual(t[use], ay[use], max(0.4 * duration, 1.0))
                    extent = float(np.hypot(np.percentile(ax_res, 98) - np.percentile(ax_res, 2),
                                            np.percentile(ay_res, 98) - np.percentile(ay_res, 2)))
                except ValueError:
                    extent = float("nan")
            slide_t.append(start + window_s / 2.0)
            slide_span.append(span)
            slide_extent.append(extent)

    # reductions between consecutive segments (reference-like -> next)
    for index in range(1, len(segment_stats)):
        previous, current = segment_stats[index - 1], segment_stats[index]
        reduction = {}
        if previous.get("intensity", {}).get("span", 0.0) > 0.0 and current.get("intensity", {}).get("span", 0.0) > 0.0:
            reduction["span_reduction"] = previous["intensity"]["span"] / current["intensity"]["span"]
        if previous.get("xy", {}).get("extent") and current.get("xy", {}).get("extent"):
            reduction["extent_reduction"] = previous["xy"]["extent"] / current["xy"]["extent"]
        if reduction:
            current["reduction_vs_previous"] = reduction

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    axes[0, 0].plot(t, mean, lw=0.4, color="#0b4f8a")
    axes[0, 0].set(xlabel="t (s)", ylabel="ROI mean (DN)", title=f"Intensity — {path.name}")
    for time, label in sorted((float(event["host_time_s"]), str(event["label"]))
                              for event in metadata.get("events", [])):
        axes[0, 0].axvline(time, color="#b45309", lw=0.8, ls="--")
        axes[0, 0].annotate(label, (time, axes[0, 0].get_ylim()[1]), rotation=90,
                            fontsize=6, va="top", color="#b45309")
    finite = np.isfinite(ax) & np.isfinite(ay)
    axes[0, 1].plot(ax[finite], ay[finite], ".", ms=1, color="0.55", alpha=0.3, rasterized=True)
    axes[0, 1].set(xlabel="ax", ylabel="ay", title="Raw XY locus (drift included)", aspect="equal")
    if slide_t:
        axes[1, 1].plot(slide_t, slide_span, lw=1.2, color="#0b4f8a", label="span P98-P2 (DN)")
        axes[1, 1].set_xlabel("t (s)")
        axes[1, 1].set_ylabel("span (DN)", color="#0b4f8a")
        axis_two = axes[1, 1].twinx()
        valid = np.isfinite(slide_extent)
        if np.any(valid):
            axis_two.plot(np.asarray(slide_t)[valid], np.asarray(slide_extent)[valid],
                          lw=1.2, color="#b45309", label="XY extent")
        axis_two.set_ylabel("XY extent", color="#b45309")
        axes[1, 1].set_title(f"Sliding {window_s:g} s metrics")
    axes[1, 0].set_title("Detrended XY locus per segment")
    for index, stats in enumerate(segment_stats):
        plot = stats.get("xy_plot")
        if not plot:
            continue
        axes[1, 0].plot(plot["ax"], plot["ay"], ".", ms=1, alpha=0.5, rasterized=True,
                        color=plt.cm.viridis(index / max(1, len(segment_stats) - 1)),
                        label=f"{stats['label']} (extent {stats.get('xy', {}).get('extent', float('nan')):.4f})")
    axes[1, 0].set(xlabel="ax", ylabel="ay", aspect="equal")
    axes[1, 0].legend(fontsize=6, loc="upper right")
    for axis in axes.flat:
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"Interference diagnostic analysis — {path.name}"
        + (f" — {fps:g} fps" if fps else ""),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figure_path = out_dir / f"{path.stem}_interference_analysis.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    summary = {
        "source": str(path),
        "fps": fps,
        "samples": int(len(samples)),
        "duration_s": t_end,
        "drive_note": metadata.get("initial_settings", {}).get("drive_note_manual"),
        "segments": segment_stats,
    }
    summary_path = out_dir / f"{path.stem}_interference_analysis.json"
    summary_path.write_text(json.dumps(summary, indent=1))
    return {"figure": str(figure_path), "summary": str(summary_path), "segments": segment_stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recordings", nargs="+", type=Path, help="diagnostic .npz files")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: next to each file)")
    parser.add_argument("--window", type=float, default=5.0, help="sliding metric window in s")
    parser.add_argument("--step", type=float, default=1.0, help="sliding metric step in s")
    args = parser.parse_args()
    for path in args.recordings:
        out_dir = args.out if args.out is not None else path.parent
        result = analyse_recording(path, out_dir, window_s=args.window, step_s=args.step)
        print(f"{path.name}:")
        for stats in result["segments"]:
            line = f"  [{stats['start_s']:6.1f}-{stats['stop_s']:6.1f} s] {stats['label']}"
            if "intensity" in stats:
                line += f" | span {stats['intensity']['span']:.1f} DN"
            if "xy" in stats:
                line += f" | XY extent {stats['xy']['extent']:.4f}"
            if "reduction_vs_previous" in stats:
                parts = [f"{key} {value:.2f}x" for key, value in stats["reduction_vs_previous"].items()]
                line += f" | reduction: {', '.join(parts)}"
            print(line)
        print(f"  figure: {result['figure']}")


if __name__ == "__main__":
    main()
