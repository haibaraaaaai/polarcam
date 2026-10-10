"""Stationary-spot distributions: (A) p(r) where peaks are FOUND, (B) p(theta) for reading.
Adopted model: hole+Fresnel (DX notebook): A=0.097840, B=0.011915, C=0.101852, r_sat=0.927995.
Author: ClawsGO Science Agent"""
from pathlib import Path
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

BASE = Path(__file__).resolve().parent
A, B, C = 0.097840, 0.011915, 0.101852
RSAT = C / (A + B)
BW_R, BW_T = 0.045, 4.0

PEAKS_R = {
    "20260918-145406": [0.847],
    "20260918-151106": [0.401, 0.855],
    "20260918-151218": [0.866],
    "20260918-152241": [0.866],
    "20260918-152256": [0.862],
    "20260918-152740": [0.891],
    "20260918-154622": [0.443, 0.892],
    "20260918-155126": [0.402, 0.852],
}

def theta_of_r(r):
    r = np.asarray(r, float)
    s2 = np.clip(A * r / (C - B * r), 0.0, 1.0)   # clamp: beyond r_sat -> exactly 90
    return np.degrees(np.arcsin(np.sqrt(s2)))

def r_of_theta(t):
    s2 = np.sin(np.radians(t)) ** 2
    return C * s2 / (A + B * s2)

def kde(x, grid, bw):
    counts = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2).sum(1)
    return counts / (len(x) * bw * np.sqrt(2 * np.pi))

def weak_shoulders(r, peaks_stable):
    """Local maxima resolved at the finer bandwidth (0.025) with prominence
    >= 0.5% of the fine-KDE max, in [0.30, 0.65], and not within 0.03 of a
    stable peak. Real-but-fragile features the coarse KDE smears out."""
    from scipy.signal import find_peaks
    counts, _ = np.histogram(r, bins=np.arange(-0.30, 1.5001 + 0.0005, 0.0005))
    d = gaussian_filter1d(counts.astype(float) / (len(r) * 0.0005), 0.025 / 0.0005, mode="constant")
    gate = 0.005 * d.max()
    pk, pr = find_peaks(d, prominence=gate, distance=int(0.05 / 0.0005))
    out = []
    for q, prom in zip(pk, pr["prominences"]):
        g = -0.30 + (q + 0.5) * 0.0005
        if not 0.30 <= g <= 0.65:
            continue
        if any(abs(g - s) <= 0.03 for s in peaks_stable):
            continue
        out.append(float(g))
    return out

def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "savefig.dpi": 300,
    })
    blue, orange, dark, muted = "#0072B2", "#C44E00", "#23313C", "#596875"
    ref = "#89939C"
    files = sorted(BASE.glob("audit_*.npz"))
    assert len(files) == 8
    grid_r = np.arange(0, 1.0001, 0.0005)
    grid_t = np.arange(0, 90.001, 0.1)

    # ---------- FIGURE A: p(r), peaks found in r ----------
    fig, axs = plt.subplots(2, 4, figsize=(14.5, 7.0), sharex=True, sharey=True)
    table = []
    for i, (ax, p) in enumerate(zip(axs.flat, files)):
        with np.load(p, allow_pickle=False) as z:
            r = z["r_saved"]; r = r[np.isfinite(r)]
        dens = kde(r, grid_r, BW_R)
        ax.fill_between(grid_r, dens, 0, color=blue, alpha=0.16, lw=0)
        ax.plot(grid_r, dens, color=blue, lw=1.7)
        for rp in PEAKS_R[p.stem.removeprefix("audit_")]:
            d = float(np.interp(rp, grid_r, dens))
            ax.scatter([rp], [d], s=26, color=orange, zorder=5, edgecolors="white", linewidths=0.5)
            ax.annotate(f"{theta_of_r(rp):.1f}°", (rp, d), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8.6, color=orange)
        for ws in weak_shoulders(r, PEAKS_R[p.stem.removeprefix("audit_")]):
            d_ws = float(np.interp(ws, grid_r, kde(r, grid_r, 0.025)))
            ax.scatter([ws], [d_ws], s=30, marker="D", facecolors="none",
                       edgecolors=orange, linewidths=1.0, zorder=5, alpha=0.85)
            ax.annotate(f"{theta_of_r(ws):.1f}°", (ws, d_ws), xytext=(0, -13),
                        textcoords="offset points", ha="center", fontsize=7.8,
                        color=orange, alpha=0.8)
        ax.axvline(RSAT, color=ref, lw=1.1, ls="--")
        ax.set_xlim(0, 1.0); ax.set_ylim(bottom=0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"{p.stem[-6:]}  ·  n = {len(r):,}", loc="left", color=dark, pad=26)
        sec = ax.secondary_xaxis("top", functions=(theta_of_r, r_of_theta))
        sec.set_xticks([0, 30, 60, 90])
        sec.set_xticklabels(["0°", "30°", "60°", "90°"], fontsize=8, color=muted)
        if i % 4 == 0:
            ax.set_ylabel("spot density")
        if i >= 4:
            ax.set_xlabel("anisotropy radius r")
        table.append((p.stem[-6:], len(r)))
    fig.suptitle("A · peaks found in r  —  ● bandwidth-stable peak, ◇ weak shoulder, ┄ r_sat = 0.928  (θ on top axis)",
                 x=0.05, y=1.0, ha="left", va="top", fontsize=14, fontweight="bold", color=dark)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.10, wspace=0.18, hspace=0.42)
    out = Path(os.environ["FIGURE_OUT"])
    fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".svg"))

    # ---------- FIGURE B: p(theta), with both markers to show the Jacobian shift ----------
    fig2, axs2 = plt.subplots(2, 4, figsize=(14.5, 7.0), sharex=True, sharey=True)
    for i, (ax, p) in enumerate(zip(axs2.flat, files)):
        with np.load(p, allow_pickle=False) as z:
            r = z["r_saved"]; r = r[np.isfinite(r)]
        th_all = theta_of_r(r[r <= RSAT])
        dens = kde(th_all, grid_t, BW_T)
        ax.fill_between(grid_t, dens, 0, color=blue, alpha=0.16, lw=0)
        ax.plot(grid_t, dens, color=blue, lw=1.7)
        # theta-space modes (what the eye sees)
        from scipy.signal import find_peaks
        pk, _ = find_peaks(dens, prominence=0.02 * dens.max(), distance=int(4 / 0.1))
        for q in pk:
            ax.scatter([grid_t[q]], [dens[q]], s=26, color=orange, zorder=5,
                       edgecolors="white", linewidths=0.5)
            ax.annotate(f"{grid_t[q]:.1f}°", (grid_t[q], dens[q]), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8.6, color=orange)
        # mapped r-space peaks (open circles): where the r analysis puts them
        for rp in PEAKS_R[p.stem.removeprefix("audit_")]:
            t_mapped = float(theta_of_r(rp))
            d = float(np.interp(t_mapped, grid_t, dens))
            ax.scatter([t_mapped], [d], s=42, facecolors="none", edgecolors=dark, linewidths=1.1, zorder=5)
        ax.set_xlim(0, 90); ax.set_ylim(bottom=0)
        ax.set_xticks([0, 30, 60, 90])
        ax.set_title(f"{p.stem[-6:]}", loc="left", color=dark, pad=7)
        if i % 4 == 0:
            ax.set_ylabel("spot density")
        if i >= 4:
            ax.set_xlabel("θ (deg)")
    fig2.suptitle("B · the same spots in θ  —  ● mode of p(θ), ○ θ(r-peak) from panel A (Jacobian shift)",
                  x=0.05, y=1.0, ha="left", va="top", fontsize=14, fontweight="bold", color=dark)
    fig2.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.10, wspace=0.18, hspace=0.42)
    out2 = out.with_name(out.stem + "_theta_space" + out.suffix)
    fig2.savefig(out2, dpi=300); fig2.savefig(out2.with_suffix(".pdf")); fig2.savefig(out2.with_suffix(".svg"))
    for name, n in table:
        print(name, n)
    print("figures written")

if __name__ == "__main__":
    main()
