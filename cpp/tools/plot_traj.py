"""
Top-view trajectories of benchmark.py runs, Sim(3)-aligned to GT (as for the ATE), one panel per sequence.

usage: python3 plot_traj.py --out traj.png [--suite kitti] [--mode mono] [--seqs REGEX] [--title T] LABEL=DIR [...]
  first LABEL=DIR is the reference (gray), the rest get categorical colors in fixed order; GT is drawn dark.
  view: x (right) vs z (forward) of the GT world (KITTI: camera-0 frame, ICRA/ARKit: y is up)
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import benchmark as bm
from plot_bench import INK, INK2, GRID, REF, SERIES, SURFACE


def aligned(job, d):
    """traj of job in benchmark dir d, Sim(3)-aligned to its GT -> (C_aligned, G, drift %), or None"""
    name = f"{job['suite']}_{job['seq']}_{job['mode']}"
    traj, log = Path(d) / f"{name}.txt", Path(d) / f"{name}.log"
    if not traj.exists() or traj.stat().st_size == 0:
        return None
    m = re.search(r"first pose at frame (\d+)", log.read_text()) if log.exists() else None
    f0 = int(m.group(1)) if m else 0
    C = np.loadtxt(traj).reshape(-1, 12)[:, [3, 7, 11]]
    G = np.loadtxt(job["gt"]).reshape(-1, 12)[job["start"]::job["stride"]][f0:f0 + len(C)][:, [3, 7, 11]]
    C = C[:len(G)]
    if len(C) < 3:
        return None
    mc, mg = C.mean(0), G.mean(0)
    X, Y = C - mc, G - mg
    U, D, Vt = np.linalg.svd(Y.T @ X / len(C))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    s = np.trace(np.diag(D) @ S) / (X ** 2).sum(1).mean()
    A = s * X @ (U @ S @ Vt).T + mg
    path = np.linalg.norm(np.diff(G, axis=0), axis=1).sum()
    return A, G, 100 * np.sqrt(((A - G) ** 2).sum(1).mean()) / max(path, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="kitti")
    ap.add_argument("--mode", default="mono")
    ap.add_argument("--seqs", default=".")
    ap.add_argument("--title", default="")
    ap.add_argument("runs", nargs="+", help="LABEL=DIR")
    o = ap.parse_args()
    runs = [tuple(s.split("=", 1)) for s in o.runs]
    if o.suite == "kitti":
        jobs = list(bm.kitti_jobs(o.seqs))
    else:  # GT already converted by benchmark.py into <dir>/gt
        jobs = [j for j in bm.icra_jobs(o.seqs, [o.mode], Path(runs[0][1])) if j["mode"] == o.mode]
    colors = [REF] + SERIES[: len(runs) - 1]

    ncol = 4
    nrow = (len(jobs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), facecolor=SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        ax.axis("off")
    for ax, job in zip(axes.ravel(), jobs):
        ax.axis("on")
        G = None
        notes = []
        for (label, d), c in zip(runs, colors):
            r = aligned(job, d)
            if r is None:
                notes.append(f"{label}: -")
                continue
            A, Gi, drift = r
            if G is None or len(Gi) > len(G):
                G = Gi
            ax.plot(A[:, 0], A[:, 2], color=c, linewidth=1.6, label=label)
            notes.append(f"{label}: {drift:.2f}%")
        if G is not None:
            ax.plot(G[:, 0], G[:, 2], color=INK, linewidth=1.0, linestyle=(0, (4, 2)), label="GT", zorder=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color=GRID, linewidth=0.6)
        ax.tick_params(labelsize=7, colors=INK2)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(job["seq"], color=INK, fontsize=10, loc="left")
        ax.text(0.02, 0.02, "\n".join(notes), transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom",
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.8, pad=1.5))
    h, l = next(ax for ax in axes.ravel() if ax.lines).get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=len(l), frameon=False, fontsize=9)
    fig.suptitle(o.title or "top view, Sim(3)-aligned to GT (drift % = ATE / path)", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(o.out, dpi=110, facecolor=SURFACE)
    print("saved", o.out)


if __name__ == "__main__":
    main()
