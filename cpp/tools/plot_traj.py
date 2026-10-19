"""
Top-view trajectories of benchmark.py runs, Sim(3)-aligned to GT (as for the ATE), one panel per sequence.

usage: python3 plot_traj.py --out traj.png [--suite kitti] [--mode mono] [--seqs REGEX] [--title T] [--paper] LABEL=DIR [...]
  first LABEL=DIR is the reference (gray), the rest get categorical colors in fixed order; GT is drawn dark.
  --paper: SLAM-paper style figure (white, serif, GT dashed black, start marker, x/z [m], shared legend,
           per-panel ATE "first -> last run", † = tracked < 95 %); also writes a PDF next to the PNG.
  view: x (right) vs z (forward) of the GT world (KITTI: camera-0 frame, ICRA/ARKit: y is up), EuRoC: x vs y (z is up)
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


def paper_figure(jobs, runs, out, title):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.edgecolor": "black", "axes.linewidth": 0.6, "xtick.direction": "in",
                         "ytick.direction": "in", "figure.facecolor": "white", "axes.facecolor": "white"})
    colors = ["#eb6834", "#2a78d6", "#1baf7a", "#eda100"]  # 검증된 팔레트 (orange = 이전, blue = 이후)
    ncol = min(5, len(jobs))
    nrow = (len(jobs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(3.3 * ncol, 7.0), 3.2 * nrow + 0.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    handles = {}
    for ax, job in zip(axes.ravel(), jobs):
        up = 1 if job["suite"] == "euroc" else 2  # 위에서 본 평면: EuRoC 월드는 z 가 위 -> x-y, KITTI/ICRA 카메라 월드는 x-z
        ax.axis("on")
        G, ates = None, []
        for (label, d), c in zip(runs, colors):
            r = aligned(job, d)
            if r is None:
                ates.append("-")
                continue
            A, Gi, drift = r
            if G is None or len(Gi) > len(G):
                G = Gi
            h, = ax.plot(A[:, 0], A[:, up], color=c, linewidth=1.3, label=label)
            handles.setdefault(label, h)
            ate = drift / 100 * np.linalg.norm(np.diff(Gi, axis=0), axis=1).sum()
            short = len(A) < 0.95 * len(np.loadtxt(job["gt"]).reshape(-1, 12)[job["start"]::job["stride"]])
            ates.append(f"{ate:.1f}{'†' if short else ''}")
        if G is not None:
            g, = ax.plot(G[:, 0], G[:, up], color="black", linewidth=0.9, linestyle=(0, (4, 2)), label="Ground truth", zorder=0)
            st, = ax.plot(G[0, 0], G[0, up], marker="s", color="black", markersize=5, linestyle="none", label="Start", zorder=5)
            handles.setdefault("Ground truth", g), handles.setdefault("Start", st)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color="#d9d9d9", linewidth=0.4)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("yz"[up - 1] + " [m]")
        ax.set_title(f"Sequence {job['seq']}", fontsize=10)
        ax.text(0.03, 0.97, "ATE [m]: " + " → ".join(ates), transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    order = ["Ground truth"] + [l for l, _ in runs] + ["Start"]
    h_in = fig.get_figheight()  # 제목 (있으면) 아래에 범례: 둘이 겹치지 않게 인치 단위로 자리 잡기
    fig.legend([handles[k] for k in order if k in handles], [k for k in order if k in handles], loc="upper center",
               bbox_to_anchor=(0.5, 1 - (0.3 / h_in if title else 0)), ncol=len(order), frameon=False, fontsize=9.5)
    if title:
        fig.suptitle(title, y=1 - 0.05 / h_in, va="top", fontsize=11)
    fig.text(0.005, 0.005, "Trajectories Sim(3)-aligned to ground truth (as for the ATE). "
             "† tracked < 95 % of the sequence (ATE over the tracked part).", fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.025, 1, 1 - (0.65 if title else 0.35) / h_in))
    fig.savefig(out, dpi=200)
    fig.savefig(Path(out).with_suffix(".pdf"))
    print("saved", out, "and", Path(out).with_suffix(".pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="kitti")
    ap.add_argument("--mode", default="mono")
    ap.add_argument("--seqs", default=".")
    ap.add_argument("--title", default="")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("runs", nargs="+", help="LABEL=DIR")
    o = ap.parse_args()
    runs = [tuple(s.split("=", 1)) for s in o.runs]
    if o.suite == "kitti":
        jobs = list(bm.kitti_jobs(o.seqs))
    elif o.suite == "euroc":
        jobs = list(bm.euroc_jobs(o.seqs))
    else:  # GT already converted by benchmark.py into <dir>/gt
        jobs = [j for j in bm.icra_jobs(o.seqs, [o.mode], Path(runs[0][1])) if j["mode"] == o.mode]
    if o.paper:
        return paper_figure(jobs, runs, o.out, o.title)
    colors = [REF] + SERIES[: len(runs) - 1]

    ncol = min(4, len(jobs))
    nrow = (len(jobs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), facecolor=SURFACE, squeeze=False)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        ax.axis("off")
    for ax, job in zip(axes.ravel(), jobs):
        up = 1 if job["suite"] == "euroc" else 2  # 위에서 본 평면: EuRoC 월드는 z 가 위 -> x-y, KITTI/ICRA 카메라 월드는 x-z
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
            ax.plot(A[:, 0], A[:, up], color=c, linewidth=1.6, label=label)
            notes.append(f"{label}: {drift:.2f}%")
        if G is not None:
            ax.plot(G[:, 0], G[:, up], color=INK, linewidth=1.0, linestyle=(0, (4, 2)), label="GT", zorder=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color=GRID, linewidth=0.6)
        ax.tick_params(labelsize=7, colors=INK2)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(job["seq"], color=INK, fontsize=10, loc="left")
        ax.text(0.02, 0.02, "\n".join(notes), transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom",
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.8, pad=1.5))
    h, l = next(ax for ax in axes.ravel() if ax.lines).get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.suptitle(o.title or "top view, Sim(3)-aligned to GT (drift % = ATE / path)", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(o.out, dpi=110, facecolor=SURFACE)
    print("saved", o.out)


if __name__ == "__main__":
    main()
