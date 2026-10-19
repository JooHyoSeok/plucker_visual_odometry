"""
KITTI Sim(3) ATE RMSE per sequence of benchmark.py runs next to ORB-SLAM (Mur-Artal et al., TRO 2015, Table: monocular,
with loop closure, keyframe trajectory) and ORB-SLAM + global BA (20 its).

usage: python3 plot_vs_orbslam.py --out plot.png [--title T] LABEL=DIR [...]
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_bench import GRID, INK, INK2, SERIES, SURFACE

ORB = {"00": 6.68, "02": 21.75, "03": 1.59, "04": 1.79, "05": 8.23, "06": 14.68, "07": 3.36, "08": 46.58, "09": 7.62,
       "10": 8.68}
ORB_GBA = {"00": 5.33, "02": 21.28, "03": 1.51, "04": 1.62, "05": 4.85, "06": 12.34, "07": 2.26, "08": 46.68, "09": 6.62,
           "10": 8.80}  # 01: ORB-SLAM 실패


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("runs", nargs="+", help="LABEL=DIR")
    o = ap.parse_args()
    seqs = sorted(ORB)
    series = [("ORB-SLAM", ORB, "#2b2b2b"), ("ORB-SLAM + GBA", ORB_GBA, "#9a9994")]
    for i, spec in enumerate(o.runs):
        label, d = spec.split("=", 1)
        rows = {r["seq"]: r for r in csv.DictReader(open(Path(d) / "results.csv")) if r["suite"] == "kitti"}
        series.append((label, {s: float(rows[s]["ate_sim3_m"]) if s in rows else np.nan for s in seqs}, SERIES[i]))
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(14, 4.8))
    n, w = len(series), 0.8 / len(series)
    x = np.arange(len(seqs))
    for i, (label, vals, c) in enumerate(series):
        v = np.array([vals.get(s, np.nan) for s in seqs])
        ax.bar(x - 0.4 + w * (i + 0.5), v, w * 0.9, color=c, label=label, edgecolor=SURFACE, linewidth=0.5)
        for xi, vi in zip(x, v):
            if np.isfinite(vi):
                ax.text(xi - 0.4 + w * (i + 0.5), vi, f"{vi:.0f}" if vi >= 10 else f"{vi:.1f}", ha="center", va="bottom",
                        fontsize=6.5, color=INK2, rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(x, seqs)
    ax.set_ylabel("Sim(3) ATE RMSE [m] (log)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, ncol=n, loc="upper left", fontsize=9)
    wins = []
    for label, vals, _ in series[2:]:  # ORB-SLAM (GBA 없음) 보다 좋은 시퀀스 수
        wins.append(f"{label}: better than ORB-SLAM on {sum(vals[s] < ORB[s] for s in seqs if np.isfinite(vals[s]))}/{len(seqs)}")
    ax.set_title((o.title + "\n" if o.title else "") + " | ".join(wins), loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(o.out, dpi=120)
    print("saved", o.out)


if __name__ == "__main__":
    main()
