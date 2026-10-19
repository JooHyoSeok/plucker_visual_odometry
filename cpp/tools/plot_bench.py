"""
Plot benchmark.py results of several variants side by side.

usage: python3 plot_bench.py --out plot.png [--title T] [--metrics drift_pct,t_rel_pct,r_rel_deg100m] LABEL=DIR [...]
  first LABEL=DIR is the reference (drawn gray), the rest get categorical colors in fixed order.
  one row per metric:
  left : value per sequence, hatched = tracked < 95 % (metric on the tracked part only), "x" = nothing tracked,
         bars above the axis limit (3x the reference median max) are clipped and labelled with their value
  right: median per variant, with the number of sequences tracked >= 95 %
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REF = "#9a9994"                                              # reference variant: neutral gray
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # categorical slots, fixed order
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"drift_pct": "drift %  (Sim(3) ATE / path)", "t_rel_pct": "t_rel %  (KITTI, 100-800 m)",
          "r_rel_deg100m": "r_rel deg/100m  (KITTI)", "scale_trend_pct_km": "scale trend %/km  (0 = no systematic drift)"}


def load(d):
    return {(r["suite"], r["seq"], r["mode"]): r for r in csv.DictReader(open(Path(d) / "results.csv"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--metrics", default="drift_pct")
    ap.add_argument("runs", nargs="+", help="LABEL=DIR")
    o = ap.parse_args()
    runs = [(s.split("=", 1)[0], load(s.split("=", 1)[1])) for s in o.runs]
    keys = sorted(set.intersection(*(set(r) for _, r in runs)))
    colors = [REF] + SERIES[: len(runs) - 1]
    metrics = o.metrics.split(",")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
    fig, axes = plt.subplots(len(metrics), 2, figsize=(15, 4.6 * len(metrics)), gridspec_kw={"width_ratios": [3.2, 1]},
                             squeeze=False)
    n = len(runs)
    w = 0.8 / n
    x = np.arange(len(keys))
    for row, met in enumerate(metrics):
        ax, ax2 = axes[row]
        val = lambda r, k: float(r[k][met])
        ref_vals = [val(runs[0][1], k) for k in keys]
        ymax = 3 * np.nanmax(ref_vals) if np.isfinite(np.nanmax(ref_vals)) else None
        for i, ((label, r), c) in enumerate(zip(runs, colors)):
            for xi, k in zip(x, keys):
                d, trk = val(r, k), float(r[k]["tracked_pct"])
                pos = xi - 0.4 + w * (i + 0.5)
                if np.isnan(d):
                    ax.text(pos, 0, "x", ha="center", va="bottom", color=INK2, fontsize=8)
                    continue
                clip = ymax is not None and d > ymax
                ax.bar(pos, min(d, ymax) if clip else d, w * 0.9, color=c, hatch="////" if trk < 95 else None,
                       edgecolor=SURFACE, linewidth=0.5, label=label if (xi == 0 and row == 0) else None)
                if clip:
                    ax.text(pos, ymax, f"{d:.0f}", ha="center", va="bottom", color=INK2, fontsize=7, rotation=90)
        if ymax is not None:
            ymin = min(0, np.nanmin([val(r, k) for _, r in runs for k in keys]))  # 음수 가능한 지표 (스케일 추세)
            ax.set_ylim(ymin * 1.12, ymax * 1.12)
        ax.set_xticks(x, [k[1] if k[0] == "kitti" else f"{k[1]}\n{k[2]}" for k in keys])
        ax.set_ylabel(LABELS.get(met, met))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if row == 0:
            ax.legend(frameon=False, ncol=min(n, 6), loc="upper left", fontsize=9)
            ax.set_title("per sequence (hatched: tracked < 95 %, x: not tracked, clipped bars show their value)",
                         color=INK, fontsize=10, loc="left")
        med = [np.nanmedian([val(r, k) for k in keys]) for _, r in runs]
        ok = [sum(float(r[k]["tracked_pct"]) >= 95 for k in keys) for _, r in runs]
        y = np.arange(n)[::-1]
        xmax = 3 * med[0] if np.isfinite(med[0]) and med[0] > 0 else np.nanmax(med)  # 기준의 3배 넘으면 잘라서 표시
        ax2.barh(y, np.minimum(med, xmax), 0.6, color=colors)
        for yi, m, kk in zip(y, med, ok):
            ax2.text(min(m, xmax), yi, f"  {m:.3g}  ({kk}/{len(keys)} ok)", va="center", color=INK, fontsize=9)
        ax2.set_yticks(y, [label for label, _ in runs])
        ax2.set_xlim(0, xmax * 1.9)
        ax2.set_xlabel("median " + LABELS.get(met, met).split("  ")[0])
        ax2.grid(axis="x", color=GRID, linewidth=0.8)
        ax2.set_axisbelow(True)
        for sp in ("top", "right"):
            ax2.spines[sp].set_visible(False)
        if row == 0:
            ax2.set_title("median over sequences", color=INK, fontsize=10, loc="left")
    if o.title:
        fig.suptitle(o.title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(o.out, dpi=120)
    print("saved", o.out)


if __name__ == "__main__":
    main()
