"""
Paper-style LaTeX table of KITTI ATE (Sim(3) RMSE [m], sequence 01 excluded) with our run next to published numbers,
top-3 per column coloured (1st bold + green, 2nd light green, 3rd yellow), compiled to PDF and PNG.

usage: python3 kitti_table_tex.py OUT_PREFIX "GROUP TITLE=RESULTS_DIR" [...]
  each RESULTS_DIR (kitti_SS_mono.txt/.log + _pregba.txt) becomes a group with rows "Ours w/o global BA" and "Ours"
  (the last group's "Ours" in bold) -> OUT_PREFIX.tex / .pdf / .png
Baselines: DROID-SLAM ... CLoSeR from the CLoSeR paper (Table 2), ORB-SLAM monocular from Mur-Artal et al., TRO 2015.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402

SEQS = ["00", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
FRAMES = [4541, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201]
SCALE_KM = [3.7, 5.1, 0.6, 0.4, 2.2, 1.2, 0.7, 3.2, 1.7, 0.9]
LOOP = [1, 1, 0, 0, 1, 1, 1, 0, 1, 0]
N = float("nan")
GROUPS = [
    ("Optimization-based (Calibrated)", True, [
        ("DROID-SLAM", [92.10, 107.61, 2.38, 1.00, 118.50, 62.47, 21.78, 161.60, 72.32, 118.70]),
        ("DPV-SLAM", [112.80, 123.53, 2.50, 0.81, 57.80, 54.86, 18.77, 110.49, 76.66, 13.65]),
        ("DPV-SLAM++", [8.30, 39.64, 2.50, 0.78, 5.74, 11.60, 1.52, 110.90, 76.70, 13.70])]),
    ("Feedforward (Uncalibrated)", False, [
        ("VGGT-Long", [8.67, 32.08, 6.12, 4.23, 8.31, 5.34, 4.63, 53.10, 41.99, 18.37]),
        ("VGGT-SLAM 2.0", [108.39, 140.15, 13.45, 2.87, 10.12, 29.17, 17.27, 75.33, 27.70, 24.09]),
        ("FastVGGT", [N, N, 21.53, 9.51, N, 40.56, 51.35, N, 201.54, 196.22]),
        ("InfiniteVGGT", [186.46, 289.16, 166.74, 68.00, 143.84, 117.57, 85.33, 221.56, 215.41, 156.92]),
        ("CUT3R", [190.38, 264.39, 20.40, 7.31, 92.25, 67.54, 22.48, 145.08, 67.42, 40.00]),
        ("TTT3R", [119.94, 238.07, 16.83, 3.98, 36.38, 47.20, 11.62, 107.33, 86.96, 33.58]),
        ("AMB3R", [177.78, 233.65, 22.08, 6.81, 157.62, 61.93, 35.02, 255.53, 134.62, 50.41]),
        ("$\\pi^3$-Chunk", [26.65, 157.92, 5.13, 1.09, 12.79, 27.66, 5.94, 61.26, 56.31, 21.96]),
        ("LingBot-Map", [23.22, 71.15, 2.81, 0.91, 17.40, 4.91, 5.16, 14.24, 21.71, 14.54]),
        ("LoGeR", [62.34, 39.64, 4.89, 1.82, 41.27, 13.99, 16.24, 26.46, 22.71, 8.84]),
        ("LoGeR*", [30.47, 36.32, 5.38, 1.95, 26.34, 6.60, 5.55, 24.41, 10.12, 10.11]),
        ("CLoSeR", [6.58, 28.38, 4.88, 1.82, 6.58, 4.77, 4.87, 26.51, 5.64, 8.83])]),
    ("Feature-based (Calibrated)", False, [
        ("ORB-SLAM", [6.68, 21.75, 1.59, 1.79, 8.23, 14.68, 3.36, 46.58, 7.62, 8.68]),
        ("ORB-SLAM + GBA", [5.33, 21.28, 1.51, 1.62, 4.85, 12.34, 2.26, 46.68, 6.62, 8.80])]),
]
COLORS = ["c1", "c2", "c3"]  # 1st / 2nd / 3rd per column


def ours(d, suffix):
    jobs = {j["seq"]: j for j in bm.kitti_jobs("^(" + "|".join(SEQS) + ")$")}
    vals = []
    for s in SEQS:
        log, traj = d / f"kitti_{s}_mono.log", d / f"kitti_{s}_mono{suffix}.txt"
        ok = s in jobs and log.exists() and traj.exists()
        vals.append(bm.evaluate(jobs[s], traj, log.read_text(errors="replace"), 0)["ate_sim3_m"] if ok else N)
    return vals


def main():
    if len(sys.argv) < 3 or any("=" not in a for a in sys.argv[2:]):
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    mine = [a.split("=", 1) for a in sys.argv[2:]]
    groups = GROUPS + [(t, False, [("Ours w/o global BA", ours(Path(d), "_pregba")),
                                   ("\\textbf{Ours}" if k == len(mine) - 1 else "Ours", ours(Path(d), ""))])
                       for k, (t, d) in enumerate(mine)]
    rows = [(n, v + [np.mean(v) if not np.isnan(v).any() else N]) for _, _, g in groups for n, v in g]
    rank = {}  # (row, col) -> 0 / 1 / 2
    for c in range(len(SEQS) + 1):
        col = sorted({round(v[c], 2) for _, v in rows if not np.isnan(v[c])})
        for i, (_, v) in enumerate(rows):
            if not np.isnan(v[c]) and round(v[c], 2) in col[:3]:
                rank[i, c] = col.index(round(v[c], 2))

    n_base = sum(len(g) for _, _, g in GROUPS)  # 이 앞 행은 논문 숫자 (빈 칸 = OOM), 뒤는 우리 실행 (빈 칸 = 안 돌림)

    def cell(i, c, v, gray):
        if np.isnan(v):
            return "OOM" if c < len(SEQS) and i < n_base else "--"
        t = f"{v:.2f}"
        if (i, c) in rank:
            t = f"\\cellcolor{{{COLORS[rank[i, c]]}}}" + (f"\\textbf{{{t}}}" if rank[i, c] == 0 else t)
        return f"\\textcolor{{gray}}{{{t}}}" if gray and (i, c) not in rank else t

    ncol = len(SEQS) + 2
    head = " & ".join(f"\\texttt{{{s}}}" for s in SEQS)
    L = [r"\documentclass[border=6pt]{standalone}", r"\usepackage{newtxtext,newtxmath}", r"\usepackage[table]{xcolor}",  # colortbl 먼저, 그 뒤 arydshln (반대면 점선 열 ':' 이 깨짐)
         r"\usepackage{booktabs,arydshln}", r"\usepackage{pifont}", r"\newsavebox{\tbox}",
         r"\definecolor{c1}{RGB}{176,222,199}\definecolor{c2}{RGB}{214,234,190}\definecolor{c3}{RGB}{252,246,196}",
         r"\begin{document}", r"\setlength{\tabcolsep}{4.2pt}",
         r"\sbox{\tbox}{\begin{tabular}{l" + "r" * len(SEQS) + ":r}", r"\toprule",
         f"Method & {head} & \\textbf{{Avg.}} \\\\", r"\midrule",
         r"\textit{\#Frames} & " + " & ".join(map(str, FRAMES)) + f" & {np.mean(FRAMES):.0f} \\\\",
         r"\textit{Scale} & " + " & ".join(f"{x}km" for x in SCALE_KM) + f" & {np.mean(SCALE_KM):.1f}km \\\\",
         r"\textit{Contains Loop?} & " + " & ".join("\\ding{51}" if x else "\\ding{55}" for x in LOOP) + r" & -- \\",
         r"\midrule"]
    i = 0
    for gi, (gname, gray, g) in enumerate(groups):
        if gi:
            L.append(r"\midrule")
        L.append(f"\\rowcolor{{gray!12}}\\multicolumn{{{ncol}}}{{c}}{{\\textbf{{\\textit{{{gname}}}}}}} \\\\")
        for n, _ in g:
            v = rows[i][1]
            name = f"\\textcolor{{gray}}{{{n}}}" if gray else n
            L.append(f"{name} & " + " & ".join(cell(i, c, v[c], gray) for c in range(ncol - 1)) + " \\\\")
            i += 1
    L += [r"\bottomrule", r"\end{tabular}}",  # 표 너비를 재서 설명 문단을 같은 너비로
          r"\begin{minipage}{\wd\tbox}",
          r"\textbf{Comparison of absolute trajectory error} (ATE RMSE $\downarrow$ [m]) on KITTI, sequence \texttt{01} excluded "
          r"(not initialised by our system). Colours: 1st (bold) / 2nd / 3rd per column over all methods. Baselines from the "
          r"CLoSeR paper and ORB-SLAM (TRO 2015, keyframe trajectories); ours: one threaded real-time run, Sim(3)-aligned "
          r"(same as \texttt{evo\_ape -as}).\\[5pt]",
          r"\usebox{\tbox}", r"\end{minipage}", r"\end{document}"]
    tex = out.with_suffix(".tex")
    tex.write_text("\n".join(L) + "\n")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=tex.parent, check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["pdftoppm", "-png", "-r", "250", "-singlefile", str(out.with_suffix(".pdf")), str(out)], check=True)
    for ext in (".aux", ".log"):
        out.with_suffix(ext).unlink(missing_ok=True)
    print("saved", tex, out.with_suffix(".pdf"), out.with_suffix(".png"))


if __name__ == "__main__":
    main()
