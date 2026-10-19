"""
Benchmark run_vo over many sequences at once, so changes are judged on medians instead of one noisy run.

  suites : kitti  -> KITTI odometry 00-10 (image_2, or image_3 if image_2 is empty; intrinsics from calib.txt)
           icra   -> ICRA Stray Scanner sequences (rgb.mp4 + odometry.csv + camera_matrix.csv),
                     --resize 0.5 --stride 2 --init-min-points 50, start = first frame with GT speed > 0.3 m/s
                     (pure in-place rotation at the start cannot be initialized by monocular VO)
  modes  : mono (all), depth (icra: iPhone LiDAR depth/, confidence >= 1)
  metric : on the tracked part (from the first pose until tracking stops)
           drift % = Sim(3)-aligned ATE RMSE / GT path length * 100, ok = tracked >= 95 % of frames
           t_rel % / r_rel deg/100m = KITTI devkit relative error over 100..800 m segments (start every 10 frames),
           translation scaled by the Sim(3) scale (monocular) -> local accuracy, less dominated by where it drifted
           depth mode also: SE(3)-aligned ATE without scale fit (metric scale)

usage:
  python3 benchmark.py --out DIR [--suite kitti,icra] [--modes mono,depth] [--seqs REGEX] [--jobs 8]
                       [--max-frames N] [--args="extra run_vo args"] [--run-vo PATH]
  python3 benchmark.py --compare DIR_A DIR_B          # per-sequence drift %, A vs B
  python3 benchmark.py --reeval DIR                    # recompute results.csv from the saved trajectories/logs
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
KITTI = Path.home() / "Desktop/hyoseok/KITTI_Dataset/dataset/sequences"
KITTI_GT = Path.home() / "Desktop/hyoseok/KITTI_poses/dataset/poses"
ICRA = Path.home() / "Desktop/hyoseok/2026_ICRA_Dataset/ICRA_Final_Dataset"
FIELDS = ["suite", "seq", "mode", "frames", "tracked", "tracked_pct", "path_m", "ate_sim3_m", "drift_pct",
          "t_rel_pct", "r_rel_deg100m", "scale_trend_pct_km", "ate_se3_m", "scale", "fps", "fail"]


def kitti_jobs(seqs):
    for s in sorted(p.name for p in KITTI.iterdir() if p.is_dir()):
        if not re.search(seqs, s) or not (KITTI_GT / f"{s}.txt").exists():
            continue
        img = "image_2" if any((KITTI / s / "image_2").glob("*.png")) else "image_3"
        P = [float(x) for x in open(KITTI / s / "calib.txt").readline().split()[1:]]
        yield dict(suite="kitti", seq=s, mode="mono", gt=KITTI_GT / f"{s}.txt", stride=1, start=0,
                   args=["--video", f"{KITTI / s / img}/%06d.png", "--calib", f"{P[0]},{P[5]},{P[2]},{P[6]}"])


def icra_jobs(seqs, modes, out):
    for d in sorted(p for p in ICRA.iterdir() if (p / "rgb.mp4").exists() and (p / "odometry.csv").exists()):
        if not re.search(seqs, d.name):
            continue
        a = np.loadtxt(d / "odometry.csv", delimiter=",", skiprows=1)
        gt = out / "gt" / f"{d.name}.txt"
        if not gt.exists():
            gt.parent.mkdir(parents=True, exist_ok=True)
            R = Rotation.from_quat(a[:, 5:9]).as_matrix()
            np.savetxt(gt, np.concatenate([R, a[:, 2:5, None]], axis=2).reshape(-1, 12), fmt="%.9f")
        v = np.linalg.norm(a[30:, 2:5] - a[:-30, 2:5], axis=1)
        start = int(np.argmax(v > 0.3)) if (v > 0.3).any() else 0
        K = np.loadtxt(d / "camera_matrix.csv", delimiter=",")
        base = ["--video", str(d / "rgb.mp4"), "--calib", f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}",
                "--resize", "0.5", "--stride", "2", "--init-min-points", "50", "--start", str(start)]
        for m in modes:
            extra = [] if m == "mono" else ["--depth", str(d / "depth/%06d.png"), "--depth-conf",
                                            str(d / "confidence/%06d.png"), "--depth-conf-min", "1"]
            yield dict(suite="icra", seq=d.name, mode=m, gt=gt, stride=2, start=start, args=base + extra)


def run(job, run_vo, out, extra, max_frames):
    name = f"{job['suite']}_{job['seq']}_{job['mode']}"
    traj, log = out / f"{name}.txt", out / f"{name}.log"
    cmd = [run_vo] + ([str(max_frames)] if max_frames else []) + job["args"] + ["--no-viewer", "--traj", str(traj)] + extra
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return evaluate(job, traj, log.read_text(), time.time() - t0)


def kitti_rpe(Pe, Pg, s, lengths=range(100, 801, 100), step=10):
    """KITTI devkit relative error. Pe/Pg: (n,3,4) T_wc, est translations scaled by s.
    returns (t_rel %, r_rel deg/100m) averaged over all segments, nan if the path is shorter than 100 m"""
    T = lambda P: np.vstack([np.hstack([P[:, :3], P[:, 3:]]), [0, 0, 0, 1]])
    Te = [T(np.hstack([P[:, :3], s * P[:, 3:]])) for P in Pe]
    Tg = [T(P) for P in Pg]
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pg[:, :, 3], axis=0), axis=1))]
    te, re = [], []
    for i in range(0, len(Pg), step):
        for L in lengths:
            j = np.searchsorted(dist, dist[i] + L)
            if j >= len(Pg):
                continue
            E = np.linalg.inv(np.linalg.inv(Te[i]) @ Te[j]) @ (np.linalg.inv(Tg[i]) @ Tg[j])
            te.append(np.linalg.norm(E[:3, 3]) / L)
            re.append(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)) / L)
    return (100 * np.mean(te), np.degrees(np.mean(re)) * 100) if te else (np.nan, np.nan)


def scale_trend(C, G, win=200):
    """systematic scale drift [%/km]: local scale = VO path / GT path over `win` frames (median-normalized),
    log-linear fit against distance travelled. 0 = no trend (random-walk drift averages out), nan if path < 300 m"""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1))]
    lc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    ks = [k for k in range(len(G) - win) if d[k + win] - d[k] > 20]  # 거의 정지한 창 제외
    if d[-1] < 300 or len(ks) < 10:
        return np.nan
    loc = np.array([(lc[k + win] - lc[k]) / (d[k + win] - d[k]) for k in ks])
    slope = np.polyfit(d[ks] / 1000, np.log(loc / np.median(loc)), 1)[0]
    return float(100 * (np.exp(slope) - 1))


def evaluate(job, traj, log, wall):
    m = re.search(r"(\d+)/(\d+) frames tracked", log)
    tracked, frames = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    f0 = int(re.search(r"first pose at frame (\d+)", log).group(1)) if "first pose at frame" in log else 0
    fps = float(re.search(r"\| ([\d.]+) fps", log).group(1)) if re.search(r"\| ([\d.]+) fps", log) else 0
    fail = re.search(r"FAILED at \d+: (.*?) \|", log)
    row = dict(suite=job["suite"], seq=job["seq"], mode=job["mode"], frames=frames, tracked=tracked,
               tracked_pct=100 * tracked / max(frames, 1), fps=fps, fail=fail.group(1) if fail else "",
               path_m=0, ate_sim3_m=np.nan, drift_pct=np.nan, t_rel_pct=np.nan, r_rel_deg100m=np.nan,
               scale_trend_pct_km=np.nan, ate_se3_m=np.nan, scale=np.nan)
    if tracked < 3 or not traj.exists():
        return row
    Pe = np.loadtxt(traj).reshape(-1, 3, 4)
    P = np.loadtxt(job["gt"]).reshape(-1, 12)[job["start"]::job["stride"]][f0:f0 + len(Pe)]
    Pe = Pe[:len(P)]
    C, G = Pe[:, :, 3], P[:, [3, 7, 11]]
    row["path_m"] = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
    row["scale_trend_pct_km"] = scale_trend(C, G)
    for with_scale in (True, False):
        mc, mg = C.mean(0), G.mean(0)
        X, Y = C - mc, G - mg
        U, D, Vt = np.linalg.svd(Y.T @ X / len(C))
        S = np.eye(3)
        S[2, 2] = np.sign(np.linalg.det(U @ Vt))
        s = np.trace(np.diag(D) @ S) / (X ** 2).sum(1).mean() if with_scale else 1.0
        ate = float(np.sqrt(((s * X @ (U @ S @ Vt).T - Y) ** 2).sum(1).mean()))
        if with_scale:
            row["ate_sim3_m"], row["scale"] = ate, float(s)
            row["drift_pct"] = 100 * ate / max(row["path_m"], 1e-9)
            row["t_rel_pct"], row["r_rel_deg100m"] = kitti_rpe(Pe, P.reshape(-1, 3, 4), s)
        elif job["mode"] == "depth":
            row["ate_se3_m"] = ate
    return row


def summarize(rows):
    print(f"\n{'run':34s} {'tracked':>8s} {'path':>8s} {'Sim3 ATE':>9s} {'drift':>7s} {'t_rel':>7s} {'r_rel':>6s} "
          f"{'SE3 ATE':>8s} {'fps':>5s}  fail")
    for r in rows:
        print(f"{r['suite'] + '_' + r['seq'] + '_' + r['mode']:34s} {r['tracked_pct']:7.0f}% {r['path_m']:7.0f}m "
              f"{r['ate_sim3_m']:8.2f}m {r['drift_pct']:6.2f}% {r['t_rel_pct']:6.2f}% {r['r_rel_deg100m']:6.2f} "
              f"{r['ate_se3_m']:7.2f}m {r['fps']:5.1f}  {r['fail']}")
    print()
    for key in sorted({(r["suite"], r["mode"]) for r in rows}):
        g = [r for r in rows if (r["suite"], r["mode"]) == key]
        d = np.array([r["drift_pct"] for r in g], float)
        tr = np.array([r["t_rel_pct"] for r in g], float)
        rr = np.array([r["r_rel_deg100m"] for r in g], float)
        st = np.array([r["scale_trend_pct_km"] for r in g], float)
        ok = sum(r["tracked_pct"] >= 95 for r in g)
        print(f"[{key[0]} {key[1]}] {len(g)} runs | tracked >= 95%: {ok}/{len(g)} | drift % median {np.nanmedian(d):.2f}, "
              f"mean {np.nanmean(d):.2f} | t_rel % median {np.nanmedian(tr):.2f} | r_rel deg/100m median "
              f"{np.nanmedian(rr):.3f} | scale trend %/km median {np.nanmedian(st):+.1f} | mean fps {np.mean([float(r['fps']) for r in g]):.1f}")


def compare(a, b):
    load = lambda d: {(r["suite"], r["seq"], r["mode"]): r for r in csv.DictReader(open(Path(d) / "results.csv"))}
    A, B = load(a), load(b)
    print(f"{'run':34s} {'A drift':>8s} {'B drift':>8s} {'A trk':>6s} {'B trk':>6s}")
    da, db = [], []
    for k in sorted(set(A) & set(B)):
        x, y = float(A[k]["drift_pct"]), float(B[k]["drift_pct"])
        da.append(x), db.append(y)
        print(f"{'_'.join(k):34s} {x:7.2f}% {y:7.2f}% {float(A[k]['tracked_pct']):5.0f}% {float(B[k]['tracked_pct']):5.0f}%")
    print(f"median drift %: A {np.nanmedian(da):.2f} | B {np.nanmedian(db):.2f}   (B better on "
          f"{sum(y < x for x, y in zip(da, db))}/{len(da)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--suite", default="kitti,icra")
    ap.add_argument("--modes", default="mono,depth")
    ap.add_argument("--seqs", default=".")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--args", default="", help='extra run_vo arguments for every run, as --args="--opt ..."')
    ap.add_argument("--run-vo", default=str(HERE.parent / "build/run_vo"))
    ap.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    ap.add_argument("--reeval", metavar="DIR")
    o = ap.parse_args()
    if o.compare:
        return compare(*o.compare)
    if o.reeval:
        out = Path(o.reeval)
        jobs = list(kitti_jobs(".")) + list(icra_jobs(".", ["mono", "depth"], out))
        rows = [evaluate(j, out / f"{j['suite']}_{j['seq']}_{j['mode']}.txt",
                         (out / f"{j['suite']}_{j['seq']}_{j['mode']}.log").read_text(), 0)
                for j in jobs if (out / f"{j['suite']}_{j['seq']}_{j['mode']}.log").exists()]
        with open(out / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows)
        return summarize(rows)
    if not o.out:
        sys.exit(__doc__)
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    suites, modes = o.suite.split(","), o.modes.split(",")
    jobs = (list(kitti_jobs(o.seqs)) if "kitti" in suites and "mono" in modes else []) + \
           (list(icra_jobs(o.seqs, modes, out)) if "icra" in suites else [])
    print(f"{len(jobs)} runs, {o.jobs} in parallel, run_vo = {o.run_vo} {o.args}")
    t0 = time.time()
    with ThreadPoolExecutor(o.jobs) as ex:
        rows = list(ex.map(lambda j: run(j, o.run_vo, out, o.args.split(), o.max_frames), jobs))
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"\n{time.time() - t0:.0f} s, results in {out}/results.csv")


if __name__ == "__main__":
    main()
