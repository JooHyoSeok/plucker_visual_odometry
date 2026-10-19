"""
Result video of run_vo runs on one sequence: camera image (top), top view of the trajectories so far (bottom left),
position error vs distance so far (bottom right). Runs are Sim(3)-aligned to GT with their final trajectory (as for
the ATE); a run with an online log (--traj-online) is drawn with the pose estimated at that moment, so drift building
up and the jump at a loop correction are visible ("LOOP a <-> b" flashes when the correction lands).

usage: python3 make_video.py --images 'seq/image_2/%06d.png' --gt poses/00.txt --out out.mp4 [--every 2] [--fps 30]
                             [--title T] [--align sim3|origin] [--panel error|scale] LABEL=PREFIX [...]
  --align origin: start pose on GT, scale from the first 100 m only (as the viewer) -> drift shows as divergence
  --panel scale : bottom right = local scale (VO / GT distance over 20 frames, 1 = scale of the first 100 m)
  PREFIX.txt          final trajectory (run_vo --traj), starts at the first tracked frame
  PREFIX.log          run_vo stdout (first pose frame, loop events), optional
  PREFIX_online.txt   run_vo --traj-online, optional (+ PREFIX_online_snap.txt: whole trajectory at each loop correction,
                      so past poses are redrawn corrected like in the viewer)
  first run is drawn gray (reference), the rest in categorical colors; GT dashed dark.
"""
import argparse
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

GRAY, SERIES = (148, 153, 154), [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237)]  # BGR
BG, INK, INK2, GRID = (251, 252, 252), (11, 11, 11), (78, 81, 82), (223, 227, 228)


def sim3(C, G):  # Umeyama: G ~ s R C + t
    mc, mg = C.mean(0), G.mean(0)
    U, D, Vt = np.linalg.svd((G - mg).T @ (C - mc) / len(C))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    s = np.trace(np.diag(D) @ S) / ((C - mc) ** 2).sum(1).mean()
    R = U @ S @ Vt
    return lambda X: s * (X - mc) @ R.T + mg


def origin_align(C, G):  # 시작 포즈를 GT 에 맞추고 스케일은 처음 100 m (최소 20 프레임) 로만
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1))]
    n = max(20, int(np.searchsorted(d, 100)))
    s = d[n] / max(np.linalg.norm(np.diff(C[: n + 1], axis=0), axis=1).sum(), 1e-9)
    return lambda X: s * (X - C[0]) + G[0]  # KITTI 처럼 시작 방향이 같은 world 가정 (VO world = 첫 카메라)


def load_run(prefix, G, how="sim3"):
    pre = Path(prefix)
    log = pre.with_suffix(".log").read_text() if pre.with_suffix(".log").exists() else ""
    m = re.search(r"first pose at frame (\d+)", log)
    f0 = int(m.group(1)) if m else 0
    C = np.loadtxt(pre.with_suffix(".txt")).reshape(-1, 12)[:, [3, 7, 11]][: len(G) - f0]
    align = (sim3 if how == "sim3" else origin_align)(C, G[f0:f0 + len(C)])
    final_ate = float(np.sqrt((np.linalg.norm(align(C) - G[f0:f0 + len(C)], axis=1) ** 2).mean()))
    pos = np.full((len(G), 3), np.nan)
    snaps = {}  # 보정이 들어온 프레임 k -> 그때 궤적 (len(G), 3)
    on = Path(str(pre) + "_online.txt")
    if on.exists():
        a = np.loadtxt(on)
        k = a[:, 0].astype(int)
        pos[k[k < len(G)]] = align(a[k < len(G)][:, [4, 8, 12]])
        sn = Path(str(pre) + "_online_snap.txt")
        if sn.exists() and sn.stat().st_size:
            b = np.loadtxt(sn)
            for kk in np.unique(b[:, 0]).astype(int):
                r = b[b[:, 0] == kk]
                P = np.full((len(G), 3), np.nan)
                f = r[:, 1].astype(int)
                P[f[f < len(G)]] = align(r[f < len(G)][:, 2:5])
                snaps[kk] = P
    else:
        pos[f0:f0 + len(C)] = align(C)
    # 보정이 실제로 들어간 프레임 (스레드: "tracking at frame N", 동기식: 검출 키프레임 직후)
    loops = [(int(m.group(3)) if m.group(3) else int(m.group(1)) + 1, int(m.group(1)), int(m.group(2)))
             for m in re.finditer(r"loop @frame (\d+) <-> (\d+).*?(?:tracking at frame (\d+))?\)?$", log, re.M)
             if "PGO cost" in m.group(0)]
    return pos, loops, final_ate, snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--title", default="")
    ap.add_argument("--align", default="sim3", choices=["sim3", "origin"])
    ap.add_argument("--panel", default="error", choices=["error", "scale"])
    ap.add_argument("runs", nargs="+", help="LABEL=PREFIX")
    o = ap.parse_args()
    G = np.loadtxt(o.gt).reshape(-1, 12)[:, [3, 7, 11]]
    runs = []
    for i, spec in enumerate(o.runs):
        label, prefix = spec.split("=", 1)
        pos, loops, final_ate, snaps = load_run(prefix, G, o.align)
        runs.append(dict(label=label, pos=pos, loops=loops, final_ate=final_ate, err=np.linalg.norm(pos - G, axis=1),
                         snaps=snaps, snap_keys=np.array(sorted(snaps)), color=GRAY if i == 0 else SERIES[(i - 1) % len(SERIES)]))
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1))]

    img0 = cv2.imread(o.images % 0)
    W = 1280
    ih = int(round(img0.shape[0] * W / img0.shape[1] / 2)) * 2
    ph = 420
    H = ih + ph
    # 위에서 본 화면 (x 오른쪽, z 앞): 전체 시퀀스 범위로 고정
    allp = np.vstack([G] + [r["pos"][~np.isnan(r["pos"][:, 0])] for r in runs])[:, [0, 2]]
    lo, hi = allp.min(0), allp.max(0)
    pw = W // 2
    sc = min((pw - 40) / max(hi[0] - lo[0], 1e-6), (ph - 50) / max(hi[1] - lo[1], 1e-6))
    off = np.array([pw / 2, ih + 30 + (ph - 50) / 2]) - sc * np.array([(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2])
    to_px = lambda P: np.c_[off[0] + sc * P[:, 0], off[1] - sc * P[:, 2]].astype(np.int32)
    W20 = 20  # 로컬 스케일 창
    for r in runs:  # 지금 프레임까지 시스템이 믿은 위치 기준 로컬 스케일
        P = r["pos"]
        r["scale"] = np.full(len(G), np.nan)
        for k in range(W20, len(G)):
            dg = np.linalg.norm(G[k] - G[k - W20])
            if dg > 3 and not np.isnan(P[k, 0]) and not np.isnan(P[k - W20, 0]):
                r["scale"][k] = np.linalg.norm(P[k] - P[k - W20]) / dg
        r["curve"] = r["err"] if o.panel == "error" else r["scale"]
    if o.panel == "error":
        emax = np.nanpercentile(np.concatenate([r["err"] for r in runs]), 99) * 1.1 + 1e-6
        emin = 0.0
    else:
        allc = np.concatenate([r["scale"] for r in runs])
        emin, emax = 0.0, max(2.0, np.nanpercentile(allc, 99.5) * 1.05)
    ex0, ex1, ey0, ey1 = pw + 50, W - 20, ih + 40, H - 40
    to_err = lambda d, e: np.c_[ex0 + (ex1 - ex0) * d / dist[-1],
                                ey1 - (ey1 - ey0) * (np.clip(e, emin, emax) - emin) / (emax - emin)].astype(np.int32)

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(o.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                           "-pix_fmt", "yuv420p", o.out], stdin=subprocess.PIPE)
    gt_px = to_px(G)
    for k in range(0, len(G), o.every):
        img = cv2.imread(o.images % k)
        if img is None:
            break
        F = np.full((H, W, 3), BG, np.uint8)
        F[:ih] = cv2.resize(img, (W, ih), interpolation=cv2.INTER_AREA)
        cv2.line(F, (pw, ih), (pw, H), GRID, 1)
        for a in range(0, len(gt_px) - 1, 6):  # GT 점선
            if (a // 6) % 2 == 0:
                cv2.line(F, tuple(gt_px[a]), tuple(gt_px[min(a + 6, len(gt_px) - 1)]), INK2, 1, cv2.LINE_AA)
        cv2.rectangle(F, (ex0, ey0), (ex1, ey1), GRID, 1)
        if o.panel == "error":
            cv2.putText(F, "position error now [m] vs distance", (ex0, ey0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK2, 1, cv2.LINE_AA)
        else:
            cv2.putText(F, "local scale (VO / GT, 20 frames; 1 = first 100 m)", (ex0, ey0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK2, 1, cv2.LINE_AA)
            y1 = int(ey1 - (ey1 - ey0) * (1 - emin) / (emax - emin))
            cv2.line(F, (ex0, y1), (ex1, y1), INK2, 1, cv2.LINE_AA)
            cv2.putText(F, "1", (ex0 - 15, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, INK2, 1, cv2.LINE_AA)
        cv2.putText(F, f"{emax:.0f}" if o.panel == "error" else f"{emax:.1f}", (ex0 - 40, ey0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, INK2, 1, cv2.LINE_AA)
        cv2.putText(F, "0", (ex0 - 15, ey1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, INK2, 1, cv2.LINE_AA)
        cv2.putText(F, f"{dist[-1] / 1000:.1f} km", (ex1 - 50, ey1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, INK2, 1, cv2.LINE_AA)
        y_txt = ih + 22
        for r in runs:
            ok = ~np.isnan(r["pos"][: k + 1, 0])
            P = r["pos"][: k + 1][ok]
            # 지금 시점에 시스템이 믿는 궤적: 가장 최근 보정 스냅샷 + 그 뒤 프레임은 그때그때의 포즈 (뷰어와 같음)
            Q = r["pos"][: k + 1].copy()
            ks = r["snap_keys"][r["snap_keys"] <= k] if len(r["snap_keys"]) else []
            if len(ks):
                S = r["snaps"][ks[-1]]
                Q[: ks[-1] + 1] = S[: ks[-1] + 1]
            Q = Q[~np.isnan(Q[:, 0])]
            if len(P) > 1:
                cv2.polylines(F, [to_px(Q)], False, r["color"], 2, cv2.LINE_AA)
                cv2.circle(F, tuple(to_px(P[-1:])[0]), 5, r["color"], -1, cv2.LINE_AA)
                e = r["err"][: k + 1][ok]
                cv_ = r["curve"][: k + 1]
                okc = ~np.isnan(cv_)
                if okc.sum() > 1:
                    cv2.polylines(F, [to_err(dist[: k + 1][okc], cv_[okc])], False, r["color"], 2, cv2.LINE_AA)
                n_loop = sum(f <= k for f, _, _ in r["loops"])
                txt = f"{r['label']}: now {e[-1]:5.1f} m | final {'ATE' if o.align == 'sim3' else 'RMSE (origin)'} {r['final_ate']:5.1f} m" + \
                      (f" | loops {n_loop}" if r["loops"] else "")
                cv2.putText(F, txt, (12, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.45, r["color"], 1, cv2.LINE_AA)
                y_txt += 20
            recent = [(f, a, b) for f, a, b in r["loops"] if 0 <= k - f < 15]  # 보정이 들어간 직후 잠깐, 최근 것 하나
            if recent:
                _, a, b = max(recent)
                cv2.putText(F, f"LOOP {a} <-> {b}", (W - 330, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5, cv2.LINE_AA)
                cv2.putText(F, f"LOOP {a} <-> {b}", (W - 330, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, r["color"], 2, cv2.LINE_AA)
        cv2.putText(F, f"{o.title}  frame {k}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(F, f"{o.title}  frame {k}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INK, 1, cv2.LINE_AA)
        cv2.putText(F, "GT dashed, " + ("Sim(3)-aligned with the final trajectory" if o.align == "sim3" else
                                         "start pose on GT, scale from the first 100 m only"), (12, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, INK2, 1, cv2.LINE_AA)
        ff.stdin.write(F.tobytes())
    ff.stdin.close()
    ff.wait()
    print("saved", o.out)


if __name__ == "__main__":
    main()
