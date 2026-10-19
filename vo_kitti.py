"""
Monocular point + Plücker-line VO on real data: KITTI odometry sequence 06 (left grayscale camera).

Data: KITTI 06 video + GT poses as redistributed in the pyslam repo (luigifreda/pyslam, data/videos/kitti06).
      1101 frames, 1226x370, ~10 Hz, ~1.2 km loop.

Frontend (real, no GT association)
  points : Shi-Tomasi + KLT (forward-backward check) + fundamental-matrix RANSAC
  lines  : LSD detection; each line is tracked by KLT on 7 samples along it, refit, and then
           snapped to the nearest LSD detection in the new frame (angle + perpendicular distance)
Backend (vo_plucker.run_vo, "decoupled" mode)
  init     : cv2.findEssentialMat (RANSAC) + recoverPose, then Sampson refinement
  per frame: rotation + translation direction from 2D-2D (essential RANSAC + Sampson refinement),
             translation magnitude (scale) from the map: point reprojection + Plücker-line reprojection
             -> 6-DoF map-only PnP had a systematic ~0.09 deg/frame yaw bias on KITTI (no BA), this fixes it
  mapping  : N-view triangulation of points and lines (Plücker from stacked back-projected planes)

usage:  python3 vo_kitti.py [n_frames] [--ba] [--video PATH] [--gt PATH] [--calib fx,fy,cx,cy]
        python3 make_video_kitti.py [같은 --video ... 인자]
"""
import sys
import time
import pickle
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vo_plucker as v

# ---------------- 입력 설정 (명령행 인자) ----------------
#   --video PATH   : 영상 파일(mp4 등) 또는 이미지 시퀀스 패턴 (예: .../sequences/06/image_0/%06d.png)
#   --gt PATH      : KITTI 포맷 GT 포즈 (한 줄 = 3x4 T_w_c 12개 값)
#   --calib fx,fy,cx,cy : 핀홀 내부 파라미터 (왜곡 없는 rectified 영상 가정)
#   기본값 = pyslam 저장소의 KITTI 06 (data_dl/pyslam/data/videos/kitti06)
def _arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


DATA = "data_dl/pyslam/data/videos/kitti06"
VIDEO = _arg("--video", f"{DATA}/video.mp4")
GT_FILE = _arg("--gt", f"{DATA}/06.txt")
N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100000

# KITTI 내부 파라미터: 00-02 = 718.856,718.856,607.1928,185.2157 / 03 = 721.5377,721.5377,609.5593,172.854
#                      04-12 = 707.0912,707.0912,601.8873,183.1104
fx, fy, cx, cy = map(float, _arg("--calib", "707.0912,707.0912,601.8873,183.1104").split(","))
_cap = cv2.VideoCapture(VIDEO)
_ok, _f0 = _cap.read()
if not _ok:
    raise SystemExit(f"cannot read video/images: {VIDEO}")
H, W = _f0.shape[:2]
_cap.release()
v.fx, v.fy, v.cx, v.cy, v.W, v.H = fx, fy, cx, cy, W, H
v.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
v.Kinv = np.linalg.inv(v.K)
v.KL = np.array([[fy, 0, 0], [0, fx, 0], [-fy * cx, -fx * cy, fx * fy]])
v.PIX_NOISE = 1.0
v.INIT_FRAME = 3          # KITTI는 프레임당 ~1 m 전진 -> 3프레임이면 baseline 충분
v.TRI_GAP = 2
# KITTI는 먼 점이 많음 -> 점 삼각측량 최소 parallax를 1° -> 2°로 (3.9% -> 3.6% ATE, 전체 시퀀스 기준)
import functools
v.point_ok = functools.partial(v.point_ok, min_deg=2.0)

# ---------------- frontend params ----------------
MAX_PTS, MIN_PTS = 1000, 800
MAX_LINES, MIN_LINES = 150, 100
MIN_LINE_LEN = 50
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
N_SAMPLES = 7


def klt(prev, cur, p0, fb_thresh=1.0, min_eig=1e-4, normals=None, guess=None):
    """forward-backward KLT. returns tracked pts and validity mask.
    min_eig: 선 위 점은 gradient가 한 방향뿐이라(rank-1) 기본값(1e-4)이면 status=0으로 버려짐 -> 낮춰서 사용"""
    if len(p0) == 0:
        return p0, np.zeros(0, bool)
    p0 = p0.reshape(-1, 1, 2).astype(np.float32)
    if guess is None:
        p1, s1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, minEigThreshold=min_eig, **LK)
    else:                                   # 초기 flow 추정값에서 시작 (큰 움직임 대응)
        p1, s1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, guess.reshape(-1, 1, 2).astype(np.float32).copy(),
                                             minEigThreshold=min_eig, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **LK)
    p0b, s2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, p0.copy(), minEigThreshold=min_eig,
                                          flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **LK)
    dfb = (p0 - p0b).reshape(-1, 2)
    # 선 샘플: edge를 따라 미끄러지는 성분(aperture)은 의미 없으니 법선 방향 FB 오차만 본다
    fb = np.linalg.norm(dfb, axis=1) if normals is None else np.abs((dfb * normals).sum(1))
    p1 = p1.reshape(-1, 2)
    ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (fb < fb_thresh)
    ok &= (p1[:, 0] >= 0) & (p1[:, 0] < W) & (p1[:, 1] >= 0) & (p1[:, 1] < H)
    return p1, ok


def seg_samples(seg):
    s = np.linspace(0.05, 0.95, N_SAMPLES)[:, None]
    return seg[:2][None] * (1 - s) + seg[2:][None] * s


def fit_segment(pts):
    """total least squares line fit -> (segment from extreme projections, max perpendicular residual)"""
    mu = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - mu)
    d = Vt[0]
    n = Vt[1]
    res = np.abs((pts - mu) @ n).max()
    s = (pts - mu) @ d
    return np.concatenate([mu + s.min() * d, mu + s.max() * d]), res


def snap_to_lsd(seg, det, det_dir, det_mid):
    """tracked segment -> nearest LSD detection (angle < 3 deg, perpendicular dist < 3 px, overlapping)"""
    if len(det) == 0:
        return None
    d = seg[2:] - seg[:2]
    L = np.linalg.norm(d)
    d = d / L
    n = np.array([-d[1], d[0]])
    cosang = np.abs(det_dir @ d)
    perp = np.abs((det_mid - seg[:2]) @ n)
    along = (det_mid - seg[:2]) @ d
    cand = np.where((cosang > np.cos(np.radians(3))) & (perp < 3.0) & (along > -0.2 * L) & (along < 1.2 * L))[0]
    if len(cand) == 0:
        return None
    best = cand[np.argmin(perp[cand])]
    return det[best]


def run_frontend():
    cap = cv2.VideoCapture(VIDEO)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    obs = []
    pts = np.zeros((0, 2), np.float32)
    pid = np.zeros(0, int)
    lines = np.zeros((0, 4))
    lid = np.zeros(0, int)
    next_p = next_l = 0
    prev = None
    t0 = time.time()
    for k in range(N_MAX):
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        det = lsd.detect(g)[0]
        det = np.zeros((0, 4)) if det is None else det.reshape(-1, 4).astype(float)
        det = det[np.linalg.norm(det[:, 2:] - det[:, :2], axis=1) >= MIN_LINE_LEN]
        det_dir = det[:, 2:] - det[:, :2]
        det_dir /= np.linalg.norm(det_dir, axis=1, keepdims=True) + 1e-12
        det_mid = (det[:, :2] + det[:, 2:]) / 2

        if prev is not None:
            # ---- track points and line samples in one KLT call ----
            samp = np.concatenate([seg_samples(s) for s in lines]) if len(lines) else np.zeros((0, 2))
            # 선 위 샘플은 edge를 따라 미끄러지는 aperture 문제가 있음 -> FB는 법선 성분만 검사,
            # 선 방향 위치는 어차피 의미 없음 (Plücker 잔차는 직선까지 거리만 봄)
            np_ = len(pts)
            p1p, okp = klt(prev, g, pts, 1.0)
            nrm = []
            for sg in lines:
                d = (sg[2:] - sg[:2]) / np.linalg.norm(sg[2:] - sg[:2])
                nrm.append(np.repeat([[-d[1], d[0]]], N_SAMPLES, axis=0))
            nrm = np.concatenate(nrm) if nrm else np.zeros((0, 2))
            # 선 샘플 초기 위치 = 주변 점 트랙 5개의 중앙값 flow 만큼 이동시킨 곳
            guess = samp.copy()
            if okp.sum() >= 5 and len(samp):
                good0, flow = pts[okp], (p1p.reshape(-1, 2) - pts)[okp]
                d2 = ((samp[:, None, :] - good0[None]) ** 2).sum(-1)
                nn = np.argpartition(d2, 5, axis=1)[:, :5]
                guess = samp + np.median(flow[nn], axis=1)
            p1l, okl = klt(prev, g, samp, 1.0, 1e-6, normals=nrm, guess=guess)
            p1, okm = np.concatenate([p1p.reshape(-1, 2), p1l.reshape(-1, 2)]), np.concatenate([okp, okl])
            pts_new, ok_p = p1[:np_], okm[:np_]

            # RANSAC fundamental on point tracks
            if ok_p.sum() >= 8:
                _, inl = cv2.findFundamentalMat(pts[ok_p], pts_new[ok_p], cv2.FM_RANSAC, 1.0, 0.999)
                if inl is not None:
                    idx = np.where(ok_p)[0]
                    ok_p[idx[inl.ravel() == 0]] = False
            pts, pid = pts_new[ok_p], pid[ok_p]

            new_lines, new_lid = [], []
            for i in range(len(lines)):
                sl = slice(np_ + i * N_SAMPLES, np_ + (i + 1) * N_SAMPLES)
                q, okq = p1[sl], okm[sl]
                if okq.sum() < 4:
                    continue
                seg, res = fit_segment(q[okq])
                if res > 1.5:
                    continue
                snapped = snap_to_lsd(seg, det, det_dir, det_mid)
                if snapped is None:
                    continue
                if np.linalg.norm(snapped[2:] - snapped[:2]) < MIN_LINE_LEN * 0.8:
                    continue
                new_lines.append(snapped)
                new_lid.append(lid[i])
            lines = np.array(new_lines).reshape(-1, 4)
            lid = np.array(new_lid, int)

        # ---- replenish points ----
        if len(pts) < MIN_PTS:
            mask = np.full(g.shape, 255, np.uint8)
            for p in pts:
                cv2.circle(mask, (int(p[0]), int(p[1])), 8, 0, -1)
            newp = cv2.goodFeaturesToTrack(g, MAX_PTS - len(pts), 0.005, 8, mask=mask)
            if newp is not None:
                newp = newp.reshape(-1, 2)
                pts = np.concatenate([pts, newp]).astype(np.float32)
                pid = np.concatenate([pid, np.arange(next_p, next_p + len(newp))])
                next_p += len(newp)

        # ---- replenish lines ----
        if len(lines) < MIN_LINES and len(det):
            mask = np.full(g.shape, 255, np.uint8)
            for s in lines:
                cv2.line(mask, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), 0, 10)
            order = np.argsort(-np.linalg.norm(det[:, 2:] - det[:, :2], axis=1))
            add = []
            for i in order:
                if len(lines) + len(add) >= MAX_LINES:
                    break
                sp = seg_samples(det[i]).astype(int)
                sp[:, 0] = np.clip(sp[:, 0], 0, W - 1)
                sp[:, 1] = np.clip(sp[:, 1], 0, H - 1)
                if (mask[sp[:, 1], sp[:, 0]] == 0).sum() >= 3:
                    continue
                add.append(det[i])
                cv2.line(mask, (int(det[i][0]), int(det[i][1])), (int(det[i][2]), int(det[i][3])), 0, 10)
            if add:
                lines = np.concatenate([lines, np.array(add)])
                lid = np.concatenate([lid, np.arange(next_l, next_l + len(add))])
                next_l += len(add)

        obs.append(({int(j): p.astype(float) for j, p in zip(pid, pts)},
                    {int(j): s.astype(float) for j, s in zip(lid, lines)}))
        prev = g
        if k % 100 == 0:
            print(f"  frontend {k:4d}: {len(pts)} pts, {len(lines)} lines  ({time.time() - t0:.0f}s)", flush=True)
    return obs


def robust_init(obs):
    op0, opI = obs[0][0], obs[v.INIT_FRAME][0]
    ids = sorted(set(op0) & set(opI))
    p1 = np.array([op0[j] for j in ids])
    p2 = np.array([opI[j] for j in ids])
    E, inl = cv2.findEssentialMat(p1, p2, v.K, cv2.RANSAC, 0.999, 1.0)
    _, R, t, inl2 = cv2.recoverPose(E, p1, p2, v.K, mask=inl)
    m = inl2.ravel() > 0
    x1 = (np.c_[p1, np.ones(len(p1))] @ v.Kinv.T)[m, :2]
    x2 = (np.c_[p2, np.ones(len(p2))] @ v.Kinv.T)[m, :2]
    R, t = v.refine_relative_pose(R, t.ravel() / np.linalg.norm(t), x1, x2)
    print(f"  init: {m.sum()}/{len(ids)} inliers")
    return R, t


def relative_poses(obs):
    """연속 프레임 사이 (R_rel, t방향): essential RANSAC + Sampson 정제. 스케일은 백엔드가 맵으로 결정."""
    rel = {}
    for k in range(1, len(obs)):
        a, b = obs[k - 1][0], obs[k][0]
        ids = [j for j in b if j in a]
        if len(ids) < 15:
            continue
        x1 = (np.c_[np.array([a[j] for j in ids]), np.ones(len(ids))] @ v.Kinv.T)[:, :2]
        x2 = (np.c_[np.array([b[j] for j in ids]), np.ones(len(ids))] @ v.Kinv.T)[:, :2]
        E, inl = cv2.findEssentialMat(x1, x2, np.eye(3), cv2.RANSAC, 0.999, 1.0 / fx)
        if E is None or E.shape != (3, 3):
            continue
        _, R, t, inl2 = cv2.recoverPose(E, x1, x2, np.eye(3), mask=inl)
        m = inl2.ravel() > 0
        if m.sum() < 15:
            continue
        R, t = v.refine_relative_pose(R, t.ravel(), x1[m], x2[m], iters=15)
        rel[k] = (R, t)
    return rel


def load_gt(n):
    P = np.loadtxt(GT_FILE)[:n]
    return P[:, [3, 7, 11]]          # camera centers (T_w_c translation)


_P0 = np.loadtxt(GT_FILE, max_rows=1).reshape(3, 4)   # GT frame 0 (T_w_c) -> T_cw
GT_R0, GT_t0 = _P0[:, :3].T, -_P0[:, :3].T @ _P0[:, 3]


if __name__ == "__main__":
    import loop_closure as lc
    import local_ba as lba
    USE_BA = "--ba" in sys.argv
    print(f"video {VIDEO}, gt {GT_FILE}, {W}x{H}, K=({fx},{fy},{cx},{cy}), local BA: {USE_BA}")
    obs = run_frontend()
    n = len(obs)
    C_gt = load_gt(n)
    init = robust_init(obs)
    rel = relative_poses(obs)
    kfs = list(range(0, n, 3))
    t0 = time.time()
    feats = lc.keyframe_features(VIDEO, kfs)
    print(f"  ORB keyframe features: {time.time() - t0:.0f}s")
    path_len = np.linalg.norm(np.diff(C_gt, axis=0), axis=1).sum()
    rmse = lambda e: np.sqrt((e ** 2).mean())

    results = {}
    for name, ul in [("points only", False), ("points + lines", True)]:
        t0 = time.time()
        hist = {}
        hook = lba.make_ba_hook() if USE_BA else None
        Rs, ts, mp, ml, st = v.run_vo(obs, ul, history=hist, init_pose=init, rel_pose=rel, post_frame_hook=hook)
        n_ok = n if st["failed_at"] is None else st["failed_at"]
        Rs, ts = Rs[:n_ok], ts[:n_ok]
        # 원점 고정 정렬 (같은 점·방향에서 출발, 스케일만 처음 50프레임으로)
        C_vo = v.align_first_frame(Rs, ts, GT_R0, GT_t0, C_gt[:n_ok])
        err_vo = np.linalg.norm(C_vo - C_gt[:n_ok], axis=1)
        t1 = time.time()
        lcr = lc.run_loop_closure(Rs, ts, feats, [k for k in kfs if k < n_ok])
        C_lc = lc.anchor_centers(lcr["centers"], GT_R0, GT_t0, C_gt[:n_ok])
        err_lc = np.linalg.norm(C_lc - C_gt[:n_ok], axis=1)
        events = [(f, lc.anchor_centers(Cc, GT_R0, GT_t0, C_gt[:n_ok])) for f, Cc in lcr["events"]]
        results[name] = dict(C=C_vo, err=err_vo, C_lc=C_lc, err_lc=err_lc, n_ok=n_ok, hist=hist, events=events,
                             loops=[(i, j, S[0], k) for i, j, S, k in lcr["loops"]],
                             poses=(Rs, ts))
        print(f"[{name:>14}] {n_ok}/{n} frames ({path_len:.0f} m) | ATE VO {rmse(err_vo):.1f} m "
              f"({100 * rmse(err_vo) / path_len:.1f}%) -> + loop closure / Sim(3) PGO {rmse(err_lc):.1f} m "
              f"({100 * rmse(err_lc) / path_len:.1f}%) | loops {len(lcr['loops'])} | "
              f"VO {t1 - t0:.0f}s, loop {time.time() - t1:.0f}s" + ("" if st["failed_at"] is None else f" | FAILED at {n_ok}"))

    out = "kitti06_run_ba.pkl" if USE_BA else "kitti06_run.pkl"
    with open(out, "wb") as f:
        pickle.dump(dict(obs=obs, C_gt=C_gt, results=results, kfs=kfs, use_ba=USE_BA), f)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].plot(C_gt[:, 0], C_gt[:, 2], "k-", lw=2.5, label="ground truth")
    r = results["points + lines"]
    ax[0].plot(r["C"][:, 0], r["C"][:, 2], ":", color="tab:purple", lw=1.4, label="VO (points + lines)")
    ax[0].plot(results["points only"]["C_lc"][:, 0], results["points only"]["C_lc"][:, 2], "--", color="tab:orange",
               lw=1.6, label="+ loop/Sim(3) PGO (points only)")
    ax[0].plot(r["C_lc"][:, 0], r["C_lc"][:, 2], "-", color="tab:purple", lw=1.8, label="+ loop/Sim(3) PGO (points + lines)")
    for i, j, sc, k in r["loops"]:
        ax[0].plot([r["C_lc"][i, 0], r["C_lc"][j, 0]], [r["C_lc"][i, 2], r["C_lc"][j, 2]], "g-", lw=2)
    ax[0].set(title="KITTI 06 (origin-anchored; green = detected loops)", xlabel="x [m]", ylabel="z [m]", aspect="equal")
    ax[0].legend(fontsize=8, loc="lower right")
    for name, c in [("points only", "tab:orange"), ("points + lines", "tab:purple")]:
        ax[1].plot(results[name]["err"], ":", color=c, label=f"VO ({name})")
        ax[1].plot(results[name]["err_lc"], "-", color=c, label=f"+ loop/PGO ({name})")
    for i, j, sc, k in r["loops"]:
        ax[1].axvline(i, color="g", lw=0.8, alpha=0.6)
    ax[1].set(title="position error (origin-anchored)", xlabel="frame", ylabel="[m]")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    png = "kitti06_result_ba.png" if USE_BA else "kitti06_result.png"
    plt.savefig(png, dpi=120)
    print(f"saved {png}, {out}")