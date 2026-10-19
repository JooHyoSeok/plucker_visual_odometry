"""
Monocular point + Plücker-line VO on real data: KITTI odometry sequence 06 (left grayscale camera).

Data: KITTI 06 video + GT poses as redistributed in the pyslam repo (luigifreda/pyslam, data/videos/kitti06).
      1101 frames, 1226x370, ~10 Hz, ~1.2 km loop.

Frontend (frontend.py; real, no GT association)
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
import frontend as fe

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
fe.setup_camera(fx, fy, cx, cy, W, H)


def run_frontend():
    cap = cv2.VideoCapture(VIDEO)
    front = fe.Frontend()
    obs = []
    t0 = time.time()
    for k in range(N_MAX):
        ok, frame = cap.read()
        if not ok:
            break
        obs.append(front.step(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        if k % 100 == 0:
            print(f"  frontend {k:4d}: {len(front.pts)} pts, {len(front.lines)} lines  ({time.time() - t0:.0f}s)", flush=True)
    return obs


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
    init = fe.robust_init(obs[0][0], obs[v.INIT_FRAME][0])
    rel = {k: r for k in range(1, n) if (r := fe.relative_pose(obs[k - 1][0], obs[k][0])) is not None}
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