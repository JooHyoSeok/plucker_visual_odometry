"""
배치 파이프라인 결과를 영상으로 -> batch_vo.mp4  (먼저 python3 kitti_batch.py 실행)
  위   : 실제 카메라 영상. 노랑 점 = KLT 트랙, 청록 = 관측(LSD) 선분,
         자홍 = 추정 포즈로 다시 투영한 맵의 Plücker 직선, 초록 o = 다시 투영한 맵 점(일부)
  아래 왼쪽 : top view — GT / VO(points+lines) / loop closure + Sim(3) PGO (온라인: 루프가 잡힐 때마다 보정)
             원점 고정 정렬 (같은 점·방향에서 출발, 스케일만 처음 50프레임으로)
  아래 오른쪽: 위치 오차 — VO vs 루프 보정 (청록 세로선 = 루프 검출 시점)
"""
import pickle
import sys
import numpy as np
import cv2

import kitti_batch as vk
import vo_plucker as v

OUT = "batch_vo.mp4"
FPS = 20
CW = 1280
IMG_H = int(round(vk.H * CW / vk.W))       # 386
PANEL_H = 420
COL = {"gt": (235, 235, 235), "vo": (200, 110, 150), "lc": (60, 200, 60)}  # BGR


def clip_line(l):
    a, b, c = l
    pts = []
    if abs(b) > 1e-9:
        for x in (0, vk.W):
            y = -(a * x + c) / b
            if 0 <= y <= vk.H:
                pts.append((x, y))
    if abs(a) > 1e-9:
        for y in (0, vk.H):
            x = -(b * y + c) / a
            if 0 <= x <= vk.W:
                pts.append((x, y))
    return pts[:2] if len(pts) >= 2 else None


def main():
    d = pickle.load(open("batch_run_ba.pkl" if "--ba" in sys.argv else "batch_run.pkl", "rb"))
    obs, C_gt, res = d["obs"], d["C_gt"], d["results"]
    n = len(obs)

    R_ = res["points + lines"]
    C_vo, events, loops = R_["C"], R_["events"], R_["loops"]
    n_ok = R_["n_ok"]

    def online_centers(k):
        """프레임 k 시점에 알고 있는 마지막 루프 보정 결과 (없으면 VO)"""
        cur = C_vo
        for f, Cc in events:
            if f <= k:
                cur = Cc
        return cur

    err_vo = R_["err"]
    err_on = np.array([np.linalg.norm(online_centers(f)[f] - C_gt[f]) for f in range(n_ok)])

    # top-view 좌표 변환 (x, z) -> 패널 픽셀
    allxz = np.concatenate([C_gt[:, [0, 2]], C_vo[:, [0, 2]]] + [Cc[:, [0, 2]] for _, Cc in events])
    lo, hi = allxz.min(0) - 10, allxz.max(0) + 10
    tv_w, tv_h = 560, PANEL_H - 50
    sc = min(tv_w / (hi[0] - lo[0]), tv_h / (hi[1] - lo[1]))
    ox = 30 + (tv_w - sc * (hi[0] - lo[0])) / 2
    oy = 35

    def to_tv(xz):
        return np.c_[ox + sc * (xz[:, 0] - lo[0]), oy + tv_h - sc * (xz[:, 1] - lo[1])].astype(np.int32)

    cap = cv2.VideoCapture(vk.VIDEO)
    vw = cv2.VideoWriter("tmp_batch.avi", cv2.VideoWriter_fourcc(*"MJPG"), FPS, (CW, IMG_H + PANEL_H))
    hist_l = res["points + lines"]["hist"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for k in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        img = frame.copy()
        op, ol = obs[k]
        R, t, mp, ml = hist_l[k]
        for u in op.values():
            cv2.circle(img, (int(u[0]), int(u[1])), 1, (0, 220, 255), -1)
        for sgm in ol.values():
            cv2.line(img, (int(sgm[0]), int(sgm[1])), (int(sgm[2]), int(sgm[3])), (255, 255, 0), 3, cv2.LINE_AA)
        if ml:
            for l in v.project_lines(np.array(list(ml.values())), R, t):
                p = clip_line(l)
                if p:
                    cv2.line(img, (int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])), (255, 0, 255), 1, cv2.LINE_AA)
        if mp:
            X = np.array(list(mp.values()))[::4] @ R.T + t
            X = X[X[:, 2] > 0.1]
            uv = X @ v.K.T
            uv = uv[:, :2] / uv[:, 2:3]
            for u in uv:
                cv2.circle(img, (int(u[0]), int(u[1])), 4, (0, 255, 0), 1, cv2.LINE_AA)
        img = cv2.resize(img, (CW, IMG_H))
        cv2.rectangle(img, (0, 0), (CW, 34), (0, 0, 0), -1)
        cv2.putText(img, f"real data  frame {k}/{n - 1}   tracked pts {len(op)}  lines {len(ol)}  "
                    f"map lines in view {len(ml)}", (10, 24), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        panel = np.full((PANEL_H, CW, 3), 30, np.uint8)
        kk = min(k + 1, n_ok)
        Con = online_centers(k)
        # ---- top view ----
        cv2.putText(panel, "top view (same origin, scale from first 50 frames)", (30, 25), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.polylines(panel, [to_tv(C_gt[:, [0, 2]])], False, (70, 70, 70), 1, cv2.LINE_AA)
        cv2.polylines(panel, [to_tv(C_gt[: k + 1, [0, 2]])], False, COL["gt"], 2, cv2.LINE_AA)
        cv2.polylines(panel, [to_tv(C_vo[:kk, [0, 2]])], False, COL["vo"], 1, cv2.LINE_AA)
        cv2.polylines(panel, [to_tv(Con[:kk, [0, 2]])], False, COL["lc"], 2, cv2.LINE_AA)
        for i, j, scl, ninl in loops:
            if i <= k:
                pa, pb = to_tv(Con[[i, j]][:, [0, 2]])
                cv2.line(panel, tuple(pa), tuple(pb), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(to_tv(C_gt[k: k + 1, [0, 2]])[0]), 5, COL["gt"], -1)
        cv2.circle(panel, tuple(to_tv(Con[kk - 1: kk, [0, 2]])[0]), 5, COL["lc"], -1)
        y0 = PANEL_H - 92
        for i, (name, c) in enumerate([("ground truth", COL["gt"]), ("VO (points + lines)", COL["vo"]),
                                       ("+ loop closure, Sim(3) PGO", COL["lc"]), ("detected loop", (0, 255, 255))]):
            cv2.line(panel, (390, y0 + 22 * i), (420, y0 + 22 * i), c, 3)
            cv2.putText(panel, name, (428, y0 + 5 + 22 * i), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # ---- error plot ----
        px0, py0, pw, ph = 690, 45, 560, 290
        cv2.putText(panel, "position error [m]", (px0, 25), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (px0, py0), (px0 + pw, py0 + ph), (90, 90, 90), 1)
        emax = max(float(err_vo.max()), 1.0) * 1.05
        ys = lambda e_: (py0 + ph - np.clip(e_, 0, emax) / emax * ph).astype(np.int32)
        for lv in np.arange(0, emax, 50):
            y = int(ys(np.array([lv]))[0])
            cv2.line(panel, (px0, y), (px0 + pw, y), (60, 60, 60), 1)
            cv2.putText(panel, f"{lv:.0f}", (px0 - 38, y + 5), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        xs = (px0 + np.arange(kk) / (n - 1) * pw).astype(np.int32)
        if kk >= 2:
            cv2.polylines(panel, [np.c_[xs, ys(err_vo[:kk])]], False, COL["vo"], 1, cv2.LINE_AA)
            cv2.polylines(panel, [np.c_[xs, ys(err_on[:kk])]], False, COL["lc"], 2, cv2.LINE_AA)
        for i, j, scl, ninl in loops:
            if i <= k:
                x = int(px0 + i / (n - 1) * pw)
                cv2.line(panel, (x, py0), (x, py0 + ph), (0, 200, 200), 1)
        cv2.putText(panel, "frame", (px0 + pw - 50, py0 + ph + 22), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(panel, f"error now: VO {err_vo[kk-1]:.1f} m   with loop closure {err_on[kk-1]:.1f} m",
                    (px0, py0 + ph + 50), font, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        recent = [(i, j, scl) for i, j, scl, _ in loops if 0 <= k - i < 40]
        if recent:
            i, j, scl = recent[-1]
            cv2.rectangle(img, (0, IMG_H - 40), (CW, IMG_H), (0, 90, 90), -1)
            cv2.putText(img, f"LOOP CLOSED: frame {i} <-> {j}   scale ratio {scl:.2f}  ->  Sim(3) pose graph optimized",
                        (10, IMG_H - 13), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        vw.write(np.vstack([img, panel]))
        if k in (330, 1000):
            cv2.imwrite(f"batch_vo_frame{k}.png", np.vstack([img, panel]))
    vw.release()
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "tmp_batch.avi", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", OUT], check=True)
    print("saved", OUT)


if __name__ == "__main__":
    main()