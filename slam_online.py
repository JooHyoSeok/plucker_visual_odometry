"""
Online monocular SLAM with points + Plücker lines: feed one image, get one pose.

  every frame : frontend (KLT points + LSD lines) -> 2D-2D relative pose -> tracking (scale from the map)
                -> mapping -> (optional) local BA
  keyframe    : ORB + thumbnail -> loop check on the previous keyframe -> Sim(3) PGO -> map correction
  feedback    : the PGO result is written back into poses, map points, map lines, loop 3D points and old
                loop edges, so the next frame is tracked against the corrected map (--no-feedback: PGO result
                is only reported, VO keeps its own map = same as the batch vo_kitti.py pipeline)

library:
    slam = OnlineSLAM(fx, fy, cx, cy, W, H)
    for img in images:                 # BGR or grayscale, rectified
        T_wc = slam.track(img)         # 4x4 camera->world, None while initializing / after tracking lost
CLI:
    python3 slam_online.py [n_frames] --video PATH|CAM_INDEX [--gt PATH] [--calib fx,fy,cx,cy]
                           [--ba] [--no-lines] [--no-feedback] [--show] [--out video.mp4]
    python3 slam_online.py --selftest
"""
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import cv2

import vo_plucker as v
import frontend as fe
import loop_closure as lc
import local_ba as lba

KF_EVERY = 3        # 루프 검사용 키프레임 간격 [frames] (배치와 동일)


class OnlineSLAM:
    def __init__(self, fx, fy, cx, cy, W, H, use_lines=True, loop=True, feedback=True, ba=False):
        fe.setup_camera(fx, fy, cx, cy, W, H)
        self.front = fe.Frontend()
        self.vo = v.VOTracker(use_lines, post_frame_hook=lba.make_ba_hook() if ba else None)
        self.loop, self.feedback = loop, feedback
        self.orb = cv2.ORB_create(1500, fastThreshold=10)
        self.feats, self.l3d, self.kfs = {}, {}, []
        self.loops = []                     # 검출된 루프 (i, j, 측정 S_ij: cam_j -> cam_i, n_inliers)
        # PGO 엣지는 측정값 원본을 보관 (보정 때 단위만 맞춤): odom[a] = 키프레임 a -> 다음 키프레임, loop_edges = (i, j, S_ij)
        self.odom, self.loop_edges = {}, []
        self.S_kf = None                    # --no-feedback: 마지막 PGO 결과 (보고용)
        self.last_p, self.last_l = {}, {}   # 랜드마크를 마지막으로 본 프레임 (맵 보정 기준)

    # ---------------- public ----------------
    def track(self, img):
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vo = self.vo
        k = len(vo.obs)
        op, ol = self.front.step(g)
        rel = fe.relative_pose(vo.obs[-1][0], op) if k else None
        init = fe.robust_init(vo.obs[0][0], op) if k == v.INIT_FRAME else None
        vo.add(op, ol, rel=rel, init_pose=init)
        for j in op:
            self.last_p[j] = k
        for j in ol:
            self.last_l[j] = k
        if self.loop and k % KF_EVERY == 0 and vo.stats["failed_at"] is None:
            self._keyframe(k, g)
        return self.pose(k)

    def pose(self, k):
        """프레임 k의 4x4 T_wc (현재 world 기준, 루프 보정 후 바뀔 수 있음) 또는 None"""
        R, t = self.vo.Rs[k], self.vo.ts[k]
        if R is None:
            return None
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R.T, -R.T @ t
        return T

    def n_ok(self):
        """앞에서부터 포즈가 연속으로 있는 프레임 수"""
        return next((f for f, R in enumerate(self.vo.Rs) if R is None), len(self.vo.Rs))

    def centers(self):
        """현재 최선의 카메라 중심 (n_ok, 3). no-feedback 모드에서는 VO + 마지막 PGO 결과를 전파한 값"""
        n = self.n_ok()
        Rs, ts = self.vo.Rs[:n], self.vo.ts[:n]
        if self.feedback or self.S_kf is None:
            return np.array([v.center(R, t) for R, t in zip(Rs, ts)]).reshape(-1, 3)
        return lc.propagate(Rs, ts, sorted(self.S_kf), self.S_kf, upto=n)

    # ---------------- loop closure ----------------
    def _keyframe(self, k, g):
        Rs, ts = self.vo.Rs, self.vo.ts
        self.feats[k] = lc.frame_features(self.orb, g)
        self.kfs.append(k)
        if len(self.kfs) < 2 or Rs[k] is None:
            return
        a = self.kfs[-2]                    # 직전 키프레임: 이제 다음 키프레임이 생겨서 로컬 3D 계산 가능
        self.l3d.update(lc.local_3d(self.feats, [a, k], Rs, ts))
        res = lc.detect_loop(a, self.feats, self.l3d, self.kfs[:-2])
        if not res:
            return
        j, S, ninl = res
        self.loops.append((a, j, S, ninl))
        self.loop_edges.append((a, j, S))
        sub = self.kfs[:-1]                 # 키프레임 a까지 (a 이후 프레임은 a 기준 상대포즈로 따라감)
        S_kf, cost = self._pgo(sub)
        print(f"    loop @frame {a} <-> {j}: {ninl} Sim(3) inliers, scale ratio {S[0]:.2f} -> PGO cost {cost:.1f}"
              + (" -> map corrected" if self.feedback else ""), flush=True)
        if self.feedback:
            self._correct(S_kf, sub)
        else:
            self.S_kf = S_kf

    def _pgo(self, sub):
        """키프레임 sub 전체 Sim(3) PGO. 초기값 = 현재 포즈, 엣지 = 보관된 측정값.
        아직 없는 오도메트리 엣지는 지금 포즈로 측정 (마지막 보정 이후 VO 그대로라 원본 측정과 같음).
        보정된 포즈로 오도메트리를 다시 재면 이전 보정이 측정값에 섞여 들어가서 joint 문제가 아니게 됨"""
        Rs, ts = self.vo.Rs, self.vo.ts
        S0 = [(1.0, Rs[f], ts[f]) for f in sub]
        pos = {f: n for n, f in enumerate(sub)}
        for a, b in zip(sub[:-1], sub[1:]):
            if a not in self.odom:
                self.odom[a] = lc.sim3_mul(S0[pos[b]], lc.sim3_inv(S0[pos[a]]))      # cam_a -> cam_b
        edges = [(pos[a], pos[b], self.odom[a], lc.W_ODOM) for a, b in zip(sub[:-1], sub[1:])]
        edges += [(pos[j], pos[i], S_ij, lc.W_LOOP) for i, j, S_ij in self.loop_edges]   # cam_j -> cam_i
        S, cost = lc.pgo(S0, edges, fixed=(0,))
        return {f: S[pos[f]] for f in sub}, cost

    def _correct(self, S_kf, sub):
        """PGO 결과를 되먹임. S_kf[a] = 키프레임 a의 Sim(3) world->cam (X_c = s R X_w + t, X_c는 VO 단위).
        비키프레임 f는 직전 키프레임 기준 VO 상대포즈를 붙여 S_f를 만들고, 새 SE(3) 포즈 = (R_f, t_f / s_f)
        (카메라 좌표가 1/s_f 배). 랜드마크는 마지막으로 본 프레임 f 기준으로 X <- S_f^-1 S_f,old X.
        보관 중인 PGO 엣지 측정값(cam_j -> cam_i)도 같은 단위로: S_ij <- (1/s_i) S_ij (s_j)."""
        vo = self.vo
        Rs, ts = vo.Rs, vo.ts
        corr, scale, new = {}, {}, {}
        kk = 0
        for f in range(len(Rs)):
            if Rs[f] is None:
                continue
            while kk + 1 < len(sub) and sub[kk + 1] <= f:
                kk += 1
            a = sub[kk]
            Rr = Rs[f] @ Rs[a].T
            tr = ts[f] - Rr @ ts[a]
            sa, Ra, ta = S_kf[a]
            S_f = (sa, Rr @ Ra, Rr @ ta + tr)
            corr[f] = lc.sim3_mul(lc.sim3_inv(S_f), (1.0, Rs[f], ts[f]))     # old world -> new world
            scale[f] = sa
            new[f] = (S_f[1], S_f[2] / sa)
        for f, (R, t) in new.items():
            Rs[f], ts[f] = R, t
        for j, X in vo.map_p.items():
            s, R, t = corr[self.last_p[j]]
            vo.map_p[j] = s * R @ X + t
        for j, L in vo.map_l.items():
            s, R, t = corr[self.last_l[j]]
            d = R @ L[:3]
            vo.map_l[j] = np.concatenate([d, s * R @ L[3:] + np.cross(t, d)])   # m = p x d 규약
        for a in self.l3d:                                                    # 카메라 좌표 = 1/s 배
            self.l3d[a] = {i: X / scale[a] for i, X in self.l3d[a].items()}
        rescale = lambda S, sj, si: (S[0] * sj / si, S[1], S[2] / si)
        nxt = dict(zip(self.kfs[:-1], self.kfs[1:]))
        self.odom = {a: rescale(M, scale[a], scale[nxt[a]]) for a, M in self.odom.items()}
        self.loop_edges = [(i, j, rescale(S, scale[j], scale[i])) for i, j, S in self.loop_edges]


# ============================================================
# 시각화 (프레임마다 한 장)
# ============================================================
CW, PANEL_H, FPS = 1280, 420, 20
COL = {"gt": (235, 235, 235), "vo": (200, 110, 150), "est": (60, 200, 60), "loop": (0, 255, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def clip_line(l, W, H):
    a, b, c = l
    pts = []
    if abs(b) > 1e-9:
        for x in (0, W):
            y = -(a * x + c) / b
            if 0 <= y <= H:
                pts.append((x, y))
    if abs(a) > 1e-9:
        for y in (0, H):
            x = -(b * y + c) / a
            if 0 <= x <= W:
                pts.append((x, y))
    return pts[:2] if len(pts) >= 2 else None


class Viewer:
    """위: 영상 + 추적 점/선 + 맵 투영. 아래: top view (+ GT 있으면 정렬 후 오차 그래프)"""

    def __init__(self, W, H, gt=None):
        self.W, self.H = W, H
        self.img_h = int(round(H * CW / W))
        self.gt = gt                        # (C_gt, R0, t0) 또는 None
        self.lo = self.hi = None
        self.err = []                       # 각 프레임 시점의 온라인 오차

    def aligned(self, C):
        if self.gt is None or not len(C):
            return C
        C_gt, R0, t0 = self.gt
        return lc.anchor_centers(C, R0, t0, C_gt[: len(C)])

    def draw(self, slam, frame, k):
        vo = slam.vo
        img = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        op, ol = vo.obs[k]
        R, t = vo.Rs[k], vo.ts[k]
        for u in op.values():
            cv2.circle(img, (int(u[0]), int(u[1])), 1, (0, 220, 255), -1)
        for s in ol.values():
            cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (255, 255, 0), 3, cv2.LINE_AA)
        ml = [vo.map_l[j] for j in ol if j in vo.map_l]
        if R is not None and ml:
            for l in v.project_lines(np.array(ml), R, t):
                p = clip_line(l, self.W, self.H)
                if p:
                    cv2.line(img, (int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])), (255, 0, 255), 1, cv2.LINE_AA)
        mp = [vo.map_p[j] for j in op if j in vo.map_p]
        if R is not None and mp:
            X = np.array(mp)[::4] @ R.T + t
            X = X[X[:, 2] > 0.1]
            uv = X @ v.K.T
            for u in uv[:, :2] / uv[:, 2:3]:
                cv2.circle(img, (int(u[0]), int(u[1])), 4, (0, 255, 0), 1, cv2.LINE_AA)
        img = cv2.resize(img, (CW, self.img_h))
        cv2.rectangle(img, (0, 0), (CW, 34), (0, 0, 0), -1)
        state = "LOST" if vo.stats["failed_at"] is not None else ("init" if R is None else "tracking")
        cv2.putText(img, f"frame {k}  {state}  tracked pts {len(op)}  lines {len(ol)}  map lines in view {len(ml)}  "
                    f"loops {len(slam.loops)}", (10, 24), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        recent = [(i, j, S[0]) for i, j, S, _ in slam.loops if 0 <= k - i < 40]
        if recent:
            i, j, sc = recent[-1]
            cv2.rectangle(img, (0, self.img_h - 40), (CW, self.img_h), (0, 90, 90), -1)
            cv2.putText(img, f"LOOP CLOSED: frame {i} <-> {j}   scale ratio {sc:.2f}  ->  Sim(3) PGO"
                        + (" + map corrected" if slam.feedback else ""),
                        (10, self.img_h - 13), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([img, self.panel(slam)])

    def panel(self, slam):
        panel = np.full((PANEL_H, CW, 3), 30, np.uint8)
        C = self.aligned(slam.centers())
        C_vo = None if slam.feedback else self.aligned(np.array([v.center(R, t) for R, t in
                                                                 zip(slam.vo.Rs[:len(C)], slam.vo.ts[:len(C)])]))
        n = len(C)
        if not n:
            return panel
        C_gt = None if self.gt is None else self.gt[0]
        # ---- top view: 범위는 늘어나기만 함 (화면이 흔들리지 않게) ----
        xz = np.concatenate([C[:, [0, 2]]] + ([] if C_gt is None else [C_gt[:, [0, 2]]]))
        lo, hi = xz.min(0), xz.max(0)
        self.lo = lo if self.lo is None else np.minimum(self.lo, lo)
        self.hi = hi if self.hi is None else np.maximum(self.hi, hi)
        pad = 0.05 * (self.hi - self.lo).max() + 1.0
        lo, hi = self.lo - pad, self.hi + pad
        tv_w, tv_h = 560, PANEL_H - 50
        sc = min(tv_w / (hi[0] - lo[0]), tv_h / (hi[1] - lo[1]))
        ox, oy = 30 + (tv_w - sc * (hi[0] - lo[0])) / 2, 35

        def to_tv(p):
            return np.c_[ox + sc * (p[:, 0] - lo[0]), oy + tv_h - sc * (p[:, 2] - lo[1])].astype(np.int32)

        cv2.putText(panel, "top view" + ("" if C_gt is None else " (same origin, scale from first 50 frames)"),
                    (30, 25), FONT, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        legend = [("estimate (online)", COL["est"]), ("detected loop", COL["loop"])]
        if C_gt is not None:
            cv2.polylines(panel, [to_tv(C_gt)], False, (70, 70, 70), 1, cv2.LINE_AA)
            cv2.polylines(panel, [to_tv(C_gt[:n])], False, COL["gt"], 2, cv2.LINE_AA)
            legend.insert(0, ("ground truth", COL["gt"]))
        if C_vo is not None:
            cv2.polylines(panel, [to_tv(C_vo)], False, COL["vo"], 1, cv2.LINE_AA)
            legend.insert(1, ("VO (no feedback)", COL["vo"]))
        cv2.polylines(panel, [to_tv(C)], False, COL["est"], 2, cv2.LINE_AA)
        for i, j, _, _ in slam.loops:
            if i < n and j < n:
                pa, pb = to_tv(C[[i, j]])
                cv2.line(panel, tuple(pa), tuple(pb), COL["loop"], 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(to_tv(C[-1:])[0]), 5, COL["est"], -1)
        y0 = PANEL_H - 22 * len(legend) - 4
        for i, (name, c) in enumerate(legend):
            cv2.line(panel, (390, y0 + 22 * i), (420, y0 + 22 * i), c, 3)
            cv2.putText(panel, name, (428, y0 + 5 + 22 * i), FONT, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        if C_gt is None:
            return panel

        # ---- error plot: 각 프레임에서 그 순간의 추정 오차 ----
        self.err.append(float(np.linalg.norm(C[-1] - C_gt[n - 1])))
        e = np.array(self.err)
        px0, py0, pw, ph = 690, 45, 560, 290
        cv2.putText(panel, "position error at estimation time [m]", (px0, 25), FONT, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (px0, py0), (px0 + pw, py0 + ph), (90, 90, 90), 1)
        emax = max(float(e.max()), 1.0) * 1.05
        step = 10.0 ** np.floor(np.log10(emax))
        for lv in np.arange(0, emax, step):
            y = int(py0 + ph - lv / emax * ph)
            cv2.line(panel, (px0, y), (px0 + pw, y), (60, 60, 60), 1)
            cv2.putText(panel, f"{lv:.0f}", (px0 - 38, y + 5), FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        if len(e) >= 2:
            xs = px0 + np.arange(len(e)) / (len(e) - 1) * pw
            cv2.polylines(panel, [np.c_[xs, py0 + ph - e / emax * ph].astype(np.int32)], False, COL["est"], 2, cv2.LINE_AA)
        cv2.putText(panel, f"error now {e[-1]:.1f} m   RMSE of current trajectory "
                    f"{np.sqrt(((C - C_gt[:n]) ** 2).sum(1).mean()):.1f} m",
                    (px0, py0 + ph + 50), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        return panel


# ============================================================
# self-check: 맵 보정이 관측 기하를 보존하는지
# ============================================================
def _selftest():
    rng = np.random.default_rng(0)
    slam = OnlineSLAM.__new__(OnlineSLAM)          # 카메라 설정 없이 (vo_plucker 기본 K 사용)
    slam.vo = v.VOTracker(True)
    Rs, ts = slam.vo.Rs, slam.vo.ts
    for f in range(12):
        Rs.append(v.so3_exp(rng.normal(0, 0.3, 3)))
        ts.append(rng.normal(0, 2, 3))
    sub = [0, 3, 6, 9]
    S_kf = {a: (1.0 if a == 0 else float(np.exp(rng.normal(0, 0.3))),
                v.so3_exp(rng.normal(0, 0.1, 3)) @ Rs[a], ts[a] + rng.normal(0, 0.5, 3)) for a in sub}
    slam.last_p, slam.last_l = {}, {}
    before = {}
    for j in range(40):
        f = int(rng.integers(12))
        Xc = np.r_[rng.uniform(-3, 3), rng.uniform(-2, 2), rng.uniform(4, 20)]
        slam.vo.map_p[j] = Rs[f].T @ (Xc - ts[f])
        slam.last_p[j] = f
        A, B = (Rs[f].T @ (np.r_[rng.uniform(-3, 3, 2), rng.uniform(4, 20)] - ts[f]) for _ in range(2))
        slam.vo.map_l[j] = v.plucker_from_endpoints(A, B)
        slam.last_l[j] = f
        uv = v.K @ Xc
        l = v.project_lines(slam.vo.map_l[j][None], Rs[f], ts[f])[0]
        before[j] = (uv[:2] / uv[2], l / np.linalg.norm(l[:2]))
    X3 = np.r_[0.5, -0.3, 8.0]
    S93 = lc.sim3_mul((1.0, Rs[9], ts[9]), lc.sim3_inv((1.0, Rs[3], ts[3])))   # cam3 -> cam9
    X9 = S93[0] * S93[1] @ X3 + S93[2]
    M36 = lc.sim3_mul((1.0, Rs[6], ts[6]), lc.sim3_inv((1.0, Rs[3], ts[3])))   # odometry cam3 -> cam6
    X6 = M36[1] @ X3 + M36[2]
    slam.kfs, slam.odom = sub, {3: M36}
    slam.l3d, slam.loop_edges = {3: {0: X3.copy()}}, [(9, 3, S93)]

    slam._correct(S_kf, sub)

    for j, (u0, l0) in before.items():
        f = slam.last_p[j]
        Xc = Rs[f] @ slam.vo.map_p[j] + ts[f]
        assert np.allclose((v.K @ Xc)[:2] / (v.K @ Xc)[2], u0, atol=1e-6), "point reprojection changed"
        l = v.project_lines(slam.vo.map_l[j][None], Rs[f], ts[f])[0]
        l = l / np.linalg.norm(l[:2])
        assert np.allclose(l, l0, atol=1e-6) or np.allclose(l, -l0, atol=1e-6), "line reprojection changed"
    for a in sub:
        s, R, t = S_kf[a]
        assert np.allclose(Rs[a], R) and np.allclose(ts[a], t / s), "keyframe pose != PGO result"
    S = slam.loop_edges[0][2]
    s3, s9 = S_kf[3][0], S_kf[9][0]
    assert np.allclose(S[0] * S[1] @ (X3 / s3) + S[2], X9 / s9), "loop edge not rescaled"
    M = slam.odom[3]
    assert np.allclose(M[0] * M[1] @ (X3 / s3) + M[2], X6 / S_kf[6][0]), "odometry edge not rescaled"
    assert np.allclose(slam.l3d[3][0], X3 / s3), "loop 3D not rescaled"
    print("selftest OK")


# ============================================================
# CLI: 영상 / 이미지 시퀀스 / 카메라에서 한 장씩 읽어서 track()
# ============================================================
def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    src = _arg("--video", None)
    if src is None:
        raise SystemExit(__doc__)
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10 ** 9
    fx, fy, cx, cy = map(float, _arg("--calib", "707.0912,707.0912,601.8873,183.1104").split(","))
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"cannot read video/images/camera: {src}")
    H, W = frame.shape[:2]
    slam = OnlineSLAM(fx, fy, cx, cy, W, H, use_lines="--no-lines" not in sys.argv,
                      feedback="--no-feedback" not in sys.argv, ba="--ba" in sys.argv)
    gt = None
    if _arg("--gt", None):
        P = np.loadtxt(_arg("--gt", None))
        R0 = P[0].reshape(3, 4)[:, :3].T
        gt = (P[:, [3, 7, 11]], R0, -R0 @ P[0].reshape(3, 4)[:, 3])
    show, out = "--show" in sys.argv, _arg("--out", None)
    viewer = Viewer(W, H, gt) if (show or out) else None
    writer = None
    print(f"source {src}, {W}x{H}, K=({fx},{fy},{cx},{cy}), lines {slam.vo.use_lines}, "
          f"feedback {slam.feedback}, local BA {'--ba' in sys.argv}")

    t0 = time.time()
    k = 0
    while ok and k < n_max:
        slam.track(frame)
        if viewer is not None:
            canvas = viewer.draw(slam, frame, k)
            if out:
                if writer is None:
                    tmp = out + ".tmp.avi" if shutil.which("ffmpeg") else out      # ffmpeg 있으면 x264로 재인코딩
                    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*("MJPG" if tmp != out else "mp4v")),
                                             FPS, canvas.shape[1::-1])
                writer.write(canvas)
            if show:
                cv2.imshow("slam_online", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if k % 100 == 0:
            print(f"  frame {k:4d}: {len(slam.vo.map_p)} map pts, {len(slam.vo.map_l)} map lines, "
                  f"loops {len(slam.loops)}  ({(k + 1) / (time.time() - t0):.1f} fps)", flush=True)
        k += 1
        ok, frame = cap.read()
    if writer is not None:
        writer.release()
        if tmp != out:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "26", out], check=True)
            os.remove(tmp)
        print("saved", out)

    n = slam.n_ok()
    np.savetxt("online_traj.txt", np.array([slam.pose(f)[:3].ravel() for f in range(n)]).reshape(-1, 12), fmt="%.6e")
    print(f"{n}/{k} frames tracked, {len(slam.loops)} loops, {k / (time.time() - t0):.1f} fps -> saved online_traj.txt"
          + ("" if slam.vo.stats["failed_at"] is None else f" | FAILED at {slam.vo.stats['failed_at']}"))
    if gt is not None and n:
        C_gt = gt[0][:n]
        path_len = np.linalg.norm(np.diff(C_gt, axis=0), axis=1).sum()
        rmse = lambda C: np.sqrt(((lc.anchor_centers(C, gt[1], gt[2], C_gt) - C_gt) ** 2).sum(1).mean())
        msg = f"ATE (origin-anchored, {path_len:.0f} m): final {rmse(slam.centers()):.1f} m"
        if not slam.feedback:
            msg += f" | VO only {rmse(np.array([v.center(R, t) for R, t in zip(slam.vo.Rs[:n], slam.vo.ts[:n])])):.1f} m"
        print(msg)


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
