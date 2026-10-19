"""
Loop closure + Sim(3) pose graph optimization for the monocular VO (no ground truth used).

  1) keyframe features : ORB (1500) + a tiny normalized thumbnail as a global descriptor
  2) local 3D          : ORB matches between keyframe i and the next keyframe, triangulated with the VO
                         poses -> 3D points in camera-i coordinates at the *local* VO scale of that moment
  3) loop detection    : thumbnail correlation shortlist (frames >= MIN_GAP apart) -> ORB ratio-test matching
                         -> fundamental RANSAC -> 3D-3D Sim(3) RANSAC (Umeyama on 3 points, inliers by
                         reprojection in both images). The Sim(3) carries the scale ratio between the two
                         visits = exactly the accumulated scale drift.
  4) Sim(3) PGO        : vertices = keyframe poses S_i (world->cam, X_c = s R X_w + t), odometry edges from VO
                         (scale 1), loop edges from 3). residual = (rotvec, t, log s) of S_meas^-1 S_j S_i^-1.
                         Own sparse LM (finite differences with graph-colored column groups).
"""
import numpy as np
import cv2
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

import vo_plucker as v

MIN_GAP = 150          # 루프 후보는 이 프레임 수 이상 떨어진 과거 키프레임만
SHORTLIST = 15
MIN_F_INLIERS = 80
MIN_SIM3_INLIERS = 20
REPROJ_PX = 3.0


# ---------------- Sim(3) helpers: S = (s, R, t),  X' = s R X + t ----------------
def sim3_mul(a, b):
    sa, Ra, ta = a
    sb, Rb, tb = b
    return (sa * sb, Ra @ Rb, sa * Ra @ tb + ta)


def sim3_inv(a):
    s, R, t = a
    return (1.0 / s, R.T, -(1.0 / s) * R.T @ t)


def rotvec(R):
    return cv2.Rodrigues(R)[0].ravel()


def umeyama_sim3(A, B):
    """B ≈ s R A + t"""
    s, R, t = v.umeyama(A, B)
    return (s, R, t)


# ---------------- features ----------------
def keyframe_features(video_path, kf_frames):
    orb = cv2.ORB_create(1500, fastThreshold=10)
    want = set(kf_frames)
    feats = {}
    cap = cv2.VideoCapture(video_path)
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(g, None)
            th = cv2.resize(g, (64, 20), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            th = (th - th.mean()) / (th.std() + 1e-6)
            feats[k] = dict(pts=np.array([p.pt for p in kp]), des=des, thumb=th / np.sqrt(len(th)))
        k += 1
    return feats


def ratio_match(d1, d2, ratio=0.75):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    m = bf.knnMatch(d1, d2, k=2)
    good = [(a.queryIdx, a.trainIdx) for a, b in (x for x in m if len(x) == 2) if a.distance < ratio * b.distance]
    return np.array(good, int).reshape(-1, 2)


def local_3d(feats, kfs, Rs, ts):
    """각 키프레임: 다음 키프레임과 ORB 매칭 -> VO 포즈로 삼각측량 -> {orb index: X in cam_i}"""
    out = {}
    for a, b in zip(kfs[:-1], kfs[1:]):
        fa, fb = feats[a], feats[b]
        mm = ratio_match(fa["des"], fb["des"])
        pts3 = {}
        if len(mm) >= 8:
            pa, pb = fa["pts"][mm[:, 0]], fb["pts"][mm[:, 1]]
            _, inl = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.0, 0.999)
            if inl is not None:
                Rr = Rs[b] @ Rs[a].T
                tr = ts[b] - Rr @ ts[a]                      # cam_a -> cam_b (VO 단위)
                I, Z = np.eye(3), np.zeros(3)
                for (ia, ib), ok in zip(mm, inl.ravel()):
                    if not ok:
                        continue
                    X = v.tri_point_multi([I, Rr], [Z, tr], [fa["pts"][ia], fb["pts"][ib]])
                    if X is not None and v.point_ok(X, [I, Rr], [Z, tr], [fa["pts"][ia], fb["pts"][ib]], min_deg=1.0):
                        pts3[ia] = X
        out[a] = pts3
    return out


def project(X):
    p = X @ v.K.T
    return p[:, :2] / p[:, 2:3]


def sim3_ransac(Xi, Xj, ui, uj, iters=300, rng=np.random.default_rng(0)):
    """Xj ≈ S Xi. 인라이어 = 양쪽 이미지 재투영 오차 < REPROJ_PX"""
    best, best_inl = None, np.zeros(len(Xi), bool)
    n = len(Xi)
    if n < 3:
        return None, best_inl
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            S = umeyama_sim3(Xi[idx], Xj[idx])
        except np.linalg.LinAlgError:
            continue
        s, R, t = S
        if not np.isfinite(s) or s <= 0:
            continue
        Pj = s * Xi @ R.T + t
        Si = sim3_inv(S)
        Pi = Si[0] * Xj @ Si[1].T + Si[2]
        ok = (Pj[:, 2] > 0) & (Pi[:, 2] > 0)
        e = np.full(n, np.inf)
        e[ok] = np.maximum(np.linalg.norm(project(Pj[ok]) - uj[ok], axis=1),
                           np.linalg.norm(project(Pi[ok]) - ui[ok], axis=1))
        inl = e < REPROJ_PX
        if inl.sum() > best_inl.sum():
            best, best_inl = S, inl
    if best is None or best_inl.sum() < MIN_SIM3_INLIERS:
        return None, best_inl
    return umeyama_sim3(Xi[best_inl], Xj[best_inl]), best_inl       # 인라이어 전체로 재추정


def detect_loop(i, feats, l3d, kfs_before):
    """키프레임 i(현재)에 대한 루프 후보를 과거 키프레임에서 찾음. 반환 (j, S_ij: cam_j -> cam_i, n_inl) 또는 None"""
    cands = [j for j in kfs_before if i - j >= MIN_GAP and j in l3d]
    if not cands or i not in l3d:
        return None
    sims = np.array([feats[i]["thumb"] @ feats[j]["thumb"] for j in cands])
    order = np.argsort(-sims)[:SHORTLIST]
    best = None
    for o in order:
        j = cands[o]
        mm = ratio_match(feats[i]["des"], feats[j]["des"], 0.7)
        if len(mm) < MIN_F_INLIERS:
            continue
        pi, pj = feats[i]["pts"][mm[:, 0]], feats[j]["pts"][mm[:, 1]]
        _, inl = cv2.findFundamentalMat(pi, pj, cv2.FM_RANSAC, 1.5, 0.999)
        if inl is None or inl.sum() < MIN_F_INLIERS:
            continue
        mm = mm[inl.ravel() > 0]
        both = [(a, b) for a, b in mm if a in l3d[i] and b in l3d[j]]
        if len(both) < MIN_SIM3_INLIERS:
            continue
        Xi = np.array([l3d[i][a] for a, b in both])
        Xj = np.array([l3d[j][b] for a, b in both])
        ui = np.array([feats[i]["pts"][a] for a, b in both])
        uj = np.array([feats[j]["pts"][b] for a, b in both])
        S, inl3 = sim3_ransac(Xj, Xi, uj, ui)          # Xi ≈ S Xj  (cam_j -> cam_i)
        if S is None:
            continue
        if best is None or inl3.sum() > best[2]:
            best = (j, S, int(inl3.sum()))
    return best


# ---------------- Sim(3) pose graph ----------------
def pgo(S0, edges, fixed=(0,), iters=30):
    """S0: list of vertex Sim(3) (world->cam). edges: (i, j, S_meas (cam_i->cam_j), weight vector(7)).
    residual: [rotvec, t, log s] of  S_meas^-1 · S_j · S_i^-1"""
    n = len(S0)
    free = [i for i in range(n) if i not in set(fixed)]
    col = {i: 7 * c for c, i in enumerate(free)}
    E = len(edges)

    def apply(x):
        S = list(S0)
        for i in free:
            d = x[col[i]: col[i] + 7]
            s, R, t = S0[i]
            S[i] = (s * np.exp(d[6]), v.so3_exp(d[:3]) @ R, t + d[3:6])
        return S

    def fun(x):
        S = apply(x)
        r = np.empty(7 * E)
        for e, (i, j, M, w) in enumerate(edges):
            Er = sim3_mul(sim3_inv(M), sim3_mul(S[j], sim3_inv(S[i])))
            r[7 * e: 7 * e + 7] = w * np.r_[rotvec(Er[1]), Er[2], np.log(Er[0])]
        return r

    # greedy graph coloring -> 같은 색 정점은 같은 edge(행)를 공유하지 않으므로 함께 유한차분 가능
    nbr = {i: set() for i in range(n)}
    for i, j, _, _ in edges:
        nbr[i].add(j)
        nbr[j].add(i)
    color = {}
    for i in free:
        used = {color[k] for k in nbr[i] if k in color}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    ncol = max(color.values()) + 1 if color else 0
    rows_of = {i: [] for i in range(n)}
    for e, (i, j, _, _) in enumerate(edges):
        rows_of[i].append(e)
        rows_of[j].append(e)

    x = np.zeros(7 * len(free))
    r = fun(x)
    cost, lam = 0.5 * r @ r, 1e-4
    for _ in range(iters):
        I, J, V = [], [], []
        for c in range(ncol):
            members = [i for i in free if color[i] == c]
            for k in range(7):
                dx = np.zeros_like(x)
                for i in members:
                    dx[col[i] + k] = 1e-6
                dr = (fun(x + dx) - r) / 1e-6
                for i in members:
                    for e in rows_of[i]:
                        rr = np.arange(7 * e, 7 * e + 7)
                        I.append(rr); J.append(np.full(7, col[i] + k)); V.append(dr[rr])
        Jm = csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(len(r), len(x)))
        Hm = (Jm.T @ Jm).tocsc()
        g = Jm.T @ r
        improved = False
        for _ in range(8):
            dx = spsolve(Hm + diags(lam * (Hm.diagonal() + 1e-9)), -g)
            r1 = fun(x + dx)
            c1 = 0.5 * r1 @ r1
            if c1 < cost:
                x, r, cost = x + dx, r1, c1
                lam = max(lam / 5, 1e-9)
                improved = True
                break
            lam *= 10
        if not improved or np.abs(dx).max() < 1e-9:
            break
    return apply(x), cost


W_ODOM = np.r_[[1 / 0.005] * 3, [1 / 0.02] * 3, 1 / 0.02]     # 오도메트리: 회전/이동/스케일 표준편차의 역수
W_LOOP = np.r_[[1 / 0.01] * 3, [1 / 0.05] * 3, 1 / 0.05]


def build_and_optimize(kfs, Rs, ts, loops):
    """kfs: keyframe frame indices. loops: list of (i_frame, j_frame, S_ij cam_j->cam_i).
    반환: {kf frame: corrected Sim(3)}"""
    S0 = [(1.0, Rs[f], ts[f]) for f in kfs]
    pos = {f: n for n, f in enumerate(kfs)}
    edges = []
    for a in range(len(kfs) - 1):
        M = sim3_mul(S0[a + 1], sim3_inv(S0[a]))                  # cam_a -> cam_{a+1}
        edges.append((a, a + 1, M, W_ODOM))
    for i_f, j_f, S_ij in loops:
        edges.append((pos[j_f], pos[i_f], S_ij, W_LOOP))          # cam_j -> cam_i
    S, cost = pgo(S0, edges, fixed=(0,))
    return {f: S[pos[f]] for f in kfs}, cost


def propagate(frames_R, frames_t, kfs, S_kf, upto=None):
    """비키프레임: 직전 키프레임 기준 VO 상대포즈를 붙여 Sim(3) 생성. 반환: 각 프레임 카메라 중심 (world)"""
    n = len(frames_R) if upto is None else upto
    C = np.zeros((n, 3))
    kfs_sorted = sorted(S_kf)
    kk = 0
    for f in range(n):
        while kk + 1 < len(kfs_sorted) and kfs_sorted[kk + 1] <= f:
            kk += 1
        a = kfs_sorted[kk]
        Rr = frames_R[f] @ frames_R[a].T
        tr = frames_t[f] - Rr @ frames_t[a]
        sa, Ra, ta = S_kf[a]
        s, R, t = sa, Rr @ Ra, Rr @ ta + tr                      # cam 좌표는 VO 단위(=cam_a 스케일)
        C[f] = -(1.0 / s) * R.T @ t
    return C


def run_loop_closure(Rs, ts, feats, kfs, verbose=True):
    """VO 포즈(Rs, ts)에 대해 키프레임 순서대로 루프를 찾고, 루프가 잡힐 때마다 Sim(3) PGO.
    (VO 자체는 보정된 포즈를 다시 쓰지 않으므로 이렇게 순서대로 재생하는 것 = 온라인 동작과 동일)
    반환: dict(loops=[(i, j, S_ij, n_inl)], events=[(frame, centers (N,3))], centers=최종 보정 중심)"""
    l3d = local_3d(feats, kfs, Rs, ts)
    loops, events = [], []
    n = len(Rs)
    for idx, i in enumerate(kfs):
        res = detect_loop(i, feats, l3d, kfs[:idx])
        if not res:
            continue
        j, S, ninl = res
        loops.append((i, j, S, ninl))
        sub = kfs[: idx + 1]
        S_kf, cost = build_and_optimize(sub, Rs, ts, [(a, b, Sab) for a, b, Sab, _ in loops])
        events.append((i, propagate(Rs, ts, sub, S_kf, upto=n)))
        if verbose:
            print(f"    loop @frame {i} <-> {j}: {ninl} Sim(3) inliers, scale ratio {S[0]:.2f} -> PGO cost {cost:.1f}")
    centers = events[-1][1] if events else np.array([v.center(R, t) for R, t in zip(Rs, ts)])
    return dict(loops=loops, events=events, centers=centers)


def anchor_centers(C, gtR0, gtt0, C_gt, n_scale=50):
    """vertex 0 = VO frame 0 고정이므로, 원점 고정 정렬(스케일만 처음 n_scale 프레임으로)을 중심 좌표에 직접 적용"""
    Y = C_gt[: len(C)] @ gtR0.T + gtt0
    m = min(n_scale, len(C))
    s = (C[:m] * Y[:m]).sum() / max((C[:m] ** 2).sum(), 1e-12)
    return (s * C - gtt0) @ gtR0