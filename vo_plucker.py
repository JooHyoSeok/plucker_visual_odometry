"""
Minimal monocular visual odometry with points + Plücker lines (synthetic corridor).

Pipeline
  1) 2-view init        : normalized 8-point + Sampson refinement (points only), |t| = 1 -> scale arbitrary
  2) per-frame tracking : LM on SE(3), point reprojection + line reprojection
  3) mapping            : (re)triangulate points (N-view DLT) and lines (N back-projected planes -> Plücker)
  4) evaluation         : Sim(3) Umeyama alignment, ATE, local scale ratio

Simplifications (on purpose, to focus on geometry)
  - data association is ground truth (no ORB / LSD matching)
  - no bundle adjustment; instead landmarks are re-triangulated from all observations until
    parallax > MATURE_DEG, then frozen; landmarks with large residual are culled and re-made
  - numerical Jacobian

Conventions
  T_cw = (R, t):  X_c = R X_w + t
  Plücker L = (d, m), m = p x d
  camera frame:   d_c = R d,  m_c = R m + [t]x R d
  image line:     l ∝ KL m_c,  KL = [[fy,0,0],[0,fx,0],[-fy cx,-fx cy, fx fy]]  (∝ K^-T)
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 0. 설정
# ============================================================
SEED = 0
N_FRAMES = 120
N_POINTS = 20        # 저텍스처 복도: 점 특징이 적음 (60 정도면 둘 다 잘 됨)
PIX_NOISE = 1.0      # px
INIT_FRAME = 8       # frame 0 과 이 프레임으로 초기화
TRI_GAP = 4          # 새 랜드마크 삼각측량 최소 프레임 간격 (baseline 확보)
LINE_MIN_DEG = 2.0   # 두 역투영 평면 사이 최소 각도 (작으면 degenerate)
OUTLIER_PX = 6.0     # 포즈 추정 후 이보다 큰 재투영 오차를 가진 랜드마크는 맵에서 제거
USE_EPIPOLAR = False  # True: 직전 프레임과의 Sampson(2D-2D) 잔차도 포즈 추정에 사용 (실데이터용)
MATURE_DEG = 5.0     # parallax가 이 이상이면 랜드마크 고정 (그 전까지는 매 프레임 재삼각측량)

fx = fy = 400.0
cx, cy = 320.0, 240.0
W, H = 640, 480
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
Kinv = np.linalg.inv(K)
KL = np.array([[fy, 0, 0], [0, fx, 0], [-fy * cx, -fx * cy, fx * fy]])


# ============================================================
# 1. 기본 유틸
# ============================================================
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w)
    Kx = skew(w / th)
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def proj_matrix(R, t):
    return K @ np.hstack([R, t[:, None]])


def center(R, t):
    return -R.T @ t


def orthonormalize(R):
    """SO(3)로 다시 투영. 이거 없으면 constant-velocity 예측 R_{k-1} R_{k-2}^T R_{k-1}에서
    수치 오차가 프레임마다 ~2.4배씩 커져서 40프레임쯤에 R이 회전행렬이 아니게 됨 (실제로 겪은 버그)."""
    U, _, Vt = np.linalg.svd(R)
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def retract(R, t, dx):
    """left perturbation: T <- Exp(dx) T"""
    dR = so3_exp(dx[:3])
    return dR @ R, dR @ t + dx[3:]


# ============================================================
# 2. Plücker
# ============================================================
def plucker_from_endpoints(A, B):
    d = B - A
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.concatenate([d, np.cross(A, d)], axis=-1)


def plucker_to_cam(L, R, t):
    d = L[:, :3] @ R.T
    m = L[:, 3:] @ R.T + np.cross(t, d)
    return d, m


def project_lines(L, R, t):
    """(M,6) world Plücker -> (M,3) pixel image lines"""
    _, m = plucker_to_cam(L, R, t)
    return m @ KL.T


# ============================================================
# 3. 합성 장면: 복도 + 카메라 궤적
# ============================================================
def make_scene(rng, length=40.0):
    segs = []
    for x in (-2.0, 2.0):
        for z in np.arange(2.0, length, 2.5):            # 문틀 (세로선)
            segs.append([[x, -1.5, z], [x, 1.5, z]])
        for y in (-1.5, 1.5):                             # 벽-천장 / 벽-바닥 모서리 (진행 방향과 평행!)
            for z in np.arange(0.0, length, 3.0):
                segs.append([[x, y, z], [x, y, z + 2.5]])
        for z in np.arange(1.0, length, 4.0):             # 몰딩
            y = rng.uniform(-1.0, 1.0)
            segs.append([[x, y, z], [x, y, z + rng.uniform(1.5, 3.0)]])
    for z in np.arange(3.0, length, 5.0):                 # 천장 조명 (가로선)
        segs.append([[-0.8, -1.5, z], [0.8, -1.5, z]])
    segs = np.array(segs)

    side = rng.integers(0, 3, N_POINTS)                   # 0,1: 벽 / 2: 바닥
    pts = np.zeros((N_POINTS, 3))
    pts[:, 2] = rng.uniform(0, length, N_POINTS)
    wall = side < 2
    pts[wall, 0] = np.where(side[wall] == 0, -2.0, 2.0)
    pts[wall, 1] = rng.uniform(-1.5, 1.5, wall.sum())
    pts[~wall, 0] = rng.uniform(-2.0, 2.0, (~wall).sum())
    pts[~wall, 1] = 1.5
    return pts, segs


def make_traj(n):
    Rs, ts = [], []
    for k in range(n):
        s = k / (n - 1)
        C = np.array([0.8 * np.sin(3 * np.pi * s), 0.15 * np.sin(6 * np.pi * s), 25.0 * s])
        yaw = 0.25 * np.sin(3 * np.pi * s + 0.8)
        R = rot_y(yaw).T            # R_cw
        Rs.append(R)
        ts.append(-R @ C)
    return Rs, ts


def in_image(u):
    return (u[:, 0] >= 0) & (u[:, 0] < W) & (u[:, 1] >= 0) & (u[:, 1] < H)


def observe(R, t, pts, segs, rng):
    """returns ({pid: uv}, {lid: [u1,v1,u2,v2]}) with noise.
    lines: only the visible part of the segment is observed (partial observation is fine for lines)."""
    obs_p, obs_l = {}, {}
    Pc = pts @ R.T + t
    uv = Pc @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    ok = (Pc[:, 2] > 0.5) & in_image(uv)
    for j in np.where(ok)[0]:
        obs_p[j] = uv[j] + rng.normal(0, PIX_NOISE, 2)

    s = np.linspace(0, 1, 60)
    for j, (A, B) in enumerate(segs):
        Xc = (A[None] + s[:, None] * (B - A)[None]) @ R.T + t
        front = Xc[:, 2] > 0.5
        u = Xc @ K.T
        u = u[:, :2] / np.maximum(u[:, 2:3], 1e-9)
        idx = np.where(front & in_image(u))[0]
        if len(idx) < 2:
            continue
        p1, p2 = u[idx[0]], u[idx[-1]]
        if np.linalg.norm(p2 - p1) < 30:
            continue
        obs_l[j] = np.concatenate([p1, p2]) + rng.normal(0, PIX_NOISE, 4)
    return obs_p, obs_l


# ============================================================
# 4. 2-view 초기화 (essential matrix)
# ============================================================
def tri_point(P1, P2, x1, x2):
    A = np.array([x1[0] * P1[2] - P1[0], x1[1] * P1[2] - P1[1],
                  x2[0] * P2[2] - P2[0], x2[1] * P2[2] - P2[1]])
    X = np.linalg.svd(A)[2][-1]
    return X[:3] / X[3]


def eight_point(x1, x2):
    """x2^T E x1 = 0, normalized coords (N,2)"""
    A = np.column_stack([x2[:, 0] * x1[:, 0], x2[:, 0] * x1[:, 1], x2[:, 0],
                         x2[:, 1] * x1[:, 0], x2[:, 1] * x1[:, 1], x2[:, 1],
                         x1[:, 0], x1[:, 1], np.ones(len(x1))])
    E = np.linalg.svd(A)[2][-1].reshape(3, 3)
    U, _, Vt = np.linalg.svd(E)
    return U @ np.diag([1.0, 1.0, 0.0]) @ Vt


def eight_point_normalized(x1, x2):
    """Hartley normalization. 전방 이동에서는 이거 없으면 t 부호가 자주 뒤집힘."""
    def T(x):
        mu = x.mean(0)
        s = np.sqrt(2) / np.mean(np.linalg.norm(x - mu, axis=1))
        return np.array([[s, 0, -s * mu[0]], [0, s, -s * mu[1]], [0, 0, 1.0]])
    T1, T2 = T(x1), T(x2)
    h = lambda x, Tm: (np.c_[x, np.ones(len(x))] @ Tm.T)[:, :2]
    E = T2.T @ eight_point(h(x1, T1), h(x2, T2)) @ T1
    U, _, Vt = np.linalg.svd(E)
    return U @ np.diag([1.0, 1.0, 0.0]) @ Vt


def sampson(R, t, x1, x2):
    """Sampson epipolar error [px 근사]"""
    E = skew(t) @ R
    a, b = np.c_[x1, np.ones(len(x1))], np.c_[x2, np.ones(len(x2))]
    Ea, Eb = a @ E.T, b @ E
    return (b * Ea).sum(1) / np.sqrt(Ea[:, 0]**2 + Ea[:, 1]**2 + Eb[:, 0]**2 + Eb[:, 1]**2) * fx


def refine_relative_pose(R, t, x1, x2, iters=30):
    """5-DoF (R, t/|t|) Gauss-Newton on Sampson error. 선형 8-point는 회전이 1~3° 틀어져서 필요."""
    for _ in range(iters):
        r = sampson(R, t, x1, x2)
        J = np.zeros((len(r), 6))
        for i in range(6):
            d = np.zeros(6)
            d[i] = 1e-7
            t2 = t + d[3:]
            J[:, i] = (sampson(so3_exp(d[:3]) @ R, t2 / np.linalg.norm(t2), x1, x2) - r) / 1e-7
        dx = -np.linalg.solve(J.T @ J + 1e-3 * np.eye(6), J.T @ r)   # damping: |t|=1 gauge 방향 처리
        R = so3_exp(dx[:3]) @ R
        t = (t + dx[3:]) / np.linalg.norm(t + dx[3:])
    return R, t


def decompose_E(E, x1, x2):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    Wm = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    best, best_cnt = None, -1
    for R in (U @ Wm @ Vt, U @ Wm.T @ Vt):
        for t in (U[:, 2], -U[:, 2]):
            P2 = np.hstack([R, t[:, None]])
            cnt = 0
            for a, b in zip(x1, x2):                       # cheirality
                X = tri_point(P1, P2, a, b)
                cnt += (X[2] > 0) and ((R @ X + t)[2] > 0)
            if cnt > best_cnt:
                best, best_cnt = (R, t), cnt
    return best


# ============================================================
# 5. 삼각측량 (점 / 선)
# ============================================================
def tri_point_multi(Rl, tl, uvs):
    """N-view DLT"""
    A = []
    for R, t, u in zip(Rl, tl, uvs):
        P = proj_matrix(R, t)
        A += [u[0] * P[2] - P[0], u[1] * P[2] - P[1]]
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-12 else None


def backproject_plane(R, t, seg):
    """이미지 선분 -> 카메라 중심과 그 선을 품는 평면 π = Pᵀl (|n| = 1로 정규화)"""
    l = np.cross([*seg[:2], 1.0], [*seg[2:], 1.0])
    pi = proj_matrix(R, t).T @ l
    return pi / np.linalg.norm(pi[:3])


def tri_line_multi(Rl, tl, segs):
    """N개의 역투영 평면을 쌓은 Π (N×4)의 null space(2차원) = 직선 위 두 점 P, Q -> Plücker.
    N=2면 두 평면의 교선과 동일: d = n1×n2, m = w1 n2 - w2 n1."""
    Pi = np.array([backproject_plane(R, t, s) for R, t, s in zip(Rl, tl, segs)])
    V = np.linalg.svd(Pi)[2]
    P, Q = V[-1], V[-2]
    d = P[3] * Q[:3] - Q[3] * P[:3]
    m = np.cross(P[:3], Q[:3])
    nd = np.linalg.norm(d)
    return None if nd < 1e-12 else np.concatenate([d, m]) / nd


def parallax_deg(X, R1, t1, R2, t2):
    a, b = X - center(R1, t1), X - center(R2, t2)
    cosang = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def plane_angle_deg(R1, t1, s1, R2, t2, s2):
    n1, n2 = backproject_plane(R1, t1, s1)[:3], backproject_plane(R2, t2, s2)[:3]
    return np.degrees(np.arcsin(np.clip(np.linalg.norm(np.cross(n1, n2)), 0, 1)))


def point_ok(X, Rl, tl, uvs, min_deg=1.0):
    if X is None:
        return False
    if parallax_deg(X, Rl[0], tl[0], Rl[-1], tl[-1]) < min_deg:
        return False
    for R, t, u in zip(Rl, tl, uvs):
        Xc = R @ X + t
        if Xc[2] < 0.5:
            return False
        p = K @ Xc
        if np.linalg.norm(p[:2] / p[2] - u) > 4.0:
            return False
    return True


def line_ok(L, Rl, tl, segs):
    if L is None:
        return False
    if plane_angle_deg(Rl[0], tl[0], segs[0], Rl[-1], tl[-1], segs[-1]) < LINE_MIN_DEG:
        return False                                   # 선이 baseline과 거의 평행 -> degenerate
    d, m0 = L[:3], L[3:]
    for R, t, sg in zip(Rl, tl, segs):
        # 관측 선분 중점의 광선과 직선의 최근접점이 카메라 앞에 있는지 (cheirality)
        C = center(R, t)
        r = R.T @ Kinv @ np.array([(sg[0] + sg[2]) / 2, (sg[1] + sg[3]) / 2, 1.0])
        p0 = np.cross(d, m0)                           # 원점에 가장 가까운 직선 위 점 (|d|=1)
        w0 = C - p0
        a, b, c = r @ r, r @ d, d @ d
        den = a * c - b * b
        if den < 1e-12:
            return False
        lam = (b * (d @ w0) - c * (r @ w0)) / den      # 광선 파라미터
        if lam < 0.5 or lam > 60:
            return False
        l = project_lines(L[None], R, t)[0]
        nl = np.hypot(l[0], l[1])
        e1 = abs(l @ [sg[0], sg[1], 1.0]) / nl
        e2 = abs(l @ [sg[2], sg[3], 1.0]) / nl
        if max(e1, e2) > 4.0:
            return False
    return True


def pick_frames(frames, k_max=12):
    """관측 프레임이 많으면 처음/끝 포함해서 균등 샘플"""
    if len(frames) <= k_max:
        return frames
    idx = np.unique(np.linspace(0, len(frames) - 1, k_max).round().astype(int))
    return [frames[i] for i in idx]


# ============================================================
# 6. 포즈 추정 (Levenberg-Marquardt, Huber)
# ============================================================
def residuals(R, t, X, uv, L, seg, epi=None):
    """epi=(R_prev, t_prev, x_prev, x_cur): 이전 프레임과의 2D-2D Sampson 잔차를 추가 (정규화 좌표).
    맵에 안 올라간 트랙까지 회전/방향을 잡아주고, 스케일은 맵(3D-2D) 잔차가 정함."""
    r = []
    if epi is not None and len(epi[2]):
        Rp, tp, x1, x2 = epi
        Rr = R @ Rp.T
        tr = t - Rr @ tp
        r.append(sampson(Rr, tr, x1, x2) if np.linalg.norm(tr) > 1e-9 else np.zeros(len(x1)))
    if len(X):
        p = (X @ R.T + t) @ K.T
        r.append((p[:, :2] / p[:, 2:3] - uv).ravel())
    if len(L):
        l = project_lines(L, R, t)
        n = np.hypot(l[:, 0], l[:, 1])
        s = np.column_stack([seg[:, :2], np.ones(len(seg))])
        e = np.column_stack([seg[:, 2:], np.ones(len(seg))])
        # 관측 선분의 두 끝점에서 투영된 직선까지의 거리 [px]
        r.append(np.column_stack([(s * l).sum(1) / n, (e * l).sum(1) / n]).ravel())
    return np.concatenate(r)


def huber_cost(r, delta):
    a = np.abs(r)
    return np.sum(np.where(a <= delta, 0.5 * a**2, delta * (a - 0.5 * delta)))


def estimate_pose(R, t, X, uv, L, seg, iters=40, huber=3.0, epi=None):
    """Levenberg-Marquardt on SE(3) (6-DoF), Huber IRLS, 수치 Jacobian.
    순수 GN은 step 크기 제어가 없어서 예측이 조금만 틀려도 엉뚱한 곳으로 수렴함."""
    lam = 1e-3
    r = residuals(R, t, X, uv, L, seg, epi)
    cost = huber_cost(r, huber)
    for _ in range(iters):
        J = np.zeros((len(r), 6))
        eps = 1e-6
        for i in range(6):
            dx = np.zeros(6)
            dx[i] = eps
            J[:, i] = (residuals(*retract(R, t, dx), X, uv, L, seg, epi) - r) / eps
        a = np.abs(r)
        w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
        Hm = J.T @ (w[:, None] * J)
        g = J.T @ (w * r)
        while True:
            dx = -np.linalg.solve(Hm + lam * np.diag(np.diag(Hm) + 1e-9), g)
            R1, t1 = retract(R, t, dx)
            r1 = residuals(R1, t1, X, uv, L, seg, epi)
            c1 = huber_cost(r1, huber) if np.all(np.isfinite(r1)) else np.inf
            if c1 < cost:
                R, t, r, cost = R1, t1, r1, c1
                lam = max(lam / 3, 1e-7)
                break
            lam *= 4
            if lam > 1e6:
                return R, t
        if np.linalg.norm(dx) < 1e-10:
            break
    return R, t


def estimate_scale(Rk, base, that, s0, X, uv, L, seg, iters=30, huber=3.0):
    """회전과 이동 방향은 고정(2D-2D에서 얻음), 이동 크기 s 하나만 맵(점+선) 재투영으로 푼다.
    t_k = base + s * that,  base = R_rel t_{k-1}"""
    f = lambda s: residuals(Rk, base + s * that, X, uv, L, seg)
    s, lam = s0, 1e-3
    r = f(s)
    cost = huber_cost(r, huber)
    for _ in range(iters):
        J = (f(s + 1e-6) - r) / 1e-6
        a = np.abs(r)
        w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
        Hs, g = J @ (w * J), J @ (w * r)
        while True:
            ds = -g / (Hs * (1 + lam) + 1e-12)
            r1 = f(s + ds)
            c1 = huber_cost(r1, huber) if np.all(np.isfinite(r1)) else np.inf
            if c1 < cost:
                s, r, cost = s + ds, r1, c1
                lam = max(lam / 3, 1e-7)
                break
            lam *= 4
            if lam > 1e6:
                return s
        if abs(ds) < 1e-10:
            break
    return s


# ============================================================
# 7. VO 메인 루프
# ============================================================
def run_vo(obs, use_lines, history=None, init_pose=None, rel_pose=None, post_frame_hook=None):
    """history(dict)를 넘기면 프레임별 (R, t, 그 프레임에 보이는 랜드마크)를 저장 -> 영상용
    init_pose=(R, t)를 넘기면 내장 8-point 초기화 대신 사용 (실데이터: RANSAC 결과)
    rel_pose={k: (R_rel, t_dir)}를 넘기면 회전/방향은 그 값(2D-2D)을 쓰고 맵으로는 스케일 1개만 푼다
    post_frame_hook(k, ctx)를 넘기면 매 프레임 매핑 후 호출 (local BA 등). ctx의 Rs/ts/map은 제자리 수정 가능"""
    N = len(obs)
    Rs, ts = [None] * N, [None] * N
    first_p, first_l = {}, {}
    for k, (op, ol) in enumerate(obs):
        for j in op:
            first_p.setdefault(j, k)
        for j in ol:
            first_l.setdefault(j, k)

    # --- init: frame 0 & INIT_FRAME ---
    op0, opI = obs[0][0], obs[INIT_FRAME][0]
    ids = sorted(set(op0) & set(opI))
    x1 = np.array([(Kinv @ [*op0[j], 1.0])[:2] for j in ids])
    x2 = np.array([(Kinv @ [*opI[j], 1.0])[:2] for j in ids])
    if init_pose is None:
        R, t = decompose_E(eight_point_normalized(x1, x2), x1, x2)
        R, t = refine_relative_pose(R, t, x1, x2)
    else:
        R, t = init_pose
    Rs[0], ts[0] = np.eye(3), np.zeros(3)
    Rs[INIT_FRAME], ts[INIT_FRAME] = R, t / np.linalg.norm(t)

    map_p, map_l = {}, {}
    mature_p, mature_l = set(), set()
    stats = {"line_rejected": 0, "failed_at": None}

    def update_map(k):
        """보이는 랜드마크를 지금까지의 모든 관측(포즈 아는 프레임)으로 다시 삼각측량.
        BA 대신 structure-only 정제. 처음 만들 때는 baseline(TRI_GAP)을 요구."""
        op, ol = obs[k]
        for j in op:
            if k - first_p[j] < TRI_GAP or j in mature_p:
                continue
            fr = pick_frames([f for f in range(first_p[j], k + 1) if Rs[f] is not None and j in obs[f][0]])
            Rl, tl, uvs = [Rs[f] for f in fr], [ts[f] for f in fr], [obs[f][0][j] for f in fr]
            X = tri_point_multi(Rl, tl, uvs)
            if point_ok(X, Rl, tl, uvs):
                map_p[j] = X
                if parallax_deg(X, Rl[0], tl[0], Rl[-1], tl[-1]) > MATURE_DEG:
                    mature_p.add(j)
        if not use_lines:
            return
        for j in ol:
            if k - first_l[j] < TRI_GAP or j in mature_l:
                continue
            fr = pick_frames([f for f in range(first_l[j], k + 1) if Rs[f] is not None and j in obs[f][1]])
            Rl, tl, sgs = [Rs[f] for f in fr], [ts[f] for f in fr], [obs[f][1][j] for f in fr]
            Lj = tri_line_multi(Rl, tl, sgs)
            if line_ok(Lj, Rl, tl, sgs):
                map_l[j] = Lj
                if plane_angle_deg(Rl[0], tl[0], sgs[0], Rl[-1], tl[-1], sgs[-1]) > MATURE_DEG:
                    mature_l.add(j)
            elif j not in map_l:
                stats["line_rejected"] += 1

    update_map(INIT_FRAME)
    ctx = dict(Rs=Rs, ts=ts, obs=obs, map_p=map_p, map_l=map_l, mature_p=mature_p, mature_l=mature_l,
               use_lines=use_lines)
    if post_frame_hook is not None:
        post_frame_hook(0, ctx)
        post_frame_hook(INIT_FRAME, ctx)
    if history is not None:
        for f in (0, INIT_FRAME):
            history[f] = (Rs[f], ts[f], {j: map_p[j] for j in obs[f][0] if j in map_p},
                          {j: map_l[j] for j in obs[f][1] if j in map_l})

    for k in range(1, N):
        if k == INIT_FRAME:
            continue
        # constant-velocity prediction
        if k >= 2:
            dR = Rs[k - 1] @ Rs[k - 2].T
            dt = ts[k - 1] - dR @ ts[k - 2]
            R0, t0 = orthonormalize(dR @ Rs[k - 1]), dR @ ts[k - 1] + dt
        else:                                   # k=1: 초기화 포즈를 선형 보간한 값에서 시작
            R0, t0 = Rs[k - 1], ts[INIT_FRAME] / INIT_FRAME

        op, ol = obs[k]
        pid = [j for j in op if j in map_p]
        lid = [j for j in ol if j in map_l] if use_lines else []
        X = np.array([map_p[j] for j in pid]).reshape(-1, 3)
        uv = np.array([op[j] for j in pid]).reshape(-1, 2)
        Lm = np.array([map_l[j] for j in lid]).reshape(-1, 6)
        sg = np.array([ol[j] for j in lid]).reshape(-1, 4)

        epi = None
        if USE_EPIPOLAR:
            common = [j for j in op if j in obs[k - 1][0]]
            if len(common) >= 8:
                x1 = (np.c_[np.array([obs[k - 1][0][j] for j in common]), np.ones(len(common))] @ Kinv.T)[:, :2]
                x2 = (np.c_[np.array([op[j] for j in common]), np.ones(len(common))] @ Kinv.T)[:, :2]
                epi = (Rs[k - 1], ts[k - 1], x1, x2)

        if 2 * (len(pid) + len(lid)) < 12:     # 대응점 부족 -> tracking lost
            stats["failed_at"] = k
            break
        else:
            def solve(X, uv, Lm, sg, Ri, ti):
                if rel_pose is not None and k in rel_pose:
                    Rr, th = rel_pose[k]
                    Rk, base = Rr @ Rs[k - 1], Rr @ ts[k - 1]
                    sc = estimate_scale(Rk, base, th, max((ti - base) @ th, 1e-3), X, uv, Lm, sg)
                    return Rk, base + sc * th
                return estimate_pose(Ri, ti, X, uv, Lm, sg, epi=epi)

            R1, t1 = solve(X, uv, Lm, sg, R0, t0)
            # outlier culling: 이 프레임에서 크게 안 맞는 랜드마크는 맵에서 빼고(다시 삼각측량 대상) 포즈 재추정
            r = residuals(R1, t1, X, uv, Lm, sg)
            rp = np.linalg.norm(r[:2 * len(pid)].reshape(-1, 2), axis=1)
            rl = np.abs(r[2 * len(pid):].reshape(-1, 2)).max(1) if len(lid) else np.zeros(0)
            keep_p, keep_l = rp < OUTLIER_PX, rl < OUTLIER_PX
            for j in np.array(pid)[~keep_p]:
                map_p.pop(j); mature_p.discard(j)
            for j in np.array(lid, dtype=int)[~keep_l]:
                map_l.pop(j); mature_l.discard(j)
            if keep_p.sum() + keep_l.sum() < 6:      # 맵이 거의 다 안 맞음 -> 실패
                stats["failed_at"] = k
                break
            if (~keep_p).any() or (~keep_l).any():
                X, uv, Lm, sg = X[keep_p], uv[keep_p], Lm[keep_l], sg[keep_l]
                R1, t1 = solve(X, uv, Lm, sg, R1, t1)
            r = residuals(R1, t1, X, uv, Lm, sg)
            if not np.all(np.isfinite(r)) or np.median(np.abs(r)) > 3.0 * PIX_NOISE:
                stats["failed_at"] = k          # tracking failure -> 여기서 멈춤 (relocalization 없음)
                break
            Rs[k], ts[k] = orthonormalize(R1), t1
        update_map(k)
        if post_frame_hook is not None:
            post_frame_hook(k, ctx)
        if history is not None:
            history[k] = (Rs[k], ts[k], {j: map_p[j] for j in op if j in map_p},   # 이 프레임에 보이는 것만
                          {j: map_l[j] for j in ol if j in map_l})

    return Rs, ts, map_p, map_l, stats


# ============================================================
# 8. 평가: Sim(3) Umeyama, ATE, local scale
# ============================================================
def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, Dg, Vt = np.linalg.svd(D.T @ S / len(src))
    Sm = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sm[2, 2] = -1
    R = U @ Sm @ Vt
    s = np.trace(np.diag(Dg) @ Sm) / ((S ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


def align_first_frame(Rs, ts, gtR0, gtt0, C_gt, n_scale=50):
    """원점 고정 정렬: 추정 frame 0 카메라 = GT frame 0 카메라로 맞추고 (회전/위치 고정),
    스케일 하나만 처음 n_scale 프레임으로 맞춘다. -> 두 궤적이 같은 점·같은 방향에서 출발하고,
    이후 벗어나는 건 전부 드리프트로 보인다 (Sim(3) 전체 정렬은 드리프트를 궤적 전체에 나눠서 숨김).
    Rs, ts: 추정 T_cw 리스트. gtR0, gtt0: GT frame 0의 T_cw. 반환: GT world 좌표의 카메라 중심 (N,3)"""
    R0, t0 = Rs[0], ts[0]
    Xc = np.array([R0 @ center(R, t) + t0 for R, t in zip(Rs, ts)])      # 추정 cam0 좌표계
    Y = C_gt[: len(Xc)] @ gtR0.T + gtt0                                   # GT도 cam0 좌표계로
    m = min(n_scale, len(Xc))
    s = (Xc[:m] * Y[:m]).sum() / max((Xc[:m] ** 2).sum(), 1e-12)
    return (s * Xc - gtt0) @ gtR0                                         # = gtR0ᵀ (s Xc - gtt0)


def local_scale(C_est, C_gt, w=10):
    a = np.linalg.norm(C_est[w:] - C_est[:-w], axis=1)
    b = np.linalg.norm(C_gt[w:] - C_gt[:-w], axis=1)
    return a / b


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    pts, segs = make_scene(rng)
    gtR, gtT = make_traj(N_FRAMES)
    obs = [observe(R, t, pts, segs, rng) for R, t in zip(gtR, gtT)]
    C_gt = np.array([center(R, t) for R, t in zip(gtR, gtT)])

    nvp = np.mean([len(o[0]) for o in obs])
    nvl = np.mean([len(o[1]) for o in obs])
    print(f"avg visible per frame: points {nvp:.1f}, lines {nvl:.1f}")

    results = {}
    for name, ul in [("points only", False), ("points + lines", True)]:
        Rs, ts, mp, ml, st = run_vo(obs, ul)
        n_ok = N_FRAMES if st["failed_at"] is None else st["failed_at"]
        C = np.array([center(R, t) for R, t in zip(Rs[:n_ok], ts[:n_ok])])
        s, Ra, ta = umeyama(C, C_gt[:n_ok])
        C_al = (s * (Ra @ C.T)).T + ta
        err = np.linalg.norm(C_al - C_gt[:n_ok], axis=1)
        results[name] = (C_al, err, local_scale(C_al, C_gt[:n_ok]))
        print(f"[{name:>14}] ATE RMSE {np.sqrt((err**2).mean()):.3f} m | "
              f"max {err.max():.3f} m | map pts {len(mp)}, lines {len(ml)} | "
              f"line tri rejections {st['line_rejected']} | "
              f"{'tracked all frames' if st['failed_at'] is None else 'TRACKING FAILED at frame ' + str(st['failed_at'])}")

    # ---------- plot ----------
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [1.3, 1, 1]})
    for A, B in segs:
        if abs(A[0] - B[0]) < 1e-6 and abs(A[1] - 1.5) < 1e-6 and abs(B[1] - 1.5) < 1e-6:
            ax[0].plot([A[2], B[2]], [A[0], B[0]], color="0.75", lw=1)
    ax[0].plot(C_gt[:, 2], C_gt[:, 0], "k-", lw=2, label="ground truth")
    cols = {"points only": "tab:orange", "points + lines": "tab:purple"}
    for name, (C_al, err, sc) in results.items():
        ax[0].plot(C_al[:, 2], C_al[:, 0], "--", color=cols[name], lw=1.8, label=name)
        if len(C_al) < N_FRAMES:                       # tracking failure 지점 표시
            ax[0].plot(C_al[-1, 2], C_al[-1, 0], "x", color=cols[name], ms=12, mew=3)
            ax[1].axvline(len(C_al), color=cols[name], ls=":", lw=1)
        ax[1].plot(err, color=cols[name], label=name)
        ax[2].plot(sc, color=cols[name], label=name)
    ax[0].set(title="top view (Sim(3)-aligned)", xlabel="z [m]", ylabel="x [m]", ylim=(-2.3, 2.3))
    ax[0].legend(loc="upper left", fontsize=9)
    ax[1].set(title="position error after alignment", xlabel="frame", ylabel="[m]")
    ax[1].legend(fontsize=9)
    ax[2].axhline(1.0, color="k", lw=0.8)
    ax[2].set(title="local scale ratio |ΔC_est| / |ΔC_gt| (10-frame window)", xlabel="frame", ylabel="ratio")
    ax[2].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("vo_result.png", dpi=130)
    print("saved vo_result.png")