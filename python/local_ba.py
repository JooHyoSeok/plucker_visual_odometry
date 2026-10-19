"""
Sliding-window local bundle adjustment for vo_plucker (points + Plücker lines).

  variables : poses of the window keyframes (the oldest N_FIXED are fixed -> fixes the 7-DoF gauge incl. scale)
              points  X (3, absolute)
              lines   orthonormal representation (Bartoli & Sturm 2005): L <-> (U in SO(3), phi)
                      m = cos(phi) U[:,0],  d = sin(phi) U[:,1]   -> d·m = 0 always, 4-DoF update (dθ, dφ)
  residuals : point reprojection [px], line = distance of observed segment endpoints to projected line [px]
  solver    : own sparse LM (finite-difference Jacobian by column groups, Huber IRLS, scipy spsolve)
"""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

import vo_plucker as v

KF_EVERY = 3       # 키프레임 간격 [frames]
WINDOW = 10        # 창에 들어가는 키프레임 수
N_FIXED = 2        # 최소 고정 키프레임 수 (단안: 2개 이상 고정해야 스케일까지 고정됨)
EXTRA_FIXED = 20   # 창 앞쪽 과거 키프레임을 최대 이만큼 고정 포즈로 포함
HUBER_PX = 2.0
MAX_ITERS = 8
MIN_WIN_PARALLAX = 0.0   # 창 안 관측들 사이 최소 parallax [deg]
MIN_OBS = 2          # 창 안 키프레임 관측 수 하한
_FIX_POSES = False   # (디버그) 포즈 고정
_FIX_STRUCT = False  # (디버그) 구조 고정
ONLY_MATURE = False  # True: parallax가 충분한(MATURE) 점만 BA에 넣음


# ---------------- Plücker <-> orthonormal ----------------
def to_orth(L):
    d, m = L[:3], L[3:]
    nm, nd = np.linalg.norm(m), np.linalg.norm(d)
    c = np.cross(m, d)
    U = np.c_[m / nm, d / nd, c / np.linalg.norm(c)]
    return U, np.arctan2(nd, nm)


def from_orth(U, phi):
    m = np.cos(phi) * U[:, 0]
    d = np.sin(phi) * U[:, 1]
    s = np.linalg.norm(d)
    return np.concatenate([d, m]) / s          # |d| = 1 로 정규화 (vo_plucker 규약)


# ---------------- BA ----------------
def bundle_adjust(frames, Rs, ts, obs, map_p, map_l, use_lines, allowed_p=None, n_fixed=None):
    """frames: 창의 키프레임 인덱스 (오름차순). Rs/ts/map_p/map_l 을 제자리에서 갱신.
    반환: 최적화에 들어간 (point ids, line ids)"""
    n_fixed = N_FIXED if n_fixed is None else n_fixed
    free = frames[n_fixed:]
    fidx = {f: i for i, f in enumerate(free)}

    # 창 안에서 2번 이상 관측된 랜드마크만
    cnt_p, cnt_l = {}, {}
    for f in frames:
        for j in obs[f][0]:
            if j in map_p:
                cnt_p[j] = cnt_p.get(j, 0) + 1
        if use_lines:
            for j in obs[f][1]:
                if j in map_l and np.linalg.norm(map_l[j][3:]) > 1e-6:
                    cnt_l[j] = cnt_l.get(j, 0) + 1
    in_free_p = {j for f in free for j in obs[f][0]}
    in_free_l = {j for f in free for j in obs[f][1]}
    pids = [j for j, c in cnt_p.items() if c >= MIN_OBS and j in in_free_p and (allowed_p is None or j in allowed_p)]
    # 창 안에서 parallax가 작은 점은 깊이가 위쪽으로 편향(ML 삼각측량 bias) -> 스케일을 끌고 감. 제외
    if MIN_WIN_PARALLAX > 0:
        keep = []
        for j in pids:
            fs = [f for f in frames if j in obs[f][0]]
            if v.parallax_deg(map_p[j], Rs[fs[0]], ts[fs[0]], Rs[fs[-1]], ts[fs[-1]]) >= MIN_WIN_PARALLAX:
                keep.append(j)
        pids = keep
    lids = [j for j, c in cnt_l.items() if c >= MIN_OBS and j in in_free_l]
    if len(pids) + len(lids) < 15:
        return [], []
    pidx = {j: i for i, j in enumerate(pids)}
    lidx = {j: i for i, j in enumerate(lids)}

    # 관측 목록
    P_f, P_j, P_uv = [], [], []
    L_f, L_j, L_seg = [], [], []
    for f in frames:
        for j, u in obs[f][0].items():
            if j in pidx:
                P_f.append(f); P_j.append(pidx[j]); P_uv.append(u)
        for j, s in obs[f][1].items():
            if j in lidx:
                L_f.append(f); L_j.append(lidx[j]); L_seg.append(s)
    P_f, P_j, P_uv = np.array(P_f), np.array(P_j), np.array(P_uv).reshape(-1, 2)
    L_f, L_j, L_seg = np.array(L_f, int), np.array(L_j, int), np.array(L_seg).reshape(-1, 4)

    R0 = {f: Rs[f] for f in frames}
    t0 = {f: ts[f] for f in frames}
    orth0 = [to_orth(map_l[j]) for j in lids]
    nP6, nX3 = 6 * len(free), 3 * len(pids)
    x0 = np.concatenate([np.zeros(nP6), np.array([map_p[j] for j in pids]).ravel(), np.zeros(4 * len(lids))])

    def unpack(x):
        Rd, td = {}, {}
        for f in frames:
            if f in fidx:
                dx = x[6 * fidx[f]: 6 * fidx[f] + 6]
                Rd[f], td[f] = v.retract(R0[f], t0[f], dx)
            else:
                Rd[f], td[f] = R0[f], t0[f]
        X = x[nP6: nP6 + nX3].reshape(-1, 3)
        Ls = []
        for i in range(len(lids)):
            dl = x[nP6 + nX3 + 4 * i: nP6 + nX3 + 4 * i + 4]
            U, phi = orth0[i]
            Ls.append(from_orth(U @ v.so3_exp(dl[:3]), phi + dl[3]))
        return Rd, td, X, np.array(Ls).reshape(-1, 6)

    fr_list = list(frames)
    fpos = {f: i for i, f in enumerate(fr_list)}

    def fun(x):
        Rd, td, X, Ls = unpack(x)
        RR = np.array([Rd[f] for f in fr_list])
        TT = np.array([td[f] for f in fr_list])
        pi = np.array([fpos[f] for f in P_f])
        Xc = np.einsum("nij,nj->ni", RR[pi], X[P_j]) + TT[pi]
        p = Xc @ v.K.T
        z = np.where(np.abs(p[:, 2:3]) < 1e-6, 1e-6, p[:, 2:3])
        r = [(p[:, :2] / z - P_uv).ravel()]
        if len(L_f):
            li = np.array([fpos[f] for f in L_f])
            d = np.einsum("nij,nj->ni", RR[li], Ls[L_j, :3])
            m = np.einsum("nij,nj->ni", RR[li], Ls[L_j, 3:]) + np.cross(TT[li], d)
            l = m @ v.KL.T
            nl = np.hypot(l[:, 0], l[:, 1])[:, None]
            s = np.c_[L_seg[:, :2], np.ones(len(L_seg))]
            e = np.c_[L_seg[:, 2:], np.ones(len(L_seg))]
            r.append((np.c_[(s * l).sum(1), (e * l).sum(1)] / nl).ravel())
        return np.concatenate(r)

    # ---------- sparse LM (직접 구현) ----------
    # scipy least_squares(trf)는 이 문제에서 수렴이 매우 느렸음 (100회 반복에도 미수렴, dense LM은 6회).
    # 그래서 열 그룹 유한차분으로 희소 Jacobian을 만들고 (JᵀWJ + λD)δ = -JᵀWr 를 sparse solve.
    nres = 2 * len(P_f) + 2 * len(L_f)
    rows = np.arange(nres)
    obs_frame = np.r_[np.repeat(P_f, 2), np.repeat(L_f, 2)]
    pose_col = np.array([6 * fidx[f] if f in fidx else -1 for f in obs_frame], int)
    lm_col = np.r_[nP6 + 3 * np.repeat(P_j, 2), nP6 + nX3 + 4 * np.repeat(L_j, 2)].astype(int)
    is_pt = np.r_[np.ones(2 * len(P_f), bool), np.zeros(2 * len(L_f), bool)]

    groups = []                                   # (perturbation column set, row->column map)
    for c in range(6):                            # 서로 다른 프레임의 포즈는 같은 행을 공유하지 않음
        cols = np.array([6 * i + c for i in range(len(free))], int)
        rc = np.where(pose_col >= 0, pose_col + c, -1)
        groups.append((cols, rc))
    for c in range(3):                            # 서로 다른 점도 같은 행을 공유하지 않음
        cols = nP6 + 3 * np.arange(len(pids)) + c
        groups.append((cols, np.where(is_pt, lm_col + c, -1)))
    for c in range(4):
        cols = nP6 + nX3 + 4 * np.arange(len(lids)) + c
        groups.append((cols, np.where(~is_pt, lm_col + c, -1)))

    def jac(x, r):
        I, Jc, V = [], [], []
        for cols, rc in groups:
            if len(cols) == 0:
                continue
            h = 1e-6
            dx = np.zeros_like(x)
            dx[cols] = h
            dr = (fun(x + dx) - r) / h
            m = rc >= 0
            I.append(rows[m]); Jc.append(rc[m]); V.append(dr[m])
        return csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(Jc))), shape=(nres, len(x)))

    def cost_of(r):
        a = np.abs(r)
        return np.sum(np.where(a <= HUBER_PX, 0.5 * a ** 2, HUBER_PX * (a - 0.5 * HUBER_PX)))

    x = x0.copy()
    r = fun(x)
    cost, lam = cost_of(r), 1e-4
    for _ in range(MAX_ITERS):
        J = jac(x, r)
        a = np.abs(r)
        w = np.where(a <= HUBER_PX, 1.0, HUBER_PX / np.maximum(a, 1e-12))
        JW = J.multiply(w[:, None]).tocsr()
        Hm = (J.T @ JW).tocsc()
        g = JW.T @ r
        dH = Hm.diagonal()
        improved = False
        for _ in range(8):
            dx = spsolve(Hm + diags(lam * (dH + 1e-9)), -g)
            if _FIX_POSES:
                dx[:nP6] = 0
            if _FIX_STRUCT:
                dx[nP6:] = 0
            r1 = fun(x + dx)
            c1 = cost_of(r1) if np.all(np.isfinite(r1)) else np.inf
            if c1 < cost:
                x, r, cost = x + dx, r1, c1
                lam = max(lam / 5, 1e-8)
                improved = True
                break
            lam *= 10
        if not improved or np.abs(dx).max() < 1e-8:
            break

    class _R:
        pass
    res = _R()
    res.x = x
    Rd, td, X, Ls = unpack(res.x)

    # 결과 반영: 키프레임 포즈, 그리고 키프레임 사이 비키프레임은 앞 키프레임과 같은 world 보정을 적용
    for f in free:
        Rs[f], ts[f] = v.orthonormalize(Rd[f]), td[f]
    # 잔차가 큰 랜드마크는 맵에서 제거
    r = fun(res.x)
    rp = np.linalg.norm(r[: 2 * len(P_f)].reshape(-1, 2), axis=1)
    bad_p = set(np.array(P_j)[rp > 3 * HUBER_PX])
    rl = np.abs(r[2 * len(P_f):].reshape(-1, 2)).max(1) if len(L_f) else np.zeros(0)
    bad_l = set(np.array(L_j)[rl > 3 * HUBER_PX]) if len(L_f) else set()
    for i, j in enumerate(pids):
        if i in bad_p:
            map_p.pop(j, None)
        else:
            map_p[j] = X[i]
    for i, j in enumerate(lids):
        if i in bad_l:
            map_l.pop(j, None)
        else:
            map_l[j] = Ls[i]
    return [j for i, j in enumerate(pids) if i not in bad_p], [j for i, j in enumerate(lids) if i not in bad_l]


def make_ba_hook():
    """vo_plucker.run_vo(..., post_frame_hook=hook) 용. 키프레임마다 창 BA 실행."""
    state = {"kfs": [], "n_ba": 0}

    def hook(k, ctx):
        if k % KF_EVERY != 0 and k != v.INIT_FRAME:
            return
        kfs = state["kfs"]
        if kfs and k <= kfs[-1]:
            return
        kfs.append(k)
        if len(kfs) < N_FIXED + 2:
            return
        # 창 = 고정 키프레임(과거, 같은 랜드마크를 본 것들) + 자유 키프레임(최근 WINDOW개)
        # 과거 관측을 고정 포즈로 같이 넣어야 스케일이 오래된(검증된) 구조에 묶임 (ORB-SLAM local BA 방식)
        n_fix = max(N_FIXED, min(EXTRA_FIXED, len(kfs) - WINDOW))
        win = kfs[-(WINDOW + n_fix):] if len(kfs) > WINDOW + n_fix else kfs
        n_fix = len(win) - min(WINDOW, len(win) - N_FIXED)
        Rs, ts = ctx["Rs"], ctx["ts"]
        old = {f: (Rs[f], ts[f]) for f in range(win[0], k + 1) if Rs[f] is not None}
        pids, lids = bundle_adjust(win, Rs, ts, ctx["obs"], ctx["map_p"], ctx["map_l"], ctx["use_lines"],
                                   allowed_p=set(ctx["mature_p"]) if ONLY_MATURE else None, n_fixed=n_fix)
        # BA가 맡은 랜드마크는 DLT 재삼각측량 대상에서 제외
        ctx["mature_p"].update(pids)
        ctx["mature_l"].update(lids)
        # 비키프레임 포즈: 직전 키프레임의 보정을 같이 적용 (T_f <- T_f T_kf_old^-1 T_kf_new)
        for f in range(win[0], k + 1):
            if f in win or Rs[f] is None or f < win[n_fix]:
                continue
            kf = max(w for w in win if w <= f)
            Rko, tko = old[kf]
            Rkn, tkn = Rs[kf], ts[kf]
            # T_f_new = T_f_old * T_kf_old^-1 * T_kf_new
            Rc = Rko.T @ Rkn
            tc = Rko.T @ (tkn - tko)
            Rf, tf = old[f]
            Rs[f], ts[f] = v.orthonormalize(Rf @ Rc), Rf @ tc + tf
        state["n_ba"] += 1

    hook.state = state
    return hook