"""
Loop-closure plugin server for the C++ run_vo (--loop-pr / --loop-ver). One process, models loaded once (GPU),
line protocol on stdin/stdout (one reply line per request):

  add KF PATH                           -> ok                       (register a keyframe image)
  query KF K J1 J2 ...                  -> cand J S J S ...         (top-K of the candidates, similarity score)
  verify I J fx fy cx cy NI (u v d)*NI NJ (u v d)*NJ
                                        -> loop s r00 .. r22 t0 t1 t2 inliers   |   none
       (u v d) = keyframe pixels of VO map points with their VO depth -> per-keyframe VO-units-per-meter scale,
       result = Sim(3) cam_J -> cam_I in VO units (the same convention as the built-in ORB verifier)
  anchor KF N (u v d)*N                 -> anchor A n               (--depth da3: VO units per meter at keyframe KF =
                                                                     median(VO depth / DA3 metric depth) at the given
                                                                     pixels, image = the one added last for KF)
  depthat PATH N (u v)*N                 -> depth z1 .. zN           (--depth da3: DA3 metric depth at the pixels, 0 = invalid;
                                                                     used by run_vo --da3-init on the first frame)
  quit

place recognition (--pr): netvlad (hloc, VGG16-NetVLAD-Pitts30K), salad (DINOv2-SALAD),
                          mast3r (ASMK on the MASt3R encoder features, as in MASt3R-SLAM)
verification     (--ver): mast3r (metric ViT-L): descriptor reciprocal-NN matches -> essential with our K (R, t dir),
                          |t| from PnP on the pointmap, VO/metric depth ratios at the given pixels -> Sim(3) scale.
--pr mast3r --ver mast3r share one model: the encoder runs once per keyframe (features cached), retrieval uses those
features, verification runs only the decoder on the cached pair.
Model code is imported from the installed packages (hloc, SALAD repo, MASt3R); weights from the paths below.
Licenses differ per model (MASt3R: CC BY-NC-SA 4.0) - check before non-research use.

depth anchor     (--depth): da3 (Depth Anything 3 DA3NESTED-GIANT-LARGE metric depth at 1008 px; KITTI 06: frame-to-frame
                          spread of the median ratio 6 %, 18 % global bias that does not matter for a scale anchor;
                          fitting the depth-dependent bias per frame made the spread worse, 9.6 %)

usage: python3 loop_server.py --pr mast3r --ver mast3r [--depth da3] [--device cuda]
"""
import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HOME = Path.home() / "Desktop/hyoseok"
SALAD_REPO = HOME / "3D_Reconstructions/VGGT-SLAM/salad"
SALAD_CKPT = Path(torch.hub.get_dir()) / "checkpoints/dino_salad.ckpt"
MAST3R_CKPT = HOME / "MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
DA3_REPO = HOME / "Depth-Anything-3"
MAST3R_RETRIEVAL = HOME / "MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------- place recognition ----------------
class NetVLAD:
    def __init__(self, device):
        from hloc.extractors.netvlad import NetVLAD as Net
        self.net, self.device = Net({}).eval().to(device), device

    @torch.no_grad()
    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self.device)
        return torch.nn.functional.normalize(self.net({"image": x})["global_descriptor"][0], dim=0)


class Salad:
    def __init__(self, device):
        # backbone + aggregator 만 조립 (salad.vpr_model 은 pytorch_lightning 을 끌어와서 import 가 깨짐)
        sys.path.insert(0, str(SALAD_REPO))
        from salad.models_salad import helper
        net = torch.nn.Module()
        net.backbone = helper.get_backbone("dinov2_vitb14",
                                           {"num_trainable_blocks": 4, "return_token": True, "norm_layer": True})
        net.aggregator = helper.get_aggregator("SALAD", {"num_channels": 768, "num_clusters": 64, "cluster_dim": 128,
                                                         "token_dim": 256})
        net.load_state_dict(torch.load(SALAD_CKPT, map_location="cpu"))
        self.net, self.device = net.eval().to(device), device

    @torch.no_grad()
    def __call__(self, bgr):
        h = 224
        w = max(14, int(round(bgr.shape[1] * h / bgr.shape[0] / 14)) * 14)  # 종횡비 유지, 14의 배수 (DINOv2 patch)
        rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return torch.nn.functional.normalize(self.net.aggregator(self.net.backbone(x[None].to(self.device)))[0], dim=0)


# ---------------- MASt3R: 검색 (ASMK) + 검증, 모델 하나 ----------------
class Mast3r:
    """MASt3R-SLAM 방식: 키프레임마다 encoder 1회 -> 특징 캐시. 같은 특징으로 ASMK 검색 (--pr mast3r),
    후보 쌍은 캐시된 특징으로 decoder 만 돌려서 검증 (--ver mast3r)."""
    W = 512
    MIN_FRAC = 0.2
    MAX_T = 10.0  # [m] 루프 = 거의 같은 자리 재방문. 06: 같은 방향 56-115 m 뒤 쌍은 매칭은 많아도 Sim(3) 스케일이 엉망

    def __init__(self, device, retrieval):
        from mast3r.model import AsymmetricMASt3R
        self.net = AsymmetricMASt3R.from_pretrained(str(MAST3R_CKPT)).eval().to(device)
        self.device = device
        self.enc = {}  # kf -> (encoder feat fp16 [cpu], pos, true_shape, (sx, sy))
        self.db = None
        if retrieval:
            from mast3r_slam.retrieval_database import RetrievalDatabase
            self.db = RetrievalDatabase(str(MAST3R_RETRIEVAL), backbone=self.net, device=device)
            self.ivf, self.q = {}, {}  # kf -> ASMK 이미지 id (0, 1, 2, ... 순서대로여야 함), 로컬 특징

    @torch.no_grad()
    def add(self, kf, bgr):
        """width 512, height 16의 배수 (비등방 resize) -> encoder"""
        H, W = bgr.shape[:2]
        h = max(16, int(round(H * self.W / W / 16)) * 16)
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.W, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(127.5).sub(1)[None].to(self.device)
        shape = torch.tensor([[h, self.W]], device=self.device)
        feat, pos, _ = self.net._encode_image(x, shape)
        self.enc[kf] = (feat.half().cpu(), pos.cpu(), shape.cpu(), (self.W / W, h / H))
        if self.db is not None:
            q = self.db.prep_features(feat)[0].cpu().numpy()
            ids = np.full(len(q), len(self.ivf), np.int64)
            self.db.add_to_database(q, ids, None)
            self.ivf[kf], self.q[kf] = int(ids[0]), q

    def query(self, kf, cands, k):
        q = self.q[kf]
        ranks, ranked, _ = self.db.query(q, np.zeros(len(q), np.int64))
        score = np.empty(ranked.shape[1])
        score[ranks[0]] = ranked[0]
        return sorted(((float(score[self.ivf[j]]), j) for j in cands), reverse=True)[:k]

    @torch.no_grad()
    def decode(self, i, j):
        (fi, pi, si, _), (fj, pj, sj, _) = self.enc[i], self.enc[j]
        d = self.device
        dec1, dec2 = self.net._decoder(fi.to(d).float(), pi.to(d), fj.to(d).float(), pj.to(d))
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            r1 = self.net._downstream_head(1, [t.float() for t in dec1], si.to(d))
            r2 = self.net._downstream_head(2, [t.float() for t in dec2], sj.to(d))
        return r1, r2  # r2["pts3d"]: view j 점을 cam i 좌표로

    def verify(self, i, j, K, pts_i, pts_j):
        r1, r2 = self.decode(i, j)
        sx, sy = self.enc[i][3]
        X1 = r1["pts3d"][0].cpu().numpy()   # view i points, cam i frame [m]
        X2 = r2["pts3d"][0].cpu().numpy()   # view j points, cam i frame [m]
        c1 = r1["conf"][0].cpu().numpy()
        c2 = r2["conf"][0].cpu().numpy()
        h, w = X1.shape[:2]
        Kn = np.array([[K[0] * sx, 0, K[2] * sx], [0, K[1] * sy, K[3] * sy], [0, 0, 1]])
        # 회전 + 이동 방향: MASt3R 디스크립터 상호 최근접 매칭 -> 우리 K로 essential RANSAC
        # (pointmap을 바로 PnP하면 MASt3R가 암묵적으로 쓰는 focal과 K가 달라서 회전이 수 도 틀어짐)
        from mast3r.fast_nn import fast_reciprocal_NNs
        xy_i, xy_j = fast_reciprocal_NNs(r1["desc"][0], r2["desc"][0], subsample_or_initxy1=4,
                                         device=self.device, dist="dot", block_size=2 ** 13)
        # 게이트: essential 인라이어 / 4px 그리드 수. KITTI 06 참 루프 0.55-0.78, 반대 방향 오검출 0.01-0.03
        n_min = self.MIN_FRAC * (h // 4) * (w // 4)
        if len(xy_i) < n_min:
            return None
        E, inl = cv2.findEssentialMat(xy_i.astype(np.float64), xy_j.astype(np.float64), Kn, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3) or inl.sum() < n_min:
            return None
        n_in, R_ji, t_dir, _ = cv2.recoverPose(E, xy_i.astype(np.float64), xy_j.astype(np.float64), Kn, mask=inl)
        # 이동 크기 [m]: view-j 픽셀 <-> 그 3D (cam i) PnP 의 baseline 길이 (방향보다 크기가 안정적)
        vv, uu = np.mgrid[0:h:2, 0:w:2]
        m = c2[vv, uu] > np.percentile(c2, 50)
        obj, img = X2[vv[m], uu[m]].astype(np.float64), np.c_[uu[m], vv[m]].astype(np.float64)
        ok, rvec, tvec, pin = cv2.solvePnPRansac(obj, img, Kn, None, iterationsCount=200, reprojectionError=2.0,
                                                 flags=cv2.SOLVEPNP_EPNP)
        if not ok or pin is None or len(pin) < 50:
            return None
        if np.linalg.norm(tvec) > self.MAX_T:
            return None
        t_ji = t_dir.ravel() * np.linalg.norm(tvec)
        # VO 단위/미터 = median(VO depth / MASt3R depth) at the given keyframe pixels
        def ratio(pts, depth_map):
            if len(pts) == 0:
                return None
            u = np.clip((pts[:, 0] * sx).astype(int), 0, w - 1)
            v = np.clip((pts[:, 1] * sy).astype(int), 0, h - 1)
            z = depth_map[v, u]
            ok = z > 1e-3
            return float(np.median(pts[ok, 2] / z[ok])) if ok.sum() >= 10 else None
        a_i = ratio(pts_i, X1[..., 2])
        a_j = ratio(pts_j, (X2 @ R_ji.T + t_ji)[..., 2])           # view-j depth in its own frame
        if a_i is None or a_j is None:
            return None
        R_ij, t_ij = R_ji.T, -R_ji.T @ t_ji                          # cam j -> cam i [m]
        return a_i / a_j, R_ij, a_i * t_ij, int(n_in)           # VO units: X_i = (a_i/a_j) R X_j + a_i t


# ---------------- metric depth anchor ----------------
class Da3Depth:
    RES, ZMIN, ZMAX = 1008, 3.0, 40.0  # DA3 깊이 이 범위의 점만 (가까운 곳은 편향 크고 먼 곳은 VO 깊이가 부정확)

    def __init__(self, device, cache=None):
        # cache: 디렉터리면 영상 해시별 깊이 맵 (fp16) 을 저장 / 재사용 -> 실행끼리 결정적 + 캐시가 차면 GPU 안 씀 (모델은 처음 놓칠 때 로드)
        self.device, self.cache, self.net = device, Path(cache) if cache else None, None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        else:
            self.load()

    def load(self):
        sys.path.insert(0, str(DA3_REPO / "src"))
        from depth_anything_3.api import DepthAnything3
        self.net = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to(self.device).eval()
        if self.cache:
            log("[loop_server] DA3 model loaded (depth cache miss)")

    @torch.no_grad()
    def depth_map(self, bgr):
        f = self.cache / f"{hashlib.md5(bgr.tobytes()).hexdigest()}_{self.RES}.npy" if self.cache else None
        if f is not None and f.exists():
            return np.load(f).astype(np.float32)
        if self.net is None:
            self.load()
        d = self.net.inference([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)], process_res=self.RES).depth[0]
        if f is not None:
            np.save(f, d.astype(np.float16))
            d = d.astype(np.float16).astype(np.float32)  # 캐시를 쓴 실행과 똑같은 값
        return d

    def at(self, bgr, uv):  # DA3 metric depth [m] at pixels uv (n, 2). 깊이 맵은 sample() 용으로 남겨 둠
        self.dep = self.depth_map(bgr)
        self.sy, self.sx = self.dep.shape[0] / bgr.shape[0], self.dep.shape[1] / bgr.shape[1]
        return self.sample(uv)

    def sample(self, uv):  # 마지막 at() 깊이 맵에서
        h, w = self.dep.shape
        return self.dep[np.clip((uv[:, 1] * self.sy).astype(int), 0, h - 1), np.clip((uv[:, 0] * self.sx).astype(int), 0, w - 1)]

    def __call__(self, bgr, pts):
        z = self.at(bgr, pts[:, :2])
        ok = (z > self.ZMIN) & (z < self.ZMAX) & (pts[:, 2] > 0)
        return (float(np.median(pts[ok, 2] / z[ok])), int(ok.sum()), np.where(ok, z, 0)) if ok.sum() >= 20 else None


def main():
    reply_out = sys.stdout  # 프로토콜 응답 전용. 라이브러리가 stdout에 찍는 메시지는 stderr로 돌림
    sys.stdout = sys.stderr
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", default="none", choices=["none", "netvlad", "salad", "mast3r"])
    ap.add_argument("--ver", default="none", choices=["none", "mast3r"])
    ap.add_argument("--depth", default="none", choices=["none", "da3"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--depth-cache", default="", help="directory for DA3 depth maps keyed by image hash (deterministic, GPU-free reruns)")
    ap.add_argument("--depth-range", default="3,40", help="DA3 depth used for anchors / priors / 3D [m] (indoor: 0.5,20)")
    o = ap.parse_args()
    Da3Depth.ZMIN, Da3Depth.ZMAX = (float(x) for x in o.depth_range.split(","))
    m3r = Mast3r(o.device, retrieval=o.pr == "mast3r") if "mast3r" in (o.pr, o.ver) else None  # 검색/검증이 공유
    pr = {"none": None, "netvlad": NetVLAD, "salad": Salad, "mast3r": None}[o.pr]
    pr = pr(o.device) if pr else None
    depth = Da3Depth(o.device, o.depth_cache or None) if o.depth == "da3" else None
    log(f"[loop_server] ready: pr={o.pr} ver={o.ver} depth={o.depth} device={o.device}")
    desc, last_img, depth_kf = {}, {}, -1
    for line in sys.stdin:
        t = line.split()
        if not t:
            continue
        try:
            if t[0] == "quit":
                break
            if t[0] == "add":
                kf, img = int(t[1]), cv2.imread(t[2])
                if pr:
                    desc[kf] = pr(img)
                if m3r:
                    m3r.add(kf, img)
                if depth:
                    last_img = {kf: img}  # anchor 요청은 add 바로 뒤에 옴
                reply = "ok"
            elif t[0] == "query":
                kf, k, cands = int(t[1]), int(t[2]), [int(x) for x in t[3:]]
                s = m3r.query(kf, cands, k) if o.pr == "mast3r" else \
                    sorted(((float(desc[kf] @ desc[j]), j) for j in cands), reverse=True)[:k]
                reply = "cand " + " ".join(f"{j} {v:.4f}" for v, j in s)
            elif t[0] == "anchor":
                kf, n = int(t[1]), int(t[2])
                pts = np.array(t[3:3 + 3 * n], float).reshape(-1, 3)
                r = depth(last_img.pop(kf), pts) if depth and kf in last_img else None
                depth_kf = kf
                reply = "none" if r is None else f"anchor {r[0]:.9g} {r[1]} " + " ".join(f"{v:.3f}" for v in r[2])
            elif t[0] == "depthkf":  # 바로 앞 anchor 키프레임의 깊이 맵 재사용 (DA3 추가 추론 없음)
                kf, n = int(t[1]), int(t[2])
                uv = np.array(t[3:3 + 2 * n], float).reshape(-1, 2)
                z = depth.sample(uv) if depth and n and kf == depth_kf else np.zeros(n)
                z = np.where((z > Da3Depth.ZMIN) & (z < Da3Depth.ZMAX), z, 0)  # 앵커와 같은 범위만
                reply = "depth " + " ".join(f"{v:.3f}" for v in z)
            elif t[0] == "depthat":
                n = int(t[2])
                uv = np.array(t[3:3 + 2 * n], float).reshape(-1, 2)
                z = depth.at(cv2.imread(t[1]), uv) if depth and n else np.zeros(0)
                reply = "depth " + " ".join(f"{v:.4f}" for v in z)
            elif t[0] == "verify":
                i, j = int(t[1]), int(t[2])
                K = [float(x) for x in t[3:7]]
                ni = int(t[7])
                pi = np.array(t[8:8 + 3 * ni], float).reshape(-1, 3)
                nj = int(t[8 + 3 * ni])
                pj = np.array(t[9 + 3 * ni:9 + 3 * ni + 3 * nj], float).reshape(-1, 3)
                r = m3r.verify(i, j, K, pi, pj) if o.ver == "mast3r" else None
                reply = "none" if r is None else \
                    "loop " + " ".join(f"{x:.9g}" for x in [r[0], *r[1].ravel(), *r[2]]) + f" {r[3]}"
            else:
                reply = "error unknown command"
        except Exception as e:  # 한 요청이 실패해도 서버는 계속
            log(f"[loop_server] {t[0]} failed: {e!r}")
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()  # 다른 GPU 작업과 겹쳐 잠깐 모자란 경우 다음 요청은 되게
            reply = "none" if t[0] == "verify" else "error"
        print(reply, file=reply_out, flush=True)


if __name__ == "__main__":
    main()
