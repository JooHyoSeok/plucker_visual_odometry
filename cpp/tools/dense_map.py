"""
Dense map v1 (offline): DA3 metric depth on the keyframes of a run_vo result, back-projected with the final keyframe poses
and the per-keyframe VO-units-per-meter a_k (run_vo --kf-out), fused into one point cloud (voxel grid).
Optional evaluation against a GT point cloud (EuRoC pointcloud0/data.ply): the dense cloud is mapped to the GT world with
the Sim(3) that aligns the keyframe camera centers to the GT camera centers of the same frames (as for the ATE), then
  accuracy     = distance from each dense point to the nearest GT point (median, % < 5 / 10 cm)
  completeness = % of GT points with a dense point within 5 / 10 cm (GT restricted to the part seen by the keyframes).

usage: python3 dense_map.py --kf KF.txt --images 'seq/image_0/%06d.png' --calib fx,fy,cx,cy --out cloud.ply
                            [--every 2] [--stride 4] [--range 0.5,20] [--voxel 0.05]
                            [--gt-traj poses.txt --gt-cloud data.ply]   (GT: KITTI format T_wc per frame)
       python3 dense_map.py --kf KF.txt --cloud online.ply --out aligned.ply --gt-traj poses.txt --gt-cloud data.ply
         (evaluate run_vo --dense-out instead: no DA3 inference, same alignment / metrics)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

DA3_SRC = Path.home() / "Desktop/hyoseok/Depth-Anything-3/src"


def load_kf(path):
    a = np.loadtxt(path)
    return a[:, 0].astype(int), a[:, 1], a[:, 2:11].reshape(-1, 3, 3), a[:, 11:14]


def sim3(P, Q):  # Q ≈ s R P + t (Umeyama)
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(Y.T @ X / len(P))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (X ** 2).sum(1).mean()
    return s, R, mq - s * R @ mp


def voxel(points, size):
    keys = np.floor(points / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kf", required=True)
    ap.add_argument("--images")
    ap.add_argument("--calib")
    ap.add_argument("--cloud", help="run_vo --dense-out PLY (binary xyz float + rgb uchar, VO units) instead of DA3 inference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--range", default="0.5,20")
    ap.add_argument("--voxel", type=float, default=0.05, help="[m] (converted with the median a_k)")
    ap.add_argument("--edge", type=float, default=0.05, help="drop pixels with |grad d| / d above this (flying pixels)")
    ap.add_argument("--gt-traj")
    ap.add_argument("--gt-cloud")
    o = ap.parse_args()
    frames, a, R, t = load_kf(o.kf)
    if o.cloud:
        W = read_ply(o.cloud)
        print(f"{o.cloud}: {len(W)} points")
    else:
        W = infer(o, frames, a, R, t)
    evaluate(o, W, frames, R, t)


def read_ply(path):  # run_vo --dense-out: binary_little_endian, float x y z + uchar r g b
    raw = Path(path).read_bytes()
    head = raw[:raw.index(b"end_header\n") + 11]
    n = int(next(l for l in head.decode().splitlines() if l.startswith("element vertex")).split()[2])
    v = np.frombuffer(raw[len(head):], dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "u1", 3)]), count=n)
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def infer(o, frames, a, R, t):
    fx, fy, cx, cy = (float(x) for x in o.calib.split(","))
    zmin, zmax = (float(x) for x in o.range.split(","))
    sys.path.insert(0, str(DA3_SRC))
    from depth_anything_3.api import DepthAnything3
    net = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to("cuda").eval()
    clouds = []
    for n in range(0, len(frames), o.every):
        img = cv2.imread(o.images % frames[n])
        h, w = img.shape[:2]
        with torch.no_grad():
            d = net.inference([cv2.cvtColor(img, cv2.COLOR_BGR2RGB)], process_res=1008).depth[0]
        d = cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        gy, gx = np.gradient(d)
        ok = (d > zmin) & (d < zmax) & (np.hypot(gx, gy) / np.maximum(d, 1e-6) < o.edge)
        v, u = np.mgrid[0:h:o.stride, 0:w:o.stride]
        m = ok[v, u]
        z = d[v, u][m] * a[n]  # DA3 미터 -> 이 키프레임의 VO 단위
        Xc = np.stack([(u[m] - cx) / fx * z, (v[m] - cy) / fy * z, z], 1)
        clouds.append((Xc - t[n]) @ R[n])  # world = R^T (Xc - t)
        print(f"keyframe {frames[n]:5d}: {m.sum():6d} points (a_k {a[n]:.4f})", flush=True)
    W = np.vstack(clouds)
    W = voxel(W, o.voxel * np.median(a))
    print(f"fused: {len(W)} points (voxel {o.voxel} m)")
    return W


def evaluate(o, W, frames, R, t):
    out = W
    if o.gt_traj:
        G = np.loadtxt(o.gt_traj).reshape(-1, 3, 4)[:, :, 3]
        C = -np.einsum("nji,nj->ni", R, t)  # 키프레임 중심 = -R^T t
        k = frames < len(G)
        s, Rg, tg = sim3(C[k], G[frames[k]])
        print(f"keyframe ATE after Sim(3): {np.sqrt((((s * C[k] @ Rg.T + tg) - G[frames[k]]) ** 2).sum(1).mean()):.3f} m")
        out = s * W @ Rg.T + tg
        if o.gt_cloud:
            g = np.loadtxt(o.gt_cloud, skiprows=11, usecols=(0, 1, 2), max_rows=4_000_000)
            dg, _ = cKDTree(g).query(out, workers=-1)
            print(f"accuracy (dense -> GT): median {np.median(dg) * 100:.1f} cm, mean {dg.mean() * 100:.1f} cm, "
                  f"< 5 cm {np.mean(dg < 0.05) * 100:.0f} %, < 10 cm {np.mean(dg < 0.10) * 100:.0f} %")
            near = cKDTree(out).query(g, workers=-1)[0]
            seen = near < 0.5  # 키프레임이 본 쪽만 (먼 GT 는 완전성에서 뺌)
            print(f"completeness (GT seen part, {seen.sum()} pts): < 5 cm {np.mean(near[seen] < 0.05) * 100:.0f} %, "
                  f"< 10 cm {np.mean(near[seen] < 0.10) * 100:.0f} %")
    with open(o.out, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(out)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, out, fmt="%.4f")
    print("saved", o.out)


if __name__ == "__main__":
    main()
