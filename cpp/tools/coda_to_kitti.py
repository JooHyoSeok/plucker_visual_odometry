"""
CODa (UT Campus Object Dataset) -> KITTI-style ground truth for run_vo on the rectified cam0 images.

  CODA/poses/dense/SEQ.txt                      ts x y z qw qx qy qz: os1 LiDAR pose in the world, one row per frame
  CODA/calibrations/SEQ/calib_os1_to_cam0.yaml  extrinsic_matrix = T_cam0_os1 (LiDAR -> unrectified cam0)
  CODA/calibrations/SEQ/calib_cam0_intrinsics.yaml  rectification_matrix R_rect (cam0 -> rectified), projection_matrix P
  -> OUT/poses.txt  T_w_cam of the rectified cam0 per frame (3x4): T_w_os1 * T_cam0_os1^-1 * [R_rect^T 0; 0 1]
     OUT/calib.txt  "fx fy cx cy" of P (the 2d_rect images), OUT/times.txt (frame timestamps [s])
  images: CODA/2d_rect/cam0/SEQ/2d_rect_cam0_SEQ_%d.png, frame k <-> pose row k

usage: python3 coda_to_kitti.py CODA_ROOT SEQ [OUT_DIR]   (default OUT_DIR = CODA_ROOT/kitti/SEQ)
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def mat(path, key, rows, cols):
    return np.array(yaml.safe_load(Path(path).read_text())[key]["data"], dtype=float).reshape(rows, cols)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    root, seq = Path(sys.argv[1]), sys.argv[2]
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else root / "kitti" / seq
    cal = root / "calibrations" / seq
    T_c_l = mat(cal / "calib_os1_to_cam0.yaml", "extrinsic_matrix", 4, 4)
    R_rect = mat(cal / "calib_cam0_intrinsics.yaml", "rectification_matrix", 3, 3)
    P = mat(cal / "calib_cam0_intrinsics.yaml", "projection_matrix", 3, 4)
    p = np.loadtxt(root / "poses" / "dense" / f"{seq}.txt")
    T_w_l = np.tile(np.eye(4), (len(p), 1, 1))
    T_w_l[:, :3, :3] = Rotation.from_quat(p[:, [5, 6, 7, 4]]).as_matrix()  # qw qx qy qz -> scipy x y z w
    T_w_l[:, :3, 3] = p[:, 1:4]
    Rr = np.eye(4)
    Rr[:3, :3] = R_rect.T
    T_w_c = T_w_l @ np.linalg.inv(T_c_l) @ Rr
    n_img = len(list((root / "2d_rect" / "cam0" / seq).glob(f"2d_rect_cam0_{seq}_*.png")))
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "poses.txt", T_w_c[:, :3, :].reshape(-1, 12), fmt="%.9f")
    np.savetxt(out / "times.txt", p[:, 0], fmt="%.6f")
    (out / "calib.txt").write_text(f"{P[0, 0]:.6f} {P[1, 1]:.6f} {P[0, 2]:.6f} {P[1, 2]:.6f}\n")
    path = np.linalg.norm(np.diff(T_w_c[:, :3, 3], axis=0), axis=1).sum()
    print(f"CODa {seq}: {len(p)} poses, {n_img} images, path {path:.1f} m, {p[-1, 0] - p[0, 0]:.0f} s, "
          f"K = {P[0, 0]:.2f} {P[1, 1]:.2f} {P[0, 2]:.2f} {P[1, 2]:.2f} -> {out}")


if __name__ == "__main__":
    main()
