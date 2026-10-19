"""
EuRoC MAV (ASL folder) -> KITTI-style input for run_vo: undistorted cam0 images + camera GT poses.

  SEQ/mav0/cam0/{data.csv, data/*.png, sensor.yaml}, SEQ/mav0/state_groundtruth_estimate0/data.csv
  -> SEQ/kitti/image_0/%06d.png  (radtan undistorted, new K with all pixels valid, same size)
     SEQ/kitti/poses.txt         (T_w_cam0 per kept frame, 3x4 = 12 numbers: GT body pose x T_BS, slerp to image time)
     SEQ/kitti/calib.txt         ("fx fy cx cy" of the undistorted images), times.txt (image timestamps [s])
  only frames inside the GT time range are kept.

usage: python3 euroc_to_kitti.py SEQ_DIR [SEQ_DIR ...]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def nums(text, key):  # sensor.yaml: key 다음에 처음 나오는 [...] (T_BS 는 cols/rows/data: [...] 블록)
    i = text.index(key + ":")
    a, b = text.index("[", i), text.index("]", i)
    return np.array([float(x) for x in text[a + 1:b].split(",")])


def convert(seq):
    cam = seq / "mav0" / "cam0"
    y = (cam / "sensor.yaml").read_text()
    fu, fv, cu, cv = nums(y, "intrinsics")
    D = nums(y, "distortion_coefficients")
    T_BS = nums(y, "T_BS").reshape(4, 4)
    w, h = nums(y, "resolution").astype(int)
    K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
    Kn, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)  # alpha 0: 검은 테두리 없이
    mx, my = cv2.initUndistortRectifyMap(K, D, None, Kn, (w, h), cv2.CV_32FC1)

    im = np.loadtxt(cam / "data.csv", delimiter=",", dtype=str, comments="#")
    t_im = im[:, 0].astype(np.int64)
    g = np.loadtxt(seq / "mav0" / "state_groundtruth_estimate0" / "data.csv", delimiter=",", comments="#")
    t_g = g[:, 0].astype(np.int64)
    keep = (t_im >= t_g[0]) & (t_im <= t_g[-1])
    ts = t_im[keep]
    q = Rotation.from_quat(g[:, [5, 6, 7, 4]])  # (w x y z) -> scipy (x y z w)
    R_wb = Slerp(t_g.astype(float), q)(ts.astype(float)).as_matrix()
    p_wb = np.stack([np.interp(ts, t_g, g[:, k]) for k in (1, 2, 3)], axis=1)
    T = np.zeros((len(ts), 4, 4))
    T[:, :3, :3], T[:, :3, 3], T[:, 3, 3] = R_wb, p_wb, 1
    T = T @ T_BS  # T_w_cam0

    out = seq / "kitti"
    (out / "image_0").mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(im[keep, 1]):
        img = cv2.imread(str(cam / "data" / name.strip()), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(out / "image_0" / f"{k:06d}.png"), cv2.remap(img, mx, my, cv2.INTER_LINEAR))
    np.savetxt(out / "poses.txt", T[:, :3, :].reshape(-1, 12), fmt="%.9f")
    np.savetxt(out / "times.txt", ts / 1e9, fmt="%.9f")
    (out / "calib.txt").write_text(f"{Kn[0, 0]:.6f} {Kn[1, 1]:.6f} {Kn[0, 2]:.6f} {Kn[1, 2]:.6f}\n")
    print(f"{seq.name}: {keep.sum()}/{len(t_im)} frames in GT range, K' = {Kn[0, 0]:.1f} {Kn[1, 1]:.1f} {Kn[0, 2]:.1f} {Kn[1, 2]:.1f}"
          f", path {np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1).sum():.1f} m -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for s in sys.argv[1:]:
        convert(Path(s))
