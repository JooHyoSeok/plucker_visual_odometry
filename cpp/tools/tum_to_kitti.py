"""
TUM RGB-D (rgbd_dataset_freiburgN_*) -> KITTI-style input for run_vo: undistorted RGB images + camera GT poses.

  SEQ/{rgb.txt, rgb/*.png, groundtruth.txt}
  -> SEQ/kitti/image_0/%06d.png  (colour; freiburg1/2 undistorted with the published parameters, new K with all pixels
                                  valid; freiburg3 is already undistorted -> copied)
     SEQ/kitti/poses.txt         (T_w_cam per kept frame, 3x4: groundtruth.txt is the colour camera's optical centre,
                                  interpolated to the image time: linear position, slerp rotation)
     SEQ/kitti/calib.txt         ("fx fy cx cy"), times.txt (image timestamps [s])
  only frames inside the GT time range are kept.

usage: python3 tum_to_kitti.py SEQ_DIR [SEQ_DIR ...]
"""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats (ROS default / calibrated intrinsics, d0-d4)
CAMS = {"freiburg1": ([517.3, 516.5, 318.6, 255.3], [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]),
        "freiburg2": ([520.9, 521.0, 325.1, 249.7], [0.2312, -0.7849, -0.0033, -0.0001, 0.9172]),
        "freiburg3": ([535.4, 539.2, 320.1, 247.6], [0, 0, 0, 0, 0])}


def convert(seq):
    cam = next(k for k in CAMS if k in seq.name)
    (fx, fy, cx, cy), D = CAMS[cam]
    K, D = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), np.array(D)
    im = [l.split() for l in (seq / "rgb.txt").read_text().splitlines() if l and not l.startswith("#")]
    g = np.array([[float(x) for x in l.split()] for l in (seq / "groundtruth.txt").read_text().splitlines()
                  if l and not l.startswith("#")])
    g = g[np.unique(g[:, 0], return_index=True)[1]]  # 시간순, 같은 시각 중복 제거 (fr2_large_no_loop 에 있음)
    t_im = np.array([float(a) for a, _ in im])
    keep = (t_im >= g[0, 0]) & (t_im <= g[-1, 0])
    ts = t_im[keep]
    R_wc = Slerp(g[:, 0], Rotation.from_quat(g[:, 4:8]))(ts).as_matrix()  # qx qy qz qw = scipy 순서
    p_wc = np.stack([np.interp(ts, g[:, 0], g[:, k]) for k in (1, 2, 3)], axis=1)
    out = seq / "kitti"
    (out / "image_0").mkdir(parents=True, exist_ok=True)
    w, h = 640, 480
    Kn = K
    if D.any():
        Kn, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)  # alpha 0: 검은 테두리 없이
        mx, my = cv2.initUndistortRectifyMap(K, D, None, Kn, (w, h), cv2.CV_32FC1)
    for k, (_, name) in enumerate(x for x, ok in zip(im, keep) if ok):
        dst = out / "image_0" / f"{k:06d}.png"
        if D.any():
            cv2.imwrite(str(dst), cv2.remap(cv2.imread(str(seq / name)), mx, my, cv2.INTER_LINEAR))
        else:
            shutil.copyfile(seq / name, dst)
    T = np.concatenate([R_wc, p_wc[:, :, None]], axis=2)
    np.savetxt(out / "poses.txt", T.reshape(-1, 12), fmt="%.9f")
    np.savetxt(out / "times.txt", ts, fmt="%.6f")
    (out / "calib.txt").write_text(f"{Kn[0, 0]:.6f} {Kn[1, 1]:.6f} {Kn[0, 2]:.6f} {Kn[1, 2]:.6f}\n")
    path = np.linalg.norm(np.diff(p_wc, axis=0), axis=1).sum()
    print(f"{seq.name}: {keep.sum()}/{len(im)} frames in GT range, K' = {Kn[0, 0]:.1f} {Kn[1, 1]:.1f} {Kn[0, 2]:.1f} "
          f"{Kn[1, 2]:.1f}, path {path:.1f} m, {ts[-1] - ts[0]:.0f} s -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for a in sys.argv[1:]:
        convert(Path(a).resolve())
