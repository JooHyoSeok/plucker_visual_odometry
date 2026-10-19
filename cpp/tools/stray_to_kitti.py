"""
Stray Scanner (iPhone ARKit) odometry.csv -> KITTI pose file for run_vo --gt.

odometry.csv: "timestamp, frame, x, y, z, qx, qy, qz, qw" per RGB frame, T_wc with an OpenCV-style camera
(x right, y down, z forward: checked on ICRA Basement_01, walking direction = camera +z).
output: one 3x4 T_wc per line (12 numbers), same row order as rgb.mp4 frames.

usage: python3 stray_to_kitti.py odometry.csv out_kitti.txt
"""
import sys

import numpy as np
from scipy.spatial.transform import Rotation

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
a = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
R = Rotation.from_quat(a[:, 5:9]).as_matrix()          # scipy order (x, y, z, w)
np.savetxt(sys.argv[2], np.concatenate([R, a[:, 2:5, None]], axis=2).reshape(-1, 12), fmt="%.9f")
print(f"{len(a)} poses -> {sys.argv[2]}")
