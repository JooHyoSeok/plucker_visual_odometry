"""
Python reference for the C++ port: runs the Python frontend + VOTracker and dumps
  obs.txt        : tracker input per frame (points, lines, 2D-2D relative pose, init pose) - exact doubles
  ref_poses.txt  : Python VOTracker poses (T_cw) per frame

usage: python3 dump_reference.py --video PATH [--calib fx,fy,cx,cy] [n_frames] [--no-lines] --out DIR
format (one record per line):
  camera fx fy cx cy W H
  frame k
  p id u v
  l id x1 y1 x2 y2
  rel  r00 .. r22 t0 t1 t2      (R row-major)
  init r00 .. r22 t0 t1 t2
ref_poses.txt: "k 1 r00 .. r22 t0 t1 t2" or "k 0" (no pose)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # vo_plucker/
import vo_plucker as v
import frontend as fe


def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def num(x):
    return " ".join(repr(float(a)) for a in np.ravel(x))


def main():
    src, out = _arg("--video", None), _arg("--out", None)
    if src is None or out is None:
        raise SystemExit(__doc__)
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10 ** 9
    fx, fy, cx, cy = map(float, _arg("--calib", "707.0912,707.0912,601.8873,183.1104").split(","))
    use_lines = "--no-lines" not in sys.argv
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    H, W = frame.shape[:2]
    fe.setup_camera(fx, fy, cx, cy, W, H)
    front = fe.Frontend()
    vo = v.VOTracker(use_lines)
    t_vo = 0.0
    with open(f"{out}/obs.txt", "w") as f:
        f.write(f"camera {num([fx, fy, cx, cy])} {W} {H}\n")
        k = 0
        while ok and k < n_max:
            op, ol = front.step(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            rel = fe.relative_pose(vo.obs[-1][0], op) if k else None
            init = fe.robust_init(vo.obs[0][0], op) if k == v.INIT_FRAME else None
            f.write(f"frame {k}\n")
            f.writelines(f"p {j} {num(u)}\n" for j, u in op.items())
            f.writelines(f"l {j} {num(s)}\n" for j, s in ol.items())
            if rel is not None:
                f.write(f"rel {num(rel[0])} {num(rel[1])}\n")
            if init is not None:
                f.write(f"init {num(init[0])} {num(init[1])}\n")
            t0 = time.perf_counter()
            vo.add(op, ol, rel=rel, init_pose=init)
            t_vo += time.perf_counter() - t0
            k += 1
            ok, frame = cap.read()
    with open(f"{out}/ref_poses.txt", "w") as f:
        for i, (R, t) in enumerate(zip(vo.Rs, vo.ts)):
            f.write(f"{i} 0\n" if R is None else f"{i} 1 {num(R)} {num(t)}\n")
    print(f"{k} frames, lines {use_lines}, failed_at {vo.stats['failed_at']}, "
          f"python VOTracker {1000 * t_vo / k:.1f} ms/frame -> {out}")


if __name__ == "__main__":
    main()
