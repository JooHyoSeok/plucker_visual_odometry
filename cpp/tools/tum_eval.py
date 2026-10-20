"""
TUM RGB-D evaluation of a run_vo trajectory with evo (like `evo_ape tum groundtruth.txt est.txt -as`): the KITTI-format
trajectory (one T_wc per frame from the first posed frame) gets the image timestamps of SEQ/kitti/times.txt, is written
in TUM format next to it (*.tum) and associated with groundtruth.txt by time (max 20 ms), so frames without GT (e.g. the
90 s gap of freiburg2_large_no_loop) are left out instead of being compared with interpolated GT.

usage: python3 tum_eval.py SEQ_DIR TRAJ.txt [RUN.log]   (RUN.log: for "first pose at frame N", default 0)
"""
import re
import sys
from pathlib import Path

import numpy as np
from evo.core import metrics, sync
from evo.tools import file_interface
from scipy.spatial.transform import Rotation


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    seq, traj = Path(sys.argv[1]), Path(sys.argv[2])
    f0 = 0
    if len(sys.argv) > 3:
        m = re.search(r"first pose at frame (\d+)", Path(sys.argv[3]).read_text(errors="replace"))
        f0 = int(m.group(1)) if m else 0
    T = np.loadtxt(traj).reshape(-1, 3, 4)
    ts = np.loadtxt(seq / "kitti" / "times.txt")[f0:f0 + len(T)]
    q = Rotation.from_matrix(T[:len(ts), :, :3]).as_quat()  # x y z w
    tum = traj.with_suffix(".tum")
    np.savetxt(tum, np.column_stack([ts, T[:len(ts), :, 3], q]), fmt="%.6f")
    ref = file_interface.read_tum_trajectory_file(str(seq / "groundtruth.txt"))
    est = file_interface.read_tum_trajectory_file(str(tum))
    ref, est = sync.associate_trajectories(ref, est, max_diff=0.02)
    est.align(ref, correct_scale=True)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, est))
    st = ape.get_all_statistics()
    print(f"{seq.name}: {len(T)} poses, {ref.num_poses} with GT | Sim(3) ATE rmse {st['rmse']:.3f} m "
          f"(mean {st['mean']:.3f}, max {st['max']:.3f}) -> {tum}")


if __name__ == "__main__":
    main()
