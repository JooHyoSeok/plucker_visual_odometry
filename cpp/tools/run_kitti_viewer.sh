#!/bin/bash
# KITTI odometry 를 Pangolin 뷰어로 한 시퀀스씩 보면서 돌리고 (궤적 / 로그 저장), 끝에 시퀀스별 Sim(3) ATE + 평균 표.
# 시퀀스가 끝나면 ("done - close the viewer window to exit") 창을 닫으면 다음 시퀀스로. 전체 중단은 Ctrl+C.
# usage: cpp/tools/run_kitti_viewer.sh [OUT_DIR] [SEQ ...]
#   기본 OUT_DIR = out_bench/viewer_kitti, SEQ = 00 02 03 04 05 06 07 08 09 10 (01 은 초기화가 안 돼서 뺌)
#   run_vo 인자 추가는 VO_ARGS (예: VO_ARGS="--no-loop" cpp/tools/run_kitti_viewer.sh out_bench/viewer_noloop)
#   표만 다시 만들기: TABLE_ONLY=1 cpp/tools/run_kitti_viewer.sh OUT_DIR [SEQ ...]
ROOT=/home/hyoseok/Desktop/hyoseok/Visual_SLAM_for_Study/vo_plucker
KITTI=/home/hyoseok/Desktop/hyoseok/KITTI_Dataset/dataset/sequences
GT=/home/hyoseok/Desktop/hyoseok/KITTI_poses/dataset/poses
PY=/home/hyoseok/miniconda3/envs/mast3r/bin/python
OUT=$(realpath -m "${1:-$ROOT/out_bench/viewer_kitti}")
[ $# -gt 0 ] && shift
SEQS=${*:-00 02 03 04 05 06 07 08 09 10}
mkdir -p "$OUT"

if [ -z "$TABLE_ONLY" ]; then
  for s in $SEQS; do
    img=image_2
    [ -e "$KITTI/$s/image_2/000000.png" ] || img=image_3
    echo "=== KITTI $s ($img) -> $OUT ==="
    stdbuf -oL -eL "$ROOT/cpp/build/run_vo" --video "$KITTI/$s/$img/%06d.png" --gt "$GT/$s.txt" \
      --traj "$OUT/kitti_${s}_mono.txt" --autostart $VO_ARGS 2>&1 | tee "$OUT/kitti_${s}_mono.log"
  done
fi

"$PY" - "$OUT" $SEQS <<'EOF'
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/hyoseok/Desktop/hyoseok/Visual_SLAM_for_Study/vo_plucker/cpp/tools")
import benchmark as bm
out, seqs = Path(sys.argv[1]), sys.argv[2:]
jobs = {j["seq"]: j for j in bm.kitti_jobs("^(" + "|".join(seqs) + ")$")}
lines = ["| Method | " + " | ".join(seqs) + " | Avg. |", "|" + "---|" * (len(seqs) + 2)]
for name, suffix in (("Ours (before global BA)", "_pregba"), ("Ours", "")):
    cells, vals = [], []
    for s in seqs:
        log, traj = out / f"kitti_{s}_mono.log", out / f"kitti_{s}_mono{suffix}.txt"
        if s not in jobs or not log.exists() or not traj.exists():
            cells.append("-")
            continue
        r = bm.evaluate(jobs[s], traj, log.read_text(errors="replace"), 0)
        if not np.isfinite(r["ate_sim3_m"]):
            cells.append("-")
            continue
        vals.append(r["ate_sim3_m"])
        cells.append(f"{r['ate_sim3_m']:.2f}" + ("*" if r["tracked_pct"] < 95 else ""))
    avg = "-" if not vals else f"{np.mean(vals):.2f}" + ("" if len(vals) == len(seqs) else f" ({len(vals)}/{len(seqs)})")
    lines.append(f"| {name} | " + " | ".join(cells) + f" | {avg} |")
table = "\n".join(lines) + "\n\nSim(3) ATE [m], same run before / after the final global BA. * = tracked < 95 %\n"
print(table)
(out / "ate_table.md").write_text(table)
print("saved", out / "ate_table.md")
EOF
