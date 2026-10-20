#!/bin/bash
# Microsoft 7-Scenes (Kinect v1 RGB-D, 640x480, 30 Hz, handheld) 한 장면 / 시퀀스를 뷰어로 돌리고 Sim(3) ATE.
# 없으면 내려받고 (장면 zip, 예: chess 3.1 GB) 그 시퀀스만 풀고, GT 를 KITTI 형식으로 바꿔서 run_vo (지금 기본 설정).
# usage: cpp/tools/run_7scenes.sh [SCENE=chess] [SEQ=01] [run_vo 인자 ...]
#   데이터: /home/hyoseok/Desktop/hyoseok/7-Scenes/SCENE/seq-SEQ/frame-%06d.color.png, frame-%06d.pose.txt (4x4 camera-to-world)
#   GT: SCENE/seq-SEQ/kitti/poses.txt (프레임마다 3x4), 결과: out_bench/7scenes/SCENE_seqSEQ{.txt,_pregba.txt,.log}
#   내부 파라미터: 데이터셋이 RGB 카메라 보정값을 주지 않음 -> 보정 추정값 fx 532.57, fy 531.54, cx 320, cy 240
#   (Kinect 공칭 525 보다 같은 DA3 깊이 결정적 실행에서 chess 46.6 -> 46.4, heads 73.4 -> 61.1, stairs 46.3 -> 33.0 mm)
ROOT=/home/hyoseok/Desktop/hyoseok/Visual_SLAM_for_Study/vo_plucker
DATA=/home/hyoseok/Desktop/hyoseok/7-Scenes
PY=/home/hyoseok/miniconda3/envs/mast3r/bin/python
SCENE=${1:-chess}
SEQ=${2:-01}
[ $# -gt 0 ] && shift
[ $# -gt 0 ] && shift
D=$DATA/$SCENE/seq-$SEQ
set -e
if [ ! -e "$D/frame-000000.color.png" ]; then
  mkdir -p "$DATA"
  [ -e "$DATA/$SCENE/seq-$SEQ.zip" ] || {
    wget -c "https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/$SCENE.zip" -O "$DATA/$SCENE.zip"
    unzip -q -o "$DATA/$SCENE.zip" -d "$DATA"
  }
  unzip -q -o "$DATA/$SCENE/seq-$SEQ.zip" -d "$DATA/$SCENE"
fi
mkdir -p "$D/kitti" "$ROOT/out_bench/7scenes"
[ -e "$D/kitti/poses.txt" ] || "$PY" -I - "$D" <<'EOF'
import sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1])
P = [np.loadtxt(p) for p in sorted(d.glob("frame-*.pose.txt"))]
np.savetxt(d / "kitti" / "poses.txt", np.array([T[:3, :].reshape(-1) for T in P]), fmt="%.9f")
C = np.array([T[:3, 3] for T in P])
print(f"{d}: {len(P)} GT poses, path {np.linalg.norm(np.diff(C, axis=0), axis=1).sum():.1f} m -> {d / 'kitti' / 'poses.txt'}")
EOF
set +e
OUT=$ROOT/out_bench/7scenes/${SCENE}_seq$SEQ
stdbuf -oL -eL "$ROOT/cpp/build/run_vo" --video "$D/frame-%06d.color.png" --calib 532.57,531.54,320,240 --gt "$D/kitti/poses.txt" \
  --traj "$OUT.txt" --autostart "$@" 2>&1 | tee "$OUT.log"
echo "=== $SCENE seq-$SEQ ==="
grep -E "frames tracked|^ATE" "$OUT.log"
