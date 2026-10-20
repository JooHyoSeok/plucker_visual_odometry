# Point + Plücker-line monocular VO with loop closure (KITTI)

## Files
| file | role |
|---|---|
| `cpp/` | C++ port (Eigen + OpenCV, Pangolin viewer): `cpp/build/run_vo` |
| `results/` | result images / videos |
| `python/vo_plucker.py` | geometry core: Plücker lines, triangulation, pose estimation, frame-by-frame `VOTracker` (`run_vo` = batch wrapper; also a synthetic-corridor demo when run directly) |
| `python/frontend.py` | real-data frontend, one frame at a time: KLT points + LSD lines, 2D-2D init / relative pose, camera setup |
| `python/slam_online.py` | **online SLAM**: `OnlineSLAM.track(img) -> pose`; loop closure + Sim(3) PGO corrects the map and tracking continues on it; live view / video |
| `python/kitti_batch.py` | batch pipeline (offline): frontend over all frames, VO, loop closure + Sim(3) PGO, evaluation |
| `python/local_ba.py` | sliding-window BA (points + lines in orthonormal 4-DoF), enabled with `--ba` |
| `python/loop_closure.py` | loop detection (thumbnail shortlist + ORB + F-RANSAC + 3D-3D Sim(3) RANSAC) and Sim(3) pose graph |
| `python/kitti_batch_video.py` | renders the result video from the run file |

All Python modules live in `python/` and import each other (run them as `python3 python/<file>.py`).

## Setup
```bash
python3 -m pip install numpy scipy matplotlib opencv-python   # OpenCV >= 4.5.1 (needs LSD)
# ffmpeg CLI is needed only for the video (macOS: brew install ffmpeg)
```

## Data
Either the original KITTI odometry grayscale images + poses:
```
dataset/sequences/06/image_0/000000.png ...
dataset/poses/06.txt
```
or the video version used here (KITTI 06 redistributed in the pyslam repo):
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/luigifreda/pyslam.git data_dl/pyslam
cd data_dl/pyslam && git sparse-checkout set data/videos/kitti06 && cd ../..
```

## C++ (`cpp/`, real-time)
```bash
git submodule update --init cpp/third_party/g2o
cmake -S cpp -B cpp/build -DPangolin_DIR=<Pangolin build dir> -DVP_PYTHON=<python with Depth Anything 3>  # viewer optional
cmake --build cpp/build -j8
cpp/build/run_vo --video dataset/sequences/00/image_2/%06d.png --gt dataset/poses/00.txt
```
- The default is the best configuration so far (PROGRESS.md sections 10-12): loop closure (thumbnail retrieval, ORB / Sim(3)
  verification refined by bidirectional reprojection, Sim(3) pose graph in a loop thread), DA3 metric depth
  (DA3METRIC-LARGE in `python/loop_server.py`, Apache 2.0, ~3 GB GPU, 27 ms per KITTI keyframe, inferred online) as scale anchors in the pose
  graph, per-point log-depth priors in local BA, single-view Plücker lines and 3D for loop verification, local BA in a
  LocalMapping thread (bad observations dropped, points kept), map reuse by ORB projection search, tracking-failure recovery
  and a fallback initialization from DA3 depth, and one global BA at the end (`*_pregba.txt` = the same run before it).
- Options: `--no-loop`, `--no-da3` (plain monocular, no Python / GPU), `--no-viewer`, `--autostart` (skip the viewer start
  menu, where loop closure / detector / verifier / lines / DA3 can be chosen). Without `--calib fx,fy,cx,cy` the intrinsics
  come from `calib.txt` next to the image folder (KITTI odometry, EuRoC converter output). Unknown options are an error.
- Results: KITTI 00, 02-10 threaded with online DA3 (one run each) Sim(3) ATE 2.99 / 15.55 / 2.22 / 1.17 / 2.83 / 4.40 /
  0.84 / 10.96 / 7.57 / 2.68 m, mean 5.12 m (ORB-SLAM + GBA 11.13); EuRoC MH_01 / V1_01 / V2_01 (`--sync`) 0.47 / 0.23 / 0.12 m.
  Table with other methods: `cpp/tools/kitti_table_tex.py` (LaTeX -> PDF / PNG).
- Experiment options: `[N_FRAMES]`, `--sync` (loop closure and local BA inline: deterministic), `--da3-cache DIR` (DA3 depth
  maps stored by image hash: identical runs without GPU, for comparisons only; fill with `cpp/tools/da3_cache.py`),
  `--kf-out FILE`, `--realtime HZ`, `--start N`, `--stride N`, `--resize S`, `--no-lines`, `--traj-online FILE`,
  `--dump-map FILE`, `--snapshot FILE`, `--loop-pr thumb|netvlad|salad|mast3r`, `--loop-ver orb|mast3r|orb+mast3r` (learned
  modules in `python/loop_server.py`, MASt3R is CC BY-NC-SA), `--loop-python PY`, `--da3-range MIN,MAX` (default from the DA3
  depth of the first frame: indoor 0.5-20 m, outdoor 3-40 m), `--da3-model metric|nested` (nested = DA3NESTED-GIANT-LARGE,
  1.40B, CC BY-NC, 3.3x slower), `--depth PATTERN` (per-frame metric depth, e.g. iPhone LiDAR).
- KITTI in the viewer, one sequence after another, then an ATE table: `cpp/tools/run_kitti_viewer.sh [OUT_DIR] [SEQ ...]`.
- Result tools: `cpp/tools/make_video.py` (image + trajectory + error/local-scale panel; run_vo `--traj-online`),
  `cpp/tools/plot_traj.py --paper` (paper-style trajectory figure, PNG + PDF), `cpp/tools/plot_vs_orbslam.py`.
- EuRoC MAV: `python3 cpp/tools/euroc_to_kitti.py ~/Desktop/hyoseok/EuRoC/*/SEQ` (undistorted cam0 + camera GT in
  KITTI layout), then `benchmark.py --suite euroc`.
- iPhone (Stray Scanner): `python3 cpp/tools/stray_to_kitti.py odometry.csv gt.txt`, then
  `run_vo --video rgb.mp4 --gt gt.txt --calib <camera_matrix.csv> --resize 0.5 --stride 2`.
- Benchmark over many sequences: `python3 cpp/tools/benchmark.py --suite kitti --args="--sync --da3-cache DIR"`,
  plots: `cpp/tools/plot_bench.py`, `cpp/tools/plot_traj.py`.

## Run (online)
```python
from slam_online import OnlineSLAM
slam = OnlineSLAM(fx, fy, cx, cy, W, H)          # use_lines=True, loop=True, feedback=True, ba=False
for img in images:                               # BGR or gray, rectified, one at a time
    T_wc = slam.track(img)                       # 4x4 camera->world, None while initializing / after lost
```
```bash
# video file, image pattern, or camera index (--video 0); --gt only for evaluation / plot
python3 python/slam_online.py --video dataset/sequences/06/image_0/%06d.png --gt dataset/poses/06.txt \
                              --calib 707.0912,707.0912,601.8873,183.1104 --out online06.mp4   # --show: live window
python3 python/slam_online.py --selftest          # map-correction check
```
Options: first positional number = first N frames, `--ba`, `--no-lines`, `--no-feedback` (loop result only reported,
VO not corrected = same result as the batch pipeline). Writes `online_traj.txt` (KITTI format, T_wc per frame).

## Run (batch)
```bash
# default = data_dl/pyslam/data/videos/kitti06 (video.mp4 + 06.txt)
python3 python/kitti_batch.py
python3 python/kitti_batch_video.py

# original KITTI images (image sequence pattern works directly)
python3 python/kitti_batch.py --video dataset/sequences/06/image_0/%06d.png \
                              --gt dataset/poses/06.txt \
                              --calib 707.0912,707.0912,601.8873,183.1104
python3 python/kitti_batch_video.py --video dataset/sequences/06/image_0/%06d.png --gt dataset/poses/06.txt
```
Options for `kitti_batch.py`
- first positional number: use only the first N frames (e.g. `python3 python/kitti_batch.py 300`)
- `--ba`: turn on local BA (outputs `batch_run_ba.pkl` / `batch_result_ba.png`; pass `--ba` to the video script too)
- `--calib fx,fy,cx,cy`: KITTI 00-02 `718.856,718.856,607.1928,185.2157`, 03 `721.5377,721.5377,609.5593,172.854`, 04-12 `707.0912,707.0912,601.8873,183.1104`

Outputs (current directory): `batch_result.png` (trajectory + error), `batch_run.pkl` (all results), `batch_vo.mp4` (video).
Runtime for the full 1101-frame sequence: about 6-8 min (+ about 10 min more with `--ba`).

## Notes
- GT is used only for evaluation (origin-anchored alignment: same start pose, scale fitted on the first 50 frames), never for tracking or loop detection.
- Inputs must be rectified (no lens distortion). For other datasets, convert GT to KITTI format (one 3x4 `T_w_c` per line, 12 numbers).
- Tuning knobs: `frontend.py` (`MAX_PTS`, `MAX_LINES`, `MIN_LINE_LEN`, `INIT_FRAME`, `TRI_GAP` in `setup_camera`), `loop_closure.py` (`MIN_GAP`, `SHORTLIST`, `MIN_SIM3_INLIERS`, PGO weights `W_ODOM`, `W_LOOP`), `local_ba.py` (`WINDOW`, `EXTRA_FIXED`).