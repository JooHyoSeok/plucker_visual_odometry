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
cmake -S cpp -B cpp/build -DPangolin_DIR=<Pangolin build dir>   # viewer is optional (built only if Pangolin is found)
cmake --build cpp/build -j8
cpp/build/run_vo --video dataset/sequences/00/image_2/%06d.png --gt dataset/poses/00.txt \
                 --calib 718.856,718.856,607.1928,185.2157 --loop --sim3-align
```
- Viewer start screen: choose loop closure, detector, verifier and lines in the menu, then press Start
  (`--autostart` skips it, `--no-viewer` uses the flags only).
- Loop closure: `--loop` = built-in thumbnail retrieval + ORB / 3D-3D Sim(3) verification + Sim(3) pose graph (g2o),
  written back into poses and map. Pluggable modules: `--loop-pr thumb|netvlad|salad|mast3r`,
  `--loop-ver orb|mast3r|orb+mast3r` (orb+mast3r: ORB geometric check first, then MASt3R Sim(3)).
  The learned modules run in `python/loop_server.py`, started with `--loop-python <python with hloc / SALAD / MASt3R>`
  (checkpoint paths at the top of that file; MASt3R is CC BY-NC-SA).
- Loop closing runs in its own thread (ORB-SLAM3 style: tracking holds the map lock per frame, PGO runs without it).
  `--loop-sync` runs it inline (deterministic, for A/B comparisons). `--loop-skip-chi2 X` (default 50): a new loop
  edge that already fits the current estimate is added without re-running the PGO.
- KITTI Sim(3) ATE with the built-in loop closure: 00 101 m -> about 5 m, 06 48 m -> 14-23 m (varies run to run).
- Metric depth scale anchor `--depth-prior da3` (Depth Anything 3 DA3NESTED-GIANT-LARGE in `python/loop_server.py`, needs
  `--loop-python` with depth_anything_3; ~8-10 GB GPU): per keyframe VO-units-per-meter from DA3 vs VO depth -> log-scale
  prior edges in the Sim(3) pose graph every 10 keyframes (loop thread). KITTI 00-10 without loops: drift 2.20 -> 0.56 %,
  t_rel 12.2 -> 2.35 %. Best combination so far: `--depth-prior da3 --loop`. `--da3-init`: initialize from DA3 depth on frame 0.
- With DA3 (PROGRESS.md section 10): `--kf-parallax 25` (fewer keyframes, 2.4x faster), `--local-ba --ba-depth-prior 0.3`
  (local BA with a per-point DA3 log-depth prior from the anchor depths, BA can no longer drift the window scale),
  `--loop-da3-3d --loop-min-f 40` (ORB loop verification falls back to DA3-depth 3D of the ORB keypoints when the local
  3D is missing / poor; used only for the first link of a new revisit). Best configuration:
  `--depth-prior da3 --loop --kf-parallax 25 --local-ba --ba-depth-prior 0.3 --loop-da3-3d --loop-min-f 40 --loop-python <py>`.
  At most 2 DA3 servers on one 32 GB GPU.
- More options (PROGRESS.md section 11): `--final-gba` (global BA at the end: all keyframes + points + DA3 prior + loop
  pose edges, writes `*_pregba.txt` too), `--da3-lines` (Plücker lines from one view: inverse depth is affine along the image
  segment -> robust fit to DA3 depth, also lines parallel to the motion that cannot be triangulated), `--da3-points` (DA3
  depth for keyframe tracks without enough parallax), `--da3-range MIN,MAX` (DA3 depth used, indoor 0.5,20),
  `--realtime HZ` (feed frames at camera rate), `--fld` (FastLineDetector instead of LSD).
- Options (off by default, mixed results per sequence, see PROGRESS.md section 9): `--hybrid --hybrid-fix-rot` (translation
  refined by map + epipolar), `--loop-consist N --loop-prune R` (loop consistency / outlier edges), `--loop-fuse` (map point
  fusion + covisibility candidate exclusion), `--local-ba [--ba-lines]` (keyframe BA in a LocalMapping thread, Plücker lines in
  the orthonormal 4-DoF parameterization with analytic Jacobians; `--ba-sync` runs it inline).
- Result tools: `cpp/tools/make_video.py` (image + trajectory + error/local-scale panel; run_vo `--traj-online`),
  `cpp/tools/plot_traj.py --paper` (paper-style trajectory figure, PNG + PDF), `cpp/tools/plot_vs_orbslam.py`.
- EuRoC MAV: `python3 cpp/tools/euroc_to_kitti.py ~/Desktop/hyoseok/EuRoC/*/SEQ` (undistorted cam0 + camera GT in
  KITTI layout), then `benchmark.py --suite euroc`.
- Benchmark over many sequences: `python3 cpp/tools/benchmark.py --suite kitti --args="--loop"`,
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