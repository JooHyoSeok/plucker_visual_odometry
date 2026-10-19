// C++ monocular VO (points + Plücker lines): images in one by one -> pose out. No loop closure yet.
// usage: run_vo [n_frames] --video PATH|CAM_INDEX [--gt PATH] [--calib fx,fy,cx,cy] [--no-lines] [--traj OUT.txt]
//               [--init-frame N | --init-px P] [--kf-mapping] [--no-reattach]
//               [--rel-static] [--drop-dynamic] [--rel-min-px P] [--resize S] [--stride N] [--start N] [--min-parallax DEG] [--init-min-points N] [--sim3-align] [--pose-g2o] [--local-ba] [--ba-window N] [--ba-fixed N] [--kf-parallax PX] [--no-viewer] [--snapshot OUT.png]
//   --resize S : 영상과 intrinsic을 S배 (고해상도 입력용, 예: iPhone 1920x1440 -> 0.5)
//   --stride N : N 프레임마다 하나만 사용 (고 fps / 느린 이동용). GT도 같은 간격으로 맞춤
//   --start N  : 처음 N 프레임 건너뛰기 (예: 시작이 제자리 회전뿐인 핸드헬드 영상). GT도 N부터
//   --depth PATTERN [--depth-conf PATTERN --depth-conf-min C] [--depth-max M] [--depth-scale S] [--depth-2d2d]
//              : 프레임별 metric depth 영상 (예: iPhone LiDAR depth/%06d.png, uint16 mm -> scale 0.001).
//                영상 번호 = 원본 프레임 번호. 깊이 초기화 + 깊이 맵 점 + 맵 기준 6-DoF 추적 (--depth-2d2d: 2D-2D 유지)
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <string>
#include <thread>

#include "vo_plucker/frontend.h"
#ifdef VP_WITH_VIEWER
#include "vo_plucker/viewer.h"
#endif

using namespace vp;
using Clock = std::chrono::steady_clock;

static const char* arg(int argc, char** argv, const char* name, const char* def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], name)) return argv[i + 1];
  return def;
}

static bool flag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], name)) return true;
  return false;
}

static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 두 프레임 공통 점 트랙의 픽셀 이동 중앙값 (공통 트랙이 min_common 미만이면 0)
static double median_shift(const FrameObs& a, const FrameObs& b, size_t min_common) {
  std::vector<double> d;
  for (size_t i = 0; i < b.pid.size(); ++i)
    if (const Vec2* u = a.point(b.pid[i])) d.push_back((*u - b.uv[i]).norm());
  if (d.size() < min_common) return 0;
  std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
  return d[d.size() / 2];
}

// 점별 깊이 [m]: 깊이 영상(작아도 됨, 같은 시야) nearest, conf >= conf_min, 3x3 이웃이 10% 넘게 다르면 (물체 경계) 버림
static int sample_depth(FrameObs& o, const cv::Mat& d16, const cv::Mat& conf, int conf_min, double max_m, double scale,
                        int W, int H) {
  o.depth.assign(o.pid.size(), NAN);
  if (d16.empty()) return 0;
  const double sx = double(d16.cols) / W, sy = double(d16.rows) / H;
  int n = 0;
  for (size_t i = 0; i < o.pid.size(); ++i) {
    const int x = int(o.uv[i](0) * sx), y = int(o.uv[i](1) * sy);
    if (x < 1 || y < 1 || x >= d16.cols - 1 || y >= d16.rows - 1) continue;
    if (!conf.empty() && conf.at<uchar>(y, x) < conf_min) continue;
    double lo = 1e9, hi = 0;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const double v = d16.at<uint16_t>(y + dy, x + dx) * scale;
        lo = std::min(lo, v), hi = std::max(hi, v);
      }
    if (lo <= 0 || hi > 1.1 * lo || hi > max_m) continue;
    o.depth[i] = d16.at<uint16_t>(y, x) * scale, ++n;
  }
  return n;
}

// KITTI pose file: one 3x4 T_wc per line
static std::vector<Eigen::Matrix<double, 3, 4>> load_kitti(const char* path) {
  std::vector<Eigen::Matrix<double, 3, 4>> P;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    Eigen::Matrix<double, 3, 4> T;
    for (int i = 0; i < 12; ++i) ss >> T(i / 4, i % 4);
    P.push_back(T);
  }
  return P;
}

// ATE RMSE: (a) origin-anchored (frame 0 카메라를 GT에 겹치고 스케일만 처음 50프레임으로, 드리프트가 그대로 보임)
//           (b) Sim(3) Umeyama 전체 정렬 (논문 표준)
static void print_ate(const std::vector<Vec3>& C, const std::vector<Eigen::Matrix<double, 3, 4>>& P) {
  const int n = std::min<int>(C.size(), P.size());
  if (n < 3) return;
  std::vector<Vec3> G(n);
  double path = 0;
  for (int f = 0; f < n; ++f) G[f] = P[f].col(3), path += f ? (G[f] - G[f - 1]).norm() : 0;
  const Mat3 R0 = P[0].leftCols<3>().transpose();
  const Vec3 t0 = -R0 * P[0].col(3);
  double num = 0, den = 0, se_a = 0, se_s = 0;
  for (int f = 0; f < std::min(n, 50); ++f) num += C[f].dot(R0 * G[f] + t0), den += C[f].squaredNorm();
  const double s = num / std::max(den, 1e-12);
  for (int f = 0; f < n; ++f) se_a += (R0.transpose() * (s * C[f] - t0) - G[f]).squaredNorm();
  Eigen::Matrix<double, 3, Eigen::Dynamic> X(3, n), Y(3, n);
  for (int f = 0; f < n; ++f) X.col(f) = C[f], Y.col(f) = G[f];
  const Eigen::Matrix4d S = Eigen::umeyama(X, Y, true), E = Eigen::umeyama(X, Y, false);
  double se_e = 0;
  for (int f = 0; f < n; ++f) {
    se_s += (S.topLeftCorner<3, 3>() * C[f] + S.topRightCorner<3, 1>() - G[f]).squaredNorm();
    se_e += (E.topLeftCorner<3, 3>() * C[f] + E.topRightCorner<3, 1>() - G[f]).squaredNorm();
  }
  std::printf("ATE (%.1f m): origin-anchored %.2f m | Sim(3)-aligned %.2f m (scale %.3f) | SE(3)-aligned, no scale fit %.2f m\n",
              path, std::sqrt(se_a / n), std::sqrt(se_s / n), std::cbrt(S.topLeftCorner<3, 3>().determinant()),
              std::sqrt(se_e / n));
}

int main(int argc, char** argv) {
  const char* src = arg(argc, argv, "--video", nullptr);
  if (!src) {
    std::fprintf(stderr,
                 "usage: %s [n_frames] --video PATH|CAM_INDEX [--gt PATH] [--calib fx,fy,cx,cy] [--no-lines] "
                 "[--traj OUT.txt] [--init-frame N | --init-px P] [--kf-mapping] [--no-reattach] [--rel-static] [--drop-dynamic] [--no-viewer] [--snapshot OUT.png]\n",
                 argv[0]);
    return 2;
  }
  const int n_max = argc > 1 && std::isdigit(argv[1][0]) ? std::atoi(argv[1]) : 1 << 30;
  double fx, fy, cx, cy;
  std::sscanf(arg(argc, argv, "--calib", "707.0912,707.0912,601.8873,183.1104"), "%lf,%lf,%lf,%lf", &fx, &fy, &cx, &cy);
  TrackerParams prm;
  prm.use_lines = !flag(argc, argv, "--no-lines");
  prm.kf_mapping = flag(argc, argv, "--kf-mapping");
  prm.rel_min_px = std::atof(arg(argc, argv, "--rel-min-px", "0"));
  // 점 삼각측량 최소 parallax [deg] (KITTI 기준 2), 초기화에 필요한 맵 점 수: 실내/느린 핸드헬드는 1, 50 정도
  prm.point_min_deg = std::atof(arg(argc, argv, "--min-parallax", "2"));
  prm.init_min_points = std::atoi(arg(argc, argv, "--init-min-points", "100"));
  const double rp_dist = std::atof(arg(argc, argv, "--rel-dist", "1e6"));  // 프레임 간 2D-2D recoverPose distanceThresh
  const char* depth_pat = arg(argc, argv, "--depth", nullptr);
  const char* conf_pat = arg(argc, argv, "--depth-conf", nullptr);
  const int conf_min = std::atoi(arg(argc, argv, "--depth-conf-min", "2"));
  const double depth_max = std::atof(arg(argc, argv, "--depth-max", "10"));
  const double depth_scale = std::atof(arg(argc, argv, "--depth-scale", "0.001"));
  prm.use_depth = depth_pat != nullptr;
  prm.depth_6dof = !flag(argc, argv, "--depth-2d2d");
  prm.pose_g2o = flag(argc, argv, "--pose-g2o");  // 6-DoF 추정을 g2o motion-only BA로
  prm.local_ba = flag(argc, argv, "--local-ba");  // 키프레임마다 local BA (g2o, 점만)
  prm.ba_window = std::atoi(arg(argc, argv, "--ba-window", "10"));
  prm.ba_fixed = std::atoi(arg(argc, argv, "--ba-fixed", "10"));
  prm.kf_parallax_px = std::atof(arg(argc, argv, "--kf-parallax", "10"));  // 키프레임: 회전 보정 parallax 중앙값 [px]
  // 초기화: --init-frame N이면 그 프레임, 아니면 frame 0 대비 픽셀 이동 중앙값이 --init-px 이상이 될 때 (07처럼 천천히 출발해도)
  const int init_fixed = std::atoi(arg(argc, argv, "--init-frame", "0"));
  const double init_px = std::atof(arg(argc, argv, "--init-px", "20"));  // 이 이상이면 초기화 시도, 품질이 안 되면 다음 프레임 재시도
  const double resize = std::atof(arg(argc, argv, "--resize", "1"));
  const int stride = std::max(1, std::atoi(arg(argc, argv, "--stride", "1")));
  const int start = std::max(0, std::atoi(arg(argc, argv, "--start", "0")));
  const char* gt_path = arg(argc, argv, "--gt", nullptr);
  std::vector<Eigen::Matrix<double, 3, 4>> gt;
  if (gt_path) {
    const auto all = load_kitti(gt_path);
    for (size_t i = start; i < all.size(); i += stride) gt.push_back(all[i]);
  }

  cv::VideoCapture cap;
  std::isdigit(src[0]) && !std::strchr(src, '.') ? cap.open(std::atoi(src)) : cap.open(src);
  cv::Mat frame, gray;
  auto read = [&]() {  // stride만큼 건너뛰고 한 장, resize 적용
    for (int s = 1; s < stride; ++s)
      if (!cap.grab()) return false;
    if (!cap.read(frame)) return false;
    if (resize != 1) cv::resize(frame, frame, cv::Size(), resize, resize, cv::INTER_AREA);
    return true;
  };
  for (int s = 0; s < start; ++s) cap.grab();
  if (!cap.read(frame)) {
    std::fprintf(stderr, "cannot read video/images/camera: %s\n", src);
    return 1;
  }
  if (resize != 1) cv::resize(frame, frame, cv::Size(), resize, resize, cv::INTER_AREA);
  const Camera cam(fx * resize, fy * resize, cx * resize, cy * resize, frame.cols, frame.rows);
  Frontend front;
  VOTracker vo(cam, prm);
  std::printf("source %s, %dx%d, K=(%g,%g,%g,%g), stride %d, lines %d\n", src, frame.cols, frame.rows, cam.fx, cam.fy,
              cam.cx, cam.cy, stride, prm.use_lines);
#ifdef VP_WITH_VIEWER
  std::unique_ptr<Viewer> viewer;
  if (!flag(argc, argv, "--no-viewer")) {
    std::vector<Pose> gt_cw;
    for (const auto& T : gt) gt_cw.push_back(Pose{T.leftCols<3>().transpose(), -T.leftCols<3>().transpose() * T.col(3)});
    viewer = std::make_unique<Viewer>(cam, gt_cw, flag(argc, argv, "--sim3-align"));
  }
#endif

  // 실험용 (기본 꺼짐, KITTI 06/07에서 이득 불분명): 2D-2D를 맵 점 트랙만으로 / 자기운동에 안 맞는 트랙 제거
  const bool no_reattach = flag(argc, argv, "--no-reattach"), rel_all = !flag(argc, argv, "--rel-static"),
             keep_dynamic = !flag(argc, argv, "--drop-dynamic");
  long n_reattach = 0, n_dynamic = 0, n_depth_obs = 0;
  double t_read = 0, t_front = 0, t_rel = 0, t_vo = 0, t_view = 0;
  const auto t_start = Clock::now();
  int k = 0;
  for (; k < n_max; ++k) {
    if (k) {
      const auto t0 = Clock::now();
      if (!read()) break;
      t_read += ms_since(t0);
    }
    auto t0 = Clock::now();
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    FrameObs o = front.step(gray);
    if (const auto pred = vo.predict(); pred && !no_reattach) n_reattach += front.reattach(cam, vo, *pred, o);
    if (depth_pat) {
      const int fi = start + k * stride;  // 원본 프레임 번호
      const cv::Mat d = cv::imread(cv::format(depth_pat, fi), cv::IMREAD_UNCHANGED);
      const cv::Mat c = conf_pat ? cv::imread(cv::format(conf_pat, fi), cv::IMREAD_UNCHANGED) : cv::Mat();
      n_depth_obs += sample_depth(o, d, c, conf_min, depth_max, depth_scale, cam.W, cam.H);
    }
    t_front += ms_since(t0);
    t0 = Clock::now();
    const std::optional<Pose> rel = k ? relative_pose(cam, vo.obs.back(), o, rel_all ? nullptr : &vo.map_p, rp_dist) : std::nullopt;
    std::optional<Pose> init;
    if (!vo.initialized() && !prm.use_depth && k > 0 &&
        (init_fixed > 0 ? k == init_fixed : median_shift(vo.obs[0], o, 50) >= init_px)) {
      init = robust_init(cam, vo.obs[0], o);
    }
    t_rel += ms_since(t0);
    t0 = Clock::now();
    const bool was_init = vo.initialized();
    vo.add(std::move(o), rel, init);
    if (init && !was_init)
      std::printf("  init at frame %d: %s\n", k,
                  vo.initialized() ? "ok" : ("rejected (" + vo.init_note + "), retry next frame").c_str());
    if (!keep_dynamic) {
      const std::vector<int> dyn = vo.inconsistent_tracks(3.0);
      front.drop(dyn), n_dynamic += dyn.size();
    }
    t_vo += ms_since(t0);
#ifdef VP_WITH_VIEWER
    if (viewer) {
      t0 = Clock::now();
      viewer->update(frame, vo);
      t_view += ms_since(t0);
      while (viewer->paused() && !viewer->closed()) std::this_thread::sleep_for(std::chrono::milliseconds(30));
      if (viewer->closed()) break;
    }
#endif
    if (k % 100 == 0)
      std::printf("  frame %4d: %zu map pts, %zu map lines  (%.1f fps)\n", k, vo.map_p.size(), vo.map_l.size(),
                  (k + 1) / (ms_since(t_start) / 1000));
  }
  const double total = ms_since(t_start);
  int f0 = 0, n_ok = 0;  // 포즈가 처음 생긴 프레임부터 연속으로 포즈가 있는 프레임 수
  while (f0 < k && !vo.has_pose[f0]) ++f0;
  while (f0 + n_ok < k && vo.has_pose[f0 + n_ok]) ++n_ok;
  if (f0) std::printf("  first pose at frame %d (trajectory and ATE start there)\n", f0);
  std::printf("%d/%d frames tracked%s | %.1f fps (%.1f ms/frame)\n", n_ok, k,
              vo.failed_at >= 0 ? (" | FAILED at " + std::to_string(vo.failed_at) + ": " + vo.fail_reason).c_str() : "",
              k / (total / 1000),
              total / k);
  std::printf("  keyframes %zu | stationary frames %d | low-parallax frames solved 6-DoF on map %d | mapping %s\n",
              vo.kfs.size(), vo.n_still, vo.n_map6dof,
              prm.kf_mapping ? "keyframes" : "every frame");
  std::printf("  map points re-attached after track loss: %ld (%.1f / frame) | dynamic tracks dropped: %ld (%.1f / frame)\n",
              n_reattach, double(n_reattach) / k, n_dynamic, double(n_dynamic) / k);
  if (prm.local_ba)
    std::printf("  local BA: %d runs, %.1f ms/run (%.1f ms/frame), %.0f points/run, %.1f outlier points removed/run, "
                "reprojection RMS %.2f -> %.2f px\n",
                vo.n_ba, vo.ba_ms / std::max(vo.n_ba, 1), vo.ba_ms / k, double(vo.ba_points) / std::max(vo.n_ba, 1),
                double(vo.ba_outliers) / std::max(vo.n_ba, 1), vo.ba_rms_before / std::max(vo.n_ba, 1),
                vo.ba_rms_after / std::max(vo.n_ba, 1));
  if (prm.use_depth)
    std::printf("  depth: %.0f points/frame with valid depth -> %ld map points from depth\n", double(n_depth_obs) / k,
                vo.n_depth_pts);
  std::printf("  per frame: read %.1f | frontend %.1f (LSD %.1f) | 2D-2D %.1f | tracker %.1f | viewer %.1f ms\n",
              t_read / k, t_front / k, front.lsd_ms / k, t_rel / k, t_vo / k, t_view / k);

  // ---- trajectory (KITTI format, T_wc) ----
  const char* traj = arg(argc, argv, "--traj", "cpp_traj.txt");
  std::ofstream ft(traj);
  ft.precision(9);
  std::vector<Vec3> C;
  for (int f = f0; f < f0 + n_ok; ++f) {
    const Mat3 Rwc = vo.poses[f].R.transpose();
    const Vec3 twc = center(vo.poses[f]);
    for (int r = 0; r < 3; ++r) ft << Rwc(r, 0) << ' ' << Rwc(r, 1) << ' ' << Rwc(r, 2) << ' ' << twc(r) << (r < 2 ? ' ' : '\n');
    C.push_back(twc);
  }
  std::printf("saved %s\n", traj);
  print_ate(C, std::vector<Eigen::Matrix<double, 3, 4>>(gt.begin() + std::min<size_t>(f0, gt.size()), gt.end()));
#ifdef VP_WITH_VIEWER
  if (viewer) viewer->finish(arg(argc, argv, "--snapshot", ""));
#endif
  return 0;
}
