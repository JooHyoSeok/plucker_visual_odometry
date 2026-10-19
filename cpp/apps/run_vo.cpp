// C++ monocular VO (points + Plücker lines): images in one by one -> pose out. No loop closure yet.
// usage: run_vo [n_frames] --video PATH|CAM_INDEX [--gt PATH] [--calib fx,fy,cx,cy] [--no-lines] [--traj OUT.txt]
//               [--no-viewer] [--snapshot OUT.png]
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
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
  const Eigen::Matrix4d S = Eigen::umeyama(X, Y, true);
  for (int f = 0; f < n; ++f) se_s += (S.topLeftCorner<3, 3>() * C[f] + S.topRightCorner<3, 1>() - G[f]).squaredNorm();
  std::printf("ATE (%.0f m): origin-anchored %.1f m | Sim(3)-aligned %.1f m\n", path, std::sqrt(se_a / n),
              std::sqrt(se_s / n));
}

int main(int argc, char** argv) {
  const char* src = arg(argc, argv, "--video", nullptr);
  if (!src) {
    std::fprintf(stderr,
                 "usage: %s [n_frames] --video PATH|CAM_INDEX [--gt PATH] [--calib fx,fy,cx,cy] [--no-lines] "
                 "[--traj OUT.txt] [--no-viewer] [--snapshot OUT.png]\n",
                 argv[0]);
    return 2;
  }
  const int n_max = argc > 1 && std::isdigit(argv[1][0]) ? std::atoi(argv[1]) : 1 << 30;
  double fx, fy, cx, cy;
  std::sscanf(arg(argc, argv, "--calib", "707.0912,707.0912,601.8873,183.1104"), "%lf,%lf,%lf,%lf", &fx, &fy, &cx, &cy);
  TrackerParams prm;
  prm.use_lines = !flag(argc, argv, "--no-lines");
  const char* gt_path = arg(argc, argv, "--gt", nullptr);
  const auto gt = gt_path ? load_kitti(gt_path) : std::vector<Eigen::Matrix<double, 3, 4>>{};

  cv::VideoCapture cap;
  std::isdigit(src[0]) && !std::strchr(src, '.') ? cap.open(std::atoi(src)) : cap.open(src);
  cv::Mat frame, gray;
  if (!cap.read(frame)) {
    std::fprintf(stderr, "cannot read video/images/camera: %s\n", src);
    return 1;
  }
  const Camera cam(fx, fy, cx, cy, frame.cols, frame.rows);
  Frontend front;
  VOTracker vo(cam, prm);
  std::printf("source %s, %dx%d, K=(%g,%g,%g,%g), lines %d\n", src, frame.cols, frame.rows, fx, fy, cx, cy,
              prm.use_lines);
#ifdef VP_WITH_VIEWER
  std::unique_ptr<Viewer> viewer;
  if (!flag(argc, argv, "--no-viewer")) {
    std::vector<Vec3> gc;
    Pose cw0;
    for (const auto& T : gt) gc.push_back(T.col(3));
    if (!gt.empty()) cw0 = Pose{gt[0].leftCols<3>().transpose(), -gt[0].leftCols<3>().transpose() * gt[0].col(3)};
    viewer = std::make_unique<Viewer>(cam, gc, cw0);
  }
#endif

  double t_read = 0, t_front = 0, t_rel = 0, t_vo = 0, t_view = 0;
  const auto t_start = Clock::now();
  int k = 0;
  for (; k < n_max; ++k) {
    if (k) {
      const auto t0 = Clock::now();
      if (!cap.read(frame)) break;
      t_read += ms_since(t0);
    }
    auto t0 = Clock::now();
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    FrameObs o = front.step(gray);
    t_front += ms_since(t0);
    t0 = Clock::now();
    const std::optional<Pose> rel = k ? relative_pose(cam, vo.obs.back(), o) : std::nullopt;
    std::optional<Pose> init;
    if (k == prm.init_frame) init = robust_init(cam, vo.obs[0], o);
    t_rel += ms_since(t0);
    t0 = Clock::now();
    vo.add(std::move(o), rel, init);
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
  int n_ok = 0;
  while (n_ok < k && vo.has_pose[n_ok]) ++n_ok;
  std::printf("%d/%d frames tracked%s | %.1f fps (%.1f ms/frame)\n", n_ok, k,
              vo.failed_at >= 0 ? (" | FAILED at " + std::to_string(vo.failed_at)).c_str() : "", k / (total / 1000),
              total / k);
  std::printf("  per frame: read %.1f | frontend %.1f (LSD %.1f) | 2D-2D %.1f | tracker %.1f | viewer %.1f ms\n",
              t_read / k, t_front / k, front.lsd_ms / k, t_rel / k, t_vo / k, t_view / k);

  // ---- trajectory (KITTI format, T_wc) ----
  const char* traj = arg(argc, argv, "--traj", "cpp_traj.txt");
  std::ofstream ft(traj);
  ft.precision(9);
  std::vector<Vec3> C;
  for (int f = 0; f < n_ok; ++f) {
    const Mat3 Rwc = vo.poses[f].R.transpose();
    const Vec3 twc = center(vo.poses[f]);
    for (int r = 0; r < 3; ++r) ft << Rwc(r, 0) << ' ' << Rwc(r, 1) << ' ' << Rwc(r, 2) << ' ' << twc(r) << (r < 2 ? ' ' : '\n');
    C.push_back(twc);
  }
  std::printf("saved %s\n", traj);
  print_ate(C, gt);
#ifdef VP_WITH_VIEWER
  if (viewer) viewer->finish(arg(argc, argv, "--snapshot", ""));
#endif
  return 0;
}
