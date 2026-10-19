// C++ monocular VO (points + Plücker lines): images in one by one -> pose out. No loop closure yet.
// usage: run_vo [n_frames] --video PATH|CAM_INDEX [--gt PATH] [--calib fx,fy,cx,cy] [--no-lines] [--traj OUT.txt]
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <string>

#include "vo_plucker/frontend.h"

using namespace vp;
using Clock = std::chrono::steady_clock;

static const char* arg(int argc, char** argv, const char* name, const char* def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], name)) return argv[i + 1];
  return def;
}

static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
  const char* src = arg(argc, argv, "--video", nullptr);
  if (!src) {
    std::fprintf(stderr,
                 "usage: %s [n_frames] --video PATH|CAM_INDEX [--gt PATH] [--calib fx,fy,cx,cy] [--no-lines] "
                 "[--traj OUT.txt]\n",
                 argv[0]);
    return 2;
  }
  const int n_max = argc > 1 && std::isdigit(argv[1][0]) ? std::atoi(argv[1]) : 1 << 30;
  double fx, fy, cx, cy;
  std::sscanf(arg(argc, argv, "--calib", "707.0912,707.0912,601.8873,183.1104"), "%lf,%lf,%lf,%lf", &fx, &fy, &cx, &cy);
  TrackerParams prm;
  for (int i = 1; i < argc; ++i) prm.use_lines &= std::strcmp(argv[i], "--no-lines") != 0;

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

  double t_read = 0, t_front = 0, t_rel = 0, t_vo = 0;
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
  std::printf("  per frame: read %.1f | frontend %.1f (LSD %.1f) | 2D-2D %.1f | tracker %.1f ms\n", t_read / k,
              t_front / k, front.lsd_ms / k, t_rel / k, t_vo / k);

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

  // ---- ATE: 원점 고정 정렬 (frame 0 카메라를 GT에 겹치고, 스케일만 처음 50프레임으로) ----
  if (const char* gt = arg(argc, argv, "--gt", nullptr); gt && n_ok > 1) {
    std::ifstream fg(gt);
    std::vector<Eigen::Matrix<double, 3, 4>> P;
    std::string line;
    while (std::getline(fg, line) && int(P.size()) < n_ok) {
      std::istringstream ss(line);
      Eigen::Matrix<double, 3, 4> T;
      for (int i = 0; i < 12; ++i) ss >> T(i / 4, i % 4);
      P.push_back(T);
    }
    const int n = std::min<int>(n_ok, P.size());
    const Mat3 R0 = P[0].leftCols<3>().transpose();
    const Vec3 t0 = -R0 * P[0].col(3);
    double num = 0, den = 0, se = 0, path = 0;
    for (int f = 0; f < std::min(n, 50); ++f) {
      const Vec3 Y = R0 * P[f].col(3) + t0;  // GT in cam0 coords
      num += C[f].dot(Y), den += C[f].squaredNorm();
    }
    const double s = num / std::max(den, 1e-12);
    for (int f = 0; f < n; ++f) {
      const Vec3 Ca = R0.transpose() * (s * C[f] - t0);
      se += (Ca - P[f].col(3)).squaredNorm();
      if (f) path += (P[f].col(3) - P[f - 1].col(3)).norm();
    }
    std::printf("ATE (origin-anchored, %.0f m): %.1f m\n", path, std::sqrt(se / n));
  }
  return 0;
}
