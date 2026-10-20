// C++ monocular VO (points + Plücker lines): images in one by one -> pose out. Optional loop closure + Sim(3) PGO (--loop, pluggable modules).
// usage: run_vo --video PATH|CAM_INDEX [--gt POSES.txt] [--calib fx,fy,cx,cy] [--traj OUT.txt] [--no-loop] [--no-da3]
//               [--no-viewer] [--autostart]
//   기본 = 지금 최고 설정 (PROGRESS.md 섹션 12): 루프 + DA3 앵커 (BA 깊이 prior, 단일 시점 라인, 루프 3D, 추적 복구) + local BA
//   스레드 + 맵 재사용 + 끝에 전역 BA. --no-loop / --no-da3 로만 끔. --calib 이 없으면 영상 폴더 옆 calib.txt (KITTI / EuRoC 변환본)
//   실험용: [N_FRAMES] --sync (루프 / BA 를 트래킹 안에서, 결정적) --da3-cache DIR (DA3 깊이맵 저장, 같은 결과 + GPU 없음)
//           --kf-out FILE --realtime HZ --start N --stride N --resize S --no-lines --ba-lines --traj-online FILE --dump-map FILE
//           --snapshot FILE --loop-pr thumb|netvlad|salad|mast3r --loop-ver orb|mast3r|orb+mast3r --loop-python PY
//           --da3-range MIN,MAX (기본: 첫 프레임 DA3 깊이로 실내 0.5,20 / 실외 3,40)
//           --da3-model metric|nested (기본 DA3METRIC-LARGE 0.35B: NESTED-GIANT-LARGE 1.40B 보다 3.3 배 빠르고 KITTI 가 더 정확)
//   --resize S : 영상과 intrinsic을 S배 (고해상도 입력용, 예: iPhone 1920x1440 -> 0.5)
//   --stride N : N 프레임마다 하나만 사용 (고 fps / 느린 이동용). GT도 같은 간격으로 맞춤
//   --start N  : 처음 N 프레임 건너뛰기 (예: 시작이 제자리 회전뿐인 핸드헬드 영상). GT도 N부터
//   --depth PATTERN [--depth-conf PATTERN --depth-conf-min C] [--depth-max M] [--depth-scale S]
//              : 프레임별 metric depth 영상 (예: iPhone LiDAR depth/%06d.png, uint16 mm -> scale 0.001).
//                영상 번호 = 원본 프레임 번호. 깊이 초기화 + 깊이 맵 점 + 맵 기준 6-DoF 추적
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <string>
#include <thread>

#include "vo_plucker/frontend.h"
#include "vo_plucker/local_mapping.h"
#include "vo_plucker/loop_closing.h"
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

// 인자 목록: 모르는 인자 (예전 실험 인자 포함) 는 에러 -> 조용히 기본값으로 도는 일이 없게
static const char* kValueOpts[] = {"--video", "--gt", "--calib", "--traj", "--da3-cache", "--kf-out", "--realtime", "--start",
                                   "--stride", "--resize", "--traj-online", "--dump-map", "--snapshot", "--loop-pr", "--loop-ver",
                                   "--loop-python", "--da3-range", "--da3-model", "--ba-line-prior", "--ba-vp", "--depth", "--depth-conf", "--depth-conf-min", "--depth-max",
                                   "--depth-scale"};
static const char* kFlagOpts[] = {"--no-loop", "--no-da3", "--no-viewer", "--autostart", "--sync", "--no-lines", "--ba-lines", "--ba-line-nowrite"};

static bool known_args(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    auto in = [&](const char* const* b, const char* const* e) { return std::any_of(b, e, [&](const char* o) { return !std::strcmp(argv[i], o); }); };
    if (in(std::begin(kValueOpts), std::end(kValueOpts)) && i + 1 < argc) {
      ++i;
    } else if (!in(std::begin(kFlagOpts), std::end(kFlagOpts)) && !(i == 1 && std::isdigit(argv[1][0]))) {
      std::fprintf(stderr, "unknown option %s\n", argv[i]);
      return false;
    }
  }
  return true;
}

// --calib 이 없을 때: 영상 폴더의 부모 (KITTI sequences/NN, EuRoC 변환본 SEQ/kitti) 또는 같은 폴더의 calib.txt.
// KITTI odometry "P0: fx 0 cx 0 0 fy cy 0 ..." (첫 줄, P0-P3 의 fx fy cx cy 같음) 또는 "fx fy cx cy"
static bool calib_file(const std::string& video, double& fx, double& fy, double& cx, double& cy) {
  const size_t a = video.find_last_of('/');
  std::vector<std::string> dirs;
  if (a != std::string::npos && a > 0) {
    const size_t b = video.find_last_of('/', a - 1);
    if (b != std::string::npos) dirs.push_back(video.substr(0, b));
    dirs.push_back(video.substr(0, a));
  }
  for (const std::string& d : dirs) {
    std::ifstream f(d + "/calib.txt");
    std::string first;
    if (!(f >> first)) continue;
    std::vector<double> v;
    if (!std::isalpha(first[0])) v.push_back(std::atof(first.c_str()));
    for (double x; v.size() < 12 && f >> x;) v.push_back(x);
    if (std::isalpha(first[0]) && v.size() >= 7) return fx = v[0], cx = v[2], fy = v[5], cy = v[6], true;
    if (!std::isalpha(first[0]) && v.size() >= 4) return fx = v[0], fy = v[1], cx = v[2], cy = v[3], true;
  }
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
static void print_ate(const std::vector<Vec3>& C, const std::vector<Eigen::Matrix<double, 3, 4>>& P, const char* label = "ATE") {
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
  std::printf("%s (%.1f m): origin-anchored %.2f m | Sim(3)-aligned %.2f m (scale %.3f) | SE(3)-aligned, no scale fit %.2f m\n",
              label, path, std::sqrt(se_a / n), std::sqrt(se_s / n), std::cbrt(S.topLeftCorner<3, 3>().determinant()),
              std::sqrt(se_e / n));
}

int main(int argc, char** argv) {
  const char* src = arg(argc, argv, "--video", nullptr);
  if (!src || !known_args(argc, argv)) {
    std::fprintf(stderr,
                 "usage: %s --video PATH|CAM_INDEX [--gt POSES.txt] [--calib fx,fy,cx,cy] [--traj OUT.txt] [--no-loop] [--no-da3] "
                 "[--no-viewer] [--autostart]\n  (--calib default: calib.txt next to the image folder; experiment options: see run_vo.cpp)\n",
                 argv[0]);
    return 2;
  }
  const int n_max = argc > 1 && std::isdigit(argv[1][0]) ? std::atoi(argv[1]) : 1 << 30;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  if (const char* c = arg(argc, argv, "--calib", nullptr)) {
    std::sscanf(c, "%lf,%lf,%lf,%lf", &fx, &fy, &cx, &cy);
  } else if (!calib_file(src, fx, fy, cx, cy)) {
    std::fprintf(stderr, "no --calib and no calib.txt next to %s\n", src);
    return 2;
  }
  // 기본 = 실험으로 정한 최고 설정 (PROGRESS.md 섹션 10-12). 켜고 끄는 건 루프와 DA3 앵커만
  const bool sync = flag(argc, argv, "--sync");  // 루프 / local BA 를 트래킹 안에서 (결정적, 비교 실험용)
  bool use_loop = !flag(argc, argv, "--no-loop");
  // DA3 metric depth: 스케일 앵커 (주기적 Sim(3) PGO) + local BA 점 깊이 prior + 단일 시점 Plücker 라인 + 루프 검증 3D + 추적 복구
  bool use_anchor = !flag(argc, argv, "--no-da3");
  TrackerParams prm;
  prm.use_lines = !flag(argc, argv, "--no-lines");
  prm.kf_parallax_px = 25;  // 키프레임: 회전 보정 parallax 중앙값 [px] (10 -> 25: 키프레임 57 % 적고 같은 정확도)
  prm.ba_drop_obs = true;   // local BA 이상치는 관측만 빼고 점은 유지 (점을 통째로 지우면 PGO 보정 뒤 창이 비어 EuRoC 추적 실패)
  prm.ba_lines = flag(argc, argv, "--ba-lines");  // 실험: local / 전역 BA 에 Plücker 라인 (orthonormal 4-DoF)
  prm.ba_line_depth_sigma = std::atof(arg(argc, argv, "--ba-line-prior", "0"));  // 실험: 라인 끝점 DA3 log-depth prior σ
  prm.ba_line_nowrite = flag(argc, argv, "--ba-line-nowrite");  // 실험: BA 라인을 맵에 안 써넣음 (포즈 제약만)
  prm.ba_vp_sigma = std::atof(arg(argc, argv, "--ba-vp", "0"));  // 실험: local BA 구조 방향 (소실점) 제약 σ [px]
  const char* depth_pat = arg(argc, argv, "--depth", nullptr);  // iPhone LiDAR 같은 프레임별 metric depth 영상 (RGB-D 모드)
  const char* conf_pat = arg(argc, argv, "--depth-conf", nullptr);
  const int conf_min = std::atoi(arg(argc, argv, "--depth-conf-min", "2"));
  const double depth_max = std::atof(arg(argc, argv, "--depth-max", "10"));
  const double depth_scale = std::atof(arg(argc, argv, "--depth-scale", "0.001"));
  prm.use_depth = depth_pat != nullptr;
  // local BA: 별도 스레드 (LocalMapper, ORB-SLAM3 LocalMapping), --sync 면 트래킹 안에서
  const bool ba_thread = !sync;
  prm.local_ba = sync;
  std::string loop_pr = arg(argc, argv, "--loop-pr", "thumb"), loop_ver = arg(argc, argv, "--loop-ver", "orb");
  if ((loop_pr != "thumb" && loop_pr != "netvlad" && loop_pr != "salad" && loop_pr != "mast3r") || (loop_ver != "orb" && loop_ver != "mast3r" && loop_ver != "orb+mast3r")) {
    std::fprintf(stderr, "--loop-pr: thumb|netvlad|salad|mast3r, --loop-ver: orb|mast3r|orb+mast3r\n");
    return 1;
  }
  LoopParams lprm;
  lprm.threaded = !sync;
  lprm.skip_chi2 = 50;         // 새 루프 엣지가 이미 현재 추정과 맞으면 (chi2 < 50) 엣지만 추가, PGO 생략
  lprm.anchor_min_dev = 0.05;  // 최근 앵커 스케일 편차가 이보다 작으면 앵커 PGO 생략
  lprm.odom_refresh = 20;      // PGO 오도메트리: 최근 20 쌍은 지금 (local BA 뒤) 포즈로
  lprm.odom_sigma_s = 0.005;   // 이웃 키프레임 상대 스케일을 단단하게 (앵커 잡음이 창 스케일을 흔들지 못하게)
  // DA3 깊이 범위 [m]: --da3-range 가 없으면 첫 프레임 DA3 깊이 중앙값으로 실내 0.5-20 / 실외 3-40 (KITTI 19-38 m, EuRoC 1.6-3.1 m)
  const char* da3_range = arg(argc, argv, "--da3-range", nullptr);
  double zmin = 3, zmax = 40;
  if (da3_range) std::sscanf(da3_range, "%lf,%lf", &zmin, &zmax);
  const double init_px = 20;  // frame 0 대비 픽셀 이동 중앙값이 이 이상이면 초기화 시도, 품질이 안 되면 다음 프레임 재시도
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
#ifdef VP_WITH_VIEWER
  // 뷰어가 있으면 시작 화면에서 루프 클로저 / 검출기 / 검증기 / 라인을 고르고 Start (기본값 = 인자)
  std::unique_ptr<Viewer> viewer;
  if (!flag(argc, argv, "--no-viewer")) {
    std::vector<Pose> gt_cw;
    for (const auto& T : gt) gt_cw.push_back(Pose{T.leftCols<3>().transpose(), -T.leftCols<3>().transpose() * T.col(3)});
    const bool ask = !flag(argc, argv, "--autostart");  // --autostart: 시작 화면 없이 인자대로 바로
    viewer = std::make_unique<Viewer>(cam, gt_cw, !gt.empty(),
                                      ask ? std::optional<RunOptions>(RunOptions{use_loop, loop_pr, loop_ver, prm.use_lines, use_anchor})
                                          : std::nullopt);
    if (ask) {
      std::printf("viewer: choose options in the menu and press Start\n");
      const auto o = viewer->wait_start();
      if (!o) return 0;  // Start 전에 창을 닫음
      use_loop = o->loop, loop_pr = o->pr, loop_ver = o->ver, prm.use_lines = o->lines, use_anchor = o->depth_prior;
    }
  }
#endif
  prm.keyframes = use_loop || use_anchor || ba_thread;
  prm.ba_depth_sigma = use_anchor ? 0.3 : 0;  // local BA 점 깊이 prior (log 깊이 σ, 앵커 깊이)
  prm.da3_lines = use_anchor;                 // 키프레임 선분을 DA3 깊이로 바로 Plücker 라인 (진행 방향 라인 포함)
  prm.recover = use_anchor;                   // 추적 실패 다음 프레임: 외삽 포즈 + DA3 깊이 트랙 점으로 같은 좌표계 / 스케일에서 계속
  prm.depth_init = use_anchor;                // 2-view 초기화가 계속 안 되면 한 프레임을 DA3 깊이로 초기화 (아래 프레임 루프)
  lprm.shortlist = std::atoi(arg(argc, argv, "--loop-shortlist", loop_ver == "orb" ? "15" : "3"));  // MASt3R는 쌍당 비쌈
  // 후보 유사도 하한: netvlad/salad 코사인 0.5, mast3r ASMK 점수 0.02 (06: 참 루프 0.06-0.11, 그 외 <= 0.018)
  const double loop_min_sim = std::atof(arg(argc, argv, "--loop-min-sim", loop_pr == "mast3r" ? "0.02" : "0.5"));
  FrontendParams fprm;
  fprm.reuse_every = 5;  // 맵 재사용: 5 프레임마다 ORB 투영 검색으로 옛 맵 점을 다시 붙임
  Frontend front(fprm);
  VOTracker vo(cam, prm);
  std::unique_ptr<LoopCloser> loop;
  std::shared_ptr<LoopServer> srv;  // Python 플러그인 서버 (루프 모듈 / DA3)
  {
    const bool py_pr = use_loop && loop_pr != "thumb", py_ver = use_loop && loop_ver != "orb";
    setenv("HF_HUB_OFFLINE", "1", 0);  // 모델은 로컬에 있음: 허브 접속 안 함
    if (py_pr || py_ver || use_anchor)
      srv = start_loop_server(std::string(arg(argc, argv, "--loop-python", VP_PYTHON)) + " " VP_LOOP_SERVER " --pr " +
                              (py_pr ? loop_pr : "none") + " --ver " + (py_ver ? "mast3r" : "none") +
                              (use_anchor ? " --depth da3 --depth-range " + std::to_string(zmin) + "," + std::to_string(zmax) +
                                                " --depth-model " + arg(argc, argv, "--da3-model", "metric") +
                                                " --depth-focal " + std::to_string((cam.fx + cam.fy) / 2) +
                                                (arg(argc, argv, "--da3-cache", nullptr) ? std::string(" --depth-cache ") + arg(argc, argv, "--da3-cache", "") : "")
                                          : ""));
  }
  if (use_loop || use_anchor) {
    const bool py_pr = use_loop && loop_pr != "thumb";
    std::unique_ptr<PlaceRecognizer> pr;
    std::unique_ptr<LoopVerifier> ver;
    if (use_loop) {
      pr = py_pr ? make_py_recognizer(srv, loop_min_sim) : make_thumbnail_recognizer();
      // --loop-da3-3d: ORB 검증에서 로컬 3D 로 Sim(3) 가 안 되면 DA3 깊이 3D 로 (--depth-prior da3 필요)
      // ORB 검증: F-RANSAC 인라이어 40 이상, 로컬 3D 로 Sim(3) 가 안 되면 DA3 깊이 3D 로 (재방문 첫 연결), 양방향 재투영 정제
      ver = loop_ver == "orb"      ? make_orb_verifier(cam, 40, use_anchor ? srv : nullptr, true)
            : loop_ver == "mast3r" ? make_py_verifier(srv, cam)
                                   : make_cascade_verifier(cam, make_py_verifier(srv, cam));
    }
    loop = std::make_unique<LoopCloser>(vo, std::move(pr), std::move(ver), lprm, use_anchor ? srv : nullptr);
  }
  // --traj-online: 프레임마다 그 순간의 추정 포즈 (나중 루프 보정 전 값) "k T_wc(3x4)" -> 영상 도구용
  //   + 루프 보정이 들어올 때마다 그 시점 전체 궤적 스냅샷 "k f x y z" (위치만) -> OUT_snap.txt
  std::ofstream fon, fsnap;
  if (const char* p = arg(argc, argv, "--traj-online", nullptr)) {
    fon.open(p), fon.precision(9);
    std::string sp(p);
    sp = (sp.size() > 4 && sp.substr(sp.size() - 4) == ".txt" ? sp.substr(0, sp.size() - 4) : sp) + "_snap.txt";
    fsnap.open(sp), fsnap.precision(7);
  }
  int snap_seen = 0;
  // --dump-map (진단): 프레임마다 관측 "o k pid u v" + 그 프레임에 보이는 맵 점 "m k pid X Y Z mature" (world, 매핑 후)
  std::ofstream fdump;
  if (const char* p = arg(argc, argv, "--dump-map", nullptr)) fdump.open(p), fdump.precision(7);
  std::unique_ptr<LocalMapper> mapper;  // loop 보다 뒤에 선언 -> 먼저 소멸 (둘 다 vo 보다 먼저)
  if (ba_thread) mapper = std::make_unique<LocalMapper>(vo, prm.ba_window, prm.ba_fixed, prm.ba_lines && prm.use_lines,
                                                         prm.ba_mature_only);
  std::map<int, cv::Mat> recent_img;  // 키프레임 등록용 최근 영상 (BGR, 초기화 때 키프레임이 몇 프레임 늦게 정해짐)
  size_t kf_done = 0;
  double t_loop = 0;
  std::printf("source %s, %dx%d, K=(%g,%g,%g,%g), stride %d, lines %d, loop %s\n", src, frame.cols, frame.rows, cam.fx,
              cam.fy, cam.cx, cam.cy, stride, prm.use_lines,
              use_loop ? (loop_pr + " + " + loop_ver + (lprm.threaded ? ", thread" : ", sync")).c_str() : "off");
  if (use_anchor) std::printf("depth prior: DA3 metric depth scale anchor every %d keyframes (sigma %.2f)\n", lprm.anchor_every, lprm.anchor_sigma);

  long n_reattach = 0, n_dynamic = 0, n_depth_obs = 0, n_reuse = 0;
  double t_read = 0, t_front = 0, t_rel = 0, t_vo = 0, t_view = 0, t_wait = 0, t_wait_max = 0;
  // --realtime HZ: 프레임을 카메라 속도로 넣음 (KITTI 10, EuRoC 20) -> 스레드 모드를 실제 센서 속도에서 평가
  const double realtime_hz = std::atof(arg(argc, argv, "--realtime", "0"));
  const auto t_start = Clock::now();
  int k = 0, n_init_rejected = 0;
  for (; k < n_max; ++k) {
    if (realtime_hz > 0)
      std::this_thread::sleep_until(t_start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(k / realtime_hz)));
    if (k) {
      const auto t0 = Clock::now();
      if (!read()) break;
      t_read += ms_since(t0);
    }
    auto t0 = Clock::now();
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    FrameObs o = front.step(gray);
    if (k == 0 && srv && use_anchor) {
      // DA3 모델을 첫 프레임에 미리 올림 (스레드 모드: 첫 앵커 때 로딩 20-30 s 동안 루프 스레드가 밀려 첫 구간 키프레임이 DA3 3D 없이
      // 남음 -> KITTI 00 끝 루프를 검증 못 함) + 깊이 범위 자동: 가운데 7x7 점 DA3 깊이 중앙값 < 8 m 면 실내
      std::vector<Vec2> g;
      for (int y = 1; y < 8; ++y)
        for (int x = 1; x < 8; ++x) g.emplace_back(frame.cols * (0.125 + 0.75 * x / 8), frame.rows * (0.25 + 0.5 * y / 8));
      std::vector<double> z = server_depth_at(*srv, frame, g);
      std::nth_element(z.begin(), z.begin() + z.size() / 2, z.end());
      if (!da3_range) {
        const bool indoor = !z.empty() && z[z.size() / 2] < 8;
        zmin = indoor ? 0.5 : 3, zmax = indoor ? 20 : 40;
        server_depth_range(*srv, zmin, zmax);
      }
      std::printf("  DA3 depth range %.1f-%.0f m (frame 0 median depth %.1f m)\n", zmin, zmax, z.empty() ? 0.0 : z[z.size() / 2]);
    }
    int common0 = 0;  // 초기화 기준 프레임 0 과 공통 트랙 수 (초기화 전에는 다른 스레드가 vo 를 안 씀)
    if (!vo.initialized() && k > 0)
      for (int j : o.pid) common0 += vo.obs[0].point(j) != nullptr;
    const bool da3_init_now = srv && use_anchor && !vo.initialized() && k > 0 && (n_init_rejected >= 15 || (k >= 30 && common0 < 50));
    // 2-view 초기화가 안 됨 (먼 장면 + 느린 핸드헬드로 시차 부족 15 번, 또는 프레임 0 트랙이 거의 사라짐) -> 이 프레임 하나를 DA3
    // metric 깊이로 초기화 (처음부터 미터 단위, 시차가 생기면 삼각측량으로 교체). KITTI / EuRoC 는 거부 0-4 번이라 해당 없음
    if (da3_init_now || (srv && vo.wants_depth())) {  // + 추적 실패 다음 프레임: DA3 metric 깊이 (범위 안) 를 트랙 점에
      const std::vector<double> z = server_depth_at(*srv, frame, o.uv);
      o.depth.assign(o.pid.size(), std::numeric_limits<float>::quiet_NaN());
      for (size_t i = 0; i < z.size() && i < o.depth.size(); ++i)
        if (z[i] > zmin && z[i] < zmax) o.depth[i] = float(z[i]);
    }
    // 여기부터 vo 를 읽고 씀 -> map lock (루프 스레드가 보정을 써넣는 중이면 그동안 대기)
    const auto tw = Clock::now();
    const bool threads = (loop && lprm.threaded) || mapper;  // 루프 / 매핑 스레드가 vo 를 같이 씀
    auto map_lock = threads ? vo.map_lock.tracking() : std::unique_lock<std::mutex>();
    const double wait = ms_since(tw);
    t_wait += wait, t_wait_max = std::max(t_wait_max, wait);
    if (const auto pred = vo.predict()) {
      n_reattach += front.reattach(cam, vo, *pred, o);
      n_reuse += front.reuse(cam, vo, *pred, o);
    }
    if (depth_pat) {
      const int fi = start + k * stride;  // 원본 프레임 번호
      const cv::Mat d = cv::imread(cv::format(depth_pat, fi), cv::IMREAD_UNCHANGED);
      const cv::Mat c = conf_pat ? cv::imread(cv::format(conf_pat, fi), cv::IMREAD_UNCHANGED) : cv::Mat();
      n_depth_obs += sample_depth(o, d, c, conf_min, depth_max, depth_scale, cam.W, cam.H);
    }
    t_front += ms_since(t0);
    t0 = Clock::now();
    std::optional<Pose> rel = k ? relative_pose(cam, vo.obs.back(), o, nullptr, 1e6) : std::nullopt;
    std::optional<Pose> init;
    if (!vo.initialized() && !prm.use_depth && k > 0 && median_shift(vo.obs[0], o, 50) >= init_px) {
      init = robust_init(cam, vo.obs[0], o);
    }
    t_rel += ms_since(t0);
    t0 = Clock::now();
    const bool was_init = vo.initialized();
    if (loop) {
      recent_img[k] = frame.clone();
      while (recent_img.size() > 40) recent_img.erase(recent_img.begin());
    }
    vo.add(std::move(o), rel, init);
    if (fdump.is_open() && vo.has_pose[k]) {
      const FrameObs& ob = vo.obs[k];
      for (size_t i = 0; i < ob.pid.size(); ++i) {
        fdump << "o " << k << " " << ob.pid[i] << " " << ob.uv[i](0) << " " << ob.uv[i](1) << "\n";
        if (const auto it = vo.map_p.find(ob.pid[i]); it != vo.map_p.end())
          fdump << "m " << k << " " << ob.pid[i] << " " << it->second(0) << " " << it->second(1) << " " << it->second(2)
                << " " << vo.mature_p.count(ob.pid[i]) << "\n";
      }
    }
    if (fon.is_open() && vo.has_pose[k]) {
      const Pose& p = vo.poses[k];
      const Mat3 Rwc = p.R.transpose();
      const Vec3 C = -Rwc * p.t;
      fon << k;
      for (int r = 0; r < 3; ++r) fon << " " << Rwc(r, 0) << " " << Rwc(r, 1) << " " << Rwc(r, 2) << " " << C(r);
      fon << "\n";
    }
    std::vector<std::pair<int, cv::Mat>> new_kfs;  // 프레임 처리 끝 (map lock 해제 뒤) 에 루프 / 매핑 스레드로
    if (loop || mapper) {
      if (vo.kfs.size() < kf_done) kf_done = 0;  // 초기화 재시도로 키프레임 목록이 비워짐
      for (; kf_done < vo.kfs.size(); ++kf_done)
        if (vo.failed_at < 0) {
          const auto it = recent_img.find(vo.kfs[kf_done]);
          new_kfs.emplace_back(vo.kfs[kf_done], it != recent_img.end() ? it->second : cv::Mat());
        }
    }
    if (loop) {
      if (fsnap.is_open() && loop->n_corrections != snap_seen) {  // map lock 아래라 보정 중간 상태는 안 보임
        snap_seen = loop->n_corrections;
        for (int f = 0; f <= k; ++f)
          if (vo.has_pose[f]) {
            const Vec3 C = center(vo.poses[f]);
            fsnap << k << " " << f << " " << C(0) << " " << C(1) << " " << C(2) << "\n";
          }
      }
    }
    if (init && !was_init && !da3_init_now) {
      n_init_rejected += !vo.initialized();
      std::printf("  init at frame %d: %s\n", k,
                  vo.initialized() ? "ok" : ("rejected (" + vo.init_note + "), retry next frame").c_str());
    }
    if (da3_init_now)
      std::printf("  init at frame %d from DA3 depth: %s\n", k, vo.initialized() ? "ok" : "not enough tracks with depth");
    t_vo += ms_since(t0);
#ifdef VP_WITH_VIEWER
    if (viewer) {
      t0 = Clock::now();
      viewer->update(frame, vo);
      t_view += ms_since(t0);
      if (map_lock) map_lock.unlock();
      while (viewer->paused() && !viewer->closed()) std::this_thread::sleep_for(std::chrono::milliseconds(30));
      if (viewer->closed()) break;
    }
#endif
    if (map_lock) map_lock.unlock();
    {  // lock 을 놓은 뒤: 동기식 루프 (process 가 lock 을 잡음) 와 매핑 스레드를 같이 써도 교착 없음
      const auto tl = Clock::now();
      for (auto& [kf, img] : new_kfs) {
        if (loop && !img.empty()) loop->push(kf, img);
        if (mapper) mapper->push(kf);
      }
      t_loop += ms_since(tl);
    }
    if (k % 100 == 0)
      std::printf("  frame %4d: %zu map pts, %zu map lines  (%.1f fps)\n", k, vo.map_p.size(), vo.map_l.size(),
                  (k + 1) / (ms_since(t_start) / 1000));
  }
  const double total = ms_since(t_start);
  if (mapper) mapper->finish();
  if (loop) loop->finish();  // 남은 키프레임 처리 + 마지막 보정까지 반영한 뒤 궤적 저장
  int f0 = 0, n_ok = 0;  // 포즈가 처음 생긴 프레임부터 연속으로 포즈가 있는 프레임 수
  while (f0 < k && !vo.has_pose[f0]) ++f0;
  while (f0 + n_ok < k && vo.has_pose[f0 + n_ok]) ++n_ok;
  if (f0) std::printf("  first pose at frame %d (trajectory and ATE start there)\n", f0);
  std::printf("%d/%d frames tracked%s | %.1f fps (%.1f ms/frame)\n", n_ok, k,
              vo.failed_at >= 0 ? (" | FAILED at " + std::to_string(vo.failed_at) + ": " + vo.fail_reason).c_str() : "",
              k / (total / 1000),
              total / k);
  std::printf("  keyframes %zu | stationary frames %d | low-parallax frames solved 6-DoF on map %d | "
              "2D-2D rejected -> 6-DoF on map %d | 6-DoF on mature landmarks %d | mapping %s\n",
              vo.kfs.size(), vo.n_still, vo.n_map6dof, vo.n_rescued, vo.n_track_mature,
              prm.kf_mapping ? "keyframes" : "every frame");
  if (prm.recover) std::printf("  tracking failures recovered with DA3 depth: %d\n", vo.n_recovered);
  std::printf("  map points re-attached after track loss: %ld (%.1f / frame) | re-used by ORB projection search: %ld | "
              "dynamic tracks dropped: %ld (%.1f / frame)\n", n_reattach, double(n_reattach) / k, n_reuse, n_dynamic,
              double(n_dynamic) / k);
  if (loop) {
    std::printf("  loop closure (%s): %zu loops | %d queries, detect %.1f ms/query, PGO + correction %.0f ms total | "
                "%d keyframes dropped, %d without DA3 (loop thread behind)\n",
                lprm.threaded ? "thread" : "sync", loop->loops.size(), loop->n_queries,
                loop->ms_detect / std::max(loop->n_queries, 1), loop->ms_pgo, loop->n_skipped, loop->n_depth_skipped);
    if (lprm.da3_adaptive)
      std::printf("  DA3 noise (mature points): per-frame scale %.3f, per-point %.3f (log) -> anchor sigma %.3f, BA prior sigma %.3f\n",
                  loop->da3_noise_frame(), loop->da3_noise_point(), std::max(lprm.anchor_sigma, 1.3 * loop->da3_noise_frame()),
                  vo.da3_sigma > 0 ? vo.da3_sigma : prm.ba_depth_sigma);
    if (prm.da3_points) std::printf("  DA3 map points (pseudo RGB-D): %ld\n", loop->n_da3_points);
    if (prm.da3_lines) std::printf("  DA3 map lines (single-view Plücker from DA3 depth): %ld\n", loop->n_da3_lines);
    if (loop->n_rejected_pgo) std::printf("  loops rejected after PGO (edge still inconsistent): %d\n", loop->n_rejected_pgo);
    if (loop->n_bad_scale) std::printf("  loop candidates rejected for Sim(3) scale ratio outside 1/3..3: %d\n", loop->n_bad_scale);
    if (use_anchor)
      std::printf("  depth prior: %d keyframes anchored, %d anchor PGO runs, %d skipped (scale already consistent)\n",
                  loop->n_anchors, loop->n_anchor_pgo, loop->n_anchor_skip);
    if (lprm.fuse)
      std::printf("  loop fusion: %d map points fused with the old map, %d loop candidates excluded (already covisible)\n",
                  loop->n_fused, loop->n_excluded);
    if (lprm.consist_n > 1 || lprm.prune_ratio > 0)
      std::printf("  loop robustness: %d detections not confirmed (consistency %d), %d loop edges pruned after PGO\n",
                  loop->n_unconfirmed, lprm.consist_n, loop->n_pruned);
    const int n_pgo = int(loop->loops.size()) - loop->n_pgo_skipped + loop->n_anchor_pgo;
    std::printf("  PGO: %d runs (%d loops already consistent -> skipped) | g2o optimize %.0f ms total (%.1f ms/run), "
                "map write-back %.0f ms total\n", n_pgo, loop->n_pgo_skipped, loop->ms_opt,
                loop->ms_opt / std::max(n_pgo, 1), loop->ms_blocked);
    std::printf("  tracking stalled by loop closure: map lock held by loop thread %.0f ms total (max %.1f ms), "
                "tracking waited %.1f ms/frame (max %.1f ms), in-thread loop work %.1f ms/frame\n",
                loop->ms_blocked, loop->ms_blocked_max, t_wait / k, t_wait_max, t_loop / k);
  }
  if (prm.local_ba)
    std::printf("  local BA: %d runs, %.1f ms/run (%.1f ms/frame), %.0f points/run, %.1f outlier points removed/run, "
                "reprojection RMS %.2f -> %.2f px\n",
                vo.n_ba, vo.ba_ms / std::max(vo.n_ba, 1), vo.ba_ms / k, double(vo.ba_points) / std::max(vo.n_ba, 1),
                double(vo.ba_outliers) / std::max(vo.n_ba, 1), vo.ba_rms_before / std::max(vo.n_ba, 1),
                vo.ba_rms_after / std::max(vo.n_ba, 1));
  if (mapper)
    std::printf("  local BA (thread): %d runs (%d discarded after a map correction, %d queued keyframes merged), "
                "solve %.1f ms/run, %.0f points/run, %.1f outliers/run, %.0f lines/run, RMS %.2f -> %.2f px | "
                "map lock held %.0f ms total (max %.1f ms)\n",
                mapper->n_ba, mapper->n_discarded, mapper->n_dropped, mapper->ms_solve / std::max(mapper->n_ba, 1),
                double(mapper->points) / std::max(mapper->n_ba, 1), double(mapper->outliers) / std::max(mapper->n_ba, 1),
                double(mapper->lines_used) / std::max(mapper->n_ba, 1), mapper->rms_before / std::max(mapper->n_ba, 1),
                mapper->rms_after / std::max(mapper->n_ba, 1), mapper->ms_blocked, mapper->ms_blocked_max);
  if (prm.local_ba && prm.ba_lines)
    std::printf("  local BA lines: %.0f lines/run, %.1f outlier lines removed/run\n",
                double(vo.ba_lines_used) / std::max(vo.n_ba, 1), double(vo.ba_line_outliers) / std::max(vo.n_ba, 1));
  if (prm.ba_vp_sigma > 0)
    std::printf("  local BA structural directions: %.0f segment constraints/run, %zu directions (sigma %.1f px)\n",
                double(mapper ? mapper->vp_edges : vo.ba_vp_edges) / std::max(mapper ? mapper->n_ba : vo.n_ba, 1),
                vo.dirs.size(), prm.ba_vp_sigma);
  if (prm.ba_depth_sigma > 0)
    std::printf("  local BA DA3 depth priors: %.0f /run (sigma %.2f in log depth)\n",
                double(mapper ? mapper->depth_priors : vo.ba_depth_priors) / std::max(mapper ? mapper->n_ba : vo.n_ba, 1),
                prm.ba_depth_sigma);
  if (prm.use_depth)
    std::printf("  depth: %.0f points/frame with valid depth -> %ld map points from depth\n", double(n_depth_obs) / k,
                vo.n_depth_pts);
  std::printf("  per frame: read %.1f | frontend %.1f (LSD %.1f) | 2D-2D %.1f | tracker %.1f | viewer %.1f ms\n",
              t_read / k, t_front / k, front.lsd_ms / k, t_rel / k, t_vo / k, t_view / k);

  // ---- trajectory (KITTI format, T_wc) ----
  const char* traj = arg(argc, argv, "--traj", "cpp_traj.txt");
  auto save_traj = [&](const std::string& path) {
    std::ofstream ft(path);
    ft.precision(9);
    std::vector<Vec3> C;
    for (int f = f0; f < f0 + n_ok; ++f) {
      const Mat3 Rwc = vo.poses[f].R.transpose();
      const Vec3 twc = center(vo.poses[f]);
      for (int r = 0; r < 3; ++r) ft << Rwc(r, 0) << ' ' << Rwc(r, 1) << ' ' << Rwc(r, 2) << ' ' << twc(r) << (r < 2 ? ' ' : '\n');
      C.push_back(twc);
    }
    std::printf("saved %s\n", path.c_str());
    return C;
  };
  // 시퀀스 끝에 전역 BA 한 번 (ORB-SLAM "+ Global BA" 처럼, 루프가 있을 때). 같은 실행의 BA 전 궤적은 *_pregba.txt
  if (use_loop && loop) {
    std::string pre(traj);
    pre = (pre.size() > 4 && pre.substr(pre.size() - 4) == ".txt" ? pre.substr(0, pre.size() - 4) : pre) + "_pregba.txt";
    const std::vector<Vec3> Cpre = save_traj(pre);
    std::vector<Vec3> Kpre;  // 키프레임 위치: 전역 BA 가 궤적을 얼마나 옮겼나
    for (int f : vo.kfs) Kpre.push_back(center(vo.poses[f]));
#ifdef VP_WITH_VIEWER
    if (viewer) viewer->update(frame, vo), viewer->global_ba_start();  // 전역 BA 전 ATE 를 뷰어 칸에 고정
#endif
    std::vector<LoopPose> lp;
    for (const LoopEdge& e : loop->loop_edges()) lp.push_back({e.i, e.j, Pose{e.S.R, e.S.t}});
    const LocalBAStats s = global_ba(vo, lp, prm.ba_lines && prm.use_lines);
    std::printf("  global BA: %d keyframes (%d fixed), %d points, %zu loop edges, %.0f ms, RMS %.2f -> %.2f px, "
                "%d outlier points removed, %d DA3 priors\n", s.kfs, s.fixed, s.points, lp.size(), s.ms, s.rms_before,
                s.rms_after, s.outliers, s.depth_priors);
    std::vector<double> mv;
    for (size_t i = 0; i < Kpre.size(); ++i) mv.push_back((center(vo.poses[vo.kfs[i]]) - Kpre[i]).norm());
    if (!mv.empty()) {  // VO 단위 -> 미터 (DA3 기준 VO 단위 / 미터가 있으면)
      std::sort(mv.begin(), mv.end());
      const double u = vo.da3_ref > 0 ? 1 / vo.da3_ref : 1;
      std::printf("  global BA moved keyframes: median %.2f, max %.2f %s\n", mv[mv.size() / 2] * u, mv.back() * u,
                  vo.da3_ref > 0 ? "m (DA3 scale)" : "VO units");
    }
    print_ate(Cpre, std::vector<Eigen::Matrix<double, 3, 4>>(gt.begin() + std::min<size_t>(f0, gt.size()), gt.end()),
              "ATE before global BA");
#ifdef VP_WITH_VIEWER
    if (viewer) viewer->global_ba_done(s.ms);
#endif
  }
  const std::vector<Vec3> C = save_traj(traj);
  // --kf-out FILE: 키프레임마다 "frame a_k R_cw(9) t_cw(3)" (덴스 맵용). a_k = 최종 맵에서 다시 잰 VO 단위 / DA3 미터
  //   (그 키프레임 앵커 점들의 VO 깊이 / DA3 깊이 중앙값, 없으면 da3_ref)
  if (const char* kp = arg(argc, argv, "--kf-out", nullptr)) {
    std::ofstream fk(kp);
    fk.precision(9);
    for (int f : vo.kfs) {
      if (!vo.has_pose[f]) continue;
      const Pose& P = vo.poses[f];
      std::vector<double> r;
      if (const auto it = vo.da3.find(f); it != vo.da3.end())
        for (const auto& [j, d] : it->second)
          if (const auto X = vo.map_p.find(j); X != vo.map_p.end())
            if (const double z = (P.R * X->second + P.t)(2); z > 0) r.push_back(z / d);
      double a = vo.da3_ref;
      if (r.size() >= 20) std::nth_element(r.begin(), r.begin() + r.size() / 2, r.end()), a = r[r.size() / 2];
      fk << f << ' ' << a;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) fk << ' ' << P.R(i, j);
      fk << ' ' << P.t(0) << ' ' << P.t(1) << ' ' << P.t(2) << '\n';
    }
    std::printf("saved %s (%zu keyframes)\n", kp, vo.kfs.size());
  }
  print_ate(C, std::vector<Eigen::Matrix<double, 3, 4>>(gt.begin() + std::min<size_t>(f0, gt.size()), gt.end()));
#ifdef VP_WITH_VIEWER
  if (viewer && loop) viewer->update(frame, vo);  // 마지막 보정 반영
  if (viewer) viewer->finish(arg(argc, argv, "--snapshot", ""));
#endif
  return 0;
}
