#include "vo_plucker/loop_closing.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_set>

namespace vp {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// PGO 엣지 정보행렬 = 1 / 표준편차^2: (회전, 이동, log 스케일). 오도메트리 0.005 / 0.02 / 0.02, 루프 0.01 / 0.05 / 0.05
Eigen::Matrix<double, 7, 7> info(double r, double t, double s) {
  Eigen::Matrix<double, 7, 7> I = Eigen::Matrix<double, 7, 7>::Zero();
  for (int k = 0; k < 3; ++k) I(k, k) = 1 / (r * r), I(3 + k, 3 + k) = 1 / (t * t);
  I(6, 6) = 1 / (s * s);
  return I;
}
const Eigen::Matrix<double, 7, 7> kInfoOdom = info(0.005, 0.02, 0.02), kInfoLoop = info(0.01, 0.05, 0.05);

cv::Mat to_gray(const cv::Mat& img) {
  if (img.channels() == 1) return img;
  cv::Mat g;
  cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
  return g;
}

// ---------------- built-in place recognition: 64x20 정규화 썸네일 상관 ----------------
class ThumbnailRecognizer : public PlaceRecognizer {
 public:
  void add(int kf, const cv::Mat& img) override {
    cv::Mat th;
    cv::resize(to_gray(img), th, cv::Size(64, 20), 0, 0, cv::INTER_AREA);
    th.convertTo(th, CV_32F);
    cv::Scalar mu, sd;
    cv::meanStdDev(th, mu, sd);
    th = (th - mu[0]) / (sd[0] + 1e-6) / std::sqrt(float(th.total()));
    thumbs_[kf] = th.reshape(1, 1).clone();
  }
  std::vector<int> query(int kf, const std::vector<int>& cands, int k) override {
    std::vector<std::pair<double, int>> s;
    for (int j : cands) s.emplace_back(-thumbs_[kf].dot(thumbs_[j]), j);
    std::sort(s.begin(), s.end());
    std::vector<int> out;
    for (int i = 0; i < std::min<int>(k, s.size()); ++i) out.push_back(s[i].second);
    return out;
  }

 private:
  std::unordered_map<int, cv::Mat> thumbs_;
};

// ---------------- built-in verifier: ORB + F-RANSAC + VO 로컬 3D의 3D-3D Sim(3) RANSAC ----------------
constexpr int kMinF = 80, kMinSim3 = 20;
constexpr double kReprojPx = 3.0;

std::vector<std::pair<int, int>> ratio_match(const cv::Mat& a, const cv::Mat& b, float ratio) {
  std::vector<std::pair<int, int>> out;
  if (a.rows < 2 || b.rows < 2) return out;
  std::vector<std::vector<cv::DMatch>> m;
  cv::BFMatcher(cv::NORM_HAMMING).knnMatch(a, b, m, 2);
  for (const auto& x : m)
    if (x.size() == 2 && x[0].distance < ratio * x[1].distance) out.emplace_back(x[0].queryIdx, x[0].trainIdx);
  return out;
}

std::vector<char> f_ransac(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b, double px) {
  cv::Mat inl;
  if (a.size() >= 8) cv::findFundamentalMat(a, b, cv::FM_RANSAC, px, 0.999, 1000, inl);
  std::vector<char> ok(a.size(), 0);
  for (int i = 0; i < inl.rows; ++i) ok[i] = inl.at<uchar>(i) != 0;
  return ok;
}

Sim3 umeyama(const std::vector<Vec3>& A, const std::vector<Vec3>& B) {  // B ≈ s R A + t
  Eigen::Matrix<double, 3, Eigen::Dynamic> X(3, A.size()), Y(3, A.size());
  for (size_t i = 0; i < A.size(); ++i) X.col(i) = A[i], Y.col(i) = B[i];
  const Eigen::Matrix4d T = Eigen::umeyama(X, Y, true);
  const double s = std::cbrt(T.topLeftCorner<3, 3>().determinant());
  return Sim3{s, T.topLeftCorner<3, 3>() / s, T.topRightCorner<3, 1>()};
}

class OrbVerifier : public LoopVerifier {
 public:
  explicit OrbVerifier(const Camera& cam) : cam_(cam), orb_(cv::ORB::create(1500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 10)) {}

  int add(const KFState& st, const cv::Mat& img) override {
    KF& d = kf_[st.kf];
    std::vector<cv::KeyPoint> kp;
    orb_->detectAndCompute(to_gray(img), cv::noArray(), kp, d.des);
    for (const auto& p : kp) d.pts.push_back(p.pt);
    const int prev = st.prev;
    if (prev < 0 || !kf_.count(prev)) return -1;
    // 직전 키프레임 a의 로컬 3D: a <-> kf ORB 매칭 -> VO 포즈로 삼각측량 -> {ORB index: X in cam_a}
    KF& a = kf_[prev];
    const auto mm = ratio_match(a.des, d.des, 0.75f);
    std::vector<cv::Point2f> pa, pb;
    for (auto [i, j] : mm) pa.push_back(a.pts[i]), pb.push_back(d.pts[j]);
    const auto ok = f_ransac(pa, pb, 1.0);
    const Mat3 Rr = st.pose.R * st.prev_pose.R.transpose();
    const Vec3 tr = st.pose.t - Rr * st.prev_pose.t;  // cam_a -> cam_b (VO 단위)
    const std::vector<Pose> P{Pose{}, Pose{Rr, tr}};
    for (size_t m = 0; m < mm.size(); ++m) {
      if (!ok[m]) continue;
      const std::vector<Vec2> uv{Vec2(pa[m].x, pa[m].y), Vec2(pb[m].x, pb[m].y)};
      const auto X = tri_point_multi(cam_, P, uv);
      if (point_ok(cam_, X, P, uv, 1.0)) a.l3d[mm[m].first] = *X;
    }
    return prev;
  }

  // 기하 검증만: ORB ratio 매칭 -> F-RANSAC 인라이어 수 (캐스케이드 게이트로도 씀)
  int f_inliers(int i, int j, std::vector<std::pair<int, int>>& mm, std::vector<cv::Point2f>& pi,
                std::vector<cv::Point2f>& pj, std::vector<char>& ok) {
    const KF &ki = kf_[i], &kj = kf_[j];
    mm = ratio_match(ki.des, kj.des, 0.7f);
    if (int(mm.size()) < kMinF) return 0;
    for (auto [a, b] : mm) pi.push_back(ki.pts[a]), pj.push_back(kj.pts[b]);
    ok = f_ransac(pi, pj, 1.5);
    return std::count(ok.begin(), ok.end(), 1);
  }
  bool geometric_ok(int i, int j) {
    std::vector<std::pair<int, int>> mm;
    std::vector<cv::Point2f> pi, pj;
    std::vector<char> ok;
    return f_inliers(i, j, mm, pi, pj, ok) >= kMinF;
  }

  std::optional<LoopEdge> verify(int i, int j) override {
    const KF &ki = kf_[i], &kj = kf_[j];
    std::vector<std::pair<int, int>> mm;
    std::vector<cv::Point2f> pi, pj;
    std::vector<char> ok;
    if (f_inliers(i, j, mm, pi, pj, ok) < kMinF) return std::nullopt;
    std::vector<Vec3> Xi, Xj;
    std::vector<Vec2> ui, uj;
    int nf = 0;
    for (size_t m = 0; m < mm.size(); ++m) {
      if (!ok[m]) continue;
      ++nf;
      const auto a = ki.l3d.find(mm[m].first), b = kj.l3d.find(mm[m].second);
      if (a == ki.l3d.end() || b == kj.l3d.end()) continue;
      Xi.push_back(a->second), Xj.push_back(b->second);
      ui.emplace_back(pi[m].x, pi[m].y), uj.emplace_back(pj[m].x, pj[m].y);
    }
    if (nf < kMinF || int(Xi.size()) < kMinSim3) return std::nullopt;
    // Sim(3) RANSAC: Xi ≈ S Xj, 인라이어 = 양쪽 이미지 재투영 < kReprojPx
    const int n = Xi.size();
    auto inliers = [&](const Sim3& S, std::vector<char>& in) {
      const Sim3 Si = S.inverse();
      int c = 0;
      in.assign(n, 0);
      for (int m = 0; m < n; ++m) {
        const Vec3 a = cam_.K * S.map(Xj[m]), b = cam_.K * Si.map(Xi[m]);
        if (a(2) <= 0 || b(2) <= 0) continue;
        const double e = std::max((a.head<2>() / a(2) - ui[m]).norm(), (b.head<2>() / b(2) - uj[m]).norm());
        if (e < kReprojPx) in[m] = 1, ++c;
      }
      return c;
    };
    std::vector<char> best_in, in;
    int best = 0;
    std::uniform_int_distribution<int> pick(0, n - 1);
    for (int it = 0; it < 300; ++it) {
      int a = pick(rng_), b = pick(rng_), c = pick(rng_);
      if (a == b || b == c || a == c) continue;
      const Sim3 S = umeyama({Xj[a], Xj[b], Xj[c]}, {Xi[a], Xi[b], Xi[c]});
      if (!std::isfinite(S.s) || S.s <= 0) continue;
      const int cnt = inliers(S, in);
      if (cnt > best) best = cnt, best_in = in;
    }
    if (best < kMinSim3) return std::nullopt;
    std::vector<Vec3> A, B;
    for (int m = 0; m < n; ++m)
      if (best_in[m]) A.push_back(Xj[m]), B.push_back(Xi[m]);
    return LoopEdge{i, j, umeyama(A, B), best};  // 인라이어 전체로 재추정
  }

  void rescale(int kf, double s) override {
    const auto it = kf_.find(kf);
    if (it == kf_.end()) return;
    for (auto& [m, X] : it->second.l3d) X /= s;
  }

 private:
  struct KF {
    std::vector<cv::Point2f> pts;
    cv::Mat des;
    std::unordered_map<int, Vec3> l3d;  // ORB index -> X in cam (VO 단위)
  };
  Camera cam_;
  cv::Ptr<cv::ORB> orb_;
  std::unordered_map<int, KF> kf_;
  std::mt19937 rng_{0};
};
}  // namespace

// ---------------- Python 플러그인 서버: sh -c CMD 를 자식 프로세스로, stdin/stdout 파이프로 한 줄씩 요청/응답 ----------------
struct LoopServer {
  explicit LoopServer(const std::string& cmd) {
    int in[2], out[2];
    if (pipe(in) || pipe(out)) throw std::runtime_error("pipe failed");
    signal(SIGPIPE, SIG_IGN);  // 서버가 죽으면 쓰기 실패 -> 아래 ask 에서 예외
    pid = fork();
    if (pid == 0) {
      dup2(in[0], 0), dup2(out[1], 1);
      close(in[0]), close(in[1]), close(out[0]), close(out[1]);
      execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char*>(nullptr));
      _exit(127);
    }
    close(in[0]), close(out[1]);
    to = fdopen(in[1], "w"), from = fdopen(out[0], "r");
    char tmpl[] = "/tmp/vp_loop_XXXXXX";
    dir = mkdtemp(tmpl);
  }
  ~LoopServer() {
    std::fputs("quit\n", to);
    std::fclose(to), std::fclose(from);
    waitpid(pid, nullptr, 0);
    rmdir(dir.c_str());
  }
  std::string ask(const std::string& req) {
    std::fputs((req + "\n").c_str(), to);
    std::fflush(to);
    std::string r;
    char buf[4096];
    while (std::fgets(buf, sizeof buf, from)) {
      r += buf;
      if (r.back() == '\n') return r.substr(0, r.size() - 1);
    }
    throw std::runtime_error("loop server died (see its stderr): " + req.substr(0, 40));
  }
  void add(int kf, const cv::Mat& img) {  // 후보 검색과 검증이 같은 키프레임을 등록하므로 한 번만. 영상은 임시 BMP로 (무압축, 빠름)
    if (!added.insert(kf).second) return;
    const std::string path = dir + "/" + std::to_string(kf) + ".bmp";
    cv::imwrite(path, img);
    ask("add " + std::to_string(kf) + " " + path);
    std::remove(path.c_str());
  }
  pid_t pid;
  FILE *to, *from;
  std::string dir;
  std::unordered_set<int> added;
};

namespace {
class PyRecognizer : public PlaceRecognizer {
 public:
  PyRecognizer(std::shared_ptr<LoopServer> s, double min_sim) : s_(std::move(s)), min_sim_(min_sim) {}
  void add(int kf, const cv::Mat& img) override { s_->add(kf, img); }
  std::vector<int> query(int kf, const std::vector<int>& cands, int k) override {
    std::string q = "query " + std::to_string(kf) + " " + std::to_string(k);
    for (int j : cands) q += " " + std::to_string(j);
    std::istringstream r(s_->ask(q));
    std::string tag;
    r >> tag;
    std::vector<int> out;
    int j;
    double sim;
    while (r >> j >> sim)
      if (sim >= min_sim_) out.push_back(j);  // 전역 디스크립터 코사인 유사도가 낮으면 검증도 안 함
    return out;
  }

 private:
  std::shared_ptr<LoopServer> s_;
  double min_sim_;
};

class PyVerifier : public LoopVerifier {
 public:
  PyVerifier(std::shared_ptr<LoopServer> s, const Camera& cam) : s_(std::move(s)), cam_(cam) {}
  int add(const KFState& st, const cv::Mat& img) override {
    s_->add(st.kf, img);
    uvd_[st.kf] = st.uvd;  // 서버가 MASt3R metric 깊이와 비교해 VO 단위/미터를 구함
    return st.kf;
  }
  std::optional<LoopEdge> verify(int i, int j) override {
    std::ostringstream q;
    q.precision(9);
    q << "verify " << i << " " << j << " " << cam_.fx << " " << cam_.fy << " " << cam_.cx << " " << cam_.cy;
    for (int f : {i, j}) {
      q << " " << uvd_[f].size();
      for (const Vec3& x : uvd_[f]) q << " " << x(0) << " " << x(1) << " " << x(2);
    }
    std::istringstream r(s_->ask(q.str()));
    std::string tag;
    r >> tag;
    if (tag != "loop") return std::nullopt;
    LoopEdge e{i, j, {}, 0};
    r >> e.S.s;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) r >> e.S.R(a, b);
    r >> e.S.t(0) >> e.S.t(1) >> e.S.t(2) >> e.inliers;
    if (!r) return std::nullopt;
    return e;
  }
  void rescale(int kf, double s) override {
    for (Vec3& x : uvd_[kf]) x(2) /= s;
  }

 private:
  std::shared_ptr<LoopServer> s_;
  Camera cam_;
  std::unordered_map<int, std::vector<Vec3>> uvd_;
};
}  // namespace

std::unique_ptr<PlaceRecognizer> make_thumbnail_recognizer() { return std::make_unique<ThumbnailRecognizer>(); }
std::unique_ptr<LoopVerifier> make_orb_verifier(const Camera& cam) { return std::make_unique<OrbVerifier>(cam); }
std::shared_ptr<LoopServer> start_loop_server(const std::string& cmd) { return std::make_shared<LoopServer>(cmd); }
std::unique_ptr<PlaceRecognizer> make_py_recognizer(std::shared_ptr<LoopServer> srv, double min_sim) {
  return std::make_unique<PyRecognizer>(std::move(srv), min_sim);
}
std::unique_ptr<LoopVerifier> make_py_verifier(std::shared_ptr<LoopServer> srv, const Camera& cam) {
  return std::make_unique<PyVerifier>(std::move(srv), cam);
}

namespace {
// ORB F-RANSAC 기하 검증을 통과한 후보만 main (MASt3R 등) 으로 Sim(3) 추정
class CascadeVerifier : public LoopVerifier {
 public:
  CascadeVerifier(const Camera& cam, std::unique_ptr<LoopVerifier> main) : gate_(cam), main_(std::move(main)) {}
  int add(const KFState& st, const cv::Mat& img) override {
    gate_.add(st, img);
    return main_->add(st, img);
  }
  std::optional<LoopEdge> verify(int i, int j) override {
    return gate_.geometric_ok(i, j) ? main_->verify(i, j) : std::nullopt;
  }
  void rescale(int kf, double s) override { main_->rescale(kf, s); }

 private:
  OrbVerifier gate_;
  std::unique_ptr<LoopVerifier> main_;
};
}  // namespace

std::unique_ptr<LoopVerifier> make_cascade_verifier(const Camera& cam, std::unique_ptr<LoopVerifier> main) {
  return std::make_unique<CascadeVerifier>(cam, std::move(main));
}

LoopCloser::LoopCloser(VOTracker& vo, std::unique_ptr<PlaceRecognizer> pr, std::unique_ptr<LoopVerifier> ver,
                       const LoopParams& prm)
    : vo_(vo), pr_(std::move(pr)), ver_(std::move(ver)), prm_(prm) {
  if (prm_.threaded) th_ = std::thread(&LoopCloser::run, this);
}

LoopCloser::~LoopCloser() { finish(); }

void LoopCloser::push(int kf, const cv::Mat& img) {
  if (!prm_.threaded) return process(kf, img);
  {
    std::lock_guard<std::mutex> l(q_mx_);
    queue_.emplace_back(kf, img);
  }
  q_cv_.notify_one();
}

void LoopCloser::finish() {
  if (!th_.joinable()) return;
  {
    std::lock_guard<std::mutex> l(q_mx_);
    stop_ = true;
  }
  q_cv_.notify_one();
  th_.join();
}

void LoopCloser::run() {
  for (;;) {
    std::unique_lock<std::mutex> l(q_mx_);
    q_cv_.wait(l, [&] { return stop_ || !queue_.empty(); });
    if (queue_.empty()) return;  // stop_ 이고 남은 것도 다 처리함
    // 밀려 있으면 오래된 키프레임은 버리고 (등록도 안 함) 최신 쪽만 등록 + 검출. 루프 DB 가 듬성해질 뿐 검출은 계속됨
    // (등록만 하고 검출을 건너뛰면, 계속 밀리는 무거운 모듈에서는 검출을 한 번도 못 함)
    while (int(queue_.size()) > prm_.max_backlog + 1) queue_.pop_front(), ++n_skipped;
    auto [kf, img] = std::move(queue_.front());
    queue_.pop_front();
    l.unlock();
    process(kf, img);
  }
}

std::unique_lock<std::mutex> LoopCloser::lock_map() {
  if (!prm_.threaded) return {};
  while (want_map_) std::this_thread::yield();  // 루프 스레드가 기다리는 중이면 먼저 보냄
  return std::unique_lock<std::mutex>(map_mx_);
}

std::unique_lock<std::mutex> LoopCloser::take_map() {
  want_map_ = true;
  std::unique_lock<std::mutex> l(map_mx_);
  want_map_ = false;
  return l;
}

KFState LoopCloser::snapshot(int kf) const {  // map lock 아래에서
  KFState st;
  st.kf = kf, st.pose = vo_.poses[kf];
  if (!kfs_.empty()) st.prev = kfs_.back(), st.prev_pose = vo_.poses[st.prev];
  std::vector<Vec3> d;
  const FrameObs& o = vo_.obs[kf];
  for (size_t m = 0; m < o.pid.size(); ++m)
    if (const auto it = vo_.map_p.find(o.pid[m]); it != vo_.map_p.end())
      if (const double z = (st.pose.R * it->second + st.pose.t)(2); z > 0) d.emplace_back(o.uv[m](0), o.uv[m](1), z);
  for (size_t m = 0, n = std::min<size_t>(300, d.size()); m < n; ++m) st.uvd.push_back(d[m * d.size() / n]);
  return st;
}

void LoopCloser::process(int kf, const cv::Mat& img) {
  auto t0 = Clock::now();
  KFState st;
  {
    auto l = take_map();
    const auto tb = Clock::now();
    const bool ok = vo_.has_pose[kf] && (kfs_.empty() || kf > kfs_.back());  // 초기화 재시도로 다시 온 키프레임 무시
    if (ok) st = snapshot(kf);
    blocked(ms_since(tb));
    if (!ok) return;
  }
  kfs_.push_back(kf);
  pr_->add(kf, img);
  const int a = ver_->add(st, img);
  if (a < 0) return void(ms_detect += ms_since(t0));
  std::vector<int> cands;
  for (int j : ready_)
    if (a - j >= prm_.min_gap) cands.push_back(j);
  ready_.push_back(a);
  std::optional<LoopEdge> best;
  if (!cands.empty()) {
    ++n_queries;
    for (int j : pr_->query(a, cands, prm_.shortlist))
      if (auto e = ver_->verify(a, j); e && (!best || e->inliers > best->inliers)) best = e;
  }
  ms_detect += ms_since(t0);
  if (!best) return;

  t0 = Clock::now();
  loops.push_back(*best);
  edges_.push_back(*best);
  std::vector<int> sub(kfs_.begin(), std::find(kfs_.begin(), kfs_.end(), a) + 1);  // 키프레임 a까지
  std::vector<Pose> P;  // PGO 초기값 (옛 키프레임 포즈는 루프 스레드만 바꾸므로 써넣을 때까지 그대로)
  {
    auto l = take_map();
    const auto tb = Clock::now();
    for (int f : sub) P.push_back(vo_.poses[f]);
    blocked(ms_since(tb));
  }
  // 새 루프 엣지를 지금 포즈로 평가 (EdgeSim3 와 같은 error). 이미 맞으면 엣지만 남기고 PGO 생략 -
  // ORB-SLAM3 는 이미 연결된 (covisible) 키프레임을 후보에서 빼서 재방문 구간에서 같은 루프를 다시 풀지 않음
  const auto at_sub = [&](int f) { return P[std::find(sub.begin(), sub.end(), f) - sub.begin()]; };
  const Pose Pa = at_sub(a), Pj = at_sub(best->j);
  const Eigen::Matrix<double, 7, 1> r =
      (g2o::Sim3(best->S.R, best->S.t, best->S.s) * g2o::Sim3(Pj.R, Pj.t, 1.0) * g2o::Sim3(Pa.R, Pa.t, 1.0).inverse()).log();
  const double chi2_new = r.dot(kInfoLoop * r);
  if (chi2_new < prm_.skip_chi2) {
    ++n_pgo_skipped;
    std::printf("    loop @frame %d <-> %d: %d Sim(3) inliers, scale ratio %.2f, edge chi2 %.1f -> consistent, PGO skipped\n",
                a, best->j, best->inliers, best->S.s, chi2_new);
    ms_pgo += ms_since(t0);
    return;
  }
  double cost = 0;
  const auto S_kf = pgo(P, sub, cost);  // lock 없이: 트래킹은 계속
  int at = -1;
  if (prm_.feedback) {
    auto l = take_map();
    const auto tb = Clock::now();
    correct(S_kf, sub);
    at = int(vo_.poses.size()) - 1;
    ++n_corrections;
    blocked(ms_since(tb));
  }
  std::printf("    loop @frame %d <-> %d: %d Sim(3) inliers, scale ratio %.2f, edge chi2 %.1f -> PGO cost %.1f", a,
              best->j, best->inliers, best->S.s, chi2_new, cost);
  if (at >= 0) std::printf(" -> map corrected (tracking at frame %d)", at);
  std::printf("\n");
  ms_pgo += ms_since(t0);
}

std::unordered_map<int, Sim3> LoopCloser::pgo(const std::vector<Pose>& P, const std::vector<int>& sub, double& cost) {
  // 정점 = 키프레임 Sim(3) (world -> cam), 초기값 = P (sub 순서). 엣지 = 보관된 측정 (없는 오도메트리는 P 로)
  auto pose_sim3 = [&](size_t n) { return Sim3{1.0, P[n].R, P[n].t}; };
  for (size_t n = 0; n + 1 < sub.size(); ++n)
    if (!odom_.count(sub[n])) odom_[sub[n]] = pose_sim3(n + 1) * pose_sim3(n).inverse();

  // ORB-SLAM3 essential graph 와 같은 설정: Sim(3) 정점뿐이라 고정 7x7 블록 (BlockSolverX 보다 반복당 ~2.4배 빠름),
  // lambda 1e-16 (거의 Gauss-Newton 으로 시작), 20회. 같은 그래프에서 30회 X 솔버와 같은 cost 로 수렴 (00, 06 확인)
  using Block = g2o::BlockSolver_7_3;
  g2o::SparseOptimizer opt;
  auto* lm = new g2o::OptimizationAlgorithmLevenberg(
      std::make_unique<Block>(std::make_unique<g2o::LinearSolverEigen<Block::PoseMatrixType>>()));
  lm->setUserLambdaInit(1e-16);
  opt.setAlgorithm(lm);
  std::unordered_map<int, g2o::VertexSim3Expmap*> v;
  for (size_t n = 0; n < sub.size(); ++n) {
    auto* x = new g2o::VertexSim3Expmap();
    x->setId(n);
    x->setEstimate(g2o::Sim3(P[n].R, P[n].t, 1.0));
    x->setFixed(n == 0);
    opt.addVertex(x);
    v[sub[n]] = x;
  }
  auto add_edge = [&](int from, int to, const Sim3& M, const Eigen::Matrix<double, 7, 7>& I) {  // M: cam_from -> cam_to
    auto* e = new g2o::EdgeSim3();
    e->setVertex(0, v[from]);
    e->setVertex(1, v[to]);
    e->setMeasurement(g2o::Sim3(M.R, M.t, M.s));
    e->setInformation(I);
    opt.addEdge(e);
  };
  for (size_t n = 0; n + 1 < sub.size(); ++n) add_edge(sub[n], sub[n + 1], odom_[sub[n]], kInfoOdom);
  for (const auto& e : edges_)
    if (v.count(e.i) && v.count(e.j)) add_edge(e.j, e.i, e.S, kInfoLoop);
  opt.initializeOptimization();
  const auto t0 = Clock::now();
  opt.optimize(20);
  ms_opt += ms_since(t0);
  cost = opt.chi2();
  std::unordered_map<int, Sim3> out;
  for (auto& [f, x] : v) {
    const g2o::Sim3& S = x->estimate();
    out[f] = Sim3{S.scale(), S.rotation().toRotationMatrix(), S.translation()};
  }
  return out;
}

void LoopCloser::correct(const std::unordered_map<int, Sim3>& S_kf, const std::vector<int>& sub) {
  VOTracker& vo = vo_;  // map lock 아래에서
  // 프레임 f: 직전 키프레임 a 기준 VO 상대포즈를 붙인 S_f -> 새 SE(3) = (R, t / s) (카메라 좌표 1/s 배).
  // 랜드마크는 마지막으로 본 프레임 기준으로 X <- S_f^-1 S_f,old X
  const int n = vo.poses.size();
  std::vector<Sim3> corr(n);
  std::vector<double> scale(n, 1.0);
  std::vector<Pose> fresh(n);
  size_t kk = 0;
  for (int f = 0; f < n; ++f) {
    if (!vo.has_pose[f]) continue;
    while (kk + 1 < sub.size() && sub[kk + 1] <= f) ++kk;
    const int a = sub[kk];
    if (f < a) continue;  // 첫 키프레임 이전 (깊이 초기화 등): 그대로
    const Mat3 Rr = vo.poses[f].R * vo.poses[a].R.transpose();
    const Vec3 tr = vo.poses[f].t - Rr * vo.poses[a].t;
    const Sim3& Sa = S_kf.at(a);
    const Sim3 Sf{Sa.s, Rr * Sa.R, Rr * Sa.t + tr};
    corr[f] = Sf.inverse() * Sim3{1.0, vo.poses[f].R, vo.poses[f].t};  // old world -> new world
    scale[f] = Sa.s;
    fresh[f] = Pose{orthonormalize(Sf.R), Sf.t / Sa.s};  // 안 하면 R R^T 오차가 보정마다 2배로 커짐 (00에서 발산)
  }
  for (int f = sub.front(); f < n; ++f)
    if (vo.has_pose[f]) vo.poses[f] = fresh[f];
  for (auto& [j, X] : vo.map_p)
    if (const auto it = vo.last_p.find(j); it != vo.last_p.end() && it->second >= sub.front()) X = corr[it->second].map(X);
  for (auto& [j, L] : vo.map_l)
    if (const auto it = vo.last_l.find(j); it != vo.last_l.end() && it->second >= sub.front()) {
      const Sim3& C = corr[it->second];
      const Vec3 d = C.R * L.head<3>();
      L << d, C.s * C.R * L.tail<3>() + C.t.cross(d);  // m = p x d 규약
    }
  for (int f : kfs_) ver_->rescale(f, scale[f]);
  // 보관된 측정 (cam_from -> cam_to): S' = (1/s_to) S (s_from)
  auto rescale = [&](const Sim3& S, int from, int to) { return Sim3{S.s * scale[from] / scale[to], S.R, S.t / scale[to]}; };
  for (auto& [a, M] : odom_) {
    const auto it = std::upper_bound(kfs_.begin(), kfs_.end(), a);
    if (it != kfs_.end()) M = rescale(M, a, *it);
  }
  for (auto& e : edges_) e.S = rescale(e.S, e.j, e.i);
}

}  // namespace vp
