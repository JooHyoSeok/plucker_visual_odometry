#include "vo_plucker/loop_closing.h"

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>

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
const Eigen::Matrix<double, 7, 7> kInfoLoop = info(0.01, 0.05, 0.05);
// DA3 깊이 3D 루프: 스케일 a_i/a_j 는 DA3 앵커 엣지와 같은 정보 (이중 계산 X), 이동은 깊이 잡음 (~6 % x 거리) 만큼 덜 믿음
const Eigen::Matrix<double, 7, 7> kInfoLoopDa3 = info(0.01, 0.1, 1.0);
// DA3 앵커가 있으면 메트릭 스케일은 앵커가 줌 -> 모든 루프 엣지의 스케일은 약하게 (EuRoC V1_01: GT 로 참인 루프의 스케일 비가
// 0.49-2.87 로 틀려서 PGO 를 비틂). 루프는 회전 / 이동만 고침
const Eigen::Matrix<double, 7, 7> kInfoLoopAnchored = info(0.01, 0.05, 1.0);
constexpr double kMaxLoopChi2Post = 300;  // PGO 뒤 새 루프 엣지 chi2 가 이보다 크면 잘못된 루프 (KITTI 참 루프 최대 254, EuRoC 나쁜 루프 500-8000)

// metric depth 앵커: 키프레임 Sim(3) 정점의 log 스케일 = 측정 (log a_k - log a_ref). Jacobian 은 g2o 수치 미분
class EdgeScalePrior : public g2o::BaseUnaryEdge<1, double, g2o::VertexSim3Expmap> {
 public:
  void computeError() override {
    _error[0] = std::log(static_cast<const g2o::VertexSim3Expmap*>(_vertices[0])->estimate().scale()) - _measurement;
  }
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

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
constexpr int kMinSim3 = 20;

std::vector<Vec3> subsample(const std::vector<Vec3>& d, size_t n) {  // 고르게 최대 n 개
  std::vector<Vec3> out;
  for (size_t m = 0, k = std::min(n, d.size()); m < k; ++m) out.push_back(d[m * d.size() / k]);
  return out;
}
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
  explicit OrbVerifier(const Camera& cam, int min_f = 80, std::shared_ptr<LoopServer> da3 = nullptr, bool refine = true)
      : cam_(cam), orb_(cv::ORB::create(1500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 10)), kMinF(min_f),
        da3_(std::move(da3)), refine_(refine) {}

  int add(const KFState& st, const cv::Mat& img) override {
    KF& d = kf_[st.kf];
    std::vector<cv::KeyPoint> kp;
    orb_->detectAndCompute(to_gray(img), cv::noArray(), kp, d.des);
    for (const auto& p : kp) d.pts.push_back(p.pt);
    if (da3_ && st.da3_scale > 0) {  // DA3 깊이 3D (RGB-D 처럼): X = a_k z K^-1 [u v 1], 서버가 앵커와 같은 범위만 (밖은 0)
      std::vector<Vec2> uv;
      for (const auto& p : d.pts) uv.emplace_back(p.x, p.y);
      const std::vector<double> z = server_depth_kf(*da3_, st.kf, uv);
      for (size_t m = 0; m < z.size(); ++m)
        if (z[m] > 0) d.d3[m] = st.da3_scale * z[m] * cam_.Kinv * Vec3(uv[m](0), uv[m](1), 1);
    }
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
    // 로컬 (2-view 삼각측량) 3D 로 먼저, 안 되면 DA3 깊이 3D 로 (시작 직후처럼 로컬 3D 가 부정확 / 적은 곳)
    if (auto e = sim3(i, j, ki.l3d, kj.l3d, mm, pi, pj, ok)) return e;
    if (ki.d3.empty() || kj.d3.empty()) return std::nullopt;
    auto e = sim3(i, j, ki.d3, kj.d3, mm, pi, pj, ok);
    if (e) e->da3 = true, std::fprintf(stderr, "    [loop %d <-> %d: Sim(3) from DA3 depth, %d inliers]\n", i, j, e->inliers);
    return e;
  }

  // F-RANSAC 인라이어 중 양쪽에 3D 가 있는 매칭으로 Sim(3) RANSAC
  std::optional<LoopEdge> sim3(int i, int j, const std::unordered_map<int, Vec3>& Ai, const std::unordered_map<int, Vec3>& Aj,
                               const std::vector<std::pair<int, int>>& mm, const std::vector<cv::Point2f>& pi,
                               const std::vector<cv::Point2f>& pj, const std::vector<char>& ok) {
    std::vector<Vec3> Xi, Xj;
    std::vector<Vec2> ui, uj;
    for (size_t m = 0; m < mm.size(); ++m) {
      if (!ok[m]) continue;
      const auto a = Ai.find(mm[m].first), b = Aj.find(mm[m].second);
      if (a == Ai.end() || b == Aj.end()) continue;
      Xi.push_back(a->second), Xj.push_back(b->second);
      ui.emplace_back(pi[m].x, pi[m].y), uj.emplace_back(pj[m].x, pj[m].y);
    }
    if (int(Xi.size()) < kMinSim3) return std::nullopt;
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
    const Sim3 S0 = umeyama(A, B);  // 인라이어 전체로 재추정 (3D-3D)
    if (!refine_) return LoopEdge{i, j, S0, best};
    // 양방향 재투영으로 Sim(3) 정제 (ORB-SLAM OptimizeSim3 의 생각): 3D-3D 는 깊이 오차를 그대로 쓰지만, 재투영은 광선 방향
    // 깊이 오차에 둔감 -> 작은 기선 / DA3 깊이 3D 에서도 회전 / 이동이 맞음. 정제 후 헤시안 = 이 루프 측정의 정보행렬 (H)
    g2o::SparseOptimizer opt;
    using Block = g2o::BlockSolverX;
    opt.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<Block>(std::make_unique<g2o::LinearSolverEigen<Block::PoseMatrixType>>())));
    auto* v = new g2o::VertexSim3Expmap();
    v->setId(0), v->setEstimate(g2o::Sim3(S0.R, S0.t, S0.s)), v->_fix_scale = false;
    v->_principle_point1 = v->_principle_point2 = Eigen::Vector2d(cam_.cx, cam_.cy);
    v->_focal_length1 = v->_focal_length2 = Eigen::Vector2d(cam_.fx, cam_.fy);
    opt.addVertex(v);
    std::vector<g2o::OptimizableGraph::Edge*> es;
    int id = 1;
    for (int m = 0; m < n; ++m) {
      if (!best_in[m]) continue;
      auto* pj = new g2o::VertexPointXYZ();
      auto* pi = new g2o::VertexPointXYZ();
      pj->setId(id++), pj->setEstimate(Xj[m]), pj->setFixed(true), opt.addVertex(pj);
      pi->setId(id++), pi->setEstimate(Xi[m]), pi->setFixed(true), opt.addVertex(pi);
      auto* e1 = new g2o::EdgeSim3ProjectXYZ();  // j 의 점 -> S -> i 영상
      e1->setVertex(0, pj), e1->setVertex(1, v), e1->setMeasurement(ui[m]), e1->setInformation(Eigen::Matrix2d::Identity());
      auto* e2 = new g2o::EdgeInverseSim3ProjectXYZ();  // i 의 점 -> S^-1 -> j 영상
      e2->setVertex(0, pi), e2->setVertex(1, v), e2->setMeasurement(uj[m]), e2->setInformation(Eigen::Matrix2d::Identity());
      for (auto* e : {static_cast<g2o::OptimizableGraph::Edge*>(e1), static_cast<g2o::OptimizableGraph::Edge*>(e2)}) {
        auto* rk = new g2o::RobustKernelHuber();
        rk->setDelta(std::sqrt(10.0));
        e->setRobustKernel(rk), opt.addEdge(e), es.push_back(e);
      }
    }
    opt.initializeOptimization();
    opt.optimize(5);
    int nin = 0;
    for (size_t q = 0; q + 1 < es.size(); q += 2) {  // 짝 (e1, e2) 중 하나라도 크면 그 대응을 뺌 (ORB-SLAM: chi2 > 10)
      es[q]->computeError(), es[q + 1]->computeError();
      const bool bad = es[q]->chi2() > 10 || es[q + 1]->chi2() > 10;
      es[q]->setLevel(bad), es[q + 1]->setLevel(bad), nin += !bad;
    }
    if (nin < kMinSim3) return std::nullopt;
    opt.initializeOptimization(0);
    opt.optimize(5);
    // 헤시안 (7x7, g2o Sim3 왼쪽 섭동 [w u s] = PGO EdgeSim3 오차와 같은 좌표): 인라이어 재투영 잔차의 수치 Jacobian
    Eigen::Matrix<double, 7, 7> H = Eigen::Matrix<double, 7, 7>::Zero();
    std::vector<Eigen::Matrix<double, 2, 7>> J(es.size());
    for (int k = 0; k < 7; ++k)
      for (int sgn : {1, -1}) {
        double d[7] = {0, 0, 0, 0, 0, 0, 0};
        d[k] = sgn * 1e-6;
        v->push(), v->oplus(d);
        for (size_t q = 0; q < es.size(); ++q)
          if (es[q]->level() == 0) {
            es[q]->computeError();
            const Eigen::Map<const Eigen::Vector2d> r(es[q]->errorData());
            J[q].col(k) += sgn * r / 2e-6;
          }
        v->pop();
      }
    for (size_t q = 0; q < es.size(); ++q)
      if (es[q]->level() == 0) H += J[q].transpose() * J[q];
    const g2o::Sim3 Sr = v->estimate();
    LoopEdge le{i, j, Sim3{Sr.scale(), Sr.rotation().toRotationMatrix(), Sr.translation()}, nin};
    le.H = H;
    return le;
  }

  void rescale(int kf, double s) override {
    const auto it = kf_.find(kf);
    if (it == kf_.end()) return;
    for (auto& [m, X] : it->second.l3d) X /= s;
    for (auto& [m, X] : it->second.d3) X /= s;
  }

 private:
  struct KF {
    std::vector<cv::Point2f> pts;
    cv::Mat des;
    std::unordered_map<int, Vec3> l3d;  // ORB index -> X in cam (VO 단위)
    std::unordered_map<int, Vec3> d3;   // ORB index -> DA3 깊이 X in cam (VO 단위)
  };
  Camera cam_;
  cv::Ptr<cv::ORB> orb_;
  std::unordered_map<int, KF> kf_;
  std::mt19937 rng_{0};
  int kMinF;  // F-RANSAC 인라이어 최소 수
  std::shared_ptr<LoopServer> da3_;
  bool refine_;  // 재투영 Sim(3) 정제 (끄면 3D-3D Umeyama 그대로, 비교용)
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
    std::lock_guard<std::mutex> lk(mx);  // 메인 스레드 (--da3-init 첫 프레임) 와 루프 스레드가 같이 씀
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
  std::vector<double> depth_at(const cv::Mat& img, const std::vector<Vec2>& uv) {  // DA3 metric depth [m], 0 = 없음
    const std::string path = dir + "/depthat.bmp";
    cv::imwrite(path, img);
    std::ostringstream q;
    q.precision(7);
    q << "depthat " << path << " " << uv.size();
    for (const Vec2& x : uv) q << " " << x(0) << " " << x(1);
    std::istringstream r(ask(q.str()));
    std::remove(path.c_str());
    std::string tag;
    r >> tag;
    std::vector<double> z(uv.size(), 0.0);
    for (double& v : z) r >> v;
    return z;
  }
  std::vector<double> depth_kf(int kf, const std::vector<Vec2>& uv) {  // 마지막 anchor 키프레임의 DA3 깊이 맵에서 [m]
    std::ostringstream q;
    q.precision(7);
    q << "depthkf " << kf << " " << uv.size();
    for (const Vec2& x : uv) q << " " << x(0) << " " << x(1);
    std::istringstream r(ask(q.str()));
    std::string tag;
    r >> tag;
    std::vector<double> z(uv.size(), 0.0);
    for (double& v : z) r >> v;
    return r ? z : std::vector<double>(uv.size(), 0.0);
  }
  // 방금 add 한 키프레임의 VO 단위/미터. z: 점별 DA3 깊이 [m] (쓴 범위 밖이면 0)
  std::optional<double> anchor(int kf, const std::vector<Vec3>& uvd, std::vector<double>* z = nullptr) {
    std::ostringstream q;
    q.precision(9);
    q << "anchor " << kf << " " << uvd.size();
    for (const Vec3& x : uvd) q << " " << x(0) << " " << x(1) << " " << x(2);
    std::istringstream r(ask(q.str()));
    std::string tag;
    double a = 0;
    int n = 0;
    r >> tag >> a >> n;
    if (tag != "anchor" || !r || !(a > 0)) return std::nullopt;
    if (z) {
      z->assign(uvd.size(), 0.0);
      for (double& v : *z) r >> v;
      if (!r) z->clear();
    }
    return a;
  }
  pid_t pid;
  FILE *to, *from;
  std::mutex mx;
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
    uvd_[st.kf] = subsample(st.uvd, 300);  // 서버가 MASt3R metric 깊이와 비교해 VO 단위/미터를 구함
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
std::unique_ptr<LoopVerifier> make_orb_verifier(const Camera& cam, int min_f, std::shared_ptr<LoopServer> da3, bool refine) {
  return std::make_unique<OrbVerifier>(cam, min_f, std::move(da3), refine);
}
std::shared_ptr<LoopServer> start_loop_server(const std::string& cmd) { return std::make_shared<LoopServer>(cmd); }
std::vector<double> server_depth_at(LoopServer& s, const cv::Mat& img, const std::vector<Vec2>& uv) { return s.depth_at(img, uv); }
std::vector<double> server_depth_kf(LoopServer& s, int kf, const std::vector<Vec2>& uv) { return s.depth_kf(kf, uv); }
void server_depth_range(LoopServer& s, double zmin, double zmax) {
  std::ostringstream q;
  q << "range " << zmin << " " << zmax;
  s.ask(q.str());
}
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
                       const LoopParams& prm, std::shared_ptr<LoopServer> anchor)
    : vo_(vo), pr_(std::move(pr)), ver_(std::move(ver)), prm_(prm), anchor_(std::move(anchor)) {
  if (prm_.threaded) th_ = std::thread(&LoopCloser::run, this);
}

LoopCloser::~LoopCloser() { finish(); }

const Eigen::Matrix<double, 7, 7>& LoopCloser::loop_info(const LoopEdge& e) const {
  return e.da3 ? kInfoLoopDa3 : a_ref_ > 0 ? kInfoLoopAnchored : kInfoLoop;
}


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
    // 밀려 있으면 DA3 (키프레임당 ~200 ms) 만 건너뛰고 루프 등록 + 검출 (썸네일 + ORB ~25 ms) 은 모든 키프레임에.
    // 그래도 크게 밀리면 (무거운 검출 모듈) 오래된 키프레임은 버림: 등록만 하고 검출을 건너뛰면 검출을 한 번도 못 할 수 있음
    while (int(queue_.size()) > 4 * (prm_.max_backlog + 1)) queue_.pop_front(), ++n_skipped;
    const bool depth = int(queue_.size()) <= prm_.max_backlog + 1;
    auto [kf, img] = std::move(queue_.front());
    queue_.pop_front();
    l.unlock();
    if (!depth && anchor_) ++n_depth_skipped;
    process(kf, img, depth);
  }
}

// DA3 깊이로 선분 하나에서 바로 3D 라인 (삼각측량 불가능한 진행 방향 라인도): 3D 직선 위 점의 역깊이는 영상 선분 매개변수 s 의
// 1차 함수 (1/z = α + β s) -> 샘플 (s, 1/z) 에 Theil-Sen 직선 -> 양 끝 깊이 -> 두 점 -> Plücker. 가림 경계는 앞쪽 (최소 깊이)
void LoopCloser::add_da3_lines(const KFState& st, double a_k) {
  constexpr int kS = 8;
  constexpr double kOff = 3.0;  // 선분 법선 방향 오프셋 [px]
  std::vector<Vec2> uv;
  for (const Vec4& g : st.new_seg) {
    const Vec2 p = g.head<2>(), q = g.tail<2>(), d = (q - p).normalized(), nrm(-d(1), d(0));
    for (int i = 0; i < kS; ++i) {
      const Vec2 x = p + (0.1 + 0.8 * i / (kS - 1)) * (q - p);
      uv.push_back(x), uv.push_back(x + kOff * nrm), uv.push_back(x - kOff * nrm);
    }
  }
  const std::vector<double> z = server_depth_kf(*anchor_, st.kf, uv);
  std::vector<int> ids;
  std::vector<std::pair<Vec3, Vec3>> ends;  // 카메라 좌표 양 끝점 (VO 단위) -> 아래에서 world Plücker
  for (size_t l = 0; l < st.new_seg.size(); ++l) {
    std::vector<std::pair<double, double>> sr;  // (s, 1/z)
    for (int i = 0; i < kS; ++i) {
      double zm = 0;
      for (int k = 0; k < 3; ++k)
        if (const double v = z[(l * kS + i) * 3 + k]; v > 0 && (zm == 0 || v < zm)) zm = v;
      if (zm > 0) sr.emplace_back(0.1 + 0.8 * i / (kS - 1), 1 / zm);
    }
    if (sr.size() < 5) continue;
    std::vector<double> sl, ic;
    for (size_t i = 0; i < sr.size(); ++i)
      for (size_t j = i + 1; j < sr.size(); ++j) sl.push_back((sr[j].second - sr[i].second) / (sr[j].first - sr[i].first));
    std::nth_element(sl.begin(), sl.begin() + sl.size() / 2, sl.end());
    const double beta = sl[sl.size() / 2];
    for (const auto& [s, r] : sr) ic.push_back(r - beta * s);
    std::nth_element(ic.begin(), ic.begin() + ic.size() / 2, ic.end());
    const double alpha = ic[ic.size() / 2];
    std::vector<double> res;
    for (const auto& [s, r] : sr) res.push_back(std::abs(r - (alpha + beta * s)) / r);
    std::nth_element(res.begin(), res.begin() + res.size() / 2, res.end());
    const double r0 = alpha, r1 = alpha + beta;  // 선분 양 끝 (s = 0, 1) 역깊이
    if (res[res.size() / 2] > 0.1 || r0 <= 0 || r1 <= 0) continue;
    const Vec4& g = st.new_seg[l];
    ends.emplace_back(a_k / r0 * (vo_.cam.Kinv * Vec3(g(0), g(1), 1)), a_k / r1 * (vo_.cam.Kinv * Vec3(g(2), g(3), 1)));
    ids.push_back(st.new_lid[l]);
  }
  auto lk = take_map();
  const auto tb = Clock::now();
  const Pose& P = vo_.poses[st.kf];
  for (size_t n = 0; n < ids.size(); ++n) {
    const int j = ids[n];
    if (vo_.mature_l.count(j)) continue;
    const Vec3 A = P.R.transpose() * (ends[n].first - P.t), B = P.R.transpose() * (ends[n].second - P.t);
    const Vec3 d = B - A;
    Line L;
    L << d / d.norm(), A.cross(B) / d.norm();
    vo_.map_l[j] = L, vo_.da3_l.insert(j), ++n_da3_lines;
  }
  blocked(ms_since(tb));
}

KFState LoopCloser::snapshot(int kf) const {  // map lock 아래에서
  KFState st;
  st.kf = kf, st.pose = vo_.poses[kf];
  if (!kfs_.empty()) st.prev = kfs_.back(), st.prev_pose = vo_.poses[st.prev];
  std::vector<Vec3> d;
  std::vector<int> ids;
  const FrameObs& o = vo_.obs[kf];
  for (size_t m = 0; m < o.pid.size(); ++m)
    if (const auto it = vo_.map_p.find(o.pid[m]); it != vo_.map_p.end())
      if (const double z = (st.pose.R * it->second + st.pose.t)(2); z > 0)
        d.emplace_back(o.uv[m](0), o.uv[m](1), z), ids.push_back(o.pid[m]);
  // ORB 검증기는 맵 점 전부 (3D 연결), 서버로는 300 개만 보냄
  for (size_t m = 0, n = std::min<size_t>(3000, d.size()); m < n; ++m) {
    const int j = ids[m * d.size() / n];
    st.uvd.push_back(d[m * d.size() / n]), st.pid.push_back(j), st.mature.push_back(vo_.mature_p.count(j) > 0);
  }
  if (vo_.prm.da3_points)
    for (size_t m = 0; m < o.pid.size(); ++m)
      if (!vo_.mature_p.count(o.pid[m])) st.new_uv.push_back(o.uv[m]), st.new_pid.push_back(o.pid[m]);
  if (vo_.prm.da3_lines)
    for (size_t m = 0; m < o.lid.size(); ++m)
      if (!vo_.mature_l.count(o.lid[m])) st.new_seg.push_back(o.seg[m]), st.new_lid.push_back(o.lid[m]);
  if (vo_.prm.ba_lines && vo_.prm.ba_line_depth_sigma > 0)
    for (size_t m = 0; m < o.lid.size(); ++m)
      if (vo_.map_l.count(o.lid[m])) st.map_seg.push_back(o.seg[m]), st.map_lid.push_back(o.lid[m]);
  return st;
}

void LoopCloser::process(int kf, const cv::Mat& img, bool depth) {
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
  if (pr_) pr_->add(kf, img);
  if (prm_.fuse) {  // 맵 점 위치의 ORB 디스크립터 (KLT 점에 디스크립터를 붙임) + 대표 id 계산용 트랙 목록
    static thread_local cv::Ptr<cv::ORB> orb = cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    std::vector<cv::KeyPoint> kp;
    for (size_t m = 0, n = std::min<size_t>(1000, st.uvd.size()); m < n; ++m) {
      const size_t q = m * st.uvd.size() / n;
      kp.emplace_back(cv::Point2f(st.uvd[q](0), st.uvd[q](1)), 31.f, -1.f, 0.f, 0, int(q));
    }
    MapDesc& md = map_desc_[kf];
    orb->compute(to_gray(img), kp, md.des);  // 가장자리 키포인트는 빠짐 -> class_id 로 원래 점
    for (const auto& k : kp) md.pid.push_back(st.pid[k.class_id]), md.uv.emplace_back(k.pt.x, k.pt.y);
    kf_pids_[kf] = st.pid;
  }
  if (anchor_ && depth) {  // metric depth (DA3) vs VO 깊이 -> 이 키프레임의 VO 단위/미터
    anchor_->add(kf, img);
    const std::vector<Vec3> uvd = subsample(st.uvd, 300);
    std::vector<double> z;
    if (const auto av = anchor_->anchor(kf, uvd, &z)) {
      anchors_[kf] = *av, ++n_anchors;
      st.da3_scale = *av;
      if (prm_.da3_adaptive) {  // 성숙한 (삼각측량이 좋은) 점에서 DA3 잡음: 점별 = log(VO/DA3) 의 MAD, 프레임별 = log a_k 변화의 MAD/√2
        auto med = [](std::vector<double> v) { std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; };
        std::vector<double> lr;
        for (size_t m = 0; m < z.size(); ++m)
          if (z[m] > 0 && st.mature[m * st.uvd.size() / z.size()]) lr.push_back(std::log(uvd[m](2) / z[m]));
        if (lr.size() >= 20) {
          const double c = med(lr);
          for (double& x : lr) x = std::abs(x - c);
          sp_hist_.push_back(1.4826 * med(lr)), la_hist_.push_back(c);
          if (la_hist_.size() > 30) la_hist_.pop_front(), sp_hist_.pop_front();
          if (la_hist_.size() >= 5) {
            std::vector<double> dl;
            for (size_t q = 1; q < la_hist_.size(); ++q) dl.push_back(la_hist_[q] - la_hist_[q - 1]);
            const double cd = med(dl);
            for (double& x : dl) x = std::abs(x - cd);
            da3_sig_f_ = 1.4826 * med(dl) / std::sqrt(2.0);
            da3_sig_p_ = med(std::vector<double>(sp_hist_.begin(), sp_hist_.end()));
          }
        }
      }
      if (first_anchors_.size() < 10) {
        first_anchors_.push_back(*av);
        std::vector<double> f = first_anchors_;
        std::nth_element(f.begin(), f.begin() + f.size() / 2, f.end());
        if (first_anchors_.size() == 10) a_ref_ = f[f.size() / 2];
      }
      if (!st.new_uv.empty()) {  // pseudo RGB-D: 시차가 아직 작은 트랙을 DA3 깊이 (이 키프레임 VO 단위) 로 맵에. 같은 깊이맵 재사용
        const std::vector<double> zn = server_depth_kf(*anchor_, kf, st.new_uv);
        std::vector<std::pair<int, Vec3>> Xc;
        for (size_t m = 0; m < zn.size(); ++m)
          if (zn[m] > 0) Xc.emplace_back(st.new_pid[m], *av * zn[m] * (vo_.cam.Kinv * Vec3(st.new_uv[m](0), st.new_uv[m](1), 1)));
        auto l = take_map();
        const auto tb = Clock::now();
        const Pose& P = vo_.poses[kf];  // 지금 (BA 로 바뀌었을 수 있는) 키프레임 포즈 기준
        for (const auto& [j, x] : Xc)
          if (!vo_.mature_p.count(j)) vo_.map_p[j] = P.R.transpose() * (x - P.t), vo_.da3_p.insert(j), ++n_da3_points;
        blocked(ms_since(tb));
      }
      if (!st.new_seg.empty()) add_da3_lines(st, *av);
      if (vo_.prm.ba_depth_sigma > 0) {  // local BA 의 점 깊이 prior 용 (subsample 과 같은 순서로 점 id)
        std::vector<std::pair<int, float>> d;
        for (size_t m = 0; m < z.size(); ++m)
          if (z[m] > 0) d.emplace_back(st.pid[m * st.pid.size() / z.size()], float(z[m]));
        std::vector<VOTracker::LineDepth> ld;  // 라인 끝점 (안쪽 10 %: 끝점은 가림 경계일 때가 많음) DA3 깊이, 같은 깊이맵
        if (!st.map_seg.empty()) {
          std::vector<Vec2> uv;
          for (const Vec4& g : st.map_seg)
            uv.push_back(g.head<2>() + 0.1 * (g.tail<2>() - g.head<2>())), uv.push_back(g.head<2>() + 0.9 * (g.tail<2>() - g.head<2>()));
          const std::vector<double> zl = server_depth_kf(*anchor_, kf, uv);
          for (size_t m = 0; m < st.map_seg.size() && 2 * m + 1 < zl.size(); ++m)
            if (zl[2 * m] > 0 || zl[2 * m + 1] > 0)
              ld.push_back({st.map_lid[m], uv[2 * m], uv[2 * m + 1], float(zl[2 * m]), float(zl[2 * m + 1])});
        }
        auto l = take_map();
        const auto tb = Clock::now();
        vo_.da3[kf] = std::move(d);
        vo_.da3_ref = a_ref_;
        if (!ld.empty()) vo_.da3_lines[kf] = std::move(ld);
        if (prm_.da3_adaptive && da3_sig_p_ > 0) vo_.da3_sigma = std::max(vo_.prm.ba_depth_sigma, 2.5 * da3_sig_p_);
        blocked(ms_since(tb));
      }
    }
  }
  int a = -1;
  std::optional<LoopEdge> best;
  if (ver_) {
    a = ver_->add(st, img);
    if (a >= 0) {
      std::vector<int> cands;
      for (int j : ready_)
        if (a - j >= prm_.min_gap) cands.push_back(j);
      ready_.push_back(a);
      if (!cands.empty()) {
        ++n_queries;
        for (int j : pr_->query(a, cands, prm_.shortlist)) {
          if (prm_.fuse && shared(a, j) >= prm_.covis_min) {  // 융합으로 이미 연결된 곳 (ORB-SLAM3 DetectNBestCandidates)
            ++n_excluded;
            continue;
          }
          // 스케일 비가 1/3..3 밖이면 버림: 퇴화된 Sim(3) (EuRoC V1_01: 95-136 배 -> PGO 가 맵을 무너뜨림). KITTI 참 루프 0.66-1.53
          // 로컬 3D 루프가 있으면 그것 (더 정확), 같은 종류끼리는 인라이어 많은 것
          auto e = ver_->verify(a, j);
          if (e && (e->S.s < 1.0 / 3 || e->S.s > 3)) e.reset(), ++n_bad_scale;
          if (e && (!best || (best->da3 && !e->da3) || (best->da3 == e->da3 && e->inliers > best->inliers))) best = e;
        }
      }
    }
  }
  // DA3 3D 루프는 새 재방문의 첫 연결용: 직전 루프 엣지와 kDa3Gap 프레임 안이면 버림 (06: 재방문 구간마다 수십 개가 쌓여 PGO 가 흔들림)
  constexpr int kDa3Gap = 100;
  if (best && best->da3 && !edges_.empty() && best->i - edges_.back().i < kDa3Gap) best.reset();
  ms_detect += ms_since(t0);
  // ORB-SLAM3 식 연속 일관성 (mnLoopNumCoincidences): 연속으로 질의한 키프레임들이 같은 장소를 가리켜야 채택.
  // 앞의 검출은 보류했다가 확정되면 같이 넣음, 확정된 사슬이 이어지는 동안은 바로 채택
  if (prm_.consist_n > 1 && a >= 0) {
    const bool chained = !pending_.empty() && a - pending_.back().i <= prm_.consist_frames &&
                         best && std::abs(best->j - pending_.back().j) <= prm_.consist_frames;
    if (best) {
      if (!chained) n_unconfirmed += pending_.size() < size_t(prm_.consist_n) ? pending_.size() : 0, pending_.clear();
      pending_.push_back(*best);
      if (int(pending_.size()) < prm_.consist_n) {
        best.reset();
      } else if (int(pending_.size()) == prm_.consist_n) {  // 방금 확정: 보류했던 것까지 그래프에
        for (size_t q = 0; q + 1 < pending_.size(); ++q) loops.push_back(pending_[q]), edges_.push_back(pending_[q]);
      }
    } else if (!pending_.empty() && a - pending_.back().i > prm_.consist_frames) {
      if (int(pending_.size()) < prm_.consist_n) n_unconfirmed += pending_.size();
      pending_.clear();
    }
  }
  bool anchor_due = a_ref_ > 0 && ++since_pgo_ >= prm_.anchor_every;
  if (anchor_due && prm_.anchor_min_dev > 0) {  // 최근 앵커가 지금 스케일과 거의 맞으면 PGO 생략 (좋은 구간에 노이즈를 안 넣게)
    std::vector<double> dev;
    for (auto it = kfs_.rbegin(); it != kfs_.rend() && int(dev.size()) < since_pgo_; ++it)
      if (const auto f = anchors_.find(*it); f != anchors_.end()) dev.push_back(std::log(f->second / a_ref_));
    std::nth_element(dev.begin(), dev.begin() + dev.size() / 2, dev.end());
    if (!dev.empty() && std::abs(dev[dev.size() / 2]) < prm_.anchor_min_dev) anchor_due = false, since_pgo_ = 0, ++n_anchor_skip;
  }
  if (!best && !anchor_due) return;

  t0 = Clock::now();
  if (best) loops.push_back(*best), edges_.push_back(*best);
  const int upto = best ? a : kf;  // 루프: 키프레임 a 까지, 앵커만: 지금 키프레임까지
  std::vector<int> sub(kfs_.begin(), std::find(kfs_.begin(), kfs_.end(), upto) + 1);
  std::vector<Pose> P;  // PGO 초기값 (옛 키프레임 포즈는 루프 스레드만 바꾸므로 써넣을 때까지 그대로)
  {
    auto l = take_map();
    const auto tb = Clock::now();
    for (int f : sub) P.push_back(vo_.poses[f]);
    blocked(ms_since(tb));
  }
  // 새 루프 엣지를 지금 포즈로 평가 (EdgeSim3 와 같은 error). 이미 맞으면 엣지만 남기고 PGO 생략 -
  // ORB-SLAM3 는 이미 연결된 (covisible) 키프레임을 후보에서 빼서 재방문 구간에서 같은 루프를 다시 풀지 않음
  double chi2_new = 0;
  if (best) {
    const auto at_sub = [&](int f) { return P[std::find(sub.begin(), sub.end(), f) - sub.begin()]; };
    const Pose Pa = at_sub(a), Pj = at_sub(best->j);
    const Eigen::Matrix<double, 7, 1> r =
        (g2o::Sim3(best->S.R, best->S.t, best->S.s) * g2o::Sim3(Pj.R, Pj.t, 1.0) * g2o::Sim3(Pa.R, Pa.t, 1.0).inverse()).log();
    chi2_new = r.dot(loop_info(*best) * r);
  }
  if (best && !anchor_due && chi2_new < prm_.skip_chi2) {
    ++n_pgo_skipped;
    std::printf("    loop @frame %d <-> %d: %d Sim(3) inliers, scale ratio %.2f, edge chi2 %.1f -> consistent, PGO skipped\n",
                a, best->j, best->inliers, best->S.s, chi2_new);
    ms_pgo += ms_since(t0);
    return;
  }
  double cost = 0;
  const auto S_kf = pgo(P, sub, cost);  // lock 없이: 트래킹은 계속
  since_pgo_ = 0;
  if (!best) ++n_anchor_pgo;
  // PGO 뒤에도 새 루프 엣지가 크게 안 맞으면 (오도메트리 + 앵커와 모순) 잘못된 루프 -> 보정 안 하고 엣지 제거.
  // 참 루프는 큰 드리프트여도 오도메트리가 휘어서 흡수 (EuRoC V1_01: 사후 비용 6.9e9 인 루프가 스케일을 1만 배로 터뜨림)
  double chi2_post = 0;
  if (best) {
    const auto sim = [&](int f) { const Sim3& S = S_kf.at(f); return g2o::Sim3(S.R, S.t, S.s); };
    const Eigen::Matrix<double, 7, 1> r = (g2o::Sim3(best->S.R, best->S.t, best->S.s) * sim(best->j) * sim(a).inverse()).log();
    chi2_post = r.dot(loop_info(*best) * r);
    if (!(chi2_post < kMaxLoopChi2Post)) {
      edges_.erase(std::remove_if(edges_.begin(), edges_.end(), [&](const LoopEdge& e) { return e.i == a && e.j == best->j; }),
                   edges_.end());
      loops.pop_back(), ++n_rejected_pgo;
      std::printf("    loop @frame %d <-> %d: %d Sim(3) inliers, scale ratio %.2f, edge chi2 %.1f -> after PGO %.3g: rejected\n", a,
                  best->j, best->inliers, best->S.s, chi2_new, chi2_post);
      ms_pgo += ms_since(t0);
      return;
    }
  }
  int at = -1;
  if (prm_.feedback) {
    auto l = take_map();
    const auto tb = Clock::now();
    correct(S_kf, sub);
    if (best && prm_.fuse) {  // 옛 키프레임 j 와 그 앞뒤 키프레임까지 (연결을 넓게)
      const auto it = std::lower_bound(kfs_.begin(), kfs_.end(), best->j);
      for (auto q = std::max(kfs_.begin(), it - prm_.fuse_nb); q < kfs_.end() && q <= it + prm_.fuse_nb && *q < a; ++q)
        n_fused += fuse(a, *q);
    }
    at = int(vo_.poses.size()) - 1;
    ++n_corrections;
    ++vo_.map_version;
    blocked(ms_since(tb));
  }
  if (best) {
    std::printf("    loop @frame %d <-> %d: %d Sim(3) inliers, scale ratio %.2f, edge chi2 %.1f -> PGO cost %.1f (edge %.1f)", a,
                best->j, best->inliers, best->S.s, chi2_new, cost, chi2_post);
    if (at >= 0) std::printf(" -> map corrected (tracking at frame %d)", at);
    {  // 측정한 상대 포즈 (GT 와 비교용): 회전 (축각 벡터, rad), 이동 (cam_j -> cam_i, VO 단위), 스케일
      const Eigen::AngleAxisd aa(best->S.R);
      const Vec3 w = aa.angle() * aa.axis();
      std::printf(" | S w %.5f %.5f %.5f t %.4f %.4f %.4f s %.4f", w(0), w(1), w(2), best->S.t(0), best->S.t(1), best->S.t(2),
                  best->S.s);
    }
    if (best->H.trace() > 0) {  // 재투영 정제가 주는 측정 표준편차 (1 px 잡음 가정): 정보행렬 보정용
      const Eigen::Matrix<double, 7, 7> C = best->H.inverse();
      std::printf(" | refine sigma r %.4f t %.4f s %.4f", std::sqrt(C.diagonal().head<3>().mean()),
                  std::sqrt(C.diagonal().segment<3>(3).mean()), std::sqrt(C(6, 6)));
    }
    std::printf("\n");
  }
  ms_pgo += ms_since(t0);
}

std::unordered_map<int, Sim3> LoopCloser::pgo(const std::vector<Pose>& P, const std::vector<int>& sub, double& cost) {
  // 정점 = 키프레임 Sim(3) (world -> cam), 초기값 = P (sub 순서). 엣지 = 보관된 측정 (없는 오도메트리는 P 로)
  auto pose_sim3 = [&](size_t n) { return Sim3{1.0, P[n].R, P[n].t}; };
  for (size_t n = 0; n + 1 < sub.size(); ++n)
    if (!odom_.count(sub[n]) || n + 1 + prm_.odom_refresh >= sub.size()) odom_[sub[n]] = pose_sim3(n + 1) * pose_sim3(n).inverse();

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
  const Eigen::Matrix<double, 7, 7> info_odom = info(0.005, prm_.odom_sigma_t, prm_.odom_sigma_s);
  for (size_t n = 0; n + 1 < sub.size(); ++n) add_edge(sub[n], sub[n + 1], odom_[sub[n]], info_odom);
  for (const auto& e : edges_)
    if (v.count(e.i) && v.count(e.j)) add_edge(e.j, e.i, e.S, loop_info(e));
  if (a_ref_ > 0)  // metric depth 앵커: log s_k = log a_k - log a_ref (보정 뒤 VO 단위/미터 a_k / s_k = a_ref)
    for (const auto& [f, ak] : anchors_)
      if (v.count(f) && f != sub.front()) {
        auto* e = new EdgeScalePrior();
        e->setVertex(0, v[f]);
        e->setMeasurement(std::log(ak / a_ref_));
        const double sa = prm_.da3_adaptive ? std::max(prm_.anchor_sigma, 1.3 * da3_sig_f_) : prm_.anchor_sigma;  // KITTI σ_f ~0.06 -> 0.08
        e->setInformation(Eigen::Matrix<double, 1, 1>::Constant(1 / (sa * sa)));
        auto* rk = new g2o::RobustKernelHuber();
        rk->setDelta(2.0);  // 2 sigma 넘는 앵커 (가림, 하늘, 차) 는 선형으로
        e->setRobustKernel(rk);
        opt.addEdge(e);
      }
  opt.initializeOptimization();
  const auto t0 = Clock::now();
  opt.optimize(20);
  if (prm_.prune_ratio > 0 && edges_.size() >= 3) {  // 다른 루프/오도메트리와 안 맞는 루프 엣지는 빼고 한 번 더
    std::vector<std::pair<double, g2o::EdgeSim3*>> le;
    for (auto* e : opt.edges())
      if (auto* x = dynamic_cast<g2o::EdgeSim3*>(e); x && x->information()(0, 0) == kInfoLoop(0, 0)) {
        x->computeError();
        le.emplace_back(x->chi2(), x);
      }
    std::vector<double> c;
    for (auto& [chi, x] : le) c.push_back(chi);
    std::nth_element(c.begin(), c.begin() + c.size() / 2, c.end());
    const double thr = std::max(100.0, prm_.prune_ratio * c[c.size() / 2]);
    std::vector<std::pair<int, int>> bad;  // (j, i) 프레임
    for (auto& [chi, x] : le)
      if (chi > thr)
        bad.emplace_back(sub[x->vertex(0)->id()], sub[x->vertex(1)->id()]), opt.removeEdge(x);
    if (!bad.empty()) {
      for (auto [j, i] : bad)
        edges_.erase(std::remove_if(edges_.begin(), edges_.end(), [&](const LoopEdge& e) { return e.i == i && e.j == j; }),
                     edges_.end());
      n_pruned += bad.size();
      opt.initializeOptimization();
      opt.optimize(20);
    }
  }
  ms_opt += ms_since(t0);
  cost = opt.chi2();
  std::unordered_map<int, Sim3> out;
  for (auto& [f, x] : v) {
    const g2o::Sim3& S = x->estimate();
    out[f] = Sim3{S.scale(), S.rotation().toRotationMatrix(), S.translation()};
  }
  return out;
}

int LoopCloser::root(int p) {
  int r = p;
  for (auto it = canon_.find(r); it != canon_.end(); it = canon_.find(r)) r = it->second;
  for (auto it = canon_.find(p); it != canon_.end() && it->second != r; it = canon_.find(p)) {  // 경로 압축
    const int nx = it->second;
    it->second = r, p = nx;
  }
  return r;
}

int LoopCloser::shared(int a, int j) {
  const auto ia = kf_pids_.find(a), ij = kf_pids_.find(j);
  if (ia == kf_pids_.end() || ij == kf_pids_.end()) return 0;
  std::unordered_set<int> s;
  for (int p : ia->second) s.insert(root(p));
  int n = 0;
  for (int p : ij->second) n += s.count(root(p));
  return n;
}

int LoopCloser::fuse(int a, int j) {
  // 보정 뒤 (map lock 아래): 키프레임 a 의 맵 점 디스크립터 <-> 옛 키프레임 j 의 것. 옛 점을 a 에 투영해서 가까우면 같은 점 ->
  // 현재 트랙의 3D 를 옛 점 위치로 (고정), 대표 id 기록. 그 뒤 트래킹은 옛 맵을 재사용하고, 이 트랙을 보는 키프레임은 j 와 연결됨
  const auto ia = map_desc_.find(a), ij = map_desc_.find(j);
  if (ia == map_desc_.end() || ij == map_desc_.end() || ia->second.des.empty() || ij->second.des.empty()) return 0;
  const MapDesc &A = ia->second, &J = ij->second;
  std::vector<std::vector<cv::DMatch>> mm;
  cv::BFMatcher(cv::NORM_HAMMING).knnMatch(A.des, J.des, mm, 2);
  const Pose& Pa = vo_.poses[a];
  int n = 0;
  for (const auto& x : mm) {
    if (x.size() < 2 || x[0].distance > 64 || x[0].distance > 0.8 * x[1].distance) continue;
    const int pa = A.pid[x[0].queryIdx], pj = J.pid[x[0].trainIdx];
    const auto ita = vo_.map_p.find(pa), itj = vo_.map_p.find(pj);
    if (ita == vo_.map_p.end() || itj == vo_.map_p.end() || root(pa) == root(pj)) continue;
    const Vec3 q = vo_.cam.K * (Pa.R * itj->second + Pa.t);
    if (q(2) <= 0 || (q.head<2>() / q(2) - A.uv[x[0].queryIdx]).norm() > prm_.fuse_px) continue;
    canon_[root(pa)] = root(pj);
    ita->second = itj->second;
    vo_.mature_p.insert(pa);
    ++n;
  }
  return n;
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
  if (ver_)
    for (int f : kfs_) ver_->rescale(f, scale[f]);
  for (auto& [f, ak] : anchors_) ak /= scale[f];  // 카메라 좌표가 1/s 배 -> VO 깊이 / 미터도 1/s 배
  // 보관된 측정 (cam_from -> cam_to): S' = (1/s_to) S (s_from)
  auto rescale = [&](const Sim3& S, int from, int to) { return Sim3{S.s * scale[from] / scale[to], S.R, S.t / scale[to]}; };
  for (auto& [a, M] : odom_) {
    const auto it = std::upper_bound(kfs_.begin(), kfs_.end(), a);
    if (it != kfs_.end()) M = rescale(M, a, *it);
  }
  for (auto& e : edges_) e.S = rescale(e.S, e.j, e.i);
}

}  // namespace vp
