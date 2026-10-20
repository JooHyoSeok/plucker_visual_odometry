#include "vo_plucker/optimizer.h"

#include "vo_plucker/g2o_line.h"

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/edge_project_xyz.h>
#include <g2o/types/sba/edge_project_xyz_onlypose.h>
#include <g2o/types/sba/edge_se3_expmap.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>

#include <chrono>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace vp {

namespace {
// 고정된 대응 (직전 / 현재 프레임 정규화 좌표) 의 Sampson epipolar 오차 [px]. 정점 = 현재 T_cw, 직전 T_cw 는 고정.
// Jacobian 은 g2o 기본 (수치 미분)
class EdgeEpipolar : public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap> {
 public:
  Vec3 x0, x1;
  g2o::SE3Quat T_prev_inv;
  double f = 1;
  void computeError() override {
    const g2o::SE3Quat Tr = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0])->estimate() * T_prev_inv;
    const Vec3 t = Tr.translation();
    if (t.norm() < 1e-12) {
      _error[0] = 0;
      return;
    }
    const Mat3 E = skew(t.normalized()) * Tr.rotation().toRotationMatrix();
    const Vec3 Ex0 = E * x0, Etx1 = E.transpose() * x1;
    _error[0] = f * x1.dot(Ex0) / std::sqrt(Ex0.head<2>().squaredNorm() + Etx1.head<2>().squaredNorm() + 1e-18);
  }
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

// 점의 키프레임 깊이 prior: log z_c - log(측정 깊이). 정점 = 점, 키프레임 T_cw. Jacobian 은 g2o 기본 (수치 미분)
class EdgeLogDepth : public g2o::BaseBinaryEdge<1, double, g2o::VertexPointXYZ, g2o::VertexSE3Expmap> {
 public:
  void computeError() override {
    const Vec3 Xc = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1])->estimate().map(
        static_cast<const g2o::VertexPointXYZ*>(_vertices[0])->estimate());
    _error[0] = std::log(std::max(Xc(2), 1e-6)) - _measurement;
  }
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};
}  // namespace

Pose hybrid_ba(const Camera& c, const Pose& init, const MapObs& m, const Pose& prev,
               const std::vector<std::pair<Vec2, Vec2>>& tracks, double w_epi, int iters, double huber_px,
               bool fix_rotation) {
  if (fix_rotation) {  // 회전 고정: 이동 3-DoF 만 (g2o SE3 정점은 회전/이동 분리 고정이 안 돼서 작은 LM 으로)
    Vec3 t = init.t;
    auto cost = [&](const Vec3& tt, VecX* r) {
      const Pose p{init.R, tt};
      const VecX rm = residuals(c, p, MapObs{m.X, m.uv, {}, {}});
      const Mat3 Rr = init.R * prev.R.transpose();
      const Vec3 tr = tt - Rr * prev.t;
      VecX re(tracks.size());
      const Mat3 E = tr.norm() > 1e-12 ? Mat3(skew(tr.normalized()) * Rr) : Mat3::Zero();
      for (size_t i = 0; i < tracks.size(); ++i) {
        const Vec3 x0 = c.Kinv * Vec3(tracks[i].first(0), tracks[i].first(1), 1.0);
        const Vec3 x1 = c.Kinv * Vec3(tracks[i].second(0), tracks[i].second(1), 1.0);
        const Vec3 Ex0 = E * x0, Etx1 = E.transpose() * x1;
        re(i) = std::sqrt(w_epi) * c.fx * x1.dot(Ex0) / std::sqrt(Ex0.head<2>().squaredNorm() + Etx1.head<2>().squaredNorm() + 1e-18);
      }
      VecX all(rm.size() + re.size());
      all << rm, re;
      if (r) *r = all;
      double s = 0;  // Huber
      for (int i = 0; i < all.size(); ++i) {
        const double a = std::abs(all(i));
        s += a < huber_px ? 0.5 * a * a : huber_px * (a - 0.5 * huber_px);
      }
      return s;
    };
    VecX r;
    double f0 = cost(t, &r), lam = 1e-3;
    for (int it = 0; it < iters; ++it) {
      Eigen::MatrixXd J(r.size(), 3);
      for (int d = 0; d < 3; ++d) {
        Vec3 tp = t;
        tp(d) += 1e-7;
        VecX rp;
        cost(tp, &rp);
        J.col(d) = (rp - r) / 1e-7;
      }
      VecX w(r.size());
      for (int i = 0; i < r.size(); ++i) w(i) = std::abs(r(i)) < huber_px ? 1.0 : huber_px / std::abs(r(i));
      const Mat3 H = J.transpose() * w.asDiagonal() * J;
      const Vec3 g = J.transpose() * w.cwiseProduct(r);
      bool stepped = false;
      while (lam < 1e8) {
        const Vec3 dt = -(H + lam * Mat3(H.diagonal().asDiagonal())).ldlt().solve(g);
        VecX r1;
        const double f1 = cost(t + dt, &r1);
        if (f1 < f0) {
          t += dt, r = r1, f0 = f1, lam = std::max(lam / 3, 1e-7), stepped = true;
          break;
        }
        lam *= 4;
      }
      if (!stepped) break;
    }
    return Pose{init.R, t};
  }
  using Block = g2o::BlockSolver_6_3;
  g2o::SparseOptimizer opt;
  opt.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
      std::make_unique<Block>(std::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>())));
  auto* v = new g2o::VertexSE3Expmap();  // T_cw
  v->setId(0);
  v->setEstimate(g2o::SE3Quat(init.R, init.t));
  opt.addVertex(v);
  auto huber = [&]() {
    auto* rk = new g2o::RobustKernelHuber();
    rk->setDelta(huber_px);
    return rk;
  };
  for (size_t i = 0; i < m.X.size(); ++i) {
    auto* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
    e->setVertex(0, v);
    e->setMeasurement(m.uv[i]);
    e->setInformation(Eigen::Matrix2d::Identity());
    e->setRobustKernel(huber());
    e->fx = c.fx, e->fy = c.fy, e->cx = c.cx, e->cy = c.cy;
    e->Xw = m.X[i];
    opt.addEdge(e);
  }
  const g2o::SE3Quat Tp_inv = g2o::SE3Quat(prev.R, prev.t).inverse();
  for (const auto& [u0, u1] : tracks) {
    auto* e = new EdgeEpipolar();
    e->setVertex(0, v);
    e->x0 = c.Kinv * Vec3(u0(0), u0(1), 1.0), e->x1 = c.Kinv * Vec3(u1(0), u1(1), 1.0);
    e->T_prev_inv = Tp_inv, e->f = c.fx;
    e->setInformation(Eigen::Matrix<double, 1, 1>::Constant(w_epi));
    e->setRobustKernel(huber());
    opt.addEdge(e);
  }
  opt.initializeOptimization();
  opt.optimize(iters);
  const g2o::SE3Quat T = v->estimate();
  return Pose{T.rotation().toRotationMatrix(), T.translation()};
}

Pose motion_only_ba(const Camera& c, const Pose& init, const MapObs& m, int iters, double huber_px) {
  using Block = g2o::BlockSolver_6_3;
  g2o::SparseOptimizer opt;
  opt.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
      std::make_unique<Block>(std::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>())));

  auto* v = new g2o::VertexSE3Expmap();  // T_cw
  v->setId(0);
  v->setEstimate(g2o::SE3Quat(init.R, init.t));
  opt.addVertex(v);
  for (size_t i = 0; i < m.X.size(); ++i) {
    auto* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
    e->setVertex(0, v);
    e->setMeasurement(m.uv[i]);
    e->setInformation(Eigen::Matrix2d::Identity());
    auto* rk = new g2o::RobustKernelHuber();
    rk->setDelta(huber_px);
    e->setRobustKernel(rk);
    e->fx = c.fx, e->fy = c.fy, e->cx = c.cx, e->cy = c.cy;
    e->Xw = m.X[i];
    opt.addEdge(e);
  }
  opt.initializeOptimization();
  opt.optimize(iters);
  const g2o::SE3Quat T = v->estimate();
  return Pose{T.rotation().toRotationMatrix(), T.translation()};
}

struct LocalBAProblem {
  g2o::SparseOptimizer opt;
  std::vector<int> free;
  std::unordered_map<int, g2o::VertexSE3Expmap*> vk;
  std::unordered_map<int, g2o::SE3Quat> pose0;  // build 때 포즈 / 점 (BA 전 오차 계산용)
  std::unordered_map<int, Vec3> X0;
  struct Obs { g2o::EdgeSE3ProjectXYZ* e; int j, f; };
  struct LObs { EdgeLineProjection* e; int j, f; };
  std::vector<Obs> edges;
  std::vector<LObs> ledges;
  std::vector<std::pair<EdgeLogDepth*, int>> dedges;  // DA3 깊이 prior (엣지, 점 id)
  std::unordered_map<int, g2o::VertexPointXYZ*> vp;
  std::unordered_map<int, VertexLineOrth*> vl;
  LocalBAStats st;
  int version = 0;  // build 때 vo.map_version (루프 보정이 들어오면 apply 에서 버림)
};
void LocalBAProblemDeleter::operator()(LocalBAProblem* p) const { delete p; }

static constexpr double kBaChi2 = 5.991;  // 2 DoF, 95% (ORB-SLAM), 1 px 잡음 가정

LocalBAPtr local_ba_build(const VOTracker& vo, int window, int max_fixed, bool lines, bool mature_only) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<int>& kfs = vo.kfs;
  if (kfs.size() < 3) return nullptr;
  LocalBAPtr P(new LocalBAProblem);
  P->version = vo.map_version;
  const int n_win = std::min<int>(window, kfs.size());
  std::vector<int> free(kfs.end() - n_win, kfs.end());

  // 창 키프레임이 보는 맵 점
  std::unordered_set<int> pts;
  for (int f : free)
    for (int j : vo.obs[f].pid)
      if (vo.map_p.count(j) && (!mature_only || vo.mature_p.count(j))) pts.insert(j);
  // 고정 키프레임: 창 앞쪽에서 거꾸로, 같은 점을 보는 것
  std::vector<int> fixed;
  for (int i = int(kfs.size()) - n_win - 1; i >= 0 && int(fixed.size()) < max_fixed; --i) {
    const int f = kfs[i];
    for (int j : vo.obs[f].pid)
      if (pts.count(j)) {
        fixed.push_back(f);
        break;
      }
  }
  while (fixed.size() < 2 && free.size() > 2) fixed.push_back(free.front()), free.erase(free.begin());  // gauge
  P->free = free;

  g2o::SparseOptimizer& opt = P->opt;
  if (lines) {  // 정점 차원이 6 (포즈) / 3 (점) / 4 (라인) 로 섞임 -> 동적 블록
    using Block = g2o::BlockSolverX;
    opt.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<Block>(std::make_unique<g2o::LinearSolverEigen<Block::PoseMatrixType>>())));
  } else {
    using Block = g2o::BlockSolver_6_3;
    opt.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<Block>(std::make_unique<g2o::LinearSolverEigen<Block::PoseMatrixType>>())));
  }
  auto& vk = P->vk;
  auto add_kf = [&](int f, bool fix) {
    auto* v = new g2o::VertexSE3Expmap();
    v->setId(vk.size());
    P->pose0[f] = g2o::SE3Quat(vo.poses[f].R, vo.poses[f].t);
    v->setEstimate(P->pose0[f]);
    v->setFixed(fix);
    opt.addVertex(v);
    vk[f] = v;
  };
  for (int f : free) add_kf(f, false);
  for (int f : fixed) add_kf(f, true);

  const double huber = std::sqrt(kBaChi2);
  int next_id = vk.size();
  for (int j : pts) {
    std::vector<std::pair<int, const Vec2*>> seen;  // (키프레임, 관측)
    for (auto& [f, v] : vk)
      if (const Vec2* u = vo.obs[f].point(j); u && !vo.bad_obs.count(VOTracker::obs_key(f, j))) seen.emplace_back(f, u);
    if (seen.size() < 2) continue;
    auto* p = new g2o::VertexPointXYZ();
    p->setId(next_id++);
    P->X0[j] = vo.map_p.at(j);
    p->setEstimate(P->X0[j]);
    p->setMarginalized(true);
    opt.addVertex(p);
    P->vp[j] = p;
    for (auto& [f, u] : seen) {
      auto* e = new g2o::EdgeSE3ProjectXYZ();
      e->setVertex(0, p);
      e->setVertex(1, vk[f]);
      e->setMeasurement(*u);
      e->setInformation(Eigen::Matrix2d::Identity());
      auto* rk = new g2o::RobustKernelHuber();
      rk->setDelta(huber);
      e->setRobustKernel(rk);
      e->fx = vo.cam.fx, e->fy = vo.cam.fy, e->cx = vo.cam.cx, e->cy = vo.cam.cy;
      opt.addEdge(e);
      P->edges.push_back({e, j, f});
    }
  }
  // DA3 깊이 prior: 키프레임이 보는 점의 깊이 = da3_ref x DA3 깊이 (BA 가 창의 스케일을 흘리지 못하게)
  if (vo.prm.ba_depth_sigma > 0 && vo.da3_ref > 0)
    for (auto& [f, v] : vk)
      if (const auto it = vo.da3.find(f); it != vo.da3.end())
        for (const auto& [j, z] : it->second)
          if (const auto p = P->vp.find(j); p != P->vp.end()) {
            auto* e = new EdgeLogDepth();
            e->setVertex(0, p->second);
            e->setVertex(1, v);
            e->setMeasurement(std::log(vo.da3_ref * z));
            const double sg = vo.da3_sigma > 0 ? vo.da3_sigma : vo.prm.ba_depth_sigma;
            e->setInformation(Eigen::Matrix<double, 1, 1>::Constant(1 / (sg * sg)));
            auto* rk = new g2o::RobustKernelHuber();
            rk->setDelta(std::sqrt(3.841));  // 1 DoF 95 %
            e->setRobustKernel(rk);
            opt.addEdge(e);
            P->dedges.emplace_back(e, j);
          }
  P->st.depth_priors = P->dedges.size();
  // 라인: 창 키프레임이 보고, 창 + 고정 키프레임에서 2번 이상 관측된 맵 라인
  if (lines) {
    std::unordered_set<int> ls;
    for (int f : free)
      for (int j : vo.obs[f].lid)
        if (vo.map_l.count(j)) ls.insert(j);
    for (int j : ls) {
      std::vector<std::pair<int, const Vec4*>> seen;
      for (auto& [f, v] : vk)
        if (const Vec4* sg = vo.obs[f].line(j); sg && !vo.bad_lobs.count(VOTracker::obs_key(f, j))) seen.emplace_back(f, sg);
      if (seen.size() < 2) continue;
      // 퇴화 라인 제외: 창 안 첫 / 마지막 관측의 역투영 평면 각도가 작으면 (진행 방향을 향하는 라인 등)
      // 라인 위치가 거의 안 정해져서 포즈 회전을 끌고 감 (06: 라인 BA 만 켜면 마지막 방향 오차 1.3 -> 21 deg)
      const auto [fa, fb] = std::minmax_element(seen.begin(), seen.end());
      if (plane_angle_deg(vo.cam, vo.poses[fa->first], *fa->second, vo.poses[fb->first], *fb->second) < vo.prm.mature_deg)
        continue;
      auto* l = new VertexLineOrth();
      l->setId(next_id++);
      l->setEstimate(to_orth(vo.map_l.at(j)));
      l->setMarginalized(true);
      opt.addVertex(l);
      P->vl[j] = l;
      for (auto& [f, sg] : seen) {
        auto* e = new EdgeLineProjection();
        e->setVertex(0, l);
        e->setVertex(1, vk[f]);
        e->setMeasurement(*sg);
        e->setInformation(Eigen::Matrix2d::Identity());
        auto* rk = new g2o::RobustKernelHuber();
        rk->setDelta(huber);
        e->setRobustKernel(rk);
        e->KL = vo.cam.KL;
        opt.addEdge(e);
        P->ledges.push_back({e, j, f});
      }
    }
  }
  P->st.kfs = free.size(), P->st.fixed = fixed.size(), P->st.points = P->vp.size(), P->st.edges = P->edges.size();
  P->st.lines = P->vl.size();
  P->st.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (P->vp.empty()) return nullptr;
  return P;
}

void local_ba_solve(LocalBAProblem& P) {
  const auto t0 = std::chrono::steady_clock::now();
  auto rms = [&]() {  // 재투영 오차 RMS [px] (2차에서 쓰는 관측 = level 0, chi2는 Huber 전 값)
    double s = 0;
    int n = 0;
    for (auto& o : P.edges)
      if (o.e->level() == 0) o.e->computeError(), s += o.e->chi2(), ++n;
    return n ? std::sqrt(s / n) : 0.0;
  };
  P.opt.initializeOptimization();
  P.opt.optimize(5);
  for (auto& o : P.edges)  // 1차 후 큰 오차 / 카메라 뒤 관측은 빼고 Huber 없이 다시
    if (o.e->chi2() > kBaChi2 || !o.e->isDepthPositive()) o.e->setLevel(1);
    else o.e->setRobustKernel(nullptr);
  for (auto& o : P.ledges)  // 라인도 같은 기준 (끝점 거리 2개 = 2 DoF)
    if (o.e->computeError(), o.e->chi2() > kBaChi2) o.e->setLevel(1);
    else o.e->setRobustKernel(nullptr);
  {  // 재투영 인라이어가 하나도 안 남은 점은 깊이 prior 도 빼기 (prior 만으로는 점 3-DoF 가 안 정해짐)
    std::unordered_set<int> live;
    for (auto& o : P.edges)
      if (o.e->level() == 0) live.insert(o.j);
    for (auto& [e, j] : P.dedges)
      if (!live.count(j)) e->setLevel(1);
  }
  {  // BA 전 오차: 같은 (인라이어) 관측을 build 때 포즈/점으로 계산하고 1차 결과로 되돌림
    std::unordered_map<int, g2o::SE3Quat> pk;
    std::unordered_map<int, Vec3> pp;
    for (auto& [f, v] : P.vk) pk[f] = v->estimate(), v->setEstimate(P.pose0[f]);
    for (auto& [j, p] : P.vp) pp[j] = p->estimate(), p->setEstimate(P.X0[j]);
    P.st.rms_before = rms();
    for (auto& [f, v] : P.vk) v->setEstimate(pk[f]);
    for (auto& [j, p] : P.vp) p->setEstimate(pp[j]);
  }
  P.opt.initializeOptimization(0);
  P.opt.optimize(10);
  P.st.rms_after = rms();
  P.st.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

LocalBAStats local_ba_apply(VOTracker& vo, LocalBAProblem& P) {
  const auto t0 = std::chrono::steady_clock::now();
  LocalBAStats& st = P.st;
  if (vo.map_version != P.version) {  // BA 하는 동안 루프 보정이 맵을 바꿈 -> 낡은 결과는 버림
    st.discarded = true;
    return st;
  }
  // ---- write back ---- (BA 동안 트래킹이 지운 점/라인은 되살리지 않음)
  std::unordered_map<int, Pose> old;
  for (int f : P.free) {
    old[f] = vo.poses[f];
    const g2o::SE3Quat T = P.vk[f]->estimate();
    vo.poses[f] = Pose{T.rotation().toRotationMatrix(), T.translation()};
  }
  std::unordered_set<int> bad;
  std::unordered_map<int, int> good;
  for (auto& o : P.edges) {
    o.e->computeError();
    if (o.e->chi2() > kBaChi2 || !o.e->isDepthPositive()) {
      bad.insert(o.j);
      if (vo.prm.ba_drop_obs) vo.bad_obs.insert(VOTracker::obs_key(o.f, o.j));
    } else {
      ++good[o.j];
    }
  }
  if (vo.prm.ba_drop_obs)  // 나쁜 관측만 뺐으니 좋은 관측이 2개 이상 남은 점은 유지
    for (auto it = bad.begin(); it != bad.end();) it = good[*it] >= 2 ? bad.erase(it) : std::next(it);
  for (auto& [j, p] : P.vp) {
    if (!vo.map_p.count(j)) continue;
    if (bad.count(j)) {
      vo.map_p.erase(j), vo.mature_p.erase(j);
    } else {
      vo.map_p[j] = p->estimate();
      vo.mature_p.insert(j);  // BA가 맡은 점: 매 프레임 DLT 재삼각측량에서 제외
    }
  }
  st.outliers = bad.size();
  std::unordered_set<int> lbad;  // 라인도 점과 같이: 나쁜 관측만 빼고 좋은 관측이 2개 이상이면 유지
  std::unordered_map<int, int> lgood;
  for (auto& o : P.ledges) {
    o.e->computeError();
    if (o.e->chi2() > kBaChi2) {
      lbad.insert(o.j);
      if (vo.prm.ba_drop_obs) vo.bad_lobs.insert(VOTracker::obs_key(o.f, o.j));
    } else {
      ++lgood[o.j];
    }
  }
  if (vo.prm.ba_drop_obs)
    for (auto it = lbad.begin(); it != lbad.end();) it = lgood[*it] >= 2 ? lbad.erase(it) : std::next(it);
  for (auto& [j, l] : P.vl) {
    if (!vo.map_l.count(j)) continue;
    if (lbad.count(j)) {
      vo.map_l.erase(j), vo.mature_l.erase(j);
    } else {
      vo.map_l[j] = from_orth(l->estimate());
      vo.mature_l.insert(j);  // BA 가 맡은 라인: 재삼각측량에서 제외
    }
  }
  st.line_outliers = lbad.size();
  // 창 안 비키프레임 + BA 동안 트래킹한 프레임: 직전 (자유) 키프레임의 보정을 같이 (T_f <- T_f T_kf,old^-1 T_kf,new)
  const int k_last = int(vo.poses.size()) - 1;
  for (int f = P.free.front(); f <= k_last; ++f) {
    if (!vo.has_pose[f] || old.count(f)) continue;
    auto it = std::upper_bound(P.free.begin(), P.free.end(), f);
    if (it == P.free.begin()) continue;
    const int kf = *(it - 1);
    const Pose &ko = old[kf], &kn = vo.poses[kf];
    const Mat3 Rc = ko.R.transpose() * kn.R;
    const Vec3 tc = ko.R.transpose() * (kn.t - ko.t);
    vo.poses[f] = Pose{orthonormalize(vo.poses[f].R * Rc), vo.poses[f].R * tc + vo.poses[f].t};
  }
  st.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return st;
}

LocalBAStats global_ba(VOTracker& vo, const std::vector<LoopPose>& loops, bool lines) {
  LocalBAPtr P = local_ba_build(vo, vo.kfs.size(), 0, lines, false);
  if (!P) return LocalBAStats{};
  Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Zero();  // [회전, 이동], PGO 루프 엣지와 같은 잡음
  for (int k = 0; k < 3; ++k) I(k, k) = 1 / (0.01 * 0.01), I(3 + k, 3 + k) = 1 / (0.05 * 0.05);
  for (const LoopPose& l : loops) {
    const auto a = P->vk.find(l.j), b = P->vk.find(l.i);
    if (a == P->vk.end() || b == P->vk.end()) continue;
    auto* e = new g2o::EdgeSE3Expmap();  // error = log(T_i^-1 M T_j), M = T_i T_j^-1
    e->setVertex(0, a->second);
    e->setVertex(1, b->second);
    e->setMeasurement(g2o::SE3Quat(l.T.R, l.T.t));
    e->setInformation(I);
    P->opt.addEdge(e);
  }
  local_ba_solve(*P);
  return local_ba_apply(vo, *P);
}

LocalBAStats local_ba(VOTracker& vo, int window, int max_fixed, bool lines, bool mature_only) {
  LocalBAPtr P = local_ba_build(vo, window, max_fixed, lines, mature_only);
  if (!P) return LocalBAStats{};
  local_ba_solve(*P);
  return local_ba_apply(vo, *P);
}

}  // namespace vp
