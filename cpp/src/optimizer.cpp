#include "vo_plucker/optimizer.h"

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/edge_project_xyz.h>
#include <g2o/types/sba/edge_project_xyz_onlypose.h>
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

LocalBAStats local_ba(VOTracker& vo, int window, int max_fixed) {
  const auto t0 = std::chrono::steady_clock::now();
  LocalBAStats st;
  const std::vector<int>& kfs = vo.kfs;
  if (kfs.size() < 3) return st;
  const int n_win = std::min<int>(window, kfs.size());
  std::vector<int> free(kfs.end() - n_win, kfs.end());

  // 창 키프레임이 보는 맵 점
  std::unordered_set<int> pts;
  for (int f : free)
    for (int j : vo.obs[f].pid)
      if (vo.map_p.count(j)) pts.insert(j);
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

  using Block = g2o::BlockSolver_6_3;
  g2o::SparseOptimizer opt;
  opt.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
      std::make_unique<Block>(std::make_unique<g2o::LinearSolverEigen<Block::PoseMatrixType>>())));
  std::unordered_map<int, g2o::VertexSE3Expmap*> vk;
  auto add_kf = [&](int f, bool fix) {
    auto* v = new g2o::VertexSE3Expmap();
    v->setId(vk.size());
    v->setEstimate(g2o::SE3Quat(vo.poses[f].R, vo.poses[f].t));
    v->setFixed(fix);
    opt.addVertex(v);
    vk[f] = v;
  };
  for (int f : free) add_kf(f, false);
  for (int f : fixed) add_kf(f, true);

  const double chi2_th = 5.991, huber = std::sqrt(chi2_th);  // 2 DoF, 95% (ORB-SLAM), 1 px 잡음 가정
  struct Obs { g2o::EdgeSE3ProjectXYZ* e; int j; };
  std::vector<Obs> edges;
  std::unordered_map<int, g2o::VertexPointXYZ*> vp;
  int next_id = vk.size();
  for (int j : pts) {
    std::vector<std::pair<int, const Vec2*>> seen;  // (키프레임, 관측)
    for (auto& [f, v] : vk)
      if (const Vec2* u = vo.obs[f].point(j)) seen.emplace_back(f, u);
    if (seen.size() < 2) continue;
    auto* p = new g2o::VertexPointXYZ();
    p->setId(next_id++);
    p->setEstimate(vo.map_p[j]);
    p->setMarginalized(true);
    opt.addVertex(p);
    vp[j] = p;
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
      edges.push_back({e, j});
    }
  }
  st.kfs = free.size(), st.fixed = fixed.size(), st.points = vp.size(), st.edges = edges.size();
  if (vp.empty()) return st;

  // 재투영 오차 RMS [px] (2차에서 쓰는 관측 = level 0, chi2는 Huber 전 값)
  auto rms = [&]() {
    double s = 0;
    int n = 0;
    for (auto& o : edges)
      if (o.e->level() == 0) o.e->computeError(), s += o.e->chi2(), ++n;
    return n ? std::sqrt(s / n) : 0.0;
  };
  opt.initializeOptimization();
  opt.optimize(5);
  for (auto& o : edges)  // 1차 후 큰 오차 / 카메라 뒤 관측은 빼고 Huber 없이 다시
    if (o.e->chi2() > chi2_th || !o.e->isDepthPositive()) o.e->setLevel(1);
    else o.e->setRobustKernel(nullptr);
  {  // BA 전 오차: 같은 (인라이어) 관측을 원래 포즈/점으로 계산하고 1차 결과로 되돌림
    std::unordered_map<int, g2o::SE3Quat> pk;
    std::unordered_map<int, Vec3> pp;
    for (auto& [f, v] : vk) pk[f] = v->estimate(), v->setEstimate(g2o::SE3Quat(vo.poses[f].R, vo.poses[f].t));
    for (auto& [j, p] : vp) pp[j] = p->estimate(), p->setEstimate(vo.map_p[j]);
    st.rms_before = rms();
    for (auto& [f, v] : vk) v->setEstimate(pk[f]);
    for (auto& [j, p] : vp) p->setEstimate(pp[j]);
  }
  opt.initializeOptimization(0);
  opt.optimize(10);
  st.rms_after = rms();

  // ---- write back ----
  std::unordered_map<int, Pose> old;
  for (int f : free) {
    old[f] = vo.poses[f];
    const g2o::SE3Quat T = vk[f]->estimate();
    vo.poses[f] = Pose{T.rotation().toRotationMatrix(), T.translation()};
  }
  std::unordered_set<int> bad;
  for (auto& o : edges) {
    o.e->computeError();
    if (o.e->chi2() > chi2_th || !o.e->isDepthPositive()) bad.insert(o.j);
  }
  for (auto& [j, p] : vp) {
    if (bad.count(j)) {
      vo.map_p.erase(j), vo.mature_p.erase(j);
    } else {
      vo.map_p[j] = p->estimate();
      vo.mature_p.insert(j);  // BA가 맡은 점: 매 프레임 DLT 재삼각측량에서 제외
    }
  }
  st.outliers = bad.size();
  // 창 안 비키프레임: 직전 (자유) 키프레임의 보정을 같이 (T_f <- T_f T_kf,old^-1 T_kf,new)
  const int k_last = int(vo.poses.size()) - 1;
  for (int f = free.front(); f <= k_last; ++f) {
    if (!vo.has_pose[f] || old.count(f)) continue;
    auto it = std::upper_bound(free.begin(), free.end(), f);
    if (it == free.begin()) continue;
    const int kf = *(it - 1);
    const Pose &ko = old[kf], &kn = vo.poses[kf];
    const Mat3 Rc = ko.R.transpose() * kn.R;
    const Vec3 tc = ko.R.transpose() * (kn.t - ko.t);
    vo.poses[f] = Pose{orthonormalize(vo.poses[f].R * Rc), vo.poses[f].R * tc + vo.poses[f].t};
  }
  st.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return st;
}

}  // namespace vp
