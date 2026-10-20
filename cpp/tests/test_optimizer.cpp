// g2o motion-only BA vs the existing LM (estimate_pose) on synthetic data, points only (lines come with the line edge).
//   no outliers : Huber inactive -> plain least squares -> both must reach the same optimum
//   10% outliers: Huber differs (LM: per residual component, g2o: per point 2D norm) -> compare accuracy vs GT only
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

#include <g2o/core/jacobian_workspace.h>

#include "vo_plucker/g2o_line.h"
#include "vo_plucker/optimizer.h"

using namespace vp;

static bool trial_set(double outlier_ratio) {
  const Camera cam(707.0912, 707.0912, 601.8873, 183.1104, 1226, 370);
  std::mt19937 rng(0);
  std::normal_distribution<double> n01(0, 1);
  std::uniform_real_distribution<double> uni(0, 1);
  double max_dR = 0, max_dt = 0, t_lm = 0, t_g2o = 0, max_err_lm = 0, max_err_g2o = 0;
  const int trials = 200;
  for (int it = 0; it < trials; ++it) {
    const Pose gt{so3_exp(Vec3(n01(rng), n01(rng), n01(rng)) * 0.2), Vec3(n01(rng), n01(rng), n01(rng))};
    MapObs m;
    while (m.X.size() < 300) {
      const Vec3 Xc((uni(rng) - 0.5) * 20, (uni(rng) - 0.5) * 6, 3 + uni(rng) * 40);
      const Vec3 q = cam.K * Xc;
      Vec2 u = q.head<2>() / q(2);
      if (u(0) < 0 || u(0) >= cam.W || u(1) < 0 || u(1) >= cam.H) continue;
      u += 0.5 * Vec2(n01(rng), n01(rng));
      if (uni(rng) < outlier_ratio) u += Vec2(n01(rng), n01(rng)) * 30;  // 이상치
      m.X.push_back(gt.R.transpose() * (Xc - gt.t)), m.uv.push_back(u);
    }
    const Pose init{so3_exp(Vec3(n01(rng), n01(rng), n01(rng)) * 0.01) * gt.R, gt.t + 0.2 * Vec3(n01(rng), n01(rng), n01(rng))};
    auto t0 = std::chrono::steady_clock::now();
    const Pose a = estimate_pose(cam, init, m);
    t_lm += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now();
    const Pose b = motion_only_ba(cam, init, m);
    t_g2o += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    max_dR = std::max(max_dR, (a.R - b.R).cwiseAbs().maxCoeff());
    max_dt = std::max(max_dt, (a.t - b.t).norm());
    max_err_lm = std::max(max_err_lm, (center(a) - center(gt)).norm());
    max_err_g2o = std::max(max_err_g2o, (center(b) - center(gt)).norm());
  }
  std::printf("%d trials, 300 points, %.0f%% outliers: LM %.2f ms/call, g2o %.2f ms/call\n", trials, 100 * outlier_ratio,
              t_lm / trials, t_g2o / trials);
  std::printf("  LM vs g2o: max |dR| %.2e, max |dt| %.2e | max center error vs GT: LM %.4f, g2o %.4f\n", max_dR, max_dt,
              max_err_lm, max_err_g2o);
  const bool ok = outlier_ratio == 0 ? max_dR < 1e-6 && max_dt < 1e-5 : max_err_g2o < 1.5 * max_err_lm + 1e-3;
  std::printf("  %s\n", ok ? (outlier_ratio == 0 ? "OK: same optimum" : "OK: g2o as accurate as LM") : "MISMATCH");
  return ok;
}

// hybrid_ba: 전방 주행 + yaw 회전 (회전 구간), 정확한 맵 (점 50개) + 많은 트랙 (600개). 2D-2D 처럼 회전 0.05 deg,
// 이동 방향 3 deg 틀린 초기값에서 GT 로 돌아와야 함 (50회 중앙값; 노이즈 0.5 px 에서 트랙만의 하한이 ~0.15 deg)
static bool hybrid_check() {
  const Camera cam(707.0912, 707.0912, 601.8873, 183.1104, 1226, 370);
  std::mt19937 rng(1);
  std::normal_distribution<double> n01(0, 1);
  std::uniform_real_distribution<double> uni(0, 1);
  auto deg = [](const Mat3& R) { return std::acos(std::clamp((R.trace() - 1) / 2, -1.0, 1.0)) * 180 / M_PI; };
  std::vector<double> er, et;
  double init_r = 0, init_t = 0;
  for (int it = 0; it < 50; ++it) {
    const Pose prev{so3_exp(Vec3(n01(rng), n01(rng), n01(rng)) * 0.3), Vec3(n01(rng), n01(rng), n01(rng)) * 5};  // 임의 직전 포즈
    const Mat3 Rr = so3_exp(Vec3(0, 3.0 * M_PI / 180, 0));   // yaw 3 deg / frame
    const Vec3 tr = Vec3(0.02, 0, -0.5);                      // cam_{k-1} -> cam_k, 0.5 m forward (t = -C)
    const Pose cur{Rr * prev.R, Rr * prev.t + tr};
    auto project = [&](const Pose& p, const Vec3& Xw, Vec2& u) {
      const Vec3 q = cam.K * (p.R * Xw + p.t);
      if (q(2) <= 0.1) return false;
      u = q.head<2>() / q(2) + 0.5 * Vec2(n01(rng), n01(rng));
      return u(0) >= 0 && u(0) < cam.W && u(1) >= 0 && u(1) < cam.H;
    };
    MapObs m;
    std::vector<std::pair<Vec2, Vec2>> tracks;
    while (tracks.size() < 600) {
      const Vec3 Xw = prev.R.transpose() * (Vec3((uni(rng) - 0.5) * 40, (uni(rng) - 0.5) * 8, 4 + uni(rng) * 40) - prev.t);
      Vec2 u0, u1;
      if (!project(prev, Xw, u0) || !project(cur, Xw, u1)) continue;
      tracks.emplace_back(u0, u1);
      if (m.X.size() < 50) m.X.push_back(Xw), m.uv.push_back(u1);
    }
    // 2D-2D 같은 초기값: 회전 0.05 deg, 이동 방향을 옆으로 3 deg 틀림 (크기는 맞음)
    const Mat3 Ri = so3_exp(Vec3(0, -0.05 * M_PI / 180, 0)) * Rr;
    const Pose init{Ri * prev.R, Ri * prev.t + so3_exp(Vec3(0, 3.0 * M_PI / 180, 0)) * tr};
    const Pose h = hybrid_ba(cam, init, m, prev, tracks);
    auto rel = [&](const Pose& p) { return Pose{p.R * prev.R.transpose(), p.t - p.R * prev.R.transpose() * prev.t}; };
    auto errs = [&](const Pose& p) {
      const Pose q = rel(p);
      return std::pair(deg(q.R * Rr.transpose()),
                       std::acos(std::clamp(q.t.normalized().dot(tr.normalized()), -1.0, 1.0)) * 180 / M_PI);
    };
    init_r = errs(init).first, init_t = errs(init).second;
    er.push_back(errs(h).first), et.push_back(errs(h).second);
  }
  std::nth_element(er.begin(), er.begin() + 25, er.end()), std::nth_element(et.begin(), et.begin() + 25, et.end());
  std::printf("hybrid_ba (yaw 3 deg/frame, 50 map points, 600 tracks): rotation error %.3f -> %.4f deg, "
              "translation direction error %.2f -> %.2f deg (median of 50)\n", init_r, er[25], init_t, et[25]);
  const bool ok = er[25] < 0.02 && et[25] < 0.5;
  std::printf("  %s\n", ok ? "OK: hybrid recovers rotation and direction" : "FAIL");
  return ok;
}

// EdgeLineProjection: 해석 Jacobian vs 중앙 차분 (라인 4, 포즈 6), orthonormal <-> Plücker 왕복, 기존 project_line 과 같은 오차
static bool line_edge_check() {
  const Camera cam(707.0912, 707.0912, 601.8873, 183.1104, 1226, 370);
  std::mt19937 rng(2);
  std::normal_distribution<double> n01(0, 1);
  std::uniform_real_distribution<double> uni(0, 1);
  double worst_j = 0, worst_rt = 0, worst_res = 0;
  for (int it = 0; it < 200; ++it) {
    Pose T{so3_exp(Vec3(n01(rng), n01(rng), n01(rng)) * 0.3), Vec3(n01(rng), n01(rng), n01(rng)) * 2};
    const bool far = it % 2;  // 홀수: 카메라가 세계 원점에서 멀리 (KITTI 처럼), 라인 원점 c = 카메라 중심 근처 (local BA 처럼)
    const Vec3 Cf = Vec3(n01(rng), n01(rng), n01(rng)) * 300;
    if (far) T.t = -T.R * Cf;
    const Vec3 pc((uni(rng) - 0.5) * 10, (uni(rng) - 0.5) * 4, 5 + uni(rng) * 25);  // 카메라 앞 점
    const Vec3 dc = Vec3(n01(rng), n01(rng), n01(rng) * 0.3).normalized();
    const Vec3 pw = T.R.transpose() * (pc - T.t), dw = T.R.transpose() * dc;
    Line L;
    L << dw, pw.cross(dw);
    auto proj = [&](const Vec3& X) { const Vec3 q = cam.K * (T.R * X + T.t); return Vec2(q.head<2>() / q(2)); };
    const Vec2 a = proj(pw - 2 * dw) + Vec2(n01(rng), n01(rng)), b = proj(pw + 2 * dw) + Vec2(n01(rng), n01(rng));
    const Vec4 seg(a(0), a(1), b(0), b(1));
    const Line L2 = from_orth(to_orth(L));  // 왕복: 방향 부호까지 같아야 (d.m = 0, |d| = 1)
    worst_rt = std::max(worst_rt, (L2 - L).cwiseAbs().maxCoeff());

    const Vec3 c = far ? Vec3(Cf + Vec3(n01(rng), n01(rng), n01(rng)) * 2) : Vec3::Zero();
    worst_rt = std::max(worst_rt, (shift_line(shift_line(L, c), -c) - L).cwiseAbs().maxCoeff());
    auto* vl = new VertexLineOrth();
    vl->setEstimate(to_orth(shift_line(L, c)));
    auto* vp = new g2o::VertexSE3Expmap();
    vp->setEstimate(g2o::SE3Quat(T.R, T.t));
    EdgeLineProjection e;
    e.setVertex(0, vl), e.setVertex(1, vp);
    e.setMeasurement(seg);
    e.KL = cam.KL, e.c = c;
    e.computeError();
    MapObs mo;
    mo.L.push_back(L), mo.seg.push_back(seg);
    worst_res = std::max(worst_res, (e.error() - residuals(cam, T, mo)).cwiseAbs().maxCoeff());  // 기존 선 오차와 같음
    g2o::JacobianWorkspace ws;  // 고정 크기 엣지의 Jacobian 은 작업공간 메모리를 가리키는 Map (최적화기가 보통 할당)
    ws.updateSize(&e);
    ws.allocate();
    e.linearizeOplus(ws);
    const Eigen::Matrix<double, 2, 4> Jl = e.J_line();
    const Eigen::Matrix<double, 2, 6> Jp = e.J_pose();
    auto numeric = [&](g2o::OptimizableGraph::Vertex* v, int dim, Eigen::MatrixXd& J) {
      J.resize(2, dim);
      for (int k = 0; k < dim; ++k) {
        double d[6] = {0, 0, 0, 0, 0, 0};
        const double h = 1e-6;
        v->push(), d[k] = h, v->oplus(d), e.computeError();
        const Vec2 ep = e.error();
        v->pop(), v->push(), d[k] = -h, v->oplus(d), e.computeError();
        J.col(k) = (ep - e.error()) / (2 * h);
        v->pop();
      }
    };
    Eigen::MatrixXd Nl, Np;
    numeric(vl, 4, Nl), numeric(vp, 6, Np);
    const double scale = std::max({Nl.cwiseAbs().maxCoeff(), Np.cwiseAbs().maxCoeff(), 1.0});
    worst_j = std::max({worst_j, (Jl - Nl).cwiseAbs().maxCoeff() / scale, (Jp - Np).cwiseAbs().maxCoeff() / scale});
    delete vl, delete vp;
  }
  std::printf("line edge (200 random lines/poses): Jacobian analytic vs numeric max rel diff %.2e | orth round trip %.2e | "
              "error vs project_line residual %.2e\n", worst_j, worst_rt, worst_res);
  const bool ok = worst_j < 1e-6 && worst_rt < 1e-9 && worst_res < 1e-9;
  std::printf("  %s\n", ok ? "OK: analytic line Jacobians match" : "FAIL");
  return ok;
}

// 구조 방향: 세 방향 (Manhattan) 의 3D 선분을 투영 (0.5 px 잡음 + 무작위 선분 30 %) -> assign_dirs 가 첫 호출에 2 개,
// 두 번째 호출 (같은 키프레임) 에 나머지 1 개를 만들고 (참 방향과 0.5° 안), 배정된 선분이 맞는 방향인지 (BA 가 Huber / chi2 로 거름)
static bool vp_check() {
  const Camera cam(707.0912, 707.0912, 601.8873, 183.1104, 1226, 370);
  std::mt19937 rng(1);
  std::normal_distribution<double> n01(0, 1);
  std::uniform_real_distribution<double> uni(0, 1);
  const Mat3 Rw = so3_exp(Vec3(0.3, -0.5, 0.2));  // 세계 Manhattan 축
  const Mat3 R = so3_exp(Vec3(0.05, 0.4, -0.03)) * Rw.transpose();  // R_cw: 카메라에서 축이 비스듬히 보이게
  std::vector<Vec4> seg;
  std::vector<int> truth;  // 선분의 참 방향 (-1 = 무작위)
  while (seg.size() < 300) {
    const int k = uni(rng) < 0.3 ? -1 : int(uni(rng) * 3);
    const Vec3 X((uni(rng) - 0.5) * 30, (uni(rng) - 0.5) * 8, 4 + uni(rng) * 30);  // 카메라 좌표
    const Vec3 d = k < 0 ? Vec3(n01(rng), n01(rng), n01(rng)).normalized() : Vec3(R * Rw.col(k));
    const Vec3 a = cam.K * X, b = cam.K * (X + d * (1 + 3 * uni(rng)));
    if (a(2) <= 0 || b(2) <= 0) continue;
    Vec4 s;
    s << a.head<2>() / a(2), b.head<2>() / b(2);
    if (std::hypot(s(2) - s(0), s(3) - s(1)) < 40 || s.minCoeff() < 0 || s(0) > cam.W || s(2) > cam.W || s(1) > cam.H ||
        s(3) > cam.H)
      continue;
    for (int i = 0; i < 4; ++i) s(i) += 0.5 * n01(rng);
    seg.push_back(s), truth.push_back(k);
  }
  std::vector<Vec3> dirs;
  const size_t n1 = assign_dirs(cam, seg, R, dirs, 0).size(), d1 = dirs.size();
  const auto a = assign_dirs(cam, seg, R, dirs, 1);
  double worst = 0;
  for (const Vec3& v : dirs) {
    double best = 180;
    for (int k = 0; k < 3; ++k) best = std::min(best, std::acos(std::min(1.0, std::abs(v.dot(Rw.col(k))))) * 180 / M_PI);
    worst = std::max(worst, best);
  }
  int wrong = 0, structural = 0;
  for (const auto& [i, j] : a) {
    if (truth[i] < 0) continue;
    ++structural;
    wrong += std::acos(std::min(1.0, std::abs(dirs[j].dot(Rw.col(truth[i]))))) > 0.02;
  }
  std::printf("structural directions: first call %zu dirs (%zu segments), second %zu dirs, %zu assigned (%d structural, %d wrong), "
              "worst direction error %.2f deg, vp_dist at true VP %.2e px\n",
              d1, n1, dirs.size(), a.size(), structural, wrong, worst,
              std::abs(vp_dist(Vec4(100, 100, 200, 150), Vec3(300, 200, 1))));
  const bool ok = d1 == 2 && dirs.size() == 3 && worst < 0.5 && wrong * 50 <= structural && structural > 150;  // 짧은 선분 잡음 오배정 2 % 까지
  std::printf("  %s\n", ok ? "OK: directions found and segments assigned" : "FAIL");
  return ok;
}

int main() { return trial_set(0.0) && trial_set(0.1) && hybrid_check() && line_edge_check() && vp_check() ? 0 : 1; }
