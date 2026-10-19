// g2o motion-only BA vs the existing LM (estimate_pose) on synthetic data, points only (lines come with the line edge).
//   no outliers : Huber inactive -> plain least squares -> both must reach the same optimum
//   10% outliers: Huber differs (LM: per residual component, g2o: per point 2D norm) -> compare accuracy vs GT only
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

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

int main() { return trial_set(0.0) && trial_set(0.1) && hybrid_check() ? 0 : 1; }
