// g2o motion-only BA vs the existing LM (estimate_pose) on synthetic data, points only (lines come with the line edge).
//   no outliers : Huber inactive -> plain least squares -> both must reach the same optimum
//   10% outliers: Huber differs (LM: per residual component, g2o: per point 2D norm) -> compare accuracy vs GT only
#include <chrono>
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

int main() { return trial_set(0.0) && trial_set(0.1) ? 0 : 1; }
