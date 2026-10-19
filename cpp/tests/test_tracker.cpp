// Replays tools/dump_reference.py output through the C++ VOTracker and compares with the Python poses.
// usage: test_tracker DIR [--no-lines]     (DIR/obs.txt, DIR/ref_poses.txt)
//        test_tracker DIR --corrupt-rel K  (프레임 K 의 2D-2D 회전을 30 deg 틀어서: 맵 기준 6-DoF 구제로 추적이 계속돼야 통과)
#include <chrono>
#include <cstdio>
#include <cstring>

#include "obs_io.h"

using namespace vp;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s DIR [--no-lines]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  TrackerParams prm;
  prm.use_lines = !(argc > 2 && std::strcmp(argv[2], "--no-lines") == 0);
  const int corrupt = argc > 3 && std::strcmp(argv[2], "--corrupt-rel") == 0 ? std::atoi(argv[3]) : -1;
  prm.kf_mapping = false, prm.still_px = 0, prm.init_min_points = 0, prm.catchup_6dof = false;  // Python vo_plucker.VOTracker와 같은 동작
  Camera cam;
  std::vector<ObsRecord> recs = read_obs(dir + "/obs.txt", cam);

  VOTracker vo(cam, prm);
  const auto t0 = std::chrono::steady_clock::now();
  for (size_t k = 0; k < recs.size(); ++k) {
    auto& r = recs[k];
    if (int(k) == corrupt && r.rel) r.rel->R = Eigen::AngleAxisd(30 * M_PI / 180, Vec3::UnitY()).toRotationMatrix() * r.rel->R;
    vo.add(std::move(r.o), r.rel, r.init);
  }
  if (corrupt >= 0) {
    std::printf("corrupted 2D-2D at frame %d: failed_at %d, rescued %d -> %s\n", corrupt, vo.failed_at, vo.n_rescued,
                vo.failed_at < 0 && vo.n_rescued > 0 ? "OK" : "FAIL");
    return vo.failed_at < 0 && vo.n_rescued > 0 ? 0 : 1;
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  // ---- compare ----
  std::ifstream fr(dir + "/ref_poses.txt");
  std::string line;
  int n = 0, mismatch_has = 0, first_bad = -1;
  double max_dR = 0, max_dt = 0, path = 0;
  Vec3 C_prev = Vec3::Zero();
  while (std::getline(fr, line)) {
    std::istringstream ss(line);
    int k, has;
    ss >> k >> has;
    ++n;
    if (bool(has) != bool(vo.has_pose[k])) {
      ++mismatch_has;
      continue;
    }
    if (!has) continue;
    const Pose p = read_pose(ss);
    const double dR = (p.R - vo.poses[k].R).cwiseAbs().maxCoeff();
    const double dt = (p.t - vo.poses[k].t).norm() / (1.0 + p.t.norm());
    max_dR = std::max(max_dR, dR), max_dt = std::max(max_dt, dt);
    if (first_bad < 0 && (dR > 1e-6 || dt > 1e-6)) first_bad = k;
    if (k) path += (center(p) - C_prev).norm();
    C_prev = center(p);
  }
  const Vec3 dC = center(vo.poses[n - 1]) - C_prev;
  std::printf("%zu frames, lines %d | C++ tracker %.2f ms/frame | failed_at %d\n", recs.size(), prm.use_lines,
              ms / recs.size(), vo.failed_at);
  std::printf("vs python: pose-exists mismatch %d, max |dR| %.2e, max rel |dt| %.2e, first frame > 1e-6: %d, "
              "final center diff %.3f (path %.1f, VO units)\n",
              mismatch_has, max_dR, max_dt, first_bad, dC.norm(), path);
  return mismatch_has == 0 ? 0 : 1;
}
