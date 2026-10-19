// Replays tools/dump_reference.py output through the C++ VOTracker and compares with the Python poses.
// usage: test_tracker DIR [--no-lines]     (DIR/obs.txt, DIR/ref_poses.txt)
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "vo_plucker/tracker.h"

using namespace vp;

static Pose read_pose(std::istringstream& ss) {
  Pose p;
  for (int i = 0; i < 9; ++i) ss >> p.R(i / 3, i % 3);
  for (int i = 0; i < 3; ++i) ss >> p.t(i);
  return p;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s DIR [--no-lines]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  TrackerParams prm;
  prm.use_lines = !(argc > 2 && std::strcmp(argv[2], "--no-lines") == 0);

  // ---- read obs ----
  std::ifstream fo(dir + "/obs.txt");
  std::string line, tag;
  Camera cam;
  struct Rec {
    FrameObs o;
    std::optional<Pose> rel, init;
  };
  std::vector<Rec> recs;
  while (std::getline(fo, line)) {
    std::istringstream ss(line);
    ss >> tag;
    if (tag == "camera") {
      double fx, fy, cx, cy;
      int W, H;
      ss >> fx >> fy >> cx >> cy >> W >> H;
      cam = Camera(fx, fy, cx, cy, W, H);
    } else if (tag == "frame") {
      recs.emplace_back();
    } else if (tag == "p") {
      int j;
      Vec2 u;
      ss >> j >> u(0) >> u(1);
      recs.back().o.pid.push_back(j), recs.back().o.uv.push_back(u);
    } else if (tag == "l") {
      int j;
      Vec4 s;
      ss >> j >> s(0) >> s(1) >> s(2) >> s(3);
      recs.back().o.lid.push_back(j), recs.back().o.seg.push_back(s);
    } else if (tag == "rel") {
      recs.back().rel = read_pose(ss);
    } else if (tag == "init") {
      recs.back().init = read_pose(ss);
    }
  }

  // ---- run ----
  VOTracker vo(cam, prm);
  const auto t0 = std::chrono::steady_clock::now();
  for (auto& r : recs) vo.add(std::move(r.o), r.rel, r.init);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  // ---- compare ----
  std::ifstream fr(dir + "/ref_poses.txt");
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
