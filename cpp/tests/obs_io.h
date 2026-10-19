// Reader for tools/dump_reference.py output (obs.txt / ref_poses.txt), shared by the tests.
#pragma once
#include <fstream>
#include <sstream>
#include <string>

#include "vo_plucker/tracker.h"

namespace vp {

struct ObsRecord {
  FrameObs o;
  std::optional<Pose> rel, init;
};

inline Pose read_pose(std::istringstream& ss) {
  Pose p;
  for (int i = 0; i < 9; ++i) ss >> p.R(i / 3, i % 3);
  for (int i = 0; i < 3; ++i) ss >> p.t(i);
  return p;
}

inline std::vector<ObsRecord> read_obs(const std::string& path, Camera& cam) {
  std::ifstream f(path);
  std::string line, tag;
  std::vector<ObsRecord> recs;
  while (std::getline(f, line)) {
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
  for (auto& r : recs) r.o.index();
  return recs;
}

}  // namespace vp
