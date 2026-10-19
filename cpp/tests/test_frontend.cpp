// Runs the C++ frontend on the same images and compares its observations / 2D-2D poses with the Python dump.
// usage: test_frontend DIR VIDEO      (DIR = tools/dump_reference.py output, VIDEO = same --video path)
// OpenCV 버전이 다르면 (Python 4.10 / C++ 4.6) KLT·RANSAC이 미세하게 달라질 수 있어서 첫 불일치 프레임과 이후 통계를 보고
#include <cstdio>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <set>

#include "obs_io.h"
#include "vo_plucker/frontend.h"

using namespace vp;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s DIR VIDEO\n", argv[0]);
    return 2;
  }
  Camera cam;
  const std::vector<ObsRecord> ref = read_obs(std::string(argv[1]) + "/obs.txt", cam);
  cv::VideoCapture cap(argv[2]);
  Frontend front;
  const int init_frame = TrackerParams().init_frame;
  std::vector<FrameObs> obs;
  int first_bad = -1, exact = 0, n_rel_bad = 0, worst_rel = -1;
  double max_dp = 0, max_dl = 0, max_drel = 0, sum_jac = 0;
  cv::Mat frame, gray;
  for (size_t k = 0; k < ref.size() && cap.read(frame); ++k) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    obs.push_back(front.step(gray));
    const FrameObs &c = obs.back(), &p = ref[k].o;
    const std::optional<Pose> rel = k ? relative_pose(cam, obs[k - 1], c) : std::nullopt;
    std::optional<Pose> init;
    if (int(k) == init_frame) init = robust_init(cam, obs[0], c);

    bool same = c.pid == p.pid && c.lid == p.lid && bool(rel) == bool(ref[k].rel);
    if (same) {
      double dp = 0, dl = 0;
      for (size_t i = 0; i < c.pid.size(); ++i) dp = std::max(dp, (c.uv[i] - p.uv[i]).norm());
      for (size_t i = 0; i < c.lid.size(); ++i) dl = std::max(dl, (c.seg[i] - p.seg[i]).norm());
      double dr = 0;
      if (rel) dr = (rel->R - ref[k].rel->R).norm() + (rel->t - ref[k].rel->t).norm();
      if (init) dr = std::max(dr, (init->R - ref[k].init->R).norm() + (init->t - ref[k].init->t).norm());
      if (dr > 1e-6) ++n_rel_bad;
      if (dr > max_drel) max_drel = dr, worst_rel = k;
      max_dp = std::max(max_dp, dp), max_dl = std::max(max_dl, dl);
      same = dp < 1e-3 && dl < 1e-3;
    }
    exact += same;
    if (!same && first_bad < 0) {
      first_bad = k;
      std::printf("first mismatch @frame %zu: points %zu vs py %zu, lines %zu vs py %zu, rel %d vs py %d\n", k,
                  c.pid.size(), p.pid.size(), c.lid.size(), p.lid.size(), bool(rel), bool(ref[k].rel));
      for (size_t i = 0; i < std::min(c.seg.size(), p.seg.size()); ++i)
        if ((c.seg[i] - p.seg[i]).norm() > 1e-3) {
          std::printf("  line #%zu id %d: c++ [%.2f %.2f %.2f %.2f] py [%.2f %.2f %.2f %.2f]\n", i, c.lid[i],
                      c.seg[i](0), c.seg[i](1), c.seg[i](2), c.seg[i](3), p.seg[i](0), p.seg[i](1), p.seg[i](2),
                      p.seg[i](3));
          break;
        }
    }
    const std::set<int> a(c.pid.begin(), c.pid.end()), b(p.pid.begin(), p.pid.end());
    int inter = 0;
    for (int j : a) inter += b.count(j);
    sum_jac += double(inter) / (a.size() + b.size() - inter);
  }
  std::printf("%zu frames: identical %d | max |du| point %.2e px, line %.2e px | "
              "mean point-id overlap (Jaccard) %.3f | LSD %.1f ms/frame\n",
              obs.size(), exact, max_dp, max_dl, sum_jac / obs.size(), front.lsd_ms / obs.size());
  std::printf("2D-2D rel/init pose: %d frames differ > 1e-6, max %.2e @frame %d\n", n_rel_bad, max_drel, worst_rel);
  return 0;
}
