// Real-data frontend, one frame at a time (port of frontend.py).
//   points : Shi-Tomasi + KLT (forward-backward check) + fundamental-matrix RANSAC
//   lines  : LSD; each line tracked by KLT on n_samples samples, refit, snapped to the nearest LSD detection
//   2D-2D  : robust_init (essential RANSAC + recoverPose + Sampson), relative_pose between consecutive frames
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vo_plucker/tracker.h"

namespace vp {

struct FrontendParams {
  int max_pts = 1000, min_pts = 800;
  int max_lines = 150, min_lines = 100;
  double min_line_len = 50;  // px
  int n_samples = 7;          // KLT samples per line
};

class Frontend {
 public:
  explicit Frontend(const FrontendParams& prm = {});
  FrameObs step(const cv::Mat& gray);  // grayscale CV_8U, rectified

  FrontendParams prm;
  std::vector<cv::Point2f> pts;  // current point tracks
  std::vector<int> pid;
  std::vector<Vec4> lines;  // current line tracks [x1, y1, x2, y2]
  std::vector<int> lid;
  double lsd_ms = 0;  // 누적 LSD 시간 (프로파일용)

 private:
  cv::Ptr<cv::LineSegmentDetector> lsd_;
  cv::Mat prev_;
  int next_p_ = 0, next_l_ = 0;
};

// frame 0 / init_frame point observations -> (R, t)
Pose robust_init(const Camera& c, const FrameObs& o0, const FrameObs& oI);
// consecutive frames a (k-1), b (k) -> (R_rel, t direction), nullopt if too few matches
std::optional<Pose> relative_pose(const Camera& c, const FrameObs& a, const FrameObs& b);

}  // namespace vp
