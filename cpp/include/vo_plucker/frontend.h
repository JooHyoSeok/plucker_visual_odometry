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
  // 끊긴 맵 점 재연결 (reattach): 최근 이만큼의 프레임에서 본 맵 점까지, 투영 위치에서 이 px 안에서만 받아들임
  int reattach_frames = 30;
  double reattach_px = 20;
};

class Frontend {
 public:
  explicit Frontend(const FrontendParams& prm = {});
  FrameObs step(const cv::Mat& gray);  // grayscale CV_8U, rectified
  // 가림 등으로 KLT 트랙이 끊긴 맵 점을 다시 붙임 (ORB-SLAM SearchByProjection 역할, 디스크립터 대신 KLT):
  // 최근 본 맵 점을 예측 포즈 pred로 투영 -> 마지막으로 본 프레임에서 현재 프레임으로 KLT (투영 위치를 초기값으로)
  // -> FB 검사 + 투영 위치 근처면 원래 id로 o와 추적 목록에 추가. 같은 자리의 맵에 없는 새 트랙은 원래 id로 합침
  // step() 직후, vo.add() 전에 호출. 반환: 붙인 점 수
  int reattach(const Camera& cam, const VOTracker& vo, const Pose& pred, FrameObs& o);

  FrontendParams prm;
  std::vector<cv::Point2f> pts;  // current point tracks
  std::vector<int> pid;
  std::vector<Vec4> lines;  // current line tracks [x1, y1, x2, y2]
  std::vector<int> lid;
  double lsd_ms = 0;  // 누적 LSD 시간 (프로파일용)

 private:
  cv::Ptr<cv::LineSegmentDetector> lsd_;
  cv::Mat prev_;
  std::vector<std::vector<cv::Mat>> hist_;  // 최근 reattach_frames+1 장의 KLT 피라미드 (back() = 현재 프레임)
  int k_ = -1;                 // 현재 프레임 번호
  int next_p_ = 0, next_l_ = 0;
};

// frame 0 / init_frame point observations -> (R, t)
Pose robust_init(const Camera& c, const FrameObs& o0, const FrameObs& oI);
// consecutive frames a (k-1), b (k) -> (R_rel, t direction), nullopt if too few matches
std::optional<Pose> relative_pose(const Camera& c, const FrameObs& a, const FrameObs& b);

}  // namespace vp
