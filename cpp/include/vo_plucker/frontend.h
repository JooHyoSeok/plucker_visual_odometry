// Real-data frontend, one frame at a time (port of frontend.py).
//   points : Shi-Tomasi + KLT (forward-backward check) + fundamental-matrix RANSAC
//   lines  : LSD; each line tracked by KLT on n_samples samples, refit, snapped to the nearest LSD detection
//   2D-2D  : robust_init (essential RANSAC + recoverPose + Sampson), relative_pose between consecutive frames
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ximgproc/fast_line_detector.hpp>

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
  bool fld = false;  // 선 검출을 LSD 대신 ximgproc FastLineDetector 로 (빠름)
  // 맵 재사용 (reuse): reuse_every 프레임마다 ORB 디스크립터로 최근 reuse_frames 안에 본 (지금 트랙 없는) 맵 점을 다시 찾음
  int reuse_every = 0;      // 0 = 끔
  int reuse_frames = 300;
  double reuse_px = 15;     // 예측 투영 위치에서 이 안의 키포인트만
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
  void drop(const std::vector<int>& ids);  // 점 트랙 제거 (동적 물체 등)
  // 맵 재사용 (ORB-SLAM SearchByProjection, 디스크립터로): 지금 추적 중인 맵 점의 ORB 디스크립터를 갱신하고, 최근 본 맵 점 중
  // 트랙이 끊긴 것을 예측 포즈로 투영 -> 근처 ORB 키포인트와 Hamming 매칭 (비율 검사, 1:1) -> 그 자리 새 트랙을 옛 점 id 로 /
  // 새 관측으로 추가. 재방문 때 새 점을 만들지 않고 옛 맵을 씀. reattach 다음에 호출. 반환: 다시 붙인 점 수
  int reuse(const Camera& cam, const VOTracker& vo, const Pose& pred, FrameObs& o);

  FrontendParams prm;
  std::vector<cv::Point2f> pts;  // current point tracks
  std::vector<int> pid;
  std::vector<Vec4> lines;  // current line tracks [x1, y1, x2, y2]
  std::vector<int> lid;
  double lsd_ms = 0;  // 누적 LSD 시간 (프로파일용)

 private:
  cv::Ptr<cv::LineSegmentDetector> lsd_;
  cv::Ptr<cv::ximgproc::FastLineDetector> fld_;
  cv::Ptr<cv::ORB> orb_;
  std::unordered_map<int, cv::Mat> desc_;  // 맵 점 id -> 마지막으로 본 자리의 ORB 디스크립터 (upright)
  cv::Mat prev_;
  std::vector<std::vector<cv::Mat>> hist_;  // 최근 reattach_frames+1 장의 KLT 피라미드 (back() = 현재 프레임)
  int k_ = -1;                 // 현재 프레임 번호
  int next_p_ = 0, next_l_ = 0;
};

// 2D-2D에서 recoverPose의 distanceThresh: 이 값(|t|=1 기준) 넘게 먼 점은 인라이어에서 빠짐. OpenCV 기본값 50은
// baseline이 짧으면 (30 fps 핸드헬드: 프레임당 ~2 cm) 몇 m 밖의 점을 다 버림 -> 큰 값 = cheirality만 검사.
// 50 = Python frontend와 동일 (일치 테스트용)
constexpr double kRecoverPoseDist = 1e6;

// frame 0 / init frame point observations -> (R, t)
Pose robust_init(const Camera& c, const FrameObs& o0, const FrameObs& oI, double dist_thresh = kRecoverPoseDist);
// consecutive frames a (k-1), b (k) -> (R_rel, t direction), nullopt if too few matches.
// static_ids를 주면 그 id(맵 점 = 검증된 정적 배경)의 트랙만 사용 (30개 미만이면 전체): 움직이는 물체가 많을 때
// RANSAC이 물체의 움직임을 카메라 움직임으로 잡는 걸 막음
std::optional<Pose> relative_pose(const Camera& c, const FrameObs& a, const FrameObs& b,
                                  const std::unordered_map<int, Vec3>* static_ids = nullptr,
                                  double dist_thresh = kRecoverPoseDist);

}  // namespace vp
