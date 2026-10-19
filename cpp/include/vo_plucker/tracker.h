// Frame-by-frame monocular VO, points + Plücker lines (port of vo_plucker.VOTracker, "decoupled" mode).
//   rotation + translation direction from 2D-2D (given per frame as rel), translation magnitude from the map
//   mapping: N-view triangulation of points / lines until parallax > mature_deg, then frozen
#pragma once
#include <unordered_map>
#include <unordered_set>

#include "vo_plucker/geometry.h"

namespace vp {

struct FrameObs {
  std::vector<int> pid;  // point track ids (frontend order)
  std::vector<Vec2> uv;
  std::vector<int> lid;  // line track ids
  std::vector<Vec4> seg;  // [x1, y1, x2, y2]

  void index();  // pid/lid -> 위치 맵 생성 (point()/line() 전에 호출)
  const Vec2* point(int j) const;
  const Vec4* line(int j) const;

 private:
  std::unordered_map<int, int> pi_, li_;
};

struct TrackerParams {
  int init_frame = 3;        // frame 0 과 이 프레임으로 초기화
  int tri_gap = 2;           // 새 랜드마크 삼각측량 최소 프레임 간격
  double pix_noise = 1.0;    // px, tracking failure 판정 기준
  double point_min_deg = 2.0;
  double line_min_deg = 2.0;  // 두 역투영 평면 사이 최소 각도
  double mature_deg = 5.0;    // parallax가 이 이상이면 랜드마크 고정
  double outlier_px = 6.0;    // 포즈 추정 후 이보다 큰 재투영 오차 랜드마크는 맵에서 제거
  bool use_lines = true;
};

class VOTracker {
 public:
  VOTracker(const Camera& cam, const TrackerParams& prm) : cam(cam), prm(prm) {}

  // rel = (R_rel, t방향) k-1 -> k, init = (R, t) frame 0 -> init_frame (k == init_frame일 때 필수)
  // 반환: 이번 호출에서 포즈가 정해진 프레임 번호
  std::vector<int> add(FrameObs o, const std::optional<Pose>& rel, const std::optional<Pose>& init);

  // state: 바깥(local BA, 루프 보정)에서 제자리 수정 가능 -> 다음 프레임 추적에 그대로 반영
  Camera cam;
  TrackerParams prm;
  std::vector<FrameObs> obs;
  std::vector<Pose> poses;
  std::vector<char> has_pose;
  std::unordered_map<int, Vec3> map_p;
  std::unordered_map<int, Line> map_l;
  std::unordered_set<int> mature_p, mature_l;
  std::unordered_map<int, int> first_p, first_l;
  std::unordered_map<int, Pose> rel;
  int failed_at = -1;  // tracking 실패 프레임 (-1 = 정상), 실패 후에는 추적 안 함 (relocalization 없음)
  int line_rejected = 0;

 private:
  bool track(int k);
  void update_map(int k);
  bool fail(int k) {
    failed_at = k;
    return false;
  }
};

}  // namespace vp
