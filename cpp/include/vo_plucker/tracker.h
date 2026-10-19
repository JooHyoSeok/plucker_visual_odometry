// Frame-by-frame monocular VO, points + Plücker lines (port of vo_plucker.VOTracker, "decoupled" mode).
//   rotation + translation direction from 2D-2D (given per frame as rel), translation magnitude from the map
//   mapping: N-view triangulation of points / lines until parallax > mature_deg, then frozen
//            kf_mapping: only at keyframes, from keyframe observations (else every frame, all observations)
//   init   : whenever an init pose (frame 0 -> current) is passed to add() (the caller decides when: parallax)
//   still  : median flow of map-point tracks < still_px -> stationary, pose kept, no mapping
//   depth  : (use_depth) metric depth per point (e.g. LiDAR / depth network): init from one frame, points with depth
//            go straight into the map (frozen), tracking 6-DoF against the map; points without depth are triangulated
#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "vo_plucker/geometry.h"

namespace vp {

struct FrameObs {
  std::vector<int> pid;  // point track ids (frontend order)
  std::vector<Vec2> uv;
  std::vector<int> lid;  // line track ids
  std::vector<Vec4> seg;  // [x1, y1, x2, y2]
  std::vector<float> depth;  // (선택) 점별 metric depth [m], pid와 같은 순서, 없으면 NaN. 비어 있으면 깊이 없음

  void index();  // pid/lid -> 위치 맵 생성 (point()/line() 전에 호출)
  const Vec2* point(int j) const;
  const Vec4* line(int j) const;

 private:
  std::unordered_map<int, int> pi_, li_;
};

struct TrackerParams {
  int tri_gap = 2;           // (kf_mapping=false) 새 랜드마크 삼각측량 최소 프레임 간격
  double pix_noise = 1.0;    // px, tracking failure 판정 기준
  double point_min_deg = 2.0;
  double line_min_deg = 2.0;  // 두 역투영 평면 사이 최소 각도
  double mature_deg = 5.0;    // parallax가 이 이상이면 랜드마크 고정
  double outlier_px = 6.0;    // 포즈 추정 후 이보다 큰 재투영 오차 랜드마크는 맵에서 제거
  bool use_lines = true;
  double still_px = 0.5;      // 맵 점 트랙 flow 중앙값이 이보다 작으면 정지 (0 = 끔)
  // 초기화 직후 0..I 사이 프레임은 맵(frame 0, I로 삼각측량) 기준 6-DoF로 추정: 느린 출발에서 2D-2D 이동 방향이
  // 노이즈가 되는 문제 (KITTI 07 시작). false = 2D-2D (Python과 동일)
  bool catchup_6dof = true;
  // 그 밖의 프레임도 회전 보정 flow 중앙값이 이보다 작으면 6-DoF (0 = 끔. 10이면 KITTI 06에서 yaw 편향으로 나빠짐)
  double rel_min_px = 0;
  int init_min_points = 100;  // 초기화 직후 삼각측량된 맵 점이 이보다 적거나 1..I-1 추적이 실패하면 초기화 취소 (다음 프레임에 재시도)
  bool kf_mapping = false;    // true: 키프레임에서만 매핑 (KITTI 06에서 오히려 나빠짐), false: 매 프레임 (Python과 동일)
  double kf_parallax_px = 10; // 직전 키프레임 대비 회전 보정 parallax 중앙값이 이 이상이면 키프레임
  int kf_min_map = 150;       // 추적에 쓴 맵 대응이 이보다 적으면 (parallax 2 px 이상일 때) 키프레임
  int kf_max_gap = 8;         // 키프레임 최대 간격 [frames] (parallax 2 px 이상일 때)
  bool use_depth = false;     // FrameObs::depth 사용 (깊이 초기화 + 깊이 맵 점)
  bool depth_6dof = true;     // use_depth일 때 2D-2D 대신 맵 기준 6-DoF로 추적
  bool pose_g2o = false;      // 6-DoF 추정을 g2o motion-only BA로 (점만, 선 edge는 다음 단계). false = 자체 LM (점+선)
};

class VOTracker {
 public:
  VOTracker(const Camera& cam, const TrackerParams& prm) : cam(cam), prm(prm) {}

  // rel = (R_rel, t방향) k-1 -> k, init = (R, t) frame 0 -> k: 처음 주어진 프레임에서 초기화 (그 전 프레임은 버퍼링)
  // 반환: 이번 호출에서 포즈가 정해진 프레임 번호
  std::vector<int> add(FrameObs o, const std::optional<Pose>& rel, const std::optional<Pose>& init);
  // 다음 프레임(obs.size())의 constant-velocity 예측 포즈. 직전 두 프레임 포즈가 없으면 nullopt
  std::optional<Pose> predict() const;
  bool initialized() const { return init_frame_ >= 0; }
  // 마지막 프레임 k: k-1 -> k 추정 운동에 안 맞는 (맵에 없는) 점 트랙 id = 움직이는 물체 / 잘못된 트랙.
  // 이동이 거의 없으면 (|ΔC| / 맵 점 중앙 깊이 < 0.002) 회전만으로 예측한 위치, 아니면 Sampson 오차로 판정 [px]
  std::vector<int> inconsistent_tracks(double px) const;

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
  std::vector<int> kfs;  // 키프레임 번호 (오름차순)
  int n_still = 0;       // 정지로 판정한 프레임 수
  int n_map6dof = 0;     // parallax가 작아서 맵 기준 6-DoF로 추정한 프레임 수
  long n_depth_pts = 0;  // 깊이로 만든 맵 점 수
  int failed_at = -1;  // tracking 실패 프레임 (-1 = 정상), 실패 후에는 추적 안 함 (relocalization 없음)
  std::string fail_reason;
  std::string init_note;  // 마지막 초기화 시도가 거절된 이유
  int line_rejected = 0;

 private:
  bool track(int k);
  void update_map(int k);
  bool is_keyframe(int k, int n_map_used) const;
  void add_depth_points(int k);  // 프레임 k의 깊이 있는 트랙 -> 맵 점 (확정)
  void reset_init();  // 초기화 취소: 맵/포즈를 비우고 초기화 전 상태로
  int init_frame_ = -1;
  bool fail(int k, const char* why) {
    failed_at = k, fail_reason = why;
    return false;
  }
};

}  // namespace vp
