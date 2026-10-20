// Frame-by-frame monocular VO, points + Plücker lines (port of vo_plucker.VOTracker, "decoupled" mode).
//   rotation + translation direction from 2D-2D (given per frame as rel), translation magnitude from the map
//   mapping: N-view triangulation of points / lines until parallax > mature_deg, then frozen
//            kf_mapping: only at keyframes, from keyframe observations (else every frame, all observations)
//   init   : whenever an init pose (frame 0 -> current) is passed to add() (the caller decides when: parallax)
//   still  : median flow of map-point tracks < still_px -> stationary, pose kept, no mapping
//   depth  : (use_depth) metric depth per point (e.g. LiDAR / depth network): init from one frame, points with depth
//            go straight into the map (frozen), tracking 6-DoF against the map; points without depth are triangulated
#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "vo_plucker/geometry.h"

namespace vp {
// 트래킹 / 루프 / 로컬 매핑 스레드가 vo 를 같이 쓸 때의 lock (ORB-SLAM3 mMutexMapUpdate). 트래킹은 프레임 처리 동안 잡고,
// 다른 스레드가 기다리고 있으면 다음 프레임 전에 먼저 양보
struct MapLock {
  std::mutex mx;
  std::atomic<int> want{0};
  std::unique_lock<std::mutex> tracking() {
    while (want > 0) std::this_thread::yield();
    return std::unique_lock<std::mutex>(mx);
  }
  std::unique_lock<std::mutex> worker() {
    ++want;
    std::unique_lock<std::mutex> l(mx);
    --want;
    return l;
  }
};
}  // namespace vp

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
  bool hybrid = false;        // 2D-2D 로 푼 프레임을 맵 재투영 + 트랙 epipolar 로 6-DoF 정제 (hybrid_ba)
  double hybrid_w = 1.0;      // epipolar 항 가중치 (맵 재투영 대비)
  double hybrid_min_deg = 0;  // 2D-2D 회전이 이 각도 [deg/frame] 이상인 프레임에만 (회전 구간)
  bool hybrid_fix_rot = false; // 회전은 2D-2D 그대로, 이동만 정제
  bool local_ba = false;      // 키프레임마다 local BA (g2o, 점만): 키프레임 선택은 is_keyframe
  int ba_window = 10;         // local BA에서 최적화하는 최근 키프레임 수
  int ba_fixed = 10;          // 같은 점을 보는 그 앞 키프레임을 최대 이만큼 고정
  bool ba_lines = false;      // local BA 에 라인도 (orthonormal 4-DoF, 해석 Jacobian)
  bool depth_init = false;    // 깊이가 붙은 첫 프레임 하나로 초기화 (점은 고정 안 함, 시차가 생기면 삼각측량으로 교체), 그 뒤 단안
  bool ba_mature_only = false; // local BA 에 성숙한 점만
  bool track_mature = false;  // 성숙한 랜드마크가 40개 이상이면 매 프레임 그것만으로 6-DoF (ORB-SLAM TrackLocalMap 처럼)
  bool da3_lines = false;     // 키프레임 선분을 DA3 깊이로 바로 Plücker 라인 (진행 방향 퇴화 라인 포함), 평면 각 mature_deg 넘으면 삼각측량으로
  bool da3_points = false;    // 키프레임에서 시차가 작은 트랙을 DA3 깊이로 맵 점 (pseudo RGB-D), 시차 mature_deg 넘으면 삼각측량으로
  double ba_depth_sigma = 0;  // local BA 에 DA3 점 깊이 prior (log 깊이 표준편차, 0 = 끔. --depth-prior da3 의 앵커 깊이 사용)
  bool ba_drop_obs = false;   // local BA 이상치: 안 맞는 관측만 빼고 (ORB-SLAM), 좋은 관측이 2개 미만인 점만 맵에서 제거
  double ba_line_depth_sigma = 0;  // (ba_lines) 라인 끝점 DA3 log-depth prior σ (0 = 끔): 광선 방향 위치 / 진행 방향 라인을 묶음
  bool ba_line_nowrite = false;  // (ba_lines) 실험: BA 라인은 포즈만 제약하고 맵 라인에는 안 써넣음 (이상치 라인 삭제만)
  double ba_vp_sigma = 0;     // local BA 에 구조 방향 (소실점) 제약 [px] (0 = 끔): 선분이 세계 방향 vo.dirs 와 나란하게 -> 회전 드리프트
  bool recover = false;       // 추적 실패 다음 프레임: 외삽 포즈 + DA3 깊이 (da3_ref 로 VO 단위) 트랙 점 -> 같은 좌표계 / 스케일로 계속
  bool keyframes = false;     // 매핑과 상관없이 키프레임 목록 유지 (루프 클로저용)
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
  std::unordered_map<int, int> last_p, last_l;  // 랜드마크를 마지막으로 본 프레임 (루프 보정 기준)
  std::unordered_map<int, Pose> rel;
  std::vector<int> kfs;  // 키프레임 번호 (오름차순)
  int map_version = 0;   // 루프/앵커 보정이 맵을 바꿀 때마다 +1 (map lock 아래) -> 그 전에 만든 local BA 결과는 버림
  MapLock map_lock;
  // DA3 앵커 때 루프 스레드가 map lock 아래 기록: 키프레임 -> (맵 점 id, DA3 깊이 [m]), da3_ref = 기준 VO 단위/미터 (0 = 아직)
  std::unordered_map<int, std::vector<std::pair<int, float>>> da3;
  double da3_ref = 0;
  double da3_sigma = 0;  // (da3_adaptive) 온라인 추정으로 정한 BA DA3 prior σ (0 = prm.ba_depth_sigma)
  std::unordered_set<int> da3_p, da3_l;  // (--da3-points) DA3 깊이로 만든 맵 점: 삼각측량은 시차가 mature_deg 를 넘을 때만 바꿈
  std::unordered_set<long long> bad_obs;  // (ba_drop_obs) BA 가 뺀 점 관측 obs_key(프레임, 점 id): 그 뒤 BA / 삼각측량에서 안 씀
  std::unordered_set<long long> bad_lobs;  // 같은 것, 라인 관측 obs_key(프레임, 라인 id)
  // (ba_line_depth_sigma) 키프레임 -> 맵 라인 선분 끝점 (안쪽 10 %) 의 DA3 깊이 [m] (0 = 없음), 루프 스레드가 앵커 때 기록
  struct LineDepth { int lid; Vec2 u0, u1; float z0, z1; };
  std::unordered_map<int, std::vector<LineDepth>> da3_lines;
  // (ba_vp_sigma) 구조 방향 = 나란한 선분 묶음의 세계 단위 방향 (소실점), 키프레임 -> (선분 번호 obs[f].seg, 방향 번호)
  std::vector<Vec3> dirs;
  std::unordered_map<int, std::vector<std::pair<int, int>>> dir_obs;
  static long long obs_key(int f, int j) { return (long long)f << 32 | unsigned(j); }
  int n_still = 0;       // 정지로 판정한 프레임 수
  int n_map6dof = 0;     // parallax가 작아서 맵 기준 6-DoF로 추정한 프레임 수
  int n_track_mature = 0;  // --track-mature 로 성숙한 랜드마크 6-DoF 로 푼 프레임 수
  int n_rescued = 0;     // 2D-2D 포즈가 맵과 안 맞아서 맵 기준 6-DoF 로 다시 푼 프레임 수 (안 그러면 추적 실패)
  long n_depth_pts = 0;  // 깊이로 만든 맵 점 수
  int n_ba = 0;          // local BA 실행 수
  double ba_ms = 0;      // local BA 누적 시간
  long ba_points = 0, ba_outliers = 0, ba_lines_used = 0, ba_line_outliers = 0, ba_depth_priors = 0, ba_vp_edges = 0;
  double ba_rms_before = 0, ba_rms_after = 0;  // 누적 (실행 수로 나누면 평균)
  int failed_at = -1;  // tracking 실패 프레임 (-1 = 정상), 실패 후에는 추적 안 함 (prm.recover 면 다음 프레임에 DA3 깊이로 복구)
  std::string fail_reason;
  int n_recovered = 0;
  // (prm.recover) 실패 상태 + DA3 기준 스케일이 있음 -> 다음 프레임 obs.depth 에 DA3 metric 깊이 [m] 를 채워서 add
  bool wants_depth() const { return prm.recover && failed_at >= 0 && da3_ref > 0; }
  std::string init_note;  // 마지막 초기화 시도가 거절된 이유
  int line_rejected = 0;

 private:
  bool track(int k);
  void update_map(int k);
  bool is_keyframe(int k, int n_map_used) const;
  void add_depth_points(int k);  // 프레임 k의 깊이 있는 트랙 -> 맵 점 (확정)
  void reset_init();  // 초기화 취소: 맵/포즈를 비우고 초기화 전 상태로
  std::vector<int> recover(int k);  // 실패 뒤 프레임 k 복구, 반환 = 포즈가 정해진 프레임 (실패면 빈 것)
  int init_frame_ = -1;
  bool fail(int k, const char* why) {
    failed_at = k, fail_reason = why;
    return false;
  }
};

}  // namespace vp
