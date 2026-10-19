// g2o-based optimization: motion-only BA (tracking), local BA on keyframes (points only for now).
// Next: line edge (orthonormal 4-DoF), depth prior, Sim(3) pose graph.
#pragma once
#include <memory>

#include "vo_plucker/tracker.h"

namespace vp {

// 맵(점)은 고정하고 포즈만: 점 재투영 오차 [px], Huber, g2o Levenberg-Marquardt (ORB-SLAM PoseOptimization 방식).
// m.L (선)은 아직 무시 -> 선 edge는 다음 단계
Pose motion_only_ba(const Camera& c, const Pose& init, const MapObs& m, int iters = 40, double huber_px = 3.0);

// 하이브리드 motion-only BA: 맵 점 재투영 [px] + 직전 프레임 (포즈 prev 고정) 과의 트랙 epipolar Sampson 오차 [px] x w_epi.
// 회전 구간에서 2D-2D 만으로는 yaw 회전과 옆 이동이 섞여 (bas-relief) 이동 방향이 틀어지는 것을 맵이 잡아 주고,
// 맵만 쓸 때의 회전 드리프트는 많은 트랙의 epipolar 항이 잡아 줌. tracks: (직전 프레임 픽셀, 현재 픽셀)
Pose hybrid_ba(const Camera& c, const Pose& init, const MapObs& m, const Pose& prev,
               const std::vector<std::pair<Vec2, Vec2>>& tracks, double w_epi = 1.0, int iters = 10, double huber_px = 3.0,
               bool fix_rotation = false);

struct LocalBAStats {
  int kfs = 0, fixed = 0, points = 0, edges = 0, outliers = 0, lines = 0, line_outliers = 0;
  double ms = 0;
  double rms_before = 0, rms_after = 0;  // 인라이어 관측의 재투영 오차 RMS [px], BA 전 / 후
  bool discarded = false;                 // (스레드) BA 하는 동안 루프 보정이 들어와서 결과를 버림
};
// 최근 키프레임 window개의 포즈 + 그 키프레임들이 보는 맵 점을 g2o로 함께 최적화 (ORB-SLAM LocalBundleAdjustment 방식).
//   고정: 창보다 앞선 키프레임 중 같은 점을 보는 것 최대 max_fixed개 (최소 2개 고정 -> 단안 스케일 gauge)
//   점은 Schur 소거 (BlockSolver_6_3), Huber, 5회 -> chi2 큰 관측 제외 -> 10회
//   결과: 키프레임/점 제자리 갱신, 최적화된 점은 mature (재삼각측량 제외), 2차 후에도 이상치인 점은 맵에서 제거,
//         창 안 비키프레임 포즈는 직전 키프레임과 같은 보정 적용
// lines: Plücker 라인도 orthonormal 4-DoF 정점 + 끝점 거리 엣지 (해석 Jacobian, vo_plucker/g2o_line.h) 로 같이 최적화
// mature_only: 시차가 충분해서 이미 고정된 (mature) 점만 (낮은 시차 점의 깊이 편향 진단용)
LocalBAStats local_ba(VOTracker& vo, int window = 10, int max_fixed = 10, bool lines = false, bool mature_only = false);

// local BA 를 세 단계로 (LocalMapping 스레드용, ORB-SLAM3 처럼 최적화는 map lock 없이):
//   build (map lock 아래, vo 읽기만) -> solve (lock 없이) -> apply (map lock 아래, 써넣기). local_ba = 세 단계를 이어서
struct LocalBAProblem;
struct LocalBAProblemDeleter { void operator()(LocalBAProblem* p) const; };
using LocalBAPtr = std::unique_ptr<LocalBAProblem, LocalBAProblemDeleter>;
LocalBAPtr local_ba_build(const VOTracker& vo, int window, int max_fixed, bool lines, bool mature_only);  // 점이 없으면 nullptr
void local_ba_solve(LocalBAProblem& p);
LocalBAStats local_ba_apply(VOTracker& vo, LocalBAProblem& p);

}  // namespace vp
