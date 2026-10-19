// g2o-based optimization. Now: motion-only BA (pose only, map fixed) for tracking.
// Next: line edge (orthonormal 4-DoF), local BA on keyframes, Sim(3) pose graph.
#pragma once
#include "vo_plucker/geometry.h"

namespace vp {

// 맵(점)은 고정하고 포즈만: 점 재투영 오차 [px], Huber, g2o Levenberg-Marquardt (ORB-SLAM PoseOptimization 방식).
// m.L (선)은 아직 무시 -> 선 edge는 다음 단계
Pose motion_only_ba(const Camera& c, const Pose& init, const MapObs& m, int iters = 40, double huber_px = 3.0);

}  // namespace vp
