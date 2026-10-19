#include "vo_plucker/optimizer.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/edge_project_xyz_onlypose.h>
#include <g2o/types/sba/vertex_se3_expmap.h>

namespace vp {

Pose motion_only_ba(const Camera& c, const Pose& init, const MapObs& m, int iters, double huber_px) {
  using Block = g2o::BlockSolver_6_3;
  g2o::SparseOptimizer opt;
  opt.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
      std::make_unique<Block>(std::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>())));

  auto* v = new g2o::VertexSE3Expmap();  // T_cw
  v->setId(0);
  v->setEstimate(g2o::SE3Quat(init.R, init.t));
  opt.addVertex(v);
  for (size_t i = 0; i < m.X.size(); ++i) {
    auto* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
    e->setVertex(0, v);
    e->setMeasurement(m.uv[i]);
    e->setInformation(Eigen::Matrix2d::Identity());
    auto* rk = new g2o::RobustKernelHuber();
    rk->setDelta(huber_px);
    e->setRobustKernel(rk);
    e->fx = c.fx, e->fy = c.fy, e->cx = c.cx, e->cy = c.cy;
    e->Xw = m.X[i];
    opt.addEdge(e);
  }
  opt.initializeOptimization();
  opt.optimize(iters);
  const g2o::SE3Quat T = v->estimate();
  return Pose{T.rotation().toRotationMatrix(), T.translation()};
}

}  // namespace vp
