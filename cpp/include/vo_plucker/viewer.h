// Map viewer (Pangolin, own thread), like ORB-SLAM's Viewer + MapDrawer + FrameDrawer.
//   top   : 3D map points (gray, visible = red), Plücker line segments (blue), trajectory (green),
//           ground truth (gray, origin-anchored, scale from the first 50 frames), current camera
//           "Sim3 align": Umeyama Sim(3) of the trajectory so far onto GT (as for ATE) -> everything drawn in GT
//           coordinates, yellow segments = position error every 10 frames, ATE shown in the menu
//   bottom: current image with tracked points / lines and the map reprojected into it
// The tracking thread calls update() after every frame; drawing happens in the viewer thread.
// Start screen (optional): pick loop closure / detector / verifier / lines in the menu, then "Start".
#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <opencv2/core.hpp>
#include <string>
#include <thread>
#include <unordered_map>

#include "vo_plucker/tracker.h"

namespace vp {

struct RunOptions {  // 시작 화면에서 고르는 것 (기본값 = 명령행 인자)
  bool loop = false;
  std::string pr = "thumb", ver = "orb";  // --loop-pr / --loop-ver 와 같은 이름
  bool lines = true;
  bool depth_prior = false;  // --depth-prior da3
};

class Viewer {
 public:
  // gt_cw: 프레임별 GT T_cw (비워두면 GT 없음). 추정 궤적이 시작한 프레임의 GT 포즈를 원점으로 맞춰 그림
  // start: 주면 메뉴에 시작 화면 (wait_start() 로 Start 를 기다림)
  explicit Viewer(const Camera& cam, std::vector<Pose> gt_cw = {}, bool sim3_align = false,
                  std::optional<RunOptions> start = std::nullopt);
  ~Viewer();
  // Start 를 누를 때까지 대기 -> 고른 옵션. 그 전에 창을 닫으면 nullopt
  std::optional<RunOptions> wait_start();
  void update(const cv::Mat& bgr, const VOTracker& vo);  // 추적 스레드: 방금 넣은 프레임 기준 스냅샷
  bool paused() const { return pause_; }
  bool closed() const { return closed_; }
  // 시퀀스 끝: 창을 닫을 때까지 대기. snapshot 경로를 주면 한 장 저장하고 바로 종료
  void finish(const std::string& snapshot = "");
  // 전역 BA: 시작 때 지금 ATE 를 "ATE before GBA" 칸에 고정, 끝나면 "Global BA" 칸에 걸린 시간 (ATE 칸은 계속 지금 궤적)
  void global_ba_start();
  void global_ba_done(double ms);

 private:
  void run();
  void draw_map(bool pts, bool lines, bool gt, const Eigen::Matrix4d* S);
  // 지금까지 궤적 -> GT Sim(3) (X_gt = S X_vo), 대응 프레임 수 < 3이면 false
  bool sim3_to_gt(Eigen::Matrix4d& S, double& ate) const;

  Camera cam_;
  std::vector<Pose> gt_cw_;
  bool align_init_ = false;  // "Sim3 align" 체크박스 초기값
  int f0_ = -1;  // 처음 포즈가 생긴 프레임
  std::thread th_;
  std::atomic<bool> pause_{false}, closed_{false}, quit_{false};
  std::string snapshot_;
  std::string ate_pre_ = "-", gba_state_ = "-";  // 전역 BA 전 ATE, 전역 BA 상태
  std::optional<RunOptions> start_;  // 시작 화면 기본값 / Start 뒤에는 고른 값
  bool started_ = false;
  std::condition_variable start_cv_;

  std::mutex mu_;  // 아래 스냅샷 보호
  cv::Mat img_;
  bool img_new_ = false;
  std::vector<Vec3> traj_, pts_all_, pts_vis_;
  std::unordered_map<int, std::pair<Vec3, Vec3>> segs_;  // map line id -> 3D segment (마지막 관측 끝점 기준)
  std::vector<int> segs_vis_;
  Pose cur_;
  bool has_cur_ = false;
};

}  // namespace vp
