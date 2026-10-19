// Local BA in its own thread (ORB-SLAM3 LocalMapping): the tracking thread pushes new keyframes and keeps going;
// the mapper takes the newest one (older queued keyframes are dropped: the window always ends at the latest anyway),
//   build the window problem under the map lock (copies poses / points / lines / observations)
//   -> optimize without the lock -> write back under the lock (frames tracked meanwhile follow their keyframe).
// A loop / anchor correction in between bumps vo.map_version and the stale result is discarded.
#pragma once
#include <condition_variable>
#include <deque>
#include <thread>

#include "vo_plucker/optimizer.h"

namespace vp {

class LocalMapper {
 public:
  LocalMapper(VOTracker& vo, int window, int max_fixed, bool lines, bool mature_only);
  ~LocalMapper();
  void push(int kf);  // 트래킹 스레드: 새 키프레임 (map lock 은 놓은 상태로)
  void finish();      // 남은 것 처리 + 종료

  // 통계 (finish() 뒤에 읽기)
  int n_ba = 0, n_discarded = 0, n_dropped = 0;
  long points = 0, outliers = 0, lines_used = 0, line_outliers = 0;
  double ms_solve = 0, ms_blocked = 0, ms_blocked_max = 0, rms_before = 0, rms_after = 0;

 private:
  void run();
  VOTracker& vo_;
  int window_, max_fixed_;
  bool lines_, mature_only_;
  std::mutex q_mx_;
  std::condition_variable q_cv_;
  std::deque<int> queue_;
  bool stop_ = false;
  std::thread th_;
};

}  // namespace vp
