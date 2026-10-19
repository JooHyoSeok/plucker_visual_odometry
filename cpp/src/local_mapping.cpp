#include "vo_plucker/local_mapping.h"

#include <algorithm>
#include <chrono>

namespace vp {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
}  // namespace

LocalMapper::LocalMapper(VOTracker& vo, int window, int max_fixed, bool lines, bool mature_only)
    : vo_(vo), window_(window), max_fixed_(max_fixed), lines_(lines), mature_only_(mature_only) {
  th_ = std::thread(&LocalMapper::run, this);
}

LocalMapper::~LocalMapper() { finish(); }

void LocalMapper::push(int kf) {
  {
    std::lock_guard<std::mutex> l(q_mx_);
    queue_.push_back(kf);
  }
  q_cv_.notify_one();
}

void LocalMapper::finish() {
  if (!th_.joinable()) return;
  {
    std::lock_guard<std::mutex> l(q_mx_);
    stop_ = true;
  }
  q_cv_.notify_one();
  th_.join();
}

void LocalMapper::run() {
  for (;;) {
    {
      std::unique_lock<std::mutex> l(q_mx_);
      q_cv_.wait(l, [&] { return stop_ || !queue_.empty(); });
      if (queue_.empty()) return;
      n_dropped += queue_.size() - 1;  // 창은 어차피 최신 키프레임에서 끝남 -> 밀린 건 한 번에
      queue_.clear();
    }
    LocalBAPtr P;
    {
      auto l = vo_.map_lock.worker();
      const auto tb = Clock::now();
      P = local_ba_build(vo_, window_, max_fixed_, lines_, mature_only_);
      const double b = ms_since(tb);
      ms_blocked += b, ms_blocked_max = std::max(ms_blocked_max, b);
    }
    if (!P) continue;
    const auto ts = Clock::now();
    local_ba_solve(*P);  // lock 없이: 트래킹은 계속
    ms_solve += ms_since(ts);
    LocalBAStats st;
    {
      auto l = vo_.map_lock.worker();
      const auto tb = Clock::now();
      st = local_ba_apply(vo_, *P);
      const double b = ms_since(tb);
      ms_blocked += b, ms_blocked_max = std::max(ms_blocked_max, b);
    }
    if (st.discarded) {
      ++n_discarded;
      continue;
    }
    ++n_ba, points += st.points, outliers += st.outliers, lines_used += st.lines, line_outliers += st.line_outliers;
    depth_priors += st.depth_priors;
    rms_before += st.rms_before, rms_after += st.rms_after;
  }
}

}  // namespace vp
