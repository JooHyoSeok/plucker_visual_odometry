// Loop closure with pluggable stages + Sim(3) pose graph (g2o) + map correction (port of python/loop_closure.py
// and the feedback part of python/slam_online.py).
//
//   PlaceRecognizer : keyframe image -> similar past keyframes (built-in: thumbnail correlation;
//                     NetVLAD / SALAD via the Python plugin server python/loop_server.py)
//   LoopVerifier    : (current kf, candidate kf) -> relative Sim(3) cam_j -> cam_i + inliers
//                     (built-in: ORB + F-RANSAC + 3D-3D Sim(3) RANSAC on VO-triangulated local 3D;
//                      MASt3R via the plugin server: metric relative pose + VO/metric depth ratios -> Sim(3))
//   LoopCloser      : at every keyframe: register -> search -> verify -> Sim(3) PGO over all keyframes
//                     -> write the result back into poses, map points, lines and stored measurements, so tracking
//                        continues on the corrected map. Measurements are kept as originally measured and only
//                        rescaled when the camera units change (re-measuring odometry from corrected poses would
//                        bake earlier corrections into the graph).
//
// Threading (ORB-SLAM3 LoopClosing 구조): 트래킹 스레드는 프레임 처리 동안 map lock 을 잡고 (mMutexMapUpdate),
// 새 키프레임은 큐에 넣고 바로 다음 프레임으로 간다. 루프 스레드는 큐를 꺼내서
//   1) lock 잠깐: 키프레임 상태 복사 (KFState)      2) lock 없이: 등록 / 후보 검색 / 검증
//   3) lock 잠깐: PGO 정점 초기값 = 키프레임 포즈 복사  4) lock 없이: Sim(3) PGO
//   5) lock: 결과 써넣기. 최적화 동안 트래킹이 만든 프레임/키프레임은 마지막 PGO 키프레임 기준 상대포즈로 따라감
//      (ORB-SLAM3 의 spanning tree 전파와 같은 역할). 트래킹이 멈추는 건 5) 뿐.
// 옛 키프레임 포즈를 바꾸는 건 루프 스레드뿐이라 3)~5) 사이에 정점 초기값이 낡지 않는다 (local BA 를 같이 켜면
// 최근 키프레임은 예외, 작은 오차로 둠).

#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <random>
#include <string>
#include <thread>

#include "vo_plucker/tracker.h"

namespace vp {

struct Sim3 {  // X' = s R X + t
  double s = 1;
  Mat3 R = Mat3::Identity();
  Vec3 t = Vec3::Zero();
  Sim3 operator*(const Sim3& b) const { return Sim3{s * b.s, R * b.R, s * R * b.t + t}; }
  Sim3 inverse() const { return Sim3{1 / s, R.transpose(), -(1 / s) * R.transpose() * t}; }
  Vec3 map(const Vec3& X) const { return s * R * X + t; }
};

struct LoopEdge {
  int i = -1, j = -1;  // keyframe frame indices, i = current (newer)
  Sim3 S;              // cam_j -> cam_i (VO units of each keyframe)
  int inliers = 0;
};

class PlaceRecognizer {
 public:
  virtual ~PlaceRecognizer() = default;
  virtual void add(int kf, const cv::Mat& img) = 0;  // img: BGR (또는 gray)
  // kf 와 비슷한 키프레임을 cands 중에서 유사도 순으로 최대 k개
  virtual std::vector<int> query(int kf, const std::vector<int>& cands, int k) = 0;
};

// 검증기가 키프레임 등록 때 쓰는 VO 상태. 루프 스레드가 map lock 아래에서 복사해 둔다
struct KFState {
  int kf = -1;
  Pose pose;              // world -> cam
  int prev = -1;          // 직전에 등록된 키프레임 (없으면 -1), 포즈는 같은 시점의 값
  Pose prev_pose;
  std::vector<Vec3> uvd;  // 이 키프레임에 보이는 맵 점 (u, v, VO 깊이), 최대 300개
};

class LoopVerifier {
 public:
  virtual ~LoopVerifier() = default;
  // 키프레임 등록. 반환: 이제 검증 가능해진 키프레임 (내장 ORB: 다음 키프레임이 와야 로컬 3D가 생김) 또는 -1
  virtual int add(const KFState& st, const cv::Mat& img) = 0;
  virtual std::optional<LoopEdge> verify(int i, int j) = 0;
  virtual void rescale(int /*kf*/, double /*s*/) {}  // 맵 보정: 키프레임 카메라 좌표가 1/s 배
};

std::unique_ptr<PlaceRecognizer> make_thumbnail_recognizer();
std::unique_ptr<LoopVerifier> make_orb_verifier(const Camera& cam);

// Python 플러그인 서버 (stdin/stdout 줄 프로토콜, python/loop_server.py 참고). 한 프로세스를 후보 검색과 검증이 공유
struct LoopServer;
std::shared_ptr<LoopServer> start_loop_server(const std::string& cmd);
std::unique_ptr<PlaceRecognizer> make_py_recognizer(std::shared_ptr<LoopServer> srv, double min_sim);
std::unique_ptr<LoopVerifier> make_py_verifier(std::shared_ptr<LoopServer> srv, const Camera& cam);
// ORB F-RANSAC 기하 검증 (빠름) 을 먼저 통과한 후보만 main 으로 검증
std::unique_ptr<LoopVerifier> make_cascade_verifier(const Camera& cam, std::unique_ptr<LoopVerifier> main);

struct LoopParams {
  int min_gap = 150;   // 루프 후보는 이 프레임 수 이상 떨어진 과거 키프레임만
  int shortlist = 15;  // 검증할 후보 수
  bool feedback = true;  // PGO 결과를 포즈/맵에 되먹임 (false: 결과만 기록)
  bool threaded = true;  // false: 트래킹 스레드에서 그 자리에서 처리 (결과 재현용)
  int max_backlog = 2;   // 큐에 이보다 많이 밀려 있으면 오래된 키프레임은 버림 (등록도 검출도 안 함)
  double skip_chi2 = 0;  // 새 루프 엣지의 현재 포즈 chi2 가 이보다 작으면 (이미 맞음) 엣지만 추가, PGO 생략
};

class LoopCloser {
 public:
  LoopCloser(VOTracker& vo, std::unique_ptr<PlaceRecognizer> pr, std::unique_ptr<LoopVerifier> ver,
             const LoopParams& prm = {});
  ~LoopCloser();
  // 트래킹 스레드: 새 키프레임마다. threaded 면 큐에 넣고 바로 반환, 아니면 그 자리에서 처리
  void push(int kf, const cv::Mat& img);
  // 트래킹 스레드가 vo 를 읽고 쓰는 동안 잡는 lock. 루프 스레드가 기다리고 있으면 먼저 양보. threaded 아니면 빈 lock
  std::unique_lock<std::mutex> lock_map();
  // 남은 큐를 끝까지 처리하고 스레드 종료 (궤적 저장 전에 호출, map lock 은 놓은 상태로)
  void finish();

  // 아래 통계는 finish() 뒤에 읽기
  std::vector<LoopEdge> loops;  // 검출된 루프 (측정값 원본)
  double ms_detect = 0, ms_pgo = 0, ms_opt = 0;  // ms_opt: g2o optimize() 만
  double ms_blocked = 0, ms_blocked_max = 0;  // 루프 스레드가 map lock 을 잡은 시간 = 트래킹이 멈춘 시간
  int n_queries = 0, n_skipped = 0;            // n_skipped: 루프 스레드가 밀려서 버린 키프레임
  int n_pgo_skipped = 0;                       // 이미 맞는 루프라 PGO 생략
  std::atomic<int> n_corrections{0};           // 맵 보정을 써넣은 횟수 (트래킹 스레드가 map lock 아래에서 읽음)

 private:
  void run();
  void process(int kf, const cv::Mat& img);
  std::unique_lock<std::mutex> take_map();  // 루프 스레드 쪽 lock (대기 중 표시 -> 트래킹이 양보)
  void blocked(double ms) { ms_blocked += ms, ms_blocked_max = std::max(ms_blocked_max, ms); }
  KFState snapshot(int kf) const;
  std::unordered_map<int, Sim3> pgo(const std::vector<Pose>& P, const std::vector<int>& sub, double& cost);
  void correct(const std::unordered_map<int, Sim3>& S_kf, const std::vector<int>& sub);

  VOTracker& vo_;

  std::unique_ptr<PlaceRecognizer> pr_;
  std::unique_ptr<LoopVerifier> ver_;
  LoopParams prm_;
  std::vector<int> kfs_;          // 등록 순서 = 프레임 순서
  std::vector<int> ready_;        // 검증 가능한 키프레임 (루프 후보)
  std::unordered_map<int, Sim3> odom_;  // kf a -> 다음 키프레임까지 측정 (cam_a -> cam_b), 보정 때 단위만 맞춤
  std::vector<LoopEdge> edges_;   // PGO 루프 엣지 (보정 때 단위만 맞춤)

  std::mutex map_mx_;              // vo 접근 (트래킹 <-> 루프 스레드)
  std::atomic<bool> want_map_{false};
  std::mutex q_mx_;                // 큐
  std::condition_variable q_cv_;
  std::deque<std::pair<int, cv::Mat>> queue_;
  bool stop_ = false;
  std::thread th_;
};

}  // namespace vp
