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
  std::vector<Vec3> uvd;  // 이 키프레임에 보이는 맵 점 (u, v, VO 깊이), 최대 3000개
  std::vector<int> pid;   // uvd 와 같은 순서의 맵 점 (트랙) id
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
std::unique_ptr<LoopVerifier> make_orb_verifier(const Camera& cam, int min_f = 80);  // min_f: F-RANSAC 인라이어 최소

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
  int consist_n = 1;          // ORB-SLAM3 식 연속 일관성: 연속 질의 키프레임 이 수만큼 같은 장소를 가리켜야 채택 (1 = 끔)
  int consist_frames = 30;    //   "같은 장소": 후보 프레임 차이가 이 이내, 질의 키프레임 간격도 이 이내
  double prune_ratio = 0;     // PGO 뒤 루프 엣지 chi2 가 루프 엣지 중앙값의 이 배수보다 크면 (그리고 > 100) 빼고 재최적화 (0 = 끔)
  bool fuse = false;          // 루프 보정 뒤 옛 맵 점과 현재 트랙 융합 (ORB 디스크립터 + 투영) -> covisibility -> 연결된 후보 제외
  int covis_min = 15;         //   질의 키프레임과 후보가 대표 맵 점을 이만큼 공유하면 이미 연결 -> 검증 안 함
  double fuse_px = 8;         //   보정된 옛 점을 현재 키프레임에 투영했을 때 이 안이어야 융합
  int fuse_nb = 2;            //   옛 키프레임 j 앞뒤로 이만큼의 키프레임까지 융합
  int anchor_every = 10;      // metric depth 앵커: 이 키프레임 수마다 (루프 없이도) PGO
  double anchor_sigma = 0.08; // 앵커 log-scale 표준편차 (DA3 프레임 간 흔들림 ~6 %)
  double anchor_min_dev = 0;  // 지난 앵커 PGO 뒤 앵커들의 |log(a_k/a_ref)| 중앙값이 이보다 작으면 PGO 생략 (스케일이 맞음)
};

class LoopCloser {
 public:
  // pr/ver 가 없으면 루프 검출 없이 동작 (anchor 만). anchor: metric depth 서버 (--depth da3) -> 키프레임 스케일 앵커
  LoopCloser(VOTracker& vo, std::unique_ptr<PlaceRecognizer> pr, std::unique_ptr<LoopVerifier> ver,
             const LoopParams& prm = {}, std::shared_ptr<LoopServer> anchor = nullptr);
  ~LoopCloser();
  // 트래킹 스레드: 새 키프레임마다. threaded 면 큐에 넣고 바로 반환, 아니면 그 자리에서 처리
  void push(int kf, const cv::Mat& img);
  // 남은 큐를 끝까지 처리하고 스레드 종료 (궤적 저장 전에 호출, map lock 은 놓은 상태로)
  void finish();

  // 아래 통계는 finish() 뒤에 읽기
  std::vector<LoopEdge> loops;  // 검출된 루프 (측정값 원본)
  double ms_detect = 0, ms_pgo = 0, ms_opt = 0;  // ms_opt: g2o optimize() 만
  double ms_blocked = 0, ms_blocked_max = 0;  // 루프 스레드가 map lock 을 잡은 시간 = 트래킹이 멈춘 시간
  int n_queries = 0, n_skipped = 0;            // n_skipped: 루프 스레드가 밀려서 버린 키프레임
  int n_pgo_skipped = 0;                       // 이미 맞는 루프라 PGO 생략
  int n_anchors = 0, n_anchor_pgo = 0, n_anchor_skip = 0;  // 앵커가 붙은 키프레임, 앵커 PGO, 스케일이 맞아서 건너뛴 PGO
  int n_unconfirmed = 0, n_pruned = 0;         // 연속 일관성을 못 채운 검출, PGO 뒤 빠진 루프 엣지
  int n_fused = 0, n_excluded = 0;             // 융합한 맵 점, covisibility 로 제외한 루프 후보
  std::atomic<int> n_corrections{0};           // 맵 보정을 써넣은 횟수 (트래킹 스레드가 map lock 아래에서 읽음)

 private:
  void run();
  void process(int kf, const cv::Mat& img);
  std::unique_lock<std::mutex> take_map() { return vo_.map_lock.worker(); }  // 작업 스레드 쪽 (트래킹이 양보)
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
  std::shared_ptr<LoopServer> anchor_;
  std::unordered_map<int, double> anchors_;  // kf -> VO 단위/미터 (그 키프레임 카메라 단위, 보정 때 단위만 맞춤)
  std::vector<double> first_anchors_;
  double a_ref_ = 0;                         // 목표 VO 단위/미터 (처음 10개 앵커 중앙값)
  int since_pgo_ = 0;
  std::vector<LoopEdge> pending_;  // 연속 일관성 확인 중인 검출 (아직 그래프에 안 넣음)
  // 융합 / covisibility (루프 스레드만): 키프레임 맵 점 디스크립터, 트랙 id -> 대표 (가장 옛) 맵 점 id
  struct MapDesc {
    std::vector<int> pid;
    std::vector<Vec2> uv;
    cv::Mat des;
  };
  std::unordered_map<int, MapDesc> map_desc_;
  std::unordered_map<int, std::vector<int>> kf_pids_;
  std::unordered_map<int, int> canon_;
  int root(int p);
  int shared(int a, int j);  // 대표 id 기준 공유 맵 점 수
  int fuse(int a, int j);    // map lock 아래: 융합한 점 수

  std::mutex q_mx_;                // 큐
  std::condition_variable q_cv_;
  std::deque<std::pair<int, cv::Mat>> queue_;
  bool stop_ = false;
  std::thread th_;
};

}  // namespace vp
