#pragma once

// Node recorders (OpenSees `recorder Node -file ... -time -dT ... -node ... -dof ... disp|vel|
// accel`), with OpenSees' recording rule and text layout (NodeRecorder.cpp, DataFileStream.cpp,
// commit 955ceb0):
//   record when dT == 0 or time - next >= -dT * 1e-5, then next = time + dT;
//   one line per record: [time] then every node's DOFs, `setprecision(precision)` values
//   separated by one space.
//
// All data stays on the device. At a recording time one small kernel gathers just the
// recorded values into a staging array, which is copied asynchronously into pinned host
// memory (double-buffered); a separate host thread formats and writes the files, so the
// device never waits for the disk. The host waits for the device only at recording times.

#include <Kokkos_Core.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "compile/batches.hpp"
#include "io/model.hpp"

namespace femorax::exec {

class NodeRecorders {
 public:
  using Space = Kokkos::DefaultExecutionSpace;
  using Vec = Kokkos::View<double*, Space>;

  NodeRecorders(const io::Model& model, const compile::Compiled& c);
  ~NodeRecorders();
  NodeRecorders(const NodeRecorders&) = delete;
  NodeRecorders& operator=(const NodeRecorders&) = delete;

  // Start recording `name` (a process step). Returns false if it is not a Node recorder.
  bool activate(const std::string& name);
  // After a committed step at `time`: record every active recorder that is due.
  void record(double time, const Vec& disp, const Vec& vel, const Vec& accel);
  // Write everything still pending and close the files.
  void finish();

  std::int64_t records_written() const { return records_; }

 private:
  struct Recorder {
    std::string name, file;
    io::NodeResponse response;
    bool echo_time = false;
    double dT = 0.0, next = 0.0;
    int precision = 6;
    bool active = false;
    std::vector<std::int32_t> eq_host;      // per column (node-major), -1 = fixed
    Kokkos::View<std::int32_t*, Space> eq;  // the same, on the device
    std::int64_t stage_offset = 0;
  };
  struct Job {
    int recorder;
    double time;
    std::vector<double> values;
  };
  struct Pending {
    int slot;
    std::vector<std::pair<int, double>> due;  // (recorder, time)
  };

  void flush_pending();
  void writer_loop();

  std::vector<Recorder> recorders_;
  Vec stage_;
  Kokkos::View<double*, Kokkos::SharedHostPinnedSpace> pinned_[2];
  int next_slot_ = 0;
  std::vector<Pending> pending_;
  std::atomic<std::int64_t> records_{0};

  std::vector<std::unique_ptr<std::ofstream>> files_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Job> jobs_;
  bool stop_ = false;
  std::thread writer_;
};

}  // namespace femorax::exec
