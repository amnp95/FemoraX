#pragma once

// Recorders: Node (text) and VTKHDF, with OpenSees' recording rules and file layouts.
//
// Node (`recorder Node -file ... -time -dT ... -node ... -dof ... disp|vel|accel`;
// NodeRecorder.cpp, DataFileStream.cpp, commit 955ceb0):
//   record when dT == 0 or time - next >= -dT * 1e-5, then next = time + dT;
//   one line per record: [time] then every node's DOFs, `setprecision(precision)` values
//   separated by one space.
// VTKHDF (`recorder vtkhdf file -dT ... -rTolDt ... disp vel accel`; VTKHDF_Recorder.cpp):
//   the file is created when the recorder starts; the mesh (all nodes in tag order, all
//   elements in tag order, or the listed elements and their nodes) is written on the first
//   committed step; a step is recorded when dT == 0 or time - next >= -dT * rTolDt, then
//   next = time + dT. Layout: io/vtkhdf_writer.hpp.
//
// All data stays on the device. At a recording time one small kernel per due recorder
// gathers just the recorded values into a staging array, and only those ranges are copied
// asynchronously into pinned host memory (double-buffered). A separate host thread formats
// and writes every file (it also owns all HDF5 calls), so the device never waits for the
// disk. The host waits for the device only at recording times.

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
#include "exec/internal_force.hpp"
#include "io/model.hpp"
#include "io/vtkhdf_writer.hpp"

namespace femorax::exec {

class Recorders {
 public:
  using Space = Kokkos::DefaultExecutionSpace;
  using Vec = Kokkos::View<double*, Space>;

  // single_precision: VTKHDF responses written as float32 (io/vtkhdf_writer.hpp).
  Recorders(const io::Model& model, const compile::Compiled& c, bool single_precision = false);
  // Bytes the constructor will allocate in the default memory space (the staging buffer and
  // the equation maps, and on host backends the two pinned buffers), without allocating.
  static std::int64_t device_bytes_estimate(const io::Model& model, const compile::Compiled& c);
  ~Recorders();
  Recorders(const Recorders&) = delete;
  Recorders& operator=(const Recorders&) = delete;

  // The source of element output (VTKHDF stress3D6 / strain3D6): its current u and
  // committed material state at each record. Needed before the first record if any
  // recorder has element output.
  void attach(InternalForce& force);

  // Start recording `name` (a process step). Returns false if there is no such recorder.
  bool activate(const std::string& name);
  // After a committed step at `time`: record every active recorder that is due.
  void record(double time, const Vec& disp, const Vec& vel, const Vec& accel);
  // Write everything still pending and close the files. Rethrows a writer error.
  void finish();

  std::int64_t records_written() const { return records_; }

 private:
  enum class Kind { Node, Vtkhdf };
  struct Recorder {
    Kind kind = Kind::Node;
    std::string name, file;
    std::vector<io::NodeResponse> responses;  // Node: one; VTKHDF: in file order
    bool echo_time = false;
    double dT = 0.0, rtol = 1e-5, next = 0.0;
    int precision = 6;
    bool active = false, mesh_sent = false;
    std::int64_t columns = 0;               // values per response
    Kokkos::View<std::int32_t*, Space> eq;  // per column, -1 = fixed (value 0)
    std::int64_t stage_offset = 0;          // responses.size() * columns values
    std::shared_ptr<io::VtkhdfMesh> mesh;   // VTKHDF only
    std::vector<int> cell_responses;        // VTKHDF element output: 0 stress3D6, 1 strain3D6
    std::int64_t cells = 0;                 // elements of the mesh
    int selection = -1;                     // InternalForce::select_elements
  };
  struct Job {
    enum class Type { Open, Mesh, Frame } type = Type::Frame;
    int recorder = 0;
    double time = 0.0;
    std::vector<double> values;
  };
  struct Pending {
    int slot;
    std::vector<std::pair<int, double>> due;  // (recorder, time)
  };

  std::int64_t size_of(const Recorder& r) const {
    return static_cast<std::int64_t>(r.responses.size()) * r.columns +
           static_cast<std::int64_t>(r.cell_responses.size()) * r.cells * 6;
  }
  void enqueue(std::vector<Job> jobs);
  void flush_pending();
  void writer_loop();
  void write(Job& job);

  std::vector<Recorder> recorders_;
  InternalForce* force_ = nullptr;
  Vec stage_;
  Kokkos::View<double*, Kokkos::SharedHostPinnedSpace> pinned_[2];
  int next_slot_ = 0;
  std::vector<Pending> pending_;
  std::atomic<std::int64_t> records_{0};

  // Owned by the writer thread once it runs.
  std::vector<std::unique_ptr<std::ofstream>> files_;
  std::vector<std::unique_ptr<io::VtkhdfWriter>> vtkhdf_;
  bool single_precision_ = false;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Job> jobs_;
  bool stop_ = false;
  std::string error_;  // first writer error, reported by finish()
  std::thread writer_;
};

}  // namespace femorax::exec
