#include "exec/recorders.hpp"

#include <cstdio>
#include <filesystem>
#include <unordered_map>

#include "io/h5.hpp"

namespace femorax::exec {

NodeRecorders::NodeRecorders(const io::Model& m, const compile::Compiled& c) {
  std::unordered_map<std::int32_t, std::int32_t> internal;
  for (std::int32_t i = 0; i < c.dofs.num_nodes(); ++i) internal[c.dofs.node_tag[i]] = i;
  std::int64_t offset = 0;
  for (const auto& rec : m.recorders) {
    const auto* nr = std::get_if<io::NodeRecorder>(&rec.data);
    if (!nr) continue;
    Recorder r;
    r.name = rec.name;
    r.file = nr->file;
    r.response = nr->response;
    r.echo_time = nr->time;
    r.dT = nr->dT.value_or(0.0);
    r.precision = nr->precision;
    for (const auto tag : nr->nodes)
      for (const auto d : nr->dofs) r.eq_host.push_back(c.dofs.equation(internal.at(tag), d - 1));
    r.eq = Kokkos::View<std::int32_t*, Space>("recorder_eq", r.eq_host.size());
    auto h = Kokkos::create_mirror_view(r.eq);
    for (std::size_t i = 0; i < r.eq_host.size(); ++i) h(i) = r.eq_host[i];
    Kokkos::deep_copy(r.eq, h);
    r.stage_offset = offset;
    offset += static_cast<std::int64_t>(r.eq_host.size());
    recorders_.push_back(std::move(r));
  }
  stage_ = Vec("recorder_stage", static_cast<std::size_t>(offset));
  for (auto& p : pinned_)
    p = Kokkos::View<double*, Kokkos::SharedHostPinnedSpace>("recorder_pinned", static_cast<std::size_t>(offset));
  files_.resize(recorders_.size());
  writer_ = std::thread([this] { writer_loop(); });
}

NodeRecorders::~NodeRecorders() { finish(); }

bool NodeRecorders::activate(const std::string& name) {
  for (std::size_t i = 0; i < recorders_.size(); ++i) {
    auto& r = recorders_[i];
    if (r.name != name) continue;
    r.active = true;
    const std::filesystem::path p(r.file);
    std::error_code ec;
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
    files_[i] = std::make_unique<std::ofstream>(r.file, std::ios::trunc);
    if (!*files_[i]) throw io::ModelError("cannot open recorder file " + r.file);
    return true;
  }
  return false;
}

void NodeRecorders::record(double time, const Vec& disp, const Vec& vel, const Vec& accel) {
  std::vector<std::pair<int, double>> due;
  for (std::size_t i = 0; i < recorders_.size(); ++i) {
    auto& r = recorders_[i];
    if (!r.active) continue;
    // NodeRecorder::record: relDeltaTTol = 1e-5.
    if (r.dT == 0.0 || time - r.next >= -r.dT * 1e-5) {
      if (r.dT != 0.0) r.next = time + r.dT;
      due.emplace_back(static_cast<int>(i), time);
    }
  }
  if (due.empty()) return;

  // The previous record's copy is complete once the device reaches this point; hand it to
  // the writer before reusing a pinned buffer.
  if (pending_.size() >= 2) flush_pending();

  const int slot = next_slot_;
  next_slot_ ^= 1;
  for (const auto& [i, t] : due) {
    const auto& r = recorders_[i];
    const Vec src = r.response == io::NodeResponse::Disp ? disp
                    : r.response == io::NodeResponse::Vel ? vel
                                                          : accel;
    const auto eq = r.eq;
    const auto stage = stage_;
    const std::int64_t off = r.stage_offset;
    Kokkos::parallel_for(
        "recorder_gather", Kokkos::RangePolicy<Space>(0, static_cast<int>(r.eq_host.size())),
        KOKKOS_LAMBDA(const int k) {
          const std::int32_t q = eq(k);
          stage(off + k) = q >= 0 ? src(q) : 0.0;  // fixed DOFs: 0
        });
  }
  Kokkos::deep_copy(Space(), pinned_[slot], stage_);  // asynchronous on device backends
  pending_.push_back({slot, due});
}

void NodeRecorders::flush_pending() {
  Kokkos::fence("recorder copies");
  std::vector<Job> jobs;
  for (const auto& p : pending_) {
    const auto& buf = pinned_[p.slot];
    for (const auto& [i, t] : p.due) {
      const auto& r = recorders_[i];
      Job j{i, t, {}};
      j.values.assign(buf.data() + r.stage_offset, buf.data() + r.stage_offset + r.eq_host.size());
      jobs.push_back(std::move(j));
    }
  }
  pending_.clear();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& j : jobs) jobs_.push_back(std::move(j));
  }
  cv_.notify_one();
}

void NodeRecorders::writer_loop() {
  std::vector<char> line;
  for (;;) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
      if (jobs_.empty()) return;  // stop_ and nothing left
      job = std::move(jobs_.front());
      jobs_.pop_front();
    }
    const auto& r = recorders_[job.recorder];
    auto& out = *files_[job.recorder];
    char buf[64];
    std::string s;
    // DataFileStream: std::setprecision(p) in default float format = printf %.{p}g.
    if (r.echo_time) {
      std::snprintf(buf, sizeof(buf), "%.*g", r.precision, job.time);
      s += buf;
    }
    for (std::size_t k = 0; k < job.values.size(); ++k) {
      std::snprintf(buf, sizeof(buf), "%.*g", r.precision, job.values[k]);
      if (!s.empty()) s += ' ';
      s += buf;
    }
    s += '\n';
    out << s;
    ++records_;
  }
}

void NodeRecorders::finish() {
  if (!writer_.joinable()) return;
  if (!pending_.empty()) flush_pending();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_one();
  writer_.join();
  for (auto& f : files_)
    if (f) f->close();
}

}  // namespace femorax::exec
