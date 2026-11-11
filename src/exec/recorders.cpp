#include "exec/recorders.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <type_traits>
#include <unordered_map>

#include "io/h5.hpp"

namespace femorax::exec {

namespace {

const char* vtkhdf_field(io::NodeResponse r) {
  switch (r) {
    case io::NodeResponse::Disp: return "displacement";
    case io::NodeResponse::Vel: return "velocity";
    case io::NodeResponse::Accel: break;
  }
  return "acceleration";
}

io::NodeResponse parse_response(const std::string& s) {
  if (s == "disp") return io::NodeResponse::Disp;
  if (s == "vel") return io::NodeResponse::Vel;
  return io::NodeResponse::Accel;  // the reader admits disp vel accel only
}

void make_parent(const std::string& file) {
  const std::filesystem::path p(file);
  std::error_code ec;
  if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
}

// VTKHDF_Recorder::writeMesh: points and cells, in OpenSees' domain order (tags ascending)
// or, for a list of elements, in list order with their nodes in ascending tag order.
io::VtkhdfMesh build_mesh(const io::Model& m, const std::vector<std::int32_t>& only,
                          std::vector<std::int32_t>& point_nodes) {
  std::unordered_map<std::int32_t, std::size_t> node_row;
  for (std::size_t i = 0; i < m.nodes.tags.size(); ++i) node_row[m.nodes.tags[i]] = i;
  struct Ele {
    const std::int32_t* conn;
    int nn;
  };
  std::map<std::int32_t, Ele> all;
  for (const auto& g : m.elements) {
    const int nn = io::node_count(g.type);
    for (std::size_t e = 0; e < g.tags.size(); ++e) all[g.tags[e]] = {g.connectivity.data() + e * nn, nn};
  }
  std::vector<std::int32_t> ele_tags;
  if (only.empty()) {
    for (const auto& [t, _] : all) ele_tags.push_back(t);
    point_nodes = m.nodes.tags;
    std::sort(point_nodes.begin(), point_nodes.end());
  } else {
    std::set<std::int32_t> nodes;
    for (const auto t : only) {
      const auto it = all.find(t);
      if (it == all.end()) continue;  // OpenSees skips missing elements
      ele_tags.push_back(t);
      for (int k = 0; k < it->second.nn; ++k) nodes.insert(it->second.conn[k]);
    }
    point_nodes.assign(nodes.begin(), nodes.end());
  }
  std::unordered_map<std::int32_t, std::int32_t> point_of;
  io::VtkhdfMesh mesh;
  const std::size_t ndm = m.nodes.tags.empty() ? 3 : m.nodes.coords.size() / m.nodes.tags.size();
  for (std::size_t p = 0; p < point_nodes.size(); ++p) {
    point_of[point_nodes[p]] = static_cast<std::int32_t>(p);
    const std::size_t row = node_row.at(point_nodes[p]);
    for (std::size_t k = 0; k < 3; ++k) mesh.points.push_back(k < ndm ? m.nodes.coords[row * ndm + k] : 0.0);
  }
  mesh.offsets.push_back(0);
  for (const auto t : ele_tags) {
    const auto& e = all.at(t);
    for (int k = 0; k < e.nn; ++k) mesh.connectivity.push_back(point_of.at(e.conn[k]));
    mesh.offsets.push_back(static_cast<std::int32_t>(mesh.connectivity.size()));
    mesh.types.push_back(io::kVtkHexahedron);  // stdBrick, SSPbrick
  }
  mesh.node_tags = point_nodes;
  mesh.ele_tags = ele_tags;
  return mesh;
}

}  // namespace

std::int64_t Recorders::device_bytes_estimate(const io::Model& m, const compile::Compiled& c) {
  (void)c;
  std::int64_t values = 0, columns = 0;
  for (const auto& rec : m.recorders) {
    if (const auto* nr = std::get_if<io::NodeRecorder>(&rec.data)) {
      const auto n = static_cast<std::int64_t>(nr->nodes.size() * nr->dofs.size());
      columns += n;
      values += n;
    } else if (const auto* vr = std::get_if<io::VtkhdfRecorder>(&rec.data)) {
      std::vector<std::int32_t> point_nodes;
      const io::VtkhdfMesh mesh = build_mesh(m, vr->elements, point_nodes);
      const auto cols = static_cast<std::int64_t>(point_nodes.size()) * 3;
      std::int64_t node_resp = 0, cell_resp = 0;
      for (const auto& s : vr->responses) (s == "stress3D6" || s == "strain3D6" ? cell_resp : node_resp) += 1;
      columns += cols;
      values += node_resp * cols + cell_resp * static_cast<std::int64_t>(mesh.ele_tags.size()) * 6;
    }
  }
  constexpr bool host = std::is_same_v<Space::memory_space, Kokkos::HostSpace>;
  return values * 8 + columns * 4 + (host ? 2 * values * 8 : 0);
}

Recorders::Recorders(const io::Model& m, const compile::Compiled& c, bool single_precision)
    : single_precision_(single_precision) {
  std::unordered_map<std::int32_t, std::int32_t> internal;
  for (std::int32_t i = 0; i < c.dofs.num_nodes(); ++i) internal[c.dofs.node_tag[i]] = i;
  auto upload = [](const std::vector<std::int32_t>& h) {
    Kokkos::View<std::int32_t*, Space> d("recorder_eq", h.size());
    auto hm = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < h.size(); ++i) hm(i) = h[i];
    Kokkos::deep_copy(d, hm);
    return d;
  };
  std::int64_t offset = 0;
  for (const auto& rec : m.recorders) {
    Recorder r;
    r.name = rec.name;
    std::vector<std::int32_t> eq;
    if (const auto* nr = std::get_if<io::NodeRecorder>(&rec.data)) {
      r.kind = Kind::Node;
      r.file = nr->file;
      r.responses = {nr->response};
      r.echo_time = nr->time;
      r.dT = nr->dT.value_or(0.0);
      r.precision = nr->precision;
      for (const auto tag : nr->nodes)
        for (const auto d : nr->dofs) eq.push_back(c.dofs.equation(internal.at(tag), d - 1));
    } else if (const auto* vr = std::get_if<io::VtkhdfRecorder>(&rec.data)) {
      r.kind = Kind::Vtkhdf;
      r.file = vr->file;
      for (const auto& s : vr->responses) {
        if (s == "stress3D6") r.cell_responses.push_back(0);
        else if (s == "strain3D6") r.cell_responses.push_back(1);
        else r.responses.push_back(parse_response(s));
      }
      r.dT = vr->dT.value_or(0.0);
      r.rtol = std::max(0.0, vr->rTolDt.value_or(1e-5));
      std::vector<std::int32_t> point_nodes;
      r.mesh = std::make_shared<io::VtkhdfMesh>(build_mesh(m, vr->elements, point_nodes));
      r.cells = static_cast<std::int64_t>(r.mesh->ele_tags.size());
      // Node::getDisp() components 0..2 of each point (OpenSees reads up to the node's ndm).
      for (const auto tag : point_nodes) {
        const std::int32_t n = internal.at(tag);
        for (int k = 0; k < 3; ++k) eq.push_back(k < c.dofs.ndf[static_cast<std::size_t>(n)] ? c.dofs.equation(n, k) : -1);
      }
    } else {
      continue;
    }
    r.columns = static_cast<std::int64_t>(eq.size());
    r.eq = upload(eq);
    r.stage_offset = offset;
    offset += size_of(r);
    recorders_.push_back(std::move(r));
  }
  stage_ = Vec("recorder_stage", static_cast<std::size_t>(offset));
  for (auto& p : pinned_)
    p = Kokkos::View<double*, Kokkos::SharedHostPinnedSpace>("recorder_pinned", static_cast<std::size_t>(offset));
  files_.resize(recorders_.size());
  vtkhdf_.resize(recorders_.size());
  writer_ = std::thread([this] { writer_loop(); });
}

Recorders::~Recorders() {
  try {
    finish();
  } catch (...) {
  }
}

void Recorders::attach(InternalForce& force) {
  force_ = &force;
  for (auto& r : recorders_)
    if (!r.cell_responses.empty()) r.selection = force.select_elements(r.mesh->ele_tags);
}

bool Recorders::activate(const std::string& name) {
  for (std::size_t i = 0; i < recorders_.size(); ++i) {
    auto& r = recorders_[i];
    if (r.name != name) continue;
    r.active = true;
    std::vector<Job> jobs(1);
    jobs[0].type = Job::Type::Open;
    jobs[0].recorder = static_cast<int>(i);
    enqueue(std::move(jobs));
    return true;
  }
  return false;
}

void Recorders::record(double time, const Vec& disp, const Vec& vel, const Vec& accel) {
  std::vector<std::pair<int, double>> due;
  std::vector<Job> meshes;
  for (std::size_t i = 0; i < recorders_.size(); ++i) {
    auto& r = recorders_[i];
    if (!r.active) continue;
    if (r.kind == Kind::Vtkhdf && !r.mesh_sent) {  // VTKHDF_Recorder::record: writeMesh first
      r.mesh_sent = true;
      Job j;
      j.type = Job::Type::Mesh;
      j.recorder = static_cast<int>(i);
      meshes.push_back(std::move(j));
    }
    if (r.dT == 0.0 || time - r.next >= -r.dT * r.rtol) {
      if (r.dT != 0.0) r.next = time + r.dT;
      due.emplace_back(static_cast<int>(i), time);
    }
  }
  if (!meshes.empty()) enqueue(std::move(meshes));
  if (due.empty()) return;

  // The previous record's copy is complete once the device reaches this point; hand it to
  // the writer before reusing a pinned buffer.
  if (pending_.size() >= 2) flush_pending();

  const int slot = next_slot_;
  next_slot_ ^= 1;
  const auto stage = stage_;
  for (const auto& [i, t] : due) {
    const auto& r = recorders_[i];
    const auto eq = r.eq;
    for (std::size_t k = 0; k < r.responses.size(); ++k) {
      const Vec src = r.responses[k] == io::NodeResponse::Disp ? disp
                      : r.responses[k] == io::NodeResponse::Vel ? vel
                                                                : accel;
      const std::int64_t off = r.stage_offset + static_cast<std::int64_t>(k) * r.columns;
      Kokkos::parallel_for(
          "recorder_gather", Kokkos::RangePolicy<Space, Kokkos::IndexType<std::int64_t>>(0, r.columns),
          KOKKOS_LAMBDA(const std::int64_t j) {
            const std::int32_t q = eq(j);
            stage(off + j) = q >= 0 ? src(q) : 0.0;  // fixed DOFs: 0
          });
    }
    for (std::size_t k = 0; k < r.cell_responses.size(); ++k) {
      if (!force_ || r.selection < 0) throw io::ModelError("recorder " + r.name + ": element output without a source");
      const std::int64_t off = r.stage_offset + static_cast<std::int64_t>(r.responses.size()) * r.columns +
                               static_cast<std::int64_t>(k) * r.cells * 6;
      force_->element_output(r.selection, r.cell_responses[k], stage, off);
    }
    const std::pair<std::int64_t, std::int64_t> range{r.stage_offset, r.stage_offset + size_of(r)};
    // Asynchronous on device backends: only this recorder's values cross to the host.
    Kokkos::deep_copy(Space(), Kokkos::subview(pinned_[slot], range), Kokkos::subview(stage_, range));
  }
  pending_.push_back({slot, due});
}

void Recorders::enqueue(std::vector<Job> jobs) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& j : jobs) jobs_.push_back(std::move(j));
  }
  cv_.notify_one();
}

void Recorders::flush_pending() {
  Kokkos::fence("recorder copies");
  std::vector<Job> jobs;
  for (const auto& p : pending_) {
    const auto& buf = pinned_[p.slot];
    for (const auto& [i, t] : p.due) {
      const auto& r = recorders_[i];
      Job j;
      j.recorder = i;
      j.time = t;
      j.values.assign(buf.data() + r.stage_offset, buf.data() + r.stage_offset + size_of(r));
      jobs.push_back(std::move(j));
    }
  }
  pending_.clear();
  enqueue(std::move(jobs));
}

void Recorders::write(Job& job) {
  const auto& r = recorders_[job.recorder];
  const auto i = static_cast<std::size_t>(job.recorder);
  if (job.type == Job::Type::Open) {
    make_parent(r.file);
    if (r.kind == Kind::Node) {
      files_[i] = std::make_unique<std::ofstream>(r.file, std::ios::trunc);
      if (!*files_[i]) throw io::ModelError("cannot open recorder file " + r.file);
    } else {
      std::vector<std::string> fields, cells;
      for (const auto resp : r.responses) fields.emplace_back(vtkhdf_field(resp));
      for (const int c : r.cell_responses) cells.emplace_back(c == 0 ? "stress3D6" : "strain3D6");
      vtkhdf_[i] = std::make_unique<io::VtkhdfWriter>(r.file, fields, cells, single_precision_);
    }
    return;
  }
  if (job.type == Job::Type::Mesh) {
    vtkhdf_[i]->write_mesh(*r.mesh);
    return;
  }
  if (r.kind == Kind::Vtkhdf) {
    std::vector<const double*> fields;
    for (std::size_t k = 0; k < r.responses.size(); ++k)
      fields.push_back(job.values.data() + static_cast<std::int64_t>(k) * r.columns);
    for (std::size_t k = 0; k < r.cell_responses.size(); ++k)
      fields.push_back(job.values.data() + static_cast<std::int64_t>(r.responses.size()) * r.columns +
                       static_cast<std::int64_t>(k) * r.cells * 6);
    vtkhdf_[i]->write_step(job.time, fields);
    ++records_;
    return;
  }
  auto& out = *files_[i];
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

void Recorders::writer_loop() {
  for (;;) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
      if (jobs_.empty()) break;  // stop_ and nothing left
      job = std::move(jobs_.front());
      jobs_.pop_front();
    }
    try {
      write(job);
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (error_.empty()) error_ = e.what();
    }
  }
  for (auto& f : files_)
    if (f) f->close();
  for (auto& v : vtkhdf_)
    if (v) v->close();
}

void Recorders::finish() {
  if (!writer_.joinable()) return;
  if (!pending_.empty()) flush_pending();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_one();
  writer_.join();
  if (!error_.empty()) throw io::ModelError("recorder: " + error_);
}

}  // namespace femorax::exec
