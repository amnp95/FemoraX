#include "compile/dof_numbering.hpp"

#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>

#include "io/h5.hpp"

namespace femorax::compile {

namespace {

[[noreturn]] void invalid(const std::string& msg) { throw io::ModelError(msg); }

// Union-find with the smallest DOF id as the root, so the result does not depend on the
// order of the constraint rows.
struct UnionFind {
  std::vector<std::int32_t> parent;
  explicit UnionFind(std::int32_t n) : parent(static_cast<std::size_t>(n)) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  std::int32_t find(std::int32_t a) {
    while (parent[a] != a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  }
  void join(std::int32_t a, std::int32_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (b < a) std::swap(a, b);
    parent[b] = a;
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// Emulation of OpenSees' TransformationConstraintHandler (commit 955ceb0):
//   TransformationConstraintHandler::handle, TransformationDOF_Group (constructor,
//   addSP_Constraint, doneID), PlainNumberer.
// Nodes are processed in ascending tag order (OpenSees' Domain stores nodes by tag). Each
// nodal DOF ends up fixed or equal to some node's own free DOF. The result is compared
// with the constraints as written (the union-find numbering): where they differ, OpenSees
// would silently compute something else, and FemoraX refuses the model.
namespace {

constexpr std::int64_t kFixed = -1;

struct Mp {
  std::int32_t retained = -1;     // internal node index
  std::vector<std::int32_t> dofs;  // 0-based constrained (= retained) DOFs, row order
};

std::string describe(std::int64_t v, const DofNumbering& n) {
  if (v == kFixed) return "fixed";
  const auto it = std::upper_bound(n.dof_offset.begin(), n.dof_offset.end(), v) - 1;
  const auto node = static_cast<std::int32_t>(it - n.dof_offset.begin());
  return "node " + std::to_string(n.node_tag[node]) + " DOF " +
         std::to_string(v - n.dof_offset[node] + 1);
}

}  // namespace

std::vector<std::int64_t> emulate_transformation_handler(const io::Model& model,
                                                         const DofNumbering& n) {
  std::unordered_map<std::int32_t, std::int32_t> internal_of_tag;
  for (std::int32_t i = 0; i < n.num_nodes(); ++i) internal_of_tag[n.node_tag[i]] = i;
  const std::int32_t N = n.num_nodes();
  std::vector<Mp> mp(static_cast<std::size_t>(N));
  for (std::size_t r = 0; r < model.equal_dof.retained.size(); ++r) {
    Mp& m = mp[internal_of_tag.at(model.equal_dof.constrained[r])];
    m.retained = internal_of_tag.at(model.equal_dof.retained[r]);
    m.dofs.push_back(model.equal_dof.dof[r] - 1);
  }
  std::vector<std::vector<std::int32_t>> sps(static_cast<std::size_t>(N));
  for (std::size_t r = 0; r < model.fix.node.size(); ++r)
    sps[internal_of_tag.at(model.fix.node[r])].push_back(model.fix.dof[r] - 1);

  // ID arrays as OpenSees holds them: plain groups by nodal DOF; transformed groups by
  // modID position (the node's unconstrained DOFs, then its MP's DOFs). Values are a
  // node's own free DOF (its global id) or kFixed.
  std::vector<std::vector<std::int64_t>> id(static_cast<std::size_t>(N));
  std::vector<std::vector<std::int32_t>> unconstrained(static_cast<std::size_t>(N));
  for (std::int32_t i = 0; i < N; ++i) {
    const std::int32_t ndf = n.ndf[i];
    if (mp[i].retained < 0) {
      id[i].resize(static_cast<std::size_t>(ndf));
      for (std::int32_t d = 0; d < ndf; ++d) id[i][d] = n.dof_offset[i] + d;
      for (const auto s : sps[i]) id[i][s] = kFixed;  // addSP_Constraint without an MP
      continue;
    }
    const auto& D = mp[i].dofs;
    for (std::int32_t d = 0; d < ndf; ++d)
      if (std::find(D.begin(), D.end(), d) == D.end()) unconstrained[i].push_back(d);
    const auto nU = static_cast<std::int32_t>(unconstrained[i].size());
    id[i].assign(static_cast<std::size_t>(nU + D.size()), kFixed);  // retained part: -1
    for (std::int32_t p = 0; p < nU; ++p) id[i][p] = n.dof_offset[i] + unconstrained[i][p];
    for (const auto s : sps[i]) {
      // addSP_Constraint: loc = number of unconstrained DOFs below s, whatever s is.
      std::int32_t loc = 0;
      for (std::int32_t d = 0; d < s; ++d)
        if (std::find(D.begin(), D.end(), d) == D.end()) ++loc;
      id[i][loc] = kFixed;
    }
  }

  // doneID in ascending tag order: copy the retained node's current ID entries.
  std::vector<std::int32_t> by_tag(static_cast<std::size_t>(N));
  std::iota(by_tag.begin(), by_tag.end(), 0);
  std::sort(by_tag.begin(), by_tag.end(),
            [&](std::int32_t a, std::int32_t b) { return n.node_tag[a] < n.node_tag[b]; });
  for (const auto i : by_tag) {
    if (mp[i].retained < 0) continue;
    const auto nU = static_cast<std::int32_t>(unconstrained[i].size());
    const auto& other = id[mp[i].retained];
    for (std::size_t k = 0; k < mp[i].dofs.size(); ++k) {
      const std::int32_t pos = mp[i].dofs[k];
      id[i][nU + k] = pos < static_cast<std::int32_t>(other.size()) ? other[pos] : kFixed;
    }
  }

  // Nodal DOF -> value through the transformation matrix.
  std::vector<std::int64_t> value(static_cast<std::size_t>(n.num_dofs()), kFixed);
  for (std::int32_t i = 0; i < N; ++i) {
    for (std::int32_t d = 0; d < n.ndf[i]; ++d) {
      std::int64_t v;
      if (mp[i].retained < 0) {
        v = id[i][d];
      } else {
        const auto& D = mp[i].dofs;
        const auto k = std::find(D.begin(), D.end(), d);
        if (k == D.end()) {
          const auto& U = unconstrained[i];
          v = id[i][std::find(U.begin(), U.end(), d) - U.begin()];
        } else {
          v = id[i][unconstrained[i].size() + (k - D.begin())];
        }
      }
      value[n.dof_offset[i] + d] = v;
    }
  }
  return value;
}

namespace {

// Compare with the numbering: same fixed set, and OpenSees values and FemoraX equations in
// one-to-one correspondence. Throws on the first DOF where they differ.
void check_against_transformation_handler(const io::Model& model, const DofNumbering& n) {
  const auto value = emulate_transformation_handler(model, n);
  std::unordered_map<std::int64_t, std::int32_t> eq_of_value;
  std::unordered_map<std::int32_t, std::int64_t> value_of_eq;
  for (std::int32_t i = 0; i < n.num_nodes(); ++i) {
    for (std::int32_t d = 0; d < n.ndf[i]; ++d) {
      const std::int64_t v = value[n.dof_offset[i] + d];
      const std::int32_t e = n.equation(i, d);
      bool same = (v == kFixed) == (e < 0);
      if (same && e >= 0) {
        const auto [a, fa] = eq_of_value.emplace(v, e);
        const auto [b, fb] = value_of_eq.emplace(e, v);
        same = a->second == e && b->second == v;
      }
      if (!same) {
        std::string as_written = e < 0 ? std::string("fixed") : "equation " + std::to_string(e);
        throw io::ModelError(
            "node " + std::to_string(n.node_tag[i]) + " DOF " + std::to_string(d + 1) +
            ": OpenSees' Transformation handler would make it " +
            (v == kFixed ? std::string("fixed") : "equal to " + describe(v, n)) +
            ", which is not what the fix and equalDOF constraints say (" + as_written +
            " here). OpenSees mishandles this pattern silently (a chain through a "
            "constrained node, a partial chain, or a fix on a constrained DOF); FemoraX "
            "refuses it rather than give different answers. Tie every node directly to its "
            "final retained node and fix retained nodes, not constrained DOFs");
      }
    }
  }
}

}  // namespace

std::vector<std::int32_t> file_order(const io::Model& model) {
  std::vector<std::int32_t> order(model.nodes.size());
  std::iota(order.begin(), order.end(), 0);
  return order;
}

DofNumbering number_dofs(const io::Model& model, const std::vector<std::int32_t>& node_order,
                         bool check_opensees) {
  const auto N = static_cast<std::int32_t>(model.nodes.size());
  if (static_cast<std::int32_t>(node_order.size()) != N) invalid("node order has the wrong size");

  DofNumbering out;
  out.node_model_index = node_order;
  out.node_tag.resize(N);
  out.ndf.resize(N);
  out.dof_offset.assign(N + 1, 0);
  std::unordered_map<std::int32_t, std::int32_t> internal_of_tag;
  internal_of_tag.reserve(static_cast<std::size_t>(N));
  {
    std::vector<char> seen(static_cast<std::size_t>(N), 0);
    for (std::int32_t i = 0; i < N; ++i) {
      const std::int32_t m = node_order[i];
      if (m < 0 || m >= N || seen[m]) invalid("node order is not a permutation");
      seen[m] = 1;
      out.node_tag[i] = model.nodes.tags[m];
      out.ndf[i] = model.nodes.ndf[m];
      internal_of_tag[out.node_tag[i]] = i;
    }
  }
  std::int64_t total = 0;
  for (std::int32_t i = 0; i < N; ++i) {
    total += out.ndf[i];
    if (total > INT32_MAX) invalid("more than 2^31 DOFs: 32-bit DOF indices overflow");
    out.dof_offset[i + 1] = static_cast<std::int32_t>(total);
  }
  const std::int32_t ndof = out.num_dofs();
  auto dof_id = [&](std::int32_t tag, std::int32_t dof1) {
    return out.dof_offset[internal_of_tag.at(tag)] + (dof1 - 1);
  };

  // --- Constraint rules (OpenSees TransformationConstraintHandler) -------------------
  const auto& eqd = model.equal_dof;
  std::unordered_map<std::int32_t, std::int32_t> retained_of;  // constrained -> retained
  for (std::size_t r = 0; r < eqd.retained.size(); ++r) {
    const auto [it, fresh] = retained_of.emplace(eqd.constrained[r], eqd.retained[r]);
    if (!fresh && it->second != eqd.retained[r]) {
      invalid("node " + std::to_string(eqd.constrained[r]) + " is constrained by equalDOF to node " +
              std::to_string(it->second) + " and to node " + std::to_string(eqd.retained[r]) +
              ". OpenSees' Transformation handler would apply only the first of these and "
              "silently ignore the other; tie the node to one retained node");
    }
  }
  // --- Groups and equations ------------------------------------------------------------
  UnionFind uf(ndof);
  for (std::size_t r = 0; r < eqd.retained.size(); ++r)
    uf.join(dof_id(eqd.retained[r], eqd.dof[r]), dof_id(eqd.constrained[r], eqd.dof[r]));

  std::vector<char> fixed_root(static_cast<std::size_t>(ndof), 0);
  for (std::size_t r = 0; r < model.fix.node.size(); ++r)
    fixed_root[uf.find(dof_id(model.fix.node[r], model.fix.dof[r]))] = 1;

  // DOF ids increase with the internal node order, and each root is its group's smallest
  // id, so numbering roots in id order numbers groups by their smallest DOF.
  out.eq.assign(static_cast<std::size_t>(ndof), -1);
  std::vector<std::int32_t> eq_of_root(static_cast<std::size_t>(ndof), -1);
  std::int32_t next = 0;
  for (std::int32_t a = 0; a < ndof; ++a) {
    const std::int32_t root = uf.find(a);
    if (fixed_root[root]) continue;
    if (eq_of_root[root] < 0) eq_of_root[root] = next++;
    out.eq[a] = eq_of_root[root];
  }
  out.num_equations = next;

  if (check_opensees) check_against_transformation_handler(model, out);
  return out;
}

}  // namespace femorax::compile
