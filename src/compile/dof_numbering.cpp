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

std::string dof_list(const std::vector<std::int32_t>& d0) {
  std::string s;
  for (std::size_t k = 0; k < d0.size(); ++k) s += (k ? " " : "") + std::to_string(d0[k] + 1);
  return s;
}

// The message when OpenSees' handler and the constraints as written disagree: which of the
// refused patterns the model contains, what OpenSees does with it, and how to fix the model.
// Harmless forms of the same patterns (a fix on a tied DOF whose retained DOF is fixed too, a
// full chain in ascending tag order) are not blamed.
std::string refusal(const io::Model& model, const DofNumbering& n, std::int32_t fail_node,
                    std::int32_t fail_dof, std::int64_t opensees_value) {
  const auto& eqd = model.equal_dof;
  std::unordered_map<std::int32_t, std::int32_t> retained_of;           // tag -> tag
  std::unordered_map<std::int32_t, std::vector<std::int32_t>> dofs_of;  // constrained tag -> DOFs
  for (std::size_t r = 0; r < eqd.retained.size(); ++r) {
    retained_of[eqd.constrained[r]] = eqd.retained[r];
    dofs_of[eqd.constrained[r]].push_back(eqd.dof[r]);
  }
  std::unordered_map<std::int64_t, char> fixed;
  for (std::size_t r = 0; r < model.fix.node.size(); ++r)
    fixed[static_cast<std::int64_t>(model.fix.node[r]) * 64 + model.fix.dof[r]] = 1;
  auto is_fixed = [&](std::int32_t tag, std::int32_t d1) {
    return fixed.count(static_cast<std::int64_t>(tag) * 64 + d1) > 0;
  };
  auto list = [](const std::vector<std::int32_t>& d1) {
    std::string s;
    for (std::size_t k = 0; k < d1.size(); ++k) s += (k ? " " : "") + std::to_string(d1[k]);
    return s;
  };
  const std::string tail = " FemoraX refuses the model rather than give a different answer "
                           "(docs/deviations.md, \"Constraint handling\").";

  // A fix on a tied DOF whose retained DOF is free: OpenSees loses the fix.
  for (std::size_t r = 0; r < model.fix.node.size(); ++r) {
    const std::int32_t C = model.fix.node[r], d1 = model.fix.dof[r];
    const auto it = dofs_of.find(C);
    if (it == dofs_of.end() || std::find(it->second.begin(), it->second.end(), d1) == it->second.end())
      continue;
    const std::int32_t R = retained_of.at(C);
    if (is_fixed(R, d1)) continue;
    return "node " + std::to_string(C) + " DOF " + std::to_string(d1) +
           " is fixed and also tied by equalDOF to node " + std::to_string(R) + " DOF " +
           std::to_string(d1) + ", which is free. OpenSees' addSP_Constraint puts the fix at a "
           "different position of the node's transformed DOFs, so in OpenSees the fix is "
           "silently lost and node " + std::to_string(C) + " moves with node " +
           std::to_string(R) + ". Fix the model: fix node " + std::to_string(R) + " DOF " +
           std::to_string(d1) + " instead (the tie then holds node " + std::to_string(C) +
           " fixed too), or remove that DOF from the tie." + tail;
  }
  // Chains: C tied to R, R tied to M.
  for (const auto& [C, R] : retained_of) {
    const auto rm = retained_of.find(R);
    if (rm == retained_of.end()) continue;
    const std::int32_t M = rm->second;
    const auto& dCR = dofs_of.at(C);
    const auto& dRM = dofs_of.at(R);
    const std::string chain = "node " + std::to_string(C) + " is tied to node " + std::to_string(R) +
                              " (DOFs " + list(dCR) + "), which is itself tied to node " +
                              std::to_string(M) + " (DOFs " + list(dRM) + "). ";
    const std::string fix = " Fix the model: tie node " + std::to_string(C) + " directly to node " +
                            std::to_string(M) + ", the final retained node.";
    if (dCR != dRM) {
      return chain + "This is a partial chain: OpenSees resolves it through node " +
             std::to_string(R) + "'s internal DOF layout, which no longer lines up with its "
             "nodal DOFs, so in OpenSees the DOFs of node " + std::to_string(C) +
             " are silently connected to the wrong DOFs (or fixed)." + fix + tail;
    }
    if (R > C) {
      return chain + "Node " + std::to_string(R) + " has a larger tag than node " +
             std::to_string(C) + ": OpenSees processes nodes in ascending tag order and reads "
             "node " + std::to_string(R) + "'s equation numbers before they are set, so in "
             "OpenSees node " + std::to_string(C) + " is silently fixed." + fix + tail;
    }
  }
  const std::string what_os =
      opensees_value == kFixed ? std::string("fixed") : "equal to " + describe(opensees_value, n);
  return "node " + std::to_string(n.node_tag[fail_node]) + " DOF " + std::to_string(fail_dof + 1) +
         ": OpenSees' Transformation handler would make this DOF " + what_os +
         ", which is not what the fix and equalDOF constraints say. Tie every node directly to "
         "its final retained node, and fix retained nodes rather than tied DOFs." + tail;
}

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
      if (!same) throw io::ModelError(refusal(model, n, i, d, v));
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
  // One equalDOF command (one OpenSees MP constraint) per constrained node: the handler
  // applies only the first command that names a node. A command is a `command` value
  // (schema 0.2) or, without it, a (retained, constrained) pair.
  {
    struct Cmd {
      std::int64_t id;
      std::int32_t retained;
      std::vector<std::int32_t> dofs;
    };
    std::unordered_map<std::int32_t, std::vector<Cmd>> cmds;  // constrained tag -> commands
    for (std::size_t r = 0; r < eqd.retained.size(); ++r) {
      const std::int64_t id = eqd.command.empty() ? -1 - static_cast<std::int64_t>(eqd.retained[r])
                                                  : static_cast<std::int64_t>(eqd.command[r]);
      auto& list = cmds[eqd.constrained[r]];
      auto it = std::find_if(list.begin(), list.end(), [&](const Cmd& c) { return c.id == id; });
      if (it == list.end()) {
        list.push_back({id, eqd.retained[r], {}});
        it = list.end() - 1;
      }
      it->dofs.push_back(eqd.dof[r] - 1);
    }
    for (const auto& [c, list] : cmds) {
      if (list.size() < 2) continue;
      std::string ties;
      for (std::size_t k = 0; k < list.size(); ++k)
        ties += (k ? (k + 1 == list.size() ? " and " : ", ") : "") + std::string("node ") +
                std::to_string(list[k].retained) + " (DOFs " + dof_list(list[k].dofs) + ")";
      invalid("node " + std::to_string(c) + " is constrained by " + std::to_string(list.size()) +
              " equalDOF commands: to " + ties + ". OpenSees' Transformation handler applies "
              "only the first MP constraint on a node and silently ignores the others, so the "
              "later ties would have no effect in OpenSees. Fix the model: tie node " +
              std::to_string(c) + " with one equalDOF command, to one retained node, listing all "
              "its DOFs (equalDOF " + std::to_string(list[0].retained) + " " + std::to_string(c) +
              " 1 2 3). FemoraX refuses the model rather than give a different answer "
              "(docs/deviations.md, \"Constraint handling\").");
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
