#include "compile/batches.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <tuple>
#include <unordered_map>

#include "io/h5.hpp"

namespace femorax::compile {

namespace {

std::int32_t round_up(std::int32_t n, int w) {
  w = std::max(w, 1);
  return (n + w - 1) / w * w;
}

// Greedy coloring in batch order: an element takes the smallest color not used by any
// earlier element that shares one of its equations. Deterministic.
void color(Batch& b, std::int32_t num_equations) {
  std::vector<std::int32_t> element_color(static_cast<std::size_t>(b.count), -1);
  std::vector<std::vector<std::int32_t>> used(static_cast<std::size_t>(num_equations));
  std::int32_t ncolors = 0;
  std::vector<char> taken;
  for (std::int32_t e = 0; e < b.count; ++e) {
    taken.assign(static_cast<std::size_t>(ncolors) + 1, 0);
    for (int k = 0; k < b.ndof; ++k) {
      const std::int32_t q = b.eq[static_cast<std::size_t>(k) * b.padded + e];
      if (q < 0) continue;
      for (const auto c : used[q]) taken[c] = 1;
    }
    std::int32_t c = 0;
    while (taken[c]) ++c;
    element_color[e] = c;
    ncolors = std::max(ncolors, c + 1);
    for (int k = 0; k < b.ndof; ++k) {
      const std::int32_t q = b.eq[static_cast<std::size_t>(k) * b.padded + e];
      if (q >= 0 && std::find(used[q].begin(), used[q].end(), c) == used[q].end()) used[q].push_back(c);
    }
  }
  b.color_start.assign(static_cast<std::size_t>(ncolors) + 1, 0);
  for (const auto c : element_color) ++b.color_start[c + 1];
  for (std::int32_t c = 0; c < ncolors; ++c) b.color_start[c + 1] += b.color_start[c];
  b.color_elements.resize(static_cast<std::size_t>(b.count));
  std::vector<std::int32_t> fill(b.color_start.begin(), b.color_start.end() - 1);
  for (std::int32_t e = 0; e < b.count; ++e) b.color_elements[fill[element_color[e]]++] = e;
}

}  // namespace

Compiled compile_model(const io::Model& m, const CompileOptions& opt) {
  Compiled c;
  c.ordering = compute_ordering(m, opt.renumbering);
  c.dofs = number_dofs(m, c.ordering.node_order);
  const auto& n = c.dofs;
  const std::int32_t N = n.num_nodes();

  c.subdomain.owned_nodes.resize(static_cast<std::size_t>(N));
  for (std::int32_t i = 0; i < N; ++i) c.subdomain.owned_nodes[i] = i;

  std::unordered_map<std::int32_t, std::int32_t> internal_of_tag;
  for (std::int32_t i = 0; i < N; ++i) internal_of_tag[n.node_tag[i]] = i;
  c.coords.resize(static_cast<std::size_t>(3) * N);
  for (std::int32_t i = 0; i < N; ++i)
    for (int k = 0; k < 3; ++k)
      c.coords[static_cast<std::size_t>(k) * N + i] =
          m.nodes.coords[static_cast<std::size_t>(n.node_model_index[i]) * 3 + k];
  c.eq_dof.assign(static_cast<std::size_t>(n.num_slots), -1);
  for (std::int32_t i = 0; i < N; ++i)
    for (std::int32_t d = 0; d < n.ndf[i]; ++d) {
      const std::int32_t q = n.equation(i, d);
      if (q >= 0) c.eq_dof[q] = static_cast<std::int8_t>(d);
    }

  // Materials by (kind, tag): sections have a tag space of their own (io::is_section).
  std::unordered_map<std::int64_t, std::size_t> material_by_key;
  auto key = [](bool section, std::int32_t tag) { return (static_cast<std::int64_t>(section) << 32) | static_cast<std::uint32_t>(tag); };
  for (std::size_t i = 0; i < m.materials.size(); ++i)
    material_by_key[key(io::is_section(m.materials[i]), m.materials[i].tag)] = i;
  auto material_of = [&](const io::ElementGroup& g) {
    return material_by_key.at(key(io::is_beam(g.type), g.material));
  };

  // Damping region of every element tag: the last region in file order wins (OpenSees).
  std::unordered_map<std::int32_t, std::int32_t> region_of;
  for (std::size_t r = 0; r < m.rayleigh.size(); ++r)
    for (const auto e : m.rayleigh[r].elements) region_of[e] = static_cast<std::int32_t>(r);

  // Uniform damping region of every element tag (the last region wins, as for Rayleigh), and
  // each region's filters, fitted once.
  std::unordered_map<std::int32_t, std::int32_t> uniform_of;
  for (std::size_t r = 0; r < m.uniform.size(); ++r)
    for (const auto e : m.uniform[r].elements) uniform_of[e] = static_cast<std::int32_t>(r);
  std::vector<UniformDampingFilters> uniform_filters;
  for (const auto& r : m.uniform) uniform_filters.push_back(registry::uniform_damping_filters(r));
  auto uniform_region = [&](std::int32_t tag) {
    const auto it = uniform_of.find(tag);
    return it == uniform_of.end() ? -1 : it->second;
  };

  // Global element index (file order across groups) -> (group, index in group).
  std::vector<std::pair<std::int32_t, std::int32_t>> where;
  for (std::size_t g = 0; g < m.elements.size(); ++g)
    for (std::size_t e = 0; e < m.elements[g].size(); ++e)
      where.emplace_back(static_cast<std::int32_t>(g), static_cast<std::int32_t>(e));

  // One batch per registered pair; elements in renumbered order.
  std::map<int, std::vector<std::int32_t>> members;
  for (const auto ge : c.ordering.element_order) {
    const auto& g = m.elements[where[ge].first];
    const auto& mat = m.materials[material_of(g)];
    const std::int32_t ur = uniform_region(g.tags[where[ge].second]);
    const int pair =
        registry::pair_index(g, mat, ur < 0 ? 0 : uniform_filters[static_cast<std::size_t>(ur)].nFilter);
    if (pair < 0) {
      throw io::UnsupportedComponent(std::string("unsupported component: element type \"") +
                                     io::to_string(g.type) + "\" with material " +
                                     std::to_string(mat.tag) +
                                     " has no kernel in this build. Run this model with the "
                                     "OpenSees backend.");
    }
    members[pair].push_back(ge);
  }

  c.lumped_mass.assign(static_cast<std::size_t>(n.num_slots), 0.0);  // slots contracted below
  c.lumped_alpha_mass.assign(static_cast<std::size_t>(n.num_slots), 0.0);  // slots contracted below
  c.nodal_mass.assign(static_cast<std::size_t>(n.num_slots), 0.0);  // slots contracted below
  c.nodal_alpha_mass.assign(static_cast<std::size_t>(n.num_slots), 0.0);  // slots contracted below
  for (const auto& [pair, list] : members) {
    registry::for_each_pair([&]<class E, class M>(int index) {
      if (index != pair) return;
      using Elem = typename E::template Element<typename M::Material>;
      Batch b;
      b.pair = pair;
      b.element = E::type;
      b.nnode = Elem::NNODE;
      b.ndof = Elem::NDOF;
      b.nstate = Elem::NSTATE;
      if constexpr (Elem::GEOMETRY_CACHEABLE) b.geometry_doubles = Elem::NGEOM * Elem::NGP + element_constant_count<Elem>();
      {
        using Mat = typename Elem::Material;
        constexpr int NS = Mat::NSTRAIN;
        if (!Mat::CONSTANT_TANGENT) b.tangent_doubles = Elem::NGP * (is_symmetric(Mat::TANGENT) ? NS * (NS + 1) / 2 : NS * NS);
        b.tangent = Mat::TANGENT;
      }
      if constexpr (SplitMaterial<typename Elem::Material> && RebindableElement<Elem>) {
        constexpr int NS = Elem::Material::NSTRAIN;
        b.compact_bytes = Elem::NGP * ((2 * NS + NS * NS) * 8 + 2 * 4);
      }
      b.count = static_cast<std::int32_t>(list.size());
      b.padded = round_up(b.count, opt.simd_width);
      const auto P = static_cast<std::size_t>(b.padded);
      b.tag.assign(P, -1);
      b.node.assign(static_cast<std::size_t>(b.nnode) * P, -1);
      b.eq.assign(static_cast<std::size_t>(b.ndof) * P, -1);
      b.param.assign(P, 0);

      std::map<std::tuple<std::int32_t, std::int32_t, std::int32_t>, std::int32_t> param_of;  // (group, region, uniform)
      for (std::int32_t e = 0; e < b.count; ++e) {
        const auto [gi, ei] = where[list[e]];
        const auto& g = m.elements[gi];
        b.tag[e] = g.tags[ei];
        const auto rit = region_of.find(g.tags[ei]);
        const std::int32_t region = rit == region_of.end() ? -1 : rit->second;
        const std::int32_t uregion = uniform_region(g.tags[ei]);
        const auto [pit, fresh] =
            param_of.emplace(std::make_tuple(gi, region, uregion), static_cast<std::int32_t>(b.params.size()));
        if (fresh) {
          ParamSet ps;
          ps.group = gi;
          ps.material = static_cast<std::int32_t>(material_of(g));
          ps.region = region;
          if (region >= 0) {
            const auto& r = m.rayleigh[region];
            ps.damping = Rayleigh{r.alphaM, r.betaK, r.betaKinit, r.betaKcomm};
          }
          if constexpr (registry::is_damped_entry<M>) {
            const auto ur = static_cast<std::size_t>(uregion);
            ps.element = registry::ParamBlob::of(E::template params<typename M::Material>(
                g, M::params(m.materials[ps.material], m.uniform[ur], uniform_filters[ur])));
          } else {
            ps.element = registry::ParamBlob::of(
                E::template params<typename M::Material>(g, M::params(m.materials[ps.material])));
          }
          b.params.push_back(ps);
        }
        b.param[e] = pit->second;

        // Node indices, equations, coordinates (uniform NODE_NDF for the registered
        // elements so far; the DOF offsets handle any layout).
        double xe[Elem::NNODE][Elem::NDM];
        int k = 0;
        for (int a = 0; a < b.nnode; ++a) {
          const std::int32_t node =
              internal_of_tag.at(g.connectivity[static_cast<std::size_t>(ei) * b.nnode + a]);
          b.node[static_cast<std::size_t>(a) * P + e] = node;
          for (int d = 0; d < Elem::NODE_NDF[a]; ++d, ++k)
            b.eq[static_cast<std::size_t>(k) * P + e] = n.equation(node, d);
          for (int d = 0; d < Elem::NDM; ++d) xe[a][d] = c.coords[static_cast<std::size_t>(d) * N + node];
        }
        // Lumped mass of this element into its equations (fixed DOFs drop out, tied DOFs
        // add to their shared equation), and alphaM-weighted for CentralDifference.
        const ParamSet& ps = b.params[b.param[e]];
        // Per DOF for elements whose DOFs at a node have different masses (beams: rotations),
        // per node for the solids.
        double mk[Elem::NDOF];
        if constexpr (requires(double* x) { Elem::lumped_mass_dof(ps.template params<Elem>(), xe, x); }) {
          Elem::lumped_mass_dof(ps.template params<Elem>(), xe, mk);
        } else {
          double mj[Elem::NNODE];
          Elem::lumped_mass(ps.template params<Elem>(), xe, mj);
          k = 0;
          for (int a = 0; a < b.nnode; ++a)
            for (int d = 0; d < Elem::NODE_NDF[a]; ++d, ++k) mk[k] = mj[a];
        }
        for (k = 0; k < Elem::NDOF; ++k) {
          const std::int32_t q = b.eq[static_cast<std::size_t>(k) * P + e];
          if (q < 0) continue;
          c.lumped_mass[q] += mk[k];
          c.lumped_alpha_mass[q] += ps.damping.alphaM * mk[k];
        }
      }
      if (opt.build_coloring) color(b, n.num_slots);
      c.batches.push_back(std::move(b));
    });
  }

  // Nodal masses, damped by the alphaM of their node's Rayleigh region: OpenSees' `region
  // -ele` puts the elements' nodes in the region too (MeshRegion::setElements), and a node
  // applies the region's alphaM to its own mass (Node::getUnbalancedLoadIncInertia, alphaM M v;
  // Node::getDamp in the tangent). A node takes the last region in file order that holds one
  // of its elements, as elements do (docs/design.md, "Structures: mass and damping standard").
  std::unordered_map<std::int32_t, std::int32_t> node_region;
  if (!m.mass.node.empty() && !m.rayleigh.empty()) {
    std::unordered_map<std::int32_t, std::pair<std::size_t, std::size_t>> element_at;  // tag -> (group, index)
    for (std::size_t g = 0; g < m.elements.size(); ++g)
      for (std::size_t e = 0; e < m.elements[g].size(); ++e) element_at[m.elements[g].tags[e]] = {g, e};
    for (std::size_t r = 0; r < m.rayleigh.size(); ++r)
      for (const auto tag : m.rayleigh[r].elements) {
        const auto [g, e] = element_at.at(tag);
        const auto& grp = m.elements[g];
        const std::size_t nn = grp.connectivity.size() / grp.size();
        for (std::size_t a = 0; a < nn; ++a) node_region[grp.connectivity[e * nn + a]] = static_cast<std::int32_t>(r);
      }
  }
  for (std::size_t r = 0; r < m.mass.node.size(); ++r) {
    const std::int32_t q = n.equation(internal_of_tag.at(m.mass.node[r]), m.mass.dof[r] - 1);
    if (q >= 0) {
      c.lumped_mass[q] += m.mass.value[r];
      c.nodal_mass[q] += m.mass.value[r];
      const auto it = node_region.find(m.mass.node[r]);
      if (it != node_region.end()) {
        const double am = m.rayleigh[static_cast<std::size_t>(it->second)].alphaM * m.mass.value[r];
        c.nodal_alpha_mass[q] += am;
        c.lumped_alpha_mass[q] += am;  // CentralDifference's damping diagonal
      }
    }
  }
  // The nodal masses' shares: each equation's nodal mass over the element DOFs touching it.
  c.nodal_mass_share.assign(c.nodal_mass.size(), 0.0);
  {
    std::vector<std::int32_t> touching(c.nodal_mass.size(), 0);
    for (const auto& b : c.batches)
      for (std::int32_t e = 0; e < b.count; ++e)
        for (int k = 0; k < b.ndof; ++k) {
          const std::int32_t q = b.eq[static_cast<std::size_t>(k) * static_cast<std::size_t>(b.padded) + static_cast<std::size_t>(e)];
          if (q >= 0) ++touching[static_cast<std::size_t>(q)];
        }
    for (std::size_t q = 0; q < touching.size(); ++q)
      if (touching[q] > 0) c.nodal_mass_share[q] = c.nodal_mass[q] / touching[q];
  }
  // The constrained DOFs' masses onto their equations, for the explicit stages: row-sum
  // lumping of T^T M T for the embedded translations (weights summing to one: exact total mass
  // and centre of mass), the exact T^T M T blocks for rigid links and diaphragms (their
  // translation rows carry the master's rotations with lever arms). The nodal masses keep their
  // slots (the operator applies them exactly); so do the shares above (an element on a
  // constrained DOF sees only that DOF's own nodal mass in its stable step: conservative).
  if (n.num_constrained() > 0) {
    const std::int32_t neq = n.num_equations;
    auto row = [&](std::int32_t k) {
      return std::pair{n.con_start[static_cast<std::size_t>(k)], n.con_start[static_cast<std::size_t>(k) + 1]};
    };
    // Equations coupled by exact rows: union-find.
    std::vector<std::int32_t> parent(static_cast<std::size_t>(neq));
    for (std::int32_t q = 0; q < neq; ++q) parent[static_cast<std::size_t>(q)] = q;
    std::function<std::int32_t(std::int32_t)> find = [&](std::int32_t q) {
      while (parent[static_cast<std::size_t>(q)] != q) q = parent[static_cast<std::size_t>(q)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(q)])];
      return q;
    };
    std::vector<std::int32_t> coupled;  // equations in an exact row with two or more terms
    for (std::int32_t k = 0; k < n.num_constrained(); ++k) {
      if (!n.con_exact[static_cast<std::size_t>(k)]) continue;
      const auto [a, b] = row(k);
      if (b - a > 1)
        for (auto t = a; t < b; ++t) coupled.push_back(n.con_eq[static_cast<std::size_t>(t)]);
      for (auto t = a + 1; t < b; ++t) {
        const auto r1 = find(n.con_eq[static_cast<std::size_t>(a)]), r2 = find(n.con_eq[static_cast<std::size_t>(t)]);
        if (r1 != r2) parent[static_cast<std::size_t>(std::max(r1, r2))] = std::min(r1, r2);
      }
    }
    std::map<std::int32_t, std::vector<std::int32_t>> members_of;  // root -> equations, ascending
    std::sort(coupled.begin(), coupled.end());
    coupled.erase(std::unique(coupled.begin(), coupled.end()), coupled.end());
    for (const auto q : coupled) members_of[find(q)].push_back(q);
    constexpr std::size_t kMaxBlock = 12;
    auto& mb = c.mass_blocks;
    std::vector<std::int32_t> block_of(static_cast<std::size_t>(neq), -1), pos(static_cast<std::size_t>(neq), -1);
    for (const auto& [root, list] : members_of) {
      if (list.size() > kMaxBlock)
        throw io::UnsupportedComponent("unsupported component: rigid links or diaphragms couple " +
                                       std::to_string(list.size()) + " equations into one mass block (at most " +
                                       std::to_string(kMaxBlock) + " so far)");
      const auto b = static_cast<std::int32_t>(mb.count());
      for (std::size_t i = 0; i < list.size(); ++i) {
        block_of[static_cast<std::size_t>(list[i])] = b;
        pos[static_cast<std::size_t>(list[i])] = static_cast<std::int32_t>(i);
        mb.eq.push_back(list[i]);
      }
      mb.start.push_back(static_cast<std::int32_t>(mb.eq.size()));
      const auto nb = static_cast<std::int64_t>(list.size());
      mb.offset.push_back(mb.offset.back() + nb * nb);
    }
    mb.M.assign(static_cast<std::size_t>(mb.offset.back()), 0.0);
    mb.C.assign(mb.M.size(), 0.0);
    // The equations' own masses on the blocks' diagonals.
    for (std::size_t b = 0; b < mb.count(); ++b) {
      const auto nb = mb.start[b + 1] - mb.start[b];
      for (std::int32_t i = 0; i < nb; ++i) {
        const auto q = static_cast<std::size_t>(mb.eq[static_cast<std::size_t>(mb.start[b] + i)]);
        mb.M[static_cast<std::size_t>(mb.offset[b] + i * nb + i)] = c.lumped_mass[q];
        mb.C[static_cast<std::size_t>(mb.offset[b] + i * nb + i)] = c.lumped_alpha_mass[q];
      }
    }
    // The ground-motion mass (T^T M R)_q of each equation: its own mass plus w m of the slots
    // in its translational direction (a rotation's inertia takes no ground load).
    c.ground_mass.assign(c.lumped_mass.begin(), c.lumped_mass.begin() + neq);
    for (std::int32_t k = 0; k < n.num_constrained(); ++k) {
      const auto s_ = static_cast<std::size_t>(neq + k);
      const double ms = c.lumped_mass[s_], cs = c.lumped_alpha_mass[s_];
      const auto [a, b] = row(k);
      const int dir = c.eq_dof[s_];
      for (auto t = a; t < b; ++t) {
        const auto q = static_cast<std::size_t>(n.con_eq[static_cast<std::size_t>(t)]);
        if (dir < 3 && c.eq_dof[q] == dir) c.ground_mass[q] += n.con_weight[static_cast<std::size_t>(t)] * ms;
      }
      if (!n.con_exact[static_cast<std::size_t>(k)]) {
        // Dominant lumping: m |w_i| sum_j |w_j| on each equation, never below the exact
        // m w w^T (Cauchy-Schwarz), so the explicit stable step stays an upper bound. For the
        // embedded translations (weights >= 0 summing to one) it is the row-sum lumping (exact
        // total mass and centre of mass); for the rotation tie (weights summing to zero) it
        // keeps the rotary inertia that a row sum would drop.
        double S = 0.0;
        for (auto t = a; t < b; ++t) S += std::abs(n.con_weight[static_cast<std::size_t>(t)]);
        for (auto t = a; t < b; ++t) {
          const auto q = static_cast<std::size_t>(n.con_eq[static_cast<std::size_t>(t)]);
          const double w = std::abs(n.con_weight[static_cast<std::size_t>(t)]) * S;
          c.lumped_mass[q] += w * ms;
          c.lumped_alpha_mass[q] += w * cs;
          if (block_of[q] >= 0) {
            const auto bb = static_cast<std::size_t>(block_of[q]);
            const auto nb = mb.start[bb + 1] - mb.start[bb];
            const auto ii = static_cast<std::size_t>(mb.offset[bb] + pos[q] * nb + pos[q]);
            mb.M[ii] += w * ms;
            mb.C[ii] += w * cs;
          }
        }
        continue;
      }
      for (auto t1 = a; t1 < b; ++t1) {
        const auto q1 = static_cast<std::size_t>(n.con_eq[static_cast<std::size_t>(t1)]);
        const double w1 = n.con_weight[static_cast<std::size_t>(t1)];
        c.lumped_mass[q1] += w1 * w1 * ms;  // the diagonal of T^T M T
        c.lumped_alpha_mass[q1] += w1 * w1 * cs;
        if (block_of[q1] < 0) continue;  // a single equation: the diagonal is exact
        const auto bb = static_cast<std::size_t>(block_of[q1]);
        const auto nb = mb.start[bb + 1] - mb.start[bb];
        for (auto t2 = a; t2 < b; ++t2) {
          const auto q2 = static_cast<std::size_t>(n.con_eq[static_cast<std::size_t>(t2)]);
          const double w2 = n.con_weight[static_cast<std::size_t>(t2)];
          const auto ii = static_cast<std::size_t>(mb.offset[bb] + pos[q1] * nb + pos[q2]);
          mb.M[ii] += w1 * w2 * ms;
          mb.C[ii] += w1 * w2 * cs;
        }
      }
    }
    c.lumped_mass.resize(static_cast<std::size_t>(neq));
    c.lumped_alpha_mass.resize(static_cast<std::size_t>(neq));
  }
  return c;
}

std::int64_t memory_estimate_bytes(const Compiled& c) {
  const std::int64_t neq = c.dofs.num_equations, N = c.dofs.num_nodes();
  std::int64_t bytes = 0;
  bytes += 3 * N * 8;    // coordinates
  bytes += 9 * neq * 8;  // u, v (two), a, f_int, mass, alpha mass, scratch
  for (const auto& b : c.batches) {
    const std::int64_t P = b.padded;
    bytes += P * 4 * (1 + b.nnode + b.ndof + 1);  // tag, nodes, equations, param index
    bytes += 2 * P * b.nstate * 8;                // material state, committed and trial
    bytes += P * b.compact_bytes;                 // compacted corrector buffers
    for (const auto& ps : b.params) bytes += static_cast<std::int64_t>(ps.element.bytes.size()) + 64;
    bytes += static_cast<std::int64_t>(b.color_elements.size() + b.color_start.size()) * 4;
  }
  return bytes;
}

}  // namespace femorax::compile
