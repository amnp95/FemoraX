#include "compile/batches.hpp"

#include <algorithm>
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
  c.eq_dof.assign(static_cast<std::size_t>(n.num_equations), -1);
  for (std::int32_t i = 0; i < N; ++i)
    for (std::int32_t d = 0; d < n.ndf[i]; ++d) {
      const std::int32_t q = n.equation(i, d);
      if (q >= 0) c.eq_dof[q] = static_cast<std::int8_t>(d);
    }

  std::unordered_map<std::int32_t, std::size_t> material_index;
  for (std::size_t i = 0; i < m.materials.size(); ++i) material_index[m.materials[i].tag] = i;

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
    const auto& mat = m.materials[material_index.at(g.material)];
    const std::int32_t ur = uniform_region(g.tags[where[ge].second]);
    const int pair =
        registry::pair_index(g.type, mat, ur < 0 ? 0 : uniform_filters[static_cast<std::size_t>(ur)].nFilter);
    if (pair < 0) {
      throw io::UnsupportedComponent(std::string("unsupported component: element type \"") +
                                     io::to_string(g.type) + "\" with material " +
                                     std::to_string(mat.tag) +
                                     " has no kernel in this build. Run this model with the "
                                     "OpenSees backend.");
    }
    members[pair].push_back(ge);
  }

  c.lumped_mass.assign(static_cast<std::size_t>(n.num_equations), 0.0);
  c.lumped_alpha_mass.assign(static_cast<std::size_t>(n.num_equations), 0.0);
  c.nodal_mass.assign(static_cast<std::size_t>(n.num_equations), 0.0);
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
          ps.material = static_cast<std::int32_t>(material_index.at(g.material));
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
        double mj[Elem::NNODE];
        Elem::lumped_mass(ps.template params<Elem>(), xe, mj);
        k = 0;
        for (int a = 0; a < b.nnode; ++a)
          for (int d = 0; d < Elem::NODE_NDF[a]; ++d, ++k) {
            const std::int32_t q = b.eq[static_cast<std::size_t>(k) * P + e];
            if (q < 0) continue;
            c.lumped_mass[q] += mj[a];
            c.lumped_alpha_mass[q] += ps.damping.alphaM * mj[a];
          }
      }
      if (opt.build_coloring) color(b, n.num_equations);
      c.batches.push_back(std::move(b));
    });
  }

  // Nodal masses (no Rayleigh damping: node regions are not supported).
  for (std::size_t r = 0; r < m.mass.node.size(); ++r) {
    const std::int32_t q = n.equation(internal_of_tag.at(m.mass.node[r]), m.mass.dof[r] - 1);
    if (q >= 0) {
      c.lumped_mass[q] += m.mass.value[r];
      c.nodal_mass[q] += m.mass.value[r];
    }
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
