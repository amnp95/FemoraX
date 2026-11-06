#include "compile/renumber.hpp"

#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_map>

#include "io/h5.hpp"

namespace femorax::compile {

namespace {

using Index = std::int32_t;

// Model node index of every node tag.
std::unordered_map<std::int32_t, Index> index_of_tag(const io::Model& m) {
  std::unordered_map<std::int32_t, Index> idx;
  idx.reserve(m.nodes.size());
  for (std::size_t i = 0; i < m.nodes.size(); ++i) idx[m.nodes.tags[i]] = static_cast<Index>(i);
  return idx;
}

// Node adjacency (CSR, sorted, no self loops) from element connectivity and equalDOF ties.
struct Graph {
  std::vector<Index> start, adj;
  Index degree(Index v) const { return start[v + 1] - start[v]; }
};

Graph node_graph(const io::Model& m) {
  const auto N = static_cast<Index>(m.nodes.size());
  const auto idx = index_of_tag(m);
  std::vector<std::vector<Index>> nbr(static_cast<std::size_t>(N));
  for (const auto& g : m.elements) {
    const int nn = io::node_count(g.type);
    for (std::size_t e = 0; e < g.size(); ++e)
      for (int a = 0; a < nn; ++a)
        for (int b = 0; b < nn; ++b)
          if (a != b) nbr[idx.at(g.connectivity[e * nn + a])].push_back(idx.at(g.connectivity[e * nn + b]));
  }
  for (std::size_t r = 0; r < m.equal_dof.retained.size(); ++r) {
    const Index a = idx.at(m.equal_dof.retained[r]), b = idx.at(m.equal_dof.constrained[r]);
    nbr[a].push_back(b);
    nbr[b].push_back(a);
  }
  Graph G;
  G.start.assign(static_cast<std::size_t>(N) + 1, 0);
  for (Index v = 0; v < N; ++v) {
    auto& l = nbr[v];
    std::sort(l.begin(), l.end());
    l.erase(std::unique(l.begin(), l.end()), l.end());
    G.start[v + 1] = G.start[v] + static_cast<Index>(l.size());
  }
  G.adj.reserve(static_cast<std::size_t>(G.start[N]));
  for (auto& l : nbr) G.adj.insert(G.adj.end(), l.begin(), l.end());
  return G;
}

// BFS levels from root within the unvisited part; returns the last level.
std::vector<Index> bfs_last_level(const Graph& G, Index root, const std::vector<char>& done,
                                  int& depth) {
  std::vector<Index> level = {root}, last = level;
  std::vector<char> seen(done);
  seen[root] = 1;
  depth = 0;
  while (!level.empty()) {
    last = level;
    std::vector<Index> next;
    for (Index v : level)
      for (Index k = G.start[v]; k < G.start[v + 1]; ++k)
        if (!seen[G.adj[k]]) {
          seen[G.adj[k]] = 1;
          next.push_back(G.adj[k]);
        }
    if (!next.empty()) ++depth;
    level = std::move(next);
  }
  return last;
}

// George-Liu pseudo-peripheral node: start at the smallest-degree node of the component,
// move to the smallest-degree node of the last BFS level while the depth grows.
Index pseudo_peripheral(const Graph& G, Index start, const std::vector<char>& done) {
  Index root = start;
  int depth = -1;
  for (int it = 0; it < 32; ++it) {
    int d;
    auto last = bfs_last_level(G, root, done, d);
    if (d <= depth) break;
    depth = d;
    root = *std::min_element(last.begin(), last.end(), [&](Index a, Index b) {
      return G.degree(a) != G.degree(b) ? G.degree(a) < G.degree(b) : a < b;
    });
  }
  return root;
}

std::vector<Index> rcm(const Graph& G) {
  const auto N = static_cast<Index>(G.start.size() - 1);
  std::vector<char> done(static_cast<std::size_t>(N), 0);
  std::vector<Index> order;
  order.reserve(static_cast<std::size_t>(N));
  for (Index s = 0; s < N; ++s) {  // components in order of their first node in the file
    if (done[s]) continue;
    // Smallest-degree node of this component as the search start.
    Index best = s;
    {
      std::vector<Index> stack = {s};
      std::vector<char> seen(done);
      seen[s] = 1;
      while (!stack.empty()) {
        const Index v = stack.back();
        stack.pop_back();
        if (G.degree(v) < G.degree(best) || (G.degree(v) == G.degree(best) && v < best)) best = v;
        for (Index k = G.start[v]; k < G.start[v + 1]; ++k)
          if (!seen[G.adj[k]]) {
            seen[G.adj[k]] = 1;
            stack.push_back(G.adj[k]);
          }
      }
    }
    const Index root = pseudo_peripheral(G, best, done);
    // Cuthill-McKee: BFS, neighbours by increasing degree, ties by index.
    const std::size_t first = order.size();
    order.push_back(root);
    done[root] = 1;
    for (std::size_t head = first; head < order.size(); ++head) {
      const Index v = order[head];
      std::vector<Index> next;
      for (Index k = G.start[v]; k < G.start[v + 1]; ++k)
        if (!done[G.adj[k]]) {
          done[G.adj[k]] = 1;
          next.push_back(G.adj[k]);
        }
      std::sort(next.begin(), next.end(), [&](Index a, Index b) {
        return G.degree(a) != G.degree(b) ? G.degree(a) < G.degree(b) : a < b;
      });
      for (const Index w : next) order.push_back(w);  // (vector::insert trips a GCC 13 -O3 false positive)
    }
    std::reverse(order.begin() + static_cast<std::ptrdiff_t>(first), order.end());
  }
  return order;
}

std::vector<Index> inverse(const std::vector<Index>& p) {
  std::vector<Index> r(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) r[p[i]] = static_cast<Index>(i);
  return r;
}

}  // namespace

Ordering compute_ordering(const io::Model& m, Renumbering mode) {
  Ordering o;
  const auto N = static_cast<Index>(m.nodes.size());
  if (mode == Renumbering::Off) {
    o.node_order.resize(static_cast<std::size_t>(N));
    std::iota(o.node_order.begin(), o.node_order.end(), 0);
  } else {
    o.node_order = rcm(node_graph(m));
  }
  o.node_rank = inverse(o.node_order);

  // Elements by smallest internal node index (ties: file position). Off keeps file order.
  const auto idx = index_of_tag(m);
  std::vector<Index> key;
  for (const auto& g : m.elements) {
    const int nn = io::node_count(g.type);
    for (std::size_t e = 0; e < g.size(); ++e) {
      Index k = INT32_MAX;
      for (int a = 0; a < nn; ++a) k = std::min(k, o.node_rank[idx.at(g.connectivity[e * nn + a])]);
      key.push_back(k);
    }
  }
  o.element_order.resize(key.size());
  std::iota(o.element_order.begin(), o.element_order.end(), 0);
  if (mode != Renumbering::Off) {
    std::stable_sort(o.element_order.begin(), o.element_order.end(),
                     [&](Index a, Index b) { return key[a] < key[b]; });
  }
  o.element_rank = inverse(o.element_order);
  return o;
}

std::int32_t node_bandwidth(const io::Model& m, const std::vector<std::int32_t>& node_rank) {
  const auto idx = index_of_tag(m);
  std::int32_t bw = 0;
  for (const auto& g : m.elements) {
    const int nn = io::node_count(g.type);
    for (std::size_t e = 0; e < g.size(); ++e)
      for (int a = 0; a < nn; ++a)
        for (int b = 0; b < nn; ++b) {
          const auto ra = node_rank[idx.at(g.connectivity[e * nn + a])];
          const auto rb = node_rank[idx.at(g.connectivity[e * nn + b])];
          bw = std::max(bw, std::abs(ra - rb));
        }
  }
  return bw;
}

}  // namespace femorax::compile
