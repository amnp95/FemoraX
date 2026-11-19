# Explicit time-step remedies for soil-structure interaction (literature review, 2026-11-19)

The evidence behind the local time stepping decision (docs/design.md, "Local time stepping").

- `report.md`: the comparison of local time stepping, subcycling and implicit-explicit
  coupling, mass scaling (conventional, selective, reciprocal), Chebyshev / exponential /
  multirate integrators and element-level remedies, with sources and the recommendation.
- `notes/`: the four research notes behind it, one per method family, with a source for each
  claim. Claims taken only from abstracts or search summaries, and references given from
  memory, are marked unverified: check them in the papers before relying on them.

Produced with a web literature search; the primary papers (arXiv versions are open) are the
authority, especially the stabilized recurrence of Grote, Michel & Sauter (arXiv:2005.13350).
