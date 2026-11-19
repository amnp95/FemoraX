# Embedded elements, constraints and interfaces for HPC (literature review, 2026-11-19)

Requested by the user: FemoraX should support embedded elements (piles and members in a
non-conforming soil mesh) to remove the meshing challenge; this review covers embedded-beam
formulations, constraint enforcement (rigid links, diaphragms, tied and embedded
constraints) and interface/contact modelling, for explicit and implicit GPU matrix-free
analysis.

- `report.md`: the comparison and the recommendation for FemoraX.
- `notes/`: the three research notes behind it (embedded beams; constraint enforcement;
  interfaces and contact), a source for each claim; claims from abstracts or search summaries
  only, and references from memory, are marked unverified.

OpenSees findings in it (commit 955ceb0a5, read from the source, not yet confirmed by a run):
the explicit rigid-diaphragm mass (DiagonalSOE keeps only the diagonal of T^T M T), and two
ZeroLengthContactASDimplex issues (a zero sliding tangent; a restart index). To be confirmed by
small tests before they become docs/deviations.md entries.
