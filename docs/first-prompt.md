Read CLAUDE.md and docs/design.md. Then look through ../Femora to understand how a Femora
model is built and exported (start with the Tcl export code and the generated API knowledge
files), and through ../OpenSees for ElasticIsotropic, stdBrick, SSPbrick and
CentralDifference. Summarize in a few lines what you learned from each.

We are starting milestone 1a: the layered elastic soil column with explicit central
difference, end to end on the Kokkos CPU backend. My GPU laptops have not arrived yet.

Before writing any code, give me a plan for:

1. The CMake project with Kokkos (Serial and OpenMP backends now, CUDA added later) and
   the directory layout from CLAUDE.md.
2. HDF5 model schema v0, covering only what milestone 1a needs, and how Femora's assembled
   mesh maps onto it.
3. The physics layer for ElasticIsotropic, stdBrick and SSPbrick (force-only path and
   critical_dt), plus the OpenSees reference drivers and tests that compare against them.
4. The host compile step: DOF numbering with fix and equalDOF folded in, element batches,
   and the lumped mass vector.
5. The explicit time loop: central difference, Rayleigh damping without a matrix,
   UniformExcitation, and the stable time step check.
6. A node recorder and a VTKHDF recorder.
7. The system test: run the same column through OpenSees and FemoraX and compare.

For each part, list the files you will create and the tests that prove it works.
End with the open decisions you need from me (test framework, bindings, schema details).
Do not write code until I approve the plan.
