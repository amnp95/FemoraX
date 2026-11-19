# Stabilized LF-LTS: a 1D reproduction (2026-11-19)

`gms1d.py` implements the stabilized leapfrog local time stepping algorithm of Grote, Michel &
Sauter (Math. Comp. 2021, arXiv:2005.13350, section 3.3, read from the arXiv LaTeX source) on
their 1D test (section 5): wave equation on (0, 1), c = 1, Dirichlet ends, P1 elements with
mass lumping, coarse mesh on (0, 0.9) with h_c, fine mesh on [0.9, 1) with h_f = h_c / p; the
fine DOFs are the nodes of the closed fine region (the interface node included). Stability:
every eigenvalue of dt^2 A^{p,nu} strictly inside (0, 4), with A^{p,nu} built column by column
from one step (u_next = 2 u - u_prev - dt^2 A^{p,nu} u).

Results (numpy, `.venv`):
- p = 1 gives A exactly (one level is leapfrog).
- Without stabilization (nu = 0), p = 2: an unstable band of steps starts at dt = 0.7097 h_c
  (h_c = 1/80) and 0.7077 h_c (h_c = 1/160); the paper's first critical step is 0.7104 h_c.
  p = 3, h_c = 1/40: the most critical step 0.5038 dt_opt, the paper's first one 0.5011 dt_opt.
  On h_c = 1/20 the window 0.70-0.725 h_c has no unstable step (the critical steps depend a
  little on the mesh).
- With nu = 0.01, every eigenvalue stays inside (0, 4) over the same windows (smallest margin
  1.5e-3 to 3.0e-3).

Details for the implementation: the fine substeps apply A to the fine DOFs' values only
(A Pi_f z, a column restriction), so in matrix-free form they evaluate every element touching
a fine DOF with the coarse DOFs' values set to zero, and the result lands on the fine DOFs and
their coarse neighbours; the coarse force w = A Pi_c u is computed once per coarse step. The
conserved energy is E = 1/2 (|(u^{n+1} - u^n)/dt|_M^2 + a^{p,nu}(u^{n+1}, u^n)), with the
stabilized form a^{p,nu}(u, v) = v^T M A^{p,nu} u.
