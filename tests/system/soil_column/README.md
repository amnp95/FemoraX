# Soil-column system test (OpenSees vs FemoraX)

A Femora-style layered elastic soil column, written for both programs by
`make_models.py`, run by each, and compared value by value by `compare.py`.

```
D=/tmp/soil_column
.venv/bin/python tests/system/soil_column/make_models.py $D 10 10 40 1.0   # 4000 elements, 1 s
(cd $D && /path/to/OpenSees model.tcl)                                     # about 15 min
(cd $D && build/cpu-release/src/app/femorax run model.femorax.h5 --renumbering off)
.venv/bin/python tests/system/soil_column/compare.py $D                    # gate 1e-12
```

Run FemoraX in subdirectories (`cd $D/rcm && femorax run ../model.femorax.h5 --renumbering
on`, the CUDA build, ...) and compare each against the same OpenSees output by linking the
files. Tolerance and observed differences: `docs/deviations.md` ("System tests").

`--integrator CentralDifference` writes the alphaM-only variant; FemoraX deliberately
differs from OpenSees there (docs/deviations.md, "CentralDifference with lumped stdBrick").

## Implicit stages

`make_implicit.py` writes the same column with consistent mass and Femora's staging:
gravity (implicit-dynamic Newmark 0.6/0.3025, dt 1, 30 steps; or Static LoadControl), then
setTime 0, the UniformExcitation pattern, the recorders, and Newmark shaking. OpenSees runs
`algorithm Linear -factorOnce` with ProfileSPD (or Newton); FemoraX the matrix-free Krylov
solver.

```
.venv/bin/python tests/system/soil_column/make_implicit.py $D 10 10 40 0.5 1e-3 --gravity dynamic
(cd $D && /path/to/OpenSees model.tcl && build/cpu-release/src/app/femorax run model.femorax.h5)
.venv/bin/python tests/system/soil_column/compare.py $D --tol 1e-8
```

`implicit_check.sh` (ctest `system.implicit_vs_opensees`) runs the four variants on a
2x2x10 column in a few seconds.
