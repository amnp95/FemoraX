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
