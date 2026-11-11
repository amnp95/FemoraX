# OpenSees reference build and drivers

FemoraX ports are verified against the OpenSees classes they port (CLAUDE.md rule 1). This
directory builds the **full OpenSees at the pinned commit** with OpenSees' own CMake
project, and small driver programs linked against that build that call the OpenSees
classes directly and write reference data.

- `CMakeLists.txt`: a separate CMake project. It includes `../OpenSees` unchanged with
  `add_subdirectory` (outputs go to `build/reference/opensees`), checks that the checkout
  is clean and at the pinned commit (`docs/dependencies.md`), and adds the drivers.
- `drivers/`: one program per component family.
- `data/`: the generated reference data, committed, so the normal FemoraX test suite never
  needs OpenSees.

The same build provides the OpenSees executable for the system tests
(`build/reference/opensees/OpenSees`).

## Build (once per pinned commit)

System packages: `tcl-dev`, `libeigen3-dev`, plus what FemoraX already needs
(`gfortran`, `libhdf5-dev`, `liblapack-dev`, `zlib1g-dev`).

```
tests/reference/build_hdf5.sh              # HDF5 1.14.6, for OpenSees' VTKHDF recorder
cmake -S tests/reference -B build/reference -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DHDF5_ROOT=$PWD/build/_hdf5/install
cmake --build build/reference -j8          # OpenSees executable and all drivers
```

OpenSees compiles its HDF5 recorders (VTKHDF, MPCO) only with HDF5 1.12 or newer; with the
system HDF5 1.10 `recorder vtkhdf` is "not found".

### Stampede3 (TACC)

`scripts/stampede3_opensees.sh` (on a compute node, about 3 minutes) builds the same project
in `build/reference-s3-intel` with the Intel compilers, following the user's own Stampede3
OpenSees build: `icx`/`icpx`/`ifx`, MKL linked statically as BLAS/LAPACK, Tcl 8.6.11 from
Conan (`~/.conan`), TACC's HDF5 1.14.6 and Eigen 3.4.0, and `-fp-model=precise` so the
compiler keeps the order of floating-point operations. It writes a launcher,
`build/reference-s3-intel/OpenSees.sh`, that sets the Intel runtime, HDF5 and Tcl paths;
the `s3-*` presets use it for the live comparisons (`system.vtkhdf_vs_opensees`,
`system.implicit_vs_opensees`, both passing).

The stored data stays the GCC data from the laptop. The Intel drivers agree with it to
rounding (2026-11-10): constraints 11 of 11 files identical; elements 22 of 32 identical,
the others within 8.4e-15 of each file's largest value; materials 57 of 86 identical, the
others within 5.8e-14, with no plastic path taking a different branch. The remaining
differences are most likely Intel's own math library (`exp`, `pow`, ...).

## Regenerate reference data

```
build/reference/drivers/nd_material tests/reference/data/nd
build/reference/drivers/soil_material tests/reference/data/nd
build/reference/drivers/brick_element tests/reference/data/element
.venv/bin/python tests/reference/make_constraint_cases.py tests/reference/data/constraints/cases
for f in tests/reference/data/constraints/cases/*.txt; do
  build/reference/drivers/dof_numbering $f tests/reference/data/constraints/$(basename $f)
done
```

Commit the data with the port it verifies. Each file records the OpenSees commit that
produced it; a port's tests check that it matches the pinned commit.

## Notes on the build

- **Version label.** OpenSees reads its git version with `git log` in
  `CMAKE_SOURCE_DIR`, which here is this wrapper project, so the version string built into
  this OpenSees (and printed by `version`) names a FemoraX commit, not OpenSees'. The code
  itself is the pinned commit: configuration fails unless `../OpenSees` is a clean checkout
  of it, and every data file records the pinned commit.
- **Tk.** OpenSees calls `find_package(TCL REQUIRED)`, and CMake's FindTCL then also
  demands Tk. OpenSees never uses Tk on Linux, so the wrapper sets `TK_LIBRARY` and
  `TK_INCLUDE_PATH` to a marker value instead of requiring `tk-dev`; nothing links it.
- **Serial only.** MPI is disabled (OpenSeesSP/MP would need MUMPS), and MKL is disabled so a
  Windows oneAPI install visible through WSL is never used; LAPACK/BLAS come from the
  system (OpenBLAS).
- `CMAKE_FIND_PACKAGE_TARGETS_GLOBAL` makes the imported targets OpenSees finds visible to
  the drivers.

## Drivers

| Driver | Covers | Output |
|---|---|---|
| `nd_material` | nD materials along strain paths (monotonic, cyclic, volumetric load/unload, random reversals), with commit and revert | `data/nd/<Material>_<case>_<path>.txt` |
| `soil_material` | PressureDependMultiYield and PressureIndependMultiYield (the copy an element gets) along strain paths at the Gauss-point level: elastic-stage consolidation, the switch to the plastic stage (`updateParameter`, as `updateMaterialStage` does), then monotonic and cyclic constant-volume shear at several amplitudes and confinements, random reversals, confinement changes, extension, a switch with sheared stress, switching back and forth; every 7th step a discarded trial; Femora's three soils and a user G/Gmax backbone; for PressureIndependMultiYield Femora's basin clay, a frictional material and user backbones, with a switch at rest (no strain increment). Columns: stage, the discarded trial, strain, stress, tangent | `data/nd/PressureDependMultiYield_<case>_<path>.txt`, `data/nd/PressureIndependMultiYield_<case>_<path>.txt` |
| `dof_numbering` | OpenSees `TransformationConstraintHandler` + `PlainNumberer` on node/fix/equalDOF cases (`make_constraint_cases.py`): each nodal DOF's equation through the transformation matrix | `data/constraints/<case>.txt` |
| `brick_element` | OpenSees `Brick` (stdBrick) and `SSPbrick` in a real Domain: internal force with body force, tangent, initial tangent, mass (consistent and lumped for Brick), Rayleigh damping forces (all four terms) for 4 geometries x 2 materials x 4 fields. `getRayleighDampingForces` is protected in OpenSees, so the driver re-exports it from subclasses (`BrickProbe`, `SSPbrickProbe`) | `data/element/Brick_<geometry>_<material>_{state,mass}.txt`, `data/element/SSPbrick_<geometry>_<material>_{state,mass}.txt` |
