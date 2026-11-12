# Vista (TACC, NVIDIA Grace CPUs: gg nodes, and the Grace side of gh nodes) build environment
# for FemoraX. Source it, do not run it:
#
#   source scripts/vista_env.sh
#
# GCC 13.2, CMake, Ninja, HDF5 1.14.6 and Eigen 3.4.0 from TACC's modules (Vista's default
# compiler is NVIDIA's; loading gcc swaps it). Python: the project's Vista environment
# .venv-vista (numpy, h5py), separate from Stampede3's .venv because /work is shared by both
# machines and their Pythons differ. Create it once:
#   module load gcc/13.2.0 python3/3.11.8 && python3 -m venv .venv-vista &&
#   .venv-vista/bin/pip install numpy h5py
# Kokkos and GoogleTest come from build/_kokkos and build/_gtest (docs/dependencies.md).
export LMOD_PAGER=cat
module load gcc/13.2.0 > /dev/null 2>&1
module load cmake/4.1.1 hdf5/1.14.6 eigen/3.4.0 ninja/1.12.1 python3/3.11.8 > /dev/null 2>&1
for m in gcc/13.2.0 cmake/4.1.1 hdf5/1.14.6 eigen/3.4.0 ninja/1.12.1 python3/3.11.8; do
  module is-loaded "$m" || echo "vista_env.sh: module $m did not load" >&2
done
module unload xalt > /dev/null 2>&1  # TACC's usage tracking preloads a library into every program
unset LD_PRELOAD
export HDF5_ROOT=$TACC_HDF5_DIR
export Eigen3_ROOT=$TACC_EIGEN_DIR
