# Stampede3 (TACC) build environment for FemoraX. Source it, do not run it:
#
#   source scripts/stampede3_env.sh
#
# GCC 13.2 (FemoraX's own build), CMake, Ninja, Python 3.12 (its numpy, h5py, scipy and
# pandas are reused by .venv), HDF5 1.14.6 and Eigen 3.4.0 from TACC's modules. Kokkos and
# GoogleTest are not taken from modules: CMake fetches the pinned versions
# (docs/dependencies.md). The OpenSees reference build may use the Intel compiler instead
# (tests/reference/README.md).
export LMOD_PAGER=cat
module load gcc/13.2.0 cmake/4.1.1 python/3.12.11 > /dev/null 2>&1
module load hdf5/1.14.6 eigen/3.4.0 ninja > /dev/null 2>&1
for m in gcc/13.2.0 cmake python/3.12.11 hdf5/1.14.6 eigen/3.4.0 ninja; do
  module is-loaded "$m" || echo "stampede3_env.sh: module $m did not load" >&2
done
# TACC's usage tracking (XALT) preloads libxalt_init.so into every program through
# LD_PRELOAD; AddressSanitizer must be the first library loaded and refuses to start
# otherwise.
module unload xalt > /dev/null 2>&1
unset LD_PRELOAD
export HDF5_ROOT=$TACC_HDF5_DIR
export Eigen3_ROOT=$TACC_EIGEN_DIR
# Tcl's init.tcl for the OpenSees reference build (Tcl 8.6.11 from Conan, as the user's
# OpenSees build on Stampede3 uses); OpenSees cannot start without it.
for d in "$HOME"/.conan/data/tcl/8.6.11/_/_/package/*/lib/tcl8.6; do
  [ -f "$d/init.tcl" ] && export TCL_LIBRARY="$d" && break
done
