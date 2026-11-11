# Stampede3 (TACC) environment for the Intel GPU build (preset s3-sycl-pvc, pvc nodes).
# Source it, do not run it:
#
#   source scripts/stampede3_sycl_env.sh
#
# Intel oneAPI 2026.0 (icpx with SYCL; Kokkos 5.1.0's SYCL backend wants oneAPI 2025.1.1 or
# newer), and HDF5, CMake, Ninja and Python as in stampede3_env.sh.
export LMOD_PAGER=cat
module load intel/26.0 cmake/4.1.1 python/3.12.11 > /dev/null 2>&1
module load hdf5/1.14.6 ninja > /dev/null 2>&1
for m in intel/26.0 cmake python/3.12.11 hdf5/1.14.6 ninja; do
  module is-loaded "$m" || echo "stampede3_sycl_env.sh: module $m did not load" >&2
done
module unload xalt > /dev/null 2>&1
unset LD_PRELOAD
export HDF5_ROOT=$TACC_HDF5_DIR
# Eigen 3.4.0 (tests only; header-only): the intel/26.0 hierarchy offers only Eigen 5.0.
export Eigen3_ROOT=/opt/apps/intel24/eigen/3.4.0
# TACC's python/3.12.11 module ships its own, older libur_loader (SYCL's Unified Runtime)
# and puts it on the library paths ahead of the compiler's; libsycl from oneAPI 2026.0 then
# fails to link (undefined urProgramDynamicLinkExp@LIBUR_LOADER_0.12). The compiler's own
# library directory goes first.
_icpx_lib=$(cd "$(dirname "$(command -v icpx)")/../lib" && pwd)
export LIBRARY_PATH="$_icpx_lib:$LIBRARY_PATH" LD_LIBRARY_PATH="$_icpx_lib:$LD_LIBRARY_PATH"
unset _icpx_lib
# Lets the SYCL runtime report the GPU's free memory (FemoraX's memory check).
export ZES_ENABLE_SYSMAN=1
for d in "$HOME"/.conan/data/tcl/8.6.11/_/_/package/*/lib/tcl8.6; do
  [ -f "$d/init.tcl" ] && export TCL_LIBRARY="$d" && break
done
