#!/usr/bin/env bash
# The OpenSees reference build on Stampede3 (TACC), with the Intel compilers: the serial
# OpenSees at the pinned commit and the reference drivers of tests/reference, in
# build/reference-s3-intel. Run on a compute node (about 3 minutes on 112 cores):
#
#   scripts/stampede3_opensees.sh
#
# The recipe follows the user's own Stampede3 OpenSees build (Intel oneAPI, MKL linked
# statically, Tcl 8.6.11 from Conan), serial only, with -fp-model=precise so the compiler
# does not reorder floating-point arithmetic. Its output agrees with the stored GCC
# reference data (tests/reference/data) to rounding; the stored data stays the GCC one.
#
# It also writes build/reference-s3-intel/OpenSees.sh, a launcher that starts OpenSees
# with the Intel runtime, Intel HDF5 and Tcl library paths it was built with, so it runs
# from any environment (the s3-* presets point FEMORAX_OPENSEES at it).
set -euo pipefail
cd "$(dirname "$0")/.."
export LMOD_PAGER=cat
module reset > /dev/null 2>&1
module load intel hdf5/1.14.6 eigen/3.4.0 cmake/4.1.1 ninja python/3.12.11 > /dev/null 2>&1
module unload xalt > /dev/null 2>&1
unset LD_PRELOAD OMP_NUM_THREADS

TCL=$(ls -d "$HOME"/.conan/data/tcl/8.6.11/_/_/package/*/ 2>/dev/null | head -1)
[ -f "${TCL}lib/tcl8.6/init.tcl" ] || {
  echo "Tcl 8.6.11 not found in ~/.conan (conan install tcl/8.6.11@ ... as in the user's compile.sh)" >&2
  exit 1
}
M=$TACC_MKL_LIB
MKL="-Wl,--start-group $M/libmkl_intel_lp64.a $M/libmkl_sequential.a $M/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl"
B=build/reference-s3-intel

CC=icx CXX=icpx FC=ifx cmake -S tests/reference -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-fp-model=precise -Wno-error=implicit-int -Wno-error=implicit-function-declaration" \
  -DCMAKE_CXX_FLAGS="-fp-model=precise" -DCMAKE_Fortran_FLAGS="-fp-model=precise" \
  -DHDF5_ROOT="$TACC_HDF5_DIR" -DCMAKE_PREFIX_PATH="$TACC_EIGEN_DIR" \
  -DPython3_ROOT_DIR=/opt/apps/python/3.12.11 \
  -DBLA_STATIC=ON -DBLA_VENDOR=Intel10_64lp -DLAPACK_LIBRARIES="$MKL" -DBLAS_LIBRARIES="$MKL" \
  -DTCL_INCLUDE_PATH="${TCL}include" -DTCL_LIBRARY="${TCL}lib/libtcl8.6.a" > "$B.configure.log" 2>&1 || {
  tail -20 "$B.configure.log"; exit 1; }
cmake --build "$B" -j"$(nproc)" > "$B.build.log" 2>&1 || { grep -E " error|FAILED" "$B.build.log" | head -20; exit 1; }

cat > "$B/OpenSees.sh" <<EOF
#!/usr/bin/env bash
# Written by scripts/stampede3_opensees.sh: OpenSees with the library paths of its build.
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export TCL_LIBRARY="${TCL}lib/tcl8.6"
unset LD_PRELOAD
exec "$PWD/$B/opensees/OpenSees" "\$@"
EOF
chmod +x "$B/OpenSees.sh"
echo "OpenSees: $PWD/$B/OpenSees.sh; drivers: $PWD/$B/drivers/"
