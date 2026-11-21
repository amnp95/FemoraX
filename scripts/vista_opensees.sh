#!/usr/bin/env bash
# The OpenSees reference build on Vista (TACC; NVIDIA Grace, aarch64), with GCC: the serial
# OpenSees at the pinned commit and the reference drivers of tests/reference, in
# build/reference-vista. Run on a compute node (gg or gh):
#
#   scripts/vista_opensees.sh
#
# The counterpart of scripts/stampede3_opensees.sh (Intel compilers, MKL, Conan's Tcl), with what
# Vista has: GCC 13.2 and gfortran, NVIDIA's NVPL for BLAS and LAPACK (sequential, LP64), and the
# system's Tcl 8.6 (/usr/include/tcl.h, /usr/lib64/libtcl8.6.so). ../OpenSees is read only
# (tests/reference/CMakeLists.txt includes its CMake project unchanged); everything is built in
# build/reference-vista, so Stampede3's build/reference-s3-intel on the shared /work is untouched.
# -ffp-contract=off as in FemoraX's Vista builds: GCC fuses a*b+c into FMA instructions on aarch64
# by default, which changes the last bits against the x86 reference data.
#
# It also writes build/reference-vista/OpenSees.sh, a launcher that starts OpenSees with the
# library paths it was built with, so it runs from any environment.
set -euo pipefail
cd "$(dirname "$0")/.."
export LMOD_PAGER=cat
module load gcc/13.2.0 > /dev/null 2>&1
module load nvpl/24.7 cmake/4.1.1 hdf5/1.14.6 eigen/3.4.0 ninja/1.12.1 python3/3.11.8 > /dev/null 2>&1
module unload xalt > /dev/null 2>&1
unset LD_PRELOAD OMP_NUM_THREADS
for m in gcc/13.2.0 nvpl/24.7 cmake/4.1.1 hdf5/1.14.6 eigen/3.4.0; do
  module is-loaded "$m" || { echo "vista_opensees.sh: module $m did not load" >&2; exit 1; }
done
[ -f /usr/include/tcl.h ] && [ -f /usr/lib64/libtcl8.6.so ] && [ -f /usr/share/tcl8.6/init.tcl ] || {
  echo "vista_opensees.sh: the system's Tcl 8.6 (tcl.h, libtcl8.6.so, init.tcl) was not found" >&2
  exit 1
}
N=$TACC_NVPL_LIB
NVPL="$N/libnvpl_lapack_lp64_seq.so;$N/libnvpl_blas_lp64_seq.so"
B=build/reference-vista
FP="-ffp-contract=off"

CC=gcc CXX=g++ FC=gfortran cmake -S tests/reference -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="$FP -Wno-error=implicit-int -Wno-error=implicit-function-declaration" \
  -DCMAKE_CXX_FLAGS="$FP" -DCMAKE_Fortran_FLAGS="$FP" \
  -DHDF5_ROOT="$TACC_HDF5_DIR" -DCMAKE_PREFIX_PATH="$TACC_EIGEN_DIR" \
  -DLAPACK_LIBRARIES="$NVPL" -DBLAS_LIBRARIES="$N/libnvpl_blas_lp64_seq.so" \
  -DTCL_INCLUDE_PATH=/usr/include -DTCL_LIBRARY=/usr/lib64/libtcl8.6.so > "$B.configure.log" 2>&1 || {
  tail -20 "$B.configure.log"; exit 1; }
cmake --build "$B" -j"$(nproc)" > "$B.build.log" 2>&1 || { grep -E " error|FAILED" "$B.build.log" | head -20; exit 1; }

cat > "$B/OpenSees.sh" <<EOT
#!/usr/bin/env bash
# Written by scripts/vista_opensees.sh: OpenSees with the library paths of its build.
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export TCL_LIBRARY=/usr/share/tcl8.6
unset LD_PRELOAD
exec "$PWD/$B/opensees/OpenSees" "\$@"
EOT
chmod +x "$B/OpenSees.sh"
echo "OpenSees: $PWD/$B/OpenSees.sh; drivers: $PWD/$B/drivers/"
