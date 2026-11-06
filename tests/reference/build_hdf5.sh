#!/usr/bin/env bash
# HDF5 1.14.6 for the OpenSees reference build only. OpenSees enables its HDF5 recorders
# (VTKHDF, MPCO) only with HDF5 >= 1.12 (CMakeLists.txt: _HDF5), and the system library
# here is 1.10. FemoraX itself keeps using the system HDF5.
#
#   tests/reference/build_hdf5.sh            # into build/_hdf5/install
#   cmake -S tests/reference -B build/reference -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DHDF5_ROOT=$PWD/build/_hdf5/install
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
D=$ROOT/build/_hdf5
URL=https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.6/hdf5-1.14.6.tar.gz
SHA=e4defbac30f50d64e1556374aa49e574417c9e72c6b1de7a4ff88c4b1bea6e9b
mkdir -p "$D/src"
[ -f "$D/src/hdf5.tar.gz" ] || curl -sSLf -o "$D/src/hdf5.tar.gz" "$URL"
echo "$SHA  $D/src/hdf5.tar.gz" | sha256sum -c -
tar xzf "$D/src/hdf5.tar.gz" -C "$D/src"
cmake -S "$D/src/hdf5-1.14.6" -B "$D/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$D/install" -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_FORTRAN=OFF \
  -DHDF5_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DHDF5_BUILD_TOOLS=ON \
  -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=OFF
cmake --build "$D/build" -j8
cmake --install "$D/build"
