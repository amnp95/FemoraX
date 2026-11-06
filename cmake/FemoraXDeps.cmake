# Third-party dependencies, pinned by version and hash.
# Bump a version only together with its hash, and record the change in docs/dependencies.md.

include(FetchContent)

# ---------------------------------------------------------------------------
# Kokkos 5.1.0
# Host backends for the CPU build: Serial and OpenMP. A device backend (CUDA, later HIP)
# goes in its own build tree, configured with -DKokkos_ENABLE_CUDA=ON.
# ---------------------------------------------------------------------------
set(FEMORAX_KOKKOS_VERSION 5.1.0)

if(FEMORAX_USE_SYSTEM_KOKKOS)
  find_package(Kokkos ${FEMORAX_KOKKOS_VERSION} EXACT REQUIRED)
else()
  # Defaults for the CPU build; a preset or -D on the command line overrides them.
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Kokkos Serial backend")
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Kokkos OpenMP backend")
  set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(kokkos
    URL https://github.com/kokkos/kokkos/releases/download/5.1.0/kokkos-5.1.0.tar.gz
    URL_HASH SHA256=7bdbdfc88033ed7d940c7940ed8919e1f2b78a9656c69276beb76ad45c41ec4e
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM)
  FetchContent_MakeAvailable(kokkos)
endif()

# ---------------------------------------------------------------------------
# GoogleTest 1.17.0 (tests only)
# ---------------------------------------------------------------------------
if(FEMORAX_BUILD_TESTS)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM)
  FetchContent_MakeAvailable(googletest)
endif()

# ---------------------------------------------------------------------------
# HDF5 (C library, system package libhdf5-dev): model reader and recorders.
# ---------------------------------------------------------------------------
find_package(HDF5 REQUIRED COMPONENTS C)

# ---------------------------------------------------------------------------
# Python for tools and tests: the project's virtual environment (.venv, with numpy and
# h5py) when it exists. Override with -DFEMORAX_PYTHON=<interpreter>.
# ---------------------------------------------------------------------------
if(EXISTS "${PROJECT_SOURCE_DIR}/.venv/bin/python")
  set(_femorax_default_python "${PROJECT_SOURCE_DIR}/.venv/bin/python")
else()
  set(_femorax_default_python "")
endif()
set(FEMORAX_PYTHON "${_femorax_default_python}" CACHE FILEPATH
    "Python interpreter with numpy and h5py, for tools and tests")
if(FEMORAX_BUILD_TESTS AND NOT FEMORAX_PYTHON)
  message(FATAL_ERROR
    "No Python for the tests. Create the virtual environment:\n"
    "  python3 -m venv .venv && .venv/bin/pip install numpy h5py\n"
    "or pass -DFEMORAX_PYTHON=<python with numpy and h5py>.")
endif()
