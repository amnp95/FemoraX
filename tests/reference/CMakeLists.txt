# OpenSees reference build: the full OpenSees at the pinned commit, built with its own
# CMake project, plus the FemoraX reference drivers linked against that build.
#
# This is a separate CMake project from FemoraX (it needs Fortran, Tcl and Eigen, and takes
# a while to build). See tests/reference/README.md:
#
#   cmake -S tests/reference -B build/reference -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/reference --target OpenSees reference_drivers
#
# ../OpenSees is used read-only: OpenSees' CMakeLists.txt is included unchanged with
# add_subdirectory, and everything it builds goes to build/reference/opensees.

cmake_minimum_required(VERSION 3.25)
project(FemoraXReference LANGUAGES C CXX Fortran)

set(OPENSEES_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../OpenSees" CACHE PATH
    "OpenSees source tree (read-only)")
set(OPENSEES_PINNED_COMMIT "955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea")
get_filename_component(OPENSEES_SOURCE_DIR "${OPENSEES_SOURCE_DIR}" ABSOLUTE)

# The reference data is only meaningful at the pinned commit (docs/dependencies.md).
find_package(Git REQUIRED)
execute_process(COMMAND ${GIT_EXECUTABLE} -C ${OPENSEES_SOURCE_DIR} rev-parse HEAD
                OUTPUT_VARIABLE opensees_head OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE git_rc)
execute_process(COMMAND ${GIT_EXECUTABLE} -C ${OPENSEES_SOURCE_DIR} status --porcelain
                --untracked-files=no
                OUTPUT_VARIABLE opensees_dirty OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT git_rc EQUAL 0)
  message(FATAL_ERROR "${OPENSEES_SOURCE_DIR} is not a git checkout")
endif()
if(NOT opensees_head STREQUAL OPENSEES_PINNED_COMMIT)
  message(FATAL_ERROR "OpenSees is at ${opensees_head}, expected the pinned commit "
                      "${OPENSEES_PINNED_COMMIT} (docs/dependencies.md)")
endif()
if(opensees_dirty)
  message(FATAL_ERROR "OpenSees has local changes; the reference build needs a clean "
                      "checkout of ${OPENSEES_PINNED_COMMIT}:\n${opensees_dirty}")
endif()

# OpenSees calls find_package(TCL REQUIRED). CMake's FindTCL then also requires Tk, which
# OpenSees never uses on Linux (only winMain.cpp, mpiMainTest.cpp and the Java evaluator
# include tk.h, and none of them is built here) and never links. A marker value satisfies
# the check without installing tk-dev; it is never passed to the linker.
set(TK_LIBRARY "TK-NOT-USED-BY-OPENSEES" CACHE STRING "Tk is not used by OpenSees")
set(TK_INCLUDE_PATH "TK-NOT-USED-BY-OPENSEES" CACHE STRING "Tk is not used by OpenSees")

# Serial OpenSees only. MPI would add OpenSeesSP/OpenSeesMP, which need MUMPS. MKL is
# disabled so a Windows oneAPI install visible through WSL (/mnt/...) is never picked up;
# OpenSees then uses the system LAPACK/BLAS.
set(CMAKE_DISABLE_FIND_PACKAGE_MPI ON)
set(CMAKE_DISABLE_FIND_PACKAGE_MKL ON)

# Imported targets that OpenSees finds (HDF5::HDF5, ZLIB::ZLIB, Eigen3::Eigen) must be
# visible to the drivers defined in this directory.
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

# OpenSees' own project, unchanged.
add_subdirectory(${OPENSEES_SOURCE_DIR} opensees)
# Build the executable as part of the default target here (it is EXCLUDE_FROM_ALL there).
set_target_properties(OpenSees PROPERTIES EXCLUDE_FROM_ALL FALSE)

# OpenSees sets its include paths and definitions per directory; drivers need the same.
get_directory_property(OPS_INCLUDE_DIRS DIRECTORY ${OPENSEES_SOURCE_DIR} INCLUDE_DIRECTORIES)
get_directory_property(OPS_DEFINITIONS DIRECTORY ${OPENSEES_SOURCE_DIR} COMPILE_DEFINITIONS)

# Libraries and support sources of the OpenSees executable, minus its main().
set(OPS_DRIVER_SOURCES
  ${OPENSEES_SOURCE_DIR}/SRC/tcl/commands.cpp
  ${OPENSEES_SOURCE_DIR}/SRC/actor/objectBroker/FEM_ObjectBrokerAllClasses.cpp)
set(OPS_DRIVER_LIBS
  OPS_InterpTcl coordTransformation OpenSeesLIB OPS_Reliability OPS_ReliabilityTcl
  OPS_Numerics OPS_Recorder ${CMAKE_DL_LIBS} HDF5::HDF5 ${TCL_LIBRARY} ZLIB::ZLIB
  Eigen3::Eigen)

function(femorax_reference_driver name)
  add_executable(${name} drivers/${name}.cpp drivers/driver_common.cpp ${OPS_DRIVER_SOURCES})
  target_include_directories(${name} PRIVATE ${OPS_INCLUDE_DIRS} drivers)
  target_compile_definitions(${name} PRIVATE ${OPS_DEFINITIONS}
                             OPENSEES_COMMIT="${OPENSEES_PINNED_COMMIT}")
  target_link_libraries(${name} PRIVATE ${OPS_DRIVER_LIBS})
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/drivers)
  add_dependencies(reference_drivers ${name})
endfunction()

add_custom_target(reference_drivers)
femorax_reference_driver(nd_material)
femorax_reference_driver(brick_element)
femorax_reference_driver(dof_numbering)
femorax_reference_driver(brick_bench)
