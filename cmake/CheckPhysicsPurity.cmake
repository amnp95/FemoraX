# Scan the physics layer for constructs that CLAUDE.md rule 2 forbids.
#
#   cmake -DPHYSICS_DIR=<dir> -P CheckPhysicsPurity.cmake
#
# Fails (FATAL_ERROR) with one line per violation. The scan is deliberately simple and
# conservative: it reads code with comments and string literals removed, line by line.
# The standalone-compile test (-fno-exceptions -fno-rtti, no Kokkos on the include path)
# backs it up.

if(NOT DEFINED PHYSICS_DIR)
  message(FATAL_ERROR "PHYSICS_DIR is not set")
endif()

# Standard headers the physics layer may include. Anything that allocates or throws
# (<vector>, <string>, <memory>, <map>, <functional>, <iostream>, ...) is not listed.
set(allowed_std_headers
  array cassert cfloat climits cmath cstddef cstdint limits numbers type_traits utility
  concepts bit algorithm)

# Forbidden patterns: regex | message
set(rules
  "Kokkos|Kokkos is not allowed in src/physics (rule 2)"
  "(^|[^A-Za-z0-9_])new[ \t]*[A-Za-z_(]|heap allocation (new)"
  "(^|[^A-Za-z0-9_])delete([ \t]|<lb>)|heap allocation (delete)"
  "(^|[^A-Za-z0-9_])(malloc|calloc|realloc|free)[ \t]*\\(|heap allocation (malloc/free)"
  "(^|[^A-Za-z0-9_])virtual([^A-Za-z0-9_]|$)|virtual functions"
  "(^|[^A-Za-z0-9_])(throw|try|catch)([^A-Za-z0-9_]|$)|exceptions"
  "(^|[^A-Za-z0-9_])(thread_local|mutable)([^A-Za-z0-9_]|$)|mutable or thread-local state"
  "std::(vector|string|map|unordered_map|set|list|deque|function|shared_ptr|unique_ptr)|heap-allocating std type"
  "cuda[A-Z][A-Za-z]*|native CUDA API (use src/native/)"
  "hip[A-Z][A-Za-z]*\\(|native HIP API (use src/native/)")

file(GLOB_RECURSE files "${PHYSICS_DIR}/*.hpp" "${PHYSICS_DIR}/*.h" "${PHYSICS_DIR}/*.cpp")
if(NOT files)
  message(FATAL_ERROR "no physics sources found under ${PHYSICS_DIR}")
endif()

set(violations "")
set(macros_file "${PHYSICS_DIR}/macros.hpp")

foreach(f IN LISTS files)
  file(READ "${f}" text)
  # Drop block comments, then line comments and string literals, keeping newlines so
  # line numbers stay right.
  string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" "" text "${text}")
  # CMake lists split on ';' and treat '[' ']' specially, so mask all three first.
  string(REPLACE ";" "<semicolon>" text "${text}")
  string(REPLACE "[" "<lb>" text "${text}")
  string(REPLACE "]" "<rb>" text "${text}")
  string(REPLACE "\n" ";" lines "${text}")
  set(lineno 0)
  foreach(line IN LISTS lines)
    math(EXPR lineno "${lineno} + 1")
    string(REGEX REPLACE "//.*$" "" code "${line}")

    # Includes: only allowed standard headers or other physics headers.
    if(code MATCHES "^[ \t]*#[ \t]*include[ \t]*<([^>]+)>")
      list(FIND allowed_std_headers "${CMAKE_MATCH_1}" idx)
      if(idx EQUAL -1)
        list(APPEND violations "${f}:${lineno}: include <${CMAKE_MATCH_1}> is not allowed")
      endif()
    elseif(code MATCHES "^[ \t]*#[ \t]*include[ \t]*\"([^\"]+)\"")
      if(NOT CMAKE_MATCH_1 MATCHES "^physics/")
        list(APPEND violations
             "${f}:${lineno}: include \"${CMAKE_MATCH_1}\" is outside src/physics")
      endif()
    endif()

    # Blank string literals (after the include check, which needs the quoted path).
    string(REGEX REPLACE "\"([^\"\\\\]|\\\\.)*\"" "\"\"" code "${code}")

    # macros.hpp is the one place allowed to spell __host__/__device__.
    if(NOT f STREQUAL macros_file AND code MATCHES "__(host|device|global)__")
      list(APPEND violations "${f}:${lineno}: use FEMORAX_HD, not __host__/__device__")
    endif()

    foreach(rule IN LISTS rules)
      string(REPLACE "|" ";" parts "${rule}")
      list(POP_BACK parts msg)
      list(JOIN parts "|" regex)
      if(code MATCHES "${regex}")
        list(APPEND violations "${f}:${lineno}: ${msg}")
      endif()
    endforeach()

    # Static data: 'static' is allowed only for member functions and for constexpr
    # constants. A 'static' line with no '(' that is not constexpr declares static data.
    if(code MATCHES "(^|[^A-Za-z0-9_])static([^A-Za-z0-9_]|$)" AND
       NOT code MATCHES "static_assert|static_cast|constexpr|\\(")
      list(APPEND violations "${f}:${lineno}: static mutable data")
    endif()
  endforeach()
endforeach()

if(violations)
  list(LENGTH violations n)
  list(JOIN violations "\n" report)
  string(REPLACE "<semicolon>" ";" report "${report}")
  string(REPLACE "<lb>" "[" report "${report}")
  string(REPLACE "<rb>" "]" report "${report}")
  message(FATAL_ERROR "physics purity check: ${n} violation(s)\n${report}")
endif()
list(LENGTH files nfiles)
message(STATUS "physics purity check: ${nfiles} file(s) clean")
