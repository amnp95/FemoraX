// GoogleTest entry point that initializes Kokkos once for the whole test binary.
// Kokkos arguments (for example --kokkos-num-threads=8) are consumed before GoogleTest
// parses the rest.

#include "exec/runtime.hpp"

#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
  femorax::Runtime runtime(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
