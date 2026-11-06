#pragma once

namespace femorax {

// Wait until all work submitted to the default execution space has finished.
// Benchmarks call this before reading the clock, so asynchronous (GPU) launches are timed
// completely.
void fence();

// Monotonic wall-clock time in seconds.
double wall_time();

}  // namespace femorax
