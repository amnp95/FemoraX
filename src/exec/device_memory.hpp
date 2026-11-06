#pragma once

#include <cstdint>

namespace femorax::exec {

// Free memory of the default execution space's memory, in bytes: the device's free memory
// for GPU backends (vendor runtime query in src/native/), MemAvailable from /proc/meminfo for
// host backends. Returns -1 if unknown.
std::int64_t device_free_memory_bytes();

}  // namespace femorax::exec
