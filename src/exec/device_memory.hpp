#pragma once

#include <cstdint>

namespace femorax::exec {

// Free memory of the default execution space's memory, in bytes: the device's free memory
// for GPU backends (vendor runtime query in src/native/), MemAvailable from /proc/meminfo for
// host backends. Returns -1 if unknown; callers then take the safe choice (no geometry
// cache, and the pre-run memory check reports the estimate without stopping).
std::int64_t device_free_memory_bytes();

}  // namespace femorax::exec
