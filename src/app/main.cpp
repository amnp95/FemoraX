// femorax command-line program.
//
//   femorax --version        print the FemoraX version
//   femorax info             print the build configuration (Kokkos version and backends)
//   femorax check <model.h5> read and validate a model, and check this build can run it
//   femorax run <model.h5>   run a model [--renumbering off|on] [--quiet]
//
// Kokkos options (for example --kokkos-num-threads=8) are accepted by every command.
//
// Exit status: 0 success, 1 usage error, 2 not implemented,
//              3 unsupported component (the model should run on the OpenSees backend),
//              4 invalid model file.

#include "exec/runtime.hpp"
#include "femorax_version.hpp"
#include "exec/explicit_analysis.hpp"
#include "io/model_reader.hpp"
#include "io/support.hpp"

#include <cstdio>
#include <string_view>

namespace {

void print_usage() {
  std::puts(
      "usage: femorax <command> [args]\n"
      "\n"
      "commands:\n"
      "  --version          print the FemoraX version\n"
      "  info               print the build configuration\n"
      "  check <model.h5>   validate a model and check this build can run it\n"
      "  run <model.h5>     run a model [--renumbering off|on] [--quiet]\n");
}

int cmd_info() {
  std::printf("FemoraX %s\n", femorax::version_string);
  std::printf("Kokkos %s\n", femorax::kokkos_version().c_str());
  std::printf("execution spaces:");
  for (const auto& s : femorax::enabled_execution_spaces()) std::printf(" %s", s.c_str());
  std::printf("\ndefault space: %s (concurrency %d, SIMD width %d)\n",
              femorax::default_execution_space().c_str(), femorax::default_concurrency(),
              femorax::simd_width());
  return 0;
}

constexpr int kExitUnsupported = 3;
constexpr int kExitInvalid = 4;

int cmd_check(const char* path) {
  try {
    const femorax::io::Model m = femorax::io::read_model(path);
    for (const auto& w : m.warnings) std::fprintf(stderr, "warning: %s\n", w.c_str());
    femorax::io::check_supported(m, femorax::io::Capabilities{});
    std::printf("%s: schema %d.%d, %zu nodes, %zu elements in %zu groups, %zu process steps\n",
                path, m.version.major, m.version.minor, m.nodes.size(), m.element_count(),
                m.elements.size(), m.process.size());
    std::printf("ok: valid model; every component is in this build's supported set\n");
    return 0;
  } catch (const femorax::io::UnsupportedComponent& e) {
    std::fprintf(stderr, "%s: %s\n", path, e.what());
    return kExitUnsupported;
  } catch (const femorax::io::ModelError& e) {
    std::fprintf(stderr, "%s: invalid model: %s\n", path, e.what());
    return kExitInvalid;
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc >= 2 && std::string_view(argv[1]) == "--version") {
    std::printf("femorax %s\n", femorax::version_string);
    return 0;
  }

  // Host-only commands, without starting Kokkos.
  if (argc >= 2 && std::string_view(argv[1]) == "check") {
    if (argc != 3) {
      std::fprintf(stderr, "usage: femorax check <model.h5>\n");
      return 1;
    }
    return cmd_check(argv[2]);
  }

  femorax::Runtime runtime(argc, argv);  // strips --kokkos-* arguments from argv

  if (argc < 2) {
    print_usage();
    return 1;
  }
  const std::string_view cmd = argv[1];
  if (cmd == "info") return cmd_info();
  if (cmd == "run") {
    if (argc < 3) {
      std::fprintf(stderr, "usage: femorax run <model.h5> [--renumbering off|on] [--quiet]\n");
      return 1;
    }
    femorax::exec::RunOptions opt;
    for (int i = 3; i < argc; ++i) {
      const std::string_view a = argv[i];
      if (a == "--renumbering" && i + 1 < argc) {
        const std::string_view v = argv[++i];
        opt.renumbering = v == "off" ? femorax::compile::Renumbering::Off : femorax::compile::Renumbering::RCM;
      } else if (a == "--quiet") {
        opt.verbose = false;
      } else {
        std::fprintf(stderr, "femorax run: unknown option '%.*s'\n", static_cast<int>(a.size()), a.data());
        return 1;
      }
    }
    try {
      const femorax::io::Model m = femorax::io::read_model(argv[2]);
      for (const auto& w : m.warnings) std::fprintf(stderr, "warning: %s\n", w.c_str());
      femorax::io::check_supported(m, femorax::io::Capabilities{});
      const auto r = femorax::exec::run_model(m, opt);
      if (opt.verbose) {
        std::printf("femorax: done in %.3f s (compile %.3f s, setup %.3f s), %lld records written\n",
                    r.total_seconds, r.compile_seconds, r.setup_seconds, static_cast<long long>(r.records));
        std::printf("femorax: %lld equations, %lld elements; device memory estimate %.3f GB (%s free at start); "
                    "force settings %s (%s)\n",
                    static_cast<long long>(r.equations), static_cast<long long>(r.elements),
                    static_cast<double>(r.device_bytes_estimate) / 1e9,
                    r.device_bytes_free > 0
                        ? (std::to_string(static_cast<double>(r.device_bytes_free) / 1e9).substr(0, 5) + " GB").c_str()
                        : "unknown",
                    r.force_settings.c_str(), r.force_settings_source.c_str());
      }
      return 0;
    } catch (const femorax::io::UnsupportedComponent& e) {
      std::fprintf(stderr, "%s: %s\n", argv[2], e.what());
      return kExitUnsupported;
    } catch (const femorax::io::ModelError& e) {
      std::fprintf(stderr, "%s: %s\n", argv[2], e.what());
      return kExitInvalid;
    }
  }
  std::fprintf(stderr, "femorax: unknown command '%.*s'\n\n", static_cast<int>(cmd.size()),
               cmd.data());
  print_usage();
  return 1;
}
