// femorax command-line program.
//
//   femorax --version        print the FemoraX version
//   femorax info             print the build configuration (Kokkos version and backends)
//   femorax run <model.h5>   run a model (not implemented yet)
//
// Kokkos options (for example --kokkos-num-threads=8) are accepted by every command.

#include "exec/runtime.hpp"
#include "femorax_version.hpp"

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
      "  run <model.h5>     run a model (not implemented yet)\n");
}

int cmd_info() {
  std::printf("FemoraX %s\n", femorax::version_string);
  std::printf("Kokkos %s\n", femorax::kokkos_version().c_str());
  std::printf("execution spaces:");
  for (const auto& s : femorax::enabled_execution_spaces()) std::printf(" %s", s.c_str());
  std::printf("\ndefault space: %s (concurrency %d)\n",
              femorax::default_execution_space().c_str(), femorax::default_concurrency());
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc >= 2 && std::string_view(argv[1]) == "--version") {
    std::printf("femorax %s\n", femorax::version_string);
    return 0;
  }

  femorax::Runtime runtime(argc, argv);  // strips --kokkos-* arguments from argv

  if (argc < 2) {
    print_usage();
    return 1;
  }
  const std::string_view cmd = argv[1];
  if (cmd == "info") return cmd_info();
  if (cmd == "run") {
    std::fprintf(stderr, "femorax run: not implemented yet (milestone 1a in progress)\n");
    return 2;
  }
  std::fprintf(stderr, "femorax: unknown command '%.*s'\n\n", static_cast<int>(cmd.size()),
               cmd.data());
  print_usage();
  return 1;
}
