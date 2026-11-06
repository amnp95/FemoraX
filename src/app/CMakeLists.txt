# The femorax command-line program.
add_executable(femorax main.cpp)
target_link_libraries(femorax PRIVATE femorax::exec femorax::io)
femorax_set_warnings(femorax)
