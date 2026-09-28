# Benchmarks: drivers that solve a case written by the Python scripts beside
# them and write results for comparison with a reference solution. They are
# parallel programs and are built with USE_MPI only; the cases are made and
# the runs launched by the scripts (see README.md here), not by the build.

if(NOT USE_MPI)
    message(STATUS "Benchmarks: skipped (they need USE_MPI)")
    return()
endif()

add_executable(love_benchmark love_numbers/love_benchmark.cpp)
target_link_libraries(love_benchmark mfemElasticity)
