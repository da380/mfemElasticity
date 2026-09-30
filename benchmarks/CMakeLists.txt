# Benchmarks: drivers that solve a case written by the Python scripts beside
# them and write results for comparison with a reference solution. They are
# parallel programs and are built with USE_MPI only.
#
# The build makes the drivers and, beside them, launchers of the scripts:
# <build>/benchmarks/<benchmark>/run and plot start the scripts of the source
# tree with the Python environment, the drivers and the MPI launcher of this
# build, so that a benchmark is run from the build tree and writes there
# (see README.md here). BENCHMARKS_PYTHON names a Python with planetmodel,
# pyslfp and matplotlib; when empty, the poetry environment of this
# directory is looked for.

if(NOT USE_MPI)
    message(STATUS "Benchmarks: skipped (they need USE_MPI)")
    return()
endif()

set(BENCHMARKS_PYTHON "" CACHE FILEPATH
    "Python interpreter with planetmodel, pyslfp and matplotlib; the poetry environment of benchmarks/ if empty")
if(NOT BENCHMARKS_PYTHON)
    find_program(POETRY_EXECUTABLE poetry)
    if(POETRY_EXECUTABLE)
        execute_process(
            COMMAND "${POETRY_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}"
                    env info -p
            OUTPUT_VARIABLE _venv OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET RESULT_VARIABLE _result)
        if(_result EQUAL 0 AND EXISTS "${_venv}/bin/python")
            set(BENCHMARKS_PYTHON "${_venv}/bin/python" CACHE FILEPATH
                "Python interpreter with planetmodel, pyslfp and matplotlib; the poetry environment of benchmarks/ if empty"
                FORCE)
        endif()
    endif()
endif()
if(BENCHMARKS_PYTHON)
    message(STATUS "Benchmarks: scripts run with ${BENCHMARKS_PYTHON}")
else()
    message(WARNING
        "Benchmarks: no Python environment found for the scripts. Run "
        "`poetry install` in ${CMAKE_CURRENT_SOURCE_DIR} and configure again, "
        "or pass -DBENCHMARKS_PYTHON=<python>; the launchers fall back on "
        "`python3`.")
    set(BENCHMARKS_PYTHON "python3")
endif()

# benchmark_launcher(<benchmark> <script> [<argument>...]): the launcher
# <build>/benchmarks/<benchmark>/<script> of <script>.py, with the given
# arguments before the user's.
function(benchmark_launcher benchmark script)
    string(JOIN " " BENCHMARK_ARGUMENTS ${ARGN})
    set(BENCHMARK_SCRIPT
        "${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}/${script}.py")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/launcher.sh.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/${benchmark}/${script}"
                   @ONLY
                   FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                    GROUP_READ GROUP_EXECUTE
                                    WORLD_READ WORLD_EXECUTE)
endfunction()

# --- Love numbers -----------------------------------------------------------

foreach(driver love_benchmark field_benchmark partition_case
        relabelled_identity)
    add_executable(${driver} love_numbers/${driver}.cpp)
    target_link_libraries(${driver} mfemElasticity)
    set_target_properties(${driver} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/love_numbers")
endforeach()

benchmark_launcher(love_numbers run
    "--programs \"${CMAKE_CURRENT_BINARY_DIR}/love_numbers\""
    "--mpiexec \"${MPIEXEC_EXECUTABLE}\"")
benchmark_launcher(love_numbers plot)
benchmark_launcher(love_numbers make_case)
