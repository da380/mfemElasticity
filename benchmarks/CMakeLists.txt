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

# The benchmarks are organised by family, one sub-folder each with its
# own scripts and outputs (see README.md); common/ holds what they
# share. The drivers of every family collect in <build>/benchmarks/bin,
# and each family's launchers sit in <build>/benchmarks/<family>/, meant
# to be started there so that the outputs land beside them. The master
# script's launcher is <build>/benchmarks/campaign.

# benchmark_driver(<family> <driver>): the program <family>/<driver>.cpp,
# built into <build>/benchmarks/bin with common/ on its include path.
function(benchmark_driver family driver)
    add_executable(${driver} ${family}/${driver}.cpp)
    target_link_libraries(${driver} mfemElasticity)
    target_include_directories(${driver} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/common")
    set_target_properties(${driver} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
endfunction()

# benchmark_launcher(<source> <destination> <script> [<argument>...]):
# the launcher <build>/benchmarks/<destination>/<script> of the script
# <source>/<script>.py, with the given arguments before the user's.
function(benchmark_launcher source destination script)
    string(JOIN " " BENCHMARK_ARGUMENTS ${ARGN})
    set(BENCHMARK_SCRIPT
        "${CMAKE_CURRENT_SOURCE_DIR}/${source}/${script}.py")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/launcher.sh.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/${destination}/${script}"
                   @ONLY
                   FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                    GROUP_READ GROUP_EXECUTE
                                    WORLD_READ WORLD_EXECUTE)
endfunction()

set(BENCHMARK_LAUNCHER_ARGUMENTS
    "--programs \"${CMAKE_CURRENT_BINARY_DIR}/bin\""
    "--mpiexec \"${MPIEXEC_EXECUTABLE}\"")

# --- Love numbers -----------------------------------------------------------

foreach(driver love_benchmark field_benchmark partition_case)
    benchmark_driver(love_numbers ${driver})
endforeach()
benchmark_launcher(love_numbers love_numbers run
    ${BENCHMARK_LAUNCHER_ARGUMENTS})
benchmark_launcher(love_numbers love_numbers plot)
benchmark_launcher(love_numbers love_numbers cmb_report)
benchmark_launcher(common love_numbers make_case)

# --- Relabelling ------------------------------------------------------------

foreach(driver relabelled_identity aspherical_reference)
    benchmark_driver(relabelling ${driver})
endforeach()
benchmark_launcher(relabelling relabelling plot)

# --- Perturbation -----------------------------------------------------------

benchmark_launcher(perturbation perturbation perturbation_check
    ${BENCHMARK_LAUNCHER_ARGUMENTS})
benchmark_launcher(perturbation perturbation plot)

# --- The master script ------------------------------------------------------

benchmark_launcher(. . campaign ${BENCHMARK_LAUNCHER_ARGUMENTS})
