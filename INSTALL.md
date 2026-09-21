# Installing mfemElasticity

## Prerequisites

1. **CMake** 3.15 or later.
2. **A C++20 compiler** (GCC 10, Clang 11 or later).
3. **MFEM**, built or installed. For the parallel build, a parallel MFEM
   (with hypre and METIS). See the [MFEM build
   instructions](https://mfem.org/building/). CMake needs the directory
   holding `MFEMConfig.cmake`, normally the MFEM build or install directory.
4. **MPI** (optional), for the parallel build: the same MPI that MFEM was
   built with.
5. **Python 3.12 or later** (examples and tests only). The gmsh meshes the
   examples and tests read are generated at build time by the scripts in
   `meshes/`, which need the `planetmodel[meshing,mfem]` package. If the
   Python that CMake finds lacks it, a virtual environment is created under
   the build directory and the package installed there. Pass
   `-DMESHES_PYTHON=<python>` to use an environment of your own, or
   `-DGENERATE_MESHES=OFF` to skip generation. See `meshes/README.md`.
6. **Doxygen** (optional), for the API documentation.

## Configure

In-source builds are refused; configure into a separate directory.

Serial:

```bash
cmake -S . -B build_serial \
      -DMFEM_DIR=/path/to/mfem_serial_build \
      -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
```

Parallel:

```bash
cmake -S . -B build_parallel \
      -DUSE_MPI=ON \
      -DMFEM_DIR=/path/to/mfem_parallel_build \
      -DMPI_C_COMPILER=/path/to/mpicc \
      -DMPI_CXX_COMPILER=/path/to/mpic++ \
      -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
```

`CMAKE_PREFIX_PATH` may be given instead of `MFEM_DIR`. Naming the MPI
compiler wrappers matters when several MPI installations are present: the
`mpiexec` beside the wrapper is the launcher used for the MPI tests, unless
`MPIEXEC_EXECUTABLE` is set.

| Option | Default | Effect |
|---|---|---|
| `USE_MPI` | `OFF` | build against a parallel MFEM; enables the parallel classes, examples and tests |
| `BUILD_EXAMPLES` | `OFF` | build the programs in `examples/` |
| `BUILD_TESTS` | `OFF` | build the test suite in `tests/` (googletest is fetched at configure time) |
| `BUILD_DOCS` | `OFF` | generate the Doxygen documentation into `<build>/doc` as part of the build |
| `GENERATE_MESHES` | on with examples or tests | generate the gmsh meshes into `<build>/data` |
| `MESHES_PYTHON` | found or created | the Python interpreter used for mesh generation |

The usual CMake variables (`CMAKE_BUILD_TYPE`, `CMAKE_INSTALL_PREFIX`)
apply.

## Build, test, install

```bash
cmake --build build_serial -j 8
(cd build_serial && ctest)             # if BUILD_TESTS=ON
cmake --install build_serial           # optional
```

A first build with examples or tests spends a few minutes generating
meshes; later builds regenerate a mesh only when its script changes. The MPI
tests run at 1, 2 and 4 ranks.

Installation puts the library in `lib/`, the headers in `include/`, and the
CMake package files in `lib/cmake/mfemElasticity`.

## Using the installed library

```cmake
find_package(mfemElasticity REQUIRED)
target_link_libraries(my_program PRIVATE mfemElasticity)
```
