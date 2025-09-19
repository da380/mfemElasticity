# mfemElasticity

This library provides extensions to the [mfem library](https://mfem.org) for solving elastic and viscoelastic problems.

## Installation

The `mfem` library must be installed first. This project uses CMake for configuration and building.

There are two ways to configure the project: using the provided configure scripts or by calling `cmake` directly.

### Using the configure scripts (recommended)

The project includes two scripts to simplify the configuration process: `configure_serial.sh` and `configure_parallel.sh`. These scripts will create `serial_build` and `parallel_build` directories respectively.

Before running these scripts, you may need to set some environment variables to point to your dependencies.

**Serial Configuration:**

Set the `CMAKE_PREFIX_PATH` to the location of your serial MFEM installation.
```bash
export CMAKE_PREFIX_PATH=/path/to/your/mfem_serial_build
./configure_serial.sh
```

**Parallel Configuration:**

Set the `CMAKE_PREFIX_PATH` to your parallel MFEM installation. You also need to make sure CMake can find your MPI compilers. You can do this by adding the compiler's location to your `PATH`, or by setting the `MPI_C_COMPILER` and `MPI_CXX_COMPILER` environment variables.

Using `PATH`:
```bash
export CMAKE_PREFIX_PATH=/path/to/your/mfem_parallel_build
export PATH=/path/to/your/mpi/bin:$PATH
./configure_parallel.sh
```

Using `MPI_C_COMPILER` and `MPI_CXX_COMPILER`:
```bash
export CMAKE_PREFIX_PATH=/path/to/your/mfem_parallel_build
export MPI_C_COMPILER=/path/to/your/mpicc
export MPI_CXX_COMPILER=/path/to/your/mpic++
./configure_parallel.sh
```

After running the configure script, you can build the project:
```bash
cmake --build serial_build # for serial
# or
cmake --build parallel_build # for parallel
```

### Manual CMake configuration

You can also configure the project by calling `cmake` directly.

**Serial Configuration:**
```bash
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/your/mfem_serial_build
cmake --build .
```

**Parallel Configuration:**
```bash
mkdir build
cd build
cmake .. -DUSE_MPI=ON \
         -DCMAKE_PREFIX_PATH=/path/to/your/mfem_parallel_build \
         -DMPI_C_COMPILER=/path/to/your/mpicc \
         -DMPI_CXX_COMPILER=/path/to/your/mpic++
cmake --build .
```

### Build Options

You can control which optional parts of the project are built by passing CMake options to the configure scripts or to `cmake` directly.

The configure scripts enable some options by default, but you can override them. For example, to turn an option off, you would use `-D<OPTION_NAME>=OFF`.

The following options are available:

*   `BUILD_EXAMPLES`: Build the example programs in the `examples` directory. (Default: `ON` in configure scripts)
*   `BUILD_TESTS`: Build the test suite in the `tests` directory. (Default: `ON` in configure scripts)
*   `BUILD_GMSH`: Build the meshing utilities in the `meshing` directory. (Default: `ON` in configure scripts)
    *   **Requires Gmsh:** You must have Gmsh installed. You may need to set `CMAKE_PREFIX_PATH` to point to your Gmsh installation if it's in a non-standard location.
*   `BUILD_DOCS`: Generate the API documentation using Doxygen. (Default: `OFF` in configure scripts)
    *   **Requires Doxygen:** You must have Doxygen installed.

**Example:**

To configure a serial build without the Gmsh-based meshing utilities, you can run:

```bash
export CMAKE_PREFIX_PATH=/path/to/your/mfem_serial_build
./configure_serial.sh -DBUILD_GMSH=OFF
```

To enable documentation generation with the parallel script:
```bash
export CMAKE_PREFIX_PATH=/path/to/your/mfem_parallel_build
export PATH=/path/to/your/mpi/bin:$PATH
./configure_parallel.sh -DBUILD_DOCS=ON
```
