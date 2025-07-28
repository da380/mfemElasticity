# mfemElasticity

This library provides extensions to the [mfem library](https://mfem.org) for solving elastic and viscoelastic problems.

## Installation

The `mfem` library must be installed first. This project uses CMake for configuration and building.

**Basic Configuration Example:**

```bash
# Create a build directory (in-source builds are disabled)
mkdir build
cd build

# Configure for a serial build (default)
cmake ..

# Or, for a parallel build with MPI:
cmake .. \
  -DUSE_MPI=ON \
  -DMFEM_DIR=[path to mfem build or install] \
  -DMPI_C_COMPILER=[path to MPI C compiler] \
  -DMPI_CXX_COMPILER=[path to MPI C++ compiler]

# Build the project
cmake --build .
```