## Installation Instructions for `mfemElasticity`

This guide provides instructions to build and install the `mfemElasticity` project.

---

### Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **CMake (version 3.15 or higher):** A cross-platform open-source make system.
    * You can download it from [cmake.org](https://cmake.org/).
    * On Ubuntu/Debian: `sudo apt-get install cmake`
    * On macOS (with Homebrew): `brew install cmake`

2.  **C++ Compiler with C++20 Support:** A C++ compiler that supports the C++20 standard.
    * Examples include GCC (g++) 10 or newer, Clang (clang++) 11 or newer, or MSVC (Visual Studio) 2019 or newer.

3.  **MFEM Library:** The `mfemElasticity` project depends on the [MFEM library](https://mfem.org/). You'll need to have MFEM built and/or installed on your system.
    * Refer to the [MFEM documentation](https://mfem.org/build/) for build and installation instructions.
    * **Important:** You'll likely need to know the path to your MFEM build or installation directory (e.g., where `MFEMConfig.cmake` is located).

4.  **Optional: MPI (Message Passing Interface) Library:** If you intend to build `mfemElasticity` with MPI support for parallel execution.
    * Examples include Open MPI, MPICH.
    * On Ubuntu/Debian: `sudo apt-get install openmpi-bin libopenmpi-dev`

5.  **Optional: Doxygen:** If you wish to generate the project's documentation.
    * On Ubuntu/Debian: `sudo apt-get install doxygen`
    * On macOS (with Homebrew): `brew install doxygen`

---

### Build and Installation Steps

It's recommended to perform an "out-of-source" build, as enforced by the `CMakeLists.txt` file. This means you should create a separate directory for your build files.

1.  **Clone the Repository (if applicable):**
    If your project is in a Git repository, clone it:
    ```bash
    git clone <repository_url>
    cd mfemElasticity
    ```
    (Replace `<repository_url>` with the actual URL of your project's repository.)

2.  **Create a Build Directory:**
    Navigate to the root of your `mfemElasticity` project and create a build directory:
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure the Project with CMake:**
    From inside the `build` directory, run CMake. You may need to provide hints for finding the MFEM library.

    * **Basic Configuration** (assuming MFEM is found in standard locations):
        ```bash
        cmake ..
        ```

    * **Specifying MFEM Location:**
        If CMake struggles to find MFEM, explicitly tell it where to look using the `MFEM_DIR` or `mfem_DIR` CMake variables. Replace `/path/to/mfem_install` with the actual path to your MFEM installation or build directory.

        ```bash
        cmake -DMFEM_DIR=/path/to/mfem_install ..
        # OR, if you know the exact location of MFEMConfig.cmake:
        cmake -Dmfem_DIR=/path/to/mfem_install/lib/cmake/mfem ..
        ```

    * **Enabling MPI Support:**
        To build with MPI, add `-DUSE_MPI=ON`:
        ```bash
        cmake -DUSE_MPI=ON ..
        ```
        (You can combine this with MFEM path specification if needed.)

    * **Building Examples and Tests:**
        To build the example programs and test programs, add the respective options:
        ```bash
        cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON ..
        ```

    * **Generating Documentation:**
        If Doxygen is installed and you want to generate documentation, you can explicitly enable it (though it's ON by default if Doxygen is found):
        ```bash
        cmake -DBUILD_DOCS=ON ..
        ```

    You can combine multiple options in a single `cmake` command. For example:
    ```bash
    cmake -DMFEM_DIR=/path/to/mfem_install -DUSE_MPI=ON -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON ..
    ```

    After running `cmake`, it will generate the build system files (e.g., Makefiles for Unix, Visual Studio solutions for Windows).

4.  **Build the Project:**
    Compile the project using the generated build system:
    ```bash
    cmake --build . -j <number_of_cores>
    ```
    Replace `<number_of_cores>` with the number of CPU cores you want to use for compilation (e.g., `8`). This will build the `mfemElasticity` library.

    * **To build only documentation (if enabled):**
        ```bash
        cmake --build . --target doc
        ```

5.  **Install the Project:**
    Once the build is successful, you can install the library, headers, and CMake package configuration files to your system's default installation paths (or the paths specified during CMake configuration). This typically requires administrative privileges.

    ```bash
    sudo cmake --install .
    ```
    or, if you installed to a custom prefix using `CMAKE_INSTALL_PREFIX`:
    ```bash
    sudo cmake --install . --prefix /path/to/your/custom/install/location
    ```

    The `mfemElasticity` library will be installed into `lib/`, the header files into `include/`, and the CMake package configuration files (`mfemElasticityConfig.cmake`, `mfemElasticityTargets.cmake`, etc.) into `lib/cmake/mfemElasticity`.

---

### After Installation

Once installed, other CMake projects can find `mfemElasticity` using `find_package(mfemElasticity)`. Here's an example of how to use it in your own `CMakeLists.txt`:

```cmake
find_package(mfemElasticity REQUIRED)
target_link_libraries(my_program PRIVATE mfemElasticity)