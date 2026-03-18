#!/bin/bash

cmake -S . -B serial_build \
      -DCMAKE_C_COMPILER=gcc-14.1.0          \
      -DCMAKE_CXX_COMPILER=g++-14.1.0        \
      -DMFEM_DIR=$HOME/dev/mfem_serial_build \
      -DBUILD_EXAMPLES=ON \
      -DBUILD_TESTS=ON \
      -DCMAKE_INSTALL_PREFIX=install \
      -DBUILD_GMSH=ON \
      "$@"
      
