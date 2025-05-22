#!/bin/bash


cmake -S . -B serial_build \
      -DMFEM_DIR=$HOME/dev/mfem_serial_build  \
      -DBUILD_EXAMPLES=ON \
      -DBUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=install
      
