#!/bin/bash


cmake -S . -B parallel_build \
      -DUSE_MPI=ON \
      -DBUILD_EXAMPLES=ON \
      -DBUILD_GMSH=ON     \
      -DBUILD_TESTS=ON    \
      -DBUILD_DOCS=OFF \
      "$@"
      
