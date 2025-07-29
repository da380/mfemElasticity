#!/bin/bash


cmake -S . -B parallel_build \
      -DUSE_MPI=ON \
      -DMFEM_DIR=$HOME/dev/mfem_parallel_build  \
      -DMPI_C_COMPILER=$HOME/dev/petsc-install/bin/mpicc \
      -DMPI_CXX_COMPILER=$HOME/dev/petsc-install/bin/mpic++ \
      -DBUILD_EXAMPLES=ON \
      -DBUILD_TESTS=ON    \
      -DBUILD_DOCS=ON   
      
