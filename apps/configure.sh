#!/bin/bash

cmake -S . -B build \
      -DmfemElasticity_DIR=../parallel_build
      