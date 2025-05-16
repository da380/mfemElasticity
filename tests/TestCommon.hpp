#pragma once

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <tuple>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace mfem;
using namespace mfemElasticity;

using DimOrderTypeTuple = std::tuple<int, int, int>;

Mesh MakeMesh(int dim, int elementType) {
  if (dim == 1) {
    return Mesh::MakeCartesian1D(20);
  } else if (dim == 2) {
    return Mesh::MakeCartesian2D(
        20, 20, elementType == 0 ? Element::TRIANGLE : Element::QUADRILATERAL);
  } else {
    return Mesh::MakeCartesian3D(
        20, 20, 20,
        elementType == 0 ? Element::TETRAHEDRON : Element::HEXAHEDRON);
  }
}

Vector RandomVector(int dim) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> distrib(0, 1);

  auto v = Vector(dim);
  for (auto j = 0; j < dim; j++) {
    v(j) = distrib(gen);
  }
  return v;
}

DenseMatrix RandomMatrix(int dim) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> distrib(0, 1);

  auto A = DenseMatrix(dim);
  for (auto j = 0; j < dim; j++) {
    for (auto i = 0; i < dim; i++) {
      A(i, j) = distrib(gen);
    }
  }
  return A;
}
