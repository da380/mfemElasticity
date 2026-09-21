#pragma once

#include <gtest/gtest.h>

#include <iostream>
#include <limits>
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

// Entrywise max |A - B| for finalized sparse matrices of equal size.
inline double MaxDiff(const SparseMatrix& A, const SparseMatrix& B) {
  EXPECT_EQ(A.Height(), B.Height());
  EXPECT_EQ(A.Width(), B.Width());
  if (A.Height() != B.Height() || A.Width() != B.Width()) {
    return std::numeric_limits<double>::infinity();
  }
  std::unique_ptr<SparseMatrix> D(mfem::Add(1.0, A, -1.0, B));
  return D->MaxNorm();
}
