#pragma once

/*
  Helpers shared by the serial SubMesh gtests (TestSubMeshDofInjection,
  TestSubMeshMixedBilinearForm).
*/

#include "TestCommon.hpp"

// A Cartesian mesh (4x4 in 2D, 3x3x3 in 3D) split into attribute 2 for
// x < 0.5 and attribute 1 for x > 0.5.
inline Mesh MakeTwoAttributeMesh(int dim, int elementType) {
  auto mesh =
      dim == 2
          ? Mesh::MakeCartesian2D(
                4, 4,
                elementType == 0 ? Element::TRIANGLE : Element::QUADRILATERAL)
          : Mesh::MakeCartesian3D(
                3, 3, 3,
                elementType == 0 ? Element::TETRAHEDRON : Element::HEXAHEDRON);
  for (auto i = 0; i < mesh.GetNE(); i++) {
    Vector c(dim);
    mesh.GetElementCenter(i, c);
    mesh.SetAttribute(i, c[0] < 0.5 ? 2 : 1);
  }
  mesh.SetAttributes();
  return mesh;
}

// The boundary attribute of the boundary elements lying in the plane x = 0.
inline int BdrAttributeAtXZero(Mesh& mesh) {
  const auto dim = mesh.Dimension();
  for (auto i = 0; i < mesh.GetNBE(); i++) {
    auto* tr = mesh.GetBdrElementTransformation(i);
    Vector c(dim);
    tr->Transform(Geometries.GetCenter(mesh.GetBdrElementGeometry(i)), c);
    if (std::abs(c[0]) < 1e-12) {
      return mesh.GetBdrAttribute(i);
    }
  }
  return -1;
}

inline double TestFunction(const Vector& x) {
  auto v = 1.0;
  for (auto i = 0; i < x.Size(); i++) {
    v *= std::sin(2.0 * x[i] + 0.5 * i) + 0.25 * x[i] * x[i];
  }
  return v;
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
