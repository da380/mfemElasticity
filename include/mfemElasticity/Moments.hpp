#pragma once

#include <cassert>
#include <cmath>

#include "mfem.hpp"
#include "mfemElasticity/mesh_utils.hpp"

namespace mfemElasticity {

/*
mfem::Operator acting on a density field to return its moments up to
degree two.
*/
class MomentsOperator : public mfem::Operator, public mfem::Integrator {
 private:
  int _moments_dim;
  mfem::FiniteElementSpace* _fes;
  mfem::Array<int> _dom_marker;
  mfem::SparseMatrix _mat;

#ifdef MFEM_USE_MPI
  bool _parallel = false;
  mfem::ParFiniteElementSpace* _pfes;
  MPI_Comm _comm;
#endif

#ifndef MFEM_THREAD_SAFE
  mfem::Vector _c, _x, shape;
#endif

  static int RowDim(mfem::Mesh* mesh);

  // Element level calculation of sparse matrix. Pure virtual method
  // that is overridden in derived classes.
  void AssembleElementMatrix(const mfem::FiniteElement& fe,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat);

 public:
  // Serial constructors.
  MomentsOperator(mfem::FiniteElementSpace* fes,
                  const mfem::Array<int>& dom_marker);

  MomentsOperator(mfem::FiniteElementSpace* fes, mfem::Array<int>&& dom_marker)
      : MomentsOperator(fes, dom_marker) {}

  MomentsOperator(mfem::FiniteElementSpace* fes)
      : MomentsOperator(fes, AllDomainsMarker(fes->GetMesh())) {}

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  MomentsOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* fes,
                  const mfem::Array<int>& dom_marker);

  MomentsOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* fes,
                  mfem::Array<int>&& dom_marker)
      : MomentsOperator(comm, fes, dom_marker) {}

  MomentsOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* fes)
      : MomentsOperator(comm, fes, AllDomainsMarker(fes->GetMesh())) {}

#endif

  // Multiplication by the operator.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  // Transposed multiplication by the operator.
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

  // Assemble the sparse matrix associated with the operator.
  void Assemble();

  // Return the associated RAP operator.
  mfem::RAPOperator RAP() const;

  // Return the centroid given the moments vector.
  void Centroid(const mfem::Vector& moments_vector,
                mfem::Vector& centroid) const;

  mfem::Vector Centroid(const mfem::Vector& moments_vector) const;

  // Return the inertia tensor.
  void InertiaTensor(const mfem::Vector& moments_vector,
                     mfem::DenseMatrix& inertia_tensor) const;

  mfem::DenseMatrix InertiaTensor(const mfem::Vector& moments_vector) const;
};

}  // namespace mfemElasticity