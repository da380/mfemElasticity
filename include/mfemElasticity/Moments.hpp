#pragma once

#include <cassert>
#include <cmath>

#include "mfem.hpp"
#include "mfemElasticity/utils.hpp"

namespace mfemElasticity {

/*
mfem::Operator acting on a density field to return the mass, centroid, and
Moments of inertia.
*/
class MomentsOperator : public mfem::Operator, public mfem::Integrator {
 private:
  mfem::FiniteElementSpace* _fes;
  mfem::Array<int> _dom_marker;
  mfem::SparseMatrix _mat;

#ifdef MFEM_USE_MPI
  bool _parallel = false;
  mfem::ParFiniteElementSpace* _pfes;
  MPI_Comm _comm;
#endif

  static int RowDim(mfem::Mesh* mesh);

  int RowDim() const { return RowDim(_fes->GetMesh()); }

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
};

}  // namespace mfemElasticity