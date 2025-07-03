#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>

#include "mfem.hpp"

namespace mfemElasticity {

namespace DtN {

class Poisson2D : public mfem::Integrator, public mfem::Operator {
 private:
  const mfem::real_t pi = 3.141592653589793238462643383279;
  mfem::FiniteElementSpace* _fes;
  int _kmax;
  int _dtn_bdr_attr;
  mfem::SparseMatrix _mat;

#ifdef MFEM_USE_MPI
  bool _parallel = false;
  mfem::ParFiniteElementSpace* _pfes;
  MPI_Comm _comm;
#endif

#ifndef MFEM_THREAD_SAFE
  mutable mfem::Vector _c;
  mfem::Vector shape;
  mfem::DenseMatrix elmat;
#endif

  // Check the mesh has the correct properties.
  void CheckMesh() const;

  void GetDtNBoundaryAttribute();

  // Get the radius of the DtN boundary, and verify that
  // it is a circle.
  void GetRadius();

  void AssembleElementMatrix(const mfem::FiniteElement& fe,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat);

 public:
  Poisson2D(mfem::FiniteElementSpace* fes, int kmax);

#ifdef MFEM_USE_MPI
  Poisson2D(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kmax);
#endif

  void Assemble();

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override {
    Mult(x, y);
  }

  /*
#ifdef MFEM_USE_MPI
  auto RAP() const {
    auto* R = _pfes->GetRestrictionMatrix();
    auto* P = _pfes->GetProlongationMatrix();
    auto RP = mfem::ProductOperator(R, P, true, true);
    return RP;
  }
#endif
*/
};

}  // namespace DtN

}  // namespace mfemElasticity