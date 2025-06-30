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
  mfem::FiniteElementSpace* _fes;
  int _kmax;
  int _dtn_bdr_attr;
  mfem::real_t _radius;
  const mfem::real_t pi = 3.1415926535897932385;

  mfem::SparseMatrix _mat;

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

  void Assemble();

 public:
  Poisson2D(mfem::FiniteElementSpace* fes, int kmax, int dtn_bdr_attr,
            const mfem::IntegrationRule* ir = nullptr);

  int NumberOfCoefficients() const { return 2 * _kmax + 1; }

  void FourierCoefficients(const mfem::Vector& x, mfem::Vector& c) const;

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override {
    Mult(x, y);
  }
};

}  // namespace DtN

}  // namespace mfemElasticity