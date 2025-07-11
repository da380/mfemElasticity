#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>

#include "Legendre.hpp"
#include "mfem.hpp"
#include "mfemElasticity/Legendre.hpp"

namespace mfemElasticity {

namespace DtN {

/*===========================================================
      Base class for DtN operator for Poisson equation
============================================================*/
class Poisson : public mfem::Integrator, public mfem::Operator {
 protected:
  mfem::FiniteElementSpace* _fes;
  int _coeff_dim;
  mfem::Array<int> _bdr_marker;
  mfem::SparseMatrix _mat;

  static constexpr mfem::real_t pi = std::atan(1) * 4;

#ifdef MFEM_USE_MPI
  bool _parallel = false;
  mfem::ParFiniteElementSpace* _pfes;
  MPI_Comm _comm;
#endif

#ifndef MFEM_THREAD_SAFE
  mutable mfem::Vector _c;
  mfem::Vector shape, _x;
  mfem::DenseMatrix elmat;
#endif

  // Element level calculation of sparse matrix. Pure virtual method
  // that is overridden in derived classes.
  virtual void AssembleElementMatrix(const mfem::FiniteElement& fe,
                                     mfem::ElementTransformation& Trans,
                                     mfem::DenseMatrix& elmat) = 0;

  // Check that the mesh is suitable. Can be overridden.
  virtual void CheckMesh() const {}

 public:
  // Serial constructor.
  Poisson(mfem::FiniteElementSpace* fes, int coeff_dim);

#ifdef MFEM_USE_MPI
  // Parallel constructor.
  Poisson(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int coeff_dim);
#endif

  // Set the boundary marker to the default.
  void SetBoundaryMarkerToExternal() {
    auto* mesh = _fes->GetMesh();
    _bdr_marker.SetSize(mesh->bdr_attributes.Max());
    _bdr_marker = 0;
    mesh->MarkExternalBoundaries(_bdr_marker);
  }

  // Set the boundary marker.
  void SetBoundaryMarker(mfem::Array<int>& bdr_marker) {
    _bdr_marker = bdr_marker;
  }

  // Multiplication by the operator.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  // Transposed multiplication by the operator.
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override {
    Mult(x, y);
  }

  // Assemble the sparse matrix associated with the operator.
  void Assemble();

  // Return the associated RAP operator.
  mfem::RAPOperator RAPOperator() const;
};

/*===============================================================
   DtN operator for Poisson's equation on a circular boundary
=================================================================*/
class PoissonCircle : public Poisson {
 private:
  int _kMax;

  void AssembleElementMatrix(const mfem::FiniteElement& fe,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;

  void CheckMesh() const override {
    auto* mesh = _fes->GetMesh();
    assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);
  }

  void SetSizes() {
#ifndef MFEM_THREAD_SAFE
    _x.SetSize(2);
    _c.SetSize(_coeff_dim);
#endif
  }

 public:
  // Serial constructor with default boundary.
  PoissonCircle(mfem::FiniteElementSpace* fes, int kMax)
      : Poisson(fes, 2 * kMax), _kMax{kMax} {
    SetBoundaryMarkerToExternal();
    SetSizes();
  }

  // Serial constructor with specified boundary.
  PoissonCircle(mfem::FiniteElementSpace* fes, int kMax,
                mfem::Array<int>& bdr_marker)
      : Poisson(fes, 2 * kMax), _kMax{kMax} {
    SetBoundaryMarker(bdr_marker);
    SetSizes();
  }

#ifdef MFEM_USE_MPI
  // Parallel constructor with default boundary.
  PoissonCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kMax)
      : Poisson(comm, fes, 2 * kMax), _kMax{kMax} {
    SetBoundaryMarkerToExternal();
    SetSizes();
  }

  // Parallel constructor with specified boundary.
  PoissonCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kMax,
                mfem::Array<int>& bdr_marker)
      : Poisson(comm, fes, 2 * kMax), _kMax{kMax} {
    SetBoundaryMarker(bdr_marker);
    SetSizes();
  }
#endif
};

/*===============================================================
    DtN operator for Poisson's equation on a spherical boundary
=================================================================*/
class PoissonSphere : public Poisson, private LegendreHelper {
 private:
  int _lMax;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector _sin, _cos, _p, _pm1;
#endif

  void AssembleElementMatrix(const mfem::FiniteElement& fe,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;

  void CheckMesh() const override {
    auto* mesh = _fes->GetMesh();
    assert(mesh->Dimension() == 3 && mesh->SpaceDimension() == 3);
  }

  void SetSizes() {
#ifndef MFEM_THREAD_SAFE
    _x.SetSize(3);
    _c.SetSize(_coeff_dim);

    _sin.SetSize(_lMax + 1);
    _cos.SetSize(_lMax + 1);

    _p.SetSize(_lMax + 1);
    _pm1.SetSize(_lMax + 1);
#endif
  }

 public:
  // Serial constructor with default boundary.
  PoissonSphere(mfem::FiniteElementSpace* fes, int lMax)
      : Poisson(fes, (lMax + 1) * (lMax + 1)), _lMax{lMax} {
    SetBoundaryMarkerToExternal();
    SetSizes();
    SetSquareRoots(_lMax);
  }

  // Serial constructor with specified boundary.
  PoissonSphere(mfem::FiniteElementSpace* fes, int lMax,
                mfem::Array<int>& bdr_marker)
      : Poisson(fes, (lMax + 1) * (lMax + 1)), _lMax{lMax} {
    SetBoundaryMarker(bdr_marker);
    SetSizes();
    SetSquareRoots(_lMax);
  }

#ifdef MFEM_USE_MPI
  // Parallel constructor with default boundary.
  PoissonSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int lMax)
      : Poisson(comm, fes, (lMax + 1) * (lMax + 1)), _lMax{lMax} {
    SetBoundaryMarkerToExternal();
    SetSizes();
    SetSquareRoots(_lMax);
  }

  // Parallel constructor with specified boundary.
  PoissonSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int lMax,
                mfem::Array<int>& bdr_marker)
      : Poisson(comm, fes, (lMax + 1) * (lMax + 1)), _lMax{lMax} {
    SetBoundaryMarker(bdr_marker);
    SetSizes();
    SetSquareRoots(_lMax);
  }
#endif
};

}  // namespace DtN

}  // namespace mfemElasticity