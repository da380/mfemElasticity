#pragma once
#include <cassert>
#include <cmath>

#include "Legendre.hpp"
#include "mfem.hpp"
#include "mfemElasticity/Legendre.hpp"
#include "mfemElasticity/utils.hpp"

namespace mfemElasticity {

namespace DtN {

/*----------------------------------------------------------
      Base class for DtN operator for Poisson equation
------------------------------------------------------------*/
class Poisson : public mfem::Integrator, public mfem::Operator {
 protected:
  mfem::FiniteElementSpace* _fes;
  int _coeff_dim;
  mfem::Array<int> _bdr_marker;
  mfem::SparseMatrix _mat;

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
  // Serial constructors.
  Poisson(mfem::FiniteElementSpace* fes, int coeff_dim,
          const mfem::Array<int>& bdr_marker);

  Poisson(mfem::FiniteElementSpace* fes, int coeff_dim,
          mfem::Array<int>&& bdr_marker)
      : Poisson(fes, coeff_dim, bdr_marker) {}

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  Poisson(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int coeff_dim,
          const mfem::Array<int>& bdr_marker);

  Poisson(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int coeff_dim,
          mfem::Array<int>&& bdr_marker)
      : Poisson(comm, fes, coeff_dim, bdr_marker) {}

  Poisson(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int coeff_dim)
      : Poisson(comm, fes, coeff_dim, ExternalBoundaryMarker(fes->GetMesh())) {}
#endif

  // Multiplication by the operator.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  // Transposed multiplication by the operator.
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override {
    Mult(x, y);
  }

  // Assemble the sparse matrix associated with the operator.
  void Assemble();

  // Return the associated RAP operator.
  mfem::RAPOperator RAP() const;
};

/*===============================================================
   DtN operator for Poisson's equation on a circular boundary
=================================================================*/
class PoissonCircle : public Poisson {
 private:
  int _kMax;

  static constexpr mfem::real_t pi = std::atan(1) * 4;

  void AssembleElementMatrix(const mfem::FiniteElement& fe,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;

  void CheckMesh() const override {
    auto* mesh = _fes->GetMesh();
    assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);
  }

 public:
  // Serial constructors.
  PoissonCircle(mfem::FiniteElementSpace* fes, int kMax,
                const mfem::Array<int>& bdr_marker)
      : Poisson(fes, 2 * kMax, bdr_marker), _kMax{kMax} {}

  PoissonCircle(mfem::FiniteElementSpace* fes, int kMax,
                mfem::Array<int>&& bdr_marker)
      : PoissonCircle(fes, kMax, bdr_marker) {}

  PoissonCircle(mfem::FiniteElementSpace* fes, int kMax)
      : Poisson(fes, 2 * kMax, ExternalBoundaryMarker(fes->GetMesh())),
        _kMax{kMax} {}

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  PoissonCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kMax,
                const mfem::Array<int>& bdr_marker)
      : Poisson(comm, fes, 2 * kMax, bdr_marker), _kMax{kMax} {}

  PoissonCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kMax,
                mfem::Array<int>&& bdr_marker)
      : PoissonCircle(comm, fes, kMax, bdr_marker) {}

  PoissonCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kMax)
      : Poisson(comm, fes, 2 * kMax, ExternalBoundaryMarker(fes->GetMesh())),
        _kMax{kMax} {}

#endif
};

/*---------------------------------------------------------------
    DtN operator for Poisson's equation on a spherical boundary
----------------------------------------------------------------*/
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

  void SetUp() {
#ifndef MFEM_THREAD_SAFE
    _sin.SetSize(_lMax + 1);
    _cos.SetSize(_lMax + 1);
    _p.SetSize(_lMax + 1);
    _pm1.SetSize(_lMax + 1);
#endif
    SetSquareRoots(_lMax);
  }

 public:
  // Serial constructors.
  PoissonSphere(mfem::FiniteElementSpace* fes, int lMax,
                const mfem::Array<int>& bdr_marker)
      : Poisson(fes, (lMax + 1) * (lMax + 1), bdr_marker), _lMax{lMax} {
    SetUp();
  }

  PoissonSphere(mfem::FiniteElementSpace* fes, int lMax,
                mfem::Array<int>&& bdr_marker)
      : PoissonSphere(fes, lMax, bdr_marker) {}

  PoissonSphere(mfem::FiniteElementSpace* fes, int lMax)
      : Poisson(fes, (lMax + 1) * (lMax + 1),
                ExternalBoundaryMarker(fes->GetMesh())),
        _lMax{lMax} {
    SetUp();
  }

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  PoissonSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int lMax,
                const mfem::Array<int>& bdr_marker)
      : Poisson(comm, fes, (lMax + 1) * (lMax + 1), bdr_marker), _lMax{lMax} {
    SetUp();
  }

  PoissonSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int lMax,
                mfem::Array<int>&& bdr_marker)
      : PoissonSphere(comm, fes, (lMax + 1) * (lMax + 1), bdr_marker) {}

  PoissonSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int lMax)
      : Poisson(comm, fes, (lMax + 1) * (lMax + 1),
                ExternalBoundaryMarker(fes->GetMesh())),
        _lMax{lMax} {
    SetUp();
  }

#endif
};

}  // namespace DtN

}  // namespace mfemElasticity