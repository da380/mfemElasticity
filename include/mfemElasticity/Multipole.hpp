#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>

#include "mfem.hpp"
#include "mfemElasticity/Legendre.hpp"

namespace mfemElasticity {

namespace Multipole {

/*----------------------------------------------------------
    Base class for Multipole operator for Poisson equation
------------------------------------------------------------*/
class Poisson : public mfem::Integrator, public mfem::Operator {
 protected:
  mfem::FiniteElementSpace* _tr_fes;
  mfem::FiniteElementSpace* _te_fes;
  int _coeff_dim;
  mfem::real_t _bdr_radius;
  mfem::Array<int> _bdr_marker;
  mfem::Array<int> _dom_marker;
  mfem::SparseMatrix _lmat;
  mfem::SparseMatrix _rmat;

  static constexpr mfem::real_t pi = std::atan(1) * 4;

#ifdef MFEM_USE_MPI
  bool _parallel = false;
  mfem::ParFiniteElementSpace* _tr_pfes;
  mfem::ParFiniteElementSpace* _te_pfes;
  MPI_Comm _comm;
#endif

#ifndef MFEM_THREAD_SAFE
  mutable mfem::Vector _c;
  mfem::Vector shape, _x;
  mfem::DenseMatrix elmat;
#endif

  // Element level calculation of  leftsparse matrix. Pure virtual method
  // that is overridden in derived classes.
  virtual void AssembleLeftElementMatrix(const mfem::FiniteElement& fe,
                                         mfem::ElementTransformation& Trans,
                                         mfem::DenseMatrix& elmat) = 0;

  virtual void AssembleRightElementMatrix(const mfem::FiniteElement& fe,
                                          mfem::ElementTransformation& Trans,
                                          mfem::DenseMatrix& elmat) = 0;

  // Check that the mesh is suitable. Can be overridden.
  virtual void CheckMesh() const {}

  static mfem::Array<int> ExternalBoundaryMarker(mfem::Mesh* mesh);

  static mfem::Array<int> DomainMarker(mfem::Mesh* mesh);

 public:
  // Serial constructors.
  Poisson(mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
          int coeff_dim, const mfem::Array<int>& dom_marker,
          const mfem::Array<int>& bdr_marker);

  Poisson(mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
          int coeff_dim, mfem::Array<int>&& dom_marker,
          const mfem::Array<int>& bdr_marker)
      : Poisson(tr_fes, te_fes, coeff_dim, dom_marker, bdr_marker) {}

  Poisson(mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
          int coeff_dim, const mfem::Array<int>& dom_marker,
          mfem::Array<int>&& bdr_marker)
      : Poisson(tr_fes, te_fes, coeff_dim, dom_marker, bdr_marker) {}

  Poisson(mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
          int coeff_dim, mfem::Array<int>&& dom_marker,
          mfem::Array<int>&& bdr_marker)
      : Poisson(tr_fes, te_fes, coeff_dim, dom_marker, bdr_marker) {}

#ifdef MFEM_USE_MPI
  // Parallel constructors
  Poisson(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
          mfem::ParFiniteElementSpace* te_fes, int coeff_dim,
          const mfem::Array<int>& dom_marker,
          const mfem::Array<int>& bdr_marker);

  Poisson(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
          mfem::ParFiniteElementSpace* te_fes, int coeff_dim,
          mfem::Array<int>&& dom_marker, const mfem::Array<int>& bdr_marker)
      : Poisson(comm, tr_fes, te_fes, coeff_dim, dom_marker, bdr_marker) {}

  Poisson(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
          mfem::ParFiniteElementSpace* te_fes, int coeff_dim,
          mfem::Array<int>&& dom_marker, mfem::Array<int>&& bdr_marker)
      : Poisson(comm, tr_fes, te_fes, coeff_dim, dom_marker, bdr_marker) {}

  Poisson(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
          mfem::ParFiniteElementSpace* te_fes, int coeff_dim,
          const mfem::Array<int>& dom_marker, mfem::Array<int>&& bdr_marker)
      : Poisson(comm, tr_fes, te_fes, coeff_dim, dom_marker, bdr_marker) {}
#endif

  // Get the radius of the external boundary. Returns 0 if
  // the boundary is not present.
  mfem::real_t ExternalBoundaryRadius() const;

#ifdef MFEM_USE_MPI

  // Returns the radius of the external boundary for parallel
  // calculations.
  mfem::real_t ParallelExternalBoundaryRadius() const;

#endif

  // Multiplication by the operator.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  // Transposed multiplication by the operator.
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

  // Assemble the sparse matrix associated with the operator.
  void Assemble();

  // Return the associated RAP operator.
  mfem::RAPOperator RAPOperator() const;
};

/*----------------------------------------------------------
     Class for Multipole operator for Poisson 2D equation
------------------------------------------------------------*/
class PoissonCircle : public Poisson {
 private:
  int _kMax;

  static constexpr mfem::real_t pi = std::atan(1) * 4;

  void AssembleLeftElementMatrix(const mfem::FiniteElement& fe,
                                 mfem::ElementTransformation& Trans,
                                 mfem::DenseMatrix& elmat) override;

  void AssembleRightElementMatrix(const mfem::FiniteElement& fe,
                                  mfem::ElementTransformation& Trans,
                                  mfem::DenseMatrix& elmat) override;

  void CheckMesh() const override {
    assert(_te_fes->GetMesh() == _tr_fes->GetMesh());
    assert(_tr_fes->GetMesh()->Dimension() == 2 &&
           _tr_fes->GetMesh()->SpaceDimension() == 2);
  }

 public:
  // Serial constructors
  PoissonCircle(mfem::FiniteElementSpace* tr_fes,
                mfem::FiniteElementSpace* te_fes, int kMax,
                const mfem::Array<int>& dom_marker,
                const mfem::Array<int>& bdr_marker)
      : Poisson(tr_fes, te_fes, 2 * kMax, dom_marker, bdr_marker),
        _kMax{kMax} {}

  PoissonCircle(mfem::FiniteElementSpace* tr_fes,
                mfem::FiniteElementSpace* te_fes, int kMax,
                mfem::Array<int>&& dom_marker,
                const mfem::Array<int>& bdr_marker)
      : Poisson(tr_fes, te_fes, 2 * kMax, dom_marker, bdr_marker),
        _kMax{kMax} {}

  PoissonCircle(mfem::FiniteElementSpace* tr_fes,
                mfem::FiniteElementSpace* te_fes, int kMax,
                const mfem::Array<int>& dom_marker,
                mfem::Array<int>&& bdr_marker)
      : Poisson(tr_fes, te_fes, 2 * kMax, dom_marker, bdr_marker),
        _kMax{kMax} {}

  PoissonCircle(mfem::FiniteElementSpace* tr_fes,
                mfem::FiniteElementSpace* te_fes, int kMax,
                mfem::Array<int>&& dom_marker, mfem::Array<int>&& bdr_marker)
      : Poisson(tr_fes, te_fes, 2 * kMax, dom_marker, bdr_marker),
        _kMax{kMax} {}

  PoissonCircle(mfem::FiniteElementSpace* tr_fes,
                mfem::FiniteElementSpace* te_fes, int kMax)
      : Poisson(tr_fes, te_fes, 2 * kMax, DomainMarker(tr_fes->GetMesh()),
                ExternalBoundaryMarker(tr_fes->GetMesh())),
        _kMax{kMax} {}

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  PoissonCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                mfem::ParFiniteElementSpace* te_fes, int kMax,
                const mfem::Array<int>& dom_marker,
                const mfem::Array<int>& bdr_marker)
      : Poisson(comm, tr_fes, te_fes, 2 * kMax, dom_marker, bdr_marker),
        _kMax{kMax} {}

  PoissonCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                mfem::ParFiniteElementSpace* te_fes, int kMax,
                mfem::Array<int>&& dom_marker,
                const mfem::Array<int>& bdr_marker)
      : Poisson(comm, tr_fes, te_fes, 2 * kMax, dom_marker, bdr_marker),
        _kMax{kMax} {}

  PoissonCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                mfem::ParFiniteElementSpace* te_fes, int kMax,
                const mfem::Array<int>& dom_marker,
                mfem::Array<int>&& bdr_marker)
      : Poisson(comm, tr_fes, te_fes, 2 * kMax, dom_marker, bdr_marker),
        _kMax{kMax} {}

  PoissonCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                mfem::ParFiniteElementSpace* te_fes, int kMax,
                mfem::Array<int>&& dom_marker, mfem::Array<int>&& bdr_marker)
      : Poisson(comm, tr_fes, te_fes, 2 * kMax, dom_marker, bdr_marker),
        _kMax{kMax} {}

  PoissonCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                mfem::ParFiniteElementSpace* te_fes, int kMax)
      : Poisson(comm, tr_fes, te_fes, 2 * kMax, DomainMarker(tr_fes->GetMesh()),
                ExternalBoundaryMarker(tr_fes->GetMesh())),
        _kMax{kMax} {}

#endif
};

/*----------------------------------------------------------
     Class for Multipole operator for Poisson 3D equation
------------------------------------------------------------*/
class PoissonSphere : public Poisson, private LegendreHelper {
 private:
  int _lMax;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector _sin, _cos, _p, _pm1;
#endif

  void AssembleLeftElementMatrix(const mfem::FiniteElement& fe,
                                 mfem::ElementTransformation& Trans,
                                 mfem::DenseMatrix& elmat) override;

  void AssembleRightElementMatrix(const mfem::FiniteElement& fe,
                                  mfem::ElementTransformation& Trans,
                                  mfem::DenseMatrix& elmat) override;

  void CheckMesh() const override {
    assert(_te_fes->GetMesh() == _tr_fes->GetMesh());
    assert(_tr_fes->GetMesh()->Dimension() == 3 &&
           _tr_fes->GetMesh()->SpaceDimension() == 3);
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
  // Serial constructors
  PoissonSphere(mfem::FiniteElementSpace* tr_fes,
                mfem::FiniteElementSpace* te_fes, int lMax,
                const mfem::Array<int>& dom_marker,
                const mfem::Array<int>& bdr_marker)
      : Poisson(tr_fes, te_fes, (lMax + 1) * (lMax + 1), dom_marker,
                bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonSphere(mfem::FiniteElementSpace* tr_fes,
                mfem::FiniteElementSpace* te_fes, int lMax,
                mfem::Array<int>&& dom_marker,
                const mfem::Array<int>& bdr_marker)
      : Poisson(tr_fes, te_fes, (lMax + 1) * (lMax + 1), dom_marker,
                bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonSphere(mfem::FiniteElementSpace* tr_fes,
                mfem::FiniteElementSpace* te_fes, int lMax,
                const mfem::Array<int>& dom_marker,
                mfem::Array<int>&& bdr_marker)
      : Poisson(tr_fes, te_fes, (lMax + 1) * (lMax + 1), dom_marker,
                bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonSphere(mfem::FiniteElementSpace* tr_fes,
                mfem::FiniteElementSpace* te_fes, int lMax,
                mfem::Array<int>&& dom_marker, mfem::Array<int>&& bdr_marker)
      : Poisson(tr_fes, te_fes, (lMax + 1) * (lMax + 1), dom_marker,
                bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonSphere(mfem::FiniteElementSpace* tr_fes,
                mfem::FiniteElementSpace* te_fes, int lMax)
      : Poisson(tr_fes, te_fes, (lMax + 1) * (lMax + 1),
                DomainMarker(tr_fes->GetMesh()),
                ExternalBoundaryMarker(tr_fes->GetMesh())),
        _lMax{lMax} {
    SetUp();
  }

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  PoissonSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                mfem::ParFiniteElementSpace* te_fes, int lMax,
                const mfem::Array<int>& dom_marker,
                const mfem::Array<int>& bdr_marker)
      : Poisson(comm, tr_fes, te_fes, (lMax + 1) * (lMax + 1), dom_marker,
                bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                mfem::ParFiniteElementSpace* te_fes, int lMax,
                mfem::Array<int>&& dom_marker,
                const mfem::Array<int>& bdr_marker)
      : Poisson(comm, tr_fes, te_fes, (lMax + 1) * (lMax + 1), dom_marker,
                bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                mfem::ParFiniteElementSpace* te_fes, int lMax,
                const mfem::Array<int>& dom_marker,
                mfem::Array<int>&& bdr_marker)
      : Poisson(comm, tr_fes, te_fes, (lMax + 1) * (lMax + 1), dom_marker,
                bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                mfem::ParFiniteElementSpace* te_fes, int lMax,
                mfem::Array<int>&& dom_marker, mfem::Array<int>&& bdr_marker)
      : Poisson(comm, tr_fes, te_fes, (lMax + 1) * (lMax + 1), dom_marker,
                bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                mfem::ParFiniteElementSpace* te_fes, int lMax)
      : Poisson(comm, tr_fes, te_fes, (lMax + 1) * (lMax + 1),
                DomainMarker(tr_fes->GetMesh()),
                ExternalBoundaryMarker(tr_fes->GetMesh())),
        _lMax{lMax} {
    SetUp();
  }

#endif
};

}  // namespace Multipole
}  // namespace mfemElasticity
