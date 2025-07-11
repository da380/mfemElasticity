#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>

#include "mfem.hpp"

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

 public:
  // Serial constructor.
  Poisson(mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
          int coeff_dim);

#ifdef MFEM_USE_MPI
  // Parallel constructor.
  Poisson(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
          mfem::ParFiniteElementSpace* te_fes, int coeff_dim);
#endif

  // Set the boundary marker to the default.
  void SetBoundaryMarkerToExternal() {
    auto* mesh = _tr_fes->GetMesh();
    _bdr_marker.SetSize(mesh->bdr_attributes.Max());
    _bdr_marker = 0;
    mesh->MarkExternalBoundaries(_bdr_marker);
  }

  // Set the domain marker to the default.
  void SetDomainMarkerToAll() {
    auto* mesh = _tr_fes->GetMesh();
    _dom_marker.SetSize(mesh->attributes.Max());
    _dom_marker = 1;
  }

  // Set the boundary marker.
  void SetBoundaryMarker(mfem::Array<int>& bdr_marker) {
    _bdr_marker = bdr_marker;
  }

  // Set the domain marker.
  void SetDomainMarker(mfem::Array<int>& dom_marker) {
    _dom_marker = dom_marker;
  }

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
  // Serial constructor with default boundary and domain.
  PoissonCircle(mfem::FiniteElementSpace* tr_fes,
                mfem::FiniteElementSpace* te_fes, int kMax)
      : Poisson(tr_fes, te_fes, 2 * kMax), _kMax{kMax} {
    SetBoundaryMarkerToExternal();
    SetDomainMarkerToAll();
    _bdr_radius = ExternalBoundaryRadius();
  }

  // Serial constructor with specified boundary and domain.
  PoissonCircle(mfem::FiniteElementSpace* tr_fes,
                mfem::FiniteElementSpace* te_fes, int kMax,
                mfem::Array<int>& bdr_marker, mfem::Array<int>& dom_marker)
      : Poisson(tr_fes, te_fes, 2 * kMax), _kMax{kMax} {
    SetBoundaryMarker(bdr_marker);
    SetDomainMarker(dom_marker);
    _bdr_radius = ExternalBoundaryRadius();
  }

#ifdef MFEM_USE_MPI
  // Parallel constructor with default boundary and domain.
  PoissonCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                mfem::ParFiniteElementSpace* te_fes, int kMax)
      : Poisson(comm, tr_fes, te_fes, 2 * kMax), _kMax{kMax} {
    SetBoundaryMarkerToExternal();
    SetDomainMarkerToAll();
    _bdr_radius = ParallelExternalBoundaryRadius();
  }

  // Parallel constructor with specified boundary and domain.
  PoissonCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                mfem::ParFiniteElementSpace* te_fes, int kMax,
                mfem::Array<int>& bdr_marker, mfem::Array<int>& dom_marker)
      : Poisson(comm, tr_fes, te_fes, 2 * kMax), _kMax{kMax} {
    SetBoundaryMarker(bdr_marker);
    SetDomainMarker(dom_marker);
    _bdr_radius = ParallelExternalBoundaryRadius();
  }
#endif
};

}  // namespace Multipole
}  // namespace mfemElasticity
