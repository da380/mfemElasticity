#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>

#include "mfem.hpp"

namespace mfemElasticity {

namespace Multipole {

/*
Class for the Dirichlet-to-Neumann mapping for the Poisson equation
in 2D space.
*/
class Poisson2D : public mfem::Integrator, public mfem::Operator {
 private:
  mfem::FiniteElementSpace* _tr_fes;
  mfem::FiniteElementSpace* _te_fes;
  int _kmax;
  mfem::real_t _bdr_radius;
  mfem::Array<int> _dom_marker;
  mfem::Array<int> _bdr_marker;
  mfem::SparseMatrix _lmat;
  mfem::SparseMatrix _rmat;

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

  // Get the radius of the external boundary. Returns 0 if
  // the boundary is not present.
  mfem::real_t ExternalBoundaryRadius() const;

#ifdef MFEM_USE_MPI

  // Returns the radius of the external boundary for parallel
  // calculations.
  mfem::real_t ParallelExternalBoundaryRadius() const;

#endif

  // Element level assembly for right matrix.
  void AssembleRightElementMatrix(const mfem::FiniteElement& fe,
                                  mfem::ElementTransformation& Trans,
                                  mfem::DenseMatrix& elmat);

  // Element level assembly for left matrix.
  void AssembleLeftElementMatrix(const mfem::FiniteElement& fe,
                                 mfem::ElementTransformation& Trans,
                                 mfem::DenseMatrix& elmat);

 public:
  /*
  Construct the operator for a serial calculation.
  */
  Poisson2D(mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
            int kmax, int dom_marker, int bdr_marker);

  /*
  Construct the operator for a serial calculation using default domain and
  boundary.
  */
  Poisson2D(mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
            int kmax);

#ifdef MFEM_USE_MPI

  /*
  Construct the operator for a serial calculation.
  */
  Poisson2D(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
            mfem::ParFiniteElementSpace* te_fes, int kmax, int dom_marker,
            int bdr_marker);

  /*
  Construct the operator for a parallel calculation using default domain and
  boundary.
  */
  Poisson2D(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
            mfem::ParFiniteElementSpace* te_fes, int kmax);

#endif

  // Multiplication method for the operator.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  // Transposed multiplication.
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

  // Assemble the sparse matrices associated with the operator.
  void Assemble();

  // Return the associated RAP operator that occurs within
  // Linearsystems involving true degrees of freedom. For serial
  // calculations the operator and its RAP operator coincide.
  mfem::RAPOperator RAP() const;
};

}  // namespace Multipole

}  // namespace mfemElasticity