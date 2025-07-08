#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>

#include "mfem.hpp"

namespace mfemElasticity {

namespace DtN {

/*
Class for the Dirichlet-to-Neumann mapping for the Poisson equation
in 2D space.
*/
class Poisson2D : public mfem::Integrator, public mfem::Operator {
 private:
  mfem::FiniteElementSpace* _fes;
  int _kmax;
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

  // Element level calculation of sparse matrix.
  void AssembleElementMatrix(const mfem::FiniteElement& fe,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat);

 public:
  /*
  Construct the operator for a serial calculation.

    fes          -- The finite element space.
    kmax         -- Order for the Fourier exapansion.
    bdr_marker   -- Marker for the boundary on which the
                    mapping is applied.

    Note that the operator is not ready for use following construction.
    Its Assemble() method must be called to assemble the neccesary
    sparse matrix.

    It is assumed that the boundary to which the DtN mapping is evaluated
    is circular with centre at the origin.
  */
  Poisson2D(mfem::FiniteElementSpace* fes, int kmax,
            mfem::Array<int>& bdr_marker);

  /*
  Construct the operator for a serial calculation using the default
  choice of boundary.

    fes          -- The finite element space.
    kmax         -- Order for the Fourier exapansion.

    Note that the operator is not ready for use following construction.
    Its Assemble() method must be called to assemble the neccesary
    sparse matrix.

    It is assumed that the boundary to which the DtN mapping is evaluated
    is circular with centre at the origin.
  */
  Poisson2D(mfem::FiniteElementSpace* fes, int kmax);

#ifdef MFEM_USE_MPI
  /*
  Construct the operator for a parallel calculation.

    comm         -- The MPI communicator.
    fes          -- The parallel finite element space.
    kmax         -- Order for the Fourier exapansion.
    bdr_marker   -- Marker for the boundary on which the
                    mapping is applied.


    Note that the operator is not ready for use following construction.
    Its Assemble() method must be called to assemble the neccesary
    sparse matrix.

    It is assumed that the boundary to which the DtN mapping is evaluated
    is circular with centre at the origin.
  */
  Poisson2D(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kmax,
            mfem::Array<int>& bdr_marker);

  /*
Construct the operator for a parallel calculation using the default
choice of boundary.

  comm         -- The MPI communicator.
  fes          -- The parallel finite element space.
  kmax         -- Order for the Fourier exapansion.

  Note that the operator is not ready for use following construction.
  Its Assemble() method must be called to assemble the neccesary
  sparse matrix.

  It is assumed that the boundary to which the DtN mapping is evaluated
  is circular with centre at the origin.
*/
  Poisson2D(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kmax);
#endif

  // Multiplication method for the operator.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  // Transposed multiplication, which is just multiplication because the
  // operator is self-adjoint.
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override {
    Mult(x, y);
  }

  // Assemble the sparse matrix associated with the operator.
  void Assemble();

  // Return the associated RAP operator that occurs within
  // linear systems involving true degrees of freedom.
  mfem::RAPOperator RAP() const;
};

/*
Class for the Dirichlet-to-Neumann mapping for the Poisson equation
in 2D space.
*/
class Poisson3D : public mfem::Integrator, public mfem::Operator {
 private:
  mfem::FiniteElementSpace* _fes;
  int _lmax;
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

  // Element level calculation of sparse matrix.
  void AssembleElementMatrix(const mfem::FiniteElement& fe,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat);

 public:
  /*
Construct the operator for a serial calculation.

  fes          -- The finite element space.
  lmax         -- Degree for the spherical harmonic exapansion.
  bdr_marker   -- Marker for the boundary on which the
                  mapping is applied.

  Note that the operator is not ready for use following construction.
  Its Assemble() method must be called to assemble the neccesary
  sparse matrix.

  It is assumed that the boundary to which the DtN mapping is evaluated
  is spherical with centre at the origin.
*/
  Poisson3D(mfem::FiniteElementSpace* fes, int lmax,
            mfem::Array<int>& bdr_marker);

  /*
  Construct the operator for a serial calculation using the default
  choice of boundary.

    fes          -- The finite element space.
    lmax         -- Degree for the spherical harmonic exapansion.

    Note that the operator is not ready for use following construction.
    Its Assemble() method must be called to assemble the neccesary
    sparse matrix.

    It is assumed that the boundary to which the DtN mapping is evaluated
    is spherical with centre at the origin.
  */
  Poisson3D(mfem::FiniteElementSpace* fes, int lmax);

#ifdef MFEM_USE_MPI
  /*
  Construct the operator for a parallel calculation.

    comm         -- The MPI communicator.
    fes          -- The parallel finite element space.
    lmax         -- Degree for the spherical harmonic exapansion.
    bdr_marker   -- Marker for the boundary on which the
                    mapping is applied.


    Note that the operator is not ready for use following construction.
    Its Assemble() method must be called to assemble the neccesary
    sparse matrix.

    It is assumed that the boundary to which the DtN mapping is evaluated
    is spherical with centre at the origin.
  */
  Poisson3D(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int lmax,
            mfem::Array<int>& bdr_marker);

  /*
  Construct the operator for a parallel calculation using the default
  choice of boundary.

  comm         -- The MPI communicator.
  fes          -- The parallel finite element space.
  lmax         -- Degree for the spherical harmonic exapansion.

  Note that the operator is not ready for use following construction.
  Its Assemble() method must be called to assemble the neccesary
  sparse matrix.

  It is assumed that the boundary to which the DtN mapping is evaluated
  is spherical with centre at the origin.
  */
  Poisson3D(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int lmax);

#endif

  // Multiplication method for the operator.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  // Transposed multiplication, which is just multiplication because the
  // operator is self-adjoint.
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override {
    Mult(x, y);
  }

  // Assemble the sparse matrix associated with the operator.
  void Assemble();

  // Return the associated RAP operator that occurs within
  // linear systems involving true degrees of freedom.
  mfem::RAPOperator RAP() const;
};

}  // namespace DtN

}  // namespace mfemElasticity