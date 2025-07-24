#pragma once

#include <cassert>
#include <cmath>
#include <memory>

#include "mesh.hpp"
#include "mfem.hpp"
#include "mfemElasticity/legendre.hpp"
#include "mfemElasticity/mesh.hpp"

namespace mfemElasticity {

/************************************************************
*************************************************************
  DtN operator for Poisson's equation on a spherical
  boundary in 2D or 3D.
*************************************************************
************************************************************/
class PoissonDtNOperator : public mfem::Operator,
                           protected LegendreHelper,
                           protected SphericalMeshHelper {
 private:
  mfem::FiniteElementSpace* _fes;
  int _dim;
  int _degree;
  int _coeff_dim;
  mfem::SparseMatrix _mat;

#ifdef MFEM_USE_MPI
  bool _parallel = false;
  mfem::ParFiniteElementSpace* _pfes;
  MPI_Comm _comm;
#endif

#ifndef MFEM_THREAD_SAFE
  mutable mfem::Vector _c;
  mfem::Vector shape, _x, _sin, _cos, _p, _pm1;
  mfem::DenseMatrix elmat;
#endif

  // Return the coefficient dimension.
  int CoeffDim() const;

  // Common set up between the serial and parallel constructors.
  void SetUp();

  // Element level calculations.
  void AssembleElementMatrix2D(const mfem::FiniteElement& fe,
                               mfem::ElementTransformation& Trans,
                               mfem::DenseMatrix& elmat);

  void AssembleElementMatrix3D(const mfem::FiniteElement& fe,
                               mfem::ElementTransformation& Trans,
                               mfem::DenseMatrix& elmat);

 public:
  // Serial constructors.
  PoissonDtNOperator(mfem::FiniteElementSpace* fes, int degree);

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  PoissonDtNOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* fes,
                     int degree);

#endif

  // Multiplication by the operator.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  // Transposed multiplication by the operator.
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override {
    Mult(x, y);
  }

  // Assemble the sparse matrix associated with the operator.
  void Assemble();

#ifdef MFEM_USE_MPI
  // Return the associated RAP operator.
  mfem::RAPOperator RAP() const;
#endif
};

/*************************************************************
**************************************************************
  Multipole operator for Poisson's equation on a spherical
  boundary in 2D or 3D.
***************************************************************
**************************************************************/

class PoissonMultipoleOperator : public mfem::Operator,
                                 protected LegendreHelper,
                                 protected SphericalMeshHelper {
 protected:
  mfem::FiniteElementSpace* _tr_fes;
  mfem::FiniteElementSpace* _te_fes;
  int _dim;
  int _degree;
  int _coeff_dim;
  mfem::Array<int> _dom_marker;
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
  mfem::Vector shape, _x, _sin, _cos, _p, _pm1;
  mfem::DenseMatrix elmat;
#endif

  // Return the coefficient dimension.
  int CoeffDim() const;

  // Common set up between the serial and parallel constructors.
  void SetUp();

  // Element level calculations.
  void AssembleLeftElementMatrix2D(const mfem::FiniteElement& fe,
                                   mfem::ElementTransformation& Trans,
                                   mfem::DenseMatrix& elmat);

  void AssembleLeftElementMatrix3D(const mfem::FiniteElement& fe,
                                   mfem::ElementTransformation& Trans,
                                   mfem::DenseMatrix& elmat);

  virtual void AssembleRightElementMatrix2D(const mfem::FiniteElement& fe,
                                            mfem::ElementTransformation& Trans,
                                            mfem::DenseMatrix& elmat);

  virtual void AssembleRightElementMatrix3D(const mfem::FiniteElement& fe,
                                            mfem::ElementTransformation& Trans,
                                            mfem::DenseMatrix& elmat);

 public:
  // Serial constructors.
  PoissonMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                           mfem::FiniteElementSpace* te_fes, int degree,
                           const mfem::Array<int>& dom_marker);

  PoissonMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                           mfem::FiniteElementSpace* te_fes, int degree,
                           mfem::Array<int>&& dom_marker)
      : PoissonMultipoleOperator(tr_fes, te_fes, degree, dom_marker) {}

  PoissonMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                           mfem::FiniteElementSpace* te_fes, int degree)
      : PoissonMultipoleOperator(tr_fes, te_fes, degree,
                                 AllDomainsMarker(tr_fes->GetMesh())) {}

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  PoissonMultipoleOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                           mfem::ParFiniteElementSpace* te_fes, int degree,
                           const mfem::Array<int>& dom_marker);

  PoissonMultipoleOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                           mfem::ParFiniteElementSpace* te_fes, int degree,
                           mfem::Array<int>&& dom_marker)
      : PoissonMultipoleOperator(comm, tr_fes, te_fes, degree, dom_marker) {}

  PoissonMultipoleOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                           mfem::ParFiniteElementSpace* te_fes, int degree)
      : PoissonMultipoleOperator(comm, tr_fes, te_fes, degree,
                                 AllDomainsMarker(tr_fes->GetMesh())) {}

#endif

  // Multiplication by the operator.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  // Transposed multiplication by the operator.
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

  // Assemble the sparse matrix associated with the operator.
  void Assemble();

#ifdef MFEM_USE_MPI
  // Return the associated RAP operator.
  mfem::RAPOperator RAP() const;
#endif
};

/*************************************************************
**************************************************************
  Linearised Multipole operator for Poisson's equation on a
  spherical boundary in 2D or 3D.
***************************************************************
**************************************************************/

class PoissonLinearisedMultipoleOperator : public mfem::Operator,
                                           protected LegendreHelper,
                                           protected SphericalMeshHelper {
 protected:
  mfem::FiniteElementSpace* _tr_fes;
  mfem::FiniteElementSpace* _te_fes;
  int _dim;
  int _degree;
  int _coeff_dim;
  mfem::Array<int> _dom_marker;
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
  mfem::Vector shape, _x, _sin, _cos, _p, _pm1, _d;
  mfem::DenseMatrix elmat, part_elmat;
#endif

  // Return the coefficient dimension.
  int CoeffDim() const;

  // Common set up between the serial and parallel constructors.
  void SetUp();

  // Element level calculations.
  void AssembleLeftElementMatrix2D(const mfem::FiniteElement& fe,
                                   mfem::ElementTransformation& Trans,
                                   mfem::DenseMatrix& elmat);

  void AssembleLeftElementMatrix3D(const mfem::FiniteElement& fe,
                                   mfem::ElementTransformation& Trans,
                                   mfem::DenseMatrix& elmat);

  virtual void AssembleRightElementMatrix2D(const mfem::FiniteElement& fe,
                                            mfem::ElementTransformation& Trans,
                                            mfem::DenseMatrix& elmat);

  virtual void AssembleRightElementMatrix3D(const mfem::FiniteElement& fe,
                                            mfem::ElementTransformation& Trans,
                                            mfem::DenseMatrix& elmat);

 public:
  // Serial constructors.
  PoissonLinearisedMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                                     mfem::FiniteElementSpace* te_fes,
                                     int degree,
                                     const mfem::Array<int>& dom_marker);

  PoissonLinearisedMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                                     mfem::FiniteElementSpace* te_fes,
                                     int degree, mfem::Array<int>&& dom_marker)
      : PoissonLinearisedMultipoleOperator(tr_fes, te_fes, degree, dom_marker) {
  }

  PoissonLinearisedMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                                     mfem::FiniteElementSpace* te_fes,
                                     int degree)
      : PoissonLinearisedMultipoleOperator(
            tr_fes, te_fes, degree, AllDomainsMarker(tr_fes->GetMesh())) {}

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  PoissonLinearisedMultipoleOperator(MPI_Comm comm,
                                     mfem::ParFiniteElementSpace* tr_fes,
                                     mfem::ParFiniteElementSpace* te_fes,
                                     int degree,
                                     const mfem::Array<int>& dom_marker);

  PoissonLinearisedMultipoleOperator(MPI_Comm comm,
                                     mfem::ParFiniteElementSpace* tr_fes,
                                     mfem::ParFiniteElementSpace* te_fes,
                                     int degree, mfem::Array<int>&& dom_marker)
      : PoissonLinearisedMultipoleOperator(comm, tr_fes, te_fes, degree,
                                           dom_marker) {}

  PoissonLinearisedMultipoleOperator(MPI_Comm comm,
                                     mfem::ParFiniteElementSpace* tr_fes,
                                     mfem::ParFiniteElementSpace* te_fes,
                                     int degree)
      : PoissonLinearisedMultipoleOperator(
            comm, tr_fes, te_fes, degree, AllDomainsMarker(tr_fes->GetMesh())) {
  }

#endif

  // Multiplication by the operator.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  // Transposed multiplication by the operator.
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

  // Assemble the sparse matrix associated with the operator.
  void Assemble();

#ifdef MFEM_USE_MPI
  // Return the associated RAP operator.
  mfem::RAPOperator RAP() const;
#endif
};

}  // namespace mfemElasticity