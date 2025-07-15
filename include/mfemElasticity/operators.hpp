#include <cassert>
#include <cmath>

#include "mfem.hpp"
#include "mfemElasticity/legendre.hpp"
#include "mfemElasticity/mesh.hpp"

namespace mfemElasticity {

/**
  DtN operator for Poissons equation on a spherical
  boundary in 2D or 3D.
**/
class PoissonDtNOperator : public mfem::Operator, private LegendreHelper {
 protected:
  mfem::FiniteElementSpace* _fes;
  int _dim;
  int _degree;
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
  mfem::Vector shape, _x, _sin, _cos, _p, _pm1;
  mfem::DenseMatrix elmat;
#endif

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
  PoissonDtNOperator(mfem::FiniteElementSpace* fes, int degree,
                     const mfem::Array<int>& bdr_marker);

  PoissonDtNOperator(mfem::FiniteElementSpace* fes, int degree,
                     mfem::Array<int>&& bdr_marker)
      : PoissonDtNOperator(fes, degree, bdr_marker) {}

  PoissonDtNOperator(mfem::FiniteElementSpace* fes, int degree)
      : PoissonDtNOperator(fes, degree,
                           ExternalBoundaryMarker(fes->GetMesh())) {}

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  PoissonDtNOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* fes,
                     int degree, const mfem::Array<int>& bdr_marker);

  PoissonDtNOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* fes,
                     int degree, mfem::Array<int>&& bdr_marker)
      : PoissonDtNOperator(comm, fes, degree, bdr_marker) {}

  PoissonDtNOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* fes,
                     int degree)
      : PoissonDtNOperator(comm, fes, degree,
                           ExternalBoundaryMarker(fes->GetMesh())) {}
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

/**
  Multipole operator for Poissons equation on a spherical
  boundary in 2D or 3D.
**/
class PoissonMultipoleOperator : public mfem::Operator, private LegendreHelper {
 protected:
  mfem::FiniteElementSpace* _tr_fes;
  mfem::FiniteElementSpace* _te_fes;
  int _dim;
  int _degree;
  int _coeff_dim;
  mfem::real_t _bdr_radius;
  mfem::Array<int> _bdr_marker;
  mfem::Array<int> _dom_marker;
  mfem::SparseMatrix _lmat;
  mfem::SparseMatrix _rmat;

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

  // Common set up between the serial and parallel constructors.
  void SetUp();

  // Element level calculations.
  void AssembleLeftElementMatrix2D(const mfem::FiniteElement& fe,
                                   mfem::ElementTransformation& Trans,
                                   mfem::DenseMatrix& elmat);

  void AssembleLeftElementMatrix3D(const mfem::FiniteElement& fe,
                                   mfem::ElementTransformation& Trans,
                                   mfem::DenseMatrix& elmat);

  void AssembleRightElementMatrix2D(const mfem::FiniteElement& fe,
                                    mfem::ElementTransformation& Trans,
                                    mfem::DenseMatrix& elmat);

  void AssembleRightElementMatrix3D(const mfem::FiniteElement& fe,
                                    mfem::ElementTransformation& Trans,
                                    mfem::DenseMatrix& elmat);

 public:
  // Serial constructors.
  PoissonMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                           mfem::FiniteElementSpace* te_fes, int degree,
                           const mfem::Array<int>& dom_marker,
                           const mfem::Array<int>& bdr_marker);

  PoissonMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                           mfem::FiniteElementSpace* te_fes, int degree,
                           mfem::Array<int>&& dom_marker,
                           const mfem::Array<int>& bdr_marker)
      : PoissonMultipoleOperator(tr_fes, te_fes, degree, dom_marker,
                                 bdr_marker) {}

  PoissonMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                           mfem::FiniteElementSpace* te_fes, int degree,
                           const mfem::Array<int>& dom_marker,
                           mfem::Array<int>&& bdr_marker)
      : PoissonMultipoleOperator(tr_fes, te_fes, degree, dom_marker,
                                 bdr_marker) {}

  PoissonMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                           mfem::FiniteElementSpace* te_fes, int degree,
                           mfem::Array<int>&& dom_marker,
                           mfem::Array<int>&& bdr_marker)
      : PoissonMultipoleOperator(tr_fes, te_fes, degree, dom_marker,
                                 bdr_marker) {}

  PoissonMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                           mfem::FiniteElementSpace* te_fes, int degree)
      : PoissonMultipoleOperator(tr_fes, te_fes, degree,
                                 AllBoundariesMarker(tr_fes->GetMesh()),
                                 ExternalBoundaryMarker(tr_fes->GetMesh())) {}

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  PoissonMultipoleOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                           mfem::ParFiniteElementSpace* te_fes, int degree,
                           const mfem::Array<int>& dom_marker,
                           const mfem::Array<int>& bdr_marker);

  PoissonMultipoleOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                           mfem::ParFiniteElementSpace* te_fes, int degree,
                           mfem::Array<int>&& dom_marker,
                           const mfem::Array<int>& bdr_marker)
      : PoissonMultipoleOperator(comm, tr_fes, te_fes, degree, dom_marker,
                                 bdr_marker) {}

  PoissonMultipoleOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                           mfem::ParFiniteElementSpace* te_fes, int degree,
                           const mfem::Array<int>& dom_marker,
                           mfem::Array<int>&& bdr_marker)
      : PoissonMultipoleOperator(comm, tr_fes, te_fes, degree, dom_marker,
                                 bdr_marker) {}

  PoissonMultipoleOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                           mfem::ParFiniteElementSpace* te_fes, int degree,
                           mfem::Array<int>&& dom_marker,
                           mfem::Array<int>&& bdr_marker)
      : PoissonMultipoleOperator(comm, tr_fes, te_fes, degree, dom_marker,
                                 bdr_marker) {}

  PoissonMultipoleOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                           mfem::ParFiniteElementSpace* te_fes, int degree)
      : PoissonMultipoleOperator(comm, tr_fes, te_fes, degree,
                                 AllBoundariesMarker(tr_fes->GetMesh()),
                                 ExternalBoundaryMarker(tr_fes->GetMesh())) {}

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

/*----------------------------------------------------------
    Base class for Multipole operator for Poisson equation
------------------------------------------------------------*/
class PoissonMultipole : public mfem::Integrator, public mfem::Operator {
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
  // Serial constructors.
  PoissonMultipole(mfem::FiniteElementSpace* tr_fes,
                   mfem::FiniteElementSpace* te_fes, int coeff_dim,
                   const mfem::Array<int>& dom_marker,
                   const mfem::Array<int>& bdr_marker);

  PoissonMultipole(mfem::FiniteElementSpace* tr_fes,
                   mfem::FiniteElementSpace* te_fes, int coeff_dim,
                   mfem::Array<int>&& dom_marker,
                   const mfem::Array<int>& bdr_marker)
      : PoissonMultipole(tr_fes, te_fes, coeff_dim, dom_marker, bdr_marker) {}

  PoissonMultipole(mfem::FiniteElementSpace* tr_fes,
                   mfem::FiniteElementSpace* te_fes, int coeff_dim,
                   const mfem::Array<int>& dom_marker,
                   mfem::Array<int>&& bdr_marker)
      : PoissonMultipole(tr_fes, te_fes, coeff_dim, dom_marker, bdr_marker) {}

  PoissonMultipole(mfem::FiniteElementSpace* tr_fes,
                   mfem::FiniteElementSpace* te_fes, int coeff_dim,
                   mfem::Array<int>&& dom_marker, mfem::Array<int>&& bdr_marker)
      : PoissonMultipole(tr_fes, te_fes, coeff_dim, dom_marker, bdr_marker) {}

#ifdef MFEM_USE_MPI
  // Parallel constructors
  PoissonMultipole(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                   mfem::ParFiniteElementSpace* te_fes, int coeff_dim,
                   const mfem::Array<int>& dom_marker,
                   const mfem::Array<int>& bdr_marker);

  PoissonMultipole(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                   mfem::ParFiniteElementSpace* te_fes, int coeff_dim,
                   mfem::Array<int>&& dom_marker,
                   const mfem::Array<int>& bdr_marker)
      : PoissonMultipole(comm, tr_fes, te_fes, coeff_dim, dom_marker,
                         bdr_marker) {}

  PoissonMultipole(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                   mfem::ParFiniteElementSpace* te_fes, int coeff_dim,
                   mfem::Array<int>&& dom_marker, mfem::Array<int>&& bdr_marker)
      : PoissonMultipole(comm, tr_fes, te_fes, coeff_dim, dom_marker,
                         bdr_marker) {}

  PoissonMultipole(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                   mfem::ParFiniteElementSpace* te_fes, int coeff_dim,
                   const mfem::Array<int>& dom_marker,
                   mfem::Array<int>&& bdr_marker)
      : PoissonMultipole(comm, tr_fes, te_fes, coeff_dim, dom_marker,
                         bdr_marker) {}
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
  mfem::RAPOperator RAP() const;
};

/*----------------------------------------------------------
     Class for Multipole operator for Poisson 2D equation
------------------------------------------------------------*/
class PoissonMultipoleCircle : public PoissonMultipole {
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
  PoissonMultipoleCircle(mfem::FiniteElementSpace* tr_fes,
                         mfem::FiniteElementSpace* te_fes, int kMax,
                         const mfem::Array<int>& dom_marker,
                         const mfem::Array<int>& bdr_marker)
      : PoissonMultipole(tr_fes, te_fes, 2 * kMax, dom_marker, bdr_marker),
        _kMax{kMax} {}

  PoissonMultipoleCircle(mfem::FiniteElementSpace* tr_fes,
                         mfem::FiniteElementSpace* te_fes, int kMax,
                         mfem::Array<int>&& dom_marker,
                         const mfem::Array<int>& bdr_marker)
      : PoissonMultipole(tr_fes, te_fes, 2 * kMax, dom_marker, bdr_marker),
        _kMax{kMax} {}

  PoissonMultipoleCircle(mfem::FiniteElementSpace* tr_fes,
                         mfem::FiniteElementSpace* te_fes, int kMax,
                         const mfem::Array<int>& dom_marker,
                         mfem::Array<int>&& bdr_marker)
      : PoissonMultipole(tr_fes, te_fes, 2 * kMax, dom_marker, bdr_marker),
        _kMax{kMax} {}

  PoissonMultipoleCircle(mfem::FiniteElementSpace* tr_fes,
                         mfem::FiniteElementSpace* te_fes, int kMax,
                         mfem::Array<int>&& dom_marker,
                         mfem::Array<int>&& bdr_marker)
      : PoissonMultipole(tr_fes, te_fes, 2 * kMax, dom_marker, bdr_marker),
        _kMax{kMax} {}

  PoissonMultipoleCircle(mfem::FiniteElementSpace* tr_fes,
                         mfem::FiniteElementSpace* te_fes, int kMax)
      : PoissonMultipole(tr_fes, te_fes, 2 * kMax,
                         AllDomainsMarker(tr_fes->GetMesh()),
                         ExternalBoundaryMarker(tr_fes->GetMesh())),
        _kMax{kMax} {}

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  PoissonMultipoleCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                         mfem::ParFiniteElementSpace* te_fes, int kMax,
                         const mfem::Array<int>& dom_marker,
                         const mfem::Array<int>& bdr_marker)
      : PoissonMultipole(comm, tr_fes, te_fes, 2 * kMax, dom_marker,
                         bdr_marker),
        _kMax{kMax} {}

  PoissonMultipoleCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                         mfem::ParFiniteElementSpace* te_fes, int kMax,
                         mfem::Array<int>&& dom_marker,
                         const mfem::Array<int>& bdr_marker)
      : PoissonMultipole(comm, tr_fes, te_fes, 2 * kMax, dom_marker,
                         bdr_marker),
        _kMax{kMax} {}

  PoissonMultipoleCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                         mfem::ParFiniteElementSpace* te_fes, int kMax,
                         const mfem::Array<int>& dom_marker,
                         mfem::Array<int>&& bdr_marker)
      : PoissonMultipole(comm, tr_fes, te_fes, 2 * kMax, dom_marker,
                         bdr_marker),
        _kMax{kMax} {}

  PoissonMultipoleCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                         mfem::ParFiniteElementSpace* te_fes, int kMax,
                         mfem::Array<int>&& dom_marker,
                         mfem::Array<int>&& bdr_marker)
      : PoissonMultipole(comm, tr_fes, te_fes, 2 * kMax, dom_marker,
                         bdr_marker),
        _kMax{kMax} {}

  PoissonMultipoleCircle(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                         mfem::ParFiniteElementSpace* te_fes, int kMax)
      : PoissonMultipole(comm, tr_fes, te_fes, 2 * kMax,
                         AllDomainsMarker(tr_fes->GetMesh()),
                         ExternalBoundaryMarker(tr_fes->GetMesh())),
        _kMax{kMax} {}

#endif
};

/*----------------------------------------------------------
     Class for Multipole operator for Poisson 3D equation
------------------------------------------------------------*/
class PoissonMultipoleSphere : public PoissonMultipole, private LegendreHelper {
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
  PoissonMultipoleSphere(mfem::FiniteElementSpace* tr_fes,
                         mfem::FiniteElementSpace* te_fes, int lMax,
                         const mfem::Array<int>& dom_marker,
                         const mfem::Array<int>& bdr_marker)
      : PoissonMultipole(tr_fes, te_fes, (lMax + 1) * (lMax + 1), dom_marker,
                         bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonMultipoleSphere(mfem::FiniteElementSpace* tr_fes,
                         mfem::FiniteElementSpace* te_fes, int lMax,
                         mfem::Array<int>&& dom_marker,
                         const mfem::Array<int>& bdr_marker)
      : PoissonMultipole(tr_fes, te_fes, (lMax + 1) * (lMax + 1), dom_marker,
                         bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonMultipoleSphere(mfem::FiniteElementSpace* tr_fes,
                         mfem::FiniteElementSpace* te_fes, int lMax,
                         const mfem::Array<int>& dom_marker,
                         mfem::Array<int>&& bdr_marker)
      : PoissonMultipole(tr_fes, te_fes, (lMax + 1) * (lMax + 1), dom_marker,
                         bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonMultipoleSphere(mfem::FiniteElementSpace* tr_fes,
                         mfem::FiniteElementSpace* te_fes, int lMax,
                         mfem::Array<int>&& dom_marker,
                         mfem::Array<int>&& bdr_marker)
      : PoissonMultipole(tr_fes, te_fes, (lMax + 1) * (lMax + 1), dom_marker,
                         bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonMultipoleSphere(mfem::FiniteElementSpace* tr_fes,
                         mfem::FiniteElementSpace* te_fes, int lMax)
      : PoissonMultipole(tr_fes, te_fes, (lMax + 1) * (lMax + 1),
                         AllDomainsMarker(tr_fes->GetMesh()),
                         ExternalBoundaryMarker(tr_fes->GetMesh())),
        _lMax{lMax} {
    SetUp();
  }

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  PoissonMultipoleSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                         mfem::ParFiniteElementSpace* te_fes, int lMax,
                         const mfem::Array<int>& dom_marker,
                         const mfem::Array<int>& bdr_marker)
      : PoissonMultipole(comm, tr_fes, te_fes, (lMax + 1) * (lMax + 1),
                         dom_marker, bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonMultipoleSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                         mfem::ParFiniteElementSpace* te_fes, int lMax,
                         mfem::Array<int>&& dom_marker,
                         const mfem::Array<int>& bdr_marker)
      : PoissonMultipole(comm, tr_fes, te_fes, (lMax + 1) * (lMax + 1),
                         dom_marker, bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonMultipoleSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                         mfem::ParFiniteElementSpace* te_fes, int lMax,
                         const mfem::Array<int>& dom_marker,
                         mfem::Array<int>&& bdr_marker)
      : PoissonMultipole(comm, tr_fes, te_fes, (lMax + 1) * (lMax + 1),
                         dom_marker, bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonMultipoleSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                         mfem::ParFiniteElementSpace* te_fes, int lMax,
                         mfem::Array<int>&& dom_marker,
                         mfem::Array<int>&& bdr_marker)
      : PoissonMultipole(comm, tr_fes, te_fes, (lMax + 1) * (lMax + 1),
                         dom_marker, bdr_marker),
        _lMax{lMax} {
    SetUp();
  }

  PoissonMultipoleSphere(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                         mfem::ParFiniteElementSpace* te_fes, int lMax)
      : PoissonMultipole(comm, tr_fes, te_fes, (lMax + 1) * (lMax + 1),
                         AllDomainsMarker(tr_fes->GetMesh()),
                         ExternalBoundaryMarker(tr_fes->GetMesh())),
        _lMax{lMax} {
    SetUp();
  }

#endif
};

class FirstMoments : public mfem::Operator, public mfem::Integrator {
 private:
  int _dim;
  mfem::FiniteElementSpace* _fes;
  mfem::Array<int> _dom_marker;
  mfem::SparseMatrix _mat;

#ifdef MFEM_USE_MPI
  bool _parallel = false;
  mfem::ParFiniteElementSpace* _pfes;
  MPI_Comm _comm;
#endif

#ifndef MFEM_THREAD_SAFE
  mfem::Vector _x, shape;
#endif

  // Element level calculation of sparse matrix. Pure virtual method
  // that is overridden in derived classes.
  void AssembleElementMatrix(const mfem::FiniteElement& fe,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat);

 public:
  // Serial constructors.
  FirstMoments(mfem::FiniteElementSpace* fes,
               const mfem::Array<int>& dom_marker);

  FirstMoments(mfem::FiniteElementSpace* fes, mfem::Array<int>&& dom_marker)
      : FirstMoments(fes, dom_marker) {}

  FirstMoments(mfem::FiniteElementSpace* fes)
      : FirstMoments(fes, AllDomainsMarker(fes->GetMesh())) {}

#ifdef MFEM_USE_MPI
  // Parallel constructors.
  FirstMoments(MPI_Comm comm, mfem::ParFiniteElementSpace* fes,
               const mfem::Array<int>& dom_marker);

  FirstMoments(MPI_Comm comm, mfem::ParFiniteElementSpace* fes,
               mfem::Array<int>&& dom_marker)
      : FirstMoments(comm, fes, dom_marker) {}

  FirstMoments(MPI_Comm comm, mfem::ParFiniteElementSpace* fes)
      : FirstMoments(comm, fes, AllDomainsMarker(fes->GetMesh())) {}

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