#pragma once

#include <cassert>
#include <cmath>
#include <memory>

#include "mesh.hpp"  // Assuming this is mfemElasticity/mesh.hpp
#include "mfem.hpp"
#include "mfemElasticity/legendre.hpp"
#include "mfemElasticity/mesh.hpp"  // Redundant if the first "mesh.hpp" is already this one, but safe

namespace mfemElasticity {

/**
 * @brief Dirichlet-to-Neumann (DtN) operator for Poisson's equation on a
 * spherical boundary.
 *
 * This class implements the Dirichlet-to-Neumann operator, $u \mapsto
 * \partial_r u$, for Poisson's equation ($-\nabla^2 u = f$) in a domain with a
 * spherical external boundary. The operator maps Dirichlet boundary data
 * (values of $u$) on the sphere to Neumann boundary data (normal derivatives of
 * $u$) on the same spherical surface.
 *
 * It inherits from `mfem::Operator` for its matrix-vector product capabilities,
 * `LegendreHelper` for spherical harmonic related computations, and
 * `SphericalMeshHelper` for managing spherical mesh properties.
 *
 * The implementation considers both 2D (circular) and 3D (spherical) cases.
 */
class PoissonDtNOperator : public mfem::Operator,
                           protected LegendreHelper,
                           protected SphericalMeshHelper {
 private:
  /** @brief Pointer to the finite element space on which the operator acts. */
  mfem::FiniteElementSpace* _fes;
  /** @brief Spatial dimension of the problem (2 for 2D, 3 for 3D). */
  int _dim;
  /** @brief Polynomial degree of the finite element space. */
  int _degree;
  /** @brief Dimension of the coefficient space (e.g., 1 for scalar Poisson). */
  int _coeff_dim;
  /** @brief The sparse matrix representing the assembled DtN operator. */
  mfem::SparseMatrix _mat;

#ifdef MFEM_USE_MPI
  /** @brief Flag indicating if the operator is used in a parallel context. */
  bool _parallel = false;
  /** @brief Pointer to the parallel finite element space (if in parallel). */
  mfem::ParFiniteElementSpace* _pfes;
  /** @brief MPI communicator used for parallel operations. */
  MPI_Comm _comm;
#endif

#ifndef MFEM_THREAD_SAFE
  /** @brief Mutable workspace vector for coefficient evaluation. */
  mutable mfem::Vector _c;
  /** @brief Workspace for shape functions. */
  mfem::Vector shape;
  /** @brief Workspace for physical coordinates. */
  mfem::Vector _x;
  /** @brief Workspace for sine values in spherical coordinates. */
  mfem::Vector _sin;
  /** @brief Workspace for cosine values in spherical coordinates. */
  mfem::Vector _cos;
  /** @brief Workspace for Legendre polynomials $P_l^m$. */
  mfem::Vector _p;
  /** @brief Workspace for Legendre polynomials $P_{l-1}^m$. */
  mfem::Vector _pm1;
  /** @brief Workspace for element matrix assembly. */
  mfem::DenseMatrix elmat;
#endif

  /**
   * @brief Returns the dimension of the coefficient space.
   * @return The coefficient dimension.
   */
  int CoeffDim() const;

  /**
   * @brief Common setup routine called by both serial and parallel
   * constructors. Initializes common parameters and helper classes.
   */
  void SetUp();

  /**
   * @brief Assembles the 2D element matrix for the DtN operator.
   *
   * This method computes the local contribution of the operator for a
   * given 2D finite element, typically a boundary segment.
   *
   * @param fe The finite element.
   * @param Trans The element transformation.
   * @param elmat The dense matrix to store the element contribution.
   */
  void AssembleElementMatrix2D(const mfem::FiniteElement& fe,
                               mfem::ElementTransformation& Trans,
                               mfem::DenseMatrix& elmat);

  /**
   * @brief Assembles the 3D element matrix for the DtN operator.
   *
   * This method computes the local contribution of the operator for a
   * given 3D finite element, typically a boundary face.
   *
   * @param fe The finite element.
   * @param Trans The element transformation.
   * @param elmat The dense matrix to store the element contribution.
   */
  void AssembleElementMatrix3D(const mfem::FiniteElement& fe,
                               mfem::ElementTransformation& Trans,
                               mfem::DenseMatrix& elmat);

 public:
  /**
   * @brief Constructs a serial PoissonDtNOperator.
   * @param fes Pointer to the finite element space for the solution.
   * @param degree The polynomial degree of the FE space.
   */
  PoissonDtNOperator(mfem::FiniteElementSpace* fes, int degree);

#ifdef MFEM_USE_MPI
  /**
   * @brief Constructs a parallel PoissonDtNOperator.
   * @param comm The MPI communicator.
   * @param fes Pointer to the parallel finite element space for the solution.
   * @param degree The polynomial degree of the FE space.
   */
  PoissonDtNOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* fes,
                     int degree);

#endif

  /**
   * @brief Multiplies the DtN operator matrix by a vector.
   * Computes $y = A \cdot x$.
   * @param x The input vector (Dirichlet data).
   * @param y The output vector (Neumann data).
   */
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  /**
   * @brief Multiplies the transpose of the DtN operator matrix by a vector.
   * For this self-adjoint operator, $A^T = A$, so it calls `Mult(x, y)`.
   * @param x The input vector.
   * @param y The output vector.
   */
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override {
    Mult(x, y);
  }

  /**
   * @brief Assembles the sparse matrix associated with the DtN operator.
   * This method needs to be called after construction to build the internal
   * `_mat`.
   */
  void Assemble();

#ifdef MFEM_USE_MPI
  /**
   * @brief Returns the associated Reduced-Parallel-Assembly (RAP) operator.
   * This is useful for building parallel operators using local assembly.
   * @return An `mfem::RAPOperator` object.
   */
  mfem::RAPOperator RAP() const;
#endif
};

/**
 * @brief Multipole operator for Poisson's equation on a spherical boundary.
 *
 * This class implements a multipole expansion operator relevant for Poisson's
 * equation in spherical domains. It provides methods to construct and apply a
 * matrix operator that typically relates boundary data on one spherical surface
 * to the solution in the exterior or interior, possibly involving a fundamental
 * solution or Green's function.
 *
 * It inherits from `mfem::Operator` for its matrix-vector product capabilities,
 * `LegendreHelper` for spherical harmonic related computations, and
 * `SphericalMeshHelper` for managing spherical mesh properties.
 */
class PoissonMultipoleOperator : public mfem::Operator,
                                 protected LegendreHelper,
                                 protected SphericalMeshHelper {
 private:
  /** @brief Pointer to the trial finite element space. */
  mfem::FiniteElementSpace* _tr_fes;
  /** @brief Pointer to the test finite element space. */
  mfem::FiniteElementSpace* _te_fes;
  /** @brief Spatial dimension of the problem (2 for 2D, 3 for 3D). */
  int _dim;
  /** @brief Polynomial degree of the finite element spaces. */
  int _degree;
  /** @brief Dimension of the coefficient space (e.g., 1 for scalar Poisson). */
  int _coeff_dim;
  /** @brief Marker array indicating which domain attributes are included in the
   * operation. */
  mfem::Array<int> _dom_marker;
  /** @brief The sparse matrix for the left-hand side contribution of the
   * operator. */
  mfem::SparseMatrix _lmat;
  /** @brief The sparse matrix for the right-hand side contribution of the
   * operator. */
  mfem::SparseMatrix _rmat;

#ifdef MFEM_USE_MPI
  /** @brief Flag indicating if the operator is used in a parallel context. */
  bool _parallel = false;
  /** @brief Pointer to the parallel trial finite element space (if in
   * parallel). */
  mfem::ParFiniteElementSpace* _tr_pfes;
  /** @brief Pointer to the parallel test finite element space (if in parallel).
   */
  mfem::ParFiniteElementSpace* _te_pfes;
  /** @brief MPI communicator used for parallel operations. */
  MPI_Comm _comm;
#endif

#ifndef MFEM_THREAD_SAFE
  /** @brief Mutable workspace vector for coefficient evaluation. */
  mutable mfem::Vector _c;
  /** @brief Workspace for shape functions. */
  mfem::Vector shape;
  /** @brief Workspace for physical coordinates. */
  mfem::Vector _x;
  /** @brief Workspace for sine values in spherical coordinates. */
  mfem::Vector _sin;
  /** @brief Workspace for cosine values in spherical coordinates. */
  mfem::Vector _cos;
  /** @brief Workspace for Legendre polynomials $P_l^m$. */
  mfem::Vector _p;
  /** @brief Workspace for Legendre polynomials $P_{l-1}^m$. */
  mfem::Vector _pm1;
  /** @brief Workspace for element matrix assembly. */
  mfem::DenseMatrix elmat;
#endif

  /**
   * @brief Returns the dimension of the coefficient space.
   * @return The coefficient dimension.
   */
  int CoeffDim() const;

  /**
   * @brief Common setup routine called by both serial and parallel
   * constructors. Initializes common parameters and helper classes.
   */
  void SetUp();

  /**
   * @brief Assembles the 2D element matrix for the left-hand side of the
   * Multipole operator.
   * @param fe The finite element.
   * @param Trans The element transformation.
   * @param elmat The dense matrix to store the element contribution.
   */
  void AssembleLeftElementMatrix2D(const mfem::FiniteElement& fe,
                                   mfem::ElementTransformation& Trans,
                                   mfem::DenseMatrix& elmat);

  /**
   * @brief Assembles the 3D element matrix for the left-hand side of the
   * Multipole operator.
   * @param fe The finite element.
   * @param Trans The element transformation.
   * @param elmat The dense matrix to store the element contribution.
   */
  void AssembleLeftElementMatrix3D(const mfem::FiniteElement& fe,
                                   mfem::ElementTransformation& Trans,
                                   mfem::DenseMatrix& elmat);

  /**
   * @brief Assembles the 2D element matrix for the right-hand side of the
   * Multipole operator.
   * @param fe The finite element.
   * @param Trans The element transformation.
   * @param elmat The dense matrix to store the element contribution.
   */
  virtual void AssembleRightElementMatrix2D(const mfem::FiniteElement& fe,
                                            mfem::ElementTransformation& Trans,
                                            mfem::DenseMatrix& elmat);

  /**
   * @brief Assembles the 3D element matrix for the right-hand side of the
   * Multipole operator.
   * @param fe The finite element.
   * @param Trans The element transformation.
   * @param elmat The dense matrix to store the element contribution.
   */
  virtual void AssembleRightElementMatrix3D(const mfem::FiniteElement& fe,
                                            mfem::ElementTransformation& Trans,
                                            mfem::DenseMatrix& elmat);

 public:
  /**
   * @brief Constructs a serial PoissonMultipoleOperator.
   * @param tr_fes Pointer to the trial finite element space.
   * @param te_fes Pointer to the test finite element space.
   * @param degree The polynomial degree of the FE spaces.
   * @param dom_marker An `mfem::Array<int>` marking which domain attributes
   * (1 for inclusion, 0 for exclusion) to consider for assembly.
   */
  PoissonMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                           mfem::FiniteElementSpace* te_fes, int degree,
                           const mfem::Array<int>& dom_marker);

  /**
   * @brief Constructs a serial PoissonMultipoleOperator (move version).
   * This overload takes the `dom_marker` by rvalue reference.
   * @param tr_fes Pointer to the trial finite element space.
   * @param te_fes Pointer to the test finite element space.
   * @param degree The polynomial degree of the FE spaces.
   * @param dom_marker An `mfem::Array<int>` marking which domain attributes
   * (1 for inclusion, 0 for exclusion) to consider for assembly (moved).
   */
  PoissonMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                           mfem::FiniteElementSpace* te_fes, int degree,
                           mfem::Array<int>&& dom_marker)
      : PoissonMultipoleOperator(tr_fes, te_fes, degree, dom_marker) {}

  /**
   * @brief Constructs a serial PoissonMultipoleOperator for all domains.
   * This overload automatically uses `AllDomainsMarker` from
   * `tr_fes->GetMesh()` to include all domain attributes in the assembly.
   * @param tr_fes Pointer to the trial finite element space.
   * @param te_fes Pointer to the test finite element space.
   * @param degree The polynomial degree of the FE spaces.
   */
  PoissonMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                           mfem::FiniteElementSpace* te_fes, int degree)
      : PoissonMultipoleOperator(tr_fes, te_fes, degree,
                                 AllDomainsMarker(tr_fes->GetMesh())) {}

#ifdef MFEM_USE_MPI
  /**
   * @brief Constructs a parallel PoissonMultipoleOperator.
   * @param comm The MPI communicator.
   * @param tr_fes Pointer to the parallel trial finite element space.
   * @param te_fes Pointer to the parallel test finite element space.
   * @param degree The polynomial degree of the FE spaces.
   * @param dom_marker An `mfem::Array<int>` marking which domain attributes
   * (1 for inclusion, 0 for exclusion) to consider for assembly.
   */
  PoissonMultipoleOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                           mfem::ParFiniteElementSpace* te_fes, int degree,
                           const mfem::Array<int>& dom_marker);

  /**
   * @brief Constructs a parallel PoissonMultipoleOperator (move version).
   * This overload takes the `dom_marker` by rvalue reference.
   * @param comm The MPI communicator.
   * @param tr_fes Pointer to the parallel trial finite element space.
   * @param te_fes Pointer to the parallel test finite element space.
   * @param degree The polynomial degree of the FE spaces.
   * @param dom_marker An `mfem::Array<int>` marking which domain attributes
   * (1 for inclusion, 0 for exclusion) to consider for assembly (moved).
   */
  PoissonMultipoleOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                           mfem::ParFiniteElementSpace* te_fes, int degree,
                           mfem::Array<int>&& dom_marker)
      : PoissonMultipoleOperator(comm, tr_fes, te_fes, degree, dom_marker) {}

  /**
   * @brief Constructs a parallel PoissonMultipoleOperator for all domains.
   * This overload automatically uses `AllDomainsMarker` from
   * `tr_fes->GetMesh()` to include all domain attributes in the assembly across
   * all processors.
   * @param comm The MPI communicator.
   * @param tr_fes Pointer to the parallel trial finite element space.
   * @param te_fes Pointer to the parallel test finite element space.
   * @param degree The polynomial degree of the FE spaces.
   */
  PoissonMultipoleOperator(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                           mfem::ParFiniteElementSpace* te_fes, int degree)
      : PoissonMultipoleOperator(comm, tr_fes, te_fes, degree,
                                 AllDomainsMarker(tr_fes->GetMesh())) {}

#endif

  /**
   * @brief Multiplies the Multipole operator matrix by a vector.
   * Computes $y = A \cdot x$.
   * @param x The input vector.
   * @param y The output vector.
   */
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  /**
   * @brief Multiplies the transpose of the Multipole operator matrix by a
   * vector. Computes $y = A^T \cdot x$.
   * @param x The input vector.
   * @param y The output vector.
   */
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

  /**
   * @brief Assembles the sparse matrices associated with the Multipole
   * operator. This method needs to be called after construction to build the
   * internal `_lmat` and `_rmat`.
   */
  void Assemble();

#ifdef MFEM_USE_MPI
  /**
   * @brief Returns the associated Reduced-Parallel-Assembly (RAP) operator.
   * @return An `mfem::RAPOperator` object.
   */
  mfem::RAPOperator RAP() const;
#endif
};

/**
 * @brief Linearised Multipole operator for Poisson's equation on a spherical
 * boundary.
 *
 * This class extends `PoissonMultipoleOperator` to represent a linearised
 * version of the multipole expansion. This is typically used when the geometry
 * or other parameters depend on the solution itself, leading to a non-linear
 * problem that is then linearized.
 *
 * It inherits from `mfem::Operator`, `LegendreHelper`, and
 * `SphericalMeshHelper`.
 */
class PoissonLinearisedMultipoleOperator : public mfem::Operator,
                                           protected LegendreHelper,
                                           protected SphericalMeshHelper {
 private:
  /** @brief Pointer to the trial finite element space. */
  mfem::FiniteElementSpace* _tr_fes;
  /** @brief Pointer to the test finite element space. */
  mfem::FiniteElementSpace* _te_fes;
  /** @brief Spatial dimension of the problem (2 for 2D, 3 for 3D). */
  int _dim;
  /** @brief Polynomial degree of the finite element spaces. */
  int _degree;
  /** @brief Dimension of the coefficient space. */
  int _coeff_dim;
  /** @brief Marker array indicating which domain attributes are included in the
   * operation. */
  mfem::Array<int> _dom_marker;
  /** @brief The sparse matrix for the left-hand side contribution of the
   * operator. */
  mfem::SparseMatrix _lmat;
  /** @brief The sparse matrix for the right-hand side contribution of the
   * operator. */
  mfem::SparseMatrix _rmat;

#ifdef MFEM_USE_MPI
  /** @brief Flag indicating if the operator is used in a parallel context. */
  bool _parallel = false;
  /** @brief Pointer to the parallel trial finite element space (if in
   * parallel). */
  mfem::ParFiniteElementSpace* _tr_pfes;
  /** @brief Pointer to the parallel test finite element space (if in parallel).
   */
  mfem::ParFiniteElementSpace* _te_pfes;
  /** @brief MPI communicator used for parallel operations. */
  MPI_Comm _comm;
#endif

#ifndef MFEM_THREAD_SAFE
  /** @brief Mutable workspace vector for constant coefficient evaluation. */
  mutable mfem::Vector _c0;
  /** @brief Workspace for shape functions. */
  mfem::Vector shape;
  /** @brief Workspace for physical coordinates. */
  mfem::Vector _x;
  /** @brief Workspace for sine values in spherical coordinates. */
  mfem::Vector _sin;
  /** @brief Workspace for cosine values in spherical coordinates. */
  mfem::Vector _cos;
  /** @brief Workspace for Legendre polynomials $P_l^m$. */
  mfem::Vector _p;
  /** @brief Workspace for Legendre polynomials $P_{l-1}^m$. */
  mfem::Vector _pm1;
  /** @brief Workspace for linearised coefficient `c1`. */
  mfem::Vector _c1;
  /** @brief Workspace for linearised coefficient `c2`. */
  mfem::Vector _c2;
  /** @brief Workspace for element matrix assembly. */
  mfem::DenseMatrix elmat;
  /** @brief Workspace for partial element matrix assembly. */
  mfem::DenseMatrix part_elmat;
#endif

  /**
   * @brief Returns the dimension of the coefficient space.
   * @return The coefficient dimension.
   */
  int CoeffDim() const;

  /**
   * @brief Common setup routine called by both serial and parallel
   * constructors. Initializes common parameters and helper classes.
   */
  void SetUp();

  /**
   * @brief Assembles the 2D element matrix for the left-hand side of the
   * Linearised Multipole operator.
   * @param fe The finite element.
   * @param Trans The element transformation.
   * @param elmat The dense matrix to store the element contribution.
   */
  void AssembleLeftElementMatrix2D(const mfem::FiniteElement& fe,
                                   mfem::ElementTransformation& Trans,
                                   mfem::DenseMatrix& elmat);

  /**
   * @brief Assembles the 3D element matrix for the left-hand side of the
   * Linearised Multipole operator.
   * @param fe The finite element.
   * @param Trans The element transformation.
   * @param elmat The dense matrix to store the element contribution.
   */
  void AssembleLeftElementMatrix3D(const mfem::FiniteElement& fe,
                                   mfem::ElementTransformation& Trans,
                                   mfem::DenseMatrix& elmat);

  /**
   * @brief Assembles the 2D element matrix for the right-hand side of the
   * Linearised Multipole operator.
   * @param fe The finite element.
   * @param Trans The element transformation.
   * @param elmat The dense matrix to store the element contribution.
   */
  virtual void AssembleRightElementMatrix2D(const mfem::FiniteElement& fe,
                                            mfem::ElementTransformation& Trans,
                                            mfem::DenseMatrix& elmat);

  /**
   * @brief Assembles the 3D element matrix for the right-hand side of the
   * Linearised Multipole operator.
   * @param fe The finite element.
   * @param Trans The element transformation.
   * @param elmat The dense matrix to store the element contribution.
   */
  virtual void AssembleRightElementMatrix3D(const mfem::FiniteElement& fe,
                                            mfem::ElementTransformation& Trans,
                                            mfem::DenseMatrix& elmat);

 public:
  /**
   * @brief Constructs a serial PoissonLinearisedMultipoleOperator.
   * @param tr_fes Pointer to the trial finite element space.
   * @param te_fes Pointer to the test finite element space.
   * @param degree The polynomial degree of the FE spaces.
   * @param dom_marker An `mfem::Array<int>` marking which domain attributes
   * (1 for inclusion, 0 for exclusion) to consider for assembly.
   */
  PoissonLinearisedMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                                     mfem::FiniteElementSpace* te_fes,
                                     int degree,
                                     const mfem::Array<int>& dom_marker);

  /**
   * @brief Constructs a serial PoissonLinearisedMultipoleOperator (move
   * version). This overload takes the `dom_marker` by rvalue reference.
   * @param tr_fes Pointer to the trial finite element space.
   * @param te_fes Pointer to the test finite element space.
   * @param degree The polynomial degree of the FE spaces.
   * @param dom_marker An `mfem::Array<int>` marking which domain attributes
   * (1 for inclusion, 0 for exclusion) to consider for assembly (moved).
   */
  PoissonLinearisedMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                                     mfem::FiniteElementSpace* te_fes,
                                     int degree, mfem::Array<int>&& dom_marker)
      : PoissonLinearisedMultipoleOperator(tr_fes, te_fes, degree, dom_marker) {
  }

  /**
   * @brief Constructs a serial PoissonLinearisedMultipoleOperator for all
   * domains. This overload automatically uses `AllDomainsMarker` from
   * `tr_fes->GetMesh()` to include all domain attributes in the assembly.
   * @param tr_fes Pointer to the trial finite element space.
   * @param te_fes Pointer to the test finite element space.
   * @param degree The polynomial degree of the FE spaces.
   */
  PoissonLinearisedMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                                     mfem::FiniteElementSpace* te_fes,
                                     int degree)
      : PoissonLinearisedMultipoleOperator(
            tr_fes, te_fes, degree, AllDomainsMarker(tr_fes->GetMesh())) {}

#ifdef MFEM_USE_MPI
  /**
   * @brief Constructs a parallel PoissonLinearisedMultipoleOperator.
   * @param comm The MPI communicator.
   * @param tr_fes Pointer to the parallel trial finite element space.
   * @param te_fes Pointer to the parallel test finite element space.
   * @param degree The polynomial degree of the FE spaces.
   * @param dom_marker An `mfem::Array<int>` marking which domain attributes
   * (1 for inclusion, 0 for exclusion) to consider for assembly.
   */
  PoissonLinearisedMultipoleOperator(MPI_Comm comm,
                                     mfem::ParFiniteElementSpace* tr_fes,
                                     mfem::ParFiniteElementSpace* te_fes,
                                     int degree,
                                     const mfem::Array<int>& dom_marker);

  /**
   * @brief Constructs a parallel PoissonLinearisedMultipoleOperator (move
   * version). This overload takes the `dom_marker` by rvalue reference.
   * @param comm The MPI communicator.
   * @param tr_fes Pointer to the parallel trial finite element space.
   * @param te_fes Pointer to the parallel test finite element space.
   * @param degree The polynomial degree of the FE spaces.
   * @param dom_marker An `mfem::Array<int>` marking which domain attributes
   * (1 for inclusion, 0 for exclusion) to consider for assembly (moved).
   */
  PoissonLinearisedMultipoleOperator(MPI_Comm comm,
                                     mfem::ParFiniteElementSpace* tr_fes,
                                     mfem::ParFiniteElementSpace* te_fes,
                                     int degree, mfem::Array<int>&& dom_marker)
      : PoissonLinearisedMultipoleOperator(comm, tr_fes, te_fes, degree,
                                           dom_marker) {}

  /**
   * @brief Constructs a parallel PoissonLinearisedMultipoleOperator for all
   * domains. This overload automatically uses `AllDomainsMarker` from
   * `tr_fes->GetMesh()` to include all domain attributes in the assembly across
   * all processors.
   * @param comm The MPI communicator.
   * @param tr_fes Pointer to the parallel trial finite element space.
   * @param te_fes Pointer to the parallel test finite element space.
   * @param degree The polynomial degree of the FE spaces.
   */
  PoissonLinearisedMultipoleOperator(MPI_Comm comm,
                                     mfem::ParFiniteElementSpace* tr_fes,
                                     mfem::ParFiniteElementSpace* te_fes,
                                     int degree)
      : PoissonLinearisedMultipoleOperator(
            comm, tr_fes, te_fes, degree, AllDomainsMarker(tr_fes->GetMesh())) {
  }

#endif

  /**
   * @brief Multiplies the Linearised Multipole operator matrix by a vector.
   * Computes $y = A \cdot x$.
   * @param x The input vector.
   * @param y The output vector.
   */
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  /**
   * @brief Multiplies the transpose of the Linearised Multipole operator matrix
   * by a vector. Computes $y = A^T \cdot x$.
   * @param x The input vector.
   * @param y The output vector.
   */
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

  /**
   * @brief Assembles the sparse matrices associated with the Linearised
   * Multipole operator. This method needs to be called after construction to
   * build the internal `_lmat` and `_rmat`.
   */
  void Assemble();

#ifdef MFEM_USE_MPI
  /**
   * @brief Returns the associated Reduced-Parallel-Assembly (RAP) operator.
   * @return An `mfem::RAPOperator` object.
   */
  mfem::RAPOperator RAP() const;
#endif
};

}  // namespace mfemElasticity