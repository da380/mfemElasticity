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
 * @brief Galerkin representation of the Dirichlet-to-Neumann (DtN) operator
 * for Poisson's equation on a spherical boundary. This is associated with
 * the bilinearform
 * \f[
 * (v,u) \mapsto \int_{\partial \Omega} v \frac{\partial u}{\partial n} \dd S,
 * \f]
 * where the normal derivative is determined from the boundary values of \f$u\f$
 * using the exterior solution of Laplace's equations expressed using the
 * appropriate spectral basis (i.e., Fourier series in 2D and spherical
 * harmonics in 3D).
 *
 * In 2D the Dirichlet-to-Neumann map results in the following:
 * \f[
 * \int_{\partial \Omega} v \frac{\partial u}{\partial n} \dd S =
 * \frac{1}{\pi b}\sum_{k\ne 0} |k| v_{k} u_{k},
 * \f]
 * where
 * \f[
 * u_{k} = \left\{
 * \begin{array}{c}
 * \int_{\partial \Omega} \cos k \theta \,u(b,\theta) \dd S && k < 0 \\
 * \int_{\partial \Omega} \sin k \theta \, u(b,\theta) \dd S && k > 0
 * \end{array}
 * \right.
 * \f]
 * with the polar co-ordinates calculated relative to the boundary's centroid,
 * and similarly for \f$v_{k}\f$.
 *
 * In 3D the corresponding expression is:
 * \f[
 * \int_{\partial \Omega} v \frac{\partial u}{\partial n} \dd S =
 * \frac{1}{ b^{3}}\sum_{lm} (l+1) v_{lm} u_{lm},
 * \f]
 * where
 * \f[
 * u_{lm} = \int_{\partial \Omega} Y_{lm}(\theta,\phi) \, u(b, \theta,\phi) \dd
 * S,
 * \f]
 * with the polar co-ordinates calculated relative to the boundary's centroid,
 * and similarly for \f$v_{lm}\f$. Here we use real spherical harmonics as
 * defined in Appendix B of Dahlen & Tromp (1998).
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
  /** @brief Workspace for Legendre polynomials \f$ P_l^m \f$. */
  mfem::Vector _p;
  /** @brief Workspace for Legendre polynomials \f$ P_{l-1}^m \f$. */
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
   * Computes \f$ y = A \cdot x \f$.
   * @param x The input vector (Dirichlet data).
   * @param y The output vector (Neumann data).
   */
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  /**
   * @brief Multiplies the transpose of the DtN operator matrix by a vector.
   * For this self-adjoint operator, \f$ A^T = A \f$, so it calls `Mult(x, y)`.
   * @param x The input vector.
   * @param y The output vector.
   */
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override {
    Mult(x, y);
  }

  /**
   * @brief Assembles the sparse matrix associated with the DtN operator's
   * Galerkin representation. This method needs to be called after construction
   * to build the internal `_mat`.
   */
  void Assemble();

#ifdef MFEM_USE_MPI
  /**
   * @brief Returns the associated Restriction-Action-Prolongation (RAP)
   * operator.
   * @return An `mfem::RAPOperator` object.
   */
  mfem::RAPOperator RAP() const;
#endif
};

/**
 * @brief Galerkin representation of the Multipole operator for Poisson's
 * equation on a spherical boundary. The resulting bilinear form is:
 * \f[
 * (v,f) \mapsto \int_{\partial \Omega} v \frac{\partial u}{\partial n} \dd S,
 * \f]
 * where the normal derivative \f$\partial u / \partial n \f$ is determined from
 * the force term \f$f\f$ using a Multipole expansion of the exterior
 * solution.
 *
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
  /** @brief Workspace for Legendre polynomials \f$ P_l^m \f$. */
  mfem::Vector _p;
  /** @brief Workspace for Legendre polynomials \f$ P_{l-1}^m \f$. */
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
   * Computes \f$ y = A \cdot x \f$.
   * @param x The input vector.
   * @param y The output vector.
   */
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  /**
   * @brief Multiplies the transpose of the Multipole operator matrix by a
   * vector. Computes \f$ y = A^T \cdot x \f$.
   * @param x The input vector.
   * @param y The output vector.
   */
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

  /**
   * @brief Assembles the sparse matrices associated with the Multipole
   * operator's Galerkin representation. This method needs to be called after
   * construction to build the internal `_lmat` and `_rmat`.
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
 * @brief Galerkin representation of the linearised Multipole operator for
 * Poisson's equation on a spherical boundary. The resulting bilinear form is:
 * \f[
 * (v,\bvec{u}) \mapsto \int_{\partial \Omega} v \frac{\partial u}{\partial n}
 * \dd S,
 * \f]
 * where the normal derivative \f$\partial u / \partial n \f$ is determined from
 * the displacement term \f$\bvec{u}\f$ using a Multipole expansion of the
 * exterior solution.
 *
 *
 * It inherits from `mfem::Operator` for its matrix-vector product capabilities,
 * `LegendreHelper` for spherical harmonic related computations, and
 * `SphericalMeshHelper` for managing spherical mesh properties.
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
  /** @brief Workspace for Legendre polynomials \f$ P_l^m \f$. */
  mfem::Vector _p;
  /** @brief Workspace for Legendre polynomials \f$ P_{l-1}^m \f$. */
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
   * @brief Multiplies the Linearised Multipole operator's Galerkin matrix by a
   * vector. Computes \f$ y = A \cdot x \f$.
   * @param x The input vector.
   * @param y The output vector.
   */
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  /**
   * @brief Multiplies the transpose of the Linearised Multipole operator's
   * Galerkin matrix by a vector. Computes \f$ y = A^T \cdot x \f$.
   * @param x The input vector.
   * @param y The output vector.
   */
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

  /**
   * @brief Assembles the sparse matrices associated with the Linearised
   * Multipole operator's Galerkin representation. This method needs to be
   * called after construction to build the internal `_lmat` and `_rmat`.
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
 * @brief BilinearFormIntegrator for a transformed diffusion integrator.
 *
 * This integrator implements a generalized diffusion term of the form:
 * \f[
 * \int_{\Omega} \nabla v : \mathbf{K} \nabla u \, dx
 * \f]
 * where \f$ \mathbf{K} \f$ is a transformation matrix or tensor that can be
 * specified by a scalar, vector, or matrix coefficient.
 *
 * The integrator supports:
 * - **Identity transformation**: \f$ \mathbf{K} = \mathbf{I} \f$ (default).
 * - **Radial transformation**: \f$ \mathbf{K} = q(x) \mathbf{I} \f$, where \f$
 * q(x) \f$ is a scalar coefficient.
 * - **General transformation (diagonal)**: \f$ \mathbf{K} = \text{diag}(q_1(x),
 * q_2(x), \dots) \f$, where \f$ q(x) \f$ is a vector coefficient.
 * - **General transformation (full matrix)**: \f$ \mathbf{K} \f$ is a matrix
 * coefficient.
 */
class TransformedLaplaceIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q =
      nullptr; /**< Scalar coefficient for radial transformation. */
  mfem::VectorCoefficient* QV =
      nullptr; /**< Vector coefficient for diagonal transformation. */
  mfem::MatrixCoefficient* QM =
      nullptr; /**< Matrix coefficient for general transformation. */

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix trial_dshape, test_dshape, xi, F, A,
      B; /**< Internal buffers for shape function derivatives and intermediate
            matrices during integration. */
#endif

 public:
  /**
   * @brief Constructor for the identity transformation (\f$ \mathbf{K} =
   * \mathbf{I} \f$).
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  TransformedLaplaceIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor for a radial transformation specified by a scalar
   * function (\f$ \mathbf{K} = q \mathbf{I} \f$).
   * @param q A reference to the `mfem::Coefficient` \f$ q \f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  TransformedLaplaceIntegrator(mfem::Coefficient& q,
                               const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Constructor for a general transformation specified by a
   * VectorCoefficient (diagonal \f$ \mathbf{K} \f$).
   * @param qv A reference to the `mfem::VectorCoefficient` \f$ q \f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  TransformedLaplaceIntegrator(mfem::VectorCoefficient& qv,
                               const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /**
   * @brief Constructor for a general transformation specified by a
   * MatrixCoefficient (full \f$ \mathbf{K} \f$).
   * @param qm A reference to the `mfem::MatrixCoefficient` \f$ q \f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  TransformedLaplaceIntegrator(mfem::MatrixCoefficient& qm,
                               const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QM{&qm} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account. Variations in the coefficient are not considered.
   *
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of the element level assembly for the bilinear form.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

  /**
   * @brief Assembly method when the trial and test spaces are equal.
   * @param fe The finite element for both trial and test spaces.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix(const mfem::FiniteElement& fe,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override {
    AssembleElementMatrix2(fe, fe, Trans, elmat);
  }

 protected:
  /**
   * @brief Protected method to get the default integration rule.
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param trans The element transformation.
   * @return A constant pointer to the chosen `mfem::IntegrationRule`.
   */
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

}  // namespace mfemElasticity