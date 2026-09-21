/**
 * @file poisson.hpp
 * @brief Exterior boundary conditions for Poisson's equation on a domain
 * with a spherical (circular in 2-D) outer boundary: the Dirichlet-to-Neumann
 * operator, and the multipole operators (and their linearisation about a
 * mapped geometry) coupling a source region to that boundary.
 */

#pragma once

#include <cassert>
#include <cmath>
#include <memory>

#include "mfemElasticity/mesh.hpp"
#include "mfemElasticity/spherical_harmonics.hpp"

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
 * a `SurfaceHarmonics` basis for the harmonic expansions, and
 * `SphericalMeshHelper` for managing spherical mesh properties.
 *
 * The implementation considers both 2D (circular) and 3D (spherical) cases.
 */
class PoissonDtNOperator : public mfem::Operator,
                           protected SphericalMeshHelper {
 private:
  /** @brief Pointer to the finite element space on which the operator acts. */
  mfem::FiniteElementSpace* fes_;
  /** @brief Spatial dimension of the problem (2 for 2D, 3 for 3D). */
  int dim_;
  /** @brief Harmonic degree of the expansion */
  int degree_;
  /** @brief The harmonics about the boundary centroid, up to degree_. */
  SurfaceHarmonics basis_;
  /** @brief Number of harmonic coefficients, basis_.Size(). */
  int coeff_dim_;
  /** @brief The sparse matrix representing the assembled DtN operator. */
  mfem::SparseMatrix mat_;

#ifdef MFEM_USE_MPI
  /** @brief Flag indicating if the operator is used in a parallel context. */
  bool parallel_ = false;
  /** @brief Pointer to the parallel finite element space (if in parallel). */
  mfem::ParFiniteElementSpace* pfes_;
  /** @brief MPI communicator used for parallel operations. */
  MPI_Comm comm_;

  /** @brief Communicator for ranks owning the relevant boundary. */
  MPI_Comm bdr_comm_;
  /** @brief Global rank of the root processor in bdr_comm_. */
  int bdr_root_rank_;
  /** @brief True if this rank is part of the bdr_comm_. */
  bool has_boundary_;

#endif

#ifndef MFEM_THREAD_SAFE
  /** @brief Workspace for the harmonic coefficients in Mult(). */
  mutable mfem::Vector c_;
#endif

  /**
   * @brief Common setup routine called by both serial and parallel
   * constructors: finds the spherical boundary and its centroid.
   */
  void SetUp();

  /**
   * @brief Element matrix of the factor @f$C@f$ of the operator,
   * @f$B = C C^T@f$: the pairing of the boundary element's shape functions
   * with the weighted harmonics.
   *
   * @param fe The (boundary) finite element.
   * @param Trans The element transformation.
   * @param elmat The element matrix, dofs by harmonic coefficients.
   */
  void AssembleElementMatrix(const mfem::FiniteElement& fe,
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
   * @brief The harmonic coefficients of a field on the boundary,
   * @f$c_i = b^{1-d}\int_{\partial\Omega} u\,Y_i \dd S@f$, in the ordering
   * and normalisation of SurfaceHarmonics (as returned by
   * BoundaryHarmonicCoefficients, which serves any spherical boundary). The
   * 2-D operator has no degree-zero term, and that coefficient is returned
   * as zero.
   * @param x The input vector (Dirichlet data).
   * @param y Harmonic coefficients, sized to the basis.
   */
  void HarmonicCoefficients(const mfem::Vector& x, mfem::Vector& y) const;

  /**
   * @brief Assembles the sparse matrix associated with the DtN operator's
   * Galerkin representation. This method needs to be called after construction
   * to build the internal `mat_`.
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

  mfem::real_t BoundaryRadius() const { return bdr_radius_; }

  mfem::Vector Centroid() const { return x0_; }
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
 * a `SurfaceHarmonics` basis for the harmonic expansions, and
 * `SphericalMeshHelper` for managing spherical mesh properties.
 */
class PoissonMultipoleOperator : public mfem::Operator,
                                 protected SphericalMeshHelper {
 private:
  /** @brief Pointer to the trial finite element space. */
  mfem::FiniteElementSpace* tr_fes_;
  /** @brief Pointer to the test finite element space. */
  mfem::FiniteElementSpace* te_fes_;
  /** @brief Spatial dimension of the problem (2 for 2D, 3 for 3D). */
  int dim_;
  /** @brief Polynomial degree of the finite element spaces. */
  int degree_;
  /** @brief The harmonics about the boundary centroid, up to degree_. */
  SurfaceHarmonics basis_;
  /** @brief Number of harmonic coefficients, basis_.Size(). */
  int coeff_dim_;
  /** @brief Marker array indicating which domain attributes are included in the
   * operation. */
  mfem::Array<int> dom_marker_;
  /** @brief The sparse matrix for the left-hand side contribution of the
   * operator. */
  mfem::SparseMatrix lmat_;
  /** @brief The sparse matrix for the right-hand side contribution of the
   * operator. */
  mfem::SparseMatrix rmat_;

#ifdef MFEM_USE_MPI
  /** @brief Flag indicating if the operator is used in a parallel context. */
  bool parallel_ = false;
  /** @brief Pointer to the parallel trial finite element space (if in
   * parallel). */
  mfem::ParFiniteElementSpace* tr_pfes_;
  /** @brief Pointer to the parallel test finite element space (if in parallel).
   */
  mfem::ParFiniteElementSpace* te_pfes_;
  /** @brief MPI communicator used for parallel operations. */
  MPI_Comm comm_;
#endif

#ifndef MFEM_THREAD_SAFE
  /** @brief Workspace for the harmonic coefficients in Mult(). */
  mutable mfem::Vector c_;
#endif

  /**
   * @brief Common setup routine called by both serial and parallel
   * constructors: finds the spherical boundary and its centroid.
   */
  void SetUp();

  /**
   * @brief Element matrix of the left factor: the pairing of a boundary
   * element's shape functions with the harmonics.
   *
   * @param fe The (boundary) finite element of the test space.
   * @param Trans The element transformation.
   * @param elmat The element matrix, dofs by harmonic coefficients.
   */
  void AssembleLeftElementMatrix(const mfem::FiniteElement& fe,
                                 mfem::ElementTransformation& Trans,
                                 mfem::DenseMatrix& elmat);

  /**
   * @brief Element matrix of the right factor: the pairing of a domain
   * element's shape functions with the interior harmonics.
   *
   * @param fe The finite element of the trial space.
   * @param Trans The element transformation.
   * @param elmat The element matrix, dofs by harmonic coefficients.
   */
  void AssembleRightElementMatrix(const mfem::FiniteElement& fe,
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
   * construction to build the internal `lmat_` and `rmat_`.
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
 * a `SurfaceHarmonics` basis for the harmonic expansions, and
 * `SphericalMeshHelper` for managing spherical mesh properties.
 */

class PoissonLinearisedMultipoleOperator : public mfem::Operator,
                                           protected SphericalMeshHelper {
 private:
  /** @brief Pointer to the trial finite element space. */
  mfem::FiniteElementSpace* tr_fes_;
  /** @brief Pointer to the test finite element space. */
  mfem::FiniteElementSpace* te_fes_;
  /** @brief Pointer to mfem::Coefficient for the density */
  mfem::Coefficient* density_ = nullptr;
  /** @brief Spatial dimension of the problem (2 for 2D, 3 for 3D). */
  int dim_;
  /** @brief Polynomial degree of the finite element spaces. */
  int degree_;
  /** @brief The harmonics about the boundary centroid, up to degree_. */
  SurfaceHarmonics basis_;
  /** @brief Number of harmonic coefficients, basis_.Size(). */
  int coeff_dim_;
  /** @brief Marker array indicating which domain attributes are included in the
   * operation. */
  mfem::Array<int> dom_marker_;
  /** @brief The sparse matrix for the left-hand side contribution of the
   * operator. */
  mfem::SparseMatrix lmat_;
  /** @brief The sparse matrix for the right-hand side contribution of the
   * operator. */
  mfem::SparseMatrix rmat_;

#ifdef MFEM_USE_MPI
  /** @brief Flag indicating if the operator is used in a parallel context. */
  bool parallel_ = false;
  /** @brief Pointer to the parallel trial finite element space (if in
   * parallel). */
  mfem::ParFiniteElementSpace* tr_pfes_;
  /** @brief Pointer to the parallel test finite element space (if in parallel).
   */
  mfem::ParFiniteElementSpace* te_pfes_;
  /** @brief MPI communicator used for parallel operations. */
  MPI_Comm comm_;
#endif

#ifndef MFEM_THREAD_SAFE
  /** @brief Workspace for the harmonic coefficients in Mult(). */
  mutable mfem::Vector c_;
#endif

  /**
   * @brief Common setup routine called by both serial and parallel
   * constructors: finds the spherical boundary and its centroid.
   */
  void SetUp();

  /**
   * @brief Element matrix of the left factor: the pairing of a boundary
   * element's shape functions with the harmonics.
   *
   * @param fe The (boundary) finite element of the test space.
   * @param Trans The element transformation.
   * @param elmat The element matrix, dofs by harmonic coefficients.
   */
  void AssembleLeftElementMatrix(const mfem::FiniteElement& fe,
                                 mfem::ElementTransformation& Trans,
                                 mfem::DenseMatrix& elmat);

  /**
   * @brief Element matrix of the right factor: the pairing of a domain
   * element's shape functions with the gradients of the interior
   * harmonics, weighted by the density if one is given.
   *
   * @param fe The finite element of the trial space.
   * @param Trans The element transformation.
   * @param elmat The element matrix, vdofs (component-blocked) by harmonic
   * coefficients.
   */
  void AssembleRightElementMatrix(const mfem::FiniteElement& fe,
                                  mfem::ElementTransformation& Trans,
                                  mfem::DenseMatrix& elmat);

 public:
  /**
   * @brief Constructs a serial PoissonLinearisedMultipoleOperator.
   * @param tr_fes Pointer to the trial finite element space.
   * @param te_fes Pointer to the test finite element space.
   * @param density Reference to an mfem::Coefficient for the equilibrium
   * density.
   * @param degree The polynomial degree of the FE spaces.
   * @param dom_marker An `mfem::Array<int>` marking which domain attributes
   * (1 for inclusion, 0 for exclusion) to consider for assembly.
   */
  PoissonLinearisedMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                                     mfem::FiniteElementSpace* te_fes,
                                     mfem::Coefficient& density, int degree,
                                     const mfem::Array<int>& dom_marker);

  /**
   * @brief Constructs a serial PoissonLinearisedMultipoleOperator. This
   * overload doesn't take in a density coefficient, with the desnsity
   * defaulting to the constant field with value equal to one.
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
   * @brief Constructs a serial PoissonLinearisedMultipoleOperator. This
   * overload automatically uses `AllDomainsMarker` from `tr_fes->GetMesh()`
   * to include all domain attributes in the assembly.
   * @param tr_fes Pointer to the trial finite element space.
   * @param te_fes Pointer to the test finite element space.
   * @param density Reference to an mfem::Coefficient for the equilibrium
   * density.
   * @param degree The polynomial degree of the FE spaces.
   */
  PoissonLinearisedMultipoleOperator(mfem::FiniteElementSpace* tr_fes,
                                     mfem::FiniteElementSpace* te_fes,
                                     mfem::Coefficient& density, int degree)
      : PoissonLinearisedMultipoleOperator(
            tr_fes, te_fes, density, degree,
            AllDomainsMarker(tr_fes->GetMesh())) {}

  /**
   * @brief Constructs a serial PoissonLinearisedMultipoleOperator for all
   * domains. This overload automatically uses `AllDomainsMarker` from
   * `tr_fes->GetMesh()` to include all domain attributes in the assembly,
   * and also uses the default value for density.
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
   * @param density Reference to mfem::Coefficient for the density.
   * @param degree The polynomial degree of the FE spaces.
   * @param dom_marker An `mfem::Array<int>` marking which domain attributes
   * (1 for inclusion, 0 for exclusion) to consider for assembly.
   */
  PoissonLinearisedMultipoleOperator(MPI_Comm comm,
                                     mfem::ParFiniteElementSpace* tr_fes,
                                     mfem::ParFiniteElementSpace* te_fes,
                                     mfem::Coefficient& density, int degree,
                                     const mfem::Array<int>& dom_marker);

  /**
   * @brief Constructs a parallel PoissonLinearisedMultipoleOperator.
   * This overload uses the default density which is equal to the constant
   * field with value one.
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
   * @brief Constructs a parallel PoissonLinearisedMultipoleOperator. This
   * overload automatically uses `AllDomainsMarker` from `tr_fes->GetMesh()`
   * to include all domain attributes in the assembly across all processors.
   * @param comm The MPI communicator.
   * @param tr_fes Pointer to the parallel trial finite element space.
   * @param te_fes Pointer to the parallel test finite element space.
   * @param density Reference to mfem::Coefficient for the density.
   * @param degree The polynomial degree of the FE spaces.
   */
  PoissonLinearisedMultipoleOperator(MPI_Comm comm,
                                     mfem::ParFiniteElementSpace* tr_fes,
                                     mfem::ParFiniteElementSpace* te_fes,
                                     mfem::Coefficient& density, int degree)
      : PoissonLinearisedMultipoleOperator(
            comm, tr_fes, te_fes, density, degree,
            AllDomainsMarker(tr_fes->GetMesh())) {}

  /**
   * @brief Constructs a parallel PoissonLinearisedMultipoleOperator for all
   * domains. This overload automatically uses `AllDomainsMarker` from
   * `tr_fes->GetMesh()` to include all domain attributes in the assembly across
   * all processor, and uses the default density value.
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
   * called after construction to build the internal `lmat_` and `rmat_`.
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
