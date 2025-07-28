#pragma once

#include <array>

#include "mfem.hpp"

namespace mfemElasticity {

/**
 * @brief Base class for indexing vector, matrix, and tensor fields.
 *
 * This abstract base class provides common functionalities for indexing
 * components within vector, matrix, and tensor fields  defined
 * over a finite element space. It stores the spatial dimension (`_dim`)
 * and the number of degrees of freedom per component (`_dof`).
 */
class Index {
 private:
  int _dim; /**< The spatial dimension (e.g., 2 for 2D, 3 for 3D). */
  int _dof; /**< The number of degrees of freedom per component (e.g., number of
               nodes in an element). */

 public:
  /**
   * @brief Constructor for the Index class.
   * @param dim The spatial dimension of the field.
   * @param dof The number of degrees of freedom associated with each component.
   */
  Index(int dim, int dof) : _dim{dim}, _dof{dof} {}

  /**
   * @brief Returns the spatial dimension of the field.
   * @return The dimension.
   */
  int Dim() const { return _dim; }

  /**
   * @brief Returns the number of degrees of freedom per component.
   * @return The degrees of freedom.
   */
  int Dof() const { return _dof; }

  /**
   * @brief Pure virtual method to return the number of components in the field
   * type.
   * @return The size of one component (e.g., `Dim()` for a vector,
   * `Dim()*Dim()` for a matrix).
   */
  virtual int ComponentSize() const = 0;

  /**
   * @brief Returns the total size of the field, i.e., `_dof * ComponentSize()`.
   * @return The total size of the field.
   */
  int Size() const { return _dof * ComponentSize(); }
};

/**
 * @brief Class for indexing vector fields.
 *
 * This class extends the base `Index` class to provide specific indexing
 * for vector fields, where components are typically stored contiguously
 * for each spatial dimension across all degrees of freedom.
 */
class VectorIndex : public Index {
 public:
  /**
   * @brief Constructor for the VectorIndex class.
   * @param dim The spatial dimension of the vector.
   * @param dof The number of degrees of freedom.
   */
  VectorIndex(int dim, int dof) : Index(dim, dof) {}

  /**
   * @brief Returns the offset to the start of the `j`-th component's block of
   * data.
   * @param j The component index (0, 1, ..., Dim()-1).
   * @return The offset.
   */
  int Offset(int j) const { return j * Dof(); }

  /**
   * @brief Overloaded operator to get the global index for the `i`-th node
   * and `j`-th component of the vector field.
   * @param i The node index.
   * @param j The component index.
   * @return The global index.
   */
  int operator()(int i, int j) const { return i + Offset(j); }

  /**
   * @brief Returns the number of components in a vector field, which is equal
   * to `Dim()`.
   * @return The number of components.
   */
  int ComponentSize() const override { return Dim(); }
};

/**
 * @brief Class for indexing matrix fields.
 *
 * This class extends the base `Index` class to provide specific indexing
 * for general (dense) matrix fields, assuming a column-major storage
 * for the components.
 */
class MatrixIndex : public Index {
 public:
  /**
   * @brief Constructor for the MatrixIndex class.
   * @param dim The spatial dimension of the matrix (e.g., for a Dim x Dim
   * matrix).
   * @param dof The number of degrees of freedom.
   */
  MatrixIndex(int dim, int dof) : Index(dim, dof) {}

  /**
   * @brief Returns the offset for the `(j,k)`-th component within the matrix.
   * This assumes column-major ordering: `j + Dim() * k`.
   * @param j The row index.
   * @param k The column index.
   * @return The component offset.
   */
  virtual int ComponentOffset(int j, int k) const { return j + Dim() * k; }

  /**
   * @brief Returns the offset to the `(j,k)`-th component's block of data.
   * @param j The row index.
   * @param k The column index.
   * @return The offset to the block.
   */
  int Offset(int j, int k) const { return ComponentOffset(j, k) * Dof(); }

  /**
   * @brief Overloaded operator to get the global index for the `i`-th node
   * and `(j,k)`-th component of the matrix field.
   * @param i The node index.
   * @param j The row index.
   * @param k The column index.
   * @return The global index.
   */
  int operator()(int i, int j, int k) const { return i + Offset(j, k); }

  /**
   * @brief Returns the number of components in a full matrix field, which is
   * `Dim() * Dim()`.
   * @return The number of components.
   */
  int ComponentSize() const override { return Dim() * Dim(); }
};

/**
 * @brief Class for indexing symmetric matrix fields.
 *
 * This class extends `MatrixIndex` to provide indexing specifically for
 * symmetric matrix fields. It stores only the unique components, typically
 * the lower triangle in a column-major fashion.
 */
class SymmetricMatrixIndex : public MatrixIndex {
 public:
  /**
   * @brief Constructor for the SymmetricMatrixIndex class.
   * @param dim The spatial dimension of the symmetric matrix.
   * @param dof The number of degrees of freedom.
   */
  SymmetricMatrixIndex(int dim, int dof) : MatrixIndex(dim, dof) {}

  /**
   * @brief Returns the offset for the `(j,k)`-th component within the symmetric
   * matrix.
   *
   * This method handles symmetry, ensuring that `ComponentOffset(j,k)` is the
   * same as `ComponentOffset(k,j)`. It calculates the offset assuming a storage
   * order that keeps only the unique elements (e.g., lower triangle in
   * column-major).
   *
   * @param j The row index.
   * @param k The column index.
   * @return The component offset.
   */
  int ComponentOffset(int j, int k) const override {
    if (j < k) {
      return ComponentOffset(k, j);
    } else {
      return (j + k * Dim() - k * (k + 1) / 2);
    }
  }

  /**
   * @brief Returns the number of unique components in a symmetric matrix, which
   * is `Dim() * (Dim() + 1) / 2`.
   * @return The number of components.
   */
  int ComponentSize() const override { return Dim() * (Dim() + 1) / 2; }
};

/**
 * @brief Class for indexing trace-free symmetric matrices.
 *
 * This class extends `SymmetricMatrixIndex`. The indexing is identical
 * to that for symmetric matrices, with the implicit understanding that
 * the final diagonal element (e.g., `v_{22}` in 3D) is removed from the
 * basis to enforce the trace-free condition. This removal is not
 * explicitly checked in calls to the indexing or offset functions.
 */
class TraceFreeSymmetricMatrixIndex : public SymmetricMatrixIndex {
 public:
  /**
   * @brief Constructor for the TraceFreeSymmetricMatrixIndex class.
   * @param dim The spatial dimension.
   * @param dof The number of degrees of freedom.
   */
  TraceFreeSymmetricMatrixIndex(int dim, int dof)
      : SymmetricMatrixIndex(dim, dof) {}

  /**
   * @brief Returns the number of components in a trace-free symmetric matrix.
   * This is one less than a full symmetric matrix.
   * @return The number of components.
   */
  int ComponentSize() const override {
    return SymmetricMatrixIndex::ComponentSize() - 1;
  }
};

/**
 * @brief BilinearFormIntegrator that acts on a test vector field,
 * \f$\bvec{v}\f$, and a trial scalar field, \f$u\f$ according to:
 * \f[
 *   (\mathbf{v},u) \mapsto \int_{\Omega} \mathbf{q}\cdot \mathbf{v} \, u \dd x
 * \f]
 * where \f$\Omega\f$ is the domain and \f$\bvec{q}\f$ is a vector
 * coefficient.
 *
 * It is assumed that the vector field is defined on a finite element space
 * formed from the product of a scalar nodal space.
 */
class DomainVectorScalarIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::VectorCoefficient*
      QV; /**< Pointer to the vector coefficient \f$\bvec{q}\f$. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector trial_shape, test_shape,
      qv; /**< Internal buffers for shape functions and coefficient values. */
  mfem::DenseMatrix
      part_elmat; /**< Internal buffer for partial element matrix. */
#endif

 public:
  /**
   * @brief Constructor for DomainVectorScalarIntegrator.
   *
   * To define an instance, a vector coefficient is provided along, optionally,
   * with an integration rule. The vector coefficient must return values with
   * size equal to the spatial dimension as the finite-element space.
   *
   * @param qv A reference to the `mfem::VectorCoefficient` \f$\bvec{q}\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`. If `nullptr`,
   * a default rule will be used.
   */
  DomainVectorScalarIntegrator(mfem::VectorCoefficient& qv,
                               const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

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
   *
   * @param trial_fe The trial finite element for the scalar field $u$.
   * @param test_fe The test finite element for the vector field $v$.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

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

/**
 * @brief BilinearFormIntegrator that acts on a test vector field,
 * \f$\bvec{v}\f$, and a trial scalar field, \f$u\f$ according to:
 *
 *   \f[
 *     (\bvec{v},u) \mapsto \int_{\Omega} \bvec{v} \cdot \bvec{q} \cdot \grad u
 * \dd x,
 *   \f]
 * where \f$\Omega\f$ is the domain and \f$\bvec{q}\f$ is a matrix coefficient.
 *
 * The coefficient \f$\bvec{q}\f$ can be set as a scalar, in which case the
 * matrix coefficient is proportional to the identity matrix. It can also be set
 * as a vector, this corresponding to the matrix coefficient being diagonal.
 *
 * It is assumed that the vector field is defined on a finite element space
 * formed from the product of a scalar nodal space. The scalar field must be
 * defined on a nodal space on which the gradient operator is defined.
 */
class DomainVectorGradScalarIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr; /**< Pointer to a scalar coefficient. If not
                                     null, \f$q_{ij} = Q \delta_{ij}\f$. */
  mfem::VectorCoefficient* QV = nullptr; /**< Pointer to a vector coefficient.
                                            If not null, \f$q_{ij} = QV_i
                                            \delta_{ij}\f$. */
  mfem::MatrixCoefficient* QM =
      nullptr; /**< Pointer to a matrix coefficient. If not null, \f$q_{ij}\f$
                  is directly from QM. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector test_shape, qv; /**< Internal buffers for test shape functions
                                  and vector coefficient values. */
  mfem::DenseMatrix trial_dshape, part_elmat, qm,
      tm; /**< Internal buffers for trial derivative shape functions, partial
             element matrix, and matrix coefficient values. */
#endif

 public:
  /**
   * @brief Constructor for DomainVectorGradScalarIntegrator with an identity
   * matrix coefficient.
   * @param ir An optional pointer to an `mfem::IntegrationRule`. If `nullptr`,
   * a default rule will be used.
   */
  DomainVectorGradScalarIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor for DomainVectorGradScalarIntegrator with a scalar
   * coefficient.
   *
   * The matrix coefficient is proportional to the identity matrix (\f$q_{ij} =
   * Q
   * \delta_{ij}\f$).
   *
   * @param q A reference to the `mfem::Coefficient`, Q.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorGradScalarIntegrator(mfem::Coefficient& q,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Constructor for DomainVectorGradScalarIntegrator with a vector
   * coefficient.
   *
   * The matrix coefficient is diagonal (\f$q_{ij} = QV_i \delta_{ij}\f$).
   *
   * @param qv A reference to the `mfem::VectorCoefficient`, QV.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorGradScalarIntegrator(mfem::VectorCoefficient& qv,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /**
   * @brief Constructor for DomainVectorGradScalarIntegrator with a matrix
   * coefficient.
   *
   * @param qm A reference to the `mfem::MatrixCoefficient`, q.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorGradScalarIntegrator(mfem::MatrixCoefficient& qm,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QM{&qm} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivatives. Variations in the matrix coefficient are not considered.
   *
   * @param trial_fe The trial finite element for the scalar field \f$u\f$.
   * @param test_fe The test finite element for the vector field \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of the element level assembly for the bilinear form.
   *
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
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator that acts on a test vector field,
 * \f$\bvec{v}\f$, and a trial scalar field, \f$u\f$ according to:
 * \f[
 *    (\bvec{v},u) \mapsto \int_{\Omega} q \divg \bvec{v}\, u \dd x,
 * \f]
 * where \f$\Omega\f$ is the domain and \f$q\f$ is a scalar coefficient.
 *
 * It is assumed that the vector field is defined on a finite element space
 * formed from the product of a scalar nodal space on which the gradient
 * operator is defined.
 */
class DomainDivVectorScalarIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q; /**< Pointer to the scalar coefficient \f$q\f$. */

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix test_dshape,
      part_elmat; /**< Internal buffers for test derivative shape functions and
                     partial element matrix. */
  mfem::Vector trial_shape; /**< Internal buffer for trial shape functions. */
#endif

 public:
  /**
   * @brief Constructor for DomainDivVectorScalarIntegrator.
   *
   * The scalar coefficient is taken equal to the constant 1 if no coefficient
   * is provided.
   *
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainDivVectorScalarIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{nullptr} {}

  /**
   * @brief Constructor for DomainDivVectorScalarIntegrator with a scalar
   * coefficient.
   * @param q A reference to the `mfem::Coefficient` \f$q\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainDivVectorScalarIntegrator(mfem::Coefficient& q,
                                  const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivatives. Variations in the coefficient are not considered.
   *
   * @param trial_fe The trial finite element for the scalar field \f$u\f$.
   * @param test_fe The test finite element for the vector field \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of element level calculations for the bilinear form.
   *
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
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator that acts on a test vector field,
 * \f$\bvec{v}\f$, and a trial vector field, \f$\bvec{u}\f$ according to:
 * \f[
 *    (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} q \divg \bvec{v}\, \divg
 * \bvec{u} \dd x
 * \f]
 * where \f$\Omega\f$ is the domain and \f$q\f$ is a scalar coefficient.
 *
 * It is assumed that both vector fields are defined on finite element spaces
 * formed from the product of scalar nodal spaces on which the gradient operator
 * is defined.
 */
class DomainDivVectorDivVectorIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q =
      nullptr; /**< Pointer to the scalar coefficient \f$q\f$. */

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix trial_dshape,
      test_dshape; /**< Internal buffers for trial and test derivative shape
                      functions. */
#endif

 public:
  /**
   * @brief Constructor for DomainDivVectorDivVectorIntegrator.
   *
   * The coefficient is taken to be the constant 1 if no coefficient is
   * provided.
   *
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainDivVectorDivVectorIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor for DomainDivVectorDivVectorIntegrator with a scalar
   * coefficient.
   * @param q A reference to the `mfem::Coefficient` \f$q\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainDivVectorDivVectorIntegrator(mfem::Coefficient& q,
                                     const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with two orders removed to account for the spatial
   * derivatives. Variations in the coefficient are not considered.
   *
   * @param trial_fe The trial finite element for the vector field $u$.
   * @param test_fe The test finite element for the vector field $v$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of the element level calculations for the bilinear
   * form.
   *
   * @param trial_fe The trial finite element.
   * @param test_fe The test finite element.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat);

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
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator acting on a test vector field, \f$\bvec{v}\f$,
 * and a trial vector field, \f$\bvec{u}\f$ according to:
 * \f[
 *    (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} q \bvec{v}\cdot
 *    \grad(\bvec{w}\cdot \bvec{u}) \dd x,
 * \f]
 * where \f$\Omega\f$ is the domain,  \f$q\f$ is a scalar coefficient and
 * \f$\bvec{w}\f$ is a vector coefficient.
 *
 * It is assumed that the vector fields are defined on finite element spaces
 * formed from the product of scalar nodal spaces. On the test space, the
 * gradient operator must be defined.
 */
class DomainVectorGradVectorIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q =
      nullptr; /**< Pointer to the scalar coefficient \f$q\f$. */
  mfem::VectorCoefficient*
      QV; /**< Pointer to the vector coefficient \f$\bvec{w}\f$. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector qv, test_shape; /**< Internal buffers for vector coefficient
                                  values and test shape functions. */
  mfem::DenseMatrix trial_dshape, left_elmat, rigth_elmat_trans,
      part_elmat; /**< Internal buffers for trial derivative shape functions and
                     various intermediate matrices. */
#endif

 public:
  /**
   * @brief Constructor for DomainVectorGradVectorIntegrator with a vector
   * coefficient and default scalar coefficient (1).
   * @param qv A reference to the `mfem::VectorCoefficient` \f$\bvec{w}\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorGradVectorIntegrator(mfem::VectorCoefficient& qv,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /**
   * @brief Constructor for DomainVectorGradVectorIntegrator with both vector
   * and scalar coefficients.
   * @param qv A reference to the `mfem::VectorCoefficient` \f$\bvec{w}\f$.
   * @param q A reference to the `mfem::Coefficient` \f$q\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorGradVectorIntegrator(mfem::VectorCoefficient& qv,
                                   mfem::Coefficient& q,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q}, QV{&qv} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivative. Variations in the coefficients are not considered.
   *
   * @param trial_fe The trial finite element for the vector field
   * \f$\bvec{u}\f$.
   * @param test_fe The test finite element for the vector field \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of element level calculations for the bilinear form.
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
   * @param el The finite element for both trial and test spaces.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override {
    AssembleElementMatrix2(el, el, Trans, elmat);
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
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator acting on a test vector field, \f$\bvec{v}\f$,
 * and a trial vector field, \f$\bvec{u}\f$ according to:
 * \f[
 *    (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} \bvec{q} \cdot \bvec{v}\,
 * \divg\bvec{u} \dd \dd x,
 * \f]
 * where \f$\Omega\f$ is the domain and \f$\bvec{q}\f$ is a vector coefficient.
 *
 * This integrator assumes that the test vector field \f$\bvec{v}\f$ is defined
 * on a finite element space formed from the product of a scalar nodal space,
 * and the trial vector field \f$\bvec{u}\f$ is defined on a finite element
 * space formed from the product of a scalar nodal space on which the gradient
 * operator is defined.
 */
class DomainVectorDivVectorIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::VectorCoefficient* QV =
      nullptr; /**< Pointer to the vector coefficient \f$q\f$. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector qv, test_shape; /**< Internal buffers for vector coefficient
                                  values and test shape functions. */
  mfem::DenseMatrix trial_dshape,
      part_elmat; /**< Internal buffers for trial derivative shape functions and
                     partial element matrix. */
#endif

 public:
  /**
   * @brief Constructor for DomainVectorDivVectorIntegrator.
   * @param qv A reference to the `mfem::VectorCoefficient` \f$q\f$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainVectorDivVectorIntegrator(mfem::VectorCoefficient& qv,
                                  const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivative. Variations in the coefficients are not considered.
   *
   * @param trial_fe The trial finite element for the vector field
\f$\bvec{u}\f$.
   * @param test_fe The test finite element for the vector field \f$\bvec{v}\f$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of element level calculations for the bilinear form.
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
   * @param el The finite element for both trial and test spaces.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override {
    AssembleElementMatrix2(el, el, Trans, elmat);
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
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator acting on a test matrix field, \f$\bf{v}\f$,
 * and a trial vector field, \f$\bvec{u}\f$ according to:
 * \f[ (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} q \bvec{v}: \deriv \bvec{u} \dd
 * x,
 * \f]
 * where $\Omega$ is the domain and $q$ is a scalar coefficient.
 *
 * The matrix field must be defined on a nodal finite element space formed from
 * the product of a scalar space. The ordering of the matrix components
 * corresponds to a dense matrix using column-major storage (i.e., $v_{00}$,
 * $v_{10}$, $v_{20}$, $v_{01}$, ...). The vector field must be defined on a
 * nodal finite element space formed from the product of a scalar space for
 * which the gradient operator is defined. The vector and matrix fields need to
 * have compatible dimensions.
 */
class DomainMatrixDeformationGradientIntegrator
    : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr; /**< Pointer to the scalar coefficient $q$. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector test_shape; /**< Internal buffer for test shape functions. */
  mfem::DenseMatrix trial_dshape,
      part_elmat; /**< Internal buffers for trial derivative shape functions and
                     partial element matrix. */
#endif

 public:
  /**
   * @brief Constructor for DomainMatrixDeformationGradientIntegrator.
   *
   * The scalar coefficient is taken equal to the constant 1 if no coefficient
   * is provided.
   *
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainMatrixDeformationGradientIntegrator(
      const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor for DomainMatrixDeformationGradientIntegrator with a
   * scalar coefficient.
   * @param q A reference to the `mfem::Coefficient` $q$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainMatrixDeformationGradientIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivative. Variations in the coefficients are not considered.
   *
   * @param trial_fe The trial finite element for the vector field $u$.
   * @param test_fe The test finite element for the matrix field $v$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of element-level calculations for the bilinear form.
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
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator acting on a test symmetric matrix field, $v$,
 * and a trial vector field, $u$.
 *
 * The integration is defined as:
 * $$ (v,u) \mapsto \frac{1}{2}\int_{\Omega} q v_{ij} (u_{i,j} + u_{j,i}) \, dx
 * $$ where $\Omega$ is the domain and $q$ is a scalar coefficient.
 *
 * The matrix field must be defined on a nodal finite element space formed from
 * the product of a scalar space. The ordering of the matrix components
 * corresponds to a dense matrix using column-major storage but storing only the
 * lower triangle. The vector field must be defined on a nodal finite element
 * space formed from the product of a scalar space for which the gradient
 * operator is defined. The vector and matrix fields need to have compatible
 * dimensions.
 */
class DomainSymmetricMatrixStrainIntegrator
    : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q; /**< Pointer to the scalar coefficient $q$. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector test_shape; /**< Internal buffer for test shape functions. */
  mfem::DenseMatrix part_elmat,
      trial_dshape; /**< Internal buffers for partial element matrix and trial
                       derivative shape functions. */
#endif

 public:
  /**
   * @brief Constructor for DomainSymmetricMatrixStrainIntegrator.
   *
   * The scalar coefficient is taken equal to the constant 1 if no coefficient
   * is provided.
   *
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainSymmetricMatrixStrainIntegrator(
      const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor for DomainSymmetricMatrixStrainIntegrator with a scalar
   * coefficient.
   * @param q A reference to the `mfem::Coefficient` $q$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainSymmetricMatrixStrainIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivative. Variations in the coefficients are not considered.
   *
   * @param trial_fe The trial finite element for the vector field $u$.
   * @param test_fe The test finite element for the symmetric matrix field $v$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of element-level calculations for the bilinear form.
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
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief BilinearFormIntegrator acting on a test trace-free symmetric matrix
 * field, $v$, and a trial vector field, $u$.
 *
 * The integration is defined as:
 * $$ (v,u) \mapsto \frac{1}{2}\int_{\Omega} q v_{ij} (u_{i,j} + u_{j,i} -
 * (2/\text{dim}) u_{k,k}\delta_{ij}) \, dx $$ where $\Omega$ is the domain and
 * $q$ is a scalar coefficient.
 *
 * The matrix field must be defined on a nodal finite element space formed from
 * the product of a scalar space. The ordering of the matrix components
 * corresponds to a dense matrix using column-major storage but storing only the
 * lower triangle and explicitly handling the trace-free nature. The vector
 * field must be defined on a nodal finite element space formed from the product
 * of a scalar space for which the gradient operator is defined. The vector and
 * matrix fields need to have compatible dimensions.
 */
class DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator
    : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q; /**< Pointer to the scalar coefficient $q$. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector test_shape; /**< Internal buffer for test shape functions. */
  mfem::DenseMatrix part_elmat,
      trial_dshape; /**< Internal buffers for partial element matrix and trial
                       derivative shape functions. */
#endif

 public:
  /**
   * @brief Constructor for
   * DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator.
   *
   * The scalar coefficient is taken equal to the constant 1 if no coefficient
   * is provided.
   *
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator(
      const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor for
   * DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator with a scalar
   * coefficient.
   * @param q A reference to the `mfem::Coefficient` $q$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Sets the default integration rule.
   *
   * The orders of the trial space, test space, and element transformation are
   * taken into account, with one order removed to account for the spatial
   * derivative. Variations in the coefficients are not considered.
   *
   * @param trial_fe The trial finite element for the vector field $u$.
   * @param test_fe The test finite element for the trace-free symmetric matrix
   * field $v$.
   * @param Trans The element transformation.
   * @return A constant reference to the chosen `mfem::IntegrationRule`.
   */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /**
   * @brief Implementation of element-level calculations for the bilinear form.
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
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/**
 * @brief DiscreteInterpolator that acts on a vector field, $u$, to return the
 * matrix field, $v_{ij} = u_{i,j}$.
 *
 * The matrix field components are stored using the column-major format.
 *
 * The input vector field must be defined on a nodal finite element space formed
 * from the product of a scalar space on which the gradient operator is defined.
 * The output matrix field must be defined on a nodal finite element space
 * formed from the product of a scalar space.
 */
class DeformationGradientInterpolator : public mfem::DiscreteInterpolator {
 private:
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix
      dshape; /**< Internal buffer for derivative shape functions. */
#endif

 public:
  /**
   * @brief Constructor for DeformationGradientInterpolator.
   */
  DeformationGradientInterpolator() {}

  /**
   * @brief Assembles the element matrix for the interpolation.
   * @param in_fe The input finite element for the vector field $u$.
   * @param out_fe The output finite element for the matrix field $v$.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element interpolation
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& in_fe,
                              const mfem::FiniteElement& out_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/**
 * @brief DiscreteInterpolator that acts on a vector field, $u$, to return the
 * symmetric matrix field, $v_{ij} = (u_{i,j} + u_{j,i})/2$.
 *
 * The matrix field components are stored in column-major format but only
 * keeping elements in the lower triangle.
 * - In 3D spaces, this implies the ordering: $v_{00}$, $v_{10}$, $v_{20}$,
 * $v_{11}$, $v_{21}$, $v_{22}$.
 * - In 2D spaces, this implies the ordering: $v_{00}$, $v_{10}$, $v_{11}$.
 *
 * The input vector field must be defined on a nodal finite element space formed
 * from the product of a scalar space on which the gradient operator is defined.
 * The output symmetric matrix field must be defined on a nodal finite element
 * space formed from the product of a scalar space.
 */
class StrainInterpolator : public mfem::DiscreteInterpolator {
 private:
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix
      dshape; /**< Internal buffer for derivative shape functions. */
#endif
 public:
  /**
   * @brief Constructor for StrainInterpolator.
   */
  StrainInterpolator() {}

  /**
   * @brief Assembles the element matrix for the interpolation.
   * @param in_fe The input finite element for the vector field $u$.
   * @param out_fe The output finite element for the symmetric matrix field $v$.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element interpolation
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& in_fe,
                              const mfem::FiniteElement& out_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/**
 * @brief DiscreteInterpolator that maps a vector field, $u$, into a trace-free
 * symmetric matrix field, $v$.
 *
 * The components of the output trace-free symmetric matrix $v$ are given by:
 * $$ v_{ij} = (u_{i,j} + u_{j,i})/2 - (1/\text{dim}) \cdot u_{k,k} \delta_{ij}
 * $$ where $u_{k,k}$ is the trace of the gradient of $u$ (i.e., the divergence
 * of $u$).
 *
 * The input vector field must be defined on a nodal finite element space formed
 * from the product of a scalar space on which the gradient operator is defined.
 * The output trace-free symmetric matrix field must be defined on a nodal
 * finite element space formed from the product of a scalar space.
 */
class DeviatoricStrainInterpolator : public mfem::DiscreteInterpolator {
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix
      dshape; /**< Internal buffer for derivative shape functions. */
#endif
 public:
  /**
   * @brief Constructor for DeviatoricStrainInterpolator.
   */
  DeviatoricStrainInterpolator() {}

  /**
   * @brief Assembles the element matrix for the interpolation.
   * @param in_fe The input finite element for the vector field $u$.
   * @param out_fe The output finite element for the trace-free symmetric matrix
   * field $v$.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element interpolation
   * matrix.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& in_fe,
                              const mfem::FiniteElement& out_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/**
 * @brief BilinearFormIntegrator for a transformed diffusion integrator.
 *
 * This integrator implements a generalized diffusion term of the form:
 * $$ \int_{\Omega} \nabla v : \mathbf{K} \nabla u \, dx $$
 * where $\mathbf{K}$ is a transformation matrix or tensor that can be
 * specified by a scalar, vector, or matrix coefficient.
 *
 * The integrator supports:
 * - **Identity transformation**: $\mathbf{K} = \mathbf{I}$ (default).
 * - **Radial transformation**: $\mathbf{K} = q(x) \mathbf{I}$, where $q(x)$ is
 * a scalar coefficient.
 * - **General transformation (diagonal)**: $\mathbf{K} = \text{diag}(q_1(x),
 * q_2(x), \dots)$, where $q(x)$ is a vector coefficient.
 * - **General transformation (full matrix)**: $\mathbf{K}$ is a matrix
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
   * @brief Constructor for the identity transformation (K = I).
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  TransformedLaplaceIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /**
   * @brief Constructor for a radial transformation specified by a scalar
   * function ($K = q I$).
   * @param q A reference to the `mfem::Coefficient` $q$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  TransformedLaplaceIntegrator(mfem::Coefficient& q,
                               const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /**
   * @brief Constructor for a general transformation specified by a
   * VectorCoefficient (diagonal K).
   * @param qv A reference to the `mfem::VectorCoefficient` $q$.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  TransformedLaplaceIntegrator(mfem::VectorCoefficient& qv,
                               const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /**
   * @brief Constructor for a general transformation specified by a
   * MatrixCoefficient (full K).
   * @param qm A reference to the `mfem::MatrixCoefficient` $q$.
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