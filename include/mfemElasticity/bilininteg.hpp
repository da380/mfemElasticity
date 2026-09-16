#pragma once

#include <array>

#include "mfem.hpp"
#include "mfemElasticity/index.hpp"

namespace mfemElasticity {

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
  mfem::Coefficient* Q = nullptr; /**< Pointer to the scalar coefficient \f$q\f$. */

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
 *    (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} q \,\bvec{v}\cdot
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
 * \divg\bvec{u}  \dd x,
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
   * @param trial_fe The trial finite element for \f$\bvec{u}\f$.
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
 * \f[
 *   (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} q \,\bvec{v}: \deriv \bvec{u}
 * \dd x = \int_{\Omega} q\, v_{ij} \frac{\partial u_{i}}{\partial x_{j}} \dd x,
 * \f]
 * where \f$\Omega\f$ is the domain and \f$q\f$ is a scalar coefficient.
 *
 * The matrix field must be defined on a nodal finite element space formed from
 * the product of a scalar space. The ordering of the matrix components
 * corresponds to a dense matrix using column-major storage (i.e., \f$v_{00},
 * v_{10}, v_{20}, v_{01}, \dots)\f$. The vector field must be defined on a
 * nodal finite element space formed from the product of a scalar space for
 * which the gradient operator is defined. The vector and matrix fields need to
 * have compatible dimensions.
 */
class DomainMatrixDeformationGradientIntegrator
    : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q =
      nullptr; /**< Pointer to the scalar coefficient \f$q\f$. */

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
   * @param q A reference to the `mfem::Coefficient` \f$q\f$.
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
   * @param trial_fe The trial finite element for the vector field
   * \f$\bvec{u}\f$.
   * @param test_fe The test finite element for the matrix field \f$\bvec{v}\f$.
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
 * @brief BilinearFormIntegrator acting on a test symmetric matrix field,
 * \f$\bvec{v}\f$, and a trial vector field, \f$\bvec{u}\f$ according to:
 * \f[
 *   (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} q \,\bvec{v}: \deriv \bvec{u}
 * \dd x = \frac{1}{2}\int_{\Omega} q\, v_{ij} \left(\frac{\partial
 *         u_{i}}{\partial x_{j}} + \frac{\partial
 *         u_{j}}{\partial x_{i}}\right)\dd x,
 * \f]
 * where \f$\Omega\f$ is the domain and \f$q\f$ is a scalar coefficient.
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
  mfem::Coefficient* Q = nullptr; /**< Pointer to the scalar coefficient \f$q\f$. */

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
   * @param q A reference to the `mfem::Coefficient` \f$q\f$.
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
   * @param trial_fe The trial finite element for the vector field
   * \f$\bvec{u}\f$.
   * @param test_fe The test finite element for the symmetric matrix field
   * \f$\bvec{v}\f$.
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
 * field, \f$\bvec{v}\f$, and a trial vector field, \f$\bvec{u}\f$  according
 * to:
 * \f[
 *   (\bvec{v},\bvec{u}) \mapsto \int_{\Omega} q \,\bvec{v}: \deriv \bvec{u}
 * \dd x = \frac{1}{2}\int_{\Omega} q\, v_{ij} \left(
 *         \frac{\partial u_{i}}{\partial x_{j}}
 *        + \frac{\partial u_{j}}{\partial x_{i}}
 *        - \frac{2}{n}\frac{\partial u_{k}}{\partial x_{k}}\delta_{ij}
 *        \right)\dd x,
 * \f]
 * where \f$\Omega\f$ is the domain, \f$q\f$ is a scalar coefficient, and
 * \f$n\f$ the spatial dimension.
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
  mfem::Coefficient* Q = nullptr; /**< Pointer to the scalar coefficient \f$q\f$. */

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
   * @param q A reference to the `mfem::Coefficient` \f$q\f$.
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
 * @brief Bilinear form integrator for general linear elasticity,
 * \f[
 * (\bvec{u}, \bvec{v}) \mapsto \int_{\Omega} \bvec{\varepsilon}(\bvec{v})
 * : \bvec{C} : \bvec{\varepsilon}(\bvec{u}) \, \mathrm{d}x,
 * \f]
 * where \f$\bvec{\varepsilon}\f$ is the symmetric strain and \f$\bvec{C}\f$ an
 * arbitrary elasticity tensor.
 *
 * The tensor is supplied as an \f$n_s \times n_s\f$ `mfem::MatrixCoefficient`,
 * \f$n_s = d(d+1)/2\f$, in Mandel form and `SymmetricMatrixIndex` component
 * ordering (lower triangle, column-major, with off-diagonal components scaled
 * by \f$\sqrt{2}\f$). This is the representation produced by the
 * `ElasticTensorCoefficient` classes of `elastic_tensor.hpp`; sums and scalar
 * products of them through MFEM's matrix-coefficient algebra are fine too.
 *
 * The vector field must be defined on a nodal H1 finite element space with
 * `Ordering::byNODES`. The element matrix has the layout
 * `elmat(dof c + i, dof c' + i')` and the default quadrature order is
 * `2 OrderGrad(el)`, as for `mfem::ElasticityIntegrator`. Per quadrature
 * point the reduced strain-displacement matrix \f$B\f$
 * (\f$\hat{\varepsilon} = B u\f$) is formed and \f$w B^T C B\f$ added.
 * Manifold elements are not supported; in 2-D the semantics are plane strain.
 */
class ElasticTensorIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::MatrixCoefficient* C_; /**< Pointer to the elasticity tensor
                                  coefficient in Mandel form. */

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix dshape_, gshape_, B_, Cq_,
      CB_; /**< Internal buffers: reference and physical shape gradients,
              strain-displacement matrix, tensor at the point and C B. */
#endif

 public:
  /**
   * @brief Constructor for ElasticTensorIntegrator.
   * @param C The \f$n_s \times n_s\f$ Mandel-form elasticity tensor
   * coefficient.
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  explicit ElasticTensorIntegrator(mfem::MatrixCoefficient& C,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), C_(&C) {}

  /**
   * @brief Implementation of element-level calculations for the bilinear form.
   * @param el The finite element (shared by trial and test spaces).
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element stiffness
   * matrix.
   */
  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;

  /**
   * @brief Builds the reduced strain-displacement matrix \f$B\f$ (size
   * \f$n_s \times d\,\mathrm{dof}\f$) from the physical shape-function
   * gradients.
   * @param dim The spatial dimension.
   * @param gshape The physical gradients of the shape functions
   * (\f$\mathrm{dof} \times d\f$).
   * @param B The output matrix, resized as needed.
   */
  static void StrainDisplacementMatrix(int dim, const mfem::DenseMatrix& gshape,
                                       mfem::DenseMatrix& B);
};

/**
 * @brief DiscreteInterpolator that acts on a vector field, \f$\bvec{u}\f$, to
 * return the matrix field, \f$\deriv \bvec{u}\f$, with components:
 * \f[
 * (\deriv \bvec{u})_{ij} = \frac{\partial u_{i}}{\partial x_{j}},
 * \f]
 * The resulting matrix field components are stored using the column-major
 * format.
 *
 * The input vector field \f$\bvec{u}\f$ must be defined on a nodal finite
 * element space formed from the product of a scalar space on which the gradient
 * operator is defined. The output matrix field \f$\deriv \bvec{u}\f$ must be
 * defined on a nodal finite element space formed from the product of a scalar
 * space.
 */
class DeformationGradientInterpolator : public mfem::DiscreteInterpolator {
 private:
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix
      dshape; /**< Internal buffer for derivative shape functions. */
#endif

 public:
  /**
   * @brief Constructs a DeformationGradientInterpolator object.
   * This default constructor initializes the interpolator without specific
   * parameters.
   */
  DeformationGradientInterpolator() {}

  /**
   * @brief Assembles the element matrix for the interpolation of \f$\deriv
   * \bvec{u}\f$.
   *
   * This method computes the local element interpolation matrix that maps
   * degrees of freedom of the input vector field \f$u\f$ to the degrees of
   * freedom of the output matrix field \f$v = \deriv u\f$.
   *
   * @param in_fe The input finite element for the vector field.
   * @param out_fe The output finite element for the matrix field.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element interpolation
   * matrix. Its dimensions will be `out_fe.GetDof()` by `in_fe.GetDof()`.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& in_fe,
                              const mfem::FiniteElement& out_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/**
 * @brief DiscreteInterpolator that acts on a vector displacement field,
 * \f$\bvec{u}\f$, to return the symmetric strain tensor field,
 * \f$\bvec{e}\f$, with components:
 * \f[
 * e_{ij} = \frac{1}{2}\left(\frac{\partial u_{i}}{\partial x_{j}} +
 * \frac{\partial u_{j}}{\partial x_{i}}\right).
 * \f]
 *
 * The components of the symmetric matrix field \f$\bvec{e}\f$ are
 * stored in column-major format, keeping only elements in the lower triangle
 * due to symmetry.
 * - In 3D spaces, this implies the ordering: \f$e_{00}\f$,
 * \f$e_{10}\f$, \f$e_{20}\f$,
 * \f$e_{11}\f$, \f$e_{21}\f$, \f$e_{22}\f$.
 * - In 2D spaces, this implies the ordering: \f$e_{00}\f$,
 * \f$e_{10}\f$, \f$e_{11}\f$.
 *
 * The input vector field \f$\bvec{u}\f$ must be defined on a nodal finite
 * element space formed from the product of a scalar space on which the gradient
 * operator is defined. The output symmetric matrix field
 * \f$\bvec{e}\f$ must be defined on a nodal finite element space
 * formed from the product of a scalar space.
 */
class StrainInterpolator : public mfem::DiscreteInterpolator {
 private:
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix
      dshape; /**< Internal buffer for derivative shape functions. */
#endif
 public:
  /**
   * @brief Constructs a StrainInterpolator object.
   * This default constructor initializes the interpolator.
   */
  StrainInterpolator() {}

  /**
   * @brief Assembles the element matrix for the interpolation of the strain
   * tensor.
   *
   * This method computes the local element interpolation matrix that maps
   * degrees of freedom of the input displacement field \f$\bvec{u}\f$ to the
   * degrees of freedom of the output symmetric strain field \f$\bvec{e}\f$.
   *
   * @param in_fe The input finite element for the vector field.
   * @param out_fe The output finite element for the symmetric matrix field.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element interpolation
   * matrix. Its dimensions will be `out_fe.GetDof()` by `in_fe.GetDof()`.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& in_fe,
                              const mfem::FiniteElement& out_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/**
 * @brief DiscreteInterpolator that maps a vector displacement field,
 * \f$\bvec{u}\f$, into a trace-free symmetric matrix field, \f$\bvec{v}\f$.
 *
 * The components of the output trace-free symmetric matrix \f$\bvec{v}\f$ are
 * given by:
 * \f[
 * v_{ij} = \frac{1}{2}\left(\frac{\partial u_{i}}{\partial x_{j}} +
 * \frac{\partial u_{j}}{\partial x_{i}}\right) - \frac{1}{n}
 * \frac{\partial u_{k}}{\partial x_{k}} \delta_{ij}
 * \f]
 * where \f$n\f$ is the spatial dimension.
 *
 * The input vector field \f$\bvec{u}\f$ must be defined on a nodal finite
 * element space formed from the product of a scalar space on which the gradient
 * operator is defined. The output trace-free symmetric matrix field
 * \f$\bvec{v}\f$ must be defined on a nodal finite element space formed from
 * the product of a scalar space.
 */
class DeviatoricStrainInterpolator : public mfem::DiscreteInterpolator {
 private:
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix
      dshape; /**< Internal buffer for derivative shape functions. */
#endif
 public:
  /**
   * @brief Constructs a DeviatoricStrainInterpolator object.
   * This default constructor initializes the interpolator.
   */
  DeviatoricStrainInterpolator() {}

  /**
   * @brief Assembles the element matrix for the interpolation of the deviatoric
   * strain tensor.
   *
   * This method computes the local element interpolation matrix that maps
   * degrees of freedom of the input displacement field \f$u\f$ to the degrees
   * of freedom of the output trace-free symmetric matrix field \f$v\f$.
   *
   * @param in_fe The input finite element for the vector field.
   * @param out_fe The output finite element for the trace-free symmetric matrix
   * field.
   * @param Trans The element transformation.
   * @param elmat The output dense matrix representing the element interpolation
   * matrix. Its dimensions will be `out_fe.GetDof()` by `in_fe.GetDof()`.
   */
  void AssembleElementMatrix2(const mfem::FiniteElement& in_fe,
                              const mfem::FiniteElement& out_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/**
 * @brief Boundary integrator
 * \f[
 *   (\bvec{u},\bvec{v}) \mapsto \int_{\Gamma} q\,(\bvec{n}\cdot\bvec{u})
 *   (\bvec{n}\cdot\bvec{v}) \dd S,
 * \f]
 * on boundary elements, for a vector field on a nodal (H1-type) space with
 * vdim equal to the space dimension and Ordering::byNODES, and a scalar
 * coefficient \f$q\f$.
 *
 * \f$\bvec{n}\f$ is the unit normal of the boundary element obtained from
 * mfem::CalcOrtho of its transformation's Jacobian, i.e. the outward normal
 * of the mesh for consistently oriented boundary elements (the normal that
 * mfem::BoundaryNormalLFIntegrator uses). On a (Par)SubMesh this is the
 * outward normal of the submesh, on inherited and on cut boundaries alike.
 *
 * Used for the fluid–solid interface term of self-gravitating problems,
 * \f$-\int \rho_F\,(\bvec{m}\cdot\nabla\Phi_0)(\bvec{m}\cdot\bvec{u})
 * (\bvec{m}\cdot\bvec{v})\f$, with \f$q\f$ built from a
 * BoundaryNormalDotCoefficient.
 */
class BoundaryNormalNormalIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr; /**< Pointer to the coefficient \f$q\f$. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector shape, normal, nshape; /**< Internal buffers. */
#endif

 public:
  /**
   * @param q The scalar coefficient (optional; unit when null).
   * @param ir An optional pointer to an `mfem::IntegrationRule`.
   */
  explicit BoundaryNormalNormalIntegrator(
      mfem::Coefficient* q = nullptr, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{q} {}

  explicit BoundaryNormalNormalIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /** @brief Default rule: order 2 el.GetOrder() + Trans.OrderW(). */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& el, const mfem::ElementTransformation& Trans);

  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;
};

/**
 * @brief Mixed boundary integrator acting on a scalar trial field \f$p\f$
 * and a vector test field \f$\bvec{v}\f$:
 * \f[
 *   (\bvec{v}, p) \mapsto \int_{\Gamma} q\, p\,(\bvec{n}\cdot\bvec{v})
 *   \dd S,
 * \f]
 * on boundary elements, with \f$\bvec{n}\f$ the boundary element's unit
 * normal as in BoundaryNormalNormalIntegrator. The vector space must be a
 * nodal space with vdim equal to the space dimension and Ordering::byNODES.
 *
 * Its transpose (vector trial, scalar test) is obtained with
 * mfem::TransposeIntegrator. Used for the interface coupling
 * \f$-\int \rho_F\,\phi\,(\bvec{m}\cdot\bvec{v})\f$ of self-gravitating
 * fluid–solid problems.
 */
class BoundaryNormalScalarIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr; /**< Pointer to the coefficient \f$q\f$. */

#ifndef MFEM_THREAD_SAFE
  mfem::Vector trial_shape, test_shape, normal, nshape; /**< Buffers. */
#endif

 public:
  explicit BoundaryNormalScalarIntegrator(
      mfem::Coefficient* q = nullptr, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{q} {}

  explicit BoundaryNormalScalarIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /** @brief Default rule: order trial + test + Trans.OrderW(). */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

}  // namespace mfemElasticity
