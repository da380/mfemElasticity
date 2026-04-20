/**
 * @file DomainLFDeformationGradientIntegrator.hpp
 * @brief Defines a LinearFormIntegrator for terms involving the deformation
 * gradient.
 */
#pragma once

#include "mfem.hpp"

namespace mfemElasticity {

/**
 * @brief A LinearFormIntegrator that evaluates an integral involving a matrix
 * coefficient and the deformation gradient of a vector field.
 *
 * This integrator computes the integral:
 * \f[
 * \bvec{u} \mapsto \int_{\Omega} \bvec{m} : \deriv \bvec{u} \dd x =
 * \int_{\Omega} m_{ij} \frac{\partial u_{i}}{\partial x_{j}} \dd x,
 * \f]
 * where \f$\Omega\f$ is the computational domain, \f$\bvec{u}\f$ is a vector
 * field,
 * \f$\deriv \bvec{u}\f$ is its deformation gradient with components
 * \f$\frac{\partial u_{i}}{\partial x_{j}}\f$, and \f$\bvec{m}\f$ is a given
 * matrix coefficient with components \f$m_{ij}\f$.
 *
 * It is assumed that \f$\bvec{u}\f$ is defined on a product of scalar finite
 * element spaces for which the gradient operator is well-defined.
 *
 * It is also assumed that the matrix coefficient \f$\bvec{m}\f$ is square with
 * its dimension equal to the spatial dimension of the finite-element space.
 *
 * @note TODO: Extension to allow for delta-function coefficients.
 */
class DomainLFDeformationGradientIntegrator
    : public mfem::LinearFormIntegrator {
 private:
  /** @brief The matrix coefficient \f$\bvec{m}\f$ (components \f$m_{ij}\f$)
   * used in the integral. */
  mfem::MatrixCoefficient& _M;

#ifndef MFEM_THREAD_SAFE
  /** @brief Workspace vector for element vector computations (non-thread-safe).
   */
  mfem::Vector v;
  /** @brief Workspace for the shape function gradients (non-thread-safe). */
  mfem::DenseMatrix dshape;
  /** @brief Workspace for the coefficient matrix evaluation (non-thread-safe).
   */
  mfem::DenseMatrix m;
#endif

 public:
  /**
   * @brief Constructs a DomainLFDeformationGradientIntegrator.
   * @param M The matrix coefficient \f$\bvec{M}\f$ used in the integral.
   * @param ir An optional integration rule. If `nullptr`, a default rule
   * will be chosen based on the element type and order.
   */
  DomainLFDeformationGradientIntegrator(
      mfem::MatrixCoefficient& M, const mfem::IntegrationRule* ir = nullptr)
      : mfem::LinearFormIntegrator(ir), _M{M} {}

  /**
   * @brief Assembles the element vector for a given finite element.
   *
   * This method calculates the local contribution of the integral
   * \f$\int_T m_{ij} \frac{\partial u_{i}}{\partial x_{j}} dx\f$ over an
   * element \f$T\f$ to the right-hand side vector.
   *
   * @param el The finite element for which to assemble the element vector.
   * @param Trans The element transformation mapping reference coordinates to
   * physical coordinates.
   * @param elvect The output element vector to which the contribution is added.
   * Its size must be `el.GetDof()`.
   */
  void AssembleRHSElementVect(const mfem::FiniteElement& el,
                              mfem::ElementTransformation& Trans,
                              mfem::Vector& elvect) override;

  /**
   * @brief Inherit other overloads of AssembleRHSElementVect from base class.
   * This ensures that other base-class `AssembleRHSElementVect` methods (if
   * any) are also accessible.
   */
  using mfem::LinearFormIntegrator::AssembleRHSElementVect;
};

}  // namespace mfemElasticity
