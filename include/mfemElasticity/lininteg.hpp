#pragma once

#include "mfem.hpp"

namespace mfemElasticity {

/**
 * @brief A LinearFormIntegrator that acts on vector fields, $u$.
 *
 * This integrator computes the contribution to a linear form from the integral
 * term $\int_{\Omega} m_{ij} u_{i,j} dx$, where $\Omega$ is the domain,
 * $m_{ij}$ is a matrix coefficient, and $u_{i,j}$ denotes the $j$-th component
 * of the gradient of the $i$-th component of the vector field $u$.
 *
 * It is assumed that $u$ is defined on a product of scalar finite element
 * spaces for which the gradient operator is well-defined.
 *
 * It is also assumed that the matrix coefficient $m$ is square with its
 * dimension equal to the spatial dimension of the finite-element space.
 *
 * @note TODO: Extension to allow for delta-function coefficients.
 */
class DomainLFDeformationGradientIntegrator
    : public mfem::LinearFormIntegrator {
 private:
  /** @brief The matrix coefficient $m_{ij}$ used in the integral. */
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
   * @param M The matrix coefficient $m_{ij}$ used in the integral.
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
   * $\int_T m_{ij} u_{i,j} dx$ over an element $T$ to the right-hand side
   * vector.
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
