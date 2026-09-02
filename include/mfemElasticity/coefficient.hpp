/**
 * @file coefficient.hpp
 * @brief Coefficients used by the self-gravitating fluid–solid problems: the
 * normal component of a vector coefficient on boundary elements, and the
 * barotropic density gradient @f$d\rho/d\Phi_0@f$ of a fluid.
 */

#pragma once

#include <memory>

#include "mfem.hpp"

namespace mfemElasticity {

/**
 * @brief @f$\mathbf{V}\cdot\mathbf{n}@f$ on boundary elements, with
 * @f$\mathbf{n}@f$ the boundary element's unit normal (mfem::CalcOrtho of the
 * boundary transformation's Jacobian, normalised: the outward normal for
 * consistently oriented boundary elements, on a SubMesh the submesh's
 * outward normal).
 *
 * Only defined for boundary-element transformations (ElementType ==
 * BDR_ELEMENT); evaluation on any other transformation aborts. The vector
 * coefficient is evaluated on the boundary transformation, which is fine for
 * mfem::GradientGridFunctionCoefficient (MFEM evaluates the gradient in the
 * adjacent element) and for coefficients of position.
 *
 * With @f$\mathbf{V} = \nabla\Phi_0@f$ this gives @f$\mathbf{m}\cdot
 * \nabla\Phi_0 = \pm g@f$ on a fluid–solid interface, the sign selecting
 * between a fluid below and a fluid above the solid.
 */
class BoundaryNormalDotCoefficient : public mfem::Coefficient {
 public:
  explicit BoundaryNormalDotCoefficient(mfem::VectorCoefficient& V) : V_(&V) {}

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  mfem::VectorCoefficient* V_;
  mfem::Vector v_, n_;
};

/**
 * @brief The barotropic density gradient of a hydrostatic fluid,
 * @f[
 *   \rho'_F = \frac{d\rho}{d\Phi_0}
 *           = \frac{\nabla\rho\cdot\nabla\Phi_0}{|\nabla\Phi_0|^2}
 *           = g^{-1}\,\partial_r\rho \quad\text{(radial models)},
 * @f]
 * built from any two vector coefficients for @f$\nabla\rho@f$ and
 * @f$\nabla\Phi_0@f$, or from a density and a background potential given as
 * grid functions on the same mesh (their gradients are element-local, so a
 * discontinuous density in an L2 space is handled cleanly). Returns zero
 * where @f$|\nabla\Phi_0|@f$ vanishes.
 *
 * The coefficient @f$\rho'_F \phi\phi'@f$ is the fluid mass term of the
 * hydrostatic Poisson equation, eq. (2.8) of Al-Attar & Tromp (2014); it is
 * negative wherever density increases downward. Analytic radial models can
 * supply @f$g^{-1}\partial_r\rho@f$ directly instead of using this class.
 */
class BarotropicDensityGradientCoefficient : public mfem::Coefficient {
 public:
  /** @brief From the two gradients; neither is owned. */
  BarotropicDensityGradientCoefficient(mfem::VectorCoefficient& grad_rho,
                                       mfem::VectorCoefficient& grad_phi0);

  /** @brief From a density and a background potential on the same mesh
   * (scalar grid functions; the gradient coefficients are owned here). */
  BarotropicDensityGradientCoefficient(const mfem::GridFunction& rho,
                                       const mfem::GridFunction& phi0);

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  std::unique_ptr<mfem::GradientGridFunctionCoefficient> owned_rho_,
      owned_phi0_;
  mfem::VectorCoefficient* grad_rho_;
  mfem::VectorCoefficient* grad_phi0_;
  mfem::Vector gr_, gp_;
};

}  // namespace mfemElasticity
