/**
 * @file coefficient.cpp
 * @brief Implementation of the coefficients in coefficient.hpp.
 */

#include "mfemElasticity/coefficient.hpp"

namespace mfemElasticity {

using namespace mfem;

real_t BoundaryNormalDotCoefficient::Eval(ElementTransformation& T,
                                          const IntegrationPoint& ip) {
  MFEM_VERIFY(T.ElementType == ElementTransformation::BDR_ELEMENT,
              "BoundaryNormalDotCoefficient: only defined on boundary "
              "elements.");
  T.SetIntPoint(&ip);
  n_.SetSize(T.GetSpaceDim());
  CalcOrtho(T.Jacobian(), n_);
  const real_t nrm = n_.Norml2();
  if (nrm <= 0.0) {
    return 0.0;
  }
  V_->Eval(v_, T, ip);
  return (v_ * n_) / nrm;
}

BarotropicDensityGradientCoefficient::BarotropicDensityGradientCoefficient(
    VectorCoefficient& grad_rho, VectorCoefficient& grad_phi0)
    : grad_rho_(&grad_rho), grad_phi0_(&grad_phi0) {}

BarotropicDensityGradientCoefficient::BarotropicDensityGradientCoefficient(
    const GridFunction& rho, const GridFunction& phi0)
    : owned_rho_(std::make_unique<GradientGridFunctionCoefficient>(&rho)),
      owned_phi0_(std::make_unique<GradientGridFunctionCoefficient>(&phi0)),
      grad_rho_(owned_rho_.get()),
      grad_phi0_(owned_phi0_.get()) {
  MFEM_VERIFY(rho.FESpace()->GetMesh() == phi0.FESpace()->GetMesh(),
              "BarotropicDensityGradientCoefficient: the density and the "
              "potential must live on the same mesh.");
}

real_t BarotropicDensityGradientCoefficient::Eval(ElementTransformation& T,
                                                  const IntegrationPoint& ip) {
  T.SetIntPoint(&ip);
  grad_phi0_->Eval(gp_, T, ip);
  const real_t g2 = gp_ * gp_;
  if (g2 <= 0.0) {
    return 0.0;
  }
  grad_rho_->Eval(gr_, T, ip);
  return (gr_ * gp_) / g2;
}

}  // namespace mfemElasticity
