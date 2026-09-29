/**
 * @file background.hpp
 * @brief The background-state module: generation of the equilibrium
 * fields @f$(\rho, \hat C, \mathbf{S}_e, \varphi_e)@f$ that the
 * referential problem consumes (doc/gravitating_elasticity.md §6–7).
 * The moduli conversion and the relabelling transformation laws live
 * here; drivers never hand-roll them.
 */

#pragma once

#include <functional>
#include <vector>

#include "mfem.hpp"
#include "mfemElasticity/elastic_tensor.hpp"
#include "mfemElasticity/mappings.hpp"
#include "mfemElasticity/referential_problem.hpp"

namespace mfemElasticity {

/**
 * @brief The hydrostatic balance of a radial density profile: gravity
 * @f$g(r)@f$ and pressure @f$p^0(r)@f$ by cumulative quadrature, in the
 * codebase's Poisson convention @f$\nabla^2\Phi_0 = 4\pi G\rho@f$ for
 * either dimension,
 * @f[
 *   g(r) = \frac{4\pi G}{r^{d-1}} \int_0^r \rho(s)\, s^{d-1}\, ds,
 *   \qquad \frac{dp^0}{dr} = -\rho g, \qquad p^0(R) = 0,
 * @f]
 * (2-D: the gravitating-disc convention of the test models). The profile
 * is sampled on a uniform grid and interpolated linearly; piecewise-smooth
 * densities (PREM) are fine. Outside the radius @f$p^0 = 0@f$ and
 * @f$g@f$ is the exterior field of the total mass.
 */
class RadialHydrostaticState {
 public:
  /**
   * @param dim Space dimension (2 or 3).
   * @param rho Radial density @f$\rho(r)@f$.
   * @param G Gravitational constant in the units of the problem.
   * @param radius The (referential) surface radius @f$R@f$.
   * @param samples Grid intervals for the cumulative quadrature.
   */
  RadialHydrostaticState(int dim,
                         std::function<mfem::real_t(mfem::real_t)> rho,
                         mfem::real_t G, mfem::real_t radius,
                         int samples = 4000);

  mfem::real_t Gravity(mfem::real_t r) const;
  mfem::real_t Pressure(mfem::real_t r) const;
  /** @brief The profile itself (0 outside the radius). */
  mfem::real_t Density(mfem::real_t r) const;

  mfem::real_t Radius() const { return R_; }
  int SpaceDim() const { return dim_; }

 private:
  int dim_;
  std::function<mfem::real_t(mfem::real_t)> rho_;
  mfem::real_t G_, R_, h_;
  std::vector<mfem::real_t> g_, p_;
};

/**
 * @brief The hydrostatic equilibrium of a radial model as a referential
 * background at @f$\varphi_e = \mathrm{id}@f$: owns the full coefficient
 * chain the referential problem consumes — the referential density
 * @f$\rho(|\mathbf{x}|)@f$, the *bare* second elastic tensor
 * @f$\hat C@f$ (the radial moduli are SEISMOLOGICAL @f$(\kappa,\mu)@f$;
 * BareElasticTensorCoefficient applies the conversion with the
 * hydrostatic @f$p^0@f$ computed here), the equilibrium stress
 * @f$\mathbf{S}_e = -p^0\mathbf{1}@f$, and the identity mapping — and
 * the assembled ReferentialElasticRheology.
 *
 * Non-copyable: the members reference one another.
 */
class RadialHydrostaticBackground {
 public:
  using RadialFunc = std::function<mfem::real_t(mfem::real_t)>;

  RadialHydrostaticBackground(int dim, RadialFunc rho, RadialFunc kappa,
                              RadialFunc mu, mfem::real_t G,
                              mfem::real_t radius, int samples = 4000);

  RadialHydrostaticBackground(const RadialHydrostaticBackground&) = delete;
  RadialHydrostaticBackground& operator=(const RadialHydrostaticBackground&) =
      delete;

  ReferentialElasticRheology& Rheology() { return rheology_; }
  mfem::Coefficient& Density() { return rho_coef_; }
  /** @brief The bare tensor @f$\hat C@f$ (Mandel). */
  mfem::MatrixCoefficient& ElasticTensor() { return C_; }
  mfem::MatrixCoefficient& EquilibriumStress() { return S_; }
  Diffeomorphism& EquilibriumMapping() { return phi_e_; }
  mfem::Coefficient& Pressure() { return p0_coef_; }

  const RadialHydrostaticState& State() const { return state_; }
  int SpaceDim() const { return dim_; }

  mfem::real_t DensityAt(mfem::real_t r) const { return state_.Density(r); }
  mfem::real_t KappaAt(mfem::real_t r) const { return kappa_fn_(r); }
  mfem::real_t MuAt(mfem::real_t r) const { return mu_fn_(r); }
  mfem::real_t PressureAt(mfem::real_t r) const { return state_.Pressure(r); }
  mfem::real_t GravityAt(mfem::real_t r) const { return state_.Gravity(r); }

 private:
  int dim_;
  RadialFunc kappa_fn_, mu_fn_;
  RadialHydrostaticState state_;
  mfem::FunctionCoefficient rho_coef_, kappa_coef_, mu_coef_, p0_coef_;
  IsotropicElasticTensorCoefficient C_eff_;
  BareElasticTensorCoefficient C_;
  mfem::ProductCoefficient minus_p0_;
  mfem::IdentityMatrixCoefficient identity_;
  mfem::ScalarMatrixProductCoefficient S_;
  IdentityDiffeomorphism phi_e_;
  ReferentialElasticRheology rheology_;
};

/**
 * @brief The same physical equilibrium described from a relabelled
 * reference body: given a radial hydrostatic base and a relabelling
 * @f$\boldsymbol{\xi}@f$ (the identity at and outside the surface for the
 * present problem class), owns the transformed chain
 * @f[
 *   \tilde\rho = J_\xi\,\rho\circ\xi, \qquad
 *   \tilde C = J_\xi\, Q_\xi^{-T} (\hat C\circ\xi)\, Q_\xi^{-1}, \qquad
 *   \tilde S = J_\xi\, F_\xi^{-1} (\mathbf{S}_e\circ\xi)\, F_\xi^{-T},
 *   \qquad \varphi_e = \xi,
 * @f]
 * (doc/gravitating_elasticity.md §1, doc/mappings.md). The solution must
 * reproduce the base solution under composition — the tier-(ii)
 * self-benchmark. Base and @f$\boldsymbol{\xi}@f$ are not owned and must
 * outlive this object. Non-copyable.
 */
class RelabelledBackground {
 public:
  RelabelledBackground(RadialHydrostaticBackground& base, Diffeomorphism& xi);

  RelabelledBackground(const RelabelledBackground&) = delete;
  RelabelledBackground& operator=(const RelabelledBackground&) = delete;

  ReferentialElasticRheology& Rheology() { return rheology_; }
  /** @brief @f$\tilde\rho = J_\xi\,\rho\circ\xi@f$. */
  mfem::Coefficient& Density() { return rho_rel_; }
  mfem::MatrixCoefficient& ElasticTensor() { return C_rel_; }
  mfem::MatrixCoefficient& EquilibriumStress() { return S_rel_; }
  Diffeomorphism& EquilibriumMapping() { return *xi_; }

 private:
  RadialHydrostaticBackground* base_;
  Diffeomorphism* xi_;
  TransformedFunctionCoefficient rho_xi_, kappa_xi_, mu_xi_, p0_xi_;
  IsotropicElasticTensorCoefficient C_eff_xi_;
  BareElasticTensorCoefficient C_comp_;
  RelabelledElasticTensorCoefficient C_rel_;
  TransformedMatrixFunctionCoefficient S_comp_;
  PullbackStressCoefficient S_rel_;
  JacobianCoefficient jac_;
  mfem::ProductCoefficient rho_rel_;
  ReferentialElasticRheology rheology_;
};

}  // namespace mfemElasticity
