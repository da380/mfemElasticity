/**
 * @file rheology.cpp
 * @brief Implementation of GeneralisedMaxwellRheology.
 */

#include "mfemElasticity/rheology.hpp"

namespace mfemElasticity {

GeneralisedMaxwellRheology::GeneralisedMaxwellRheology(
    int dim, mfem::Coefficient& kappa, mfem::Coefficient& mu_inf,
    const std::vector<MaxwellBranch>& branches)
    : dim_(dim), kappa_(&kappa), mu_inf_(&mu_inf), branches_(branches) {
  MFEM_VERIFY(dim == 2 || dim == 3,
              "GeneralisedMaxwellRheology: dim must be 2 or 3.");
  for (const auto& b : branches_) {
    MFEM_VERIFY(b.mu && b.tau,
                "GeneralisedMaxwellRheology: every branch needs mu and tau.");
  }

  // mu_U = mu_inf + sum_k mu_k, as a chain of SumCoefficients.
  mu_u_ = mu_inf_;
  for (const auto& b : branches_) {
    owned_.push_back(std::make_unique<mfem::SumCoefficient>(*mu_u_, *b.mu));
    mu_u_ = owned_.back().get();
  }

  // lambda_U = kappa - 2 mu_U / dim.
  owned_.push_back(std::make_unique<mfem::SumCoefficient>(
      *kappa_, *mu_u_, 1.0, -2.0 / static_cast<mfem::real_t>(dim_)));
  lambda_u_ = owned_.back().get();
}

GeneralisedMaxwellRheology GeneralisedMaxwellRheology::Elastic(
    int dim, mfem::Coefficient& kappa, mfem::Coefficient& mu) {
  return GeneralisedMaxwellRheology(dim, kappa, mu);
}

GeneralisedMaxwellRheology GeneralisedMaxwellRheology::Maxwell(
    int dim, mfem::Coefficient& kappa, mfem::Coefficient& mu,
    mfem::Coefficient& tau) {
  // mu_inf = 0 is an owned constant; move it into the result so that the
  // pointer stays valid.
  auto zero = std::make_unique<mfem::ConstantCoefficient>(0.0);
  auto* zero_ptr = zero.get();
  std::vector<MaxwellBranch> branches{MaxwellBranch{&mu, &tau}};
  GeneralisedMaxwellRheology r(dim, kappa, *zero_ptr, branches);
  r.owned_.push_back(std::move(zero));
  return r;
}

}  // namespace mfemElasticity
