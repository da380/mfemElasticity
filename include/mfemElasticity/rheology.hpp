/**
 * @file rheology.hpp
 * @brief Material description shared by the elastic and viscoelastic layers:
 * an isotropic generalised Maxwell (Prony series) rheology.
 */

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"

namespace mfemElasticity {

/**
 * @brief One Prony branch of a generalised Maxwell body: a relaxable shear
 * modulus @f$\mu_k@f$ and its relaxation time @f$\tau_k = \eta_k/\mu_k@f$.
 * Both coefficients are non-owning.
 */
struct MaxwellBranch {
  mfem::Coefficient* mu = nullptr;
  mfem::Coefficient* tau = nullptr;
};

/**
 * @brief Isotropic generalised Maxwell rheology.
 *
 * With @f$d = \mathrm{dev}\,\varepsilon(u)@f$ and @f$K@f$ branches,
 * @f[
 *   \sigma = \kappa\,\mathrm{tr}(\varepsilon) I + 2\mu_\infty d
 *          + \sum_k 2\mu_k (d - m_k), \qquad
 *   \dot m_k = (d - m_k)/\tau_k .
 * @f]
 * The classical Maxwell body is @f$\mu_\infty = 0@f$ with one branch; a
 * purely elastic solid has no branches.
 *
 * The material data has one owner: the elastic problem assembles its
 * operator with the *unrelaxed* shear modulus @f$\mu_U = \mu_\infty +
 * \sum_k \mu_k@f$ obtained from here, and the viscoelastic operator reads
 * the branch data from here, so the two layers cannot disagree.
 *
 * The space dimension fixes the deviatoric convention: in 2-D the library
 * models a two-dimensional continuum (2-D deviator, @f$\lambda = \kappa -
 * \mu@f$), not plane strain of a 3-D body.
 *
 * The object holds only pointers to the caller's coefficients, which must
 * outlive it, plus the sum coefficients it builds itself. It is movable but
 * not copyable.
 */
class GeneralisedMaxwellRheology {
 public:
  /**
   * @param dim Space dimension (2 or 3).
   * @param kappa Bulk modulus.
   * @param mu_inf Long-term (fully relaxed) shear modulus.
   * @param branches Prony branches; may be empty.
   */
  GeneralisedMaxwellRheology(int dim, mfem::Coefficient& kappa,
                             mfem::Coefficient& mu_inf,
                             const std::vector<MaxwellBranch>& branches = {});

  /** @brief Purely elastic solid: no branches, @f$\mu_U = \mu@f$. */
  static GeneralisedMaxwellRheology Elastic(int dim, mfem::Coefficient& kappa,
                                            mfem::Coefficient& mu);

  /** @brief Classical Maxwell body: @f$\mu_\infty = 0@f$ and one branch. */
  static GeneralisedMaxwellRheology Maxwell(int dim, mfem::Coefficient& kappa,
                                            mfem::Coefficient& mu,
                                            mfem::Coefficient& tau);

  GeneralisedMaxwellRheology(GeneralisedMaxwellRheology&&) = default;
  GeneralisedMaxwellRheology& operator=(GeneralisedMaxwellRheology&&) = default;
  GeneralisedMaxwellRheology(const GeneralisedMaxwellRheology&) = delete;
  GeneralisedMaxwellRheology& operator=(const GeneralisedMaxwellRheology&) =
      delete;

  int SpaceDim() const { return dim_; }

  mfem::Coefficient& BulkModulus() const { return *kappa_; }

  /** @brief @f$\mu_\infty@f$. */
  mfem::Coefficient& LongTermShearModulus() const { return *mu_inf_; }

  /** @brief @f$\mu_U = \mu_\infty + \sum_k \mu_k@f$; the modulus the elastic
   * operator must be assembled with. */
  mfem::Coefficient& UnrelaxedShearModulus() const { return *mu_u_; }

  /** @brief @f$\lambda_U = \kappa - 2\mu_U/d@f$, for (lambda, mu) form. */
  mfem::Coefficient& UnrelaxedLame() const { return *lambda_u_; }

  int NumBranches() const { return static_cast<int>(branches_.size()); }

  const MaxwellBranch& Branch(int k) const { return branches_[k]; }

 private:
  int dim_;
  mfem::Coefficient* kappa_;
  mfem::Coefficient* mu_inf_;
  std::vector<MaxwellBranch> branches_;
  std::vector<std::unique_ptr<mfem::Coefficient>> owned_;
  mfem::Coefficient* mu_u_ = nullptr;
  mfem::Coefficient* lambda_u_ = nullptr;
};

}  // namespace mfemElasticity
