/**
 * @file viscoelastic.hpp
 * @brief Quasi-static generalised Maxwell viscoelasticity as a
 * TimeDependentOperator on top of a QuasiStaticLinearElasticProblem, with
 * explicit, implicit and exponential time stepping.
 */

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"
#include "mfemElasticity/elastic_problem.hpp"

namespace mfemElasticity {

namespace detail {

/**
 * @brief Weights of the exponential trapezoid step for @f$h = dt/\tau@f$:
 * @f$m \leftarrow e^{-h} m + \alpha\, d^n + \beta\, d^{n+1}@f$ with
 * @f$\alpha = (1-e^{-h})/h - e^{-h}@f$ and @f$\beta = 1 - (1-e^{-h})/h@f$
 * (series for small @p h).
 */
void ExponentialTrapezoidWeights(mfem::real_t h, mfem::real_t& e,
                                 mfem::real_t& alpha, mfem::real_t& beta);

}  // namespace detail

/**
 * @brief Generalised Maxwell viscoelastic evolution operator.
 *
 * For every displacement field @f$i@f$ of the elastic problem and every
 * Prony branch @f$k@f$ of its rheology, a symmetric tensor internal
 * variable @f$m_k@f$ lives on a discontinuous (L2) nodal space on the
 * field's mesh: trace-free when the rheology is isotropic
 * (Rheology::TraceFreeInternalVariables()), a full symmetric tensor for an
 * anisotropic one. The ODE state is the concatenation of all
 * @f$m_{ik}@f$; the displacement is slaved to it through the quasi-static
 * solve
 * @f[
 *   K_U u = f_{ext}(t) + \sum_k B^T (C_k m_k), \qquad
 *   \dot m_k = (D u - m_k)/\tau_k ,
 * @f]
 * with @f$B_{IJ} = \int \Phi_I : \varepsilon(\phi_J)@f$ (deviatoric
 * strain in the trace-free case) assembled once with a unit coefficient,
 * all material weighting applied pointwise at the internal-variable nodes
 * (@f$C_k m_k = 2\mu_k m_k@f$ for the isotropic body, a sampled Mandel
 * tensor otherwise), and @f$D@f$ the strain map @f$u \mapsto
 * \varepsilon(u)@f$ (or its deviator) at those nodes, either the Galerkin
 * projection (default; adjoint-consistent) or nodal interpolation. The
 * projection is @f$(G^{-1} \otimes M^{-1}) B@f$ with @f$M@f$ the scalar L2
 * mass matrix and @f$G_{cc'} = E_c : E_{c'}@f$ the Frobenius metric of the
 * basis tensors @f$E_c@f$ (not orthonormal: the off-diagonal ones have
 * norm 2, and in the trace-free basis the diagonal ones overlap).
 *
 * **State-dependent relaxation times.** A branch may carry a RelaxationLaw
 * (Rheology::Law(k)), @f$\tau_k = \tau_{k0} F_k(\varepsilon, \sigma, m_k)@f$
 * evaluated at the internal nodes from the current state; the elastic part
 * stays linear and the global structure is unchanged. Explicit stepping and
 * ETD1 use the times of the current state (ETD1: frozen over the step,
 * first order). The exponential trapezoid and the implicit stages use a
 * predictor–corrector: the effective modulus is built with the times of
 * the start state, the step is taken, the times are re-evaluated at the
 * midpoint (trapezoid) or end (backward Euler) state and the step repeated,
 * SetCorrectorIterations() times or until the times stop changing. Each
 * pass is one elastic solve with a different effective operator (see
 * LinearElasticProblemBase::SetPreconditionerReuse()). A linear rheology
 * (Rheology::IsLinear()) takes the old paths unchanged.
 *
 * Time stepping:
 *  - Mult(): the explicit right-hand side, for any explicit MFEM ODESolver
 *    (stable only for @f$dt \lesssim 2.8\,\tau_{min}@f$ with RK4).
 *  - ExponentialEulerStep(): first order, exact relaxation with the strain
 *    frozen; one solve per step; no step restriction.
 *  - ExponentialTrapezoidStep(): second order, exact for a strain varying
 *    linearly in time; one solve per step with an effective modulus; no
 *    step restriction. The recommended workhorse.
 *  - ImplicitSolve(): backward Euler / SDIRK stages through MFEM's implicit
 *    ODESolvers, also via an effective modulus.
 * The last two need the elastic problem to support SetRelaxationWeights()
 * (the effective modulus is @f$C_\infty + \sum_k \beta_k C_k@f$ with nodal
 * weights @f$\beta_k@f$); the operator switches the problem between its
 * unrelaxed and effective operators lazily, so mixing schemes is allowed
 * but costs a reassembly at each switch.
 *
 * Observation: after a step the displacement held by the elastic problem is
 * consistent with the new state for the trapezoid and implicit schemes, but
 * not after the explicit or exponential-Euler ones. SolveElastic() gives a
 * consistent displacement for any (m, t), re-solving only when needed;
 * SyncFields() copies the state into the registered output GridFunctions.
 *
 * The internal-variable spaces use Ordering::byNODES, so component @f$c@f$
 * of a branch occupies @f$[c\,n_d, (c+1)\,n_d)@f$, with the component
 * convention of TraceFreeSymmetricMatrixIndex / SymmetricMatrixIndex. Works on serial
 * and parallel problems alike (all operations are element-local; the
 * forces are L-vectors, which is what AddForce expects).
 */
class ViscoelasticOperator : public mfem::TimeDependentOperator {
 public:
  enum class StrainMap { Galerkin, Interpolation };

  /**
   * @param problem The elastic problem; must outlive this operator.
   * @param internal_order Polynomial order of the L2 internal-variable
   * spaces; < 0 means the smallest order resolving eps(u) exactly:
   * displacement order - 1 on simplices, the displacement order on
   * tensor-product elements. With an exactly resolving order the Galerkin
   * strain map is the identity on representable strains, the two strain
   * maps coincide, and the effective-modulus elimination is exact.
   * @param map The strain map u -> d at the internal-variable nodes.
   */
  ViscoelasticOperator(QuasiStaticLinearElasticProblem& problem,
                       int internal_order = -1,
                       StrainMap map = StrainMap::Galerkin);

  // --- state layout ---------------------------------------------------------

  int NumFields() const { return static_cast<int>(fields_.size()); }
  int NumBranches(int i) const {
    return static_cast<int>(fields_[i].itau.size());
  }
  /** @brief Block offsets of the state, (field, branch) major. */
  const mfem::Array<int>& Offsets() const { return offsets_; }
  int BranchOffset(int i, int k) const;
  /** @brief Size of one branch of field @p i (= components x scalar dofs). */
  int BranchSize(int i) const { return fields_[i].nd * fields_[i].nc; }
  /** @brief Tensor components of field @p i's internal variables (n_s - 1
   * trace-free, n_s full). */
  int NumComponents(int i) const { return fields_[i].nc; }
  bool TraceFree(int i) const { return fields_[i].tracefree; }
  /** @brief Aliasing view (no copy) of branch @p k of field @p i in @p m. */
  mfem::Vector Branch(const mfem::Vector& m, int i, int k) const;

  mfem::FiniteElementSpace& InternalVariableSpace(int i) {
    return *fields_[i].dfes;
  }
  mfem::FiniteElementSpace& InternalScalarSpace(int i) {
    return *fields_[i].sfes;
  }
  /** @brief Output view of branch @p k of field @p i; see SyncFields(). */
  const mfem::GridFunction& InternalVariable(int i, int k) const {
    return *fields_[i].m_out[k];
  }
  StrainMap Map() const { return map_; }
  QuasiStaticLinearElasticProblem& Problem() { return problem_; }

  // --- ODE interface --------------------------------------------------------

  /** @brief Explicit right-hand side at GetTime(): one elastic solve with the
   * unrelaxed operator, then the pointwise rates. */
  void Mult(const mfem::Vector& m, mfem::Vector& k) const override;

  /** @brief Backward-Euler stage: k such that k = f(m + dt k, GetTime()),
   * via the effective modulus C_inf + sum_k C_k/(1 + dt/tau_k). */
  void ImplicitSolve(mfem::real_t dt, const mfem::Vector& m,
                     mfem::Vector& k) override;

  /** @brief One exponential Euler (ETD1) step: solve at (m, t), freeze the
   * strain, relax exactly. */
  void ExponentialEulerStep(mfem::Vector& m, mfem::real_t& t, mfem::real_t dt);

  /** @brief One exponential trapezoid step (second order, implicit in the
   * new strain through the effective modulus; predictor–corrector on the
   * relaxation times when they are state-dependent). Also forms the ETD1
   * prediction of the same step for ErrorEstimate(). */
  void ExponentialTrapezoidStep(mfem::Vector& m, mfem::real_t& t,
                                mfem::real_t dt);

  /**
   * @brief Local error estimate of the last ExponentialTrapezoidStep(): the
   * RMS over the state of @f$(m - \hat m) / (atol + rtol\,\max(|m^n|,
   * |m^{n+1}|))@f$ with @f$\hat m@f$ the first-order ETD1 prediction (global
   * in parallel). Below one means the step met the tolerances.
   */
  mfem::real_t ErrorEstimate(mfem::real_t rtol, mfem::real_t atol) const;

  // --- nonlinear controls ---------------------------------------------------

  /** @brief Corrector passes of the trapezoid and implicit steps for
   * state-dependent relaxation times (default 1), stopping early when the
   * relative change of the nodal times falls below @p tol. */
  void SetCorrectorIterations(int max_passes, mfem::real_t tol = 1e-3) {
    max_corrector_ = max_passes;
    corrector_tol_ = tol;
  }
  int LastCorrectorPasses() const { return last_passes_; }

  /** @brief True if every field's rheology is linear. */
  bool IsLinear() const { return linear_; }

  /** @brief Re-evaluate field @p i's relaxation times at the state
   * (@p d strain, @p m full state vector); no-op for linear fields. */
  void EvaluateRelaxationTimes(int i, const mfem::Vector& d,
                               const mfem::Vector& m) const;

  /** @brief Current nodal @f$1/\tau_k@f$ of field @p i (state-dependent
   * laws: as of the last evaluation). */
  const mfem::Vector& InverseRelaxationTimes(int i, int k) const {
    return fields_[i].itau[k];
  }

  // --- observation ----------------------------------------------------------

  /** @brief Make the problem's displacement consistent with (m, t) using the
   * unrelaxed operator; skips the solve when it already is. */
  bool SolveElastic(const mfem::Vector& m, mfem::real_t t);

  /** @brief Copy the state into the output GridFunctions. */
  void SyncFields(const mfem::Vector& m);

  /** @brief Register the problem's fields and the internal variables. */
  void RegisterFields(mfem::DataCollection& dc);

  /** @brief Smallest relaxation time over all nodes and branches (global in
   * parallel); the explicit stability limit is a small multiple of it. */
  mfem::real_t MinRelaxationTime() const;

  /** @brief The strain map of field @p i: d = D u at the internal nodes. */
  void ComputeStrain(int i, const mfem::GridFunction& u, mfem::Vector& d) const;

  /** @brief The geometric coupling form B of field @p i. */
  const mfem::MixedBilinearForm& CouplingForm(int i) const {
    return *fields_[i].B;
  }

  /** @brief Restore the problem's unrelaxed operator (done lazily anyway). */
  void UseUnrelaxedOperator() const;

 protected:
  enum class Scheme { None, BackwardEuler, ExponentialTrapezoid };

  struct Field {
    mfem::FiniteElementSpace* ufes = nullptr;
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    std::unique_ptr<mfem::FiniteElementSpace> dfes;  ///< tensor space
    std::unique_ptr<mfem::FiniteElementSpace> sfes;  ///< scalar companion
    std::unique_ptr<mfem::MixedBilinearForm> B;      ///< geometric coupling
    std::unique_ptr<mfem::DiscreteLinearOperator> D_interp;
    std::unique_ptr<mfem::SparseMatrix> Minv;  ///< block-diagonal L2 M^{-1}
    mfem::DenseMatrix Ginv;  ///< inverse Frobenius metric of the basis tensors
    bool tracefree = true;   ///< trace-free (isotropic) internal variables
    int nd = 0;              ///< scalar dofs
    int nc = 0;              ///< tensor components
    int ns = 0;              ///< d(d+1)/2
    /// Isotropic: nodal 2 mu_k. Anisotropic: per node the n_s x n_s matrix
    /// W with sigma_s = sum_t W_st m_t on unscaled tensor components
    /// (row-major, node-major: W(p)(s,t) at p ns ns + s ns + t).
    std::vector<mfem::Vector> branch_modulus;
    mfem::Vector CU;  ///< anisotropic: nodal W-form of C_U (for the stress)
    std::vector<mfem::Vector> itau0;          ///< nodal 1 / tau_k0
    mutable std::vector<mfem::Vector> itau;   ///< current nodal 1 / tau_k
    bool linear = true;                       ///< no state-dependent law
    std::vector<const RelaxationLaw*> law;    ///< per branch, may be null
    std::vector<mfem::Vector> law_params;     ///< per branch, node-major
    std::vector<int> num_params;
    std::vector<std::unique_ptr<mfem::GridFunction>> beta;  ///< nodal weights
    std::vector<std::unique_ptr<mfem::GridFunctionCoefficient>> beta_coef;
    std::vector<mfem::Coefficient*> beta_ptrs;
    std::vector<std::unique_ptr<mfem::GridFunction>> m_out;
    mutable mfem::Vector d, d_prev, dual, force, zeta;  ///< scratch
  };

  /** @brief Pointwise rate of branch @p k: (d - m_k) / tau_k. */
  virtual void Rate(const Field& f, int k, const mfem::Vector& m_k,
                    const mfem::Vector& d, mfem::Vector& k_out) const;

  /** @brief Exact relaxation of branch @p k over @p dt with @p d frozen. */
  virtual void LocalExponentialUpdate(const Field& f, int k, mfem::real_t dt,
                                      mfem::Vector& m_k,
                                      const mfem::Vector& d) const;

  /** @brief y = C_k x pointwise at the internal nodes (2 mu_k x for the
   * isotropic body, the sampled tensor otherwise). */
  static void ApplyBranchModulus(const Field& f, int k, const mfem::Vector& x,
                                 mfem::Vector& y);

  /** @brief Push B^T zeta (zeta in the internal layout) into field i. */
  void AddCoupledForce(int i, const mfem::Vector& zeta) const;

  /** @brief AssembleForce(t), the branch forces sum_k B^T(2 mu_k m_k), and
   * Solve(). */
  bool ElasticUpdate(const mfem::Vector& m, mfem::real_t t) const;

  /** @brief d_i = D u_i for every field, into fields_[i].d. */
  void ComputeAllStrains() const;

  /** @brief Set the relaxation weights for (dt, scheme) on every field,
   * reassembling only when they change. */
  void SetEffectiveModulus(mfem::real_t dt, Scheme scheme) const;

  bool CacheMatches(const mfem::Vector& m, mfem::real_t t) const;
  void UpdateCache(const mfem::Vector& m, mfem::real_t t) const;

  /** @brief Max over fields and nodes of the relative change of 1/tau
   * between @p old_itau and the current values (global). */
  mfem::real_t RelaxationTimeChange(
      const std::vector<std::vector<mfem::Vector>>& old_itau) const;
  std::vector<std::vector<mfem::Vector>> SaveRelaxationTimes() const;
  mfem::real_t GlobalMax(mfem::real_t v) const;

  QuasiStaticLinearElasticProblem& problem_;
  StrainMap map_;
  std::vector<Field> fields_;
  mfem::Array<int> offsets_;

  mutable Scheme scheme_ = Scheme::None;
  mutable mfem::real_t effective_dt_ = -1.0;
  mutable long effective_version_ = -1;  ///< tau version the weights hold
  mutable long tau_version_ = 0;         ///< bumped by EvaluateRelaxationTimes
  bool linear_ = true;
  int max_corrector_ = 1;
  mfem::real_t corrector_tol_ = 1e-3;
  int last_passes_ = 0;

  // ETD1 predictor of the last trapezoid step, for the error estimate.
  mfem::Vector predictor_diff_, m_scale_;

  // (m, t) for which fields_[i].d and the problem's displacement are
  // consistent, if cache_valid_.
  mutable mfem::Vector cached_m_;
  mutable mfem::real_t cached_t_ = 0.0;
  mutable bool cache_valid_ = false;

  bool parallel_ = false;
#ifdef MFEM_USE_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
#endif
};

/**
 * @brief ODESolver adaptor for ViscoelasticOperator::ExponentialEulerStep().
 */
class ExponentialEulerSolver : public mfem::ODESolver {
 public:
  void Init(mfem::TimeDependentOperator& f) override;
  void Step(mfem::Vector& x, mfem::real_t& t, mfem::real_t& dt) override;

 private:
  ViscoelasticOperator* op_ = nullptr;
};

/**
 * @brief ODESolver adaptor for
 * ViscoelasticOperator::ExponentialTrapezoidStep().
 */
class ExponentialTrapezoidSolver : public mfem::ODESolver {
 public:
  void Init(mfem::TimeDependentOperator& f) override;
  void Step(mfem::Vector& x, mfem::real_t& t, mfem::real_t& dt) override;

 private:
  ViscoelasticOperator* op_ = nullptr;
};

/**
 * @brief Adaptive exponential trapezoid stepping: the embedded ETD1 /
 * trapezoid pair gives a free local error estimate
 * (ViscoelasticOperator::ErrorEstimate()), and a standard controller
 * @f$dt \leftarrow dt\,\min(grow, \max(shrink, safety\,err^{-1/2}))@f$
 * shrinks and retries a step whose estimate exceeds one. Serves linear
 * bodies (where the effective operator depends on dt) and state-dependent
 * relaxation times alike; there is no stability limit, only accuracy.
 * The estimate is the local error of the first-order companion, so the
 * tolerance is conservative for the second-order solution that is
 * propagated (by a factor of order @f$\tau/dt@f$); choose rtol
 * accordingly.
 *
 * Step(x, t, dt) takes a step of at most @p dt (never below the lower bound)
 * and returns in @p dt the step proposed next; Integrate() runs to a final
 * time, hitting it exactly.
 */
class AdaptiveExponentialTrapezoidSolver : public mfem::ODESolver {
 public:
  void Init(mfem::TimeDependentOperator& f) override;
  void Step(mfem::Vector& x, mfem::real_t& t, mfem::real_t& dt) override;

  /** @brief Integrate from @p t to @p t_final with the current step @p dt
   * (updated on return); returns the number of accepted steps. */
  int Integrate(mfem::Vector& x, mfem::real_t& t, mfem::real_t t_final,
                mfem::real_t& dt);

  void SetTolerances(mfem::real_t rtol, mfem::real_t atol) {
    rtol_ = rtol;
    atol_ = atol;
  }
  void SetStepBounds(mfem::real_t dt_min, mfem::real_t dt_max) {
    dt_min_ = dt_min;
    dt_max_ = dt_max;
  }
  void SetStepFactors(mfem::real_t shrink, mfem::real_t grow,
                      mfem::real_t safety) {
    shrink_ = shrink;
    grow_ = grow;
    safety_ = safety;
  }
  int NumAcceptedSteps() const { return accepted_; }
  int NumRejectedSteps() const { return rejected_; }
  mfem::real_t LastErrorEstimate() const { return last_err_; }

 private:
  ViscoelasticOperator* op_ = nullptr;
  mfem::real_t rtol_ = 1e-4, atol_ = 1e-10;
  mfem::real_t dt_min_ = 0.0, dt_max_ = mfem::infinity();
  mfem::real_t shrink_ = 0.2, grow_ = 4.0, safety_ = 0.9;
  int accepted_ = 0, rejected_ = 0;
  mfem::real_t last_err_ = 0.0;
  mfem::Vector trial_;
};

}  // namespace mfemElasticity
