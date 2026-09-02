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
 * Prony branch @f$k@f$ of its rheology, a trace-free symmetric tensor
 * internal variable @f$m_k@f$ lives on a discontinuous (L2) nodal space on
 * the field's mesh. The ODE state is the concatenation of all
 * @f$m_{ik}@f$; the displacement is slaved to it through the quasi-static
 * solve
 * @f[
 *   K_U u = f_{ext}(t) + \sum_k B^T (2\mu_k m_k), \qquad
 *   \dot m_k = (D u - m_k)/\tau_k ,
 * @f]
 * with @f$B_{IJ} = \int \Phi_I : \mathrm{dev}\,\varepsilon(\phi_J)@f$
 * assembled once with a unit coefficient (all material weighting is
 * applied pointwise at the internal-variable nodes) and @f$D@f$ the strain
 * map @f$u \mapsto \mathrm{dev}\,\varepsilon(u)@f$ at those nodes, either
 * the Galerkin projection (default; adjoint-consistent) or nodal
 * interpolation. The projection is @f$(G^{-1} \otimes M^{-1}) B@f$ with
 * @f$M@f$ the scalar L2 mass matrix and @f$G_{cc'} = E_c : E_{c'}@f$ the
 * Frobenius metric of the trace-free basis tensors @f$E_c@f$ (which are
 * not orthonormal: each has norm 2 and in 3-D the two diagonal ones
 * overlap).
 *
 * Time stepping:
 *  - Mult(): the explicit right-hand side, for any explicit MFEM ODESolver
 *    (stable only for @f$dt \lesssim 2.8\,\tau_{min}@f$ with RK4).
 *  - ExponentialEulerStep(): first order, exact relaxation with the strain
 *    frozen; one solve per step; no step restriction.
 *  - ExponentialTrapezoidStep(): second order, exact for a strain varying
 *    linearly in time; one solve per step with an effective shear modulus;
 *    no step restriction. The recommended workhorse.
 *  - ImplicitSolve(): backward Euler / SDIRK stages through MFEM's implicit
 *    ODESolvers, also via an effective modulus.
 * The last two need the elastic problem to support
 * SetEffectiveShearModulus(); the operator switches the problem between its
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
 * of a branch occupies @f$[c\,n_d, (c+1)\,n_d)@f$, with the trace-free
 * component convention of TraceFreeSymmetricMatrixIndex. Works on serial
 * and parallel problems alike (all operations are element-local; the
 * forces are L-vectors, which is what AddForce expects).
 */
class ViscoelasticOperator : public mfem::TimeDependentOperator {
 public:
  enum class StrainMap { Galerkin, Interpolation };

  /**
   * @param problem The elastic problem; must outlive this operator.
   * @param internal_order Polynomial order of the L2 internal-variable
   * spaces; < 0 means the smallest order resolving dev eps(u) exactly:
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
    return static_cast<int>(fields_[i].two_mu.size());
  }
  /** @brief Block offsets of the state, (field, branch) major. */
  const mfem::Array<int>& Offsets() const { return offsets_; }
  int BranchOffset(int i, int k) const;
  /** @brief Size of one branch of field @p i (= components x scalar dofs). */
  int BranchSize(int i) const { return fields_[i].nd * fields_[i].nc; }
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
   * via the effective modulus mu_inf + sum_k mu_k/(1 + dt/tau_k). */
  void ImplicitSolve(mfem::real_t dt, const mfem::Vector& m,
                     mfem::Vector& k) override;

  /** @brief One exponential Euler (ETD1) step: solve at (m, t), freeze the
   * strain, relax exactly. */
  void ExponentialEulerStep(mfem::Vector& m, mfem::real_t& t, mfem::real_t dt);

  /** @brief One exponential trapezoid step (second order, implicit in the
   * new strain through the effective modulus). */
  void ExponentialTrapezoidStep(mfem::Vector& m, mfem::real_t& t,
                                mfem::real_t dt);

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
    std::unique_ptr<mfem::FiniteElementSpace> dfes;  ///< trace-free tensors
    std::unique_ptr<mfem::FiniteElementSpace> sfes;  ///< scalar companion
    std::unique_ptr<mfem::MixedBilinearForm> B;      ///< geometric coupling
    std::unique_ptr<mfem::DiscreteLinearOperator> D_interp;
    std::unique_ptr<mfem::SparseMatrix> Minv;  ///< block-diagonal L2 M^{-1}
    mfem::DenseMatrix Ginv;  ///< inverse Frobenius metric of the basis tensors
    int nd = 0;              ///< scalar dofs
    int nc = 0;              ///< tensor components
    mfem::Vector mu_inf;     ///< nodal mu_inf
    std::vector<mfem::Vector> two_mu;  ///< nodal 2 mu_k
    std::vector<mfem::Vector> itau;    ///< nodal 1 / tau_k
    std::unique_ptr<mfem::GridFunction> mu_eff;
    std::unique_ptr<mfem::GridFunctionCoefficient> mu_eff_coef;
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

  /** @brief y = w o x with the nodal weight w applied to every component. */
  static void ApplyNodalWeight(const Field& f, const mfem::Vector& w,
                               const mfem::Vector& x, mfem::Vector& y);

  /** @brief Push B^T zeta (zeta in the internal layout) into field i. */
  void AddCoupledForce(int i, const mfem::Vector& zeta) const;

  /** @brief AssembleForce(t), the branch forces sum_k B^T(2 mu_k m_k), and
   * Solve(). */
  bool ElasticUpdate(const mfem::Vector& m, mfem::real_t t) const;

  /** @brief d_i = D u_i for every field, into fields_[i].d. */
  void ComputeAllStrains() const;

  /** @brief Set the effective modulus for (dt, scheme) on every field,
   * reassembling only when they change. */
  void SetEffectiveModulus(mfem::real_t dt, Scheme scheme) const;

  bool CacheMatches(const mfem::Vector& m, mfem::real_t t) const;
  void UpdateCache(const mfem::Vector& m, mfem::real_t t) const;

  QuasiStaticLinearElasticProblem& problem_;
  StrainMap map_;
  std::vector<Field> fields_;
  mfem::Array<int> offsets_;

  mutable Scheme scheme_ = Scheme::None;
  mutable mfem::real_t effective_dt_ = -1.0;

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

}  // namespace mfemElasticity
