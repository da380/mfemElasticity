/**
 * @file quasi_static_problem.hpp
 * @brief Linear quasi-static problems: the abstract interface used by the
 * viscoelastic layer, a base class owning the shared bookkeeping (serial and
 * parallel), and two reference problems (pure traction, clamped).
 *
 * A problem is the equilibrium of a body at a time @f$t@f$: geometry,
 * boundary conditions, loads and solver. Whether the body is elastic or
 * viscoelastic is decided by its Rheology; the problem assembles the
 * rheology's (effective) elastic stiffness and never sees the internal
 * variables, which the viscoelastic layer evolves around it.
 */

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"
#include "mfemElasticity/rheology.hpp"
#include "mfemElasticity/solvers.hpp"

namespace mfemElasticity {

/**
 * @brief Abstract interface for linear quasi-static problems.
 *
 * Per evaluation time @f$t@f$ the protocol is
 * @code
 *   AssembleForce(t);        // all time-dependent data to t; external loads;
 *                            // increments cleared
 *   AddForce(f); ...         // superpose dual vectors on the displacement
 *   Solve();                 // displacement <- K^{-1}(external + increments)
 * @endcode
 *
 * - AssembleForce(t) is called at every stage of a time integrator, with
 *   possibly non-monotone t; it must be cheap and idempotent at fixed t.
 * - AddForce(f) takes the vdof (L-vector) layout of DisplacementSpace(),
 *   i.e. the layout of a LinearForm on that space before FormLinearSystem.
 *   In parallel the problem applies the prolongation transpose inside
 *   Solve(); callers never handle true dofs. AddForce accumulates.
 * - Solve() may be internally iterative or nonlinear, but is a black box to
 *   callers; linearity in the *forces* is part of the contract. It returns
 *   false if the linear solver did not converge.
 * - Problems carrying more unknowns than the displacement (a gravitational
 *   potential, say) keep them internal: the interface only ever refers to
 *   the displacement. There is one displacement field; several solid
 *   regions share it on a (possibly disconnected) SubMesh, and regional
 *   material differences are the rheology's business.
 *
 * Implicit and exponential-trapezoid viscoelastic stepping eliminate the
 * internal variables and need the stiffness reassembled with the
 * *effective* modulus @f$C_\infty + \sum_k \beta_k C_k@f$, with pointwise
 * relaxation weights @f$\beta_k@f$ (see ElasticStiffness); problems that
 * can do so advertise it through SupportsRelaxationWeights().
 */
class LinearQuasiStaticProblem {
 public:
  virtual ~LinearQuasiStaticProblem() = default;

  /** @brief The (vector) displacement space. */
  virtual mfem::FiniteElementSpace& DisplacementSpace() = 0;

  /** @brief Read-only access to the displacement. */
  virtual const mfem::GridFunction& Displacement() const = 0;

  /** @brief The rheology the operator was assembled with. */
  virtual const mfemElasticity::Rheology& Rheology() const = 0;

  /** @brief Bring all time-dependent data to time @p t and reset forcing. */
  virtual void AssembleForce(mfem::real_t t) = 0;

  /** @brief Superpose a dual vector (LinearForm layout) on the displacement. */
  virtual void AddForce(const mfem::Vector& f) = 0;

  /** @brief Solve for the displacement(s); false on solver failure. */
  virtual bool Solve() = 0;

  /** @brief Whether SetRelaxationWeights() is available. */
  virtual bool SupportsRelaxationWeights() const { return false; }

  /**
   * @brief Reassemble the stiffness with @f$C_\infty + \sum_k
   * \beta_k C_k@f$, one weight coefficient per branch of the rheology
   * (typically nodal fields on the internal-variable mesh). The problem
   * must invalidate its solver setup and reassemble on every call (the same
   * coefficient objects may carry new values). The coefficients must outlive
   * the next call to SetRelaxationWeights() or ClearRelaxationWeights().
   */
  virtual void SetRelaxationWeights(
      const std::vector<mfem::Coefficient*>& /*beta*/) {
    MFEM_ABORT("relaxation weights not supported by this problem");
  }

  /** @brief Restore the unrelaxed modulus @f$C_U@f$. */
  virtual void ClearRelaxationWeights() {}

  /** @brief Register output fields with a DataCollection. */
  virtual void RegisterFields(mfem::DataCollection& dc) = 0;
};

/**
 * @brief Base class implementing the interface on a serial or parallel
 * displacement space.
 *
 * The stiffness integrators come from the rheology's ElasticStiffness
 * (two split mfem::ElasticityIntegrators for the isotropic body, one
 * ElasticTensorIntegrator for an anisotropic one), assembled with the
 * unrelaxed modulus or, after SetRelaxationWeights(), the effective one.
 * The operator is (re)assembled lazily in Solve() whenever it is out of
 * date.
 *
 * Serial and parallel are handled in one class: the space decides. Forms,
 * grid function and system matrix are created through their parallel
 * variants when the space is a ParFiniteElementSpace.
 *
 * A derived class:
 *  1. adds loads to ExternalLoad() (and registers time-dependent
 *     coefficients with RegisterTimeDependent) in its constructor;
 *  2. optionally calls SetEssentialBoundary() and overrides
 *     UpdateBoundaryValues();
 *  3. optionally adds further integrators to StiffnessIntegrators();
 *  4. optionally overrides SetupSolver()/SolveLinearSystem() (the defaults
 *     are preconditioned CG with Gauss-Seidel or BoomerAMG).
 */
class LinearQuasiStaticProblemBase : public LinearQuasiStaticProblem {
 public:
  /**
   * @param fes Displacement space (vdim = space dimension); serial or
   * parallel; not owned.
   * @param rheology The material; not owned, must outlive the problem.
   */
  LinearQuasiStaticProblemBase(mfem::FiniteElementSpace* fes,
                               const mfemElasticity::Rheology& rheology);

  mfem::FiniteElementSpace& DisplacementSpace() override { return *fes_; }
  const mfem::GridFunction& Displacement() const override { return *u_; }
  const mfemElasticity::Rheology& Rheology() const override {
    return *rheology_;
  }

  void AssembleForce(mfem::real_t t) override;
  void AddForce(const mfem::Vector& f) override;
  bool Solve() override;

  bool SupportsRelaxationWeights() const override { return true; }
  void SetRelaxationWeights(
      const std::vector<mfem::Coefficient*>& beta) override;
  void ClearRelaxationWeights() override;

  void RegisterFields(mfem::DataCollection& dc) override;

  /** @brief True if the displacement space is a ParFiniteElementSpace. */
  bool IsParallel() const;

  /** @brief Time of the most recent AssembleForce(). */
  mfem::real_t Time() const { return t_; }

  /** @brief The external load; add integrators here (before the first
   * AssembleForce). */
  mfem::LinearForm& ExternalLoad() { return *b_; }

  /** @brief Integrators of the stiffness; add further ones here (before the
   * first Solve). The rheology's integrators are already present. */
  mfem::BilinearForm& StiffnessIntegrators() { return *integrators_; }

  /** @brief The rheology's stiffness object of this problem (its relaxation
   * state). */
  const ElasticStiffness& Stiffness() const { return *stiffness_; }

  /** @brief Register a coefficient whose SetTime() AssembleForce() calls. */
  void RegisterTimeDependent(mfem::Coefficient& c) { td_coefs_.push_back(&c); }
  void RegisterTimeDependent(mfem::VectorCoefficient& c) {
    td_vcoefs_.push_back(&c);
  }

  /** @brief Relative tolerance of the linear solves (against the load). */
  void SetRelTol(mfem::real_t rel_tol) { rel_tol_ = rel_tol; }
  mfem::real_t RelTol() const { return rel_tol_; }

  /** @brief Print level of the default CG solver (quiet by default); takes
   * effect at the next operator assembly. */
  void SetPrintLevel(mfem::IterativeSolver::PrintLevel level) {
    print_level_ = level;
  }

  /**
   * @brief Keep the preconditioner across reassemblies of the stiffness
   * (a change of relaxation weights, e.g. every step under a variable dt or
   * state-dependent relaxation times) as long as the solver's iteration
   * count stays within @p factor of its count when the preconditioner was
   * built; then rebuild it. @p factor <= 1 rebuilds at every assembly.
   * Default 2: the BoomerAMG setup, the dominant cost of an assembly, is
   * then paid only when the operator has drifted far. The matrix itself is
   * always the current one.
   */
  void SetPreconditionerReuse(mfem::real_t factor) { prec_reuse_ = factor; }
  mfem::real_t PreconditionerReuse() const { return prec_reuse_; }

  /** @brief Number of preconditioner setups so far. */
  int NumPreconditionerSetups() const { return prec_setups_; }

  /** @brief Number of operator assemblies so far. */
  int NumAssemblies() const { return assemblies_; }

  /** @brief Number of Solve() calls so far. */
  int NumSolves() const { return solves_; }

  /** @brief Outer solver iterations accumulated over all solves. */
  long TotalIterations() const { return total_its_; }

  /** @brief The current (eliminated) system matrix, assembling if needed. */
  const mfem::OperatorHandle& SystemMatrix();

 protected:
  /** @brief Impose essential conditions on the marked boundary attributes
   * (all components). */
  void SetEssentialBoundary(const mfem::Array<int>& ess_bdr);

  /** @brief Refresh Dirichlet values in the solution at time @p t; called by
   * AssembleForce(). Default: nothing. */
  virtual void UpdateBoundaryValues(mfem::real_t /*t*/) {}

  /** @brief Build/refresh preconditioner and solver for a new operator.
   * Default: SetupDefaultCG(). */
  virtual void SetupSolver(mfem::OperatorHandle& A);

  /** @brief Solve A X = B with the solver from SetupSolver(); return
   * convergence. Default: warm-started preconditioned CG. */
  virtual bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X);

  /** @brief Preconditioned CG (Gauss-Seidel serial, BoomerAMG with elasticity
   * options in parallel) in prec_/cg_, warm-started; the preconditioner is
   * reused per SetPreconditionerReuse(). Equivalent to
   * SetupDefaultPreconditioner(A) followed by SetupCG(*A.Ptr(), *prec_). */
  void SetupDefaultCG(mfem::OperatorHandle& A);

  /** @brief The default preconditioner on A in prec_, rebuilt or reused per
   * SetPreconditionerReuse(). */
  void SetupDefaultPreconditioner(mfem::OperatorHandle& A);

  /** @brief A fresh CG in cg_ on @p op preconditioned by @p prec, with the
   * problem's tolerance and print level, in iterative mode. The operator is
   * set before the preconditioner so that the latter is not reset onto it. */
  void SetupCG(const mfem::Operator& op, mfem::Solver& prec);

  /** @brief Record a solve's iteration count against the preconditioner's
   * baseline; marks it stale when the count has grown past the reuse
   * factor. Derived solvers call this after their outer solve. */
  void NoteIterations(int its);

  /**
   * @brief Make a warm-started solve converge to the same target as a cold
   * one: sets an absolute tolerance rel_tol * sqrt((M B, B)) in the
   * preconditioner norm, which is what a cold start would use. Returns false
   * when B vanishes (solution zero, no solve needed).
   */
  bool SetWarmStartTolerance(mfem::IterativeSolver& solver, mfem::Solver& prec,
                             const mfem::Vector& B) const;

  /** @brief Inner product, global in parallel. */
  mfem::real_t Dot(const mfem::Vector& x, const mfem::Vector& y) const;

  /** @brief The vector mass matrix @f$\int \rho\, u \cdot v@f$ on the
   * displacement space (unit weight when @p rho is null), as a true-dof
   * operator. Returns the form, which must outlive @p M. */
  std::unique_ptr<mfem::BilinearForm> AssembleMassOperator(
      mfem::Coefficient* rho, mfem::OperatorHandle& M);

  /** @brief (Re)assemble the stiffness and set up the solver. */
  void AssembleOperator();

  /** @brief Assemble the operator if it is out of date. */
  void EnsureOperator();

  mfem::FiniteElementSpace* fes_;
#ifdef MFEM_USE_MPI
  mfem::ParFiniteElementSpace* pfes_ = nullptr;
#endif
  const mfemElasticity::Rheology* rheology_;
  std::unique_ptr<ElasticStiffness> stiffness_;

  std::unique_ptr<mfem::BilinearForm> integrators_;  ///< owns integrators
  std::unique_ptr<mfem::BilinearForm> a_;            ///< assembled stiffness
  std::unique_ptr<mfem::LinearForm> b_;              ///< external load
  std::unique_ptr<mfem::GridFunction> u_;            ///< displacement
  mfem::Vector increment_, rhs_;
  mfem::Array<int> ess_tdof_list_;
  mfem::OperatorHandle A_;
  mfem::Vector X_, B_;

  std::unique_ptr<mfem::Solver> prec_;
  std::unique_ptr<mfem::CGSolver> cg_;
  // Preconditioner reuse: the form and matrix the preconditioner was built
  // on are kept alive while it is reused.
  mfem::real_t prec_reuse_ = 2.0;
  bool prec_stale_ = true;
  int prec_baseline_its_ = -1;
  int prec_setups_ = 0;
  int assemblies_ = 0;
  int solves_ = 0;
  long total_its_ = 0;
  std::unique_ptr<mfem::BilinearForm> prec_form_;
  mfem::OperatorHandle prec_A_;

  mfem::real_t t_ = 0.0;
  mfem::real_t rel_tol_ = 1e-12;
  mfem::IterativeSolver::PrintLevel print_level_;
  bool operator_dirty_ = true;
  std::vector<mfem::Coefficient*> td_coefs_;
  std::vector<mfem::VectorCoefficient*> td_vcoefs_;
};

/**
 * @brief Pure traction (Neumann) problem: a traction is applied on the marked
 * boundary attributes; no essential conditions.
 *
 * The stiffness retains the rigid-body null space. CG runs on @f$P A P@f$
 * with the preconditioner @f$P M P@f$, where @f$P@f$ is the Euclidean
 * projector orthogonal to the rigid modes (MakeRigidModeProjector()); the
 * load and the warm start are projected before the solve and the solution
 * after it, so any net force or torque is removed and the displacement is
 * orthogonal to the rigid modes in the true-dof inner product.
 */
class LinearQuasiStaticTractionProblem : public LinearQuasiStaticProblemBase {
 public:
  /**
   * @param traction Boundary traction; registered as time-dependent.
   * @param bdr_marker Boundary attributes it acts on (copied).
   */
  LinearQuasiStaticTractionProblem(mfem::FiniteElementSpace* fes,
                                   const mfemElasticity::Rheology& rheology,
                                   mfem::VectorCoefficient& traction,
                                   const mfem::Array<int>& bdr_marker);

  /**
   * @brief Fix the rigid gauge of the solution by zero net momentum and
   * angular momentum, @f$\int \rho\, u \cdot (a + b \times x) = 0@f$
   * (unit @f$\rho@f$ when null), instead of orthogonality to the rigid
   * modes in the true-dof inner product (the default). Only the rigid
   * component of the displacement changes. May be called at any time.
   */
  void SetMassWeightedGauge(mfem::Coefficient* rho = nullptr);
  /** @brief Back to the true-dof (Euclidean) gauge. */
  void SetEuclideanGauge();

 protected:
  void SetupSolver(mfem::OperatorHandle& A) override;
  bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X) override;

 private:
  /** @brief The rigid-mode projector (built on first use; the space does
   * not change). */
  const NullSpaceProjector& RigidModes();

  mfem::Array<int> marker_;
  std::unique_ptr<mfem::BilinearForm> gauge_form_;
  mfem::OperatorHandle gauge_M_;  ///< mass-weighted gauge, if set
  std::unique_ptr<NullSpaceProjector> projector_;
  std::unique_ptr<ProjectedOperator> projected_op_;    ///< P A P
  std::unique_ptr<ProjectedSolver> projected_prec_;    ///< P M P
  std::unique_ptr<ProjectedSolver> projected_;         ///< wraps cg_
};

/**
 * @brief Mixed problem: the displacement is prescribed on one set of
 * boundary attributes and a traction applied on another; all other
 * boundaries are traction-free.
 */
class LinearQuasiStaticClampedProblem : public LinearQuasiStaticProblemBase {
 public:
  /**
   * @param ess_bdr Boundary attributes with prescribed displacement (copied).
   * @param traction Boundary traction; registered as time-dependent.
   * @param traction_marker Boundary attributes it acts on (copied).
   * @param dirichlet Prescribed displacement (registered as time-dependent);
   * nullptr means homogeneous.
   */
  LinearQuasiStaticClampedProblem(mfem::FiniteElementSpace* fes,
                                  const mfemElasticity::Rheology& rheology,
                                  const mfem::Array<int>& ess_bdr,
                                  mfem::VectorCoefficient& traction,
                                  const mfem::Array<int>& traction_marker,
                                  mfem::VectorCoefficient* dirichlet = nullptr);

 protected:
  void UpdateBoundaryValues(mfem::real_t t) override;

 private:
  mfem::Array<int> ess_bdr_, marker_;
  std::unique_ptr<mfem::VectorConstantCoefficient> zero_;
  mfem::VectorCoefficient* dirichlet_;
};

}  // namespace mfemElasticity
