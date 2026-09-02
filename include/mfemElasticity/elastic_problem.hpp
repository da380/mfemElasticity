/**
 * @file elastic_problem.hpp
 * @brief Quasi-static linear elastic problems: the abstract interface used by
 * the viscoelastic layer, a base class owning the shared bookkeeping (serial
 * and parallel), and two reference problems (pure traction, clamped).
 */

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"
#include "mfemElasticity/rheology.hpp"
#include "mfemElasticity/solvers.hpp"

namespace mfemElasticity {

/**
 * @brief Abstract interface for quasi-static linear elastic problems.
 *
 * Per evaluation time @f$t@f$ the protocol is
 * @code
 *   AssembleForce(t);        // all time-dependent data to t; external loads;
 *                            // increments cleared
 *   AddForce(i, f); ...      // superpose dual vectors on displacement field i
 *   Solve();                 // displacement(s) <- K^{-1}(external +
 * increments)
 * @endcode
 *
 * - AssembleForce(t) is called at every stage of a time integrator, with
 *   possibly non-monotone t; it must be cheap and idempotent at fixed t.
 * - AddForce(i, f) takes the vdof (L-vector) layout of DisplacementSpace(i),
 *   i.e. the layout of a LinearForm on that space before FormLinearSystem.
 *   In parallel the problem applies the prolongation transpose inside
 *   Solve(); callers never handle true dofs. AddForce accumulates.
 * - Solve() may be internally iterative or nonlinear, but is a black box to
 *   callers; linearity in the *forces* is part of the contract. It returns
 *   false if the linear solver did not converge.
 * - Problems carrying more unknowns than the displacement (a gravitational
 *   potential, say) keep them internal: the interface only ever refers to
 *   the displacement field(s).
 *
 * Implicit and exponential-trapezoid viscoelastic stepping eliminate the
 * internal variables and need the stiffness reassembled with the
 * *effective* modulus @f$C_\infty + \sum_k \beta_k C_k@f$, with pointwise
 * relaxation weights @f$\beta_k@f$ (see ElasticStiffness); problems that
 * can do so advertise it through SupportsRelaxationWeights().
 */
class QuasiStaticLinearElasticProblem {
 public:
  virtual ~QuasiStaticLinearElasticProblem() = default;

  /** @brief Number of solid regions carrying a displacement unknown. */
  virtual int NumDisplacementFields() const = 0;

  /** @brief The (vector) space of displacement field @p i. */
  virtual mfem::FiniteElementSpace& DisplacementSpace(int i = 0) = 0;

  /** @brief Read-only access to displacement field @p i. */
  virtual const mfem::GridFunction& Displacement(int i = 0) const = 0;

  /** @brief The rheology field @p i's operator was assembled with. */
  virtual const mfemElasticity::Rheology& Rheology(int i = 0) const = 0;

  /** @brief Bring all time-dependent data to time @p t and reset forcing. */
  virtual void AssembleForce(mfem::real_t t) = 0;

  /** @brief Superpose a dual vector (LinearForm layout) on field @p i. */
  virtual void AddForce(int i, const mfem::Vector& f) = 0;

  /** @brief Convenience for single-field problems. */
  void AddForce(const mfem::Vector& f) { AddForce(0, f); }

  /** @brief Solve for the displacement(s); false on solver failure. */
  virtual bool Solve() = 0;

  /** @brief Whether SetRelaxationWeights() is available. */
  virtual bool SupportsRelaxationWeights() const { return false; }

  /**
   * @brief Reassemble field @p i's stiffness with @f$C_\infty + \sum_k
   * \beta_k C_k@f$, one weight coefficient per branch of its rheology
   * (typically nodal fields on the internal-variable mesh). The problem
   * must invalidate its solver setup and reassemble on every call (the same
   * coefficient objects may carry new values). The coefficients must outlive
   * the next call to SetRelaxationWeights() or ClearRelaxationWeights().
   */
  virtual void SetRelaxationWeights(
      int /*i*/, const std::vector<mfem::Coefficient*>& /*beta*/) {
    MFEM_ABORT("relaxation weights not supported by this problem");
  }

  /** @brief Restore the unrelaxed modulus @f$C_U@f$ on every field. */
  virtual void ClearRelaxationWeights() {}

  /** @brief Register output fields with a DataCollection. */
  virtual void RegisterFields(mfem::DataCollection& dc) = 0;
};


/**
 * @brief Base class implementing the interface for a single displacement
 * field on a serial or parallel space.
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
class LinearElasticProblemBase : public QuasiStaticLinearElasticProblem {
 public:
  /**
   * @param fes Displacement space (vdim = space dimension); serial or
   * parallel; not owned.
   * @param rheology The material; not owned, must outlive the problem.
   */
  LinearElasticProblemBase(mfem::FiniteElementSpace* fes,
                     const mfemElasticity::Rheology& rheology);

  int NumDisplacementFields() const override { return 1; }
  mfem::FiniteElementSpace& DisplacementSpace(int i = 0) override;
  const mfem::GridFunction& Displacement(int i = 0) const override;
  const mfemElasticity::Rheology& Rheology(int i = 0) const override;

  void AssembleForce(mfem::real_t t) override;
  void AddForce(int i, const mfem::Vector& f) override;
  using QuasiStaticLinearElasticProblem::AddForce;
  bool Solve() override;

  bool SupportsRelaxationWeights() const override { return true; }
  void SetRelaxationWeights(int i,
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
   * reused per SetPreconditionerReuse(). */
  void SetupDefaultCG(mfem::OperatorHandle& A);

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
 * The stiffness retains the rigid-body null space, so the CG solve is wrapped
 * in a RigidBodySolver, which also projects the load orthogonal to the rigid
 * modes (any net force or torque is removed).
 */
class TractionProblem : public LinearElasticProblemBase {
 public:
  /**
   * @param traction Boundary traction; registered as time-dependent.
   * @param bdr_marker Boundary attributes it acts on (copied).
   */
  TractionProblem(mfem::FiniteElementSpace* fes,
                  const mfemElasticity::Rheology& rheology,
                  mfem::VectorCoefficient& traction,
                  const mfem::Array<int>& bdr_marker);

 protected:
  void SetupSolver(mfem::OperatorHandle& A) override;
  bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X) override;

 private:
  mfem::Array<int> marker_;
  std::unique_ptr<RigidBodySolver> rigid_;
};

/**
 * @brief Mixed problem: the displacement is prescribed on one set of
 * boundary attributes and a traction applied on another; all other
 * boundaries are traction-free.
 */
class ClampedProblem : public LinearElasticProblemBase {
 public:
  /**
   * @param ess_bdr Boundary attributes with prescribed displacement (copied).
   * @param traction Boundary traction; registered as time-dependent.
   * @param traction_marker Boundary attributes it acts on (copied).
   * @param dirichlet Prescribed displacement (registered as time-dependent);
   * nullptr means homogeneous.
   */
  ClampedProblem(mfem::FiniteElementSpace* fes,
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
