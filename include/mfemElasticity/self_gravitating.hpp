/**
 * @file self_gravitating.hpp
 * @brief Quasi-static elasticity of a self-gravitating body: the displacement
 * on a SubMesh of the body coupled to the gravitational potential
 * perturbation on the enclosing ball, with a Dirichlet-to-Neumann outer
 * condition.
 */

#pragma once

#include <list>
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "mfemElasticity/elastic_problem.hpp"
#include "mfemElasticity/poisson.hpp"
#include "mfemElasticity/solvers.hpp"
#include "mfemElasticity/submesh.hpp"

namespace mfemElasticity {

/**
 * @brief Self-gravitating quasi-static linear elastic problem (eq. 3 of Yu,
 * Al-Attar, Syvret & Lloyd 2025), implementing
 * QuasiStaticLinearElasticProblem so that the viscoelastic layer runs on it
 * unchanged.
 *
 * **Geometry.** The body @f$M@f$ is a (Par)SubMesh of a ball @f$B@f$ whose
 * external boundary is a sphere (a circle in 2-D). The displacement @f$u@f$
 * lives on @f$M@f$, the potential perturbation @f$\phi@f$ on @f$B@f$, and
 * the exterior of @f$B@f$ is represented by a PoissonDtNOperator.
 *
 * **Weak form** (with @f$G@f$ the gravitational constant in the chosen
 * units, @f$\rho@f$ the density on @f$M@f$ and @f$\Phi_0@f$ the background
 * potential):
 * @f[
 *   a(u,v) + \tfrac12\int_M \rho\,[v\cdot\nabla(u\cdot\nabla\Phi_0)
 *     + u\cdot\nabla(v\cdot\nabla\Phi_0) - (v\cdot\nabla\Phi_0)\,\mathrm{div}\,u
 *     - (u\cdot\nabla\Phi_0)\,\mathrm{div}\,v]
 *     + \int_M \rho\,\nabla\phi\cdot v = \ell_u(v),
 * @f]
 * @f[
 *   \frac{1}{4\pi G}\Big[\int_B \nabla\phi\cdot\nabla\psi
 *     + \mathrm{DtN}(\phi,\psi)\Big] + \int_M \rho\, u\cdot\nabla\psi
 *     = \ell_\phi(\psi),
 * @f]
 * where @f$a@f$ is the bulk/deviatoric elastic form of ElasticProblemBase.
 * A surface mass load @f$\sigma@f$ (mass per unit area, positive when mass
 * is added) on marked boundary attributes of @f$M@f$ contributes
 * @f$\ell_u(v) = -\int \sigma\,\nabla\Phi_0\cdot v@f$ and
 * @f$\ell_\phi(\psi) = -\int \sigma\,\psi@f$. Further displacement loads
 * can be added to ExternalLoad() and further potential loads to
 * ExternalPotentialLoad(); AddForce() acts on the displacement as usual.
 *
 * **Background potential.** Solved once from @f$\rho@f$ on @f$B@f$ (with the
 * DtN) unless a coefficient is supplied, e.g. from a radial model.
 *
 * **Solvers** (SetSolverType()):
 * - BlockMINRES (default): MINRES on the @f$[u;\phi]@f$ block system with a
 *   block-diagonal preconditioner (Gauss-Seidel or BoomerAMG on the
 *   displacement block and on the shifted Laplacian), projecting out the
 *   coupled near-null vectors @f$(u_r, \phi_r)@f$, @f$\phi_r =
 *   -A_{\phi\phi}^{-1} C^T u_r@f$ the discrete potential of the rigid mode
 *   (the continuous @f$-u_r\cdot\nabla\Phi_0@f$), with a ProjectedSolver.
 *   Typically an order of magnitude cheaper than SchurCG.
 * - SchurCG: the potential is eliminated and CG runs on the displacement
 *   Schur complement @f$S = A_{uu} - C A_{\phi\phi}^{-1} C^T@f$, which is
 *   symmetric and, for a gravitationally stable body, positive on the
 *   complement of the rigid modes; each application costs one inner
 *   Poisson solve. Kept as the gauge-clean reference.
 * Both warm start across solves.
 *
 * **Gauge.** The discrete rigid modes are only near-null vectors of the
 * coupled system (see RigidModeResiduals()), so the solvers regularise the
 * system by projection and the output gauge has to be fixed explicitly. Both
 * solvers return the same gauge: the displacement has no rigid-body
 * component in the Euclidean true-dof inner product and the potential is
 * the discrete solution of its equation for that displacement (for
 * BlockMINRES this costs one extra potential solve per Solve(); the MINRES
 * iterate itself is kept as the warm start). The two solvers then agree to
 * the level of the rigid-mode residuals.
 *
 * In two dimensions the constant potential is a null vector of the
 * potential block; potential loads are made compatible by subtracting a
 * uniform flux through the outer boundary, and the constant is projected out
 * of every potential solve.
 *
 * **Serial and parallel** in one class: pass ParFiniteElementSpaces on a
 * ParSubMesh and its ParMesh for the parallel path. Every coefficient is
 * evaluated on the SubMesh, so @f$\rho@f$ need only be defined on the body.
 *
 * SetEffectiveShearModulus() reassembles the displacement block only; the
 * potential block, the coupling and the DtN are built once.
 */
class SelfGravitatingElasticProblem : public ElasticProblemBase {
 public:
  enum class SolverType { SchurCG, BlockMINRES };

  /**
   * @param fes_u Displacement space (vdim = dim) on a (Par)SubMesh of the
   * ball; not owned.
   * @param fes_phi Scalar potential space on the parent (Par)Mesh; not
   * owned.
   * @param rheology The material of the body; must outlive the problem.
   * @param density Density on the body; evaluated on the SubMesh.
   * @param gravitational_constant @f$G@f$ in the units of the problem.
   * @param dtn_degree Truncation degree of the DtN expansion.
   * @param background_potential Optional @f$\Phi_0@f$ (projected onto
   * @p fes_phi); solved from @p density when null.
   */
  SelfGravitatingElasticProblem(mfem::FiniteElementSpace* fes_u,
                                mfem::FiniteElementSpace* fes_phi,
                                const GeneralisedMaxwellRheology& rheology,
                                mfem::Coefficient& density,
                                mfem::real_t gravitational_constant,
                                int dtn_degree,
                                mfem::Coefficient* background_potential =
                                    nullptr);

  // --- loads ----------------------------------------------------------------

  /**
   * @brief Add a surface mass load @f$\sigma@f$ (registered as
   * time-dependent) on the SubMesh boundary attributes marked in
   * @p bdr_marker (sized to the SubMesh's bdr_attributes.Max(); copied).
   */
  void SetSurfaceLoad(mfem::Coefficient& sigma,
                      const mfem::Array<int>& bdr_marker);

  /**
   * @brief Potential load @f$\ell_\phi@f$ as a linear form on the shadow of
   * the potential space on the SubMesh; add integrators before the first
   * AssembleForce(). Markers refer to the SubMesh.
   */
  mfem::LinearForm& ExternalPotentialLoad() { return *b_phi_; }

  // --- solver controls ------------------------------------------------------

  void SetSolverType(SolverType type);
  SolverType GetSolverType() const { return type_; }

  /** @brief Relative tolerance of the inner potential solves (default: a
   * hundredth of RelTol(), at least 1e-15). */
  void SetInnerRelTol(mfem::real_t tol);

  /** @brief Shift @f$\epsilon@f$ of the potential preconditioner
   * @f$(K + \epsilon M)/4\pi G@f$ (default 1e-3). */
  void SetPreconditionerShift(mfem::real_t eps);

  /** @brief Print level of the inner potential solves (quiet by default). */
  void SetInnerPrintLevel(mfem::IterativeSolver::PrintLevel level);

  /** @brief Replace the background potential and mark the operator stale. */
  void SetBackgroundPotential(mfem::Coefficient& phi0);

  // --- outputs --------------------------------------------------------------

  mfem::FiniteElementSpace& PotentialSpace() { return *fes_phi_; }

  /** @brief Potential perturbation on the ball. */
  const mfem::GridFunction& Potential() const { return *phi_; }

  /** @brief Potential perturbation restricted to the body (shadow space). */
  const mfem::GridFunction& PotentialOnBody() const { return *phi_shadow_; }

  const mfem::GridFunction& BackgroundPotential() const { return *phi0_; }
  const mfem::GridFunction& BackgroundPotentialOnBody() const {
    return *phi0_shadow_;
  }

  /** @brief @f$\nabla\Phi_0@f$ on the body, as used in the operator. */
  mfem::VectorCoefficient& BackgroundGravity() const {
    return *grad_phi0_shadow_;
  }

  const PoissonDtNOperator& DtN() const { return *dtn_; }
  const SubMeshDofInjection& Injection() const { return *injection_; }
  mfem::real_t GravitationalConstant() const { return G_; }
  mfem::Coefficient& Density() const { return *rho_; }

  /** @brief Iterations of the outer solver in the last Solve(). */
  int LastOuterIterations() const { return outer_its_; }

  /** @brief Inner potential-solve iterations accumulated over the last
   * Solve() (for BlockMINRES: the gauge-fixing solve only). */
  int LastInnerIterations() const { return inner_its_; }

  /**
   * @brief Diagnostic: how well the discretisation preserves the rigid-body
   * null space. Returns, for each rigid mode @f$u_r@f$ (translations then
   * rotations), @f$\|S u_r\| / (\|A_{uu}\|_{\max} \|u_r\|)@f$ with
   * @f$S@f$ the displacement Schur complement; zero for the continuous
   * problem, and decreasing with refinement for the discrete one. The
   * rigid modes are unit vectors in the Euclidean true-dof norm. Assembles
   * the operator if needed; costs one inner solve per mode.
   */
  std::vector<mfem::real_t> RigidModeResiduals();

  /** @brief Rigid-body modes (orthonormal, true dofs), translations then
   * rotations; the null space projected out of the displacement. */
  const NullSpaceProjector& RigidModes() const { return *projector_u_; }

  // --- interface overrides --------------------------------------------------

  void AssembleForce(mfem::real_t t) override;

  /** @brief Registers "displacement", "potential" and
   * "background_potential" (the latter two on the body). */
  void RegisterFields(mfem::DataCollection& dc) override;

 protected:
  void SetupSolver(mfem::OperatorHandle& A) override;
  bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X) override;

 private:
  /** @brief @f$S x = A_{uu} x - C A_{\phi\phi}^{-1} C^T x@f$. */
  class SchurOperator : public mfem::Operator {
   public:
    SchurOperator(const SelfGravitatingElasticProblem& p,
                  const mfem::Operator& A_uu);
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

   private:
    const SelfGravitatingElasticProblem* p_;
    const mfem::Operator* A_uu_;
    mutable mfem::Vector t_, w_, cw_;
  };

  void SetupPotentialOperators();
  void SetupPotentialPreconditioner();
  void SetupCoupling();
  void SetupGravityIntegrators();
  void SetupRigidModes();
  void SetupCoupledNullSpace();
  void ComputeBackgroundPotential(mfem::Coefficient* phi0);
  void SetupSchur(mfem::OperatorHandle& A);
  void SetupMinres(mfem::OperatorHandle& A);

  /** @brief Cold-started inner solve of @f$A_{\phi\phi} x = b@f$ on true
   * dofs; returns convergence, accumulates inner_its_. */
  bool SolvePotential(const mfem::Vector& b, mfem::Vector& x) const;

  /** @brief Dual L-vector of @p fes to true dofs (P^T). */
  void ToTrueDofs(const mfem::FiniteElementSpace& fes, const mfem::Vector& L,
                  mfem::Vector& T) const;

  /** @brief In 2-D: make a potential load compatible with the constant. */
  void MakeCompatible(mfem::Vector& B_phi) const;

  /** @brief Push true-dof potential @p Phi into phi_ and phi_shadow_. */
  void DistributePotential(const mfem::Vector& Phi);

  bool ParallelPotential() const;

  // geometry and spaces
  int dim_;
  mfem::FiniteElementSpace* fes_phi_;
#ifdef MFEM_USE_MPI
  mfem::ParFiniteElementSpace* pfes_phi_ = nullptr;
#endif
  std::unique_ptr<mfem::FiniteElementSpace> shadow_phi_;
  std::unique_ptr<SubMeshDofInjection> injection_;

  // physics
  mfem::Coefficient* rho_;
  mfem::real_t G_, four_pi_G_;
  int dtn_degree_;
  mfem::ConstantCoefficient one_, inv_four_pi_G_, shift_coef_;
  std::unique_ptr<mfem::ProductCoefficient> half_rho_, minus_half_rho_;
  std::unique_ptr<mfem::ScalarVectorProductCoefficient> minus_half_rho_grad_;

  // potential fields
  std::unique_ptr<mfem::GridFunction> phi0_, phi0_shadow_, phi_, phi_shadow_;
  std::unique_ptr<mfem::GradientGridFunctionCoefficient> grad_phi0_,
      grad_phi0_shadow_;
  mfem::Vector Phi_true_;

  // potential operators
  std::unique_ptr<PoissonDtNOperator> dtn_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::RAPOperator> dtn_rap_;
#endif
  const mfem::Operator* dtn_op_ = nullptr;
  std::unique_ptr<mfem::BilinearForm> k_phi_form_, k_shift_form_;
  mfem::OperatorHandle K_phi_, K_shift_;
  std::unique_ptr<mfem::SumOperator> A_phiphi_;
  std::unique_ptr<mfem::Solver> prec_phi_;
  std::unique_ptr<mfem::CGSolver> cg_phi_;
  std::unique_ptr<mfem::OrthoSolver> ortho_phi_;
  mfem::Solver* phi_solver_ = nullptr;

  // coupling C (trial phi on B, test u on M) and its transpose, true dofs
  std::unique_ptr<mfem::MixedBilinearForm> c_form_;
  mfem::OperatorHandle C_;
  std::unique_ptr<mfem::Operator> Ct_owned_;
  const mfem::Operator* C_op_ = nullptr;
  const mfem::Operator* Ct_op_ = nullptr;

  // loads
  std::unique_ptr<mfem::LinearForm> b_phi_;
  mfem::Vector B_phi_;
  std::list<mfem::Array<int>> load_markers_;
  std::vector<std::unique_ptr<mfem::Coefficient>> load_coefs_;
  std::vector<std::unique_ptr<mfem::VectorCoefficient>> load_vcoefs_;

  // 2-D compatibility
  mfem::Vector ones_, L_outer_;
  mfem::real_t outer_length_ = 0.0;

  // null spaces
  std::unique_ptr<NullSpaceProjector> projector_u_, projector_block_;
  std::vector<mfem::Vector> rigid_true_;

  // outer solvers
  SolverType type_ = SolverType::BlockMINRES;
  std::unique_ptr<SchurOperator> schur_;
  std::unique_ptr<ProjectedOperator> projected_op_;
  std::unique_ptr<ProjectedSolver> projected_;
  mfem::Array<int> offsets_;
  std::unique_ptr<mfem::BlockOperator> block_op_;
  std::unique_ptr<mfem::BlockDiagonalPreconditioner> block_prec_;
  std::unique_ptr<mfem::MINRESSolver> minres_;
  std::unique_ptr<mfem::BlockVector> X_block_, B_block_;
  mfem::Vector rhs_s_, w_;

  // tolerances and statistics
  mfem::real_t inner_rel_tol_ = 1e-14;
  bool inner_tol_set_ = false;
  mfem::real_t shift_ = 1e-3;
  mfem::IterativeSolver::PrintLevel inner_print_level_;
  int outer_its_ = 0;
  mutable int inner_its_ = 0;
};

}  // namespace mfemElasticity
