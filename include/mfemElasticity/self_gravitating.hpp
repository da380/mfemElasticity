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
#include "mfemElasticity/coefficient.hpp"
#include "mfemElasticity/elastic_problem.hpp"
#include "mfemElasticity/poisson.hpp"
#include "mfemElasticity/solvers.hpp"
#include "mfemElasticity/submesh.hpp"

namespace mfemElasticity {

/**
 * @brief A fluid region of a self-gravitating body (see
 * SelfGravitatingElasticProblem): a set of parent-mesh attributes carrying
 * no displacement unknown, its density, and its interfaces with the solid.
 *
 * - @p attributes: the parent-mesh attributes of the fluid (not part of the
 *   displacement SubMesh).
 * - @p density: @f$\rho_F@f$, evaluated on the parent's fluid elements (for
 *   the background potential and, through the fallback below, for
 *   @f$\rho'_F@f$) and, unless @p interface_density is given, on the
 *   *SubMesh's boundary elements* of the interfaces: a coefficient of
 *   position (FunctionCoefficient) serves both; a coefficient keyed by
 *   domain attribute or a GridFunctionCoefficient on the parent does not.
 * - @p density_gradient: @f$\rho'_F = d\rho/d\Phi_0 = g^{-1}\partial_r\rho@f$
 *   on the fluid elements (BarotropicDensityGradientCoefficient, or an
 *   analytic form for radial models). When null it is computed from the
 *   element-wise L2 projection of @p density and the discrete
 *   @f$\nabla\Phi_0@f$.
 * - @p interface_marker: the SubMesh boundary attributes of the fluid's
 *   interfaces with the solid (sized to the SubMesh's bdr_attributes.Max()).
 *   Nothing distinguishes a fluid below the solid from a fluid above: the
 *   sign of @f$\mathbf{m}\cdot\nabla\Phi_0@f$ does that.
 * - @p interface_density: @f$\rho_F@f$ on those boundary elements; default
 *   @p density.
 */
struct FluidRegion {
  mfem::Array<int> attributes;
  mfem::Coefficient* density = nullptr;
  mfem::Coefficient* density_gradient = nullptr;
  mfem::Array<int> interface_marker;
  mfem::Coefficient* interface_density = nullptr;
};

/**
 * @brief Self-gravitating quasi-static linear elastic problem (eq. 3 of Yu,
 * Al-Attar, Syvret & Lloyd 2025, with the fluid regions of their Appendix A
 * / Al-Attar & Tromp 2014 eq. 2.52), implementing
 * QuasiStaticLinearElasticProblem so that the viscoelastic layer runs on it
 * unchanged.
 *
 * **Geometry.** The body @f$M = M_S \cup M_F@f$ is a region of a ball
 * @f$B@f$ whose external boundary is a sphere (a circle in 2-D). The
 * displacement @f$u@f$ lives on a (Par)SubMesh of the *solid* regions
 * @f$M_S@f$ (possibly disconnected, e.g. inner core and mantle), the
 * potential perturbation @f$\phi@f$ on @f$B@f$, and the exterior of @f$B@f$
 * is represented by a PoissonDtNOperator. Fluid regions @f$M_F@f$ (a liquid
 * core) carry no displacement; they enter through their density, the
 * hydrostatic Poisson term, and the interface terms below.
 *
 * **Weak form** (with @f$G@f$ the gravitational constant in the chosen
 * units, @f$\rho@f$ the density and @f$\Phi_0@f$ the background potential,
 * @f$\mathbf{m}@f$ the outward normal of the solid on the fluid–solid
 * interfaces @f$\Sigma_F@f$, @f$\rho_F@f$ the fluid-side density there,
 * and @f$\rho'_F = d\rho/d\Phi_0@f$ in the fluid):
 * @f[
 *   a(u,v) + \tfrac12\int_{M_S} \rho\,[v\cdot\nabla(u\cdot\nabla\Phi_0)
 *     + u\cdot\nabla(v\cdot\nabla\Phi_0) - (v\cdot\nabla\Phi_0)\,\mathrm{div}\,u
 *     - (u\cdot\nabla\Phi_0)\,\mathrm{div}\,v]
 *     - \int_{\Sigma_F} \rho_F\,(\mathbf{m}\cdot\nabla\Phi_0)
 *       (\mathbf{m}\cdot u)(\mathbf{m}\cdot v)
 *     + c(\phi, v) = \ell_u(v) - c(\psi, v),
 * @f]
 * @f[
 *   \frac{1}{4\pi G}\Big[\int_B \nabla\phi\cdot\nabla\chi
 *     + \mathrm{DtN}(\phi,\chi)\Big] + \int_{M_F} \rho'_F\,\phi\,\chi
 *     + c(\chi, u) = \ell_\phi(\chi) - \int_{M_F} \rho'_F\,\psi\,\chi,
 * @f]
 * with the coupling form
 * @f$c(\phi, v) = \int_{M_S} \rho\,\nabla\phi\cdot v
 *   - \int_{\Sigma_F} \rho_F\,\phi\,(\mathbf{m}\cdot v)@f$,
 * @f$a@f$ the bulk/deviatoric elastic form of LinearElasticProblemBase, and
 * @f$\psi@f$ an optional applied (tidal) potential. Without fluid regions
 * the interface and @f$\rho'_F@f$ terms are absent and this is eq. 3 of the
 * paper. A surface mass load @f$\sigma@f$ (mass per unit area, positive
 * when mass is added) on marked boundary attributes of the SubMesh
 * contributes @f$\ell_u(v) = -\int \sigma\,\nabla\Phi_0\cdot v@f$ and
 * @f$\ell_\phi(\chi) = -\int \sigma\,\chi@f$. Further displacement loads
 * can be added to ExternalLoad() and further potential loads to
 * ExternalPotentialLoad(); AddForce() acts on the displacement as usual.
 *
 * **Background potential.** Solved once from the density of the solid
 * (injected from the SubMesh) and of the fluids (on the parent) with the
 * plain Poisson–DtN operator, unless a coefficient is supplied, e.g. from a
 * radial model. The fluid mass term @f$\int_{M_F}\rho'_F\phi\chi@f$ is
 * assembled afterwards (it may depend on @f$\nabla\Phi_0@f$) and added to
 * the potential block.
 *
 * **Solvers** (SetSolverType()):
 * - BlockMINRES (default): MINRES on the @f$[u;\phi]@f$ block system with a
 *   block-diagonal preconditioner (Gauss-Seidel or BoomerAMG on the
 *   displacement block and on the shifted Laplacian), restricted by a
 *   ProjectedSolver to displacements orthogonal to the rigid modes (and,
 *   in 2-D, potentials orthogonal to the constant). Typically an order of
 *   magnitude cheaper than SchurCG.
 * - SchurCG: the potential is eliminated and CG runs on the displacement
 *   Schur complement @f$S = A_{uu} - C A_{\phi\phi}^{-1} C^T@f$, which is
 *   symmetric and, for a gravitationally stable body, positive on the
 *   complement of the rigid modes; each application costs one inner
 *   Poisson solve. Kept as the gauge-clean reference.
 * Both warm start across solves. The inner potential solves use CG; with
 * fluid regions the potential block is @f$(K + \mathrm{DtN})/4\pi G +
 * M_F@f$ with @f$M_F \le 0@f$ where density increases downward, positive
 * for Earth-like models but not by a wide margin —
 * PotentialBlockMinEigenvalue() reports it.
 *
 * **Null space and gauge.** The global rigid modes, paired with their
 * potential @f$-u_r\cdot\nabla\Phi_0@f$ extended through the fluid, are
 * near-null vectors of the coupled system (see RigidModeResiduals()), so
 * the system is regularised by restricting the displacement to the
 * orthogonal complement of the rigid modes in the Euclidean true-dof inner
 * product. Both solvers solve exactly that restricted system (the block
 * form of BlockMINRES has the Schur complement of SchurCG), so they agree
 * to solver tolerance and share the gauge: no rigid-body component in the
 * displacement, the potential the discrete solution of its equation for
 * that displacement. A solid region enclosed by fluid (an inner core) has,
 * for a spherically symmetric model, its own near-null rotations;
 * AddRegionRotations() removes them as well. Its translations are restored
 * gravitationally (the Slichter mode) and must not be projected; being
 * soft, they are where discretisation error shows first.
 *
 * In two dimensions the constant potential is a null vector of the
 * Laplace–DtN block; potential loads are made compatible by subtracting a
 * uniform flux through the outer boundary, and the potential is restricted
 * to the complement of the constant (every potential solve runs on
 * @f$P A_{\phi\phi} P@f$, the block solver restricts the potential block
 * likewise). With fluid regions this is a regularisation: the constant is
 * not null for @f$M_F \ne 0@f$, and the 2-D problem is not gauge invariant
 * (a constant potential shift loads the interfaces through the fluid
 * pressure), so 2-D fluid results are for testing on small meshes only.
 *
 * **Serial and parallel** in one class: pass ParFiniteElementSpaces on a
 * ParSubMesh and its ParMesh for the parallel path. The solid density is
 * evaluated on the SubMesh; fluid densities on the parent (see FluidRegion).
 *
 * SetRelaxationWeights() reassembles the displacement block only (the
 * interface term is reassembled with it, harmlessly); the potential block,
 * the coupling and the DtN are built once.
 */
class SelfGravitatingElasticProblem : public LinearElasticProblemBase {
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
   * @p fes_phi); solved from the densities when null.
   * @param fluids Fluid regions (copied); their coefficients must outlive
   * the problem. Empty for a solid body.
   */
  SelfGravitatingElasticProblem(
      mfem::FiniteElementSpace* fes_u, mfem::FiniteElementSpace* fes_phi,
      const mfemElasticity::Rheology& rheology, mfem::Coefficient& density,
      mfem::real_t gravitational_constant, int dtn_degree,
      mfem::Coefficient* background_potential = nullptr,
      const std::vector<FluidRegion>& fluids = {});

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

  /**
   * @brief Apply a (tidal) potential @f$\psi@f$ (registered as
   * time-dependent; interpolated on the potential space at each
   * AssembleForce()). Its load is @f$-c(\psi, v)@f$ on the displacement and
   * @f$-\int_{M_F}\rho'_F\psi\chi@f$ on the potential, i.e. the coupling and
   * fluid-mass operators applied to the interpolant.
   */
  void SetTidalPotential(mfem::Coefficient& psi);

  // --- solver controls ------------------------------------------------------

  void SetSolverType(SolverType type);
  SolverType GetSolverType() const { return type_; }

  /** @brief Relative tolerance of the inner potential solves (default: a
   * hundredth of RelTol(); never below 1e-13, the round-off floor of CG's
   * squared-residual criterion). */
  void SetInnerRelTol(mfem::real_t tol);

  /** @brief Shift @f$\epsilon@f$ of the potential preconditioner
   * @f$(K + \epsilon M)/4\pi G@f$ (default 1e-3). */
  void SetPreconditionerShift(mfem::real_t eps);

  /** @brief Print level of the inner potential solves (quiet by default). */
  void SetInnerPrintLevel(mfem::IterativeSolver::PrintLevel level);

  /** @brief Replace the background potential and mark the operator stale. */
  void SetBackgroundPotential(mfem::Coefficient& phi0);

  /**
   * @brief Project out the rigid rotations of the solid region(s) with the
   * given SubMesh attributes (a solid inner core enclosed by fluid), which
   * are near-null vectors when density and background potential are
   * invariant under those rotations. May be called at any time before or
   * between solves; translations of the region are never projected.
   */
  void AddRegionRotations(const mfem::Array<int>& solid_attributes);

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

  /** @brief The coupling operator @f$C@f$ (potential true dofs to
   * displacement true dofs) of the weak form. */
  const mfem::Operator& Coupling() const { return *C_op_; }

  /** @brief The potential block @f$A_{\phi\phi} = (K + \mathrm{DtN})/4\pi G
   * + M_F@f$ on true dofs. */
  const mfem::Operator& PotentialOperator() const { return *A_phiphi_; }

  /** @brief The potential load @f$\ell_\phi@f$ of the last AssembleForce()
   * on true dofs (tidal part and 2-D compatibility included). */
  const mfem::Vector& PotentialLoad() const { return B_phi_; }
  const SubMeshDofInjection& Injection() const { return *injection_; }
  mfem::real_t GravitationalConstant() const { return G_; }
  mfem::Coefficient& Density() const { return *rho_; }

  /** @brief Iterations of the outer solver in the last Solve(). */
  int LastOuterIterations() const { return outer_its_; }

  /** @brief Inner potential-solve iterations accumulated over the last
   * Solve() (zero for BlockMINRES). */
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

  /** @brief The residual of RigidModeResiduals() for an arbitrary true-dof
   * displacement @p u (normalised by its norm): @f$\|S u\| /
   * (\|A_{uu}\|_{\max} \|u\|)@f$. */
  mfem::real_t ModeResidual(const mfem::Vector& u);

  /** @brief Rigid-body modes (orthonormal, true dofs), translations then
   * rotations then any region rotations; the null space projected out of
   * the displacement. */
  const NullSpaceProjector& RigidModes() const { return *projector_u_; }

  /**
   * @brief Diagnostic: the extreme Ritz values of the potential block
   * @f$A_{\phi\phi} = (K + \mathrm{DtN})/4\pi G + M_F@f$ (Euclidean
   * inner product) after @p lanczos_steps Lanczos steps from a fixed
   * pseudo-random start. Returns the smallest; the largest is stored in
   * @p largest when given. In 2-D the constant (projected out of every
   * potential solve) is deflated. Negative means the fluid mass term has
   * made the block indefinite (the inner CG solves then cannot be trusted).
   */
  mfem::real_t PotentialBlockMinEigenvalue(int lanczos_steps = 40,
                                           mfem::real_t* largest = nullptr);

  bool HasFluidRegions() const { return !fluids_.empty(); }
  const std::vector<FluidRegion>& FluidRegions() const { return fluids_; }

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
  void SetupPotentialSolver();
  void SetupFluidMass();
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

  /** @brief Interpolate the tidal potential at the current time and form
   * its loads C Psi and M_F Psi on true dofs. */
  void AssembleTidalLoad();

  /** @brief @f$\rho_F@f$ on the interfaces of region @p i (its
   * interface_density or density). */
  mfem::Coefficient& InterfaceDensity(const FluidRegion& f) const {
    return f.interface_density ? *f.interface_density : *f.density;
  }

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

  // fluid regions
  std::vector<FluidRegion> fluids_;
  std::list<mfem::Array<int>> fluid_markers_;  ///< parent-attribute markers
  std::vector<std::unique_ptr<mfem::Coefficient>> fluid_coefs_;
  std::unique_ptr<mfem::L2_FECollection> l2_fec_;
  std::unique_ptr<mfem::FiniteElementSpace> l2_fes_;
  std::vector<std::unique_ptr<mfem::GridFunction>> rho_fluid_l2_;
  std::unique_ptr<BoundaryNormalDotCoefficient> m_dot_grad_phi0_;

  // potential operators: A_lap = (K + DtN)/4piG; A_phiphi = A_lap + M_F
  std::unique_ptr<PoissonDtNOperator> dtn_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::RAPOperator> dtn_rap_;
#endif
  const mfem::Operator* dtn_op_ = nullptr;
  std::unique_ptr<mfem::BilinearForm> k_phi_form_, k_shift_form_,
      m_fluid_form_;
  mfem::OperatorHandle K_phi_, K_shift_, M_fluid_;
  std::unique_ptr<mfem::SumOperator> A_lap_, A_full_;
  const mfem::Operator* A_phiphi_ = nullptr;
  std::unique_ptr<mfem::Solver> prec_phi_;
  std::unique_ptr<mfem::CGSolver> cg_phi_;
  // 2-D: the constant is projected out, CG runs on P A_phiphi P with the
  // preconditioner P M P.
  std::unique_ptr<NullSpaceProjector> projector_c_;
  std::unique_ptr<ProjectedOperator> projected_phi_op_;
  std::unique_ptr<ProjectedSolver> projected_phi_, projected_prec_phi_;
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
  mfem::Coefficient* psi_ = nullptr;
  std::unique_ptr<mfem::GridFunction> psi_gf_;
  mfem::Vector Psi_true_, tidal_u_, B_eff_;

  // 2-D compatibility
  mfem::Vector ones_, L_outer_;
  mfem::real_t outer_length_ = 0.0;

  // null spaces
  std::unique_ptr<NullSpaceProjector> projector_u_, projector_block_;
  std::vector<mfem::Vector> rigid_true_;
  int num_global_modes_ = 0;

  // outer solvers
  SolverType type_ = SolverType::BlockMINRES;
  std::unique_ptr<SchurOperator> schur_;
  std::unique_ptr<ProjectedOperator> projected_op_;
  std::unique_ptr<ProjectedSolver> projected_;
  mfem::Array<int> offsets_;
  std::unique_ptr<mfem::BlockOperator> block_op_;
  std::unique_ptr<mfem::BlockDiagonalPreconditioner> block_prec_;
  std::unique_ptr<ProjectedSolver> projected_prec_;  ///< P M P, both solvers
  std::unique_ptr<mfem::MINRESSolver> minres_;
  std::unique_ptr<mfem::BlockVector> X_block_, B_block_;
  mfem::Vector rhs_s_, w_;

  // tolerances and statistics
  mfem::real_t inner_rel_tol_ = 1e-13;
  bool inner_tol_set_ = false;
  mfem::real_t shift_ = 1e-3;
  mfem::IterativeSolver::PrintLevel inner_print_level_;
  int outer_its_ = 0;
  mutable int inner_its_ = 0;
};

}  // namespace mfemElasticity
