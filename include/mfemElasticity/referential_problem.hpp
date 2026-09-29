/**
 * @file referential_problem.hpp
 * @brief The general linearised referential problem: quasi-static
 * elasticity about an arbitrarily pre-stressed equilibrium of a general
 * reference body, with referential self-gravity
 * (doc/gravitating_elasticity.md).
 */

#pragma once

#include <list>
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "mfemElasticity/bilininteg.hpp"
#include "mfemElasticity/mappings.hpp"
#include "mfemElasticity/null_space.hpp"
#include "mfemElasticity/poisson.hpp"
#include "mfemElasticity/quasi_static_problem.hpp"
#include "mfemElasticity/rheology.hpp"
#include "mfemElasticity/submesh.hpp"

namespace mfemElasticity {

/**
 * @brief A prescribed radial vacuum-extension operator @f$E@f$: buffer
 * displacement vdofs from body displacement vdofs
 * (doc/gravitating_elasticity.md §3.1, option (b)).
 *
 * Buffer dofs shared with the body (the @f$\partial B@f$ trace) copy their
 * body values exactly (through the SubMesh dof pairing), so
 * @f$u_{\mathrm{ext}}|_{\partial B} = u|_{\partial B}@f$ holds to
 * round-off. Interior buffer nodes at radius @f$r@f$ take the tapered
 * radial interpolation @f$t(r)\,u(x_s)@f$ with
 * @f$t = ((r_{\mathrm{out}} - r)/(r_{\mathrm{out}} - r_b))^2@f$ — so both
 * @f$t@f$ and @f$t'@f$ vanish at the outer (DtN) sphere, keeping
 * @f$a = 1@f$ there — and @f$x_s@f$ the radial projection onto the body
 * surface (pulled inside by @p pullback for robust point location; the
 * interior rule is a gauge choice, so the pull-back is harmless). Any
 * other smooth extension is an equally valid gauge: agreement of
 * observables between two @f$E@f$s is a gauge-invariance test.
 *
 * Both spaces must share the body space's FiniteElementCollection object
 * and live on SubMeshes of one parent. Serial.
 */
std::unique_ptr<mfem::SparseMatrix> NewRadialVacuumExtension(
    mfem::FiniteElementSpace& body_fes, mfem::FiniteElementSpace& buffer_fes,
    mfem::real_t r_body, mfem::real_t r_outer, mfem::real_t taper_power = 2.0,
    mfem::real_t pullback = 0.999);

#ifdef MFEM_USE_MPI
/**
 * @brief Parallel radial vacuum extension on true dofs: trace rows through
 * NewSubMeshPairingTrueDofMatrix (cross-rank), interior rows through a
 * query–reply exchange (surface projections gathered to every rank, each
 * locating what its local body elements contain and replying with
 * global-column interpolation rows; unresolved points retry at smaller
 * radii, as in serial). Semantics as the serial overload.
 */
std::unique_ptr<mfem::HypreParMatrix> NewRadialVacuumExtension(
    mfem::ParFiniteElementSpace& body_fes,
    mfem::ParFiniteElementSpace& buffer_fes, mfem::real_t r_body,
    mfem::real_t r_outer, mfem::real_t taper_power = 2.0,
    mfem::real_t pullback = 0.999);
#endif

/**
 * @brief The constitutive state of the linearised referential problem as a
 * Rheology: the second elastic tensor @f$\hat C@f$ at equilibrium (Mandel,
 * classical symmetries), the second Piola–Kirchhoff equilibrium stress
 * @f$\mathbf{S}_e@f$, and the equilibrium mapping @f$\varphi_e@f$. Its
 * stiffness is the total-Lagrangian split
 * MaterialStiffnessIntegrator + GeometricStiffnessIntegrator
 * (doc/gravitating_elasticity.md §2). No branches: relaxation weights are
 * no-ops (a viscoelastic extension supplies branch tensors later).
 *
 * All inputs are non-owning and must outlive the rheology. Remember the
 * moduli convention: @f$\hat C@f$ is the *bare* tensor
 * (BareElasticTensorCoefficient converts seismological moduli).
 */
class ReferentialElasticRheology : public Rheology {
 public:
  ReferentialElasticRheology(int dim, mfem::MatrixCoefficient& C,
                             mfem::MatrixCoefficient& S_e,
                             Diffeomorphism& phi_e);

  int SpaceDim() const override { return dim_; }
  int NumBranches() const override { return 0; }
  mfem::Coefficient& RelaxationTime(int k) const override;
  const RelaxationLaw* Law(int k) const override;
  bool TraceFreeInternalVariables() const override { return true; }
  void BranchModulus(int k, mfem::ElementTransformation& T,
                     const mfem::IntegrationPoint& ip,
                     mfem::DenseMatrix& Ck) const override;
  /** @brief The Mandel tensor @f$\hat C@f$ (the equilibrium-mapping
   * weighting is the integrator's business, not the modulus'). */
  void UnrelaxedModulus(mfem::ElementTransformation& T,
                        const mfem::IntegrationPoint& ip,
                        mfem::DenseMatrix& CU) const override;
  std::unique_ptr<ElasticStiffness> MakeStiffness() const override;

  mfem::MatrixCoefficient& ElasticTensor() const { return *C_; }
  mfem::MatrixCoefficient& EquilibriumStress() const { return *S_; }
  Diffeomorphism& EquilibriumMapping() const { return *map_; }

 private:
  int dim_;
  mfem::MatrixCoefficient* C_;
  mfem::MatrixCoefficient* S_;
  Diffeomorphism* map_;
};

/**
 * @brief The linearised quasi-static problem of a self-gravitating,
 * arbitrarily pre-stressed elastic body in the fully referential
 * formulation (doc/gravitating_elasticity.md; solids only at present).
 *
 * **Geometry.** As for LinearQuasiStaticSelfGravitatingProblem: the body
 * on a (Par)SubMesh of a ball whose outer boundary carries the DtN
 * condition; the displacement @f$u@f$ on the SubMesh, the *referential*
 * potential perturbation @f$\zeta^1@f$ on the ball. The reference body is
 * fixed; the physics of shape lives in the equilibrium mapping
 * @f$\varphi_e@f$ (a Diffeomorphism on the ball, tapering to the identity
 * at the DtN sphere — asserted, not enforced) and the referential fields.
 *
 * **Weak form** (doc/gravitating_elasticity.md §3.1, with
 * @f$a_e = J_e F_e^{-1} F_e^{-T}@f$, @f$\mathbf{g}_0 = \nabla\zeta^0@f$):
 * @f[
 *   \int_B \langle \hat C\,\widehat{\mathrm{sym}(F_e^TDu)},
 *     \widehat{\mathrm{sym}(F_e^TDv)}\rangle
 *   + \mathbf{S}_e : (Du^T Dv)
 *   + \tfrac{1}{8\pi G}\langle a''(u,v)\mathbf{g}_0, \mathbf{g}_0\rangle\,dV
 *   + c(\zeta^1, v) = \ell_u(v),
 * @f]
 * @f[
 *   \tfrac{1}{4\pi G}\Bigl[\int_{B_R} \langle a_e\nabla\zeta^1,
 *     \nabla\chi\rangle + \mathrm{DtN}(\zeta^1,\chi)\Bigr]
 *   + c(\chi, u) = \ell_\zeta(\chi),
 * @f]
 * with the coupling @f$c(\zeta^1, v) = \tfrac{1}{4\pi G}\int_B
 * \langle a'(v)\mathbf{g}_0, \nabla\zeta^1\rangle@f$. The background
 * @f$\zeta^0@f$ solves the referential Poisson equation with the fixed
 * referential density (solved at construction unless supplied).
 *
 * **Loads.** A surface mass load @f$\sigma@f$ (mass per unit *referential*
 * area) loads the potential row alone,
 * @f$\ell_\zeta(\chi) = -\int_{\partial B}\sigma\chi\,dS@f$: the Eulerian
 * form's @f$-\int\sigma\nabla\Phi_0\cdot v@f$ is absorbed by the change of
 * variables (note §3.1). Further loads via ExternalLoad() /
 * ExternalPotentialLoad() / AddForce().
 *
 * **Null space.** Rigid modes carry *no* potential partners: the null
 * pairs are @f$(t, 0)@f$ — exact discretely, every term vanishing
 * pointwise — and @f$(W\varphi_e, 0)@f$, exactly strain-free and
 * gravitationally near-null. Both are projected from the displacement in
 * the true-dof inner product; the potential needs no projection (2-D:
 * the constant is handled as in the Eulerian class).
 *
 * **Solver**: block MINRES on the symmetric saddle system with the
 * projected block-diagonal preconditioner (AMG/GS on the displacement
 * block and on the shifted mapped Laplacian). Serial and parallel in one
 * class, as throughout.
 */
class LinearQuasiStaticReferentialProblem
    : public LinearQuasiStaticProblemBase {
 public:
  /**
   * @param fes_u Displacement space (vdim = dim), not owned: either on a
   * (Par)SubMesh of the ball (the buffer terms then need a prescribed
   * extension; see SetVacuumExtension for the alternative), or on the
   * ball itself ("ball-wide" mode): the displacement carries the gauge
   * vacuum-extension field through the buffer, the constitutive
   * coefficients must vanish there, and SetVacuumExtension() supplies its
   * regularisation (doc/gravitating_elasticity.md §3.1).
   * @param fes_zeta Scalar referential-potential space on the parent
   * (Par)Mesh; not owned.
   * @param rheology The constitutive state (C, S_e, phi_e); must outlive
   * the problem. Its equilibrium mapping must extend over the ball and be
   * the identity at the DtN sphere.
   * @param density Referential density on the body (SubMesh); not owned.
   * @param gravitational_constant @f$G@f$ in the units of the problem.
   * @param dtn_degree Truncation degree of the DtN expansion.
   * @param background_zeta0 Optional @f$\zeta^0@f$ (projected onto
   * @p fes_zeta); solved from the density when null.
   */
  LinearQuasiStaticReferentialProblem(
      mfem::FiniteElementSpace* fes_u, mfem::FiniteElementSpace* fes_zeta,
      const ReferentialElasticRheology& rheology, mfem::Coefficient& density,
      mfem::real_t gravitational_constant, int dtn_degree,
      mfem::Coefficient* background_zeta0 = nullptr);

  /** @brief Surface mass load @f$\sigma@f$ per referential area
   * (registered as time-dependent) on the marked boundary attributes of
   * the displacement mesh: a potential-row load only (see the class
   * notes). */
  void SetSurfaceLoad(mfem::Coefficient& sigma,
                      const mfem::Array<int>& bdr_marker);

  /**
   * @brief Ball-wide mode: regularise the pure-gauge vacuum-extension
   * field with the harmonic penalty @f$\epsilon\mu_g\int \nabla u :
   * \nabla v@f$ on the marked (buffer) attributes, with the Tikhonov
   * refinement of the gauge machinery removing the @f$O(\epsilon)@f$ bias
   * from the observables (as for the gauged fluid: moderate epsilon and a
   * few refinements; epsilon -> 0 without refinement wrecks the
   * conditioning instead). Call before the first Solve().
   */
  void SetVacuumExtension(const mfem::Array<int>& buffer_marker,
                          mfem::Coefficient& mu_gauge, mfem::real_t epsilon,
                          int refinements = 3);

  /**
   * @brief SubMesh mode: supply the buffer's gravity terms through a
   * *prescribed* extension @f$u_{\mathrm{ext}} = E\,u@f$ on a buffer
   * displacement space (option (b) of doc/gravitating_elasticity.md
   * §3.1, the accurate route): the buffer's gravity-gravity block and
   * coupling are assembled on @p fes_buffer and folded through @f$E@f$
   * by sparse products. Exact for any smooth extension (the choice is
   * gauge); no extra unknowns and no penalty. @p fes_buffer must live on
   * a (Sub)Mesh of the ball sharing this space's collection; @p E maps
   * body vdofs to buffer vdofs (NewRadialVacuumExtension). Serial. Call
   * once, before the first Solve().
   */
  void SetPrescribedVacuumExtension(mfem::FiniteElementSpace& fes_buffer,
                                    const mfem::SparseMatrix& E);

#ifdef MFEM_USE_MPI
  /** @brief Parallel overload: @p E on true dofs
   * (NewRadialVacuumExtension parallel overload); the folds run through
   * hypre RAP/ParMult. */
  void SetPrescribedVacuumExtension(mfem::ParFiniteElementSpace& fes_buffer,
                                    const mfem::HypreParMatrix& E);
#endif

  /** @brief True when the displacement lives on the ball itself. */
  bool BallWide() const { return ball_wide_; }

  /** @brief Potential-row load as a linear form on the shadow of the
   * potential space; add integrators before the first AssembleForce(). */
  mfem::LinearForm& ExternalPotentialLoad() { return *b_zeta_; }

  mfem::FiniteElementSpace& PotentialSpace() { return *fes_zeta_; }

  /** @brief Referential potential perturbation @f$\zeta^1@f$ on the
   * ball. */
  const mfem::GridFunction& Potential() const { return *zeta_; }
  const mfem::GridFunction& PotentialOnBody() const {
    return ball_wide_ ? *zeta_ : *zeta_shadow_;
  }
  mfem::FiniteElementSpace& PotentialSpaceOnBody() { return *shadow_zeta_; }

  const mfem::GridFunction& BackgroundPotential() const { return *zeta0_; }
  const mfem::GridFunction& BackgroundPotentialOnBody() const {
    return ball_wide_ ? *zeta0_ : *zeta0_shadow_;
  }

  /** @brief @f$\nabla\zeta^0@f$ on the body, as used in the operator. */
  mfem::VectorCoefficient& BackgroundGravity() const {
    return *grad_zeta0_shadow_;
  }

  const PoissonDtNOperator& DtN() const { return *dtn_; }
  mfem::real_t GravitationalConstant() const { return G_; }

  /** @brief Iterations of the outer solver in the last Solve(). */
  int LastOuterIterations() const { return outer_its_; }

  /** @brief Rigid-body modes (orthonormal, true dofs): translations, then
   * rotations of the *mapped* positions @f$W\varphi_e@f$. */
  const NullSpaceProjector& RigidModes() const { return *projector_u_; }

  /**
   * @brief Diagnostic: @f$\|A_{\mathrm{blk}} n\| / \|A\|_{\max}@f$ for
   * each rigid null pair @f$(u_r, 0)@f$ under the full block operator.
   * Translations are exact discrete null vectors (round-off); rotations
   * are near-null, decreasing with refinement. Assembles if needed.
   */
  std::vector<mfem::real_t> RigidPairResiduals();

  void AssembleForce(mfem::real_t t) override;
  void RegisterFields(mfem::DataCollection& dc) override;

 protected:
  void SetupSolver(mfem::OperatorHandle& A) override;
  bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X) override;

  /** @brief Tikhonov refinement on the coupled system, as for the gauged
   * fluid: refinement solves carry zero potential load, and the potential
   * accumulates alongside the displacement. */
  bool GaugeRefine(mfem::Vector& X) override;

 private:
  void SetupPotentialOperators();
  void SetupCoupling();
  void SetupGravityIntegrators();
  void SetupRigidModes();
  void ComputeBackgroundPotential(mfem::Coefficient* zeta0);
  void ToTrueDofs(const mfem::FiniteElementSpace& fes, const mfem::Vector& L,
                  mfem::Vector& T) const;
  void MakeCompatible(mfem::Vector& B_zeta) const;
  void DistributePotential(const mfem::Vector& Z);
  bool ParallelPotential() const;

  // geometry and spaces
  bool ball_wide_ = false;
  int dim_;
  mfem::FiniteElementSpace* fes_zeta_;
#ifdef MFEM_USE_MPI
  mfem::ParFiniteElementSpace* pfes_zeta_ = nullptr;
#endif
  std::unique_ptr<mfem::FiniteElementSpace> shadow_zeta_;
  std::unique_ptr<SubMeshDofInjection> injection_;

  // physics
  const ReferentialElasticRheology* ref_rheology_;
  mfem::Coefficient* rho_;
  mfem::real_t G_, four_pi_G_;
  int dtn_degree_;
  mfem::ConstantCoefficient one_, inv_four_pi_G_, shift_coef_;

  // potential fields
  std::unique_ptr<mfem::GridFunction> zeta0_, zeta0_shadow_, zeta_,
      zeta_shadow_;
  std::unique_ptr<mfem::GradientGridFunctionCoefficient> grad_zeta0_shadow_;
  mfem::Vector Zeta_true_;

  // potential operators: A_zeta = (K_a + DtN)/4piG
  std::unique_ptr<PoissonDtNOperator> dtn_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::RAPOperator> dtn_rap_;
#endif
  const mfem::Operator* dtn_op_ = nullptr;
  std::unique_ptr<mfem::BilinearForm> k_zeta_form_, k_shift_form_;
  mfem::OperatorHandle K_zeta_, K_shift_;
  std::unique_ptr<mfem::SumOperator> A_zeta_op_;
  const mfem::Operator* A_zeta_ = nullptr;
  std::unique_ptr<mfem::Solver> prec_zeta_;
  std::unique_ptr<mfem::CGSolver> cg_zeta_;
  std::unique_ptr<NullSpaceProjector> projector_c_;
  std::unique_ptr<ProjectedOperator> projected_zeta_op_;
  std::unique_ptr<ProjectedSolver> projected_zeta_, projected_prec_zeta_;
  mfem::Solver* zeta_solver_ = nullptr;

  // coupling (trial zeta on the ball, test u on the body), true dofs
  std::unique_ptr<mfem::MixedBilinearForm> c_form_;
  mfem::OperatorHandle C_;
  std::unique_ptr<mfem::Operator> Ct_owned_;
  const mfem::Operator* C_op_ = nullptr;
  const mfem::Operator* Ct_op_ = nullptr;

  // prescribed vacuum extension (SubMesh mode, serial)
  std::unique_ptr<mfem::FiniteElementSpace> shadow_zeta_buffer_;
  std::unique_ptr<mfem::GridFunction> zeta0_buffer_;
  std::unique_ptr<mfem::GradientGridFunctionCoefficient> grad_zeta0_buffer_;
  std::unique_ptr<mfem::SparseMatrix> ext_EtGE_, ext_C_total_, ext_Ct_total_;
#ifdef MFEM_USE_MPI
  std::unique_ptr<mfem::HypreParMatrix> pext_EtGE_, pext_C_total_,
      pext_Ct_total_;
#endif
  mfem::OperatorHandle A_aug_;

  // loads
  std::unique_ptr<mfem::LinearForm> b_zeta_;
  mfem::Vector B_zeta_;
  std::list<mfem::Array<int>> load_markers_;
  std::vector<std::unique_ptr<mfem::Coefficient>> load_coefs_;

  // 2-D compatibility (as in the Eulerian class)
  mfem::Vector ones_, L_outer_;
  mfem::real_t outer_length_ = 0.0;

  // null space and solver
  std::unique_ptr<NullSpaceProjector> projector_u_, projector_block_;
  mfem::Array<int> offsets_;
  std::unique_ptr<mfem::BlockOperator> block_op_;
  std::unique_ptr<mfem::BlockDiagonalPreconditioner> block_prec_;
  std::unique_ptr<ProjectedOperator> projected_op_;
  std::unique_ptr<ProjectedSolver> projected_, projected_prec_;
  std::unique_ptr<mfem::MINRESSolver> minres_;
  std::unique_ptr<mfem::BlockVector> X_block_, B_block_;
  mfem::real_t shift_ = 1e-3;
  int outer_its_ = 0;
};

}  // namespace mfemElasticity
