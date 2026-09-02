/**
 * @file self_gravitating.cpp
 * @brief Implementation of SelfGravitatingElasticProblem.
 */

#include "mfemElasticity/self_gravitating.hpp"

#include <algorithm>
#include <cmath>

#include "mfemElasticity/bilininteg.hpp"
#include "mfemElasticity/detail/fem_factory.hpp"
#include "mfemElasticity/mesh.hpp"

namespace mfemElasticity {

using namespace mfem;

namespace {
constexpr real_t kPi = 3.141592653589793238462643383279502884;
}

// ---------------------------------------------------------------------------
// Construction

SelfGravitatingElasticProblem::SelfGravitatingElasticProblem(
    FiniteElementSpace* fes_u, FiniteElementSpace* fes_phi,
    const GeneralisedMaxwellRheology& rheology, Coefficient& density,
    real_t gravitational_constant, int dtn_degree,
    Coefficient* background_potential)
    : ElasticProblemBase(fes_u, rheology),
      dim_(fes_u->GetMesh()->Dimension()),
      fes_phi_(fes_phi),
      rho_(&density),
      G_(gravitational_constant),
      four_pi_G_(4.0 * kPi * gravitational_constant),
      dtn_degree_(dtn_degree),
      one_(1.0),
      inv_four_pi_G_(1.0 / (4.0 * kPi * gravitational_constant)),
      shift_coef_(shift_ / (4.0 * kPi * gravitational_constant)),
      K_phi_(Operator::MFEM_SPARSEMAT),
      K_shift_(Operator::MFEM_SPARSEMAT),
      C_(Operator::MFEM_SPARSEMAT) {
  MFEM_VERIFY(G_ > 0.0,
              "SelfGravitatingElasticProblem: G must be positive.");
  MFEM_VERIFY(fes_phi_->GetVDim() == 1,
              "SelfGravitatingElasticProblem: the potential space must be "
              "scalar.");
  MFEM_VERIFY(fes_phi_->GetMesh()->Dimension() == dim_,
              "SelfGravitatingElasticProblem: mesh dimensions differ.");

#ifdef MFEM_USE_MPI
  pfes_phi_ = dynamic_cast<ParFiniteElementSpace*>(fes_phi_);
  MFEM_VERIFY((pfes_ != nullptr) == (pfes_phi_ != nullptr),
              "SelfGravitatingElasticProblem: the displacement and potential "
              "spaces must both be serial or both be parallel.");
  if (pfes_phi_) {
    auto* psub = dynamic_cast<ParSubMesh*>(pfes_->GetParMesh());
    MFEM_VERIFY(psub, "SelfGravitatingElasticProblem: the displacement "
                      "space must live on a ParSubMesh.");
    MFEM_VERIFY(psub->GetParent() == pfes_phi_->GetParMesh(),
                "SelfGravitatingElasticProblem: the potential space must "
                "live on the parent of the displacement SubMesh.");
    shadow_phi_ = SubMeshDofInjection::MakeShadowSpace(*pfes_phi_, *psub);
    K_phi_.SetType(Operator::Hypre_ParCSR);
    K_shift_.SetType(Operator::Hypre_ParCSR);
    C_.SetType(Operator::Hypre_ParCSR);
  } else
#endif
  {
    auto* sub = dynamic_cast<SubMesh*>(fes_->GetMesh());
    MFEM_VERIFY(sub, "SelfGravitatingElasticProblem: the displacement space "
                     "must live on a SubMesh.");
    MFEM_VERIFY(sub->GetParent() == fes_phi_->GetMesh(),
                "SelfGravitatingElasticProblem: the potential space must "
                "live on the parent of the displacement SubMesh.");
    shadow_phi_ = SubMeshDofInjection::MakeShadowSpace(*fes_phi_, *sub);
  }
  injection_ = std::make_unique<SubMeshDofInjection>(*shadow_phi_, *fes_phi_);

  // Potential fields.
  phi0_ = detail::MakeGridFunction(fes_phi_);
  phi0_shadow_ = detail::MakeGridFunction(shadow_phi_.get());
  phi_ = detail::MakeGridFunction(fes_phi_);
  phi_shadow_ = detail::MakeGridFunction(shadow_phi_.get());
  *phi0_ = 0.0;
  *phi0_shadow_ = 0.0;
  *phi_ = 0.0;
  *phi_shadow_ = 0.0;
  grad_phi0_ = std::make_unique<GradientGridFunctionCoefficient>(phi0_.get());
  grad_phi0_shadow_ =
      std::make_unique<GradientGridFunctionCoefficient>(phi0_shadow_.get());
  Phi_true_.SetSize(fes_phi_->GetTrueVSize());
  Phi_true_ = 0.0;

  // The DtN operator on the outer boundary.
#ifdef MFEM_USE_MPI
  if (pfes_phi_) {
    dtn_ = std::make_unique<PoissonDtNOperator>(pfes_phi_->GetComm(),
                                                pfes_phi_, dtn_degree_);
    dtn_->Assemble();
    dtn_rap_ = std::make_unique<RAPOperator>(dtn_->RAP());
    dtn_op_ = dtn_rap_.get();
  } else
#endif
  {
    dtn_ = std::make_unique<PoissonDtNOperator>(fes_phi_, dtn_degree_);
    dtn_->Assemble();
    dtn_op_ = dtn_.get();
  }

  // Data for the 2-D compatibility condition: the constant and the outer
  // boundary functional, on true dofs.
  if (dim_ == 2) {
    auto ones = detail::MakeGridFunction(fes_phi_);
    *ones = 1.0;
    ones->GetTrueDofs(ones_);
    auto marker = ExternalBoundaryMarker(fes_phi_->GetMesh());
    auto l = detail::MakeLinearForm(fes_phi_);
    l->AddBoundaryIntegrator(new BoundaryLFIntegrator(one_), marker);
    l->Assemble();
    ToTrueDofs(*fes_phi_, *l, L_outer_);
    outer_length_ = Dot(L_outer_, ones_);
    MFEM_VERIFY(outer_length_ > 0.0,
                "SelfGravitatingElasticProblem: empty outer boundary.");
  }

  SetupPotentialOperators();
  ComputeBackgroundPotential(background_potential);
  SetupCoupling();
  SetupGravityIntegrators();
  SetupRigidModes();
  SetupCoupledNullSpace();

  b_phi_ = detail::MakeLinearForm(shadow_phi_.get());
  B_phi_.SetSize(fes_phi_->GetTrueVSize());
  B_phi_ = 0.0;
}

bool SelfGravitatingElasticProblem::ParallelPotential() const {
#ifdef MFEM_USE_MPI
  return pfes_phi_ != nullptr;
#else
  return false;
#endif
}

void SelfGravitatingElasticProblem::ToTrueDofs(const FiniteElementSpace& fes,
                                               const Vector& L,
                                               Vector& T) const {
  const Operator* P = fes.GetProlongationMatrix();
  if (P) {
    T.SetSize(P->Width());
    P->MultTranspose(L, T);
  } else {
    T = L;
  }
}

void SelfGravitatingElasticProblem::SetupPotentialOperators() {
  Array<int> empty;

  // K = (grad phi, grad psi) on the ball; A_phiphi = (K + DtN) / (4 pi G).
  k_phi_form_ = detail::MakeBilinearForm(fes_phi_);
  k_phi_form_->AddDomainIntegrator(new DiffusionIntegrator(one_));
  k_phi_form_->Assemble();
  k_phi_form_->FormSystemMatrix(empty, K_phi_);
  const real_t c = 1.0 / four_pi_G_;
  A_phiphi_ = std::make_unique<SumOperator>(K_phi_.Ptr(), c, dtn_op_, c,
                                            false, false);

  SetupPotentialPreconditioner();

#ifdef MFEM_USE_MPI
  if (pfes_phi_) {
    cg_phi_ = std::make_unique<CGSolver>(pfes_phi_->GetComm());
    if (dim_ == 2) {
      ortho_phi_ = std::make_unique<OrthoSolver>(pfes_phi_->GetComm());
    }
  } else
#endif
  {
    cg_phi_ = std::make_unique<CGSolver>();
    if (dim_ == 2) {
      ortho_phi_ = std::make_unique<OrthoSolver>();
    }
  }
  cg_phi_->SetOperator(*A_phiphi_);
  cg_phi_->SetPreconditioner(*prec_phi_);
  cg_phi_->SetRelTol(inner_rel_tol_);
  cg_phi_->SetAbsTol(0.0);
  cg_phi_->SetMaxIter(10000);
  cg_phi_->SetPrintLevel(inner_print_level_);
  cg_phi_->iterative_mode = false;
  if (ortho_phi_) {
    ortho_phi_->SetSolver(*cg_phi_);
    ortho_phi_->iterative_mode = false;
    phi_solver_ = ortho_phi_.get();
  } else {
    phi_solver_ = cg_phi_.get();
  }
}

void SelfGravitatingElasticProblem::SetupPotentialPreconditioner() {
  Array<int> empty;
  shift_coef_.constant = shift_ / four_pi_G_;
  k_shift_form_ = detail::MakeBilinearForm(fes_phi_);
  k_shift_form_->AddDomainIntegrator(new DiffusionIntegrator(inv_four_pi_G_));
  k_shift_form_->AddDomainIntegrator(new MassIntegrator(shift_coef_));
  k_shift_form_->Assemble();
  k_shift_form_->FormSystemMatrix(empty, K_shift_);
#ifdef MFEM_USE_MPI
  if (pfes_phi_) {
    auto amg = std::make_unique<HypreBoomerAMG>(*K_shift_.As<HypreParMatrix>());
    amg->SetPrintLevel(0);
    prec_phi_ = std::move(amg);
  } else
#endif
  {
    prec_phi_ = std::make_unique<GSSmoother>(*K_shift_.As<SparseMatrix>());
  }
  if (cg_phi_) {
    cg_phi_->SetPreconditioner(*prec_phi_);
  }
}

void SelfGravitatingElasticProblem::MakeCompatible(Vector& B_phi) const {
  if (dim_ != 2) {
    return;
  }
  const real_t mass = Dot(B_phi, ones_);
  B_phi.Add(-mass / outer_length_, L_outer_);
}

bool SelfGravitatingElasticProblem::SolvePotential(const Vector& b,
                                                   Vector& x) const {
  x.SetSize(b.Size());
  x = 0.0;
  phi_solver_->Mult(b, x);
  inner_its_ += cg_phi_->GetNumIterations();
  return cg_phi_->GetConverged();
}

void SelfGravitatingElasticProblem::DistributePotential(const Vector& Phi) {
  phi_->SetFromTrueDofs(Phi);
  injection_->MultTranspose(*phi_, *phi_shadow_);
}

void SelfGravitatingElasticProblem::ComputeBackgroundPotential(
    Coefficient* phi0) {
  if (phi0) {
    phi0_->ProjectCoefficient(*phi0);
  } else {
    // (K + DtN) Phi0 = -4 pi G (rho, psi)_M, i.e. A_phiphi Phi0 = -(rho, psi)_M
    // with the density integrated on the SubMesh and injected into the ball.
    auto rho_form = detail::MakeLinearForm(shadow_phi_.get());
    rho_form->AddDomainIntegrator(new DomainLFIntegrator(*rho_));
    rho_form->Assemble();
    Vector bL(fes_phi_->GetVSize()), B;
    injection_->Mult(*rho_form, bL);
    ToTrueDofs(*fes_phi_, bL, B);
    B *= -1.0;
    MakeCompatible(B);
    Vector Phi0;
    const bool ok = SolvePotential(B, Phi0);
    MFEM_VERIFY(ok, "SelfGravitatingElasticProblem: the background "
                    "potential solve did not converge.");
    phi0_->SetFromTrueDofs(Phi0);
  }
  injection_->MultTranspose(*phi0_, *phi0_shadow_);
}

void SelfGravitatingElasticProblem::SetupCoupling() {
  Array<int> empty;
#ifdef MFEM_USE_MPI
  if (pfes_phi_) {
    auto form =
        std::make_unique<ParSubMeshMixedBilinearForm>(pfes_phi_, pfes_);
    form->AddDomainIntegrator(new GradientIntegrator(*rho_));
    form->Assemble();
    form->FormRectangularSystemMatrix(empty, empty, C_);
    c_form_ = std::move(form);
    Ct_owned_.reset(C_.As<HypreParMatrix>()->Transpose());
  } else
#endif
  {
    auto form = std::make_unique<SubMeshMixedBilinearForm>(fes_phi_, fes_);
    form->AddDomainIntegrator(new GradientIntegrator(*rho_));
    form->Assemble();
    form->FormRectangularSystemMatrix(empty, empty, C_);
    c_form_ = std::move(form);
    Ct_owned_.reset(Transpose(*C_.As<SparseMatrix>()));
  }
  C_op_ = C_.Ptr();
  Ct_op_ = Ct_owned_.get();
}

void SelfGravitatingElasticProblem::SetupGravityIntegrators() {
  half_rho_ = std::make_unique<ProductCoefficient>(0.5, *rho_);
  minus_half_rho_ = std::make_unique<ProductCoefficient>(-0.5, *rho_);
  minus_half_rho_grad_ = std::make_unique<ScalarVectorProductCoefficient>(
      *minus_half_rho_, *grad_phi0_shadow_);
  auto* g1 =
      new DomainVectorGradVectorIntegrator(*grad_phi0_shadow_, *half_rho_);
  auto* g2 = new DomainVectorDivVectorIntegrator(*minus_half_rho_grad_);
  auto& integs = StiffnessIntegrators();
  integs.AddDomainIntegrator(g1);
  integs.AddDomainIntegrator(g2);
  integs.AddDomainIntegrator(new TransposeIntegrator(g1, 0));
  integs.AddDomainIntegrator(new TransposeIntegrator(g2, 0));
}

void SelfGravitatingElasticProblem::SetupRigidModes() {
#ifdef MFEM_USE_MPI
  if (pfes_) {
    projector_u_ = std::make_unique<NullSpaceProjector>(pfes_->GetComm());
  } else
#endif
  {
    projector_u_ = std::make_unique<NullSpaceProjector>();
  }
  auto gf = detail::MakeGridFunction(fes_);
  auto add = [&](VectorCoefficient& c) {
    gf->ProjectCoefficient(c);
    Vector t;
    gf->GetTrueDofs(t);
    rigid_true_.push_back(t);
    projector_u_->Add(t);
  };
  for (int c = 0; c < dim_; c++) {
    RigidTranslation tr(dim_, c);
    add(tr);
  }
  if (dim_ == 2) {
    RigidRotation rot(2, 2);
    add(rot);
  } else {
    for (int c = 0; c < 3; c++) {
      RigidRotation rot(3, c);
      add(rot);
    }
  }
}

void SelfGravitatingElasticProblem::SetupCoupledNullSpace() {
  // The near-null vectors of the block system are (u_r, phi_r) with phi_r
  // the discrete potential response to the rigid mode, phi_r = -A_phiphi^{-1}
  // C^T u_r: then the potential row is satisfied exactly and the residual of
  // the displacement row is S u_r (see RigidModeResiduals()). Defining phi_r
  // this way (rather than by interpolating -u_r . grad Phi0) keeps the
  // projector independent of the mesh partition. In 2-D the constant
  // potential is added.
#ifdef MFEM_USE_MPI
  if (pfes_) {
    projector_block_ =
        std::make_unique<NullSpaceProjector>(pfes_->GetComm());
  } else
#endif
  {
    projector_block_ = std::make_unique<NullSpaceProjector>();
  }
  offsets_.SetSize(3);
  offsets_[0] = 0;
  offsets_[1] = fes_->GetTrueVSize();
  offsets_[2] = fes_phi_->GetTrueVSize();
  offsets_.PartialSum();

  BlockVector n(offsets_);
  Vector t(fes_phi_->GetTrueVSize()), w;
  for (int i = 0; i < projector_u_->Size(); i++) {
    const Vector& ur = projector_u_->Basis(i);
    Ct_op_->Mult(ur, t);
    SolvePotential(t, w);
    n.GetBlock(0) = ur;
    n.GetBlock(1) = w;
    n.GetBlock(1) *= -1.0;
    projector_block_->Add(n);
  }
  if (dim_ == 2) {
    n.GetBlock(0) = 0.0;
    n.GetBlock(1) = ones_;
    projector_block_->Add(n);
  }
  inner_its_ = 0;
}

// ---------------------------------------------------------------------------
// Loads

void SelfGravitatingElasticProblem::SetSurfaceLoad(
    Coefficient& sigma, const Array<int>& bdr_marker) {
  MFEM_VERIFY(bdr_marker.Size() == fes_->GetMesh()->bdr_attributes.Max(),
              "SetSurfaceLoad: the marker must be sized to the SubMesh's "
              "bdr_attributes.Max().");
  RegisterTimeDependent(sigma);
  load_markers_.push_back(bdr_marker);
  auto& marker = load_markers_.back();
  auto minus_sigma = std::make_unique<ProductCoefficient>(-1.0, sigma);
  auto minus_sigma_grad = std::make_unique<ScalarVectorProductCoefficient>(
      *minus_sigma, *grad_phi0_shadow_);
  b_->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*minus_sigma_grad),
                            marker);
  b_phi_->AddBoundaryIntegrator(new BoundaryLFIntegrator(*minus_sigma),
                                marker);
  load_coefs_.push_back(std::move(minus_sigma));
  load_vcoefs_.push_back(std::move(minus_sigma_grad));
}

void SelfGravitatingElasticProblem::AssembleForce(real_t t) {
  ElasticProblemBase::AssembleForce(t);
  b_phi_->Assemble();
  Vector bL(fes_phi_->GetVSize());
  injection_->Mult(*b_phi_, bL);
  ToTrueDofs(*fes_phi_, bL, B_phi_);
  MakeCompatible(B_phi_);
}

// ---------------------------------------------------------------------------
// Controls

void SelfGravitatingElasticProblem::SetSolverType(SolverType type) {
  if (type != type_) {
    type_ = type;
    operator_dirty_ = true;
  }
}

void SelfGravitatingElasticProblem::SetInnerRelTol(real_t tol) {
  inner_rel_tol_ = tol;
  inner_tol_set_ = true;
  cg_phi_->SetRelTol(tol);
}

void SelfGravitatingElasticProblem::SetPreconditionerShift(real_t eps) {
  shift_ = eps;
  SetupPotentialPreconditioner();
  operator_dirty_ = true;
}

void SelfGravitatingElasticProblem::SetInnerPrintLevel(
    IterativeSolver::PrintLevel level) {
  inner_print_level_ = level;
  cg_phi_->SetPrintLevel(level);
}

void SelfGravitatingElasticProblem::SetBackgroundPotential(Coefficient& phi0) {
  ComputeBackgroundPotential(&phi0);
  operator_dirty_ = true;
}

void SelfGravitatingElasticProblem::RegisterFields(DataCollection& dc) {
  ElasticProblemBase::RegisterFields(dc);
  dc.RegisterField("potential", phi_shadow_.get());
  dc.RegisterField("background_potential", phi0_shadow_.get());
}

// ---------------------------------------------------------------------------
// Diagnostics

std::vector<real_t> SelfGravitatingElasticProblem::RigidModeResiduals() {
  EnsureOperator();
  const Operator& A_uu = *A_.Ptr();
  real_t a_max = 0.0;
#ifdef MFEM_USE_MPI
  if (pfes_) {
    auto* hyp = A_.As<HypreParMatrix>();
    SparseMatrix diag, offd;
    HYPRE_BigInt* cmap = nullptr;
    hyp->GetDiag(diag);
    hyp->GetOffd(offd, cmap);
    real_t local = std::max(diag.MaxNorm(), offd.MaxNorm());
    MPI_Allreduce(&local, &a_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                  pfes_->GetComm());
  } else
#endif
  {
    a_max = A_.As<SparseMatrix>()->MaxNorm();
  }
  const int saved_inner = inner_its_;
  std::vector<real_t> residuals;
  Vector t(Ct_op_->Height()), w, y(A_uu.Height()), cw(A_uu.Height());
  for (int i = 0; i < projector_u_->Size(); i++) {
    const Vector& ur = projector_u_->Basis(i);
    Ct_op_->Mult(ur, t);
    SolvePotential(t, w);
    A_uu.Mult(ur, y);
    C_op_->Mult(w, cw);
    y -= cw;
    residuals.push_back(std::sqrt(Dot(y, y)) / a_max);
  }
  inner_its_ = saved_inner;
  return residuals;
}

// ---------------------------------------------------------------------------
// Solvers

SelfGravitatingElasticProblem::SchurOperator::SchurOperator(
    const SelfGravitatingElasticProblem& p, const Operator& A_uu)
    : Operator(A_uu.Height(), A_uu.Width()), p_(&p), A_uu_(&A_uu) {}

void SelfGravitatingElasticProblem::SchurOperator::Mult(const Vector& x,
                                                        Vector& y) const {
  t_.SetSize(p_->Ct_op_->Height());
  p_->Ct_op_->Mult(x, t_);
  p_->SolvePotential(t_, w_);
  A_uu_->Mult(x, y);
  cw_.SetSize(y.Size());
  p_->C_op_->Mult(w_, cw_);
  y -= cw_;
}

void SelfGravitatingElasticProblem::SetupSolver(OperatorHandle& A) {
  if (!inner_tol_set_) {
    inner_rel_tol_ = std::max<real_t>(1e-15, 1e-2 * rel_tol_);
    cg_phi_->SetRelTol(inner_rel_tol_);
  }
  // Preconditioner (prec_) and CG (cg_) for the displacement block.
  SetupDefaultCG(A);
  if (type_ == SolverType::SchurCG) {
    SetupSchur(A);
  } else {
    SetupMinres(A);
  }
}

void SelfGravitatingElasticProblem::SetupSchur(OperatorHandle& A) {
  schur_ = std::make_unique<SchurOperator>(*this, *A.Ptr());
  projected_op_ = std::make_unique<ProjectedOperator>(*schur_, *projector_u_);
  // The operator must be set before the preconditioner: IterativeSolver
  // forwards SetOperator to its preconditioner, and the Gauss-Seidel
  // smoother of the serial path only accepts a SparseMatrix.
#ifdef MFEM_USE_MPI
  if (pfes_) {
    cg_ = std::make_unique<CGSolver>(pfes_->GetComm());
  } else
#endif
  {
    cg_ = std::make_unique<CGSolver>();
  }
  cg_->SetOperator(*projected_op_);
  cg_->SetPreconditioner(*prec_);
  cg_->SetRelTol(rel_tol_);
  cg_->SetAbsTol(0.0);
  cg_->SetMaxIter(10000);
  cg_->SetPrintLevel(print_level_);
  cg_->iterative_mode = true;

  projected_ = std::make_unique<ProjectedSolver>(*projector_u_);
  projected_->SetSolver(*cg_);
  projected_->iterative_mode = true;
}

void SelfGravitatingElasticProblem::SetupMinres(OperatorHandle& A) {
  block_op_ = std::make_unique<BlockOperator>(offsets_);
  block_op_->SetBlock(0, 0, A.Ptr());
  block_op_->SetBlock(0, 1, const_cast<Operator*>(C_op_));
  block_op_->SetBlock(1, 0, const_cast<Operator*>(Ct_op_));
  block_op_->SetBlock(1, 1, A_phiphi_.get());

  block_prec_ = std::make_unique<BlockDiagonalPreconditioner>(offsets_);
  block_prec_->SetDiagonalBlock(0, prec_.get());
  block_prec_->SetDiagonalBlock(1, prec_phi_.get());

#ifdef MFEM_USE_MPI
  if (pfes_) {
    minres_ = std::make_unique<MINRESSolver>(pfes_->GetComm());
  } else
#endif
  {
    minres_ = std::make_unique<MINRESSolver>();
  }
  projected_op_ =
      std::make_unique<ProjectedOperator>(*block_op_, *projector_block_);
  minres_->SetOperator(*projected_op_);
  minres_->SetPreconditioner(*block_prec_);
  minres_->SetRelTol(rel_tol_);
  minres_->SetAbsTol(0.0);
  minres_->SetMaxIter(10000);
  minres_->SetPrintLevel(print_level_);
  minres_->iterative_mode = true;

  projected_ = std::make_unique<ProjectedSolver>(*projector_block_);
  projected_->SetSolver(*minres_);
  projected_->iterative_mode = true;

  // The MINRES iterate is the warm-start state; keep it across operator
  // rebuilds (a change of shear modulus) when the sizes are unchanged.
  if (!X_block_ || X_block_->Size() != offsets_.Last()) {
    X_block_ = std::make_unique<BlockVector>(offsets_);
    *X_block_ = 0.0;
  }
  B_block_ = std::make_unique<BlockVector>(offsets_);
}

bool SelfGravitatingElasticProblem::SolveLinearSystem(const Vector& B,
                                                      Vector& X) {
  inner_its_ = 0;
  outer_its_ = 0;
  bool ok = true;

  if (type_ == SolverType::SchurCG) {
    // rhs_S = B - C A_phiphi^{-1} B_phi.
    ok = SolvePotential(B_phi_, w_) && ok;
    rhs_s_.SetSize(B.Size());
    C_op_->Mult(w_, rhs_s_);
    rhs_s_ *= -1.0;
    rhs_s_ += B;
    projector_u_->Project(rhs_s_);
    if (!SetWarmStartTolerance(*cg_, *prec_, rhs_s_)) {
      X = 0.0;
      Phi_true_ = w_;
    } else {
      projected_->Mult(rhs_s_, X);
      ok = cg_->GetConverged() && ok;
      outer_its_ = cg_->GetNumIterations();
      // phi = A_phiphi^{-1} (B_phi - C^T u).
      Vector t(B_phi_.Size());
      Ct_op_->Mult(X, t);
      t *= -1.0;
      t += B_phi_;
      ok = SolvePotential(t, Phi_true_) && ok;
    }
  } else {
    B_block_->GetBlock(0) = B;
    B_block_->GetBlock(1) = B_phi_;
    if (!SetWarmStartTolerance(*minres_, *block_prec_, *B_block_)) {
      X = 0.0;
      Phi_true_ = 0.0;
      *X_block_ = 0.0;
    } else {
      // Warm start from the previous MINRES iterate (its own gauge).
      projected_->Mult(*B_block_, *X_block_);
      ok = minres_->GetConverged();
      outer_its_ = minres_->GetNumIterations();
      // Output gauge: u orthogonal to the rigid modes, phi solved from u.
      X = X_block_->GetBlock(0);
      projector_u_->Project(X);
      Vector t(B_phi_.Size());
      Ct_op_->Mult(X, t);
      t *= -1.0;
      t += B_phi_;
      ok = SolvePotential(t, Phi_true_) && ok;
    }
  }
  DistributePotential(Phi_true_);
  return ok;
}

}  // namespace mfemElasticity
