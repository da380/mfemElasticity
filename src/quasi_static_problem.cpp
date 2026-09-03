/**
 * @file quasi_static_problem.cpp
 * @brief Implementation of LinearQuasiStaticProblemBase,
 * LinearQuasiStaticTractionProblem and LinearQuasiStaticClampedProblem.
 */

#include "mfemElasticity/quasi_static_problem.hpp"

#include <cmath>

#include "mfemElasticity/detail/fem_factory.hpp"

namespace mfemElasticity {

using namespace mfem;

LinearQuasiStaticProblemBase::LinearQuasiStaticProblemBase(
    FiniteElementSpace* fes, const mfemElasticity::Rheology& rheology)
    : fes_(fes),
      rheology_(&rheology),
      stiffness_(rheology.MakeStiffness()),
      A_(Operator::MFEM_SPARSEMAT) {
  const int dim = fes_->GetMesh()->Dimension();
  MFEM_VERIFY(
      fes_->GetVDim() == dim,
      "LinearQuasiStaticProblemBase: the displacement space must have vdim "
      "equal to the space dimension.");
  MFEM_VERIFY(
      rheology.SpaceDim() == dim,
      "LinearQuasiStaticProblemBase: rheology and mesh dimensions differ.");
#ifdef MFEM_USE_MPI
  pfes_ = dynamic_cast<ParFiniteElementSpace*>(fes_);
  if (pfes_) {
    A_.SetType(Operator::Hypre_ParCSR);
  }
#endif

  // The stiffness integrators live in a template form that is never
  // assembled; each assembly builds a fresh form borrowing them, so that a
  // change of modulus never has to reuse a matrix pattern.
  integrators_ = detail::MakeBilinearForm(fes_);
  stiffness_->AddIntegrators(*integrators_);

  b_ = detail::MakeLinearForm(fes_);
  u_ = detail::MakeGridFunction(fes_);
  *u_ = 0.0;
  increment_.SetSize(fes_->GetVSize());
  increment_ = 0.0;
}

bool LinearQuasiStaticProblemBase::IsParallel() const {
#ifdef MFEM_USE_MPI
  return pfes_ != nullptr;
#else
  return false;
#endif
}

void LinearQuasiStaticProblemBase::SetEssentialBoundary(
    const Array<int>& ess_bdr) {
  Array<int> marker(ess_bdr);
  fes_->GetEssentialTrueDofs(marker, ess_tdof_list_);
  operator_dirty_ = true;
}

void LinearQuasiStaticProblemBase::AssembleForce(real_t t) {
  t_ = t;
  for (auto* c : td_coefs_) {
    c->SetTime(t);
  }
  for (auto* c : td_vcoefs_) {
    c->SetTime(t);
  }
  // LinearForm::Assemble() zeroes before assembling: idempotent at fixed t.
  b_->Assemble();
  increment_ = 0.0;
  UpdateBoundaryValues(t);
}

void LinearQuasiStaticProblemBase::AddForce(const Vector& f) {
  MFEM_VERIFY(f.Size() == increment_.Size(),
              "AddForce: expected a dual vector in the vdof layout of "
              "DisplacementSpace().");
  increment_ += f;
}

void LinearQuasiStaticProblemBase::SetRelaxationWeights(
    const std::vector<Coefficient*>& beta) {
  // Always reassemble: the same coefficient objects may carry new values.
  stiffness_->SetRelaxationWeights(beta);
  operator_dirty_ = true;
}

void LinearQuasiStaticProblemBase::ClearRelaxationWeights() {
  if (stiffness_->IsRelaxed()) {
    stiffness_->ClearRelaxationWeights();
    operator_dirty_ = true;
  }
}

void LinearQuasiStaticProblemBase::AssembleOperator() {
  if (a_ && prec_ && !prec_stale_ && prec_reuse_ > 1.0 && !prec_form_) {
    // The preconditioner was built on the current matrix and stays on it:
    // keep that form and matrix alive while the preconditioner is reused.
    // (Later reassemblies leave prec_form_ alone; their matrices go.)
    prec_form_ = std::move(a_);
    prec_A_ = A_;
    prec_A_.SetOperatorOwner(A_.OwnsOperator());
    A_.SetOperatorOwner(false);
  }
  a_ = detail::MakeBilinearForm(fes_, integrators_.get());
  a_->Assemble();
  a_->FormSystemMatrix(ess_tdof_list_, A_);
  SetupSolver(A_);
  operator_dirty_ = false;
  assemblies_++;
}

void LinearQuasiStaticProblemBase::NoteIterations(int its) {
  total_its_ += its;
  if (prec_baseline_its_ < 0) {
    prec_baseline_its_ = its;
  } else if (its > prec_reuse_ * prec_baseline_its_) {
    prec_stale_ = true;
  }
}

void LinearQuasiStaticProblemBase::EnsureOperator() {
  if (operator_dirty_) {
    AssembleOperator();
  }
}

const OperatorHandle& LinearQuasiStaticProblemBase::SystemMatrix() {
  EnsureOperator();
  return A_;
}

bool LinearQuasiStaticProblemBase::Solve() {
  solves_++;
  EnsureOperator();
  rhs_ = *b_;
  rhs_ += increment_;
  // Fold the boundary data into the reduced system on the scratch copy rhs_,
  // keeping the assembled external load pristine. copy_interior = 1 keeps
  // the interior of u_ in X_ so that solvers in iterative_mode warm start.
  a_->FormLinearSystem(ess_tdof_list_, *u_, rhs_, A_, X_, B_, 1);
  const bool ok = SolveLinearSystem(B_, X_);
  a_->RecoverFEMSolution(X_, rhs_, *u_);
  return ok;
}

void LinearQuasiStaticProblemBase::SetupDefaultCG(OperatorHandle& A) {
  SetupDefaultPreconditioner(A);
  SetupCG(*A.Ptr(), *prec_);
}

void LinearQuasiStaticProblemBase::SetupDefaultPreconditioner(
    OperatorHandle& A) {
  const bool rebuild = !prec_ || prec_stale_ || prec_reuse_ <= 1.0;
  if (rebuild) {
#ifdef MFEM_USE_MPI
    if (pfes_) {
      auto amg = std::make_unique<HypreBoomerAMG>(*A.As<HypreParMatrix>());
      amg->SetElasticityOptions(pfes_);
      amg->SetPrintLevel(0);
      prec_ = std::move(amg);
    } else
#endif
    {
      prec_ = std::make_unique<GSSmoother>(*A.As<SparseMatrix>());
    }
    prec_form_.reset();
    prec_A_.Clear();
    prec_stale_ = false;
    prec_baseline_its_ = -1;
    prec_setups_++;
  }
}

void LinearQuasiStaticProblemBase::SetupCG(const Operator& op, Solver& prec) {
#ifdef MFEM_USE_MPI
  if (pfes_) {
    cg_ = std::make_unique<CGSolver>(pfes_->GetComm());
  } else
#endif
  {
    cg_ = std::make_unique<CGSolver>();
  }
  // Operator before preconditioner: SetOperator would otherwise reset the
  // (reused) preconditioner onto the new operator.
  cg_->SetOperator(op);
  cg_->SetPreconditioner(prec);
  cg_->SetRelTol(rel_tol_);
  cg_->SetAbsTol(0.0);
  cg_->SetMaxIter(10000);
  cg_->SetPrintLevel(print_level_);
  cg_->iterative_mode = true;
}

void LinearQuasiStaticProblemBase::SetupSolver(OperatorHandle& A) {
  SetupDefaultCG(A);
}

bool LinearQuasiStaticProblemBase::SolveLinearSystem(const Vector& B,
                                                     Vector& X) {
  if (!SetWarmStartTolerance(*cg_, *prec_, B)) {
    X = 0.0;
    return true;
  }
  cg_->Mult(B, X);
  NoteIterations(cg_->GetNumIterations());
  return cg_->GetConverged();
}

std::unique_ptr<BilinearForm> LinearQuasiStaticProblemBase::AssembleMassOperator(
    Coefficient* rho, OperatorHandle& M) {
  auto form = detail::MakeBilinearForm(fes_);
  form->AddDomainIntegrator(rho ? new VectorMassIntegrator(*rho)
                                : new VectorMassIntegrator());
  form->Assemble();
#ifdef MFEM_USE_MPI
  if (pfes_) {
    M.SetType(Operator::Hypre_ParCSR);
  }
#endif
  Array<int> empty;
  form->FormSystemMatrix(empty, M);
  return form;
}

real_t LinearQuasiStaticProblemBase::Dot(const Vector& x,
                                         const Vector& y) const {
#ifdef MFEM_USE_MPI
  if (pfes_) {
    return InnerProduct(pfes_->GetComm(), x, y);
  }
#endif
  return InnerProduct(x, y);
}

bool LinearQuasiStaticProblemBase::SetWarmStartTolerance(
    IterativeSolver& solver, Solver& prec, const Vector& B) const {
  Vector z(B.Size());
  prec.Mult(B, z);
  const real_t nom = Dot(B, z);
  if (!(nom > 0.0)) {
    return false;
  }
  solver.SetAbsTol(rel_tol_ * std::sqrt(nom));
  return true;
}

void LinearQuasiStaticProblemBase::RegisterFields(DataCollection& dc) {
  dc.RegisterField("displacement", u_.get());
}

// ---------------------------------------------------------------------------

LinearQuasiStaticTractionProblem::LinearQuasiStaticTractionProblem(
    FiniteElementSpace* fes, const mfemElasticity::Rheology& rheology,
    VectorCoefficient& traction, const Array<int>& bdr_marker)
    : LinearQuasiStaticProblemBase(fes, rheology), marker_(bdr_marker) {
  RegisterTimeDependent(traction);
  b_->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction), marker_);
}

const NullSpaceProjector& LinearQuasiStaticTractionProblem::RigidModes() {
  if (!projector_) {
    projector_ = MakeRigidModeProjector(*fes_);
  }
  return *projector_;
}

void LinearQuasiStaticTractionProblem::SetupSolver(OperatorHandle& A) {
  const auto& P = RigidModes();
  SetupDefaultPreconditioner(A);
  // CG on P A P with the preconditioner P M P: an unprojected preconditioner
  // amplifies the round-off component along the (near-)null rigid modes.
  projected_prec_ = std::make_unique<ProjectedSolver>(P);
  projected_prec_->SetSolver(*prec_);
  projected_op_ = std::make_unique<ProjectedOperator>(*A.Ptr(), P);
  SetupCG(*projected_op_, *projected_prec_);
  // The outer wrapper projects the load and the warm start before, and the
  // solution after, the CG solve.
  projected_ = std::make_unique<ProjectedSolver>(P);
  projected_->SetSolver(*cg_);
  projected_->iterative_mode = true;
  projected_->SetGauge(gauge_M_.Ptr());
}

void LinearQuasiStaticTractionProblem::SetMassWeightedGauge(Coefficient* rho) {
  gauge_M_.Clear();
  gauge_form_ = AssembleMassOperator(rho, gauge_M_);
  if (projected_) {
    projected_->SetGauge(gauge_M_.Ptr());
  }
}

void LinearQuasiStaticTractionProblem::SetEuclideanGauge() {
  gauge_M_.Clear();
  gauge_form_.reset();
  if (projected_) {
    projected_->SetGauge(nullptr);
  }
}

bool LinearQuasiStaticTractionProblem::SolveLinearSystem(const Vector& B,
                                                         Vector& X) {
  // (P M P B, B) = (M P B, P B): the cold-start norm of the projected load.
  if (!SetWarmStartTolerance(*cg_, *projected_prec_, B)) {
    X = 0.0;
    return true;
  }
  projected_->Mult(B, X);
  NoteIterations(cg_->GetNumIterations());
  return cg_->GetConverged();
}

// ---------------------------------------------------------------------------

LinearQuasiStaticClampedProblem::LinearQuasiStaticClampedProblem(
    FiniteElementSpace* fes, const mfemElasticity::Rheology& rheology,
    const Array<int>& ess_bdr, VectorCoefficient& traction,
    const Array<int>& traction_marker, VectorCoefficient* dirichlet)
    : LinearQuasiStaticProblemBase(fes, rheology),
      ess_bdr_(ess_bdr),
      marker_(traction_marker),
      dirichlet_(dirichlet) {
  SetEssentialBoundary(ess_bdr_);
  if (!dirichlet_) {
    Vector zero(fes_->GetVDim());
    zero = 0.0;
    zero_ = std::make_unique<VectorConstantCoefficient>(zero);
    dirichlet_ = zero_.get();
  } else {
    RegisterTimeDependent(*dirichlet_);
  }
  RegisterTimeDependent(traction);
  b_->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction), marker_);
}

void LinearQuasiStaticClampedProblem::UpdateBoundaryValues(real_t /*t*/) {
  u_->ProjectBdrCoefficient(*dirichlet_, ess_bdr_);
}

}  // namespace mfemElasticity
