/**
 * @file elastic_problem.cpp
 * @brief Implementation of ElasticProblemBase, TractionProblem and
 * ClampedProblem.
 */

#include "mfemElasticity/elastic_problem.hpp"

#include <cmath>

#include "mfemElasticity/detail/fem_factory.hpp"

namespace mfemElasticity {

using namespace mfem;

ElasticProblemBase::ElasticProblemBase(
    FiniteElementSpace* fes, const GeneralisedMaxwellRheology& rheology)
    : fes_(fes),
      rheology_(&rheology),
      mu_current_(&rheology.UnrelaxedShearModulus()),
      A_(Operator::MFEM_SPARSEMAT) {
  const int dim = fes_->GetMesh()->Dimension();
  MFEM_VERIFY(fes_->GetVDim() == dim,
              "ElasticProblemBase: the displacement space must have vdim "
              "equal to the space dimension.");
  MFEM_VERIFY(rheology.SpaceDim() == dim,
              "ElasticProblemBase: rheology and mesh dimensions differ.");
#ifdef MFEM_USE_MPI
  pfes_ = dynamic_cast<ParFiniteElementSpace*>(fes_);
  if (pfes_) {
    A_.SetType(Operator::Hypre_ParCSR);
  }
#endif

  // The stiffness integrators live in a template form that is never
  // assembled; each assembly builds a fresh form borrowing them, so that a
  // change of shear modulus never has to reuse a matrix pattern.
  integrators_ = detail::MakeBilinearForm(fes_);
  integrators_->AddDomainIntegrator(
      new ElasticityIntegrator(rheology.BulkModulus(), 1.0, 0.0));
  integrators_->AddDomainIntegrator(
      new ElasticityIntegrator(mu_current_, -2.0 / dim, 1.0));

  b_ = detail::MakeLinearForm(fes_);
  u_ = detail::MakeGridFunction(fes_);
  *u_ = 0.0;
  increment_.SetSize(fes_->GetVSize());
  increment_ = 0.0;
}

bool ElasticProblemBase::IsParallel() const {
#ifdef MFEM_USE_MPI
  return pfes_ != nullptr;
#else
  return false;
#endif
}

FiniteElementSpace& ElasticProblemBase::DisplacementSpace(int i) {
  MFEM_VERIFY(i == 0, "ElasticProblemBase: single displacement field.");
  return *fes_;
}

const GridFunction& ElasticProblemBase::Displacement(int i) const {
  MFEM_VERIFY(i == 0, "ElasticProblemBase: single displacement field.");
  return *u_;
}

const GeneralisedMaxwellRheology& ElasticProblemBase::Rheology(int i) const {
  MFEM_VERIFY(i == 0, "ElasticProblemBase: single displacement field.");
  return *rheology_;
}

void ElasticProblemBase::SetEssentialBoundary(const Array<int>& ess_bdr) {
  Array<int> marker(ess_bdr);
  fes_->GetEssentialTrueDofs(marker, ess_tdof_list_);
  operator_dirty_ = true;
}

void ElasticProblemBase::AssembleForce(real_t t) {
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

void ElasticProblemBase::AddForce(int i, const Vector& f) {
  MFEM_VERIFY(i == 0, "ElasticProblemBase: single displacement field.");
  MFEM_VERIFY(f.Size() == increment_.Size(),
              "AddForce: expected a dual vector in the vdof layout of "
              "DisplacementSpace().");
  increment_ += f;
}

void ElasticProblemBase::SetEffectiveShearModulus(int i, Coefficient& mu_eff) {
  MFEM_VERIFY(i == 0, "ElasticProblemBase: single displacement field.");
  // Always reassemble: the same coefficient object may carry new values.
  mu_current_.SetTarget(&mu_eff);
  operator_dirty_ = true;
}

void ElasticProblemBase::ClearEffectiveShearModulus() {
  auto* mu_u = &rheology_->UnrelaxedShearModulus();
  if (mu_current_.Target() != mu_u) {
    mu_current_.SetTarget(mu_u);
    operator_dirty_ = true;
  }
}

void ElasticProblemBase::AssembleOperator() {
  a_ = detail::MakeBilinearForm(fes_, integrators_.get());
  a_->Assemble();
  a_->FormSystemMatrix(ess_tdof_list_, A_);
  SetupSolver(A_);
  operator_dirty_ = false;
}

void ElasticProblemBase::EnsureOperator() {
  if (operator_dirty_) {
    AssembleOperator();
  }
}

const OperatorHandle& ElasticProblemBase::SystemMatrix() {
  EnsureOperator();
  return A_;
}

bool ElasticProblemBase::Solve() {
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

void ElasticProblemBase::SetupDefaultCG(OperatorHandle& A) {
#ifdef MFEM_USE_MPI
  if (pfes_) {
    auto amg = std::make_unique<HypreBoomerAMG>(*A.As<HypreParMatrix>());
    amg->SetElasticityOptions(pfes_);
    amg->SetPrintLevel(0);
    prec_ = std::move(amg);
    cg_ = std::make_unique<CGSolver>(pfes_->GetComm());
  } else
#endif
  {
    prec_ = std::make_unique<GSSmoother>(*A.As<SparseMatrix>());
    cg_ = std::make_unique<CGSolver>();
  }
  cg_->SetPreconditioner(*prec_);
  cg_->SetOperator(*A.Ptr());
  cg_->SetRelTol(rel_tol_);
  cg_->SetAbsTol(0.0);
  cg_->SetMaxIter(10000);
  cg_->SetPrintLevel(print_level_);
  cg_->iterative_mode = true;
}

void ElasticProblemBase::SetupSolver(OperatorHandle& A) { SetupDefaultCG(A); }

bool ElasticProblemBase::SolveLinearSystem(const Vector& B, Vector& X) {
  if (!SetWarmStartTolerance(*cg_, *prec_, B)) {
    X = 0.0;
    return true;
  }
  cg_->Mult(B, X);
  return cg_->GetConverged();
}

real_t ElasticProblemBase::Dot(const Vector& x, const Vector& y) const {
#ifdef MFEM_USE_MPI
  if (pfes_) {
    return InnerProduct(pfes_->GetComm(), x, y);
  }
#endif
  return InnerProduct(x, y);
}

bool ElasticProblemBase::SetWarmStartTolerance(IterativeSolver& solver,
                                               Solver& prec,
                                               const Vector& B) const {
  Vector z(B.Size());
  prec.Mult(B, z);
  const real_t nom = Dot(B, z);
  if (!(nom > 0.0)) {
    return false;
  }
  solver.SetAbsTol(rel_tol_ * std::sqrt(nom));
  return true;
}

void ElasticProblemBase::RegisterFields(DataCollection& dc) {
  dc.RegisterField("displacement", u_.get());
}

// ---------------------------------------------------------------------------

TractionProblem::TractionProblem(FiniteElementSpace* fes,
                                 const GeneralisedMaxwellRheology& rheology,
                                 VectorCoefficient& traction,
                                 const Array<int>& bdr_marker)
    : ElasticProblemBase(fes, rheology), marker_(bdr_marker) {
  RegisterTimeDependent(traction);
  b_->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction), marker_);
}

void TractionProblem::SetupSolver(OperatorHandle& A) {
  SetupDefaultCG(A);
#ifdef MFEM_USE_MPI
  if (pfes_) {
    rigid_ = std::make_unique<RigidBodySolver>(pfes_->GetComm(), pfes_);
  } else
#endif
  {
    rigid_ = std::make_unique<RigidBodySolver>(fes_);
  }
  rigid_->SetSolver(*cg_);
  // RigidBodySolver propagates its own iterative_mode to the wrapped solver
  // on each Mult(); the final projection inside the wrapper removes any
  // rigid component the warm start might carry along.
  rigid_->iterative_mode = true;
}

bool TractionProblem::SolveLinearSystem(const Vector& B, Vector& X) {
  if (!SetWarmStartTolerance(*cg_, *prec_, B)) {
    X = 0.0;
    return true;
  }
  rigid_->Mult(B, X);
  return cg_->GetConverged();
}

// ---------------------------------------------------------------------------

ClampedProblem::ClampedProblem(FiniteElementSpace* fes,
                               const GeneralisedMaxwellRheology& rheology,
                               const Array<int>& ess_bdr,
                               VectorCoefficient& traction,
                               const Array<int>& traction_marker,
                               VectorCoefficient* dirichlet)
    : ElasticProblemBase(fes, rheology),
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

void ClampedProblem::UpdateBoundaryValues(real_t /*t*/) {
  u_->ProjectBdrCoefficient(*dirichlet_, ess_bdr_);
}

}  // namespace mfemElasticity
