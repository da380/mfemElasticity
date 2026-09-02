/**
 * @file elastic_problem.cpp
 * @brief Implementation of LinearElasticProblemBase, TractionProblem and
 * ClampedProblem.
 */

#include "mfemElasticity/elastic_problem.hpp"

#include <cmath>

#include "mfemElasticity/detail/fem_factory.hpp"

namespace mfemElasticity {

using namespace mfem;

LinearElasticProblemBase::LinearElasticProblemBase(FiniteElementSpace* fes,
                                       const mfemElasticity::Rheology& rheology)
    : fes_(fes),
      rheology_(&rheology),
      stiffness_(rheology.MakeStiffness()),
      A_(Operator::MFEM_SPARSEMAT) {
  const int dim = fes_->GetMesh()->Dimension();
  MFEM_VERIFY(fes_->GetVDim() == dim,
              "LinearElasticProblemBase: the displacement space must have vdim "
              "equal to the space dimension.");
  MFEM_VERIFY(rheology.SpaceDim() == dim,
              "LinearElasticProblemBase: rheology and mesh dimensions differ.");
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

bool LinearElasticProblemBase::IsParallel() const {
#ifdef MFEM_USE_MPI
  return pfes_ != nullptr;
#else
  return false;
#endif
}

FiniteElementSpace& LinearElasticProblemBase::DisplacementSpace(int i) {
  MFEM_VERIFY(i == 0, "LinearElasticProblemBase: single displacement field.");
  return *fes_;
}

const GridFunction& LinearElasticProblemBase::Displacement(int i) const {
  MFEM_VERIFY(i == 0, "LinearElasticProblemBase: single displacement field.");
  return *u_;
}

const mfemElasticity::Rheology& LinearElasticProblemBase::Rheology(int i) const {
  MFEM_VERIFY(i == 0, "LinearElasticProblemBase: single displacement field.");
  return *rheology_;
}

void LinearElasticProblemBase::SetEssentialBoundary(const Array<int>& ess_bdr) {
  Array<int> marker(ess_bdr);
  fes_->GetEssentialTrueDofs(marker, ess_tdof_list_);
  operator_dirty_ = true;
}

void LinearElasticProblemBase::AssembleForce(real_t t) {
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

void LinearElasticProblemBase::AddForce(int i, const Vector& f) {
  MFEM_VERIFY(i == 0, "LinearElasticProblemBase: single displacement field.");
  MFEM_VERIFY(f.Size() == increment_.Size(),
              "AddForce: expected a dual vector in the vdof layout of "
              "DisplacementSpace().");
  increment_ += f;
}

void LinearElasticProblemBase::SetRelaxationWeights(
    int i, const std::vector<Coefficient*>& beta) {
  MFEM_VERIFY(i == 0, "LinearElasticProblemBase: single displacement field.");
  // Always reassemble: the same coefficient objects may carry new values.
  stiffness_->SetRelaxationWeights(beta);
  operator_dirty_ = true;
}

void LinearElasticProblemBase::ClearRelaxationWeights() {
  if (stiffness_->IsRelaxed()) {
    stiffness_->ClearRelaxationWeights();
    operator_dirty_ = true;
  }
}

void LinearElasticProblemBase::AssembleOperator() {
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

void LinearElasticProblemBase::NoteIterations(int its) {
  total_its_ += its;
  if (prec_baseline_its_ < 0) {
    prec_baseline_its_ = its;
  } else if (its > prec_reuse_ * prec_baseline_its_) {
    prec_stale_ = true;
  }
}

void LinearElasticProblemBase::EnsureOperator() {
  if (operator_dirty_) {
    AssembleOperator();
  }
}

const OperatorHandle& LinearElasticProblemBase::SystemMatrix() {
  EnsureOperator();
  return A_;
}

bool LinearElasticProblemBase::Solve() {
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

void LinearElasticProblemBase::SetupDefaultCG(OperatorHandle& A) {
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
#ifdef MFEM_USE_MPI
  if (pfes_) {
    cg_ = std::make_unique<CGSolver>(pfes_->GetComm());
  } else
#endif
  {
    cg_ = std::make_unique<CGSolver>();
  }
  // Operator before preconditioner: SetOperator would otherwise reset the
  // (reused) preconditioner onto the new matrix.
  cg_->SetOperator(*A.Ptr());
  cg_->SetPreconditioner(*prec_);
  cg_->SetRelTol(rel_tol_);
  cg_->SetAbsTol(0.0);
  cg_->SetMaxIter(10000);
  cg_->SetPrintLevel(print_level_);
  cg_->iterative_mode = true;
}

void LinearElasticProblemBase::SetupSolver(OperatorHandle& A) { SetupDefaultCG(A); }

bool LinearElasticProblemBase::SolveLinearSystem(const Vector& B, Vector& X) {
  if (!SetWarmStartTolerance(*cg_, *prec_, B)) {
    X = 0.0;
    return true;
  }
  cg_->Mult(B, X);
  NoteIterations(cg_->GetNumIterations());
  return cg_->GetConverged();
}

real_t LinearElasticProblemBase::Dot(const Vector& x, const Vector& y) const {
#ifdef MFEM_USE_MPI
  if (pfes_) {
    return InnerProduct(pfes_->GetComm(), x, y);
  }
#endif
  return InnerProduct(x, y);
}

bool LinearElasticProblemBase::SetWarmStartTolerance(IterativeSolver& solver,
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

void LinearElasticProblemBase::RegisterFields(DataCollection& dc) {
  dc.RegisterField("displacement", u_.get());
}

// ---------------------------------------------------------------------------

TractionProblem::TractionProblem(FiniteElementSpace* fes,
                                 const mfemElasticity::Rheology& rheology,
                                 VectorCoefficient& traction,
                                 const Array<int>& bdr_marker)
    : LinearElasticProblemBase(fes, rheology), marker_(bdr_marker) {
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
  NoteIterations(cg_->GetNumIterations());
  return cg_->GetConverged();
}

// ---------------------------------------------------------------------------

ClampedProblem::ClampedProblem(FiniteElementSpace* fes,
                               const mfemElasticity::Rheology& rheology,
                               const Array<int>& ess_bdr,
                               VectorCoefficient& traction,
                               const Array<int>& traction_marker,
                               VectorCoefficient* dirichlet)
    : LinearElasticProblemBase(fes, rheology),
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
