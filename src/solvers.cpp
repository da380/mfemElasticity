#include "mfemElasticity/solvers.hpp"

namespace mfemElasticity {

RigidTranslation::RigidTranslation(int dimension, int component)
    : mfem::VectorCoefficient(dimension), _component{component} {
  MFEM_ASSERT(component >= 0 && component < dimension,
              "component out of range");
}

void RigidTranslation::SetComponent(int component) { _component = component; }

void RigidTranslation::Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                            const mfem::IntegrationPoint &ip) {
  V.SetSize(vdim);
  V = 0.;
  V[_component] = 1;
}

RigidRotation::RigidRotation(int dimension, int component)
    : mfem::VectorCoefficient(dimension), _component{component} {
  MFEM_ASSERT(component >= 0 && component < dimension,
              "component out of range");
  MFEM_ASSERT(dimension == 3 || component == 2,
              "In two dimensions only z-rotation defined");
}

void RigidRotation::SetComponent(int component) { _component = component; }

void RigidRotation::Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                         const mfem::IntegrationPoint &ip) {
  V.SetSize(vdim);
  _x.SetSize(vdim);
  T.Transform(ip, _x);
  if (_component == 0) {
    V[0] = 0;
    V[1] = -_x[2];
    V[2] = _x[1];
  } else if (_component == 1) {
    V[0] = _x[2];
    V[1] = 0;
    V[2] = -_x[0];
  } else {
    V[0] = -_x[1];
    V[1] = _x[0];
    if (vdim == 3) V[2] = 0;
  }
}

RigidBodySolver::RigidBodySolver(mfem::FiniteElementSpace *fes)
    : mfem::Solver(0, false), _fes{fes}, _parallel{false} {
  auto vDim = _fes->GetVDim();
  MFEM_ASSERT(vDim == 2 || vDim == 3, "Dimensions must be two or three");
  SetRigidBodyFields();
}

#ifdef MFEM_USE_MPI
RigidBodySolver::RigidBodySolver(MPI_Comm comm,
                                 mfem::ParFiniteElementSpace *fes)
    : mfem::Solver(0, false),
      _fes{fes},
      _pfes{fes},
      _comm{comm},
      _parallel{true} {
  auto vDim = _fes->GetVDim();
  MFEM_ASSERT(vDim == 2 || vDim == 3, "Dimensions must be two or three");
  SetRigidBodyFields();
}
#endif

mfem::real_t RigidBodySolver::Dot(const mfem::Vector &x,
                                  const mfem::Vector &y) const {
#ifdef MFEM_USE_MPI
  return _parallel ? mfem::InnerProduct(_comm, x, y) : mfem::InnerProduct(x, y);

#else
  return mfem::InnerProduct(x, y);
#endif
}

mfem::real_t RigidBodySolver::Norm(const mfem::Vector &x) const {
  return std::sqrt(Dot(x, x));
}

void RigidBodySolver::SetRigidBodyFields() {
  auto vDim = _fes->GetVDim();

  // Set up a local grid function.
  std::unique_ptr<mfem::GridFunction> u;
  if (_parallel) {
#ifdef MFEM_USE_MPI
    u = std::make_unique<mfem::ParGridFunction>(_pfes);
#endif
  } else {
    u = std::make_unique<mfem::GridFunction>(_fes);
  }

  // Set the translations.
  for (auto component = 0; component < vDim; component++) {
    auto v = RigidTranslation(vDim, component);
    u->ProjectCoefficient(v);
    auto tv = std::make_unique<mfem::Vector>();
    u->GetTrueDofs(*tv);
    _u.push_back(std::move(tv));
  }

  // Set the rotations.
  if (vDim == 2) {
    auto v = RigidRotation(vDim, 2);
    u->ProjectCoefficient(v);
    auto tv = std::make_unique<mfem::Vector>();
    u->GetTrueDofs(*tv);
    _u.push_back(std::move(tv));

  } else {
    for (auto component = 0; component < vDim; component++) {
      auto v = RigidRotation(vDim, component);
      u->ProjectCoefficient(v);
      auto tv = std::make_unique<mfem::Vector>();
      u->GetTrueDofs(*tv);
      _u.push_back(std::move(tv));
    }
  }

  GramSchmidt();
}

void RigidBodySolver::GramSchmidt() {
  for (auto i = 0; i < GetNullDim(); i++) {
    auto &u = *_u[i];
    for (auto j = 0; j < i; j++) {
      auto &v = *_u[j];
      auto product = Dot(u, v);
      u.Add(-product, v);
    }
    auto norm = Norm(u);
    u /= norm;
  }
}

int RigidBodySolver::GetNullDim() const {
  auto vDim = _fes->GetVDim();
  return vDim * (vDim + 1) / 2;
}

void RigidBodySolver::ProjectOrthogonalToRigidBody(const mfem::Vector &x,
                                                   mfem::Vector &y) const {
  y = x;
  for (auto i = 0; i < GetNullDim(); i++) {
    auto &u = *_u[i];
    auto product = Dot(y, u);
    y.Add(-product, u);
  }
}

void RigidBodySolver::SetSolver(mfem::Solver &solver) {
  _solver = &solver;
  height = _solver->Height();
  width = _solver->Width();
  MFEM_VERIFY(height == width, "Solver must be a square operator");
}

void RigidBodySolver::SetOperator(const mfem::Operator &op) {
  MFEM_VERIFY(_solver, "Solver hasn't been set, call SetSolver() first.");
  _solver->SetOperator(op);
  height = _solver->Height();
  width = _solver->Width();
  MFEM_VERIFY(height == width, "Solver must be a square Operator!");
}

void RigidBodySolver::Mult(const mfem::Vector &b, mfem::Vector &x) const {
  ProjectOrthogonalToRigidBody(b, _b);
  _solver->iterative_mode = iterative_mode;
  _solver->Mult(_b, x);
  ProjectOrthogonalToRigidBody(x, x);
}

}  // namespace mfemElasticity