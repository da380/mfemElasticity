/**
 * @file viscoelastic.cpp
 * @brief Implementation of ViscoelasticOperator and the exponential ODE
 * solver adaptors.
 */

#include "mfemElasticity/viscoelastic.hpp"

#include <algorithm>
#include <cmath>

#include "mfemElasticity/bilininteg.hpp"
#include "mfemElasticity/detail/fem_factory.hpp"

namespace mfemElasticity {

using namespace mfem;

namespace detail {

void ExponentialTrapezoidWeights(real_t h, real_t& e, real_t& alpha,
                                 real_t& beta) {
  e = std::exp(-h);
  real_t phi1;  // (1 - e^{-h}) / h
  if (h < 1e-3) {
    phi1 =
        1.0 - h / 2.0 + h * h / 6.0 - h * h * h / 24.0 + h * h * h * h / 120.0;
  } else {
    phi1 = (1.0 - e) / h;
  }
  alpha = phi1 - e;
  beta = 1.0 - phi1;
}

}  // namespace detail

namespace {

// Block-diagonal inverse of the L2 mass matrix of a scalar space, as a
// sparse matrix (exact: the mass matrix is element-block-diagonal).
std::unique_ptr<SparseMatrix> BuildBlockMassInverse(FiniteElementSpace& sfes) {
  BilinearForm mass(&sfes);
  mass.AddDomainIntegrator(new MassIntegrator());
  auto Minv = std::make_unique<SparseMatrix>(sfes.GetVSize(), sfes.GetVSize());
  Array<int> dofs;
  DenseMatrix elmat, inv;
  for (int e = 0; e < sfes.GetNE(); e++) {
    sfes.GetElementDofs(e, dofs);
    mass.ComputeElementMatrix(e, elmat);
    DenseMatrixInverse elinv(elmat);
    elinv.GetInverseMatrix(inv);
    Minv->SetSubMatrix(dofs, dofs, inv, 0);
  }
  Minv->Finalize(0);
  return Minv;
}

// Inverse of the Frobenius metric G_{cc'} = E_c : E_{c'} of the trace-free
// symmetric basis tensors in the TraceFreeSymmetricMatrixIndex layout:
// component (j, k), j >= k, lower triangle column-major, the last diagonal
// entry dropped. E = e_j e_k^T + e_k e_j^T off the diagonal and
// E = e_j e_j^T - e_{d-1} e_{d-1}^T on it.
DenseMatrix TraceFreeMetricInverse(int dim) {
  const int nc = dim * (dim + 1) / 2 - 1;
  std::vector<DenseMatrix> E;
  for (int k = 0; k < dim; k++) {
    for (int j = k; j < dim; j++) {
      if (j == k && j == dim - 1) {
        continue;
      }
      DenseMatrix T(dim);
      T = 0.0;
      if (j == k) {
        T(j, j) = 1.0;
        T(dim - 1, dim - 1) = -1.0;
      } else {
        T(j, k) = 1.0;
        T(k, j) = 1.0;
      }
      E.push_back(T);
    }
  }
  MFEM_VERIFY(static_cast<int>(E.size()) == nc, "trace-free basis size");
  DenseMatrix G(nc);
  for (int c = 0; c < nc; c++) {
    for (int cp = 0; cp < nc; cp++) {
      double s = 0.0;
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          s += E[c](i, j) * E[cp](i, j);
        }
      }
      G(c, cp) = s;
    }
  }
  DenseMatrix Ginv(nc);
  DenseMatrixInverse inv(G);
  inv.GetInverseMatrix(Ginv);
  return Ginv;
}

// Nodal values of a coefficient at the L2 nodes of sfes.
Vector NodalValues(FiniteElementSpace& sfes, Coefficient& c) {
  GridFunction g(&sfes);
  g.ProjectCoefficient(c);
  Vector v(g);
  return v;
}

}  // namespace

ViscoelasticOperator::ViscoelasticOperator(
    QuasiStaticLinearElasticProblem& problem, int internal_order, StrainMap map)
    : TimeDependentOperator(0, 0.0, TimeDependentOperator::EXPLICIT),
      problem_(problem),
      map_(map) {
  const int nf = problem_.NumDisplacementFields();
  MFEM_VERIFY(nf > 0, "ViscoelasticOperator: no displacement fields.");
  fields_.resize(nf);
  offsets_.SetSize(1);
  offsets_[0] = 0;

  for (int i = 0; i < nf; i++) {
    Field& f = fields_[i];
    f.ufes = &problem_.DisplacementSpace(i);
    Mesh* mesh = f.ufes->GetMesh();
    const int dim = mesh->Dimension();
    const auto& rh = problem_.Rheology(i);
    MFEM_VERIFY(rh.SpaceDim() == dim,
                "ViscoelasticOperator: rheology/mesh dimension mismatch.");

    // Default internal order: the smallest L2 order containing dev eps(u)
    // exactly, p - 1 on simplices and p on tensor-product elements (the
    // gradient of a Q_p field has Q_{p,p-1} components).
    int order = internal_order;
    if (order < 0) {
      const int p = f.ufes->GetMaxElementOrder();
      const bool tensor = mesh->HasGeometry(Geometry::SQUARE) ||
                          mesh->HasGeometry(Geometry::CUBE) ||
                          mesh->HasGeometry(Geometry::PRISM) ||
                          mesh->HasGeometry(Geometry::PYRAMID);
      order = tensor ? p : std::max(p - 1, 0);
#ifdef MFEM_USE_MPI
      if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(f.ufes)) {
        int g = order;
        MPI_Allreduce(&order, &g, 1, MPI_INT, MPI_MAX, pfes->GetComm());
        order = g;
      }
#endif
    }
    f.fec = std::make_unique<L2_FECollection>(order, dim);
    f.nc = dim * (dim + 1) / 2 - 1;
    f.dfes = detail::MakeFESpace(*f.ufes, f.fec.get(), f.nc, Ordering::byNODES);
    f.sfes = detail::MakeFESpace(*f.ufes, f.fec.get(), 1);
    f.nd = f.sfes->GetVSize();
    MFEM_VERIFY(f.dfes->GetVSize() == f.nd * f.nc, "layout");

    // Geometric coupling, unit coefficient: all material weighting is
    // applied pointwise at the internal nodes.
    f.B = std::make_unique<MixedBilinearForm>(f.ufes, f.dfes.get());
    f.B->AddDomainIntegrator(
        new DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator());
    f.B->Assemble();
    f.B->Finalize();

    if (map_ == StrainMap::Interpolation) {
      f.D_interp =
          std::make_unique<DiscreteLinearOperator>(f.ufes, f.dfes.get());
      f.D_interp->AddDomainInterpolator(new DeviatoricStrainInterpolator());
      f.D_interp->Assemble();
      f.D_interp->Finalize();
    } else {
      f.Minv = BuildBlockMassInverse(*f.sfes);
      f.Ginv = TraceFreeMetricInverse(dim);
    }

    // Material data sampled once at the internal nodes (L2 nodes are
    // element-interior, so attribute-wise data is sampled correctly).
    f.mu_inf = NodalValues(*f.sfes, rh.LongTermShearModulus());
    const int K = rh.NumBranches();
    f.two_mu.resize(K);
    f.itau.resize(K);
    for (int k = 0; k < K; k++) {
      f.two_mu[k] = NodalValues(*f.sfes, *rh.Branch(k).mu);
      f.two_mu[k] *= 2.0;
      f.itau[k] = NodalValues(*f.sfes, *rh.Branch(k).tau);
      for (int p = 0; p < f.nd; p++) {
        MFEM_VERIFY(f.itau[k][p] > 0.0,
                    "ViscoelasticOperator: relaxation times must be "
                    "positive.");
        f.itau[k][p] = 1.0 / f.itau[k][p];
      }
      f.m_out.push_back(detail::MakeGridFunction(f.dfes.get()));
      *f.m_out.back() = 0.0;
      offsets_.Append(offsets_.Last() + f.nd * f.nc);
    }

    f.mu_eff = detail::MakeGridFunction(f.sfes.get());
    *f.mu_eff = 0.0;
    f.mu_eff_coef = std::make_unique<GridFunctionCoefficient>(f.mu_eff.get());

    f.d.SetSize(f.nd * f.nc);
    f.d_prev.SetSize(f.nd * f.nc);
    f.dual.SetSize(f.nd * f.nc);
    f.zeta.SetSize(f.nd * f.nc);
    f.force.SetSize(f.ufes->GetVSize());
  }

  height = width = offsets_.Last();

  parallel_ = detail::IsParallel(*fields_[0].ufes);
#ifdef MFEM_USE_MPI
  if (parallel_) {
    comm_ = static_cast<ParFiniteElementSpace*>(fields_[0].ufes)->GetComm();
  }
#endif
}

int ViscoelasticOperator::BranchOffset(int i, int k) const {
  int b = 0;
  for (int j = 0; j < i; j++) {
    b += NumBranches(j);
  }
  return offsets_[b + k];
}

Vector ViscoelasticOperator::Branch(const Vector& m, int i, int k) const {
  Vector v;
  v.MakeRef(const_cast<Vector&>(m), BranchOffset(i, k), BranchSize(i));
  return v;
}

void ViscoelasticOperator::ComputeStrain(int i, const GridFunction& u,
                                         Vector& d) const {
  const Field& f = fields_[i];
  d.SetSize(f.nd * f.nc);
  if (map_ == StrainMap::Interpolation) {
    f.D_interp->Mult(u, d);
    return;
  }
  // d = (G^{-1} (x) M^{-1}) B u: scalar mass inverse per component, then the
  // inverse basis-tensor metric across components.
  f.B->Mult(u, f.dual);
  Vector tc, duc;
  for (int c = 0; c < f.nc; c++) {
    tc.MakeRef(f.zeta, c * f.nd, f.nd);
    duc.MakeRef(f.dual, c * f.nd, f.nd);
    f.Minv->Mult(duc, tc);
  }
  for (int c = 0; c < f.nc; c++) {
    for (int p = 0; p < f.nd; p++) {
      real_t v = 0.0;
      for (int cp = 0; cp < f.nc; cp++) {
        v += f.Ginv(c, cp) * f.zeta[cp * f.nd + p];
      }
      d[c * f.nd + p] = v;
    }
  }
}

void ViscoelasticOperator::ComputeAllStrains() const {
  for (int i = 0; i < NumFields(); i++) {
    ComputeStrain(i, problem_.Displacement(i), fields_[i].d);
  }
}

void ViscoelasticOperator::ApplyNodalWeight(const Field& f, const Vector& w,
                                            const Vector& x, Vector& y) {
  y.SetSize(x.Size());
  for (int c = 0; c < f.nc; c++) {
    const int o = c * f.nd;
    for (int p = 0; p < f.nd; p++) {
      y[o + p] = w[p] * x[o + p];
    }
  }
}

void ViscoelasticOperator::AddCoupledForce(int i, const Vector& zeta) const {
  const Field& f = fields_[i];
  f.B->MultTranspose(zeta, f.force);
  problem_.AddForce(i, f.force);
}

bool ViscoelasticOperator::ElasticUpdate(const Vector& m, real_t t) const {
  problem_.AssembleForce(t);
  for (int i = 0; i < NumFields(); i++) {
    const Field& f = fields_[i];
    for (int k = 0; k < NumBranches(i); k++) {
      ApplyNodalWeight(f, f.two_mu[k], Branch(m, i, k), f.zeta);
      AddCoupledForce(i, f.zeta);
    }
  }
  return problem_.Solve();
}

void ViscoelasticOperator::Rate(const Field& f, int k, const Vector& m_k,
                                const Vector& d, Vector& k_out) const {
  for (int c = 0; c < f.nc; c++) {
    const int o = c * f.nd;
    for (int p = 0; p < f.nd; p++) {
      k_out[o + p] = (d[o + p] - m_k[o + p]) * f.itau[k][p];
    }
  }
}

void ViscoelasticOperator::LocalExponentialUpdate(const Field& f, int k,
                                                  real_t dt, Vector& m_k,
                                                  const Vector& d) const {
  for (int p = 0; p < f.nd; p++) {
    const real_t a = std::exp(-dt * f.itau[k][p]);
    const real_t b = 1.0 - a;
    for (int c = 0; c < f.nc; c++) {
      const int idx = c * f.nd + p;
      m_k[idx] = a * m_k[idx] + b * d[idx];
    }
  }
}

void ViscoelasticOperator::SetEffectiveModulus(real_t dt, Scheme scheme) const {
  if (scheme == scheme_ && dt == effective_dt_) {
    return;
  }
  MFEM_VERIFY(problem_.SupportsEffectiveShearModulus(),
              "ViscoelasticOperator: this scheme needs an elastic problem "
              "supporting SetEffectiveShearModulus(); use an explicit "
              "solver or ExponentialEulerSolver instead.");
  for (int i = 0; i < NumFields(); i++) {
    const Field& f = fields_[i];
    GridFunction& mu = *f.mu_eff;
    for (int p = 0; p < f.nd; p++) {
      real_t v = f.mu_inf[p];
      for (int k = 0; k < NumBranches(i); k++) {
        const real_t h = dt * f.itau[k][p];
        real_t factor;
        if (scheme == Scheme::BackwardEuler) {
          factor = 1.0 / (1.0 + h);
        } else {
          real_t e, alpha, beta;
          detail::ExponentialTrapezoidWeights(h, e, alpha, beta);
          factor = 1.0 - beta;
        }
        v += 0.5 * f.two_mu[k][p] * factor;
      }
      mu[p] = v;
    }
    problem_.SetEffectiveShearModulus(i, *f.mu_eff_coef);
  }
  scheme_ = scheme;
  effective_dt_ = dt;
}

void ViscoelasticOperator::UseUnrelaxedOperator() const {
  if (scheme_ != Scheme::None) {
    problem_.ClearEffectiveShearModulus();
    scheme_ = Scheme::None;
    effective_dt_ = -1.0;
  }
}

bool ViscoelasticOperator::CacheMatches(const Vector& m, real_t t) const {
  int ok = cache_valid_ && t == cached_t_ && m.Size() == cached_m_.Size();
  if (ok) {
    for (int j = 0; j < m.Size(); j++) {
      if (m[j] != cached_m_[j]) {
        ok = 0;
        break;
      }
    }
  }
#ifdef MFEM_USE_MPI
  if (parallel_) {
    int g = 0;
    MPI_Allreduce(&ok, &g, 1, MPI_INT, MPI_LAND, comm_);
    ok = g;
  }
#endif
  return ok != 0;
}

void ViscoelasticOperator::UpdateCache(const Vector& m, real_t t) const {
  cached_m_ = m;
  cached_t_ = t;
  cache_valid_ = true;
}

// --- ODE interface -----------------------------------------------------------

void ViscoelasticOperator::Mult(const Vector& m, Vector& k) const {
  const real_t t = GetTime();
  UseUnrelaxedOperator();
  MFEM_VERIFY(ElasticUpdate(m, t),
              "ViscoelasticOperator::Mult: elastic solve failed.");
  ComputeAllStrains();
  k.SetSize(m.Size());
  for (int i = 0; i < NumFields(); i++) {
    const Field& f = fields_[i];
    for (int b = 0; b < NumBranches(i); b++) {
      Vector kb = Branch(k, i, b);
      Rate(f, b, Branch(m, i, b), f.d, kb);
    }
  }
  UpdateCache(m, t);
}

void ViscoelasticOperator::ImplicitSolve(real_t dt, const Vector& m,
                                         Vector& k) {
  const real_t t = GetTime();
  SetEffectiveModulus(dt, Scheme::BackwardEuler);

  // Eliminating m^{n+1} = (m + h d^{n+1}) / (1 + h) leaves the effective
  // operator with the force sum_k B^T 2 mu_k m_k / (1 + h_k).
  problem_.AssembleForce(t);
  for (int i = 0; i < NumFields(); i++) {
    const Field& f = fields_[i];
    for (int b = 0; b < NumBranches(i); b++) {
      Vector mb = Branch(m, i, b);
      for (int c = 0; c < f.nc; c++) {
        const int o = c * f.nd;
        for (int p = 0; p < f.nd; p++) {
          const real_t h = dt * f.itau[b][p];
          f.zeta[o + p] = f.two_mu[b][p] / (1.0 + h) * mb[o + p];
        }
      }
      AddCoupledForce(i, f.zeta);
    }
  }
  MFEM_VERIFY(problem_.Solve(),
              "ViscoelasticOperator::ImplicitSolve: elastic solve failed.");
  ComputeAllStrains();

  k.SetSize(m.Size());
  cached_m_.SetSize(m.Size());
  for (int i = 0; i < NumFields(); i++) {
    const Field& f = fields_[i];
    for (int b = 0; b < NumBranches(i); b++) {
      Vector mb = Branch(m, i, b);
      Vector kb = Branch(k, i, b);
      Vector cb = Branch(cached_m_, i, b);
      for (int c = 0; c < f.nc; c++) {
        const int o = c * f.nd;
        for (int p = 0; p < f.nd; p++) {
          const real_t h = dt * f.itau[b][p];
          const real_t m_new = (mb[o + p] + h * f.d[o + p]) / (1.0 + h);
          kb[o + p] = (m_new - mb[o + p]) / dt;
          cb[o + p] = m_new;
        }
      }
    }
  }
  // The displacement is consistent with the new state m + dt k.
  cached_t_ = t;
  cache_valid_ = true;
}

void ViscoelasticOperator::ExponentialEulerStep(Vector& m, real_t& t,
                                                real_t dt) {
  SetTime(t);
  if (!CacheMatches(m, t)) {
    UseUnrelaxedOperator();
    MFEM_VERIFY(ElasticUpdate(m, t),
                "ViscoelasticOperator::ExponentialEulerStep: elastic solve "
                "failed.");
    ComputeAllStrains();
  }
  for (int i = 0; i < NumFields(); i++) {
    const Field& f = fields_[i];
    for (int b = 0; b < NumBranches(i); b++) {
      Vector mb = Branch(m, i, b);
      LocalExponentialUpdate(f, b, dt, mb, f.d);
    }
  }
  t += dt;
  SetTime(t);
  cache_valid_ = false;  // u is not consistent with the new state
}

void ViscoelasticOperator::ExponentialTrapezoidStep(Vector& m, real_t& t,
                                                    real_t dt) {
  SetTime(t);
  // d^n = D u^n with the unrelaxed operator; reuse if the previous step (or
  // a SolveElastic) left it consistent.
  if (!CacheMatches(m, t)) {
    UseUnrelaxedOperator();
    MFEM_VERIFY(ElasticUpdate(m, t),
                "ViscoelasticOperator::ExponentialTrapezoidStep: elastic "
                "solve failed.");
    ComputeAllStrains();
  }
  for (int i = 0; i < NumFields(); i++) {
    fields_[i].d_prev = fields_[i].d;
  }

  SetEffectiveModulus(dt, Scheme::ExponentialTrapezoid);
  problem_.AssembleForce(t + dt);
  for (int i = 0; i < NumFields(); i++) {
    const Field& f = fields_[i];
    for (int b = 0; b < NumBranches(i); b++) {
      Vector mb = Branch(m, i, b);
      for (int p = 0; p < f.nd; p++) {
        real_t e, alpha, beta;
        detail::ExponentialTrapezoidWeights(dt * f.itau[b][p], e, alpha, beta);
        for (int c = 0; c < f.nc; c++) {
          const int idx = c * f.nd + p;
          f.zeta[idx] = f.two_mu[b][p] * (e * mb[idx] + alpha * f.d_prev[idx]);
        }
      }
      AddCoupledForce(i, f.zeta);
    }
  }
  MFEM_VERIFY(problem_.Solve(),
              "ViscoelasticOperator::ExponentialTrapezoidStep: elastic solve "
              "failed.");
  ComputeAllStrains();  // d^{n+1}

  for (int i = 0; i < NumFields(); i++) {
    const Field& f = fields_[i];
    for (int b = 0; b < NumBranches(i); b++) {
      Vector mb = Branch(m, i, b);
      for (int p = 0; p < f.nd; p++) {
        real_t e, alpha, beta;
        detail::ExponentialTrapezoidWeights(dt * f.itau[b][p], e, alpha, beta);
        for (int c = 0; c < f.nc; c++) {
          const int idx = c * f.nd + p;
          mb[idx] = e * mb[idx] + alpha * f.d_prev[idx] + beta * f.d[idx];
        }
      }
    }
  }
  t += dt;
  SetTime(t);
  // By construction u^{n+1} satisfies K_U u = f + sum B^T 2 mu m^{n+1}.
  UpdateCache(m, t);
}

// --- observation -------------------------------------------------------------

bool ViscoelasticOperator::SolveElastic(const Vector& m, real_t t) {
  SetTime(t);
  if (CacheMatches(m, t)) {
    return true;
  }
  UseUnrelaxedOperator();
  const bool ok = ElasticUpdate(m, t);
  if (ok) {
    ComputeAllStrains();
    UpdateCache(m, t);
  }
  return ok;
}

void ViscoelasticOperator::SyncFields(const Vector& m) {
  for (int i = 0; i < NumFields(); i++) {
    for (int b = 0; b < NumBranches(i); b++) {
      *fields_[i].m_out[b] = Branch(m, i, b);
    }
  }
}

void ViscoelasticOperator::RegisterFields(DataCollection& dc) {
  problem_.RegisterFields(dc);
  for (int i = 0; i < NumFields(); i++) {
    for (int b = 0; b < NumBranches(i); b++) {
      std::string name = "internal_variable";
      if (NumFields() > 1) {
        name += "_field" + std::to_string(i);
      }
      if (NumBranches(i) > 1) {
        name += "_branch" + std::to_string(b);
      }
      dc.RegisterField(name, fields_[i].m_out[b].get());
    }
  }
}

real_t ViscoelasticOperator::MinRelaxationTime() const {
  real_t itau_max = 0.0;
  for (const auto& f : fields_) {
    for (const auto& it : f.itau) {
      if (it.Size() > 0) {
        itau_max = std::max(itau_max, it.Max());
      }
    }
  }
#ifdef MFEM_USE_MPI
  if (parallel_) {
    real_t g = 0.0;
    MPI_Allreduce(&itau_max, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                  comm_);
    itau_max = g;
  }
#endif
  return itau_max > 0.0 ? 1.0 / itau_max : infinity();
}

// --- ODESolver adaptors ------------------------------------------------------

void ExponentialEulerSolver::Init(TimeDependentOperator& f) {
  ODESolver::Init(f);
  op_ = dynamic_cast<ViscoelasticOperator*>(&f);
  MFEM_VERIFY(op_, "ExponentialEulerSolver requires a ViscoelasticOperator.");
}

void ExponentialEulerSolver::Step(Vector& x, real_t& t, real_t& dt) {
  op_->ExponentialEulerStep(x, t, dt);
}

void ExponentialTrapezoidSolver::Init(TimeDependentOperator& f) {
  ODESolver::Init(f);
  op_ = dynamic_cast<ViscoelasticOperator*>(&f);
  MFEM_VERIFY(op_,
              "ExponentialTrapezoidSolver requires a ViscoelasticOperator.");
}

void ExponentialTrapezoidSolver::Step(Vector& x, real_t& t, real_t& dt) {
  op_->ExponentialTrapezoidStep(x, t, dt);
}

}  // namespace mfemElasticity
