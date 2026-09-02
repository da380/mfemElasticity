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
#include "mfemElasticity/elastic_tensor.hpp"
#include "mfemElasticity/relaxation_law.hpp"

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

// Inverse of the Frobenius metric G_{cc'} = E_c : E_{c'} of the symmetric
// basis tensors in the (TraceFree)SymmetricMatrixIndex layout: component
// (j, k), j >= k, lower triangle column-major, with the last diagonal entry
// dropped in the trace-free case. E = e_j e_k^T + e_k e_j^T off the
// diagonal; on it E = e_j e_j^T (full) or e_j e_j^T - e_{d-1} e_{d-1}^T
// (trace-free).
DenseMatrix MetricInverse(int dim, bool tracefree) {
  const int nc = dim * (dim + 1) / 2 - (tracefree ? 1 : 0);
  std::vector<DenseMatrix> E;
  for (int k = 0; k < dim; k++) {
    for (int j = k; j < dim; j++) {
      if (tracefree && j == k && j == dim - 1) {
        continue;
      }
      DenseMatrix T(dim);
      T = 0.0;
      if (j == k) {
        T(j, j) = 1.0;
        if (tracefree) {
          T(dim - 1, dim - 1) = -1.0;
        }
      } else {
        T(j, k) = 1.0;
        T(k, j) = 1.0;
      }
      E.push_back(T);
    }
  }
  MFEM_VERIFY(static_cast<int>(E.size()) == nc, "symmetric basis size");
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

// Branch tensor C_k sampled at the L2 nodes of sfes, stored per node as the
// n_s x n_s matrix W acting on unscaled tensor components: sigma_s =
// sum_t W_st m_t, i.e. W_st = C^_st w_t / (a_s a_t) with the Mandel scales
// a and the multiplicity w_t (2 off the diagonal) of the symmetric pair.
template <class Eval>
Vector NodalTensors(FiniteElementSpace& sfes, Eval eval) {
  const int dim = sfes.GetMesh()->Dimension();
  const int ns = SymmetricTensorBasis::Size(dim);
  Vector W(sfes.GetVSize() * ns * ns);
  W = 0.0;
  std::vector<real_t> a(ns), w(ns);
  for (int s = 0; s < ns; s++) {
    int j, l;
    SymmetricTensorBasis::Component(dim, s, j, l);
    a[s] = SymmetricTensorBasis::Scale(j, l);
    w[s] = j == l ? 1.0 : 2.0;
  }
  DenseMatrix Cm(ns);
  Array<int> dofs;
  for (int e = 0; e < sfes.GetNE(); e++) {
    const FiniteElement* fe = sfes.GetFE(e);
    ElementTransformation* T = sfes.GetElementTransformation(e);
    const IntegrationRule& nodes = fe->GetNodes();
    sfes.GetElementDofs(e, dofs);
    for (int i = 0; i < dofs.Size(); i++) {
      const IntegrationPoint& ip = nodes.IntPoint(i);
      T->SetIntPoint(&ip);
      eval(*T, ip, Cm);
      real_t* Wp = W.GetData() + static_cast<std::size_t>(dofs[i]) * ns * ns;
      for (int s = 0; s < ns; s++) {
        for (int t = 0; t < ns; t++) {
          Wp[s * ns + t] = Cm(s, t) * w[t] / (a[s] * a[t]);
        }
      }
    }
  }
  return W;
}

Vector NodalBranchTensors(FiniteElementSpace& sfes, const Rheology& rh,
                          int k) {
  return NodalTensors(sfes, [&](ElementTransformation& T,
                                const IntegrationPoint& ip, DenseMatrix& C) {
    rh.BranchModulus(k, T, ip, C);
  });
}

Vector NodalUnrelaxedTensors(FiniteElementSpace& sfes, const Rheology& rh) {
  return NodalTensors(sfes, [&](ElementTransformation& T,
                                const IntegrationPoint& ip, DenseMatrix& C) {
    rh.UnrelaxedModulus(T, ip, C);
  });
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

    // Default internal order: the smallest L2 order containing eps(u)
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
    f.tracefree = rh.TraceFreeInternalVariables();
    f.ns = dim * (dim + 1) / 2;
    f.nc = f.tracefree ? f.ns - 1 : f.ns;
    f.dfes = detail::MakeFESpace(*f.ufes, f.fec.get(), f.nc, Ordering::byNODES);
    f.sfes = detail::MakeFESpace(*f.ufes, f.fec.get(), 1);
    f.nd = f.sfes->GetVSize();
    MFEM_VERIFY(f.dfes->GetVSize() == f.nd * f.nc, "layout");

    // Geometric coupling, unit coefficient: all material weighting is
    // applied pointwise at the internal nodes. Trace-free deviatoric
    // strain for the isotropic body, the full strain otherwise.
    f.B = std::make_unique<MixedBilinearForm>(f.ufes, f.dfes.get());
    if (f.tracefree) {
      f.B->AddDomainIntegrator(
          new DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator());
    } else {
      f.B->AddDomainIntegrator(new DomainSymmetricMatrixStrainIntegrator());
    }
    f.B->Assemble();
    f.B->Finalize();

    if (map_ == StrainMap::Interpolation) {
      f.D_interp =
          std::make_unique<DiscreteLinearOperator>(f.ufes, f.dfes.get());
      if (f.tracefree) {
        f.D_interp->AddDomainInterpolator(new DeviatoricStrainInterpolator());
      } else {
        f.D_interp->AddDomainInterpolator(new StrainInterpolator());
      }
      f.D_interp->Assemble();
      f.D_interp->Finalize();
    } else {
      f.Minv = BuildBlockMassInverse(*f.sfes);
      f.Ginv = MetricInverse(dim, f.tracefree);
    }

    // Material data sampled once at the internal nodes (L2 nodes are
    // element-interior, so attribute-wise data is sampled correctly).
    const int K = rh.NumBranches();
    f.branch_modulus.resize(K);
    f.itau0.resize(K);
    f.itau.resize(K);
    f.law.resize(K, nullptr);
    f.law_params.resize(K);
    f.num_params.resize(K, 0);
    f.linear = rh.IsLinear();
    linear_ = linear_ && f.linear;
    for (int k = 0; k < K; k++) {
      if (f.tracefree) {
        f.branch_modulus[k] = NodalValues(*f.sfes, rh.BranchShearModulus(k));
        f.branch_modulus[k] *= 2.0;
      } else {
        f.branch_modulus[k] = NodalBranchTensors(*f.sfes, rh, k);
      }
      f.itau0[k] = NodalValues(*f.sfes, rh.RelaxationTime(k));
      for (int p = 0; p < f.nd; p++) {
        MFEM_VERIFY(f.itau0[k][p] > 0.0,
                    "ViscoelasticOperator: relaxation times must be "
                    "positive.");
        f.itau0[k][p] = 1.0 / f.itau0[k][p];
      }
      f.itau[k] = f.itau0[k];
      const RelaxationLaw* law = rh.Law(k);
      if (law && law->IsStateDependent()) {
        f.law[k] = law;
        const int np = law->NumParameters();
        f.num_params[k] = np;
        f.law_params[k].SetSize(f.nd * np);
        for (int i = 0; i < np; i++) {
          Vector v = NodalValues(*f.sfes, law->Parameter(i));
          for (int p = 0; p < f.nd; p++) {
            f.law_params[k][p * np + i] = v[p];
          }
        }
      }
      f.m_out.push_back(detail::MakeGridFunction(f.dfes.get()));
      *f.m_out.back() = 0.0;
      offsets_.Append(offsets_.Last() + f.nd * f.nc);

      f.beta.push_back(detail::MakeGridFunction(f.sfes.get()));
      *f.beta.back() = 1.0;
      f.beta_coef.push_back(
          std::make_unique<GridFunctionCoefficient>(f.beta.back().get()));
      f.beta_ptrs.push_back(f.beta_coef.back().get());
    }

    if (!f.linear && !f.tracefree) {
      f.CU = NodalUnrelaxedTensors(*f.sfes, rh);
    }

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

void ViscoelasticOperator::ApplyBranchModulus(const Field& f, int k,
                                              const Vector& x, Vector& y) {
  y.SetSize(x.Size());
  const Vector& w = f.branch_modulus[k];
  if (f.tracefree) {
    for (int c = 0; c < f.nc; c++) {
      const int o = c * f.nd;
      for (int p = 0; p < f.nd; p++) {
        y[o + p] = w[p] * x[o + p];
      }
    }
    return;
  }
  const int ns = f.ns;
  for (int p = 0; p < f.nd; p++) {
    const real_t* Wp = w.GetData() + static_cast<std::size_t>(p) * ns * ns;
    for (int s = 0; s < ns; s++) {
      real_t v = 0.0;
      for (int t = 0; t < ns; t++) {
        v += Wp[s * ns + t] * x[t * f.nd + p];
      }
      y[s * f.nd + p] = v;
    }
  }
}

void ViscoelasticOperator::EvaluateRelaxationTimes(int i, const Vector& d,
                                                   const Vector& m) const {
  const Field& f = fields_[i];
  if (f.linear) {
    return;
  }
  const int dim = f.ufes->GetMesh()->Dimension();
  const int ns = f.ns, nc = f.nc, nd = f.nd;
  const int K = NumBranches(i);
  LocalState s(dim);
  std::vector<Vector> mk(K);
  for (int k = 0; k < K; k++) {
    mk[k] = Branch(m, i, k);
  }
  // Index of the dropped diagonal component in the trace-free layout.
  const int last_diag = SymmetricTensorBasis::Index(dim, dim - 1, dim - 1);
  auto expand = [&](const Vector& x, int p, Vector& full) {
    for (int c = 0; c < nc; c++) {
      full[c] = x[c * nd + p];
    }
    if (f.tracefree) {
      real_t tr = 0.0;
      for (int j = 0; j < dim - 1; j++) {
        tr += full[SymmetricTensorBasis::Index(dim, j, j)];
      }
      full[last_diag] = -tr;
    }
  };
  Vector tmp(ns);
  for (int p = 0; p < nd; p++) {
    expand(d, p, s.strain);
    // Stress: trace-free isotropic sum_j 2 mu_j (d - m_j) (deviatoric part
    // only); anisotropic C_U eps - sum_j C_j m_j through the nodal W forms.
    if (f.tracefree) {
      s.stress = 0.0;
      for (int j = 0; j < K; j++) {
        expand(mk[j], p, tmp);
        const real_t two_mu = f.branch_modulus[j][p];
        for (int c = 0; c < ns; c++) {
          s.stress[c] += two_mu * (s.strain[c] - tmp[c]);
        }
      }
    } else {
      const real_t* Wu = f.CU.GetData() + static_cast<std::size_t>(p) * ns * ns;
      for (int a = 0; a < ns; a++) {
        real_t v = 0.0;
        for (int b = 0; b < ns; b++) {
          v += Wu[a * ns + b] * s.strain[b];
        }
        s.stress[a] = v;
      }
      for (int j = 0; j < K; j++) {
        const real_t* Wj =
            f.branch_modulus[j].GetData() + static_cast<std::size_t>(p) * ns * ns;
        expand(mk[j], p, tmp);
        for (int a = 0; a < ns; a++) {
          real_t v = 0.0;
          for (int b = 0; b < ns; b++) {
            v += Wj[a * ns + b] * tmp[b];
          }
          s.stress[a] -= v;
        }
      }
    }
    for (int k = 0; k < K; k++) {
      if (!f.law[k]) {
        continue;
      }
      expand(mk[k], p, s.m);
      const real_t* params = f.law_params[k].GetData() + p * f.num_params[k];
      const real_t F = f.law[k]->Factor(params, s);
      MFEM_ASSERT(F > 0.0, "relaxation law factor must be positive");
      f.itau[k][p] = f.itau0[k][p] / F;
    }
  }
  tau_version_++;
}

std::vector<std::vector<Vector>> ViscoelasticOperator::SaveRelaxationTimes()
    const {
  std::vector<std::vector<Vector>> saved(NumFields());
  for (int i = 0; i < NumFields(); i++) {
    saved[i] = fields_[i].itau;
  }
  return saved;
}

real_t ViscoelasticOperator::RelaxationTimeChange(
    const std::vector<std::vector<Vector>>& old_itau) const {
  real_t change = 0.0;
  for (int i = 0; i < NumFields(); i++) {
    const Field& f = fields_[i];
    if (f.linear) {
      continue;
    }
    for (int k = 0; k < NumBranches(i); k++) {
      for (int p = 0; p < f.nd; p++) {
        const real_t a = old_itau[i][k][p], b = f.itau[k][p];
        change = std::max(change, std::abs(b - a) / std::max(a, b));
      }
    }
  }
  return GlobalMax(change);
}

real_t ViscoelasticOperator::GlobalMax(real_t v) const {
#ifdef MFEM_USE_MPI
  if (parallel_) {
    real_t g = 0.0;
    MPI_Allreduce(&v, &g, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, comm_);
    return g;
  }
#endif
  return v;
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
      ApplyBranchModulus(f, k, Branch(m, i, k), f.zeta);
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
  if (scheme == scheme_ && dt == effective_dt_ &&
      effective_version_ == tau_version_) {
    return;
  }
  MFEM_VERIFY(problem_.SupportsRelaxationWeights(),
              "ViscoelasticOperator: this scheme needs an elastic problem "
              "supporting SetRelaxationWeights(); use an explicit solver or "
              "ExponentialEulerSolver instead.");
  // Nodal weights beta_k: the effective modulus is C_inf + sum_k beta_k C_k
  // after eliminating m_k^{n+1}.
  for (int i = 0; i < NumFields(); i++) {
    const Field& f = fields_[i];
    for (int k = 0; k < NumBranches(i); k++) {
      GridFunction& beta_k = *f.beta[k];
      for (int p = 0; p < f.nd; p++) {
        const real_t h = dt * f.itau[k][p];
        if (scheme == Scheme::BackwardEuler) {
          beta_k[p] = 1.0 / (1.0 + h);
        } else {
          real_t e, alpha, beta;
          detail::ExponentialTrapezoidWeights(h, e, alpha, beta);
          beta_k[p] = 1.0 - beta;
        }
      }
    }
    problem_.SetRelaxationWeights(i, f.beta_ptrs);
  }
  scheme_ = scheme;
  effective_dt_ = dt;
  effective_version_ = tau_version_;
}

void ViscoelasticOperator::UseUnrelaxedOperator() const {
  if (scheme_ != Scheme::None) {
    problem_.ClearRelaxationWeights();
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
    EvaluateRelaxationTimes(i, f.d, m);
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
  // Eliminating m^{n+1} = (m + h d^{n+1}) / (1 + h) leaves the effective
  // operator with the force sum_k B^T C_k m_k / (1 + h_k). With
  // state-dependent times the h_k of the end state are found by
  // predictor–corrector: the current (last evaluated) times first, then
  // those of the end state, repeated.
  k.SetSize(m.Size());
  cached_m_.SetSize(m.Size());
  last_passes_ = 0;
  for (int pass = 0;; pass++) {
    SetEffectiveModulus(dt, Scheme::BackwardEuler);
    problem_.AssembleForce(t);
    for (int i = 0; i < NumFields(); i++) {
      const Field& f = fields_[i];
      for (int b = 0; b < NumBranches(i); b++) {
        Vector mb = Branch(m, i, b);
        for (int c = 0; c < f.nc; c++) {
          const int o = c * f.nd;
          for (int p = 0; p < f.nd; p++) {
            const real_t h = dt * f.itau[b][p];
            f.dual[o + p] = mb[o + p] / (1.0 + h);
          }
        }
        ApplyBranchModulus(f, b, f.dual, f.zeta);
        AddCoupledForce(i, f.zeta);
      }
    }
    MFEM_VERIFY(problem_.Solve(),
                "ViscoelasticOperator::ImplicitSolve: elastic solve failed.");
    ComputeAllStrains();

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
    last_passes_ = pass;
    if (linear_ || pass >= max_corrector_) {
      break;
    }
    // Corrector: times at the end state (d^{n+1}, m^{n+1}).
    const auto old_itau = SaveRelaxationTimes();
    for (int i = 0; i < NumFields(); i++) {
      EvaluateRelaxationTimes(i, fields_[i].d, cached_m_);
    }
    if (RelaxationTimeChange(old_itau) < corrector_tol_) {
      break;
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
    EvaluateRelaxationTimes(i, f.d, m);  // frozen over the step
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
    EvaluateRelaxationTimes(i, fields_[i].d_prev, m);  // predictor: start
  }

  // ETD1 prediction of the step (times and strain frozen at the start),
  // kept for the error estimate.
  const Vector m_start(m);
  predictor_diff_ = m;
  m_scale_.SetSize(m.Size());
  for (int i = 0; i < NumFields(); i++) {
    const Field& f = fields_[i];
    for (int b = 0; b < NumBranches(i); b++) {
      Vector pb = Branch(predictor_diff_, i, b);
      LocalExponentialUpdate(f, b, dt, pb, f.d_prev);
    }
  }

  last_passes_ = 0;
  for (int pass = 0;; pass++) {
    SetEffectiveModulus(dt, Scheme::ExponentialTrapezoid);
    problem_.AssembleForce(t + dt);
    for (int i = 0; i < NumFields(); i++) {
      const Field& f = fields_[i];
      for (int b = 0; b < NumBranches(i); b++) {
        Vector mb = Branch(m_start, i, b);
        for (int p = 0; p < f.nd; p++) {
          real_t e, alpha, beta;
          detail::ExponentialTrapezoidWeights(dt * f.itau[b][p], e, alpha,
                                              beta);
          for (int c = 0; c < f.nc; c++) {
            const int idx = c * f.nd + p;
            f.dual[idx] = e * mb[idx] + alpha * f.d_prev[idx];
          }
        }
        ApplyBranchModulus(f, b, f.dual, f.zeta);
        AddCoupledForce(i, f.zeta);
      }
    }
    MFEM_VERIFY(problem_.Solve(),
                "ViscoelasticOperator::ExponentialTrapezoidStep: elastic "
                "solve failed.");
    ComputeAllStrains();  // d^{n+1}

    for (int i = 0; i < NumFields(); i++) {
      const Field& f = fields_[i];
      for (int b = 0; b < NumBranches(i); b++) {
        Vector mb = Branch(m, i, b);
        Vector m0 = Branch(m_start, i, b);
        for (int p = 0; p < f.nd; p++) {
          real_t e, alpha, beta;
          detail::ExponentialTrapezoidWeights(dt * f.itau[b][p], e, alpha,
                                              beta);
          for (int c = 0; c < f.nc; c++) {
            const int idx = c * f.nd + p;
            mb[idx] = e * m0[idx] + alpha * f.d_prev[idx] + beta * f.d[idx];
          }
        }
      }
    }
    last_passes_ = pass;
    if (linear_ || pass >= max_corrector_) {
      break;
    }
    // Corrector: times at the midpoint state.
    const auto old_itau = SaveRelaxationTimes();
    Vector m_mid(m);
    m_mid += m_start;
    m_mid *= 0.5;
    for (int i = 0; i < NumFields(); i++) {
      Field& f = fields_[i];
      f.dual = f.d;
      f.dual += f.d_prev;
      f.dual *= 0.5;
      EvaluateRelaxationTimes(i, f.dual, m_mid);
    }
    if (RelaxationTimeChange(old_itau) < corrector_tol_) {
      break;
    }
  }

  for (int j = 0; j < m.Size(); j++) {
    predictor_diff_[j] = m[j] - predictor_diff_[j];
    m_scale_[j] = std::max(std::abs(m_start[j]), std::abs(m[j]));
  }
  t += dt;
  SetTime(t);
  // By construction u^{n+1} satisfies K_U u = f + sum B^T C m^{n+1}.
  UpdateCache(m, t);
}

real_t ViscoelasticOperator::ErrorEstimate(real_t rtol, real_t atol) const {
  real_t sum = 0.0;
  for (int j = 0; j < predictor_diff_.Size(); j++) {
    const real_t w = atol + rtol * m_scale_[j];
    const real_t r = predictor_diff_[j] / w;
    sum += r * r;
  }
  real_t n = static_cast<real_t>(predictor_diff_.Size());
#ifdef MFEM_USE_MPI
  if (parallel_) {
    real_t buf[2] = {sum, n}, g[2] = {0.0, 0.0};
    MPI_Allreduce(buf, g, 2, MPITypeMap<real_t>::mpi_type, MPI_SUM, comm_);
    sum = g[0];
    n = g[1];
  }
#endif
  return n > 0.0 ? std::sqrt(sum / n) : 0.0;
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

void AdaptiveExponentialTrapezoidSolver::Init(TimeDependentOperator& f) {
  ODESolver::Init(f);
  op_ = dynamic_cast<ViscoelasticOperator*>(&f);
  MFEM_VERIFY(op_, "AdaptiveExponentialTrapezoidSolver requires a "
                   "ViscoelasticOperator.");
}

void AdaptiveExponentialTrapezoidSolver::Step(Vector& x, real_t& t,
                                              real_t& dt) {
  real_t h = std::min(dt, dt_max_);
  h = std::max(h, dt_min_);
  for (;;) {
    trial_ = x;
    real_t t_trial = t;
    op_->ExponentialTrapezoidStep(trial_, t_trial, h);
    last_err_ = op_->ErrorEstimate(rtol_, atol_);
    if (last_err_ <= 1.0 || h <= dt_min_) {
      x = trial_;
      t = t_trial;
      accepted_++;
      break;
    }
    rejected_++;
    h = std::max(dt_min_,
                 h * std::max(shrink_, safety_ / std::sqrt(last_err_)));
  }
  const real_t factor =
      last_err_ > 0.0
          ? std::min(grow_, std::max(shrink_, safety_ / std::sqrt(last_err_)))
          : grow_;
  dt = std::min(dt_max_, std::max(dt_min_, h * factor));
}

int AdaptiveExponentialTrapezoidSolver::Integrate(Vector& x, real_t& t,
                                                  real_t t_final, real_t& dt) {
  int steps = 0;
  while (t < t_final - 1e-14 * std::max<real_t>(1.0, std::abs(t_final))) {
    real_t h = std::min(dt, t_final - t);
    // Avoid a tiny last step: split the remainder in two if needed.
    if (t + h < t_final && t_final - (t + h) < 0.1 * h) {
      h = 0.5 * (t_final - t);
    }
    Step(x, t, h);
    dt = h;
    steps++;
  }
  return steps;
}

}  // namespace mfemElasticity
