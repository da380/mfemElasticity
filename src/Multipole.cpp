#include "mfemElasticity/Multipole.hpp"

namespace mfemElasticity {

namespace Multipole {

Poisson::Poisson(mfem::FiniteElementSpace* tr_fes,
                 mfem::FiniteElementSpace* te_fes, int coeff_dim)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _coeff_dim{coeff_dim},
      _dom_marker(_tr_fes->GetMesh()->attributes.Max()),
      _bdr_marker(_tr_fes->GetMesh()->bdr_attributes.Max()),
      _lmat(te_fes->GetVSize(), _coeff_dim),
      _rmat(tr_fes->GetVSize(), _coeff_dim) {
  CheckMesh();
  _bdr_marker = 0;
  _dom_marker = 0;
#ifndef MFEM_THREAD_SAFE
  _c.SetSize(_coeff_dim);
  _x.SetSize(_tr_fes->GetMesh()->Dimension());
#endif
}

#ifdef MFEM_USE_MPI
Poisson::Poisson(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                 mfem::ParFiniteElementSpace* te_fes, int coeff_dim)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _parallel{true},
      _comm{comm},
      _tr_pfes{tr_fes},
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _te_pfes{te_fes},
      _coeff_dim{coeff_dim},
      _dom_marker(_tr_fes->GetMesh()->attributes.Max()),
      _bdr_marker(_tr_fes->GetMesh()->bdr_attributes.Max()),
      _lmat(te_fes->GetVSize(), _coeff_dim),
      _rmat(tr_fes->GetVSize(), _coeff_dim) {
  CheckMesh();
  _bdr_marker = 0;
  _dom_marker = 0;
#ifndef MFEM_THREAD_SAFE
  _c.SetSize(_coeff_dim);
  _x.SetSize(_tr_fes->GetMesh()->Dimension());
#endif
}
#endif

mfem::real_t Poisson::ExternalBoundaryRadius() const {
  using namespace mfem;
  auto* mesh = _tr_fes->GetMesh();
  auto dim = mesh->Dimension();

  auto r = real_t{0};
  auto x = Vector(dim);

  for (auto i = 0; i < _te_fes->GetNBE(); i++) {
    const auto elm_attr = mesh->GetBdrAttribute(i);
    if (_bdr_marker[elm_attr - 1] == 1) {
      const auto* fe = _te_fes->GetBE(i);
      auto* Trans = _te_fes->GetBdrElementTransformation(i);
      const auto ir = fe->GetNodes();
      const IntegrationPoint& ip = ir.IntPoint(0);
      Trans->SetIntPoint(&ip);
      Trans->Transform(ip, x);
      r = x.Norml2();
      break;
    }
  }
  return r;
}

#ifdef MFEM_USE_MPI
mfem::real_t Poisson::ParallelExternalBoundaryRadius() const {
  using namespace mfem;
  auto local_radius = ExternalBoundaryRadius();
  auto radius = real_t{0};
  int rank;
  int size;
  MPI_Comm_rank(_comm, &rank);
  MPI_Comm_size(_comm, &size);

  if (rank == 0) {
    auto radii = std::vector<real_t>(size);

    MPI_Gather(&local_radius, 1, MFEM_MPI_REAL_T, radii.data(), 1,
               MFEM_MPI_REAL_T, 0, _comm);

    for (auto i = 0; i < size; i++) {
      if (radii[i] != 0) {
        radius = radii[i];
        break;
      }
    }

    for (auto i = 0; i < size; i++) {
      assert(radii[i] == 0 || std::abs(radii[i] - radius) < 1e-6 * radius);
    }

  } else {
    MPI_Gather(&local_radius, 1, MFEM_MPI_REAL_T, nullptr, 0, MFEM_MPI_REAL_T,
               0, _comm);
  }

  MPI_Bcast(&radius, 1, MFEM_MPI_REAL_T, 0, _comm);

  return radius;
}
#endif

void Poisson::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector _c(_coeff_dim);
#endif

  _rmat.MultTranspose(x, _c);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, _c.GetData(), _coeff_dim, MFEM_MPI_REAL_T,
                  MPI_SUM, _comm);
  }
#endif
  y.SetSize(_lmat.Height());
  _lmat.Mult(_c, y);
}

void Poisson::MultTranspose(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector _c(_coeff_dim);
#endif

  _lmat.MultTranspose(x, _c);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, _c.GetData(), _coeff_dim, MFEM_MPI_REAL_T,
                  MPI_SUM, _comm);
  }
#endif

  y.SetSize(_rmat.Width());
  _rmat.Mult(_c, y);
}

void Poisson::Assemble() {
  auto* mesh = _tr_fes->GetMesh();

  auto elmat = mfem::DenseMatrix();
  auto vdofs = mfem::Array<int>();
  auto cdofs = mfem::Array<int>(_coeff_dim);
  for (auto i = 0; i < _coeff_dim; i++) {
    cdofs[i] = i;
  }

  for (auto i = 0; i < _te_fes->GetNBE(); i++) {
    const auto elm_attr = mesh->GetBdrAttribute(i);
    if (_bdr_marker[elm_attr - 1] == 1) {
      _te_fes->GetBdrElementVDofs(i, vdofs);
      const auto* fe = _te_fes->GetBE(i);
      auto* Trans = _te_fes->GetBdrElementTransformation(i);

      AssembleLeftElementMatrix(*fe, *Trans, elmat);

      _lmat.AddSubMatrix(vdofs, cdofs, elmat);
    }
  }

  _lmat.Finalize();

  for (auto i = 0; i < _tr_fes->GetNE(); i++) {
    const auto elm_attr = mesh->GetAttribute(i);
    if (_dom_marker[elm_attr - 1] == 1) {
      _tr_fes->GetElementVDofs(i, vdofs);
      const auto* fe = _tr_fes->GetFE(i);
      auto* Trans = _tr_fes->GetElementTransformation(i);

      AssembleRightElementMatrix(*fe, *Trans, elmat);

      _rmat.AddSubMatrix(vdofs, cdofs, elmat);
    }
  }

  _rmat.Finalize();
}

mfem::RAPOperator Poisson::RAPOperator() const {
  auto* P_te = _te_fes->GetProlongationMatrix();
  auto* P_tr = _tr_fes->GetProlongationMatrix();
  return mfem::RAPOperator(*P_te, *this, *P_tr);
}

void PoissonCircle::AssembleRightElementMatrix(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x;
  _c.SetSize(_coeff_dim);
  _x.SetSize(2);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, _coeff_dim);
  elmat = 0.0;

  const auto* ir = GetIntegrationRule(fe, Trans);
  if (ir == nullptr) {
    int intorder = fe.GetOrder() + Trans.OrderW();
    ir = &IntRules.Get(fe.GetGeomType(), intorder);
  }

  const auto fac = 1 / (2 * pi * _bdr_radius);

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);

    fe.CalcShape(ip, shape);

    auto radius = _x.Norml2();
    auto inverse_radius = radius > 0 ? 1 / radius : real_t{0};

    auto sin = _x[1] * inverse_radius;
    auto cos = _x[0] * inverse_radius;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    _c(0) = 1.0;

    auto ratio = radius / _bdr_radius;
    auto rfac = real_t{1.0};

    auto i = 1;
    for (auto k = 1; k <= _kMax; k++) {
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      rfac *= ratio;
      _c(i++) = rfac * cos_k;
      _c(i++) = rfac * sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    auto w = fac * Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, _c, elmat);
  }
}

void PoissonCircle::AssembleLeftElementMatrix(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x;
  _c.SetSize(_coeff_dim);
  _x.SetSize(2);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, _coeff_dim);
  elmat = 0.0;

  const auto* ir = GetIntegrationRule(fe, Trans);
  if (ir == nullptr) {
    int intorder = fe.GetOrder() + Trans.OrderW();
    ir = &IntRules.Get(fe.GetGeomType(), intorder);
  }

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);

    fe.CalcShape(ip, shape);

    auto inverse_radius = 1 / _bdr_radius;

    auto sin = _x[1] * inverse_radius;
    auto cos = _x[0] * inverse_radius;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    _c(0) = 1.;

    auto i = 1;
    for (auto k = 1; k <= _kMax; k++) {
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      _c(i++) = cos_k;
      _c(i++) = sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    auto w = Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, _c, elmat);
  }
}

}  // namespace Multipole
}  // namespace mfemElasticity