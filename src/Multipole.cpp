#include "mfemElasticity/Multipole.hpp"

namespace mfemElasticity {

namespace Multipole {

mfem::real_t Poisson2D::ExternalBoundaryRadius() const {
  using namespace mfem;
  auto* mesh = _tr_fes->GetMesh();

  auto radius = real_t{0};
  auto x = Vector(2);

  for (auto i = 0; i < _te_fes->GetNBE(); i++) {
    const auto elm_attr = mesh->GetBdrAttribute(i);
    if (_bdr_marker[elm_attr - 1] == 1) {
      const auto* fe = _te_fes->GetBE(i);
      auto* Trans = _te_fes->GetBdrElementTransformation(i);
      const auto ir = fe->GetNodes();
      const IntegrationPoint& ip = ir.IntPoint(0);
      Trans->SetIntPoint(&ip);
      Trans->Transform(ip, x);
      radius = x.Norml2();
      break;
    }
  }
  return radius;
}

#ifdef MFEM_USE_MPI

// Returns the radius of the external boundary for parallel
// calculations.
mfem::real_t Poisson2D::ParallelExternalBoundaryRadius() const {
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

void Poisson2D::AssembleRightElementMatrix(const mfem::FiniteElement& fe,
                                           mfem::ElementTransformation& Trans,
                                           mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();
  auto n = 2 * _kmax + 1;

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x;
#endif
  _x.SetSize(2);
  _c.SetSize(n);
  shape.SetSize(dof);
  elmat.SetSize(dof, n);
  elmat = 0.0;

  const auto* ir = GetIntegrationRule(fe, Trans);
  if (ir == nullptr) {
    int intorder = fe.GetOrder() + Trans.OrderW();
    ir = &IntRules.Get(fe.GetGeomType(), intorder);
  }

  const auto fac = 1 / (2 * M_PI * _bdr_radius);

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
    for (auto k = 1; k <= _kmax; k++) {
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      rfac *= ratio;
      _c(i) = rfac * cos_k;
      _c(i + 1) = rfac * sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
      i += 2;
    }

    auto w = fac * Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, _c, elmat);
  }
}

void Poisson2D::AssembleLeftElementMatrix(const mfem::FiniteElement& fe,
                                          mfem::ElementTransformation& Trans,
                                          mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();
  auto n = 2 * _kmax + 1;

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x;
#endif
  _x.SetSize(2);
  _c.SetSize(n);
  shape.SetSize(dof);
  elmat.SetSize(dof, n);
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

    //_bdr_radius = _x.Norml2();
    auto inverse_radius = 1 / _bdr_radius;

    auto sin = _x[1] * inverse_radius;
    auto cos = _x[0] * inverse_radius;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    _c(0) = 1.;

    auto i = 1;
    for (auto k = 1; k <= _kmax; k++) {
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      _c(i) = cos_k;
      _c(i + 1) = sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
      i += 2;
    }

    auto w = Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, _c, elmat);
  }
}

Poisson2D::Poisson2D(mfem::FiniteElementSpace* tr_fes,
                     mfem::FiniteElementSpace* te_fes, int kmax, int dom_marker,
                     int bdr_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _kmax{kmax},
      _dom_marker{dom_marker},
      _bdr_marker{bdr_marker},
      _lmat(te_fes->GetVSize(), 2 * _kmax + 1),
      _rmat(tr_fes->GetVSize(), 2 * _kmax + 1) {
  assert(_kmax >= 0);
  assert(_tr_fes->GetMesh() == _te_fes->GetMesh());
  auto* mesh = _tr_fes->GetMesh();
  assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);

  _bdr_radius = ExternalBoundaryRadius();
  assert(_bdr_radius > 0);
}

Poisson2D::Poisson2D(mfem::FiniteElementSpace* tr_fes,
                     mfem::FiniteElementSpace* te_fes, int kmax)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _kmax{kmax},
      _lmat(te_fes->GetVSize(), 2 * _kmax + 1),
      _rmat(tr_fes->GetVSize(), 2 * _kmax + 1) {
  assert(_kmax >= 0);
  assert(_tr_fes->GetMesh() == _te_fes->GetMesh());
  auto* mesh = _tr_fes->GetMesh();
  assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);

  _dom_marker.SetSize(mesh->attributes.Max());
  _dom_marker = 1;

  _bdr_marker.SetSize(mesh->bdr_attributes.Max());
  _bdr_marker = 0;
  mesh->MarkExternalBoundaries(_bdr_marker);

  _bdr_radius = ExternalBoundaryRadius();
  assert(_bdr_radius > 0);
}

#ifdef MFEM_USE_MPI

Poisson2D::Poisson2D(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                     mfem::ParFiniteElementSpace* te_fes, int kmax,
                     int dom_marker, int bdr_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _comm{comm},
      _tr_pfes{tr_fes},
      _te_pfes{te_fes},
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _kmax{kmax},
      _dom_marker{dom_marker},
      _bdr_marker{bdr_marker},
      _lmat(te_fes->GetVSize(), 2 * _kmax + 1),
      _rmat(tr_fes->GetVSize(), 2 * _kmax + 1),
      _parallel{true} {
  assert(_kmax >= 0);
  assert(_tr_fes->GetMesh() == _te_fes->GetMesh());
  auto* mesh = _tr_fes->GetMesh();
  assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);

  _bdr_radius = ParallelExternalBoundaryRadius();
  assert(_bdr_radius > 0);
}

Poisson2D::Poisson2D(MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
                     mfem::ParFiniteElementSpace* te_fes, int kmax)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _comm{comm},
      _tr_pfes{tr_fes},
      _te_pfes{te_fes},
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _kmax{kmax},
      _lmat(te_fes->GetVSize(), 2 * _kmax + 1),
      _rmat(tr_fes->GetVSize(), 2 * _kmax + 1),
      _parallel{true} {
  assert(_kmax >= 0);
  assert(_tr_pfes->GetMesh() == _te_pfes->GetMesh());
  auto* mesh = _te_pfes->GetMesh();
  assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);

  _dom_marker.SetSize(mesh->attributes.Max());
  _dom_marker = 1;

  _bdr_marker.SetSize(mesh->bdr_attributes.Max());
  _bdr_marker = 0;
  mesh->MarkExternalBoundaries(_bdr_marker);

  _bdr_radius = ParallelExternalBoundaryRadius();
  // assert(_bdr_radius > 0);
}

#endif

void Poisson2D::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector _c;
#endif
  _c.SetSize(2 * _kmax + 1);
  _rmat.MultTranspose(x, _c);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, _c.GetData(), 2 * _kmax + 1, MFEM_MPI_REAL_T,
                  MPI_SUM, _comm);
  }
#endif
  y.SetSize(_lmat.Height());
  _lmat.Mult(_c, y);
}

void Poisson2D::MultTranspose(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector _c;
#endif

  _c.SetSize(2 * _kmax + 1);
  _lmat.MultTranspose(x, _c);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, _c.GetData(), 2 * _kmax, MFEM_MPI_REAL_T,
                  MPI_SUM, _comm);
  }
#endif

  y.SetSize(_rmat.Width());
  _rmat.Mult(_c, y);
}

void Poisson2D::Assemble() {
  auto* mesh = _tr_fes->GetMesh();

  auto elmat = mfem::DenseMatrix();
  auto vdofs = mfem::Array<int>();
  auto cdofs = mfem::Array<int>(2 * _kmax + 1);
  for (auto i = 0; i < 2 * _kmax + 1; i++) {
    cdofs[i] = i;
  }

  // Assemble the left matrix by looping over the boundary.
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

  // Assemble the right matrix by looping over the domain.
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

mfem::RAPOperator Poisson2D::RAP() const {
  auto* P_te = _te_pfes->GetProlongationMatrix();
  auto* P_tr = _tr_pfes->GetProlongationMatrix();
  return mfem::RAPOperator(*P_te, *this, *P_tr);
}

}  // namespace Multipole
}  // namespace mfemElasticity