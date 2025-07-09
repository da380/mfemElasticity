// #include "mfemElasticity/DtN.hpp"

#include "mfemElasticity/DtN.hpp"

namespace mfemElasticity {

namespace DtN {

Poisson2D::Poisson2D(mfem::FiniteElementSpace* fes, int kMax,
                     mfem::Array<int>& bdr_marker)
    : mfem::Operator(fes->GetVSize()),
      _fes{fes},
      _kMax{kMax},
      _coeff_dim{2 * kMax},
      _bdr_marker{bdr_marker},
      _mat(_coeff_dim, fes->GetVSize()) {
  assert(_kMax > 0);
  auto* mesh = _fes->GetMesh();
  assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);
}

Poisson2D::Poisson2D(mfem::FiniteElementSpace* fes, int kMax)
    : mfem::Operator(fes->GetVSize()),
      _fes{fes},
      _kMax{kMax},
      _coeff_dim{2 * kMax},
      _mat(_coeff_dim, fes->GetVSize()) {
  assert(_kMax > 0);
  auto* mesh = _fes->GetMesh();
  assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);
  _bdr_marker.SetSize(mesh->bdr_attributes.Max());
  _bdr_marker = 0;
  mesh->MarkExternalBoundaries(_bdr_marker);
}

#ifdef MFEM_USE_MPI
Poisson2D::Poisson2D(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kMax,
                     mfem::Array<int>& bdr_marker)
    : mfem::Operator(fes->GetVSize()),
      _comm{comm},
      _pfes{fes},
      _fes{fes},
      _kMax{kMax},
      _coeff_dim{2 * kMax},
      _bdr_marker{bdr_marker},
      _mat(_coeff_dim, fes->GetVSize()),
      _parallel{true} {
  assert(_kMax > 0);
  auto* mesh = _fes->GetMesh();
  assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);
}

Poisson2D::Poisson2D(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kMax)
    : mfem::Operator(fes->GetVSize()),
      _comm{comm},
      _pfes{fes},
      _fes{fes},
      _kMax{kMax},
      _coeff_dim{2 * kMax},
      _mat(_coeff_dim, fes->GetVSize()),
      _parallel{true} {
  assert(_kMax > 0);
  auto* mesh = _fes->GetMesh();
  assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);
  _bdr_marker.SetSize(mesh->bdr_attributes.Max());
  _bdr_marker = 0;
  mesh->MarkExternalBoundaries(_bdr_marker);
}
#endif

void Poisson2D::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector _c;
#endif

  _c.SetSize(_coeff_dim);
  _mat.Mult(x, _c);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, _c.GetData(), _coeff_dim, MFEM_MPI_REAL_T,
                  MPI_SUM, _comm);
  }
#endif

  y.SetSize(x.Size());
  _mat.MultTranspose(_c, y);
}

void Poisson2D::AssembleElementMatrix(const mfem::FiniteElement& fe,
                                      mfem::ElementTransformation& Trans,
                                      mfem::DenseMatrix& elmat) {
  using namespace mfem;
  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x;
#endif
  _x.SetSize(2);
  _c.SetSize(_coeff_dim);
  shape.SetSize(dof);
  elmat.SetSize(_coeff_dim, dof);
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

    auto ri = 1 / _x.Norml2();
    auto sin = _x[1] * ri;
    auto cos = _x[0] * ri;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    auto i = 0;
    for (auto k = 1; k <= _kMax; k++) {
      auto fac = std::sqrt(M_PI * k);
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      _c(i++) = fac * cos_k;
      _c(i++) = fac * sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    auto w = ri * Trans.Weight() * ip.weight / M_PI;

    AddMult_a_VWt(w, _c, shape, elmat);
  }
}

void Poisson2D::Assemble() {
  using namespace mfem;
  auto* mesh = _fes->GetMesh();

  auto elmat = DenseMatrix();
  auto vdofs = Array<int>();
  auto rows = Array<int>(_coeff_dim);
  for (auto i = 0; i < _coeff_dim; i++) {
    rows[i] = i;
  }

  for (auto i = 0; i < _fes->GetNBE(); i++) {
    const auto elm_attr = mesh->GetBdrAttribute(i);
    if (_bdr_marker[elm_attr - 1] == 1) {
      _fes->GetBdrElementVDofs(i, vdofs);
      const auto* fe = _fes->GetBE(i);
      auto* Trans = _fes->GetBdrElementTransformation(i);

      AssembleElementMatrix(*fe, *Trans, elmat);

      _mat.AddSubMatrix(rows, vdofs, elmat);
    }
  }

  _mat.Finalize();
}

mfem::RAPOperator Poisson2D::RAP() const {
  auto* P = _fes->GetProlongationMatrix();
  return mfem::RAPOperator(*P, *this, *P);
}

Poisson3D::Poisson3D(mfem::FiniteElementSpace* fes, int lMax,
                     mfem::Array<int>& bdr_marker)
    : mfem::Operator(fes->GetVSize()),
      _fes{fes},
      _lMax{lMax},
      _coeff_dim{(lMax + 1) * (lMax + 1)},
      _bdr_marker{bdr_marker},
      _mat((lMax + 1) * (lMax + 1), fes->GetVSize()) {
  assert(_lMax >= 0);
  auto* mesh = _fes->GetMesh();
  assert(mesh->Dimension() == 3 && mesh->SpaceDimension() == 3);
}

Poisson3D::Poisson3D(mfem::FiniteElementSpace* fes, int lMax)
    : mfem::Operator(fes->GetVSize()),
      _fes{fes},
      _lMax{lMax},
      _coeff_dim{(lMax + 1) * (lMax + 1)},
      _mat((lMax + 1) * (lMax + 1), fes->GetVSize()) {
  assert(_lMax >= 0);
  auto* mesh = _fes->GetMesh();
  assert(mesh->Dimension() == 3 && mesh->SpaceDimension() == 3);

  _bdr_marker.SetSize(mesh->bdr_attributes.Max());
  _bdr_marker = 0;
  mesh->MarkExternalBoundaries(_bdr_marker);
}

#ifdef MFEM_USE_MPI
Poisson3D::Poisson3D(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int lMax,
                     mfem::Array<int>& bdr_marker)
    : mfem::Operator(fes->GetVSize()),
      _comm{comm},
      _pfes{fes},
      _fes{fes},
      _lMax{lMax},
      _coeff_dim{(lMax + 1) * (lMax + 1)},
      _bdr_marker{bdr_marker},
      _mat((lMax + 1) * (lMax + 1), fes->GetVSize()),
      _parallel{true} {
  assert(_lMax >= 0);
  auto* mesh = _fes->GetMesh();
  assert(mesh->Dimension() == 3 && mesh->SpaceDimension() == 3);
}

Poisson3D::Poisson3D(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int lMax)
    : mfem::Operator(fes->GetVSize()),
      _comm{comm},
      _pfes{fes},
      _fes{fes},
      _lMax{lMax},
      _coeff_dim{(lMax + 1) * (lMax + 1)},
      _mat((lMax + 1) * (lMax + 1), fes->GetVSize()),
      _parallel{true} {
  assert(_lMax >= 0);
  auto* mesh = _fes->GetMesh();
  assert(mesh->Dimension() == 3 && mesh->SpaceDimension() == 3);

  _bdr_marker.SetSize(mesh->bdr_attributes.Max());
  _bdr_marker = 0;
  mesh->MarkExternalBoundaries(_bdr_marker);
}
#endif

void Poisson3D::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector _c;
#endif

  _c.SetSize(_coeff_dim);
  _mat.Mult(x, _c);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, _c.GetData(), _coeff_dim, MFEM_MPI_REAL_T,
                  MPI_SUM, _comm);
  }
#endif

  y.SetSize(x.Size());
  _mat.MultTranspose(_c, y);
}

void Poisson3D::AssembleElementMatrix(const mfem::FiniteElement& fe,
                                      mfem::ElementTransformation& Trans,
                                      mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x, sines, cosines, _, _p_old;
#endif
  _x.SetSize(3);
  _c.SetSize(_coeff_dim);
  sines.SetSize(_lMax + 1);
  cosines.SetSize(_lMax + 1);
  _p.SetSize(_lMax + 1);
  _p_old.SetSize(_lMax);
  shape.SetSize(dof);
  elmat.SetSize(_coeff_dim, dof);
  elmat = 0.0;

  const auto* ir = GetIntegrationRule(fe, Trans);
  if (ir == nullptr) {
    int intorder = fe.GetOrder() + Trans.OrderW();
    ir = &IntRules.Get(fe.GetGeomType(), intorder);
  }

  sines(0) = 0.0;
  cosines(0) = 1.0;

  _p_old(0) = 0.0;
  _p(0) = 1.0;

  constexpr auto sqrt2 = std::sqrt(static_cast<real_t>(2));

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);

    fe.CalcShape(ip, shape);

    const auto r = _x.Norml2();

    const auto ri = 1 / r;
    const auto cos_theta = _x(2) * ri;

    const auto rxy = std::sqrt(_x(0) * _x(0) + _x(1) * _x(1));
    real_t cos, sin;
    if (rxy > 0) {
      cos = _x(0) / rxy;
      sin = _x(1) / rxy;
    } else {
      cos = 1;
      sin = 0;
    }

    auto rfac = std::sqrt(ri) * ri;
    _c(0) = rfac * _p(0);

    auto i = 1;
    for (auto l = 1; l <= _lMax; l++) {
      auto fac = rfac * std::sqrt(static_cast<real_t>(l + 1));

      sines(l) = sines(l - 1) * cos + cosines(l - 1) * sin;
      cosines(l) = cosines(l - 1) * cos - sines(l - 1) * sin;

      _c(i++) = fac * _p(0);

      fac *= sqrt2;
      for (auto m = 1; m <= l; m++) {
        _c(i++) = fac * _p(m) * cosines(m);
        _c(i++) = fac * _p(m) * sines(m);
      }
    }

    auto w = Trans.Weight() * ip.weight;

    AddMult_a_VWt(w, _c, shape, elmat);
  }
}

void Poisson3D::Assemble() {
  using namespace mfem;
  auto* mesh = _fes->GetMesh();

  auto elmat = DenseMatrix();
  auto vdofs = Array<int>();
  auto rows = Array<int>(_coeff_dim);
  for (auto i = 0; i < _coeff_dim; i++) {
    rows[i] = i;
  }

  for (auto i = 0; i < _fes->GetNBE(); i++) {
    const auto elm_attr = mesh->GetBdrAttribute(i);
    if (_bdr_marker[elm_attr - 1] == 1) {
      _fes->GetBdrElementVDofs(i, vdofs);
      const auto* fe = _fes->GetBE(i);
      auto* Trans = _fes->GetBdrElementTransformation(i);

      AssembleElementMatrix(*fe, *Trans, elmat);

      _mat.AddSubMatrix(rows, vdofs, elmat);
    }
  }

  _mat.Finalize();
}

}  // namespace DtN
   //
}  // namespace mfemElasticity