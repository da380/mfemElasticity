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

mfem::Vector Poisson3D::_sqrt;
mfem::Vector Poisson3D::_isqrt;

void Poisson3D::SetSquareRoots(int lMax) {
  _sqrt.SetSize(2 * lMax + 2);
  _isqrt.SetSize(2 * lMax + 2);
  for (auto l = 0; l <= 2 * lMax + 1; l++) {
    _sqrt(l) = std::sqrt(static_cast<mfem::real_t>(l));
  }
  for (auto l = 1; l <= 2 * lMax + 1; l++) {
    _isqrt(l) = 1 / _sqrt(l);
  }
}

mfem::real_t Poisson3D::LogFactorial(int m) const {
  return std::lgamma(static_cast<mfem::real_t>(m + 1));
}

mfem::real_t Poisson3D::LogDoubleFactorial(int m) const {
  return -logSqrtPi + m * log2 +
         std::lgamma(static_cast<mfem::real_t>(m + 0.5));
}

mfem::real_t Poisson3D::Pll(int l, mfem::real_t x) const {
  using namespace mfem;
  if (l == 0) return invSqrtFourPi;
  auto sin2 = 1 - x * x;
  if (std::abs(sin2) < std::numeric_limits<real_t>::min()) return 0;
  auto logValue =
      0.5 * (std::log(static_cast<real_t>(2 * l + 1)) - LogFactorial(2 * l)) +
      LogDoubleFactorial(l) + 0.5 * l * std::log(sin2);
  return MinusOnePower(l) * invSqrtFourPi * std::exp(logValue);
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
  SetSquareRoots(_lMax);
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
  SetSquareRoots(_lMax);
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
  SetSquareRoots(_lMax);
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
  SetSquareRoots(_lMax);
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
  Vector _c, shape, _x, _sin, _cos, _p, _pm1;
#endif

  _x.SetSize(3);
  _c.SetSize(_coeff_dim);

  _sin.SetSize(_lMax + 1);
  _cos.SetSize(_lMax + 1);

  _p.SetSize(_lMax + 1);
  _pm1.SetSize(_lMax + 1);

  shape.SetSize(dof);

  elmat.SetSize(_coeff_dim, dof);
  elmat = 0.0;

  const auto* ir = GetIntegrationRule(fe, Trans);
  if (ir == nullptr) {
    int intorder = fe.GetOrder() + Trans.OrderW();
    ir = &IntRules.Get(fe.GetGeomType(), intorder);
  }

  _sin(0) = 0.0;
  _cos(0) = 1.0;

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);

    const auto r = _x.Norml2();
    const auto ri = 1 / r;
    const auto cos_theta = _x(2) * ri;
    const auto rxy = std::sqrt(_x(0) * _x(0) + _x(1) * _x(1));
    const auto cos = rxy > 0 ? _x(0) / rxy : real_t{1};
    const auto sin = rxy > 0 ? _x(1) / rxy : real_t{0};

    _pm1(0) = 0.0;
    _p(0) = Pll(0, cos_theta);

    auto rfac = std::sqrt(ri) * ri;
    _c(0) = rfac * _p(0);

    auto i = 1;
    for (auto l = 1; l <= _lMax; l++) {
      auto fac = rfac * _sqrt(l + 1);

      _sin(l) = _sin(l - 1) * cos + _cos(l - 1) * sin;
      _cos(l) = _cos(l - 1) * cos - _sin(l - 1) * sin;

      for (auto m = 0; m < l; m++) {
        const auto alpha =
            _sqrt(2 * l + 1) * _sqrt(2 * l - 1) * _isqrt[l + m] * _isqrt[l - m];
        const auto beta = _sqrt(l - 1 + m) * _sqrt(l - 1 - m) *
                          _isqrt(2 * (l - 1) + 1) * _isqrt(2 * (l - 1) - 1);
        _pm1(m) = alpha * (cos_theta * _p(m) - beta * _pm1(m));
      }
      _pm1(l) = Pll(l, cos_theta);
      std::swap(_p, _pm1);

      _c(i++) = fac * _p(0);

      fac *= sqrt2;
      for (auto m = 1; m <= l; m++) {
        _c(i++) = fac * _p(m) * _cos(m);
        _c(i++) = fac * _p(m) * _sin(m);
      }
    }

    fe.CalcShape(ip, shape);
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

mfem::RAPOperator Poisson3D::RAP() const {
  auto* P = _fes->GetProlongationMatrix();
  return mfem::RAPOperator(*P, *this, *P);
}

}  // namespace DtN
   //
}  // namespace mfemElasticity