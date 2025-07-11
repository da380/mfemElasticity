// #include "mfemElasticity/DtN.hpp"

#include "mfemElasticity/DtN.hpp"

#include <cmath>

namespace mfemElasticity {

namespace DtN {

Poisson::Poisson(mfem::FiniteElementSpace* fes, int coeff_dim)
    : mfem::Operator(fes->GetVSize()),
      _fes{fes},
      _coeff_dim{coeff_dim},
      _bdr_marker(_fes->GetMesh()->bdr_attributes.Max()),
      _mat(_coeff_dim, fes->GetVSize()) {
  CheckMesh();
  _bdr_marker = 0;
#ifndef MFEM_THREAD_SAFE
  _c.SetSize(_coeff_dim);
  _x.SetSize(_fes->GetMesh()->Dimension());
#endif
}

#ifdef MFEM_USE_MPI
Poisson::Poisson(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int coeff_dim)
    : mfem::Operator(fes->GetVSize()),
      _parallel{true},
      _comm{comm},
      _pfes{fes},
      _fes{fes},
      _coeff_dim{coeff_dim},
      _bdr_marker(_fes->GetMesh()->bdr_attributes.Max()),
      _mat(_coeff_dim, fes->GetVSize()) {
  CheckMesh();
  _bdr_marker = 0;
#ifndef MFEM_THREAD_SAFE
  _c.SetSize(_coeff_dim);
  _x.SetSize(_fes->GetMesh()->Dimension());
#endif
}
#endif

void Poisson::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector _c(_coeff_dim);
#endif

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

void Poisson::Assemble() {
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

mfem::RAPOperator Poisson::RAPOperator() const {
  auto* P = _fes->GetProlongationMatrix();
  return mfem::RAPOperator(*P, *this, *P);
}

void PoissonCircle::AssembleElementMatrix(const mfem::FiniteElement& fe,
                                          mfem::ElementTransformation& Trans,
                                          mfem::DenseMatrix& elmat) {
  using namespace mfem;
  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x;
  _x.SetSize(2);
  _c.SetSize(_coeff_dim);
#endif

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
      auto fac = std::sqrt(pi * k);
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      _c(i++) = fac * cos_k;
      _c(i++) = fac * sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    auto w = ri * Trans.Weight() * ip.weight / pi;

    AddMult_a_VWt(w, _c, shape, elmat);
  }
}

void PoissonSphere::AssembleElementMatrix(const mfem::FiniteElement& fe,
                                          mfem::ElementTransformation& Trans,
                                          mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x, _sin, _cos, _p, _pm1;
  _x.SetSize(3);
  _c.SetSize(_coeff_dim);

  _sin.SetSize(_lMax + 1);
  _cos.SetSize(_lMax + 1);

  _p.SetSize(_lMax + 1);
  _pm1.SetSize(_lMax + 1);
#endif

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
      auto fac = rfac * _sqrt[l + 1];

      _sin(l) = rxy > 0 ? _sin(l - 1) * cos + _cos(l - 1) * sin : 0.0;
      _cos(l) = _cos(l - 1) * cos - _sin(l - 1) * sin;

      for (auto m = 0; m < l; m++) {
        const auto [alpha, beta] = RecursionCoefficients(l, m);
        _pm1(m) = alpha * (cos_theta * _p(m) - beta * _pm1(m));
      }
      _pm1(l) = Pll(l, cos_theta);
      _p(l) = 0.0;
      std::swap(_p, _pm1);

      _c(i++) = fac * _p(0);

      fac *= _sqrt[2];
      for (auto m = 1; m <= l; m++) {
        _c(i++) = fac * _p[m] * _cos(m);
        _c(i++) = rxy > 0 ? fac * _p[m] * _sin(m) : 0.0;
      }
    }

    fe.CalcShape(ip, shape);
    auto w = Trans.Weight() * ip.weight;

    AddMult_a_VWt(w, _c, shape, elmat);
  }
}

}  // namespace DtN
   //
}  // namespace mfemElasticity