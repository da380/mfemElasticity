#include "mfemElasticity/poisson.hpp"

namespace mfemElasticity {

/*****************************************************************
******************************************************************
******************************************************************
*****************************************************************/

int PoissonDtNOperator::CoeffDim() const {
  return _fes->GetMesh()->Dimension() == 2 ? 2 * _degree
                                           : (_degree + 1) * (_degree + 1);
}

void PoissonDtNOperator::SetUp() {
  assert(_dim == 2 || _dim == 3);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    SetBoundaryMarker(_pfes->GetParMesh());
  } else {
    SetBoundaryMarker(_fes->GetMesh());
  }
#else
  SetBoundaryMarker(_fes->GetMesh());
#endif

#ifndef MFEM_THREAD_SAFE
  _c.SetSize(_coeff_dim);
  _x.SetSize(_dim);
  if (_dim == 3) {
    _sin.SetSize(_degree + 1);
    _cos.SetSize(_degree + 1);
    _p.SetSize(_degree + 1);
    _pm1.SetSize(_degree + 1);
  }
#endif
  SetSquareRoots(_dim, _degree);
}

PoissonDtNOperator::PoissonDtNOperator(mfem::FiniteElementSpace* fes,
                                       int degree)
    : mfem::Operator(fes->GetVSize()),
      _fes{fes},
      _dim{fes->GetMesh()->Dimension()},
      _degree{degree},
      _coeff_dim{CoeffDim()},
      _mat(fes->GetVSize(), _coeff_dim) {
  SetUp();
  Assemble();
}

#ifdef MFEM_USE_MPI
PoissonDtNOperator::PoissonDtNOperator(MPI_Comm comm,
                                       mfem::ParFiniteElementSpace* fes,
                                       int degree)
    : mfem::Operator(fes->GetVSize()),
      _parallel{true},
      _comm{comm},
      _pfes{fes},
      _fes{fes},
      _dim{fes->GetMesh()->Dimension()},
      _degree{degree},
      _coeff_dim{CoeffDim()},
      _mat(fes->GetVSize(), _coeff_dim) {
  SetUp();
  Assemble();
}
#endif

void PoissonDtNOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector _c(_coeff_dim);
#endif

  _mat.MultTranspose(x, _c);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, _c.GetData(), _coeff_dim, MFEM_MPI_REAL_T,
                  MPI_SUM, _comm);
  }
#endif

  y.SetSize(x.Size());
  _mat.Mult(_c, y);
}

void PoissonDtNOperator::Assemble() {
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

      if (_dim == 2) {
        AssembleElementMatrix2D(*fe, *Trans, elmat);
      } else {
        AssembleElementMatrix3D(*fe, *Trans, elmat);
      }

      _mat.AddSubMatrix(vdofs, rows, elmat);
    }
  }

  _mat.Finalize();
}

#ifdef MFEM_USE_MPI
mfem::RAPOperator PoissonDtNOperator::RAP() const {
  auto* P = _fes->GetProlongationMatrix();
  return mfem::RAPOperator(*P, *this, *P);
}
#endif

void PoissonDtNOperator::AssembleElementMatrix2D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;
  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x;
  _x.SetSize(2);
  _c.SetSize(_coeff_dim);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, _coeff_dim);
  elmat = 0.0;

  int intorder = fe.GetOrder() + Trans.OrderW();
  auto ir = &IntRules.Get(fe.GetGeomType(), intorder);

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);
    _x -= _x0;

    fe.CalcShape(ip, shape);

    const auto ri = 1 / _x.Norml2();
    const auto sin = _x[1] * ri;
    const auto cos = _x[0] * ri;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    auto i = 0;
    for (auto k = 1; k <= _degree; k++) {
      const auto fac = sqrtPi * _sqrt(k);
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      _c(i++) = fac * cos_k;
      _c(i++) = fac * sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    auto w = ri * Trans.Weight() * ip.weight / pi;
    AddMult_a_VWt(w, shape, _c, elmat);
  }
}

void PoissonDtNOperator::AssembleElementMatrix3D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x, _sin, _cos, _p, _pm1;
  _x.SetSize(3);
  _c.SetSize(_coeff_dim);
  _sin.SetSize(_degree + 1);
  _cos.SetSize(_degree + 1);
  _p.SetSize(_degree + 1);
  _pm1.SetSize(_degree + 1);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, _coeff_dim);
  elmat = 0.0;

  int intorder = fe.GetOrder() + Trans.OrderW();
  auto ir = &IntRules.Get(fe.GetGeomType(), intorder);

  _sin(0) = 0.0;
  _cos(0) = 1.0;

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);
    _x -= _x0;

    const auto ri = 1 / _x.Norml2();
    const auto cos_theta = _x(2) * ri;
    const auto rxy = std::sqrt(_x(0) * _x(0) + _x(1) * _x(1));
    const auto cos = rxy > 0 ? _x(0) / rxy : real_t{1};
    const auto sin = rxy > 0 ? _x(1) / rxy : real_t{0};

    _pm1(0) = 0.0;
    _p(0) = Pll(0, cos_theta);

    auto rfac = std::sqrt(ri) * ri;
    _c(0) = rfac * _p(0);

    auto i = 1;
    for (auto l = 1; l <= _degree; l++) {
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

    AddMult_a_VWt(w, shape, _c, elmat);
  }
}

/*****************************************************************
******************************************************************
******************************************************************
*****************************************************************/

int PoissonMultipoleOperator::CoeffDim() const {
  auto dim = _tr_fes->GetMesh()->Dimension();
  auto vDim = _tr_fes->GetVDim();
  return dim == 2 ? 2 * _degree + 1 : (_degree + 1) * (_degree + 1);
}

void PoissonMultipoleOperator::SetUp() {
  assert(_tr_fes->GetMesh() == _te_fes->GetMesh());
#ifdef MFEM_USE_MPI
  if (_parallel) {
    SetBoundaryMarker(_tr_pfes->GetParMesh());
  } else {
    SetBoundaryMarker(_tr_fes->GetMesh());
  }
#else
  SetBoundaryMarker(_tr_fes->GetMesh());
#endif

#ifndef MFEM_THREAD_SAFE
  _c.SetSize(_coeff_dim);
  _x.SetSize(_dim);
  if (_dim == 3) {
    _sin.SetSize(_degree + 1);
    _cos.SetSize(_degree + 1);
    _p.SetSize(_degree + 1);
    _pm1.SetSize(_degree + 1);
  }
#endif
  SetSquareRoots(_dim, _degree);
}

PoissonMultipoleOperator::PoissonMultipoleOperator(
    mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
    int degree, const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _dim{tr_fes->GetMesh()->Dimension()},
      _degree{degree},
      _coeff_dim{CoeffDim()},
      _dom_marker{dom_marker},
      _lmat(te_fes->GetVSize(), _coeff_dim),
      _rmat(tr_fes->GetVSize(), _coeff_dim) {
  SetUp();
  Assemble();
}

#ifdef MFEM_USE_MPI
PoissonMultipoleOperator::PoissonMultipoleOperator(
    MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
    mfem::ParFiniteElementSpace* te_fes, int degree,
    const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _parallel{true},
      _comm{comm},
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _tr_pfes{tr_fes},
      _te_pfes{te_fes},
      _dim{tr_fes->GetMesh()->Dimension()},
      _degree{degree},
      _coeff_dim{CoeffDim()},
      _dom_marker{dom_marker},
      _lmat(te_fes->GetVSize(), _coeff_dim),
      _rmat(tr_fes->GetVSize(), _coeff_dim) {
  SetUp();
  Assemble();
}
#endif

void PoissonMultipoleOperator::Mult(const mfem::Vector& x,
                                    mfem::Vector& y) const {
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

void PoissonMultipoleOperator::MultTranspose(const mfem::Vector& x,
                                             mfem::Vector& y) const {
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

void PoissonMultipoleOperator::Assemble() {
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

      if (_dim == 2) {
        AssembleLeftElementMatrix2D(*fe, *Trans, elmat);
      } else {
        AssembleLeftElementMatrix3D(*fe, *Trans, elmat);
      }

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

      if (_dim == 2) {
        AssembleRightElementMatrix2D(*fe, *Trans, elmat);
      } else {
        AssembleRightElementMatrix3D(*fe, *Trans, elmat);
      }

      _rmat.AddSubMatrix(vdofs, cdofs, elmat);
    }
  }

  _rmat.Finalize();
}

#ifdef MFEM_USE_MPI
mfem::RAPOperator PoissonMultipoleOperator::RAP() const {
  auto* P_te = _te_fes->GetProlongationMatrix();
  auto* P_tr = _tr_fes->GetProlongationMatrix();
  return mfem::RAPOperator(*P_te, *this, *P_tr);
}
#endif

void PoissonMultipoleOperator::AssembleRightElementMatrix2D(
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

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  const auto fac = 1 / (2 * pi * _bdr_radius);

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);
    _x -= _x0;

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
    for (auto k = 1; k <= _degree; k++) {
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

void PoissonMultipoleOperator::AssembleLeftElementMatrix2D(
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

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);
    _x -= _x0;

    fe.CalcShape(ip, shape);

    auto inverse_radius = 1 / _bdr_radius;

    auto sin = _x[1] * inverse_radius;
    auto cos = _x[0] * inverse_radius;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    _c(0) = 1.;

    auto i = 1;
    for (auto k = 1; k <= _degree; k++) {
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

void PoissonMultipoleOperator::AssembleRightElementMatrix3D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x, _sin, _cos, _p, _pm1;
  _x.SetSize(3);
  _c.SetSize(_coeff_dim);
  _sin.SetSize(_degree + 1);
  _cos.SetSize(_degree + 1);
  _p.SetSize(_degree + 1);
  _pm1.SetSize(_degree + 1);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, _coeff_dim);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  _sin(0) = 0.0;
  _cos(0) = 1.0;

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);
    _x -= _x0;

    const auto r = _x.Norml2();
    const auto ri = 1 / r;
    const auto cos_theta = _x(2) * ri;
    const auto rxy = std::sqrt(_x(0) * _x(0) + _x(1) * _x(1));
    const auto cos = rxy > 0 ? _x(0) / rxy : real_t{1};
    const auto sin = rxy > 0 ? _x(1) / rxy : real_t{0};

    _pm1(0) = 0.0;
    _p(0) = Pll(0, cos_theta);

    const auto ratio = r / _bdr_radius;
    auto rfac = 1 / (_bdr_radius * _bdr_radius);
    _c(0) = rfac * _p(0);

    auto i = 1;
    for (auto l = 1; l <= _degree; l++) {
      rfac *= ratio;
      auto fac = rfac * (l + 1) / (2 * l + 1);

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
    AddMult_a_VWt(w, shape, _c, elmat);
  }
}

void PoissonMultipoleOperator::AssembleLeftElementMatrix3D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x, _sin, _cos, _p, _pm1;
  _x.SetSize(3);
  _c.SetSize(_coeff_dim);
  _sin.SetSize(_degree + 1);
  _cos.SetSize(_degree + 1);
  _p.SetSize(_degree + 1);
  _pm1.SetSize(_degree + 1);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, _coeff_dim);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  _sin(0) = 0.0;
  _cos(0) = 1.0;

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);
    _x -= _x0;

    const auto r = _x.Norml2();
    const auto ri = 1 / r;
    const auto cos_theta = _x(2) * ri;
    const auto rxy = std::sqrt(_x(0) * _x(0) + _x(1) * _x(1));
    const auto cos = rxy > 0 ? _x(0) / rxy : real_t{1};
    const auto sin = rxy > 0 ? _x(1) / rxy : real_t{0};

    _pm1(0) = 0.0;
    _p(0) = Pll(0, cos_theta);

    _c(0) = _p(0);

    auto i = 1;
    for (auto l = 1; l <= _degree; l++) {
      _sin(l) = rxy > 0 ? _sin(l - 1) * cos + _cos(l - 1) * sin : 0.0;
      _cos(l) = _cos(l - 1) * cos - _sin(l - 1) * sin;

      for (auto m = 0; m < l; m++) {
        const auto [alpha, beta] = RecursionCoefficients(l, m);
        _pm1(m) = alpha * (cos_theta * _p(m) - beta * _pm1(m));
      }
      _pm1(l) = Pll(l, cos_theta);
      _p(l) = 0.0;
      std::swap(_p, _pm1);

      _c(i++) = _p(0);

      for (auto m = 1; m <= l; m++) {
        _c(i++) = _sqrt[2] * _p[m] * _cos(m);
        _c(i++) = rxy > 0 ? _sqrt[2] * _p[m] * _sin(m) : 0.0;
      }
    }

    fe.CalcShape(ip, shape);
    auto w = Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, _c, elmat);
  }
}

/*****************************************************************
******************************************************************
******************************************************************
*****************************************************************/

int PoissonLinearisedMultipoleOperator::CoeffDim() const {
  auto dim = _tr_fes->GetMesh()->Dimension();
  auto vDim = _tr_fes->GetVDim();
  return dim == 2 ? 2 * _degree : (_degree + 1) * (_degree + 1) - 1;
}

void PoissonLinearisedMultipoleOperator::SetUp() {
  assert(_tr_fes->GetMesh() == _te_fes->GetMesh());
  assert(_tr_fes->GetMesh()->Dimension() == _tr_fes->GetVDim());
#ifdef MFEM_USE_MPI
  if (_parallel) {
    SetBoundaryMarker(_tr_pfes->GetParMesh());
  } else {
    SetBoundaryMarker(_tr_fes->GetMesh());
  }
#else
  SetBoundaryMarker(_tr_fes->GetMesh());
#endif

#ifndef MFEM_THREAD_SAFE
  _c0.SetSize(_coeff_dim);
  _c1.SetSize(_coeff_dim);
  _x.SetSize(_dim);
  if (_dim == 3) {
    _sin.SetSize(_degree + 1);
    _cos.SetSize(_degree + 1);
    _p.SetSize(_degree + 1);
    _pm1.SetSize(_degree + 1);
    _c2.SetSize(_coeff_dim);
  }
#endif
  SetSquareRoots(_dim, _degree);
}

PoissonLinearisedMultipoleOperator::PoissonLinearisedMultipoleOperator(
    mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
    int degree, const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _dim{tr_fes->GetMesh()->Dimension()},
      _degree{degree},
      _coeff_dim{CoeffDim()},
      _dom_marker{dom_marker},
      _lmat(te_fes->GetVSize(), _coeff_dim),
      _rmat(tr_fes->GetVSize(), _coeff_dim) {
  SetUp();
  Assemble();
}

#ifdef MFEM_USE_MPI
PoissonLinearisedMultipoleOperator::PoissonLinearisedMultipoleOperator(
    MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
    mfem::ParFiniteElementSpace* te_fes, int degree,
    const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _parallel{true},
      _comm{comm},
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _tr_pfes{tr_fes},
      _te_pfes{te_fes},
      _dim{tr_fes->GetMesh()->Dimension()},
      _degree{degree},
      _coeff_dim{CoeffDim()},
      _dom_marker{dom_marker},
      _lmat(te_fes->GetVSize(), _coeff_dim),
      _rmat(tr_fes->GetVSize(), _coeff_dim) {
  SetUp();
  Assemble();
}
#endif

void PoissonLinearisedMultipoleOperator::Mult(const mfem::Vector& x,
                                              mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector _c0(_coeff_dim);
#endif

  _rmat.MultTranspose(x, _c0);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, _c0.GetData(), _coeff_dim, MFEM_MPI_REAL_T,
                  MPI_SUM, _comm);
  }
#endif

  y.SetSize(_lmat.Height());
  _lmat.Mult(_c0, y);
}

void PoissonLinearisedMultipoleOperator::MultTranspose(const mfem::Vector& x,
                                                       mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector _c0(_coeff_dim);
#endif

  _lmat.MultTranspose(x, _c0);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, _c0.GetData(), _coeff_dim, MFEM_MPI_REAL_T,
                  MPI_SUM, _comm);
  }
#endif

  y.SetSize(_rmat.Width());
  _rmat.Mult(_c0, y);
}

void PoissonLinearisedMultipoleOperator::Assemble() {
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

      if (_dim == 2) {
        AssembleLeftElementMatrix2D(*fe, *Trans, elmat);
      } else {
        AssembleLeftElementMatrix3D(*fe, *Trans, elmat);
      }

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

      if (_dim == 2) {
        AssembleRightElementMatrix2D(*fe, *Trans, elmat);
      } else {
        AssembleRightElementMatrix3D(*fe, *Trans, elmat);
      }

      _rmat.AddSubMatrix(vdofs, cdofs, elmat);
    }
  }

  _rmat.Finalize();
}

#ifdef MFEM_USE_MPI
mfem::RAPOperator PoissonLinearisedMultipoleOperator::RAP() const {
  auto* P_te = _te_fes->GetProlongationMatrix();
  auto* P_tr = _tr_fes->GetProlongationMatrix();
  return mfem::RAPOperator(*P_te, *this, *P_tr);
}
#endif

void PoissonLinearisedMultipoleOperator::AssembleRightElementMatrix2D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();
  auto dim = Trans.GetSpaceDim();

#ifdef MFEM_THREAD_SAFE
  Vector _c0(_coeff_dim), _c1(_coeff_dim), shape(_coeff_dim), _x(2);
  DenseMatrix part_elmat;
#endif

  shape.SetSize(dof);
  part_elmat.SetSize(dof, _coeff_dim);

  elmat.SetSize(dim * dof, _coeff_dim);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  const auto fac = 1 / (2 * pi * _bdr_radius * _bdr_radius);

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);
    _x -= _x0;

    fe.CalcShape(ip, shape);

    auto r = _x.Norml2();
    auto ir = r > 0 ? 1 / r : real_t{0};

    auto sin = _x[1] * ir;
    auto cos = _x[0] * ir;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    auto ratio = r / _bdr_radius;
    auto rfac = real_t{1.0};

    auto i = 0;
    for (auto k = 1; k <= _degree; k++) {
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;

      _c0(i) = k * rfac * cos_k;
      _c1(i++) = -k * rfac * sin_k;

      _c0(i) = k * rfac * sin_k;
      _c1(i++) = k * rfac * cos_k;

      rfac *= ratio;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    auto w = fac * Trans.Weight() * ip.weight;

    MultVWt(shape, _c0, part_elmat);
    elmat.AddMatrix(w * cos, part_elmat, 0, 0);
    elmat.AddMatrix(w * sin, part_elmat, dof, 0);

    MultVWt(shape, _c1, part_elmat);
    elmat.AddMatrix(-w * sin, part_elmat, 0, 0);
    elmat.AddMatrix(w * cos, part_elmat, dof, 0);
  }
}

void PoissonLinearisedMultipoleOperator::AssembleLeftElementMatrix2D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c0(_coeff_dim), shape(dof), _x(2);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, _coeff_dim);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);
    _x -= _x0;

    fe.CalcShape(ip, shape);

    auto inverse_radius = 1 / _bdr_radius;

    auto sin = _x[1] * inverse_radius;
    auto cos = _x[0] * inverse_radius;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    auto i = 0;
    for (auto k = 1; k <= _degree; k++) {
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      _c0(i++) = cos_k;
      _c0(i++) = sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    auto w = Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, _c0, elmat);
  }
}

void PoissonLinearisedMultipoleOperator::AssembleRightElementMatrix3D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();
  auto dim = Trans.GetSpaceDim();

#ifdef MFEM_THREAD_SAFE
  Vector _c0, _c1, _c2, shape, _x, _sin, _cos, _p, _pm1;
  _x.SetSize(3);
  _c0.SetSize(_coeff_dim);
  _c1.SetSize(_coeff_dim);
  _c2.SetSize(_coeff_dim);
  _sin.SetSize(_degree + 1);
  _cos.SetSize(_degree + 1);
  _p.SetSize(_degree + 1);
  _pm1.SetSize(_degree + 1);
  DenseMatrix part_elmat;
#endif

  shape.SetSize(dof);
  part_elmat.SetSize(dof, _coeff_dim);

  elmat.SetSize(dim * dof, _coeff_dim);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  _sin(0) = 0.0;
  _cos(0) = 1.0;

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);
    _x -= _x0;

    const auto r = _x.Norml2();
    const auto ri = 1 / r;
    const auto rxy = std::sqrt(_x(0) * _x(0) + _x(1) * _x(1));
    const auto cos_theta = _x(2) * ri;
    const auto sin_theta = rxy * ri;
    const auto cosec_theta = rxy > 0 ? 1 / sin_theta : 0;
    const auto cos = rxy > 0 ? _x(0) / rxy : real_t{1};
    const auto sin = rxy > 0 ? _x(1) / rxy : real_t{0};

    _pm1(0) = 0.0;
    _p(0) = Pll(0, cos_theta);

    const auto ratio = r / _bdr_radius;
    auto rfac = std::pow(_bdr_radius, -3);

    auto i = 0;
    for (auto l = 1; l <= _degree; l++) {
      auto fac = rfac * (l + 1) / (2 * l + 1);

      _sin(l) = rxy > 0 ? _sin(l - 1) * cos + _cos(l - 1) * sin : 0.0;
      _cos(l) = _cos(l - 1) * cos - _sin(l - 1) * sin;

      for (auto m = 0; m < l; m++) {
        const auto [alpha, beta] = RecursionCoefficients(l, m);
        _pm1(m) = alpha * (cos_theta * _p(m) - beta * _pm1(m));
      }
      _pm1(l) = Pll(l, cos_theta);
      _p(l) = 0.0;
      std::swap(_p, _pm1);

      auto _p_th = _sqrt[l] * _sqrt[l + 1] * _p(1);
      _c0(i) = fac * l * _p(0);
      _c1(i) = fac * _p_th;
      _c2(i++) = 0.0;

      fac *= _sqrt[2];
      for (auto m = 1; m < l; m++) {
        _p_th = 0.5 * _sqrt[l - m] * _sqrt[l + m + 1] * _p[m + 1] -
                0.5 * _sqrt[l + m] * _sqrt[l - m + 1] * _p[m - 1];

        _c0(i) = fac * l * _p(m) * _cos(m);
        _c1(i) = fac * _p_th * _cos(m);
        _c2(i++) = -fac * m * cosec_theta * _p(m) * _sin(m);

        _c0(i) = fac * l * _p(m) * _sin(m);
        _c1(i) = fac * _p_th * _sin(m);
        _c2(i++) = fac * m * cosec_theta * _p(m) * _cos(m);
      }

      _p_th = -0.5 * _sqrt[2 * l] * _p(l - 1);
      _c0(i) = fac * l * _p(l) * _cos(l);
      _c1(i) = fac * _p_th * _cos(l);
      _c2(i++) = -fac * l * cosec_theta * _p(l) * _sin(l);

      _c0(i) = fac * l * _p(l) * _sin(l);
      _c1(i) = fac * _p_th * _sin(l);
      _c2(i++) = fac * l * cosec_theta * _p(l) * _cos(l);

      rfac *= ratio;
    }

    fe.CalcShape(ip, shape);
    auto w = Trans.Weight() * ip.weight;

    MultVWt(shape, _c0, part_elmat);
    elmat.AddMatrix(w * sin_theta * cos, part_elmat, 0, 0);
    elmat.AddMatrix(w * sin_theta * sin, part_elmat, dof, 0);
    elmat.AddMatrix(w * cos_theta, part_elmat, 2 * dof, 0);

    MultVWt(shape, _c1, part_elmat);
    elmat.AddMatrix(w * cos_theta * cos, part_elmat, 0, 0);
    elmat.AddMatrix(w * cos_theta * sin, part_elmat, dof, 0);
    elmat.AddMatrix(-w * sin_theta, part_elmat, 2 * dof, 0);

    MultVWt(shape, _c2, part_elmat);
    elmat.AddMatrix(-w * sin, part_elmat, 0, 0);
    elmat.AddMatrix(w * cos, part_elmat, dof, 0);
  }
}

void PoissonLinearisedMultipoleOperator::AssembleLeftElementMatrix3D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c0, shape, _x, _sin, _cos, _p, _pm1;
  _x.SetSize(3);
  _c0.SetSize(_coeff_dim);
  _sin.SetSize(_degree + 1);
  _cos.SetSize(_degree + 1);
  _p.SetSize(_degree + 1);
  _pm1.SetSize(_degree + 1);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, _coeff_dim);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  _sin(0) = 0.0;
  _cos(0) = 1.0;

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);
    _x -= _x0;

    const auto r = _x.Norml2();
    const auto ri = 1 / r;
    const auto cos_theta = _x(2) * ri;
    const auto rxy = std::sqrt(_x(0) * _x(0) + _x(1) * _x(1));
    const auto cos = rxy > 0 ? _x(0) / rxy : real_t{1};
    const auto sin = rxy > 0 ? _x(1) / rxy : real_t{0};

    _pm1(0) = 0.0;
    _p(0) = Pll(0, cos_theta);

    auto i = 0;
    for (auto l = 1; l <= _degree; l++) {
      _sin(l) = rxy > 0 ? _sin(l - 1) * cos + _cos(l - 1) * sin : 0.0;
      _cos(l) = _cos(l - 1) * cos - _sin(l - 1) * sin;

      for (auto m = 0; m < l; m++) {
        const auto [alpha, beta] = RecursionCoefficients(l, m);
        _pm1(m) = alpha * (cos_theta * _p(m) - beta * _pm1(m));
      }
      _pm1(l) = Pll(l, cos_theta);
      _p(l) = 0.0;
      std::swap(_p, _pm1);

      _c0(i++) = _p(0);

      for (auto m = 1; m <= l; m++) {
        _c0(i++) = _sqrt[2] * _p[m] * _cos(m);
        _c0(i++) = rxy > 0 ? _sqrt[2] * _p[m] * _sin(m) : 0.0;
      }
    }

    fe.CalcShape(ip, shape);
    auto w = Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, _c0, elmat);
  }
}

}  // namespace mfemElasticity