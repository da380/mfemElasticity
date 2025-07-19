#include "mfemElasticity/operators.hpp"

namespace mfemElasticity {

void PoissonDtNOperator::SetBoundaryMarkerSerial() {
  _x0 = MeshCentroid(_fes->GetMesh());
  _bdr_marker = ExternalBoundaryMarker(_fes->GetMesh());
  auto [found, same, radius] =
      BoundaryRadius(_fes->GetMesh(), _bdr_marker, _x0);
  assert(found == 1 && same == 1);
}

#ifdef MFEM_USE_MPI
void PoissonDtNOperator::SetBoundaryMarkerParallel() {
  _x0 = MeshCentroid(_pfes->GetParMesh());
  _bdr_marker = ExternalBoundaryMarker(_pfes->GetParMesh());
  auto [found, same, radius] =
      BoundaryRadius(_pfes->GetParMesh(), _bdr_marker, _x0);
  assert(found == 1 && same == 1);
}
#endif

void PoissonDtNOperator::SetUp() {
  assert(_dim == 2 || _dim == 3);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    SetBoundaryMarkerParallel();
  } else {
    SetBoundaryMarkerSerial();
  }
#else
  SetBoundaryMarkerSerial();
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
  if (_dim == 3) {
    SetSquareRoots(_degree);
  }
  Assemble();
}

PoissonDtNOperator::PoissonDtNOperator(mfem::FiniteElementSpace* fes,
                                       int degree)
    : mfem::Operator(fes->GetVSize()),
      _fes{fes},
      _dim{fes->GetMesh()->Dimension()},
      _degree{degree},
      _coeff_dim{_dim == 2 ? 2 * degree : (degree + 1) * (degree + 1)},
      _mat(fes->GetVSize(), _coeff_dim) {
  SetUp();
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
      _coeff_dim{_dim == 2 ? 2 * degree : (degree + 1) * (degree + 1)},
      _mat(fes->GetVSize(), _coeff_dim) {
  SetUp();
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

    auto ri = 1 / _x.Norml2();
    auto sin = _x[1] * ri;
    auto cos = _x[0] * ri;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    auto i = 0;
    for (auto k = 1; k <= _degree; k++) {
      auto fac = std::sqrt(pi * k);
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

void PoissonMultipoleOperator::SetBoundaryMarkerSerial() {
  auto* mesh = _tr_fes->GetMesh();
  _x0 = MeshCentroid(mesh);
  _bdr_marker = ExternalBoundaryMarker(mesh);
  auto [found, same, radius] = BoundaryRadius(mesh, _bdr_marker, _x0);
  assert(found == 1 && same == 1);
  _bdr_radius = radius;
}

#ifdef MFEM_USE_MPI
void PoissonMultipoleOperator::SetBoundaryMarkerParallel() {
  auto* mesh = _tr_pfes->GetParMesh();
  _x0 = MeshCentroid(mesh);
  _bdr_marker = ExternalBoundaryMarker(mesh);
  auto [found, same, radius] = BoundaryRadius(mesh, _bdr_marker, _x0);
  assert(found == 1 && same == 1);
  _bdr_radius = radius;
}
#endif

void PoissonMultipoleOperator::SetUp() {
  assert(_dim == 2 || _dim == 3);
  assert(_tr_fes->GetMesh() == _te_fes->GetMesh());
#ifdef MFEM_USE_MPI
  if (_parallel) {
    SetBoundaryMarkerParallel();
  } else {
    SetBoundaryMarkerSerial();
  }
#else
  SetBoundaryMarkerSerial();
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
  if (_dim == 3) {
    SetSquareRoots(_degree);
  }

  Assemble();
}

PoissonMultipoleOperator::PoissonMultipoleOperator(
    mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
    int degree, const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _dim{tr_fes->GetMesh()->Dimension()},
      _degree{degree},
      _coeff_dim{_dim == 2 ? 2 * degree : (degree + 1) * (degree + 1)},
      _dom_marker{dom_marker},
      _lmat(te_fes->GetVSize(), _coeff_dim),
      _rmat(tr_fes->GetVSize(), _coeff_dim) {
  SetUp();
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
      _coeff_dim{_dim == 2 ? 2 * degree : (degree + 1) * (degree + 1)},
      _dom_marker{dom_marker},
      _lmat(te_fes->GetVSize(), _coeff_dim),
      _rmat(tr_fes->GetVSize(), _coeff_dim) {
  SetUp();
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

}  // namespace mfemElasticity