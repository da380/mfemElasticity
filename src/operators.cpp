#include "mfemElasticity/operators.hpp"

namespace mfemElasticity {

PoissonDtN::PoissonDtN(mfem::FiniteElementSpace* fes, int coeff_dim,
                       const mfem::Array<int>& bdr_marker)
    : mfem::Operator(fes->GetVSize()),
      _fes{fes},
      _coeff_dim{coeff_dim},
      _bdr_marker{bdr_marker},
      _mat(fes->GetVSize(), _coeff_dim) {
  CheckMesh();
#ifndef MFEM_THREAD_SAFE
  _c.SetSize(_coeff_dim);
  _x.SetSize(_fes->GetMesh()->Dimension());
#endif
}

#ifdef MFEM_USE_MPI
PoissonDtN::PoissonDtN(MPI_Comm comm, mfem::ParFiniteElementSpace* fes,
                       int coeff_dim, const mfem::Array<int>& bdr_marker)
    : mfem::Operator(fes->GetVSize()),
      _parallel{true},
      _comm{comm},
      _pfes{fes},
      _fes{fes},
      _coeff_dim{coeff_dim},
      _bdr_marker{bdr_marker},
      _mat(fes->GetVSize(), _coeff_dim) {
  CheckMesh();
#ifndef MFEM_THREAD_SAFE
  _c.SetSize(_coeff_dim);
  _x.SetSize(_fes->GetMesh()->Dimension());
#endif
}
#endif

void PoissonDtN::Mult(const mfem::Vector& x, mfem::Vector& y) const {
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

void PoissonDtN::Assemble() {
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

      _mat.AddSubMatrix(vdofs, rows, elmat);
    }
  }

  _mat.Finalize();
}

mfem::RAPOperator PoissonDtN::RAP() const {
  auto* P = _fes->GetProlongationMatrix();
  return mfem::RAPOperator(*P, *this, *P);
}

void PoissonDtNCircle::AssembleElementMatrix(const mfem::FiniteElement& fe,
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
    AddMult_a_VWt(w, shape, _c, elmat);
  }
}

void PoissonDtNSphere::AssembleElementMatrix(const mfem::FiniteElement& fe,
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
  elmat.SetSize(dof, _coeff_dim);
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

    AddMult_a_VWt(w, shape, _c, elmat);
  }
}

PoissonMultipole::PoissonMultipole(mfem::FiniteElementSpace* tr_fes,
                                   mfem::FiniteElementSpace* te_fes,
                                   int coeff_dim,
                                   const mfem::Array<int>& dom_marker,
                                   const mfem::Array<int>& bdr_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _coeff_dim{coeff_dim},
      _dom_marker{dom_marker},
      _bdr_marker{bdr_marker},
      _lmat(te_fes->GetVSize(), _coeff_dim),
      _rmat(tr_fes->GetVSize(), _coeff_dim) {
  CheckMesh();
#ifndef MFEM_THREAD_SAFE
  _c.SetSize(_coeff_dim);
  _x.SetSize(_tr_fes->GetMesh()->Dimension());
#endif
  _bdr_radius = ExternalBoundaryRadius();
}

#ifdef MFEM_USE_MPI
PoissonMultipole::PoissonMultipole(MPI_Comm comm,
                                   mfem::ParFiniteElementSpace* tr_fes,
                                   mfem::ParFiniteElementSpace* te_fes,
                                   int coeff_dim,
                                   const mfem::Array<int>& dom_marker,
                                   const mfem::Array<int>& bdr_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      _parallel{true},
      _comm{comm},
      _tr_pfes{tr_fes},
      _tr_fes{tr_fes},
      _te_fes{te_fes},
      _te_pfes{te_fes},
      _coeff_dim{coeff_dim},
      _dom_marker{dom_marker},
      _bdr_marker{bdr_marker},
      _lmat(te_fes->GetVSize(), _coeff_dim),
      _rmat(tr_fes->GetVSize(), _coeff_dim) {
  CheckMesh();
#ifndef MFEM_THREAD_SAFE
  _c.SetSize(_coeff_dim);
  _x.SetSize(_tr_fes->GetMesh()->Dimension());
#endif
  _bdr_radius = ParallelExternalBoundaryRadius();
}
#endif

mfem::real_t PoissonMultipole::ExternalBoundaryRadius() const {
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
mfem::real_t PoissonMultipole::ParallelExternalBoundaryRadius() const {
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

void PoissonMultipole::Mult(const mfem::Vector& x, mfem::Vector& y) const {
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

void PoissonMultipole::MultTranspose(const mfem::Vector& x,
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

void PoissonMultipole::Assemble() {
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

mfem::RAPOperator PoissonMultipole::RAP() const {
  auto* P_te = _te_fes->GetProlongationMatrix();
  auto* P_tr = _tr_fes->GetProlongationMatrix();
  return mfem::RAPOperator(*P_te, *this, *P_tr);
}

void PoissonMultipoleCircle::AssembleRightElementMatrix(
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

void PoissonMultipoleCircle::AssembleLeftElementMatrix(
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

void PoissonMultipoleSphere::AssembleRightElementMatrix(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
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
  elmat.SetSize(dof, _coeff_dim);
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

    const auto ratio = r / _bdr_radius;
    auto rfac = 1 / (_bdr_radius * _bdr_radius);
    _c(0) = rfac * _p(0);

    auto i = 1;
    for (auto l = 1; l <= _lMax; l++) {
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

void PoissonMultipoleSphere::AssembleLeftElementMatrix(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
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
  elmat.SetSize(dof, _coeff_dim);
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

    _c(0) = _p(0);

    auto i = 1;
    for (auto l = 1; l <= _lMax; l++) {
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

FirstMoments::FirstMoments(mfem::FiniteElementSpace* fes,
                           const mfem::Array<int>& dom_marker)
    : mfem::Operator(fes->GetMesh()->Dimension(), fes->GetVSize()),
      _dim{fes->GetMesh()->Dimension()},
      _fes{fes},
      _dom_marker{dom_marker},
      _mat(fes->GetVSize(), _dim) {
#ifndef MFEM_THREAD_SAFE
  _x.SetSize(_dim);
#endif
}

#ifdef MFEM_USE_MPI
FirstMoments::FirstMoments(MPI_Comm comm, mfem::ParFiniteElementSpace* fes,
                           const mfem::Array<int>& dom_marker)
    : mfem::Operator(fes->GetMesh()->Dimension(), fes->GetVSize()),
      _parallel{true},
      _comm{comm},
      _pfes{fes},
      _dim{fes->GetMesh()->Dimension()},
      _fes{fes},
      _dom_marker{dom_marker},
      _mat(fes->GetVSize(), _dim) {
#ifndef MFEM_THREAD_SAFE
  _x.SetSize(_dim);
#endif
}

#endif

void FirstMoments::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

  y.SetSize(_dim);
  _mat.MultTranspose(x, y);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, y.GetData(), _dim, MFEM_MPI_REAL_T, MPI_SUM,
                  _comm);
  }
#endif
}

void FirstMoments::MultTranspose(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

  y.SetSize(width);
  _mat.Mult(x, y);
}

void FirstMoments::Assemble() {
  using namespace mfem;
  auto* mesh = _fes->GetMesh();

  auto elmat = DenseMatrix();
  auto vdofs = Array<int>();
  auto rows = Array<int>(_dim);
  for (auto i = 0; i < _dim; i++) {
    rows[i] = i;
  }

  for (auto i = 0; i < _fes->GetNE(); i++) {
    const auto elm_attr = mesh->GetAttribute(i);
    if (_dom_marker[elm_attr - 1] == 1) {
      _fes->GetElementVDofs(i, vdofs);
      const auto* fe = _fes->GetFE(i);
      auto* Trans = _fes->GetElementTransformation(i);

      AssembleElementMatrix(*fe, *Trans, elmat);

      _mat.AddSubMatrix(vdofs, rows, elmat);
    }
  }

  _mat.Finalize();
}

void FirstMoments::AssembleElementMatrix(const mfem::FiniteElement& fe,
                                         mfem::ElementTransformation& Trans,
                                         mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dim = _fes->GetMesh()->Dimension();
  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _x, shape;
  _x.SetSize(dim);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, _dim);
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
    auto w = Trans.Weight() * ip.weight;

    AddMult_a_VWt(w, shape, _x, elmat);
  }
}

}  // namespace mfemElasticity