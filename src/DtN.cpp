// #include "mfemElasticity/DtN.hpp"

#include "mfemElasticity/DtN.hpp"

namespace mfemElasticity {

namespace DtN {

Poisson2D::Poisson2D(mfem::FiniteElementSpace* fes, int kmax,
                     mfem::Array<int>& bdr_marker)
    : mfem::Operator(fes->GetVSize()),
      _fes{fes},
      _kmax{kmax},
      _bdr_marker{bdr_marker},
      _mat(2 * _kmax, fes->GetVSize()) {
  assert(_kmax > 0);
  auto* mesh = _fes->GetMesh();
  assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);
}

Poisson2D::Poisson2D(mfem::FiniteElementSpace* fes, int kmax)
    : mfem::Operator(fes->GetVSize()),
      _fes{fes},
      _kmax{kmax},
      _mat(2 * _kmax, fes->GetVSize()) {
  assert(_kmax > 0);
  auto* mesh = _fes->GetMesh();
  assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);
  _bdr_marker.SetSize(mesh->bdr_attributes.Max());
  _bdr_marker = 0;
  mesh->MarkExternalBoundaries(_bdr_marker);
}

#ifdef MFEM_USE_MPI
Poisson2D::Poisson2D(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kmax,
                     mfem::Array<int>& bdr_marker)
    : mfem::Operator(fes->GetVSize()),
      _comm{comm},
      _pfes{fes},
      _fes{fes},
      _kmax{kmax},
      _bdr_marker{bdr_marker},
      _mat(2 * _kmax, fes->GetVSize()),
      _parallel{true} {
  assert(_kmax > 0);
  auto* mesh = _fes->GetMesh();
  assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);
}

Poisson2D::Poisson2D(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kmax)
    : mfem::Operator(fes->GetVSize()),
      _comm{comm},
      _pfes{fes},
      _fes{fes},
      _kmax{kmax},
      _mat(2 * _kmax, fes->GetVSize()),
      _parallel{true} {
  assert(_kmax > 0);
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

  _c.SetSize(2 * _kmax);
  _mat.Mult(x, _c);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, _c.GetData(), 2 * _kmax, MFEM_MPI_REAL_T,
                  MPI_SUM, _comm);
  }
#endif

  auto j = 0;
  for (auto k = 1; k <= _kmax; k++) {
    auto fac = k * M_PI;
    _c(j) *= fac;
    _c(j + 1) *= fac;
    j += 2;
  }
  y.SetSize(x.Size());
  _mat.MultTranspose(_c, y);
}

void Poisson2D::AssembleElementMatrix(const mfem::FiniteElement& fe,
                                      mfem::ElementTransformation& Trans,
                                      mfem::DenseMatrix& elmat) {
  using namespace mfem;
  auto dof = fe.GetDof();
  auto n = 2 * _kmax;

#ifdef MFEM_THREAD_SAFE
  Vector _c, shape, _x;
#endif
  _x.SetSize(2);
  _c.SetSize(n);
  shape.SetSize(dof);
  elmat.SetSize(n, dof);
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
    for (auto k = 1; k <= _kmax; k++) {
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      _c(i) = cos_k;
      _c(i + 1) = sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
      i += 2;
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
  auto rows = Array<int>(2 * _kmax);
  for (auto i = 0; i < 2 * _kmax; i++) {
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

}  // namespace DtN
}  // namespace mfemElasticity