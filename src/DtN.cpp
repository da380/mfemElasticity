// #include "mfemElasticity/DtN.hpp"

#include "mfemElasticity/DtN.hpp"

namespace mfemElasticity {

namespace DtN {

Poisson2D::Poisson2D(mfem::FiniteElementSpace* fes, int kmax)
    : mfem::Operator(fes->GetTrueVSize()),
      _fes{fes},
      _kmax{kmax},
      _mat(2 * _kmax, fes->GetTrueVSize()) {
  assert(_kmax > 0);
  CheckMesh();
  GetDtNBoundaryAttribute();
}

#ifdef MFEM_USE_MPI
Poisson2D::Poisson2D(MPI_Comm comm, mfem::ParFiniteElementSpace* fes, int kmax)
    : mfem::Operator(fes->GetTrueVSize()),
      _comm{comm},
      _pfes{fes},
      _fes{fes},
      _kmax{kmax},
      _mat(2 * _kmax, fes->GetTrueVSize()),
      _parallel{true} {
  assert(_kmax > 0);
  CheckMesh();
  GetDtNBoundaryAttribute();
}
#endif

void Poisson2D::Mult(const mfem::Vector& x, mfem::Vector& y) const {
#ifdef MFEM_THREAD_SAFE
  mfem::Vector _c;
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
  for (auto i = 0; i < _kmax; i++) {
    auto fac = std::abs(i + 1) * pi;
    _c[j] *= fac;
    _c[j + 1] *= fac;
    j += 2;
  }
  y.SetSize(x.Size());
  _mat.MultTranspose(_c, y);
}

void Poisson2D::CheckMesh() const {
  auto* mesh = _fes->GetMesh();
  assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);
}

void Poisson2D::GetDtNBoundaryAttribute() {
  auto* mesh = _fes->GetMesh();
  assert(mesh->HasBoundaryElements());

  mfem::Array<int> bdr_marker(mesh->bdr_attributes.Max());
  bdr_marker = 0;
  mesh->MarkExternalBoundaries(bdr_marker);
  for (auto i = 0; i < bdr_marker.Size(); i++) {
    if (bdr_marker[i] == 1) _dtn_bdr_attr = i + 1;
  }
}

void Poisson2D::AssembleElementMatrix(const mfem::FiniteElement& fe,
                                      mfem::ElementTransformation& Trans,
                                      mfem::DenseMatrix& elmat) {
  auto dof = fe.GetDof();
  auto n = 2 * _kmax;

#ifdef MFEM_THREAD_SAFE
  mfem::Vector _c, shape;
#endif
  _c.SetSize(n);
  shape.SetSize(dof);
  elmat.SetSize(n, dof);
  elmat = 0.0;

  auto x = mfem::Vector(2);

  const auto* ir = GetIntegrationRule(fe, Trans);
  if (ir == nullptr) {
    int intorder = fe.GetOrder() + Trans.OrderW();
    ir = &mfem::IntRules.Get(fe.GetGeomType(), intorder);
  }

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x);

    fe.CalcShape(ip, shape);

    auto ri = 1 / x.Norml2();

    auto sin = x[1] * ri;
    auto cos = x[0] * ri;

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

    auto w = ri * Trans.Weight() * ip.weight / pi;

    mfem::AddMult_a_VWt(w, _c, shape, elmat);
  }
}

void Poisson2D::Assemble() {
  auto* mesh = _fes->GetMesh();

  auto elmat = mfem::DenseMatrix();
  auto vdofs = mfem::Array<int>();
  auto rows = mfem::Array<int>(2 * _kmax);
  for (auto i = 0; i < 2 * _kmax; i++) {
    rows[i] = i;
  }

  for (auto i = 0; i < _fes->GetNBE(); i++) {
    const auto bdr_attr = mesh->GetBdrAttribute(i);
    if (bdr_attr == _dtn_bdr_attr) {
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
}  // namespace mfemElasticity