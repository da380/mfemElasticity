// #include "mfemElasticity/DtN.hpp"

#include "mfemElasticity/DtN.hpp"

namespace mfemElasticity {

namespace DtN {

Poisson2D::Poisson2D(mfem::FiniteElementSpace* fes, int kmax, int dtn_bdr_attr,
                     const mfem::IntegrationRule* ir)
    : mfem::Integrator(ir),
      mfem::Operator(fes->GetTrueVSize()),
      _fes{fes},
      _kmax{kmax},
      _dtn_bdr_attr{dtn_bdr_attr},
      _mat(NumberOfCoefficients(), fes->GetTrueVSize()) {
  assert(_kmax > -1);

  CheckMesh();

  if (dtn_bdr_attr < 1) {
    GetDtNBoundaryAttribute();
  }
  GetRadius();
  Assemble();
}

void Poisson2D::FourierCoefficients(const mfem::Vector& x,
                                    mfem::Vector& c) const {
  _mat.Mult(x, c);
}

void Poisson2D::Mult(const mfem::Vector& x, mfem::Vector& y) const {
#ifdef MFEM_THREAD_SAFE
  mfem::Vector _c;
#endif
  _c.SetSize(NumberOfCoefficients());
  _mat.Mult(x, _c);
  auto fac = pi * _radius;
  for (auto k = -_kmax; k <= _kmax; k++) {
    _c[k + _kmax] *= fac * std::abs(k);
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

void Poisson2D::GetRadius() {
  auto* mesh = _fes->GetMesh();
  auto x = mfem::Vector();
  _radius = 0.0;
  for (auto i = 0; i < _fes->GetNBE(); i++) {
    const auto bdr_attr = mesh->GetBdrAttribute(i);
    if (bdr_attr == _dtn_bdr_attr) {
      const auto* el = _fes->GetBE(i);
      auto* Trans = _fes->GetBdrElementTransformation(i);
      const auto ir = el->GetNodes();
      for (auto j = 0; j < ir.GetNPoints(); j++) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(j);
        Trans->SetIntPoint(&ip);
        Trans->Transform(ip, x);
        auto r = x.Norml2();
        if (_radius == 0.0) {
          _radius = r;
        } else {
          auto err = std::abs(r - _radius);
          assert(err < 1e-3 * _radius);
        }
      }
    }
  }
  assert(_radius > 0);
}

void Poisson2D::AssembleElementMatrix(const mfem::FiniteElement& fe,
                                      mfem::ElementTransformation& Trans,
                                      mfem::DenseMatrix& elmat) {
  auto dof = fe.GetDof();
  auto n = NumberOfCoefficients();

#ifdef MFEM_THREAD_SAFE
  mfem::Vector _c, shape;
#endif
  _c.SetSize(n);
  shape.SetSize(dof);
  elmat.SetSize(n, dof);
  elmat = 0.0;

  auto x = mfem::Vector(2);

  auto fac = 1 / (pi * _radius);

  const auto* ir = GetIntegrationRule(fe, Trans);
  if (ir == nullptr) {
    int intorder = 2 * fe.GetOrder() + Trans.OrderW();
    ir = &mfem::IntRules.Get(fe.GetGeomType(), intorder);
  }

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x);
    auto w = fac * Trans.Weight() * ip.weight;

    fe.CalcShape(ip, shape);

    auto norm = x.Norml2();
    auto sin = x[1] / norm;
    auto cos = x[0] / norm;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    _c(_kmax) = 0.5;
    for (auto k = 1; k <= _kmax; k++) {
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      _c(-k + _kmax) = cos_k;
      _c(k + _kmax) = sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    mfem::AddMult_a_VWt(w, _c, shape, elmat);
  }
}

void Poisson2D::Assemble() {
  auto* mesh = _fes->GetMesh();

  auto elmat = mfem::DenseMatrix();
  auto vdofs = mfem::Array<int>();
  auto rows = mfem::Array<int>(NumberOfCoefficients());
  for (auto i = 0; i < rows.Size(); i++) {
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