#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>

#include "mfem.hpp"

namespace mfemElasticity {

namespace DtN {

class Poisson2D : public mfem::Integrator, public mfem::Operator {
 private:
  mfem::FiniteElementSpace* _fes;
  int _kmax;
  int _dtn_bdr_attr;
  mfem::real_t _radius;
  const mfem::real_t pi = 3.1415926535897932385;

  mfem::SparseMatrix _mat;

#ifndef MFEM_THREAD_SAFE
  mutable mfem::Vector _c;
  mfem::Vector shape;
  mfem::DenseMatrix elmat;
#endif

  // Check the mesh has the correct properties.
  void CheckMesh() {
    auto* mesh = _fes->GetMesh();
    assert(mesh->Dimension() == 2 && mesh->SpaceDimension() == 2);
  }

  void GetDtNBoundaryAttribute() {
    auto* mesh = _fes->GetMesh();
    assert(mesh->HasBoundaryElements());

    mfem::Array<int> bdr_marker(mesh->bdr_attributes.Max());
    bdr_marker = 0;
    mesh->MarkExternalBoundaries(bdr_marker);
    for (auto i = 0; i < bdr_marker.Size(); i++) {
      if (bdr_marker[i] == 1) _dtn_bdr_attr = i + 1;
    }
  }

  // Get the radius of the DtN boundary, and verify that
  // it is a circle.
  void GetRadius() {
    auto* mesh = _fes->GetMesh();
    auto x = mfem::Vector();

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

  void AssembleElementMatrix(const mfem::FiniteElement& fe,
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
      auto th = std::atan2(x[1], x[0]);
      auto w = fac * Trans.Weight() * ip.weight;

      fe.CalcShape(ip, shape);

      for (auto k = -_kmax; k < 0; k++) {
        _c(k + _kmax) = w * std::cos(k * th);
      }

      _c(_kmax) = 0.5 * w;

      for (auto k = 1; k <= _kmax; k++) {
        _c(k + _kmax) = w * std::sin(k * th);
      }

      mfem::AddMult_a_VWt(1, _c, shape, elmat);
    }
  }

  void Assemble() {
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
        const auto* fe = _fes->GetBE(i);
        auto* Trans = _fes->GetBdrElementTransformation(i);

        _fes->GetBdrElementVDofs(i, vdofs);
        AssembleElementMatrix(*fe, *Trans, elmat);

        _mat.AddSubMatrix(rows, vdofs, elmat);
      }
    }

    _mat.Finalize();
  }

 public:
  Poisson2D(mfem::FiniteElementSpace* fes, int kmax, int dtn_bdr_attr,
            const mfem::IntegrationRule* ir = nullptr)
      : mfem::Integrator(ir),
        mfem::Operator(fes->GetTrueVSize()),
        _fes{fes},
        _kmax{kmax},
        _dtn_bdr_attr{dtn_bdr_attr},
        _radius{0},
        _mat(NumberOfCoefficients(), fes->GetTrueVSize()) {
    // Check the maximum wavenumber.
    assert(_kmax > -1);

    // Check the mesh.
    CheckMesh();

    // Set the default boundary attribute.
    if (dtn_bdr_attr < 1) {
      GetDtNBoundaryAttribute();
    }

    // Get the boundary radius and check for consistency.
    GetRadius();

    // Assemble the sparse matrix.
    Assemble();
  }

  int NumberOfCoefficients() const { return 2 * _kmax + 1; }

  void FourierCoefficients(const mfem::Vector& x, mfem::Vector& c) const {
    _mat.Mult(x, c);
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
#ifdef MFEM_THREAD_SAFE
    mfem::Vector _c;
#endif
    y.SetSize(x.Size());
    _mat.Mult(x, _c);
    for (auto k = -_kmax; k <= _kmax; k++) {
      _c[k + _kmax] *= pi * std::abs(k);
    }
    _mat.MultTranspose(_c, y);
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override {
    Mult(x, y);
  }
};

}  // namespace DtN

}  // namespace mfemElasticity