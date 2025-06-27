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

 public:
  Poisson2D(mfem::FiniteElementSpace* fes, int kmax, int dtn_bdr_attr = 0)
      : mfem::Operator(fes->GetTrueVSize()),
        _fes{fes},
        _kmax{kmax},
        _dtn_bdr_attr{dtn_bdr_attr},
        _radius{0} {
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
  }

  // Maps a GridFunction onto its Fourier coefficients on the boundary.
  void FourierTransformation(const mfem::Vector& x,
                             mfem::Vector& coeffs) const {
    auto* mesh = _fes->GetMesh();
    mfem::Array<int> vdofs;
    mfem::Vector elvec, point, shape;

    coeffs.SetSize(2 * _kmax + 1);
    coeffs = 0.;

    auto fac = 1 / (pi * _radius);

    for (auto i = 0; i < _fes->GetNBE(); i++) {
      const auto bdr_attr = mesh->GetBdrAttribute(i);
      if (bdr_attr == _dtn_bdr_attr) {
        const auto* el = _fes->GetBE(i);
        auto* Trans = _fes->GetBdrElementTransformation(i);

        const auto* ir = GetIntegrationRule(*el, *Trans);
        if (ir == nullptr) {
          int intorder = 2 * el->GetOrder() + Trans->OrderW();
          ir = &mfem::IntRules.Get(el->GetGeomType(), intorder);
        }

        _fes->GetBdrElementVDofs(i, vdofs);
        x.GetSubVector(vdofs, elvec);

        auto dof = el->GetDof();
        shape.SetSize(dof);

        for (auto j = 0; j < ir->GetNPoints(); j++) {
          const auto& ip = ir->IntPoint(j);
          Trans->SetIntPoint(&ip);
          Trans->Transform(ip, point);
          auto th = std::atan2(point[1], point[0]);
          auto w = fac * Trans->Weight() * ip.weight;

          el->CalcShape(ip, shape);
          auto value = (shape * elvec) * w;

          auto l = 0;
          for (auto k = -_kmax; k < 0; k++) {
            coeffs(k + _kmax) += value * std::cos(k * th);
          }

          coeffs(_kmax) += 0.5 * value;

          for (auto k = 1; k <= _kmax; k++) {
            coeffs(k + _kmax) += value * std::sin(k * th);
          }
        }
      }
    }
  }

  // Maps Fourier coefficients to a GridFunction on the boundary.
  void InverseFourierTransformation(const mfem::Vector& coeffs,
                                    mfem::Vector& x) const {}

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
    y.SetSize(x.Size());
    y = 0.;
  }

  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override {
    Mult(x, y);
  }
};

}  // namespace DtN

}  // namespace mfemElasticity