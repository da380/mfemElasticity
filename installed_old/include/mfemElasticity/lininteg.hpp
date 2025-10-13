#pragma once

#include "mfem.hpp"

namespace mfemElasticity {

/*
LinearFormIntegrator that acts on vector fields, u, by:

u \mapsto \int_{\Omega} m_{ij} u_{i,j} dx

where \Omega is the domain, m_{ij} a matrix coefficient.

It is assumed that u is defined on a product of scalar finite
element spaces for which the gradient operator is defined.

It is also assumed that the matrix, m, is square with dimension
equal to the spatial dimension for the finite-element space.

TODO: Extension to allow for delta-function coefficiients.
*/
class DomainLFDeformationGradientIntegrator
    : public mfem::LinearFormIntegrator {
 private:
  mfem::MatrixCoefficient& _M;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector v;
  mfem::DenseMatrix dshape, m;
#endif

 public:
  DomainLFDeformationGradientIntegrator(
      mfem::MatrixCoefficient& M, const mfem::IntegrationRule* ir = nullptr)
      : mfem::LinearFormIntegrator(ir), _M{M} {}

  void AssembleRHSElementVect(const mfem::FiniteElement& el,
                              mfem::ElementTransformation& Trans,
                              mfem::Vector& elvect) override;

  using mfem::LinearFormIntegrator::AssembleRHSElementVect;
};

}  // namespace mfemElasticity