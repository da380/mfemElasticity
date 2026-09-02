#pragma once

/*
  Helpers shared by the self-gravitating tests (serial gtest and the MPI
  program): the canned two-layer meshes in ../data, the attribute
  conventions of those meshes, and a degree-2 surface load.

  Both meshes come from meshing/concentric_* : domain attribute 1 is the
  body (radius 1), attribute 2 the buffer shell; the body surface carries
  boundary attribute 1 and the outer sphere boundary attribute 2. On the
  body SubMesh the surface is the only boundary and keeps attribute 1.
*/

#include <cmath>
#include <memory>
#include <string>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

namespace self_grav_test {

using namespace mfem;
using namespace mfemElasticity;

inline std::string MeshFile(int dim) {
  return dim == 2 ? "../data/elastogravity_2d.msh"
                  : "../data/coupled_poisson.msh";
}

// Non-dimensional constants of the tests: unit radius, G such that the
// coupling is strong enough to matter (rho g R / mu = O(1)).
constexpr double kG = 0.05;
constexpr double kRho = 1.0;
constexpr double kKappa = 1.0;
constexpr double kMu = 0.5;
constexpr int kDtNDegree = 12;

// Body marker on the parent mesh.
inline Array<int> BodyMarker(Mesh& parent) {
  Array<int> m(parent.attributes.Max());
  m = 0;
  m[0] = 1;
  return m;
}

// Surface marker on the body SubMesh (its largest boundary attribute).
inline Array<int> SurfaceMarker(Mesh& body) {
  Array<int> m(body.bdr_attributes.Max());
  m = 0;
  m[body.bdr_attributes.Max() - 1] = 1;
  return m;
}

// Surface mass load: a degree-2 pattern in the polar angle, scaled by
// (1 + t) so that time dependence can be checked.
inline double SurfaceLoad(const Vector& x, double t) {
  const double r = x.Norml2();
  if (r == 0.0) {
    return 0.0;
  }
  const double c = (x.Size() == 2 ? x[1] : x[2]) / r;
  return 0.02 * (1.0 + 3.0 * c * c) * (1.0 + t);
}

// L2 norm of a (Par)GridFunction; global in parallel.
inline double L2Norm(const GridFunction& u) {
  const int vdim = u.FESpace()->GetVDim();
  if (vdim == 1) {
    ConstantCoefficient z(0.0);
    return u.ComputeL2Error(z);
  }
  Vector zero(vdim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return u.ComputeL2Error(z);
}

}  // namespace self_grav_test
