/*
  Parallel tests for LinearQuasiStaticReferentialProblem with the
  prescribed vacuum extension (see TestReferentialProblem.cpp for the
  serial suite and the model). Run with 1, 2 and 4 ranks; a standalone MPI
  program returning the number of failed checks.

  Every rank also solves the serial problem on the full mesh: serial and
  parallel build the same discrete system (the parallel extension's trace
  rows come from the cross-rank dof pairing and its interior rows from the
  same interpolation), so the global L2 norms of displacement and
  potential are compared directly.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace mfem;
using namespace mfemElasticity;

namespace {

constexpr double kG = 0.05;
constexpr double kRho = 1.0;
constexpr double kKappa = 1.0;
constexpr double kMu = 0.5;
constexpr int kDtNDegree = 12;
constexpr double kPi = std::numbers::pi;

int num_checks = 0;
int num_fails = 0;

void Check(double err, double tol, const std::string& what) {
  num_checks++;
  if (!(err <= tol)) {
    num_fails++;
    if (Mpi::Root()) {
      std::cout << "FAIL: " << what << "  (err = " << err << ", tol = " << tol
                << ")\n";
    }
  }
}

double RelErr(double a, double b) { return std::abs(a - b) / std::abs(b); }

double Pressure(const Vector& x) {
  const double r2 = x * x;
  const double p = x.Size() == 2
                       ? kPi * kG * kRho * kRho * (1.0 - r2)
                       : 2.0 * kPi * kG * kRho * kRho * (1.0 - r2) / 3.0;
  return std::max(0.0, p);
}

double SurfaceLoad(const Vector& x) {
  const double r = x.Norml2();
  const double c = (x.Size() == 2 ? x[1] : x[2]) / r;
  return 0.02 * (1.0 + 3.0 * c * c);
}

CallableDiffeomorphism IdentityMap(int dim) {
  return CallableDiffeomorphism(
      dim, [](const Vector& x, Vector& y) { y = x; },
      [](const Vector&, DenseMatrix& F) {
        F = 0.0;
        for (int i = 0; i < F.Height(); i++) {
          F(i, i) = 1.0;
        }
      });
}

double L2Norm(const GridFunction& u) {
  const int vdim = u.FESpace()->GetVDim();
  if (vdim == 1) {
    ConstantCoefficient z(0.0);
    return const_cast<GridFunction&>(u).ComputeL2Error(z);
  }
  Vector zero(vdim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return const_cast<GridFunction&>(u).ComputeL2Error(z);
}

void RunCase(int order, const std::string& label) {
  const char* mesh_file = "../data/elastogravity_2d.msh";
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  Array<int> body_attr({1}), buffer_attr({2});

  Vector bb_min, bb_max;
  smesh.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  auto phi = IdentityMap(dim);
  ConstantCoefficient kappa(kKappa), mu(kMu), rho(kRho);
  FunctionCoefficient p0(Pressure);
  auto C_eff =
      IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);
  BareElasticTensorCoefficient C(dim, C_eff, p0);
  MatrixFunctionCoefficient S(dim, [](const Vector& x, DenseMatrix& S) {
    S.SetSize(x.Size());
    S = 0.0;
    const double p = Pressure(x);
    for (int i = 0; i < x.Size(); i++) {
      S(i, i) = -p;
    }
  });
  ReferentialElasticRheology rheology(dim, C, S, phi);
  FunctionCoefficient sigma(SurfaceLoad);

  // Serial reference on every rank.
  double u_ref = 0.0, z_ref = 0.0;
  {
    SubMesh body(SubMesh::CreateFromDomain(smesh, body_attr));
    SubMesh buffer(SubMesh::CreateFromDomain(smesh, buffer_attr));
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes_u(&body, &fec, dim), fes_zeta(&smesh, &fec);
    FiniteElementSpace fes_buffer(&buffer, &fec, dim);
    LinearQuasiStaticReferentialProblem problem(&fes_u, &fes_zeta, rheology,
                                                rho, kG, kDtNDegree);
    auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
    problem.SetPrescribedVacuumExtension(fes_buffer, *E);
    Array<int> surface(body.bdr_attributes.Max());
    surface = 0;
    surface[body.bdr_attributes.Max() - 1] = 1;
    problem.SetSurfaceLoad(sigma, surface);
    problem.SetRelTol(1e-11);
    problem.AssembleForce(0.0);
    Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " serial solve");
    u_ref = L2Norm(problem.Displacement());
    z_ref = L2Norm(problem.Potential());
  }

  // Parallel problem.
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  ParSubMesh body(ParSubMesh::CreateFromDomain(pmesh, body_attr));
  ParSubMesh buffer(ParSubMesh::CreateFromDomain(pmesh, buffer_attr));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&body, &fec, dim), fes_zeta(&pmesh, &fec);
  ParFiniteElementSpace fes_buffer(&buffer, &fec, dim);
  LinearQuasiStaticReferentialProblem problem(&fes_u, &fes_zeta, rheology,
                                              rho, kG, kDtNDegree);
  auto E = NewRadialVacuumExtension(fes_u, fes_buffer, 1.0, r_out);
  problem.SetPrescribedVacuumExtension(fes_buffer, *E);
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;
  FunctionCoefficient sigma2(SurfaceLoad);
  problem.SetSurfaceLoad(sigma2, surface);
  problem.SetRelTol(1e-11);
  problem.AssembleForce(0.0);
  Check(problem.Solve() ? 0.0 : 1.0, 0.0, label + " parallel solve");
  Check(RelErr(L2Norm(problem.Displacement()), u_ref), 1e-5,
        label + " displacement norm");
  Check(RelErr(L2Norm(problem.Potential()), z_ref), 1e-5,
        label + " potential norm");
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  RunCase(1, "o1");
  RunCase(2, "o2");

  if (Mpi::Root()) {
    if (num_fails == 0) {
      std::cout << "All " << num_checks << " checks passed on "
                << Mpi::WorldSize() << " ranks.\n";
    } else {
      std::cout << num_fails << " of " << num_checks << " checks failed on "
                << Mpi::WorldSize() << " ranks.\n";
    }
  }
  return num_fails;
}
