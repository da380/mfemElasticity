/*
  Parallel tests for LinearElasticProblemBase / TractionProblem / ClampedProblem
  on ParFiniteElementSpaces. Run with 1, 2 and 4 ranks; a standalone MPI
  program returning the number of failed checks.

  Every rank also solves the serial problem on the full mesh and compares
  partition-independent quantities: the L2 norm of the clamped
  displacement (plain and with an effective shear modulus), and the exact
  uniaxial strain of the traction problem.
*/

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "ElasticTestCommon.hpp"
#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace mfem;
using namespace mfemElasticity;
using namespace elastic_test;

namespace {

int num_checks = 0;
int num_fails = 0;

double GlobalMax(double v) {
  double g = 0.0;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

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

double L2Norm(const GridFunction& u) {
  Vector zero(u.FESpace()->GetVDim());
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  return u.ComputeL2Error(z);  // global for ParGridFunction
}

void RunCase(int dim, int elementType, int order, const std::string& label) {
  auto smesh = MakeSmallMesh(dim, elementType);
  const auto x0_attr = BdrAttributeAt(smesh, 0, 0.0);
  const auto x1_attr = BdrAttributeAt(smesh, 0, 1.0);
  const auto nbdr = smesh.bdr_attributes.Max();

  int nxyz[3] = {Mpi::WorldSize(), 1, 1};
  int* partitioning = smesh.CartesianPartitioning(nxyz);
  ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning);
  delete[] partitioning;

  H1_FECollection fec(order, dim);
  FiniteElementSpace sfes(&smesh, &fec, dim);
  ParFiniteElementSpace pfes(&pmesh, &fec, dim);

  ConstantCoefficient kappa(Kappa(dim)), mu(kMu), tau(1.0);
  // Maxwell body (mu_inf = 0, one branch): the unrelaxed modulus is mu and
  // a relaxation weight beta gives mu_eff = beta mu.
  auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  auto ess_bdr = Marker(nbdr, {x0_attr});
  auto pull_marker = Marker(nbdr, {x1_attr});
  auto uni_marker = Marker(nbdr, {x0_attr, x1_attr});
  VectorFunctionCoefficient pull(dim, PullTraction);
  VectorFunctionCoefficient uni(dim, UniaxialTraction);

  // Clamped: serial reference norm vs parallel norm, then with an effective
  // modulus field on an L2 space.
  {
    ClampedProblem serial(&sfes, rheology, ess_bdr, pull, pull_marker);
    ClampedProblem par(&pfes, rheology, ess_bdr, pull, pull_marker);
    Check(par.IsParallel() ? 0.0 : 1.0, 0.0, label + ": IsParallel");

    serial.AssembleForce(0.25);
    Check(serial.Solve() ? 0.0 : 1.0, 0.0, label + ": serial solve");
    par.AssembleForce(0.25);
    Check(par.Solve() ? 0.0 : 1.0, 0.0, label + ": parallel solve");
    const auto ns = L2Norm(serial.Displacement());
    const auto np = L2Norm(par.Displacement());
    Check(GlobalMax(std::abs(ns - np) / ns), 1e-8, label + ": clamped norm");

    L2_FECollection l2fec(0, dim);
    FiniteElementSpace ssfes(&smesh, &l2fec);
    ParFiniteElementSpace psfes(&pmesh, &l2fec);
    FunctionCoefficient mu_var(
        [](const Vector& x) { return 0.3 + 0.6 * x[0]; });
    GridFunction smu(&ssfes);
    ParGridFunction pmu(&psfes);
    smu.ProjectCoefficient(mu_var);
    pmu.ProjectCoefficient(mu_var);
    GridFunctionCoefficient smu_c(&smu), pmu_c(&pmu);
    serial.SetRelaxationWeights(0, {&smu_c});
    par.SetRelaxationWeights(0, {&pmu_c});
    serial.AssembleForce(0.25);
    Check(serial.Solve() ? 0.0 : 1.0, 0.0, label + ": serial solve (eff)");
    par.AssembleForce(0.25);
    Check(par.Solve() ? 0.0 : 1.0, 0.0, label + ": parallel solve (eff)");
    const auto ns_e = L2Norm(serial.Displacement());
    const auto np_e = L2Norm(par.Displacement());
    Check(GlobalMax(std::abs(ns_e - np_e) / ns_e), 1e-8,
          label + ": clamped norm, effective modulus");
    Check(std::abs(ns_e - ns) / ns > 1e-3 ? 0.0 : 1.0, 0.0,
          label + ": effective modulus changed the solution");

    par.ClearRelaxationWeights();
    par.AssembleForce(0.25);
    Check(par.Solve() ? 0.0 : 1.0, 0.0, label + ": parallel solve (clear)");
    Check(GlobalMax(std::abs(L2Norm(par.Displacement()) - ns) / ns), 1e-8,
          label + ": clamped norm after Clear");
  }

  // Traction: exact uniaxial strain on every rank's elements.
  {
    TractionProblem par(&pfes, rheology, uni, uni_marker);
    double exx = 0.0, eyy = 0.0;
    UniaxialStrain(dim, kSigma, exx, eyy);
    par.AssembleForce(0.0);
    Check(par.Solve() ? 0.0 : 1.0, 0.0, label + ": traction solve");
    Check(GlobalMax(MaxStrainError(par.Displacement(), exx, eyy)) / exx, 1e-8,
          label + ": uniaxial strain");
    par.AssembleForce(1.0);
    Check(par.Solve() ? 0.0 : 1.0, 0.0, label + ": traction solve, t = 1");
    Check(GlobalMax(MaxStrainError(par.Displacement(), 2 * exx, 2 * eyy)) / exx,
          1e-8, label + ": uniaxial strain, t = 1");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();

  for (auto dim : {2, 3}) {
    for (auto elementType : {0, 1}) {
      for (auto order : {1, 2}) {
        auto label = "dim=" + std::to_string(dim) +
                     " et=" + std::to_string(elementType) +
                     " p=" + std::to_string(order);
        RunCase(dim, elementType, order, label);
      }
    }
  }

  if (Mpi::Root()) {
    if (num_fails == 0) {
      std::cout << "All " << num_checks << " checks passed on "
                << Mpi::WorldSize() << " ranks.\n";
    } else {
      std::cout << num_fails << " of " << num_checks << " checks FAILED on "
                << Mpi::WorldSize() << " ranks.\n";
    }
  }
  return num_fails == 0 ? 0 : 1;
}
