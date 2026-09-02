/*
  Parallel tests for ViscoelasticOperator on ParFiniteElementSpaces. Run
  with 1, 2 and 4 ranks; a standalone MPI program returning the number of
  failed checks.

  The checks are the partition-independent analytic ones of the serial
  test: Maxwell creep under constant uniaxial stress (exponential
  trapezoid, exact) and relaxation under a prescribed uniform strain
  (exponential Euler and backward Euler, exact at their discrete levels).
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

int TFIndex(int dim, int j, int k) {
  if (j < k) {
    std::swap(j, k);
  }
  return j + k * dim - k * (k + 1) / 2;
}

Vector DeviatoricPart(const DenseMatrix& A) {
  const int dim = A.Height();
  double tr = 0.0;
  for (int i = 0; i < dim; i++) {
    tr += A(i, i);
  }
  Vector d(dim * (dim + 1) / 2 - 1);
  for (int k = 0; k < dim; k++) {
    for (int j = k; j < dim; j++) {
      const int idx = TFIndex(dim, j, k);
      if (idx < d.Size()) {
        d[idx] = 0.5 * (A(j, k) + A(k, j)) - (j == k ? tr / dim : 0.0);
      }
    }
  }
  return d;
}

void ConstantUniaxial(const Vector& x, Vector& f) {
  UniaxialTraction(x, 0.0, f);
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
  ParFiniteElementSpace pfes(&pmesh, &fec, dim);
  ConstantCoefficient kappa(Kappa(dim)), mu(kMu), tau(1.0);

  // Creep, exponential trapezoid at dt = 2 tau.
  {
    auto rheology = GeneralisedMaxwellRheology::Maxwell(dim, kappa, mu, tau);
    auto marker = Marker(nbdr, {x0_attr, x1_attr});
    VectorFunctionCoefficient traction(dim, ConstantUniaxial);
    TractionProblem problem(&pfes, rheology, traction, marker);
    ViscoelasticOperator visco(problem);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0, dt = 2.0;
    for (int step = 0; step < 3; step++) {
      ode.Step(m, t, dt);
    }
    Check(visco.SolveElastic(m, t) ? 0.0 : 1.0, 0.0, label + ": creep solve");
    double exx0 = 0.0, eyy0 = 0.0;
    UniaxialStrain(dim, kSigma, exx0, eyy0);
    const double eta = kMu * 1.0;
    const double exx = exx0 + kSigma * (1.0 - 1.0 / dim) * t / (2.0 * eta);
    const double eyy = eyy0 - kSigma / dim * t / (2.0 * eta);
    Check(GlobalMax(MaxStrainError(problem.Displacement(), exx, eyy)) / exx0,
          1e-8, label + ": creep strain");
  }

  // Relaxation under a prescribed uniform strain, two branches.
  {
    ConstantCoefficient mu_inf(0.3), mu1(0.7), mu2(0.5), tau1(1.0), tau2(100.0);
    std::vector<MaxwellBranch> branches{{&mu1, &tau1}, {&mu2, &tau2}};
    GeneralisedMaxwellRheology rheology(dim, kappa, mu_inf, branches);
    DenseMatrix A(dim);
    A = 0.0;
    A(0, 0) = 0.01;
    A(0, 1) = 0.004;
    A(1, 0) = -0.002;
    A(1, 1) = -0.006;
    if (dim == 3) {
      A(2, 2) = 0.003;
      A(0, 2) = 0.005;
    }
    VectorFunctionCoefficient dirichlet(
        dim, [&](const Vector& x, Vector& u) { A.Mult(x, u); });
    Vector zero(dim);
    zero = 0.0;
    VectorConstantCoefficient no_traction(zero);
    Array<int> all(nbdr), none(nbdr);
    all = 1;
    none = 0;
    ClampedProblem problem(&pfes, rheology, all, no_traction, none, &dirichlet);
    ViscoelasticOperator visco(problem);
    Check(std::abs(visco.MinRelaxationTime() - 1.0), 1e-14,
          label + ": min relaxation time");
    const Vector d0 = DeviatoricPart(A);
    const double d0_max = d0.Normlinf();
    const int nd = visco.InternalScalarSpace(0).GetVSize();
    const double taus[2] = {1.0, 100.0};

    // Max over nodes, components and branches of the error relative to the
    // largest prescribed component (some components are exactly zero).
    auto max_err = [&](const Vector& m, auto factor) {
      double err = 0.0;
      for (int k = 0; k < 2; k++) {
        Vector mk = visco.Branch(m, 0, k);
        for (int c = 0; c < d0.Size(); c++) {
          for (int p = 0; p < nd; p++) {
            err = std::max(
                err, std::abs(mk[c * nd + p] - d0[c] * factor(k)) / d0_max);
          }
        }
      }
      return GlobalMax(err);
    };

    {
      ExponentialEulerSolver ode;
      ode.Init(visco);
      Vector m(visco.Height());
      m = 0.0;
      double t = 0.0, dt = 5.0;
      for (int step = 0; step < 4; step++) {
        ode.Step(m, t, dt);
      }
      Check(max_err(m, [&](int k) { return 1.0 - std::exp(-t / taus[k]); }),
            1e-9, label + ": ETD1 relaxation");
    }
    {
      BackwardEulerSolver ode;
      ode.Init(visco);
      Vector m(visco.Height());
      m = 0.0;
      double t = 0.0, dt = 0.5;
      const int n = 4;
      for (int step = 0; step < n; step++) {
        ode.Step(m, t, dt);
      }
      Check(
          max_err(
              m, [&](int k) { return 1.0 - std::pow(1.0 + dt / taus[k], -n); }),
          1e-9, label + ": BE relaxation");
    }
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
