// ============================================================================
// self_gravitating_elasticity_p.cpp
//
// Parallel driver for SelfGravitatingElasticProblem (mfemElasticity/self_gravitating.hpp):
// a self-gravitating elastic body under a degree-2 surface mass load, solved
// with the Schur-complement CG and/or the block MINRES solver.
//
// The mesh is a ball (disc) inside a larger ball whose outer boundary is a
// sphere (circle); domain attribute 1 is the body, the body surface is the
// SubMesh's largest boundary attribute. The canned meshes in ../data are
//     elastogravity_2d.msh     2-D, order 2, radii 1 and 1.2
//     coupled_poisson.msh      3-D, order 2, radii 1 and 2 (coarse)
//
// Sample runs:
//    mpirun -np 4 ./self_gravitating_elasticity_p -o 2
//    mpirun -np 4 ./self_gravitating_elasticity_p -m ../data/coupled_poisson.msh -o 2 -s 2
//    mpirun -np 2 ./self_gravitating_elasticity_p -o 2 -s 2 -diag
//
// With -s 2 both solvers are run and compared; since each fixes the rigid
// gauge differently (the Schur solver makes u orthogonal to the rigid modes,
// MINRES makes (u, phi) orthogonal to the coupled null vectors), the
// difference is reported before and after removing the rigid component.
// With -diag the rigid-mode residuals ||S u_r|| of the discretisation are
// printed; they measure how well the discrete operator preserves the
// rigid-body null space and should decrease with refinement.
// ============================================================================

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

#include "mfemElasticity.hpp"

using namespace mfem;
using namespace mfemElasticity;

namespace {

real_t load_amplitude = 0.02;

// Degree-2 surface mass load in the polar angle.
real_t SurfaceLoad(const Vector& x) {
  const real_t r = x.Norml2();
  if (r == 0.0) {
    return 0.0;
  }
  const real_t c = (x.Size() == 2 ? x[1] : x[2]) / r;
  return load_amplitude * (1.0 + 3.0 * c * c);
}

real_t L2Norm(const GridFunction& u) {
  const int vdim = u.FESpace()->GetVDim();
  Vector zero(vdim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  ConstantCoefficient zs(0.0);
  return vdim == 1 ? u.ComputeL2Error(zs) : u.ComputeL2Error(z);
}

struct Result {
  ParGridFunction u, phi;
  int outer = 0, inner = 0;
  double seconds = 0.0;
};

Result Run(SelfGravitatingElasticProblem& p,
           SelfGravitatingElasticProblem::SolverType type) {
  p.SetSolverType(type);
  p.AssembleForce(0.0);
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = p.Solve();
  const auto t1 = std::chrono::steady_clock::now();
  Result r{static_cast<const ParGridFunction&>(p.Displacement()),
           static_cast<const ParGridFunction&>(p.Potential()),
           p.LastOuterIterations(),
           p.LastInnerIterations(),
           std::chrono::duration<double>(t1 - t0).count()};
  const char* name =
      type == SelfGravitatingElasticProblem::SolverType::SchurCG
          ? "Schur CG   "
          : "Block MINRES";
  // The norms are collective: compute them on every rank before printing.
  const double u_norm = L2Norm(r.u), phi_norm = L2Norm(r.phi);
  if (Mpi::Root()) {
    std::cout << name << ": " << (ok ? "converged" : "FAILED") << ", outer "
              << r.outer << ", inner " << r.inner << ", " << r.seconds
              << " s, ||u|| = " << u_norm << ", ||phi|| = " << phi_norm
              << "\n";
  }
  return r;
}

}  // namespace

int main(int argc, char* argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const char* mesh_file = "../data/elastogravity_2d.msh";
  int order = 1;
  int solver = 0;
  int dtn_degree = 16;
  real_t G = 0.05;
  real_t rho = 1.0;
  real_t kappa = 1.0;
  real_t mu = 0.5;
  real_t rel_tol = 1e-10;
  bool diagnostics = false;
  bool visualization = false;
  bool paraview = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file (ball in a ball).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&solver, "-s", "--solver",
                 "0: Schur-complement CG, 1: block MINRES, 2: both.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree",
                 "Truncation degree of the DtN expansion.");
  args.AddOption(&G, "-G", "--gravitational-constant",
                 "Gravitational constant (non-dimensional).");
  args.AddOption(&rho, "-rho", "--density", "Density of the body.");
  args.AddOption(&kappa, "-kappa", "--bulk-modulus", "Bulk modulus.");
  args.AddOption(&mu, "-mu", "--shear-modulus", "Shear modulus.");
  args.AddOption(&load_amplitude, "-load", "--load-amplitude",
                 "Amplitude of the surface mass load.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
  args.AddOption(&diagnostics, "-diag", "--diagnostics", "-no-diag",
                 "--no-diagnostics", "Print the rigid-mode residuals.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Send the fields to GLVis.");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Write the fields to a ParaView collection.");
  args.Parse();
  if (!args.Good()) {
    if (Mpi::Root()) args.PrintUsage(std::cout);
    return 1;
  }
  if (Mpi::Root()) args.PrintOptions(std::cout);

  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  ParMesh parent(MPI_COMM_WORLD, smesh);
  smesh.Clear();
  Array<int> body_marker(parent.attributes.Max());
  body_marker = 0;
  body_marker[0] = 1;
  ParSubMesh body(ParSubMesh::CreateFromDomain(parent, body_marker));

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&body, &fec, dim);
  ParFiniteElementSpace fes_phi(&parent, &fec);
  // GlobalTrueVSize() is collective: call it on every rank.
  const HYPRE_BigInt n_u = fes_u.GlobalTrueVSize();
  const HYPRE_BigInt n_phi = fes_phi.GlobalTrueVSize();
  if (Mpi::Root()) {
    std::cout << "Ranks: " << Mpi::WorldSize() << ", displacement unknowns: "
              << n_u << ", potential unknowns: " << n_phi << "\n";
  }

  ConstantCoefficient kappa_coeff(kappa), mu_coeff(mu), rho_coeff(rho);
  auto rheology =
      IsotropicElasticRheology(dim, kappa_coeff, mu_coeff);
  FunctionCoefficient sigma(SurfaceLoad);
  Array<int> surface(body.bdr_attributes.Max());
  surface = 0;
  surface[body.bdr_attributes.Max() - 1] = 1;

  SelfGravitatingElasticProblem problem(&fes_u, &fes_phi, rheology, rho_coeff,
                                        G, dtn_degree);
  problem.SetSurfaceLoad(sigma, surface);
  problem.SetRelTol(rel_tol);

  const real_t g_surface = 4.0 * M_PI * G * rho / dim;  // uniform body
  if (Mpi::Root()) {
    std::cout << "Coupling strength rho g R / mu = " << rho * g_surface / mu
              << "\n";
  }

  if (diagnostics) {
    const auto res = problem.RigidModeResiduals();
    if (Mpi::Root()) {
      std::cout << "Rigid-mode residuals ||S u_r|| / ||A_uu||_max:";
      for (auto r : res) {
        std::cout << " " << r;
      }
      std::cout << "\n";
    }
  }

  std::unique_ptr<Result> schur, minres;
  if (solver == 0 || solver == 2) {
    schur = std::make_unique<Result>(
        Run(problem, SelfGravitatingElasticProblem::SolverType::SchurCG));
  }
  if (solver == 1 || solver == 2) {
    minres = std::make_unique<Result>(
        Run(problem, SelfGravitatingElasticProblem::SolverType::BlockMINRES));
  }

  if (schur && minres) {
    ParGridFunction du(minres->u), dphi(minres->phi);
    du -= schur->u;
    dphi -= schur->phi;
    const double du_rel = L2Norm(du) / L2Norm(schur->u);
    const double dphi_rel = L2Norm(dphi) / L2Norm(schur->phi);
    if (Mpi::Root()) {
      std::cout << "MINRES vs Schur: ||du||/||u|| = " << du_rel
                << ", ||dphi||/||phi|| = " << dphi_rel << "\n";
    }
    // Remove the rigid component of the displacement difference.
    Vector t;
    du.GetTrueDofs(t);
    problem.RigidModes().Project(t);
    du.SetFromTrueDofs(t);
    const double du_proj = L2Norm(du) / L2Norm(schur->u);
    if (Mpi::Root()) {
      std::cout << "   after removing the rigid modes: ||du||/||u|| = "
                << du_proj << "\n";
    }
  }

  if (paraview) {
    ParaViewDataCollection dc("self_gravitating_p", &body);
    dc.SetPrefixPath("output");
    dc.SetHighOrderOutput(order > 1);
    dc.SetLevelsOfDetail(order);
    problem.RegisterFields(dc);
    dc.SetCycle(0);
    dc.SetTime(0.0);
    dc.Save();
  }

  if (visualization) {
    socketstream sock("localhost", 19916);
    sock.precision(8);
    sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n"
         << "solution\n"
         << body << problem.Displacement() << "window_title 'displacement'"
         << std::flush;
    socketstream sock2("localhost", 19916);
    sock2.precision(8);
    sock2 << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n"
          << "solution\n"
          << parent << problem.Potential() << "window_title 'potential'"
          << std::flush;
  }
  return 0;
}
