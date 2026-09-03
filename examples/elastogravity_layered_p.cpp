// ============================================================================
// elastogravity_layered_p.cpp
//
// Parallel version of elastogravity_layered.cpp: the same layered models
// with a fluid outer core on a ParMesh / ParSubMesh, through
// SelfGravitatingElasticProblem on ParFiniteElementSpaces.
//
// Sample runs:
//    mpirun -np 4 ./elastogravity_layered_p -o 2
//    mpirun -np 4 ./elastogravity_layered_p -m ../data/elastogravity_three_layer_3d.msh -o 1 -s 2 -diag
// ============================================================================

#include <mpi.h>

#include <chrono>
#include <iostream>
#include <memory>

#include "layered_model.hpp"
#include "mfemElasticity.hpp"

using namespace mfem;
using namespace mfemElasticity;
using namespace layered;

namespace {

struct Result {
  ParGridFunction u, phi;
  int outer = 0, inner = 0;
  double seconds = 0.0;
};

Result Run(SelfGravitatingElasticProblem& p,
           SelfGravitatingElasticProblem::SolverType type) {
  p.SetSolverType(type);
  p.AssembleForce(0.0);
  MPI_Barrier(MPI_COMM_WORLD);
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = p.Solve();
  const auto t1 = std::chrono::steady_clock::now();
  Result r{static_cast<const ParGridFunction&>(p.Displacement()),
           static_cast<const ParGridFunction&>(p.Potential()),
           p.LastOuterIterations(), p.LastInnerIterations(),
           std::chrono::duration<double>(t1 - t0).count()};
  const char* name =
      type == SelfGravitatingElasticProblem::SolverType::SchurCG
          ? "Schur CG    "
          : "Block MINRES";
  const real_t u_norm = L2Norm(r.u), phi_norm = L2Norm(r.phi);
  if (Mpi::Root()) {
    std::cout.precision(10);
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

  const char* mesh_file = "../data/elastogravity_two_layer_2d.msh";
  int order = 1;
  int solver = 1;
  int dtn_degree = 16;
  real_t rel_tol = 1e-10;
  bool no_fluid_mass = false;
  bool diagnostics = false;
  bool paraview = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Two- or three-layer mesh.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&solver, "-s", "--solver",
                 "0: Schur-complement CG, 1: block MINRES, 2: both.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree",
                 "Truncation degree of the DtN expansion.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
  args.AddOption(&load_factor, "-load", "--load-factor",
                 "Factor on the surface mass load (0: none).");
  args.AddOption(&tidal_amplitude, "-tidal", "--tidal-amplitude",
                 "Amplitude of the degree-2 tidal potential [m^2/s^2].");
  args.AddOption(&solid_core, "-solid-core", "--solid-core", "-fluid-core",
                 "--fluid-core", "Treat the outer core as solid.");
  args.AddOption(&no_fluid_mass, "-no-fluid-mass", "--no-fluid-mass",
                 "-fluid-mass", "--fluid-mass",
                 "Drop the fluid mass term (rho'_F = 0), for experiments.");
  args.AddOption(&diagnostics, "-diag", "--diagnostics", "-no-diag",
                 "--no-diagnostics",
                 "Print rigid-mode residuals and potential-block Ritz values.");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Write a ParaView data collection.");
  args.Parse();
  if (!args.Good()) {
    if (Mpi::Root()) args.PrintUsage(std::cout);
    return 1;
  }
  if (Mpi::Root()) args.PrintOptions(std::cout);

  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  inner_core = smesh.attributes.Max() == 4;
  if (smesh.attributes.Max() != 3 && smesh.attributes.Max() != 4) {
    if (Mpi::Root()) {
      std::cerr << "Expected a two-layer (3 attributes) or three-layer "
                   "(4 attributes) mesh.\n";
    }
    return 1;
  }
  ParMesh parent(MPI_COMM_WORLD, smesh);
  smesh.Clear();
  if (Mpi::Root()) {
    std::cout << (inner_core ? "Three-layer" : "Two-layer") << " model, "
              << dim << "-D, " << (solid_core ? "solid" : "fluid")
              << " outer core, " << Mpi::WorldSize() << " ranks\n";
    ND.Print();
  }

  Array<int> solid_attrs = SolidAttributes();
  ParSubMesh solid(ParSubMesh::CreateFromDomain(parent, solid_attrs));
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes_u(&solid, &fec, dim), fes_phi(&parent, &fec);
  // (collective, so outside the root-only block)
  const HYPRE_BigInt n_u = fes_u.GlobalTrueVSize();
  const HYPRE_BigInt n_phi = fes_phi.GlobalTrueVSize();
  if (Mpi::Root()) {
    std::cout << "Displacement unknowns: " << n_u
              << ", potential unknowns: " << n_phi << "\n";
  }

  FunctionCoefficient rho(Density), rho_f(FluidDensity), kappa(BulkModulus),
      mu(ShearModulus), sigma(SurfaceLoad), psi(TidalPotential);
  ConstantCoefficient zero_gradient(0.0);
  auto rheology = IsotropicElasticRheology(dim, kappa, mu);
  std::vector<FluidRegion> fluids;
  if (!solid_core) {
    FluidRegion f;
    f.attributes = Array<int>({FluidAttribute()});
    f.density = &rho_f;
    if (no_fluid_mass) {
      f.density_gradient = &zero_gradient;
    }
    f.interface_marker = InterfaceMarker(solid);
    fluids.push_back(f);
  }

  SelfGravitatingElasticProblem problem(&fes_u, &fes_phi, rheology, rho, 1.0,
                                        dtn_degree, nullptr, fluids);
  if (load_factor != 0.0) {
    problem.SetSurfaceLoad(sigma, SurfaceMarker(solid));
  }
  if (tidal_amplitude != 0.0) {
    problem.SetTidalPotential(psi);
  }
  if (inner_core && !solid_core) {
    problem.AddRegionRotations(Array<int>({InnerCoreAttribute()}));
  }
  problem.SetRelTol(rel_tol);
  const real_t phi0_norm = L2Norm(problem.BackgroundPotential());
  if (Mpi::Root()) {
    std::cout.precision(10);
    std::cout << "||Phi0|| = " << phi0_norm << "\n";
  }

  if (diagnostics) {
    const auto res = problem.RigidModeResiduals();
    real_t hi = 0.0;
    const real_t lo = problem.PotentialBlockMinEigenvalue(40, &hi);
    if (Mpi::Root()) {
      std::cout << "Rigid-mode residuals:";
      for (auto r : res) {
        std::cout << " " << r;
      }
      std::cout << "\nPotential block Ritz values: " << lo << " .. " << hi
                << (lo > 0.0 ? "" : "  (INDEFINITE)") << "\n";
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
    ParGridFunction du(schur->u), dphi(schur->phi);
    du -= minres->u;
    dphi -= minres->phi;
    const real_t eu = L2Norm(du) / L2Norm(minres->u);
    const real_t ephi = L2Norm(dphi) / L2Norm(minres->phi);
    if (Mpi::Root()) {
      std::cout << "Relative difference Schur vs MINRES: u " << eu << ", phi "
                << ephi << "\n";
    }
  }

  const Result& r = minres ? *minres : *schur;
  ParGridFunction u_dim(r.u);
  ND.UnscaleDisplacement(u_dim);
  real_t umax = u_dim.Normlinf();
  MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                MPI_COMM_WORLD);
  if (Mpi::Root()) {
    std::cout << "Max displacement: " << umax << " m\n";
  }

  if (paraview) {
    ParaViewDataCollection dc("elastogravity_layered_p", &solid);
    dc.SetPrefixPath("ParaView");
    dc.SetLevelsOfDetail(order);
    dc.SetHighOrderOutput(true);
    problem.RegisterFields(dc);
    dc.SetCycle(0);
    dc.SetTime(0.0);
    dc.Save();
  }
  return 0;
}
