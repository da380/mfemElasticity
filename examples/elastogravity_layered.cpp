// ============================================================================
// elastogravity_layered.cpp
//
// Self-gravitating elastic deformation of a layered Earth-like model with a
// fluid outer core, as a driver of SelfGravitatingElasticProblem
// (mfemElasticity/self_gravitating.hpp) with FluidRegion. Replaces the
// hand-assembled elastogravity_two_layer / elastogravity_three_layer
// examples; the models, meshes and loads are those of layered_model.hpp.
//
// The solid regions (mantle, and the inner core when the mesh has one) form
// ONE displacement SubMesh, disconnected for the three-layer model; the
// fluid outer core enters through a FluidRegion (its density, the
// hydrostatic Poisson term and the fluid–solid interface terms). The
// inner core's near-null rotations are projected out.
//
// Meshes (../data): elastogravity_two_layer_2d.msh,
// elastogravity_three_layer_2d.msh, elastogravity_three_layer_3d.msh.
//
// Sample runs:
//    ./elastogravity_layered -o 2
//    ./elastogravity_layered -m ../data/elastogravity_three_layer_2d.msh -o 2 -s 2 -diag
//    ./elastogravity_layered -m ../data/elastogravity_three_layer_3d.msh -o 1
//    ./elastogravity_layered -o 2 -load 0 -tidal 1.0
//    ./elastogravity_layered -o 2 -solid-core
//    ./elastogravity_layered -m ../data/elastogravity_three_layer_3d.msh -o 1 -no-fluid-mass
//
// -s 2 runs both solvers and reports their difference (they solve the same
// restricted system and agree to solver tolerance). -diag prints the
// rigid-mode residuals (global modes, then the inner core's rotations) and
// the extreme Ritz values of the potential block. -solid-core treats the
// outer core as a solid (constant moduli), for comparison with the fluid
// physics. -tidal A applies the degree-2 tidal potential A (r/a)^2 P_2 with
// A in m^2/s^2 (with -load 0 it is the only forcing). -no-fluid-mass drops
// the hydrostatic Poisson term rho'_F phi (the fluid becomes unstratified in
// the Eulerian sense); for the PREM-like core it changes the response by a
// factor of about three, the potential block sitting at about half its
// positivity margin (doc/fluid_solid_design.md section 3.2).
// ============================================================================

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
  GridFunction u, phi;
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
  Result r{p.Displacement(), p.Potential(), p.LastOuterIterations(),
           p.LastInnerIterations(),
           std::chrono::duration<double>(t1 - t0).count()};
  const char* name =
      type == SelfGravitatingElasticProblem::SolverType::SchurCG
          ? "Schur CG    "
          : "Block MINRES";
  std::cout.precision(10);
  std::cout << name << ": " << (ok ? "converged" : "FAILED") << ", outer "
            << r.outer << ", inner " << r.inner << ", " << r.seconds
            << " s, ||u|| = " << L2Norm(r.u) << ", ||phi|| = " << L2Norm(r.phi)
            << "\n";
  return r;
}

}  // namespace

int main(int argc, char* argv[]) {
  const char* mesh_file = "../data/elastogravity_two_layer_2d.msh";
  int order = 1;
  int solver = 1;
  int dtn_degree = 16;
  real_t rel_tol = 1e-10;
  bool diagnostics = false;
  bool visualization = false;
  bool paraview = false;
  bool no_fluid_mass = false;

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
  args.AddOption(&diagnostics, "-diag", "--diagnostics", "-no-diag",
                 "--no-diagnostics",
                 "Print rigid-mode residuals and potential-block Ritz values.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "GLVis visualisation.");
  args.AddOption(&no_fluid_mass, "-no-fluid-mass", "--no-fluid-mass",
                 "-fluid-mass", "--fluid-mass",
                 "Drop the fluid mass term (rho'_F = 0), for experiments.");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Write a ParaView data collection.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(std::cout);
    return 1;
  }
  args.PrintOptions(std::cout);

  Mesh parent(mesh_file, 1, 1);
  const int dim = parent.Dimension();
  inner_core = parent.attributes.Max() == 4;
  if (parent.attributes.Max() != 3 && parent.attributes.Max() != 4) {
    std::cerr << "Expected a two-layer (3 attributes) or three-layer "
                 "(4 attributes) mesh.\n";
    return 1;
  }
  std::cout << (inner_core ? "Three-layer" : "Two-layer") << " model, "
            << dim << "-D, " << (solid_core ? "solid" : "fluid")
            << " outer core\n";
  ND.Print();

  Array<int> solid_attrs = SolidAttributes();
  SubMesh solid(SubMesh::CreateFromDomain(parent, solid_attrs));
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_u(&solid, &fec, dim), fes_phi(&parent, &fec);
  std::cout << "Displacement unknowns: " << fes_u.GetTrueVSize()
            << ", potential unknowns: " << fes_phi.GetTrueVSize() << "\n";

  FunctionCoefficient rho(Density), rho_f(FluidDensity), kappa(BulkModulus),
      mu(ShearModulus), sigma(SurfaceLoad), psi(TidalPotential);
  auto rheology = IsotropicMaxwellRheology::Elastic(dim, kappa, mu);
  std::vector<FluidRegion> fluids;
  ConstantCoefficient zero_gradient(0.0);
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
  std::cout.precision(10);
  std::cout << "||Phi0|| = " << L2Norm(problem.BackgroundPotential()) << "\n";

  if (diagnostics) {
    const auto res = problem.RigidModeResiduals();
    std::cout << "Rigid-mode residuals:";
    for (auto r : res) {
      std::cout << " " << r;
    }
    std::cout << "\n";
    real_t hi = 0.0;
    const real_t lo = problem.PotentialBlockMinEigenvalue(40, &hi);
    std::cout << "Potential block Ritz values: " << lo << " .. " << hi
              << (lo > 0.0 ? "" : "  (INDEFINITE)") << "\n";
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
    GridFunction du(schur->u), dphi(schur->phi);
    du -= minres->u;
    dphi -= minres->phi;
    std::cout << "Relative difference Schur vs MINRES: u "
              << L2Norm(du) / L2Norm(minres->u) << ", phi "
              << L2Norm(dphi) / L2Norm(minres->phi) << "\n";
  }

  const Result& r = minres ? *minres : *schur;
  GridFunction u_dim(r.u);
  ND.UnscaleDisplacement(u_dim);
  std::cout << "Max displacement: " << u_dim.Normlinf() << " m\n";

  if (paraview) {
    ParaViewDataCollection dc("elastogravity_layered", &solid);
    dc.SetPrefixPath("ParaView");
    dc.SetLevelsOfDetail(order);
    dc.SetHighOrderOutput(true);
    problem.RegisterFields(dc);
    dc.SetCycle(0);
    dc.SetTime(0.0);
    dc.Save();
  }
  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;
    socketstream u_sock(vishost, visport);
    u_sock.precision(8);
    u_sock << "solution\n"
           << solid << u_dim << "window_title 'Displacement [m]'"
           << (dim == 2 ? "\nkeys Rjlbc\n" : "\nkeys RRRilc\n") << std::flush;
    GridFunction phi_dim(problem.Potential());
    ND.UnscaleGravityPotential(phi_dim);
    socketstream phi_sock(vishost, visport);
    phi_sock.precision(8);
    phi_sock << "solution\n"
             << parent << phi_dim
             << "window_title 'Potential perturbation [m^2/s^2]'"
             << (dim == 2 ? "\nkeys Rjlbc\n" : "\nkeys RRRilc\n")
             << std::flush;
  }
  return 0;
}
