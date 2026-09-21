// ============================================================================
// self_gravitating_relaxation.cpp
//
// Viscoelastic relaxation of a self-gravitating layered Earth model with a
// fluid outer core: the layered models of layered_model.hpp (two- or
// three-layer meshes, see elastogravity_layered.cpp), a Maxwell mantle with
// a given viscosity, an elastic inner core (a CompositeRheology: the inner
// core carries an elastic rheology, the mantle a Maxwell one), and a surface
// mass load switched on at t = 0 (a Heaviside load: the elastic response is
// followed by the viscous relaxation towards isostasy). ViscoelasticOperator
// runs on LinearQuasiStaticSelfGravitatingProblem unchanged; the potential
// and the fluid core come along for free.
//
// Time is measured in Maxwell times of the mantle, tau = eta / mu evaluated
// with the mantle's mean shear modulus; the run prints, at every step, the
// L2 norms of the displacement and of the potential perturbation and the
// radial surface displacement under the load's maximum (theta = 0).
//
// Sample runs:
//    ./self_gravitating_relaxation -o 2 -n 20 -tf 5
//    ./self_gravitating_relaxation -m ../data/elastogravity_three_layer_2d.msh
//    -o 2
//    ./self_gravitating_relaxation -m ../data/elastogravity_three_layer_3d.msh
//    -o 1 -n 10
//    ./self_gravitating_relaxation -o 2 -rtol 1e-3
//    ./self_gravitating_relaxation -o 2 -eta 3e21 -pv
// ============================================================================

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

#include "layered_model.hpp"
#include "mfemElasticity.hpp"

using namespace mfem;
using namespace mfemElasticity;
using namespace layered;

int main(int argc, char* argv[]) {
  const char* mesh_file = "../data/elastogravity_two_layer_2d.msh";
  int order = 1;
  int dtn_degree = 16;
  real_t rel_tol = 1e-9;
  real_t eta_dim = 1e21;  // mantle viscosity [Pa s]
  real_t t_final = 5.0;   // in Maxwell times
  int n_steps = 20;
  real_t rtol = 0.0;
  bool paraview = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Two- or three-layer mesh.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&dtn_degree, "-deg", "--dtn-degree",
                 "Truncation degree of the DtN expansion.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative solver tolerance.");
  args.AddOption(&eta_dim, "-eta", "--viscosity", "Mantle viscosity [Pa s].");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time [Maxwell times].");
  args.AddOption(&n_steps, "-n", "--n-steps",
                 "Number of steps (output times when adaptive).");
  args.AddOption(&rtol, "-rtol", "--adaptive-rtol",
                 "Relative tolerance of adaptive stepping (0: fixed dt).");
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
  Array<int> solid_attrs = SolidAttributes();
  SubMesh solid(SubMesh::CreateFromDomain(parent, solid_attrs));
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_u(&solid, &fec, dim), fes_phi(&parent, &fec);

  // Material. The Maxwell time of the mantle from its mean shear modulus
  // (the moduli vary with radius, so tau varies too; the reported time unit
  // uses the mean). With an inner core the rheology is a composite: elastic
  // in the inner core, Maxwell in the mantle (the same kappa and mu
  // coefficients serve both; each region reads its own radii).
  const real_t mu_mantle_dim = inner_core ? 0.5 * (294e9 + 68e9)
                                          : 0.5 * (280e9 + 70e9);
  const real_t tau_dim = eta_dim / mu_mantle_dim;
  const real_t tau_nd = tau_dim / ND.Time();
  FunctionCoefficient rho(Density), rho_f(FluidDensity), kappa(BulkModulus),
      mu(ShearModulus), sigma(SurfaceLoad);
  ConstantCoefficient tau(tau_nd);
  auto mantle = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  IsotropicElasticRheology core(dim, kappa, mu);
  std::vector<RheologyRegion> regions;
  {
    Array<int> mantle_marker(solid.attributes.Max()), core_marker;
    mantle_marker = 0;
    mantle_marker[MantleAttribute() - 1] = 1;
    regions.push_back({mantle_marker, &mantle, "mantle"});
    if (inner_core) {
      core_marker.SetSize(solid.attributes.Max());
      core_marker = 0;
      core_marker[InnerCoreAttribute() - 1] = 1;
      regions.push_back({core_marker, &core, "inner_core"});
    }
  }
  CompositeRheology rheology(dim, regions);
  std::vector<FluidRegion> fluids;
  {
    FluidRegion f;
    f.attributes = Array<int>({FluidAttribute()});
    f.density = &rho_f;
    f.interface_marker = InterfaceMarker(solid);
    fluids.push_back(f);
  }
  LinearQuasiStaticSelfGravitatingProblem problem(
      &fes_u, &fes_phi, rheology, rho, 1.0, dtn_degree, nullptr, fluids);
  // A Heaviside load: the surface load coefficient is constant in time, so
  // switching it on at t = 0 is simply starting from an unloaded state.
  problem.SetSurfaceLoad(sigma, SurfaceMarker(solid));
  if (inner_core) {
    problem.AddRegionRotations(Array<int>({InnerCoreAttribute()}));
  }
  problem.SetRelTol(rel_tol);
  std::cout << (inner_core ? "Three-layer" : "Two-layer") << " model, " << dim
            << "-D; mantle Maxwell time " << tau_dim / (365.25 * 86400.0)
            << " yr (" << tau_nd << " time units)\n";

  ViscoelasticOperator visco(problem);
  ExponentialTrapezoidSolver ode;
  ode.Init(visco);
  AdaptiveExponentialTrapezoidSolver adaptive;
  if (rtol > 0.0) {
    adaptive.Init(visco);
    adaptive.SetTolerances(rtol, 1e-14);
  }

  // Observation point: the surface vertex nearest to the pole (theta = 0).
  int pole = -1;
  {
    real_t best = -1.0;
    for (int v = 0; v < solid.GetNV(); v++) {
      const real_t* x = solid.GetVertex(v);
      const real_t z = x[dim - 1];
      if (z > best) {
        best = z;
        pole = v;
      }
    }
  }
  auto radial_at_pole = [&]() {
    // Order-1 vertex dof = vertex index; for higher orders too, since the
    // vertex dofs come first in H1 spaces.
    const GridFunction& u = problem.Displacement();
    return u[fes_u.DofToVDof(pole, dim - 1)] * ND.Length();
  };

  ParaViewDataCollection dc("self_gravitating_relaxation", &solid);
  if (paraview) {
    dc.SetPrefixPath("ParaView");
    dc.SetLevelsOfDetail(order);
    dc.SetHighOrderOutput(true);
    visco.RegisterFields(dc);
  }

  Vector m(visco.Height());
  m = 0.0;
  real_t t = 0.0;
  real_t dt = t_final * tau_nd / n_steps;
  real_t dt_adaptive = 0.1 * dt;

  // The elastic response at t = 0+.
  if (!visco.SolveElastic(m, t)) {
    std::cerr << "Elastic solve failed.\n";
    return 2;
  }
  std::cout.precision(6);
  std::cout << "t/tau        ||u||        ||phi||   u_r(pole) [m]\n";
  auto report = [&](int cycle) {
    visco.SyncFields(m);
    std::cout << std::setw(6) << t / tau_nd << std::setw(14)
              << L2Norm(problem.Displacement()) << std::setw(14)
              << L2Norm(problem.Potential()) << std::setw(14)
              << radial_at_pole() << "\n";
    if (paraview) {
      dc.SetCycle(cycle);
      dc.SetTime(t / tau_nd);
      dc.Save();
    }
  };
  report(0);
  const auto w0 = std::chrono::steady_clock::now();
  for (int step = 1; step <= n_steps; step++) {
    if (rtol > 0.0) {
      adaptive.Integrate(m, t, step * t_final * tau_nd / n_steps, dt_adaptive);
    } else {
      ode.Step(m, t, dt);
    }
    if (!visco.SolveElastic(m, t)) {
      std::cerr << "Elastic solve failed at t = " << t << "\n";
      return 2;
    }
    report(step);
  }
  const auto w1 = std::chrono::steady_clock::now();
  std::cout << "Solves " << problem.NumSolves() << ", assemblies "
            << problem.NumAssemblies() << ", preconditioner setups "
            << problem.NumPreconditionerSetups() << ", "
            << std::chrono::duration<double>(w1 - w0).count() << " s";
  if (rtol > 0.0) {
    std::cout << "; adaptive steps " << adaptive.NumAcceptedSteps()
              << " accepted, " << adaptive.NumRejectedSteps() << " rejected";
  }
  std::cout << "\n";
  return 0;
}
