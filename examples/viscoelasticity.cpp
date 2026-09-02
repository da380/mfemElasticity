// ============================================================================
// viscoelasticity.cpp
//
// Quasi-static generalised Maxwell viscoelasticity with the library's
// ViscoelasticOperator on top of a QuasiStaticLinearElasticProblem
// (mfemElasticity/elastic_problem.hpp, viscoelastic.hpp).
//
// The rheology is a Maxwell body (mu_inf = 0, one branch) or, with -mu-inf,
// a standard linear solid (mu_inf > 0, one branch). The elastic problem is
// assembled with the unrelaxed modulus from the same rheology object the
// viscoelastic operator reads its branch data from.
//
// Time integrators (-s): exponential trapezoid (default; second order, one
// solve per step, no step restriction), exponential Euler, backward Euler
// and SDIRK23 through MFEM's implicit solvers, and explicit RK4 / forward
// Euler (stable only for dt < ~2.8 tau_min).
//
// Sample runs:
//    ./viscoelasticity -m ../data/star.mesh -o 2 -r 2
//    ./viscoelasticity -m ../data/star.mesh -o 2 -r 2 -s 4 -n 200
//    ./viscoelasticity -m ../data/beam-quad.mesh -p 1 -o 2 -r 1 -tau 0.5
//    ./viscoelasticity -m ../data/beam-quad.mesh -p 1 -mu-inf 0.5 -tf 20
// ============================================================================

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

int main(int argc, char* argv[]) {
  // Set the default options.
  const char* mesh_file = "../data/star.mesh";
  int order = 2;
  int m_order = -1;  // internal-variable order; < 0 means order - 1
  int ref_levels = 1;
  int problem_type = 0;
  int solver_type = 0;
  int map_type = 0;
  real_t t_final = 5.0;
  int n_steps = 50;
  real_t tau0 = 1.0;
  real_t mu_inf0 = 0.0;
  bool paraview = true;
  bool visualization = true;

  // Read in command line options and process.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order for the displacement.");
  args.AddOption(&m_order, "-mo", "--m-order",
                 "Order of the internal-variable space (< 0: order - 1).");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "Number of uniform mesh refinements.");
  args.AddOption(&problem_type, "-p", "--problem",
                 "Problem type: 0 = pure traction (any mesh), 1 = clamped "
                 "(needs two boundary attributes, e.g. beam-quad.mesh).");
  args.AddOption(&solver_type, "-s", "--solver",
                 "Time integrator: 0 = exponential trapezoid, 1 = exponential "
                 "Euler, 2 = backward Euler, 3 = SDIRK23, 4 = RK4, "
                 "5 = forward Euler.");
  args.AddOption(&map_type, "-map", "--strain-map",
                 "Strain map: 0 = Galerkin (M^{-1} B), 1 = interpolation.");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
  args.AddOption(&n_steps, "-n", "--n-steps", "Number of time steps.");
  args.AddOption(&tau0, "-tau", "--relaxation-time",
                 "Maxwell relaxation time tau = eta / mu.");
  args.AddOption(&mu_inf0, "-mu-inf", "--long-term-modulus",
                 "Long-term shear modulus (0: Maxwell body).");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Save time slices to a ParaView data collection.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Send the final displacement to a running GLVis server.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // Read in the mesh and refine if requested.
  Mesh mesh(mesh_file, 1, 1);
  const int dim = mesh.Dimension();
  for (int l = 0; l < ref_levels; l++) {
    mesh.UniformRefinement();
  }

  // Displacement space and material. kappa = 1 + 2/d so that the unrelaxed
  // state has lambda = mu = 1 when mu_inf + mu_1 = 1.
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);
  ConstantCoefficient kappa(1.0 + 2.0 / dim), mu_inf(mu_inf0),
      mu1(1.0 - mu_inf0), tau(tau0);
  std::vector<MaxwellBranch> branches{{&mu1, &tau}};
  GeneralisedMaxwellRheology rheology(dim, kappa, mu_inf, branches);

  // Loads. Problem 0: a time-scaled uniform traction t -> (0, 1 + t, ...)
  // on all external boundaries. Problem 1: boundary attribute 1 clamped,
  // a time-scaled pull t -> (0, ..., -0.05 (1 + t)) on attribute 2.
  VectorFunctionCoefficient traction(
      dim, [problem_type](const Vector& /*x*/, real_t t, Vector& f) {
        f = 0.0;
        if (problem_type == 0) {
          f[1] = 1.0 + t;
        } else {
          f[f.Size() - 1] = -0.05 * (1.0 + t);
        }
      });
  Array<int> marker(mesh.bdr_attributes.Max()), ess_bdr;
  marker = 0;

  unique_ptr<ElasticProblemBase> problem;
  if (problem_type == 0) {
    mesh.MarkExternalBoundaries(marker);
    problem = make_unique<TractionProblem>(&fes, rheology, traction, marker);
  } else if (problem_type == 1) {
    MFEM_VERIFY(mesh.bdr_attributes.Max() >= 2,
                "Problem 1 needs boundary attributes 1 (clamped) and 2 "
                "(traction), e.g. data/beam-quad.mesh.");
    ess_bdr.SetSize(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;
    marker[1] = 1;
    problem =
        make_unique<ClampedProblem>(&fes, rheology, ess_bdr, traction, marker);
  } else {
    cerr << "Unknown problem type: " << problem_type << "\n";
    return 1;
  }
  problem->SetPrintLevel(IterativeSolver::PrintLevel().Summary());

  const auto map = map_type == 0
                       ? ViscoelasticOperator::StrainMap::Galerkin
                       : ViscoelasticOperator::StrainMap::Interpolation;
  ViscoelasticOperator visco(*problem, m_order, map);
  cout << "Displacement unknowns:      " << fes.GetTrueVSize() << "\n"
       << "Internal-variable unknowns: " << visco.Height() << "\n";

  // Select the time integrator.
  unique_ptr<ODESolver> ode;
  switch (solver_type) {
    case 0:
      ode = make_unique<ExponentialTrapezoidSolver>();
      break;
    case 1:
      ode = make_unique<ExponentialEulerSolver>();
      break;
    case 2:
      ode = make_unique<BackwardEulerSolver>();
      break;
    case 3:
      ode = make_unique<SDIRK23Solver>();
      break;
    case 4:
      ode = make_unique<RK4Solver>();
      break;
    case 5:
      ode = make_unique<ForwardEulerSolver>();
      break;
    default:
      cerr << "Unknown solver type: " << solver_type << "\n";
      return 1;
  }
  ode->Init(visco);

  real_t t = 0.0;
  real_t dt = t_final / n_steps;
  Vector m(visco.Height());
  m = 0.0;

  if (solver_type >= 4 && dt > 2.5 * visco.MinRelaxationTime()) {
    cout << "Warning: dt = " << dt
         << " exceeds the explicit stability limit of roughly 2.8 tau_min = "
         << 2.8 * visco.MinRelaxationTime()
         << ". Expect blow-up; use an implicit or exponential integrator.\n";
  }

  // Time slices are written through the fields the operator registers.
  ParaViewDataCollection dc("viscoelastic", &mesh);
  if (paraview) {
    dc.SetPrefixPath("ParaView");
    dc.SetLevelsOfDetail(order);
    dc.SetDataFormat(VTKFormat::BINARY);
    dc.SetHighOrderOutput(true);
    visco.RegisterFields(dc);
  }

  // Initial state: relaxed internal variable, elastic response at t = 0.
  if (!visco.SolveElastic(m, t)) {
    cerr << "Elastic solve failed at t = " << t << "\n";
    return 2;
  }
  visco.SyncFields(m);
  if (paraview) {
    dc.SetCycle(0);
    dc.SetTime(t);
    dc.Save();
  }

  // March through time. SolveElastic() makes (u, m) consistent for output;
  // it is free after a trapezoid or implicit step and costs one solve after
  // an explicit or exponential-Euler one.
  for (int step = 1; step <= n_steps; step++) {
    ode->Step(m, t, dt);

    if (!visco.SolveElastic(m, t)) {
      cerr << "Elastic solve failed at t = " << t << "\n";
      return 2;
    }
    visco.SyncFields(m);
    cout << "step " << step << ", t = " << t << ", ||m||_2 = " << m.Norml2()
         << "\n";

    if (paraview) {
      dc.SetCycle(step);
      dc.SetTime(t);
      dc.Save();
    }
  }

  // Save the final state in MFEM's native format.
  {
    ofstream mesh_ofs("refined.mesh");
    mesh_ofs.precision(8);
    mesh.Print(mesh_ofs);
    ofstream sol_ofs("sol.gf");
    sol_ofs.precision(8);
    problem->Displacement().Save(sol_ofs);
  }

  // Visualise if glvis is open.
  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n";
    mesh.Print(sol_sock);
    problem->Displacement().Save(sol_sock);
    sol_sock << flush;
    if (dim == 2) {
      sol_sock << "keys Rjlvvvvvmm\n" << flush;
    } else {
      sol_sock << "keys m\n" << flush;
    }
  }

  return 0;
}
