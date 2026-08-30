// ============================================================================
// quasi_static_elasticity.cpp
//
// Driver for the quasi-static linear elastic problems defined in elastic.hpp,
// exercising the AssembleForce / AddForce / Solve protocol over a sequence of
// times. See elastic.hpp for the interface contract.
//
// Sample runs:
//    ./quasi_static_elasticity -m ../data/star.mesh -o 2 -r 2
//    ./quasi_static_elasticity -m ../data/star.mesh -o 2 -r 2 -inc
//    ./quasi_static_elasticity -m ../data/beam-quad.mesh -p 1 -o 2 -r 1
// ============================================================================

#include <fstream>
#include <iostream>
#include <memory>

#include "elastic.hpp"

/*----------------------------------------------------------------------------
  Driver
----------------------------------------------------------------------------*/

using namespace std;
using namespace mfem;

int main(int argc, char* argv[]) {
  // Set the default options.
  const char* mesh_file = "../data/star.mesh";
  int order = 1;
  int ref_levels = 0;
  int problem_type = 0;
  real_t t_final = 1.0;
  int n_steps = 10;
  bool demo_increment = false;
  bool paraview = true;
  bool visualization = true;

  // Read in command line options and process.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "Number of uniform mesh refinements.");
  args.AddOption(&problem_type, "-p", "--problem",
                 "Problem type: 0 = pure traction (any mesh), 1 = clamped "
                 "(needs two boundary attributes, e.g. beam-quad.mesh).");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
  args.AddOption(&n_steps, "-n", "--n-steps", "Number of time steps.");
  args.AddOption(&demo_increment, "-inc", "--increment", "-no-inc",
                 "--no-increment",
                 "Superpose an extra body force through AddForce() to "
                 "demonstrate the increment protocol.");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Save time slices to a ParaView data collection.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Send the final solution to a running GLVis server.");
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

  // Construct the requested problem behind the common interface.
  unique_ptr<QuasiStaticLinearElasticProblem> problem;
  if (problem_type == 0) {
    problem = make_unique<TractionProblem>(&mesh, order);
  } else if (problem_type == 1) {
    problem = make_unique<ClampedProblem>(&mesh, order);
  } else {
    cerr << "Unknown problem type: " << problem_type << "\n";
    return 1;
  }
  cout << "Displacement unknowns: "
       << problem->DisplacementSpace().GetTrueVSize() << "\n";

  // Optional demonstration of the AddForce() protocol: any dual vector
  // assembled against DisplacementSpace() may be superposed on the external
  // load. In the viscoelastic layer this slot will carry the effective
  // internal-variable force B^T(2 mu m).
  unique_ptr<VectorConstantCoefficient> extra_coef;
  unique_ptr<LinearForm> extra;
  if (demo_increment) {
    Vector g(dim);
    g = 0.0;
    g[0] = 0.1;
    extra_coef = make_unique<VectorConstantCoefficient>(g);
    extra = make_unique<LinearForm>(&problem->DisplacementSpace());
    extra->AddDomainIntegrator(new VectorDomainLFIntegrator(*extra_coef));
    extra->Assemble();
  }

  // Time slices are written through the fields the problem registers.
  ParaViewDataCollection dc("quasi_static", &mesh);
  if (paraview) {
    dc.SetPrefixPath("ParaView");
    dc.SetLevelsOfDetail(order);
    dc.SetDataFormat(VTKFormat::BINARY);
    dc.SetHighOrderOutput(true);
    problem->RegisterFields(dc);
  }

  // March through time: reset the forcing, superpose increments, solve.
  const real_t dt = t_final / n_steps;
  for (int step = 0; step <= n_steps; step++) {
    const real_t t = step * dt;
    cout << "\nstep " << step << ", t = " << t << "\n";

    problem->AssembleForce(t);
    if (extra) {
      problem->AddForce(*extra);
    }
    if (!problem->Solve()) {
      cerr << "Linear solver failed at t = " << t << "\n";
      return 2;
    }

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
