

/*********************************************************************************

Solves a static elastic  problem, with a constant boundary traction applied to
the mesh's external boundary. This code illustrates the use of the
RigidBodySolver class to project out the null space for the problem.

Options:

[-m, --mesh]: The mesh. Either 2D or 3D. Tractions are applied to its external
              boundary. Default is star.mesh in the data directory.

[-o, --order]: The polynomial order used in the calculations. Default is 1.

[-r, --refinement]: The number of times to refine the mesh. Default it 0.

*********************************************************************************/

#include <cmath>
#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char* argv[]) {
  // Set the default options.
  const char* mesh_file = "../data/star.mesh";
  int order = 1;
  int ref_levels = 0;

  // Read in command line options and process.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "number of mesh refinements");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // Read in the mesh and refine if requested.
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();
  for (int l = 0; l < ref_levels; l++) {
    mesh.UniformRefinement();
  }

  mesh.attributes.Print(cout);
  mesh.bdr_attributes.Print(cout);

  // Set up the finite element space.
  auto fec = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &fec, dim);
  cout << "Number of finite element unknowns: " << fes.GetTrueVSize() << endl;

  // Set up the linear form.
  auto b = LinearForm(&fes);
  auto marker = mfemElasticity::ExternalBoundaryMarker(&mesh);
  marker[0] = 0;
  marker[1] = 1;
  auto sigma = FunctionCoefficient([](const Vector& x) { return x[0] * x[1]; });
  b.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(sigma), marker);
  b.Assemble();

  marker.Print(cout);

  // Set up the bilinear form
  auto lambda = ConstantCoefficient(1);
  auto mu = ConstantCoefficient(1);
  auto a = BilinearForm(&fes);
  a.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
  a.Assemble();

  // Set up the gridfunction.
  auto x = GridFunction(&fes);
  x = 0.0;

  // Set the linear system.
  Array<int> ess_tdof_list;
  SparseMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
  cout << "Size of linear system: " << A.Height() << endl;

  // Set the preconditioner.
  GSSmoother M(A);

  // Set the solver.
  auto solver = CGSolver();
  solver.SetPreconditioner(M);
  solver.SetOperator(A);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  // Set up the rigid body solver.
  auto rigidSolver = mfemElasticity::RigidBodySolver(&fes);
  rigidSolver.SetSolver(solver);

  // Solve the equations.
  rigidSolver.Mult(B, X);
  a.RecoverFEMSolution(X, b, x);

  // =====================================================================
  // POST-PROCESSING: Calculate and remove the physical centroid shift
  // =====================================================================
  cout << "Calculating physical centroid shift..." << endl;

  // Build a mass matrix to compute actual volume integrals (L2 projection)
  BilinearForm m(&fes);
  m.AddDomainIntegrator(new VectorMassIntegrator());
  m.Assemble();

  Vector Mx(fes.GetVSize());
  m.Mult(x, Mx);

  Vector shift(dim);

  for (int d = 0; d < dim; d++) {
    // Create a constant vector field (1.0 in direction d, 0.0 elsewhere)
    Vector dir(dim);
    dir = 0.0;
    dir(d) = 1.0;
    VectorConstantCoefficient dir_coeff(dir);

    GridFunction const_vec(&fes);
    const_vec.ProjectCoefficient(dir_coeff);

    // Compute the volume integral of displacement: \int x_d d\Omega
    double int_x = const_vec * Mx;

    // Compute the total volume of the domain: \int 1 d\Omega
    Vector M_const(fes.GetVSize());
    m.Mult(const_vec, M_const);
    double volume = const_vec * M_const;

    // The physical shift is the volume-averaged displacement
    shift(d) = int_x / volume;
  }

  cout << "Calculated Shift (x, y): " << shift(0) << ", " << shift(1) << endl;

  // Subtract the calculated shift from the entire solution field
  for (int d = 0; d < dim; d++) {
    Vector dir(dim);
    dir = 0.0;
    dir(d) = -shift(d);
    VectorConstantCoefficient shift_coeff(dir);

    GridFunction shift_gf(&fes);
    shift_gf.ProjectCoefficient(shift_coeff);

    x += shift_gf;
  }
  // =====================================================================

  // Write solution to file.
  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);
  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  x.Save(sol_ofs);

  // Visualise if glvis is open.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock.precision(8);
  sol_sock << "solution\n" << mesh << x << flush;
  if (dim == 2) {
    sol_sock << "keys Rjlvvvvvmm\n" << flush;
  } else {
    sol_sock << "keys m\n" << flush;
  }

  return 0;
}