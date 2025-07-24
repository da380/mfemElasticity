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

  // Set up the finite element space.
  auto fec = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &fec, dim);
  cout << "Number of finite element unknowns: " << fes.GetTrueVSize() << endl;

  // Set up the constant traction vector coefficient.
  auto tv = Vector(dim);
  tv = 0.0;
  tv[0] = 1;
  auto tc = VectorConstantCoefficient(tv);

  // Set up the linear form.
  auto b = LinearForm(&fes);
  b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(tc));
  b.Assemble();

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
