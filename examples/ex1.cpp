

#include <cmath>
#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char* argv[]) {
  // 1. Parse command-line options.
  const char* mesh_file = "../data/star.mesh";
  int order = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();

  {
    int ref_levels = (int)floor(log((dim == 2 ? 1000 : 10000) / mesh.GetNE()) /
                                log(2.) / dim);
    for (int l = 0; l < ref_levels; l++) {
      mesh.UniformRefinement();
    }
  }

  auto fec = H1_FECollection(order, dim);
  auto fespace = FiniteElementSpace(&mesh, &fec, dim);
  cout << "Number of finite element unknowns: " << fespace.GetTrueVSize()
       << endl;

  auto x = GridFunction(&fespace);
  x = 0.0;

  // Set the essential boundary conditions to a null list.
  Array<int> ess_tdof_list;

  // Set up the linear form.
  auto f = FunctionCoefficient([](const Vector& x) {
    auto x0 = Vector{0.0, 2.0, 0.0};
    x0.Add(-1, x);
    auto r = x0.Norml2();
    return -exp(-10 * r * r);
  });
  auto b = LinearForm(&fespace);
  b.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(f));
  b.Assemble();

  // Set up the bilinear form
  auto lambda = FunctionCoefficient([](const Vector& x) { return 1; });
  auto mu = FunctionCoefficient([](const Vector& x) { return 1; });
  auto a = BilinearForm(&fespace);
  a.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
  a.Assemble();

  SparseMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
  cout << "Size of linear system: " << A.Height() << endl;

  GSSmoother M(A);
  auto solver = CGSolver();
  solver.SetPreconditioner(M);
  solver.SetOperator(A);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  auto rigidSolver = mfemElasticity::RigidBodySolver(&fespace);
  rigidSolver.SetSolver(solver);
  rigidSolver.Mult(B, X);

  a.RecoverFEMSolution(X, b, x);

  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);
  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  x.Save(sol_ofs);

  return 0;
}
