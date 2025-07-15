#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

const real_t G = 1;
const real_t rho = 1;
const real_t radius = 1;
const real_t x00 = 0.5;
const real_t x01 = 0.5;

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *mesh_file =
      "/home/david/dev/meshing/examples/circular_offset.msh";
  // const char *mesh_file =
  // "/home/david/dev/meshing/examples/circular_shell.msh";
  int order = 3;
  int ref_levels = 0;
  int kMax = 16;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&ref_levels, "-r", "--refine", "Number of mesh refinements");
  args.AddOption(&kMax, "-kMax", "--kMax", "Order for Fourier exapansion");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();

  {
    for (int l = 0; l < ref_levels; l++) {
      mesh.UniformRefinement();
    }
  }

  auto fec = H1_FECollection(order, dim);
  auto fespace = FiniteElementSpace(&mesh, &fec);
  cout << "Number of finite element unknowns: " << fespace.GetTrueVSize()
       << endl;

  BilinearForm a(&fespace);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  auto C = PoissonDtNCircle(&fespace, kMax);
  C.Assemble();

  // Set the density.
  auto rho_coefficient =
      FunctionCoefficient([=](const Vector &x) { return rho; });

  // Set up the form on the domain.
  auto domain_marker = Array<int>{1, 0};
  LinearForm b(&fespace);
  b.AddDomainIntegrator(new DomainLFIntegrator(rho_coefficient), domain_marker);
  b.Assemble();
  auto mass = b.Sum();

  // And now the form on the boundary.
  auto b2 = LinearForm(&fespace);
  auto one = ConstantCoefficient(1);
  auto boundary_marker = Array<int>{0, 1};
  b2.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), boundary_marker);
  b2.Assemble();
  auto length = b2.Sum();

  // Form the total form.
  b.Add(-mass / length, b2);
  b *= -4 * M_PI * G;

  OperatorPtr A;
  Vector B, X;
  auto x = GridFunction(&fespace);
  x = 0.0;

  Array<int> ess_tdof_list{};
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
  cout << "Size of linear system: " << A->Height() << endl;

  GSSmoother M((SparseMatrix &)(*A));

  auto D = SumOperator(A.Ptr(), 1, &C, 1, false, false);

  auto solver = CGSolver();
  solver.SetOperator(D);
  solver.SetPreconditioner(M);

  solver.SetRelTol(1e-12);
  solver.SetMaxIter(2000);
  solver.SetPrintLevel(1);

  auto orthoSolver = OrthoSolver();
  orthoSolver.SetSolver(solver);
  orthoSolver.Mult(B, X);

  a.RecoverFEMSolution(X, b, x);

  auto phi = FunctionCoefficient([=](const Vector &x) {
    auto r = (x(0) - x00) * (x(0) - x00) + (x(1) - x01) * (x(1) - x01);
    r = sqrt(r);
    if (r < radius) {
      return M_PI * G * rho * r * r;
    } else {
      return 2 * M_PI * G * rho * radius * log(r / radius) +
             M_PI * G * rho * radius * radius;
    }
  });

  auto l = LinearForm(&fespace);
  l.AddDomainIntegrator(new DomainLFIntegrator(one));
  l.Assemble();
  auto area = l.Sum();
  l /= area;

  auto y = GridFunction(&fespace);
  y.ProjectCoefficient(phi);

  auto py = l * y;
  y -= py;

  auto px = l * x;
  x -= px;

  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);

  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  x.Save(sol_ofs);

  ofstream exact_ofs("exact.gf");
  exact_ofs.precision(8);
  y.Save(exact_ofs);

  x -= y;
  ofstream diff_ofs("diff.gf");
  diff_ofs.precision(8);
  x.Save(diff_ofs);

  return 0;
}
