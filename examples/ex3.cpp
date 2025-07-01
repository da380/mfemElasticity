#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

const real_t pi = 3.14159265358979323846264338327950288;
const real_t G = 1;
const real_t rho = 1;
const real_t radius = 1;
const real_t outer_radius = 1.6;
const real_t x00 = 0.5;
const real_t x01 = 0.0;

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *mesh_file =
      "/home/david/dev/meshing/examples/circular_offset.msh";
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
  // auto mesh = Mesh::MakeCartesian2D(10, 10, Element::TRIANGLE);
  int dim = mesh.Dimension();

  {
    int ref_levels = (int)floor(log(10000. / mesh.GetNE()) / log(2.) / dim);
    for (int l = 0; l < ref_levels; l++) {
      mesh.UniformRefinement();
    }
  }

  auto fec = H1_FECollection(order, dim);
  auto fespace = FiniteElementSpace(&mesh, &fec);
  cout << "Number of finite element unknowns: " << fespace.GetTrueVSize()
       << endl;

  Array<int> ess_tdof_list;
  if (mesh.bdr_attributes.Size()) {
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  // Add in the first part of the RHS
  auto rho_coefficient = ConstantCoefficient(rho);
  auto domain_marker = Array<int>{1, 0};
  LinearForm b(&fespace);
  b.AddDomainIntegrator(new DomainLFIntegrator(rho_coefficient), domain_marker);

  // Add in the correction term.
  auto mass = pi * radius * radius * rho;
  auto boundary_coefficient =
      ConstantCoefficient(-mass / (2 * pi * outer_radius));
  auto boundary_marker = Array<int>{0, 1};
  b.AddBoundaryIntegrator(new BoundaryLFIntegrator(boundary_coefficient),
                          boundary_marker);

  // Assemble the RHS.
  b.Assemble();
  b *= -4 * pi * G;

  BilinearForm a(&fespace);
  auto mu = FunctionCoefficient([](const Vector &x) { return 1; });
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  OperatorPtr A;
  Vector B, X;
  auto x = GridFunction(&fespace);
  x = 0.0;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
  cout << "Size of linear system: " << A->Height() << endl;

  auto C = DtN::Poisson2D(&fespace, 8, 0);
  auto D = SumOperator(A.Ptr(), 1, &C, 1, false, false);

  GSSmoother M((SparseMatrix &)(*A));

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

  auto exact = FunctionCoefficient([=](const Vector &x) {
    auto r = (x(0) - x00) * (x(0) - x00) + (x(1) - x01) * (x(1) - x01);
    r = sqrt(r);

    if (r <= radius) {
      return 0.25 * r * r;
    } else {
      return 0.25 * radius * radius * (1 + log(r / radius));
    }
  });

  auto y = GridFunction(&fespace);
  y.ProjectCoefficient(exact);

  // x = y;

  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);
  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  x.Save(sol_ofs);

  return 0;
}
