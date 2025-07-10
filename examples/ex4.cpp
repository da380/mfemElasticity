#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

const real_t G = 1;
const real_t rho = 1;
const real_t radius = 0.5;
const real_t x00 = 0.0;
const real_t x01 = 0.25;
const real_t x02 = 0.0;
constexpr real_t pi = atan(1) * 4;

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *mesh_file =
      "/home/david/dev/meshing/examples/spherical_offset.msh";
  int order = 1;
  int ref_levels = 0;
  int lMax = 4;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&ref_levels, "-r", "--refine", "Number of mesh refinements");
  args.AddOption(&lMax, "-lMax", "--lMax", "Order for Fourier exapansion");

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
  auto fes = FiniteElementSpace(&mesh, &fec);
  cout << "Number of finite element unknowns: " << fes.GetTrueVSize() << endl;

  std::cout << "building bilinear form\n";
  BilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  std::cout << "building DtN operator\n";
  auto C = DtN::PoissonSphere(&fes, lMax);
  C.Assemble();

  std::cout << "Doing the rest\n";
  // Set the density.
  auto rho_coefficient =
      FunctionCoefficient([=](const Vector &x) { return rho; });

  // Set up the form on the domain.
  auto domain_marker = Array<int>{1, 0};
  LinearForm b(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(rho_coefficient), domain_marker);
  b.Assemble();
  b *= -4 * M_PI * G;

  OperatorPtr A;
  Vector B, X;
  auto x = GridFunction(&fes);
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

  solver.Mult(B, X);

  a.RecoverFEMSolution(X, b, x);

  auto phi = FunctionCoefficient([=](const Vector &x) {
    auto r = (x(0) - x00) * (x(0) - x00) + (x(1) - x01) * (x(1) - x01) +
             (x(2) - x02) * (x(2) - x02);
    r = sqrt(r);
    if (r <= radius) {
      return -2 * pi * G * rho * (3 * radius * radius - r * r) / 3;
    } else {
      return -4 * pi * G * rho * pow(radius, 3) / (3 * r);
    }
  });

  auto y = GridFunction(&fes);
  y.ProjectCoefficient(phi);

  x -= y;

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
  sol_sock << "keys RRRilmc\n" << flush;
}
