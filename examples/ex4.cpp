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
const real_t x00 = 0.25;
const real_t x01 = 0.0;
const real_t x02 = 0.0;

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *mesh_file =
      "/home/david/dev/meshing/examples/spherical_offset.msh";
  int order = 1;
  int ref_levels = 0;
  int lMax = 4;
  int l = 0;
  int m = 0;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&ref_levels, "-r", "--refine", "Number of mesh refinements");
  args.AddOption(&lMax, "-lMax", "--lMax", "Order for Fourier exapansion");

  args.AddOption(&l, "-l", "--l", "degree");
  args.AddOption(&m, "-m", "--m", "order");

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

  BilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  auto C = DtN::Poisson3D(&fes, lMax);
  C.Assemble();

  /*
  auto x = GridFunction(&fes);

  auto f = FunctionCoefficient([l, m](const Vector &x) {
    auto r = x.Norml2();
    auto R = sqrt(x(0) * x(0) + x(1) * x(1));
    auto phi = atan2(x(1), x(0));
    auto theta = atan2(R, x(2));
    auto p = Legendre(cos(theta), l);
    return (abs(m) > 0 ? sqrt(2) : 1) * pow(r, l) * p(m) *
           (m > 0 ? sin(m * phi) : cos(m * phi));
  });

  x.ProjectCoefficient(f);
  */

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

  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);

  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  x.Save(sol_ofs);

  return 0;
}
