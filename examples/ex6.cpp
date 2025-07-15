#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

const real_t G = 1;
const real_t rho_constant = 1;
const real_t radius = 0.5;
const real_t x00 = 0.0;
const real_t x01 = 0.25;
const real_t x02 = 0.0;
constexpr real_t pi = atan(1) * 4;

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *mesh_file =
      "/home/david/dev/meshing/examples/circular_offset.msh";
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

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &L2);
  cout << "Number of finite element unknowns: " << fes.GetTrueVSize() << endl;

  auto rho_coeff1 = ConstantCoefficient(rho_constant);
  auto rho_coeff2 = ConstantCoefficient(0);

  auto attr = Array<int>{1, 2};
  auto coeffs = Array<Coefficient *>{&rho_coeff1, &rho_coeff2};
  auto rho_coeff = PWCoefficient(attr, coeffs);

  auto rho = GridFunction(&fes);
  rho.ProjectCoefficient(rho_coeff);

  auto A = MomentsOperator(&fes);
  A.Assemble();

  auto m = Vector(A.Height());

  A.Mult(rho, m);

  auto mass = m(0);
  cout << "mass = " << mass << endl;

  auto centroid = A.Centroid(m);
  cout << "centroid = ";
  for (auto i = 0; i < dim; i++) {
    cout << centroid(i) << " ";
  }
  cout << endl;

  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);

  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  rho.Save(sol_ofs);

  // Visualise if glvis is open.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock.precision(8);
  sol_sock << "solution\n" << mesh << rho << flush;
  if (dim == 2) {
    sol_sock << "keys Rjlb\n" << flush;
  } else {
    sol_sock << "keys RRRilmc\n" << flush;
  }
}
