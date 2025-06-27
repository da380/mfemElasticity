#include <cmath>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

int main(int argc, char* argv[]) {
  // 1. Parse command-line options.
  const char* mesh_file = "../data/circular_shell.msh";

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
  auto fes = FiniteElementSpace(&mesh, &fec, dim);
  cout << "Number of finite element unknowns: " << fes.GetTrueVSize() << endl;

  auto f = FunctionCoefficient([](const Vector& x) {
    auto r = x.Norml2();
    auto th = atan2(x(1), x(0));
    return sin(th);
  });

  auto u = GridFunction(&fes);
  u.ProjectCoefficient(f);

  auto kmax = 4;
  auto A = DtN::Poisson2D(&fes, kmax);

  auto coeffs = Vector();
  A.FourierTransformation(u, coeffs);

  for (auto k = -kmax; k <= kmax; k++) {
    cout << k << " " << coeffs(k + kmax) << endl;
  }

  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);

  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  u.Save(sol_ofs);
}