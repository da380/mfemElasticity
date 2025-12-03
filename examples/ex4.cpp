#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity/poisson.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;
using namespace std::numbers;

int main(int argc, char *argv[]) {
  // 1. Initialize Options
  const char *mesh_file = "../data/circular_offset.msh";
  int order = 1;
  int ref = 0;
  int degree = 4;  // Expansion degree

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ref, "-r", "--refinement", "Serial refinement levels.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }

  // 2. Mesh Setup
  auto mesh = Mesh(mesh_file, 1, 1);
  auto dim = mesh.Dimension();
  for (int l = 0; l < ref; l++) {
    mesh.UniformRefinement();
  }

  // 3. Finite Element Space
  auto fec = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &fec);

  cout << "---------------------------------------------------" << endl;
  cout << " Test: HarmonicCoefficients (Serial)" << endl;
  cout << " Mesh: " << dim << "D" << endl;
  cout << " DOFs: " << fes.GetTrueVSize() << endl;
  cout << "---------------------------------------------------" << endl;

  // 4. Assemble DtN Operator
  auto dtn = PoissonDtNOperator(&fes, degree);
  dtn.Assemble();

  // 5. Project Known Function
  auto u = GridFunction(&fes);

  auto x0 = dtn.Centroid();
  auto b = dtn.BoundaryRadius();
  auto f = FunctionCoefficient([dim, b, x0](const Vector &x) -> real_t {
    auto r = x.DistanceTo(x0);
    if (dim == 2) {
      auto theta = atan2(x(1) - x0(1), x(0) - x0(0));

      return pow(r / b, 2) * sin(2 * theta);
    } else {
      auto dz = x(2) - x0(2);
      auto dy = x(1) - x0(1);
      auto dx = x(0) - x0(0);

      auto phi = atan2(dy, dx);
      auto R = hypot(dy, dx);
      auto theta = atan2(R, dz);

      auto ct = cos(theta);
      auto st = sin(theta);

      // return 0.5 / sqrt(pi); // Y_{00}
      //  return 0.25 * sqrt(5 / pi) * (3 * ct * ct - 1); //
      //  Y_{20}
      // return -0.5 * sqrt(15 / (pi)) * st * ct *
      //       cos(phi);  // Y_{2-1}
      // return 0.25 * sqrt(15 / pi) * st * st *
      //       cos(2 * phi);  // Y_{22}
      // return -0.5 * sqrt(15 / (pi)) * st * ct *
      //       sin(phi);  // Y_{21}
      // return 0.25 * sqrt(7 / pi) * ct *
      //       (5 * ct * ct - 3);  // Y_{30}
      return 0.25 * sqrt(105 / pi) * ct * st * st * sin(2 * phi);  // Y_{32}
    }
  });
  u.ProjectCoefficient(f);

  // 6. Compute Harmonic Coefficients
  auto coeffs = Vector();
  dtn.HarmonicCoefficients(u, coeffs);

  // Print out the coefficients for inspection
  if (dim == 2) {
    auto i = 0;
    cout << showpos;
    for (auto k = 1; k <= degree; k++) {
      cout << -k << " " << coeffs(i++) << endl;
      cout << k << " " << coeffs(i++) << endl;
    }
  } else {
    auto i = 0;
    cout << showpos;
    for (auto l = 0; l <= degree; l++) {
      cout << l << " " << 0 << " " << coeffs(i++) << endl;
      for (auto m = 1; m <= l; m++) {
        cout << l << " " << -m << " " << coeffs(i++) << endl;
        cout << l << " " << m << " " << coeffs(i++) << endl;
      }
    }
  }

  return 0;
}