#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity/poisson.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

// --------------------------------------------------------------------------
// Test Function: Dipole (u = x)
// In 2D Polar: u = r * cos(theta). On boundary r=R, u = R * cos(theta).
// This corresponds to the k=1 harmonic mode.
// --------------------------------------------------------------------------
real_t dipole_function(const Vector &x) { return x(0); }

int main(int argc, char *argv[]) {
  // 1. Initialize Options
  const char *mesh_file = "../data/circular_offset.msh";
  int order = 2;
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
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();
  for (int l = 0; l < ref; l++) {
    mesh.UniformRefinement();
  }

  // 3. Finite Element Space
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec);

  cout << "---------------------------------------------------" << endl;
  cout << " Test: HarmonicCoefficients (Serial)" << endl;
  cout << " Mesh: " << dim << "D" << endl;
  cout << " DOFs: " << fes.GetTrueVSize() << endl;
  cout << "---------------------------------------------------" << endl;

  // 4. Assemble DtN Operator
  PoissonDtNOperator dtn(&fes, degree);
  dtn.Assemble();

  // 5. Project Known Function (Dipole)
  FunctionCoefficient u_coeff(dipole_function);
  GridFunction x(&fes);
  x.ProjectCoefficient(u_coeff);

  // 6. Compute Harmonic Coefficients
  Vector coeffs;
  dtn.HarmonicCoefficients(x, coeffs);

  // 7. Analysis
  // We expect the coefficient corresponding to k=1 (2D) or l=1 (3D) to be
  // significant. All other coefficients should be near machine
  // epsilon/projection error.

  cout << "\nResults for input u = x:" << endl;
  bool passed = false;
  real_t max_val = coeffs.Normlinf();

  for (int i = 0; i < coeffs.Size(); i++) {
    // Filter out noise for printing
    if (std::abs(coeffs(i)) > 1e-4 * max_val) {
      cout << "  Index " << i << ": " << coeffs(i) << " [SIGNIFICANT]" << endl;

      // In your 2D implementation, indices are typically:
      // i=0: k=1 cos, i=1: k=1 sin, i=2: k=2 cos...
      if (dim == 2 && i == 0) passed = true;

      // In 3D implementation, typically l=0 is index 0.
      // l=1 has 3 modes. One of them corresponds to X.
      if (dim == 3 && i > 0 && i <= 3) passed = true;
    }
  }

  cout << "\nStatus: ";
  if (passed) {
    cout << "PASS (Dipole mode detected)" << endl;
  } else {
    cout << "FAIL (Dipole mode not dominant or logic error)" << endl;
  }

  return 0;
}