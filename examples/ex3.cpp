#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"
#include "poisson.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

constexpr real_t pi = atan(1) * 4;

int main(int argc, char *argv[]) {
  // Set default options.
  const char *mesh_file =
      "/home/david/dev/meshing/examples/circular_offset.msh";
  int order = 1;
  int serial_refinement = 0;
  int parallel_refinement = 0;
  int degree = 4;
  int residual = 0;

  // Deal with options.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&serial_refinement, "-sr", "--serial_refinement",
                 "number of serial mesh refinements");
  args.AddOption(&parallel_refinement, "-pr", "--parallel_refinement",
                 "number of parallel mesh refinements");
  args.AddOption(&degree, "-deg", "--degree", "Order for Fourier exapansion");
  args.AddOption(&residual, "-res", "--residual",
                 "Output the residual from reference solution");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // Read in mesh in serial.
  auto mesh = Mesh(mesh_file, 1, 1);
  auto dim = mesh.Dimension();
  {
    for (int l = 0; l < serial_refinement; l++) {
      mesh.UniformRefinement();
    }
  }

  // Check mesh has two attributes.
  assert(mesh.attributes.Max() == 2);

  // Get centroid and radius for inner domain.
  auto c1 = MeshCentroid(&mesh, Array<int>{1, 0});
  auto [found1, same1, r1] = BoundaryRadius(&mesh, Array<int>{1, 0}, c1);
  assert(found1 == 1 && same1 == 1);

  // Get centroid and radius for combined domain.
  auto c2 = MeshCentroid(&mesh, Array<int>{1, 1});
  auto [found2, same2, r2] = BoundaryRadius(&mesh, Array<int>{0, 1}, c2);
  assert(found2 == 1 && same2 == 1);

  // Set up the finite element space.
  auto fec = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &fec);
  cout << "Number of finite element unknowns: " << fes.GetTrueVSize() << endl;

  // Assemble the binlinear form for Poisson's equation.
  auto a = BilinearForm(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  // Assemble mass-shifted binlinear form for preconditioning.
  auto eps = ConstantCoefficient(0.01);
  auto as = BilinearForm(&fes, &a);
  as.AddDomainIntegrator(new MassIntegrator(eps));
  as.Assemble();

  /*
  auto L2 = L2_FECollection(order - 1, dim);
  auto H1 = H1_FECollection(order, dim);
  auto dfes = FiniteElementSpace(&mesh, &L2);
  auto fes = FiniteElementSpace(&mesh, &H1);
  cout << "Number of finite element unknowns: " << fes.GetTrueVSize() << endl;

  // Set piecewise coefficient for density.
  auto rho_coeff1 = ConstantCoefficient(rho);
  auto rho_coeff2 = ConstantCoefficient(0);

  auto attr = Array<int>{1, 2};
  auto coeffs = Array<Coefficient *>{&rho_coeff1, &rho_coeff2};
  auto rho_coeff = PWCoefficient(attr, coeffs);

  auto z = GridFunction(&dfes);
  z.ProjectCoefficient(rho_coeff);

  auto C = PoissonMultipoleCircle(&dfes, &fes, kMax);
  C.Assemble();

  BilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  // Set up the form on the domain.
  auto domain_marker = Array<int>{1, 0};
  LinearForm b(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff), domain_marker);
  b.Assemble();

  auto n = GridFunction(&fes);
  C.Mult(z, n);
  b -= n;

  b *= -4 * M_PI * G;

  OperatorPtr A;
  Vector B, X;
  auto x = GridFunction(&fes);
  x = 0.0;

  Array<int> ess_tdof_list{};
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
  cout << "Size of linear system: " << A->Height() << endl;

  GSSmoother M((SparseMatrix &)(*A));

  auto solver = CGSolver();
  solver.SetOperator(*A);
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

  auto one = ConstantCoefficient(1);
  auto l = LinearForm(&fes);
  l.AddDomainIntegrator(new DomainLFIntegrator(one));
  l.Assemble();
  auto area = l.Sum();
  l /= area;

  auto y = GridFunction(&fes);
  y.ProjectCoefficient(phi);

  auto py = l * y;
  y -= py;

  auto px = l * x;
  x -= px;

  ofstream exact_ofs("exact.gf");
  exact_ofs.precision(8);
  y.Save(exact_ofs);

  x -= y;
  ofstream diff_ofs("diff.gf");
  diff_ofs.precision(8);
  x.Save(diff_ofs);

  return 0;
*/
}
