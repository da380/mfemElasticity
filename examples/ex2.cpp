#include <cassert>
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

  // Set up the DtN operator.
  auto c = PoissonDtNOperator(&fes, degree);

  // Set the density coefficient.
  auto rho_coeff1 = ConstantCoefficient(1);
  auto rho_coeff2 = ConstantCoefficient(0);

  auto attr = Array<int>{1, 2};
  auto coeffs = Array<Coefficient *>{&rho_coeff1, &rho_coeff2};
  auto rho_coeff = PWCoefficient(attr, coeffs);

  // Set the linear form.
  auto b = LinearForm(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff));
  b.Assemble();

  // If in 2D, add in necessary boundary form.
  if (dim == 2) {
    auto x = GridFunction(&fes);
    x = 1.0;
    auto mass = b(x);
    auto bb = LinearForm(&fes);
    auto one = ConstantCoefficient(1);
    auto boundary_marker = Array<int>{0, 1};
    bb.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), boundary_marker);
    bb.Assemble();
    auto length = bb(x);
    b.Add(-mass / length, bb);
  }

  // Scale the linear form.
  b *= -4 * pi;

  // Set up the linear system
  auto x = GridFunction(&fes);
  x = 0.0;
  Array<int> ess_tdof_list{};
  SparseMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
  auto D = SumOperator(&A, 1, &c, 1, false, false);

  // Set up the preconditioner.
  SparseMatrix As;
  as.FormSystemMatrix(ess_tdof_list, As);
  auto P = GSSmoother(As);

  // Set up the solver.
  auto solver = CGSolver();
  solver.SetOperator(D);
  solver.SetPreconditioner(P);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  // Sovler the linear system.
  if (dim == 2) {
    auto orthoSolver = OrthoSolver();
    orthoSolver.SetSolver(solver);
    orthoSolver.Mult(B, X);
  } else {
    solver.Mult(B, X);
  }
  a.RecoverFEMSolution(X, b, x);

  auto exact = UniformSphereSolution(dim, c1, r1);
  auto exact_coeff = exact.Coefficient();

  auto y = GridFunction(&fes);
  y.ProjectCoefficient(exact_coeff);

  // In 2D, remove the mean from the solution.
  if (dim == 2) {
    auto l = LinearForm(&fes);
    auto z = GridFunction(&fes);
    z = 1.0;
    auto one = ConstantCoefficient(1);
    l.AddDomainIntegrator(new DomainLFIntegrator(one));
    l.Assemble();
    auto area = l(z);
    l /= area;
    auto px = l(x);
    x -= px;
    auto py = l(y);
    y -= py;
  }

  if (residual == 1) {
    x -= y;
  }

  // Write to file.
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
  if (dim == 2) {
    sol_sock << "keys Rjlbc\n" << flush;
  } else {
    sol_sock << "keys RRRilmc\n" << flush;
  }
}
