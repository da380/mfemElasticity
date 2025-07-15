#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

constexpr real_t pi = atan(1) * 4;
const real_t G = 1;
const real_t rho = 1;

int main(int argc, char *argv[]) {
  // Set default options.
  int dim = 2;
  int order = 1;
  int serial_refinement = 0;
  int parallel_refinement = 0;
  int degree = 4;

  // Deal with options.
  OptionsParser args(argc, argv);
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&serial_refinement, "-sr", "--serial_refinement",
                 "number of serial mesh refinements");
  args.AddOption(&parallel_refinement, "-pr", "--parallel_refinement",
                 "number of parallel mesh refinements");
  args.AddOption(&dim, "-dim", "--dimension", "dimension of the mesh");
  args.AddOption(&degree, "-deg", "--degree", "Order for Fourier exapansion");

  assert(dim == 2 || dim == 3);

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // Read in mesh in serial.
  auto mesh_file =
      dim == 2
          ? string("/home/david/dev/meshing/examples/circular_offset.msh")
          : string("/home/david/dev/meshing/examples/spherical_offset.msh");
  auto mesh = Mesh(mesh_file, 1, 1);
  {
    for (int l = 0; l < serial_refinement; l++) {
      mesh.UniformRefinement();
    }
  }

  // Check the mesh has two attributes.
  assert(mesh.attributes.Max() == 2);

  // Set up the finite element space.
  auto fec = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &fec);
  cout << "Number of finite element unknowns: " << fes.GetTrueVSize() << endl;

  // Assemble the binlinear form for Poisson's equation.
  auto a = BilinearForm(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  // Assemble the mass form for use in preconditioning.
  auto m = BilinearForm(&fes);
  m.AddDomainIntegrator(new MassIntegrator());
  m.Assemble();

  // Set up the DtN operator.
  auto c = PoissonDtNOperator(&fes, degree);

  // Set the density coefficient.
  auto rho_coeff1 = ConstantCoefficient(rho);
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
  b *= -4 * pi * G;

  // Set up the linear system
  auto x = GridFunction(&fes);
  x = 0.0;
  Array<int> ess_tdof_list{};
  SparseMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
  auto D = SumOperator(&A, 1, &c, 1, false, false);

  // Form the preconditioner, using the mass-matrix to make it
  // positive-definite.
  auto AShift = SparseMatrix(A);
  auto M = SparseMatrix();
  m.FormSystemMatrix(ess_tdof_list, M);
  auto A_norm = A.MaxNorm();
  auto M_norm = M.MaxNorm();
  auto eps = 1e-6 * A_norm / M_norm;
  AShift.Add(eps, M);

  auto P = GSSmoother(AShift);

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
