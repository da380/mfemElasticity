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
  // Initialise MPI and Hypre.
  Mpi::Init();
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

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
  if (myid == 0) {
    args.PrintOptions(cout);
  }

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

  // Form the parallel mesh.
  auto pmesh = ParMesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  {
    for (int l = 0; l < parallel_refinement; l++) {
      pmesh.UniformRefinement();
    }
  }

  // Set up the finite element space.
  auto fec = H1_FECollection(order, dim);
  ParFiniteElementSpace fes(&pmesh, &fec);
  HYPRE_BigInt size = fes.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  // Assemble the binlinear form for Poisson's equation.
  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  // Assemble the mass form for use in preconditioning.
  auto m = ParBilinearForm(&fes);
  m.AddDomainIntegrator(new MassIntegrator());
  m.Assemble();

  // Set up the DtN operator.
  auto c = PoissonDtNOperator(MPI_COMM_WORLD, &fes, degree);

  // Set the density coefficient.
  auto rho_coeff1 = ConstantCoefficient(rho);
  auto rho_coeff2 = ConstantCoefficient(0);

  auto attr = Array<int>{1, 2};
  auto coeffs = Array<Coefficient *>{&rho_coeff1, &rho_coeff2};
  auto rho_coeff = PWCoefficient(attr, coeffs);

  // Set the linear form.
  ParLinearForm b(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff));
  b.Assemble();

  // If in 2D, add in necessary boundary form.
  if (dim == 2) {
    auto x = ParGridFunction(&fes);
    x = 1.0;
    auto mass = b(x);
    auto bb = ParLinearForm(&fes);
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
  auto x = ParGridFunction(&fes);
  x = 0.0;
  Array<int> ess_tdof_list{};
  HypreParMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
  auto C = c.RAP();
  auto D = SumOperator(dynamic_cast<Operator *>(&A), 1, &C, 1, false, false);

  // Form the preconditioner, using the mass-matrix to make it
  // positive-definite.
  auto P = HypreParMatrix(A);
  auto M = HypreParMatrix();
  m.FormSystemMatrix(ess_tdof_list, M);
  auto A_norm = A.FNorm();
  auto M_norm = M.FNorm();
  auto eps = 1e-6 * A_norm / M_norm;
  P.Add(eps, M);
  auto prec = HypreBoomerAMG(P);

  // Set up the solver.
  auto solver = CGSolver(MPI_COMM_WORLD);
  solver.SetOperator(D);
  solver.SetPreconditioner(prec);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  // Sovler the linear system.
  if (dim == 2) {
    auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
    orthoSolver.SetSolver(solver);
    orthoSolver.Mult(B, X);
  } else {
    solver.Mult(B, X);
  }
  a.RecoverFEMSolution(X, b, x);

  // In 2D, remove the mean from the solution.
  if (dim == 2) {
    auto l = ParLinearForm(&fes);
    auto z = ParGridFunction(&fes);
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
  ostringstream mesh_name, sol_name;
  mesh_name << "mesh." << setfill('0') << setw(6) << myid;
  sol_name << "sol." << setfill('0') << setw(6) << myid;

  ofstream mesh_ofs(mesh_name.str().c_str());
  mesh_ofs.precision(8);
  pmesh.Print(mesh_ofs);

  ofstream sol_ofs(sol_name.str().c_str());
  sol_ofs.precision(8);
  x.Save(sol_ofs);

  // Visualise if glvis is open.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock << "parallel " << num_procs << " " << myid << "\n";
  sol_sock.precision(8);
  sol_sock << "solution\n" << pmesh << x << flush;
  if (dim == 2) {
    sol_sock << "keys Rjlbc\n" << flush;
  } else {
    sol_sock << "keys RRRilmc\n" << flush;
  }
}
