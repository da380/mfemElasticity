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
const real_t x00 = 0.5;
const real_t x01 = 0.5;

int main(int argc, char *argv[]) {
  // 1. Initialize MPI and HYPRE.
  Mpi::Init();
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  const char *mesh_file =
      "/home/david/dev/meshing/examples/circular_offset.msh";
  // const char *mesh_file = "../data/star.mesh";

  int order = 3;
  int serial_refinement = 0;
  int parallel_refinement = 0;
  int kmax = 16;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&serial_refinement, "-sr", "--serial_refinement",
                 "number of serial mesh refinements");
  args.AddOption(&parallel_refinement, "-pr", "--parallel_refinement",
                 "number of parallel mesh refinements");

  args.AddOption(&kmax, "-kmax", "--kmax", "Order for Fourier exapansion");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(cout);
  }

  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();
  {
    for (int l = 0; l < serial_refinement; l++) {
      mesh.UniformRefinement();
    }
  }

  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  {
    for (int l = 0; l < parallel_refinement; l++) {
      pmesh.UniformRefinement();
    }
  }

  auto L2 = L2_FECollection(order - 1, dim);
  auto H1 = H1_FECollection(order, dim);

  auto dfes = ParFiniteElementSpace(&pmesh, &L2);
  auto fes = ParFiniteElementSpace(&pmesh, &H1);
  HYPRE_BigInt size = fes.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  auto one_function = ParGridFunction(&fes);
  one_function = 1.0;
  auto one = ConstantCoefficient(1);

  // Set piecewise coefficient for density.
  auto rho_coeff1 = ConstantCoefficient(rho);
  auto rho_coeff2 = ConstantCoefficient(0);

  auto attr = Array<int>{1, 2};
  auto coeffs = Array<Coefficient *>{&rho_coeff1, &rho_coeff2};
  auto rho_coeff = PWCoefficient(attr, coeffs);

  auto z = ParGridFunction(&dfes);
  z.ProjectCoefficient(rho_coeff);

  auto C = Multipole::Poisson2D(MPI_COMM_WORLD, &dfes, &fes, kmax);

  C.Assemble();

  auto a = ParBilinearForm(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  // Set up the form on the domain.
  auto b = ParLinearForm(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff));
  b.Assemble();

  auto n = ParGridFunction(&fes);

  C.Mult(z, n);
  b -= n;

  b *= -4 * M_PI * G;

  // Form the Linear system.
  auto x = ParGridFunction(&fes);
  x = 0.0;
  Array<int> ess_tdof_list{};
  HypreParMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  auto prec = HypreBoomerAMG();

  // Set the solver.
  auto solver = GMRESSolver(MPI_COMM_WORLD);

  solver.SetPreconditioner(prec);
  solver.SetOperator(A);

  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  // Set the orthosolver.
  auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
  orthoSolver.SetSolver(solver);

  // Solve and recover the solution.
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

  auto l = ParLinearForm(&fes);
  l.AddDomainIntegrator(new DomainLFIntegrator(one));
  l.Assemble();
  auto area = l(one_function);
  l /= area;

  auto y = ParGridFunction(&fes);
  y.ProjectCoefficient(phi);

  auto py = l(y);
  y -= py;

  auto px = l(x);
  x -= px;

  x -= y;

  // Write to file.
  ostringstream mesh_name, sol_name;
  mesh_name << "mesh." << setfill('0') << setw(6) << myid;
  sol_name << "sol." << setfill('0') << setw(6) << myid;

  ofstream mesh_ofs(mesh_name.str().c_str());
  mesh_ofs.precision(8);
  pmesh.Print(mesh_ofs);

  ofstream sol_ofs(sol_name.str().c_str());
  sol_ofs.precision(8);
  z.Save(sol_ofs);

  // Visualise if glvis is open.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock << "parallel " << num_procs << " " << myid << "\n";
  sol_sock.precision(8);
  sol_sock << "solution\n" << pmesh << x << flush;
  sol_sock << "keys Rjlbc\n" << flush;
}
