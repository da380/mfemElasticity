#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

const real_t pi = 3.14159265358979323846264338327950288;
const real_t G = 1;
const real_t rho = 1;
const real_t radius = 1;
const real_t x00 = 0.5;
const real_t x01 = 0.0;

int main(int argc, char *argv[]) {
  // 1. Initialize MPI and HYPRE.
  Mpi::Init();
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  const char *mesh_file =
      "/home/david/dev/meshing/examples/circular_offset.msh";

  int order = 1;
  int serial_refinement = 0;
  int parallel_refinement = 0;
  int kmax = 8;

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

  auto fec = H1_FECollection(order, dim);
  ParFiniteElementSpace fespace(&pmesh, &fec);
  HYPRE_BigInt size = fespace.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  auto one_function = ParGridFunction(&fespace);
  one_function = 1.0;

  ParBilinearForm a(&fespace);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  auto C = DtN::Poisson2D(MPI_COMM_WORLD, &fespace, kmax);
  C.Assemble();

  /*

    // Set the density.
    auto rho_coefficient =
        FunctionCoefficient([=](const Vector &x) { return rho; });

    // Set up the form on the domain.
    auto domain_marker = Array<int>{1, 0};
    ParLinearForm b(&fespace);
    b.AddDomainIntegrator(new DomainLFIntegrator(rho_coefficient),
  domain_marker); b.Assemble(); auto mass = b(one_function);

    // And now the form on the boundary.
    auto b2 = ParLinearForm(&fespace);
    auto one = ConstantCoefficient(1);
    auto boundary_marker = Array<int>{0, 1};
    b2.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), boundary_marker);
    b2.Assemble();
    auto length = b2(one_function);

    // Form the total form.
    b.Add(-mass / length, b2);
    b *= -4 * pi * G;

    // Form the Linear system.
    auto x = ParGridFunction(&fespace);
    x = 0.0;
    Array<int> ess_tdof_list{};
    HypreParMatrix A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

    auto D = SumOperator(dynamic_cast<Operator *>(&A), 1, &C, 1, false, false);

    // Set the solver.
    auto solver = CGSolver(MPI_COMM_WORLD);
    // auto solver = HyprePCG(MPI_COMM_WORLD);
    // solver.SetPreconditioner(prec);
    solver.SetOperator(D);

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
        return pi * G * rho * r * r;
      } else {
        return 2 * pi * G * rho * radius * log(r / radius) +
               pi * G * rho * radius * radius;
      }
    });

    auto l = ParLinearForm(&fespace);
    l.AddDomainIntegrator(new DomainLFIntegrator(one));
    l.Assemble();
    auto area = l(one_function);
    l /= area;

    auto y = ParGridFunction(&fespace);
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
    x.Save(sol_ofs);

    // Visualise if glvis is open.
    char vishost[] = "localhost";
    int visport = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock << "parallel " << num_procs << " " << myid << "\n";
    sol_sock.precision(8);
    sol_sock << "solution\n" << pmesh << x << flush;
    sol_sock << "keys Rjlmc\n" << flush;

  */
}
