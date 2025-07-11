#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

const real_t G = 1;
const real_t rho = 1;
const real_t radius = 0.5;
const real_t x00 = 0.0;
const real_t x01 = 0.25;
const real_t x02 = 0.0;
constexpr real_t pi = atan(1) * 4;

int main(int argc, char *argv[]) {
  Mpi::Init();
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  const char *mesh_file =
      "/home/david/dev/meshing/examples/spherical_offset.msh";
  // const char *mesh_file = "../data/star.mesh";

  int order = 1;
  int serial_refinement = 0;
  int parallel_refinement = 0;

  int lMax = 4;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&serial_refinement, "-sr", "--serial_refinement",
                 "number of serial mesh refinements");
  args.AddOption(&parallel_refinement, "-pr", "--parallel_refinement",
                 "number of parallel mesh refinements");

  args.AddOption(&lMax, "-lMax", "--lMax", "Order for Fourier exapansion");

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
  auto fes = ParFiniteElementSpace(&pmesh, &fec);
  HYPRE_BigInt size = fes.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  auto eps = ConstantCoefficient(0.1);
  auto p = ParBilinearForm(&fes);
  p.AddDomainIntegrator(new DiffusionIntegrator());
  p.AddDomainIntegrator(new MassIntegrator(eps));
  p.Assemble();

  auto C = DtN::PoissonSphere(MPI_COMM_WORLD, &fes, lMax);
  C.Assemble();

  // Set the density.
  auto rho_coefficient =
      FunctionCoefficient([=](const Vector &x) { return rho; });

  // Set up the form on the domain.
  auto domain_marker = Array<int>{1, 0};
  ParLinearForm b(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(rho_coefficient), domain_marker);
  b.Assemble();
  b *= -4 * pi * G;

  // Form the Linear system.
  auto x = ParGridFunction(&fes);
  x = 0.0;
  Array<int> ess_tdof_list{};
  HypreParMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  auto RCP = C.FormSystemMatrix();
  auto D = SumOperator(dynamic_cast<Operator *>(&A), 1, &RCP, 1, false, false);

  HypreParMatrix P;
  p.FormSystemMatrix(ess_tdof_list, P);

  auto prec = HypreBoomerAMG(P);

  // Set the solver.
  auto solver = CGSolver(MPI_COMM_WORLD);
  // auto solver = GMRESSolver(MPI_COMM_WORLD);

  solver.SetOperator(D);
  solver.SetPreconditioner(prec);

  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  // Solve and recover the solution.
  solver.Mult(B, X);
  a.RecoverFEMSolution(X, b, x);

  auto phi = FunctionCoefficient([=](const Vector &x) {
    auto r = (x(0) - x00) * (x(0) - x00) + (x(1) - x01) * (x(1) - x01) +
             (x(2) - x02) * (x(2) - x02);
    r = sqrt(r);
    if (r <= radius) {
      return -2 * pi * G * rho * (3 * radius * radius - r * r) / 3;
    } else {
      return -4 * pi * G * rho * pow(radius, 3) / (3 * r);
    }
  });

  auto y = ParGridFunction(&fes);
  y.ProjectCoefficient(phi);

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
  sol_sock << "keys RRRilmc\n" << flush;
}
