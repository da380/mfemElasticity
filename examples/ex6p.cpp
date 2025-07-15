#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

const real_t G = 1;
const real_t rho_constant = 1;
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

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&serial_refinement, "-sr", "--serial_refinement",
                 "number of serial mesh refinements");
  args.AddOption(&parallel_refinement, "-pr", "--parallel_refinement",
                 "number of parallel mesh refinements");

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

  auto fes = ParFiniteElementSpace(&pmesh, &L2);
  HYPRE_BigInt size = fes.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  auto rho_coeff1 = ConstantCoefficient(rho_constant);
  auto rho_coeff2 = ConstantCoefficient(0);

  auto attr = Array<int>{1, 2};
  auto coeffs = Array<Coefficient *>{&rho_coeff1, &rho_coeff2};
  auto rho_coeff = PWCoefficient(attr, coeffs);

  auto rho = ParGridFunction(&fes);
  rho.ProjectCoefficient(rho_coeff);

  auto A = MomentsOperator(MPI_COMM_WORLD, &fes);
  A.Assemble();

  auto m = Vector(A.Height());

  A.Mult(rho, m);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    auto mass = m(0);
    cout << "mass = " << mass << endl;

    auto centroid = A.Centroid(m);
    cout << "centroid = ";
    for (auto i = 0; i < dim; i++) {
      cout << centroid(i) << " ";
    }
    cout << endl;
  }

  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);

  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  rho.Save(sol_ofs);

  // Visualise if glvis is open.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock.precision(8);
  sol_sock << "solution\n" << mesh << rho << flush;
  if (dim == 2) {
    sol_sock << "keys Rjlb\n" << flush;
  } else {
    sol_sock << "keys RRRilmc\n" << flush;
  }
}
