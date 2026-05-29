#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>
#include <vector>

#include "mfem.hpp"
#include "mfemElasticity.hpp"
#include "uniform_sphere.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

constexpr real_t pi = std::numbers::pi_v<mfem::real_t>;

int main(int argc, char* argv[]) {
  // Initialize MPI
  Mpi::Init(argc, argv);
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  // Set default options.
  const char* mesh_file = "../data/circular_offset.msh";
  int order = 1;
  int refinement = 0;
  int degree = 8;
  int residual = 0;
  int method = 0;
  int linearised = 0;

  // Deal with options.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order");
  args.AddOption(&refinement, "-r", "--refinement",
                 "number of mesh refinements");
  args.AddOption(&degree, "-deg", "--degree", "Order for Fourier expansion");
  args.AddOption(&residual, "-res", "--residual", "Output the residual");
  args.AddOption(&method, "-mth", "--method", "Solution method");
  args.AddOption(&linearised, "-lin", "--linearised",
                 "Solve linearised problem");

  args.Parse();
  if (!args.Good()) {
    if (myid == 0) args.PrintUsage(cout);
    return 1;
  }
  if (myid == 0) args.PrintOptions(cout);

  // Read in serial mesh on all ranks (or read on rank 0 and broadcast).
  auto mesh = Mesh(mesh_file, 1, 1);
  auto dim = mesh.Dimension();
  for (int l = 0; l < refinement; l++) {
    mesh.UniformRefinement();
  }

  // Properties of the first attribute.
  auto dom_marker = Array<int>(mesh.attributes.Max());
  dom_marker = 0;
  dom_marker[0] = 1;
  auto bdr_marker = Array<int>(mesh.bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[0] = 1;
  auto c1 = MeshCentroid(&mesh, dom_marker);
  auto [found1, same1, r1] = SphericalBoundaryRadius(&mesh, bdr_marker, c1);

  // --- PARALLEL MESH SETUP ---
  auto pmesh = ParMesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();  // Free serial mesh memory

  // Form the parallel submesh
  auto psub_mesh = ParSubMesh::CreateFromDomain(pmesh, Array<int>{1});
  auto marker = Array<int>{1, 0};

  // Set up the parallel FE-spaces
  auto H1 = H1_FECollection(order, dim);
  auto L2 = L2_FECollection(order, dim);

  auto pfes = ParFiniteElementSpace(&pmesh, &H1);
  auto psub_fes = ParFiniteElementSpace(&psub_mesh, &H1);

  auto rho = ConstantCoefficient(1.0);

  auto a00 = ParBilinearForm(&psub_fes);
  a00.AddDomainIntegrator(new DiffusionIntegrator());
  a00.AddDomainIntegrator(new MassIntegrator(rho));
  a00.Assemble();

  auto a11 = ParBilinearForm(&pfes);
  a11.AddDomainIntegrator(new DiffusionIntegrator());
  a11.AddDomainIntegrator(new MassIntegrator(rho));
  a11.Assemble();

  auto mu = ConstantCoefficient(0.2);
  auto a01 = ParBilinearForm(&psub_fes);
  a01.AddDomainIntegrator(new MassIntegrator(mu));
  a01.Assemble();

  auto a10 = ParBilinearForm(&pfes);
  a10.AddDomainIntegrator(new MassIntegrator(mu), marker);
  a10.Assemble();

  auto f = FunctionCoefficient(
      [&c1](const Vector& x) { return (x[1] - c1[1]) * (x[0] - c1[0]); });
  auto b0 = ParLinearForm(&psub_fes);
  b0.AddDomainIntegrator(new DomainLFIntegrator(f));
  b0.Assemble();

  auto b1 = ParLinearForm(&pfes);
  b1.Assemble();

  // Set up Parallel GridFunctions
  auto x0 = ParGridFunction(&psub_fes);
  auto x1 = ParGridFunction(&pfes);

  // Set up the linear systems
  x0 = 0.0;
  x1 = 0.0;
  Array<int> ess_tdof_list;

  HypreParMatrix A00, A11, A01, A10;
  Vector B0, X0, B1, X1;

  a00.FormLinearSystem(ess_tdof_list, x0, b0, A00, X0, B0);
  a11.FormLinearSystem(ess_tdof_list, x1, b1, A11, X1, B1);

  // Generate true-DOF Hypre matrices for cross-coupling operators
  a01.FormSystemMatrix(ess_tdof_list, A01);
  a10.FormSystemMatrix(ess_tdof_list, A10);

  // --- PARALLEL SOLVERS USING HYPRE BOOMER AMG ---
  auto amg00 = HypreBoomerAMG(A00);
  amg00.SetPrintLevel(0);
  auto solver00 = CGSolver(MPI_COMM_WORLD);

  solver00.SetRelTol(1e-12);
  solver00.SetMaxIter(10000);
  solver00.SetPrintLevel(myid == 0 ? 1 : 0);
  solver00.SetOperator(A00);
  solver00.SetPreconditioner(amg00);

  auto amg11 = HypreBoomerAMG(A11);
  amg11.SetPrintLevel(0);
  auto solver11 = CGSolver(MPI_COMM_WORLD);
  solver11.SetRelTol(1e-12);
  solver11.SetMaxIter(10000);
  solver11.SetPrintLevel(myid == 0 ? 1 : 0);
  solver11.SetOperator(A11);
  solver11.SetPreconditioner(amg11);

  solver00.iterative_mode = true;
  solver11.iterative_mode = true;

  // --- Automated Block Gauss-Seidel Outer Iterations ---

  Vector B0_base(B0);
  Vector B1_base(B1);
  Vector B0_curr(B0.Size());
  Vector B1_curr(B1.Size());

  auto x10 = ParGridFunction(&pfes);
  auto x01 = ParGridFunction(&psub_fes);

  x10 = 0.0;
  x01 = 0.0;

  // Buffers for extracting True DOFs prior to multiplication
  Vector X01_true(X0.Size());
  Vector X10_true(X1.Size());

  int max_outer_iters = 100;
  double outer_tol = 1e-8;

  if (myid == 0) cout << "\n--- Starting Outer Iterations ---\n";

  Vector X0_old(X0);
  Vector X1_old(X1);

  for (int iter = 0; iter < max_outer_iters; ++iter) {
    X0_old = X0;
    X1_old = X1;

    // ==========================================
    // Solve Submesh (Domain 0)
    // ==========================================

    // Transfer the global L-vector (x1) down to the submesh L-vector (x01)
    psub_mesh.Transfer(x1, x01);

    // Convert transferred L-vector to T-vector for linear operations
    x01.GetTrueDofs(X01_true);

    B0_curr = B0_base;
    A01.AddMult(X01_true, B0_curr, -1.0);

    solver00.Mult(B0_curr, X0);

    if (iter == 0) {
      auto norm = solver00.GetFinalNorm();
      solver00.SetAbsTol(norm);
    }

    // Distribute T-vector solution X0 back into L-vector GridFunction x0
    a00.RecoverFEMSolution(X0, b0, x0);

    // ==========================================
    // Solve Global Mesh (Domain 1)
    // ==========================================

    // Transfer the submesh L-vector (x0) up to the global L-vector (x10)
    psub_mesh.Transfer(x0, x10);

    // Convert transferred L-vector to T-vector
    x10.GetTrueDofs(X10_true);

    B1_curr = B1_base;
    if (myid == 0) {
      cout << B1_curr.Norml2() << endl;
    }
    A10.AddMult(X10_true, B1_curr, -1.0);
    if (myid == 0) {
      cout << B1_curr.Norml2() << endl;
    }

    solver11.Mult(B1_curr, X1);

    if (iter == 0) {
      auto norm = solver11.GetFinalNorm();
      solver11.SetAbsTol(norm);
    }

    // Distribute T-vector solution X1 back into L-vector GridFunction x1
    a11.RecoverFEMSolution(X1, b1, x1);

    // ==========================================
    // Convergence Check (Parallel Global Norm)
    // ==========================================
    X0_old -= X0;
    X1_old -= X1;

    // Calculate sum of squares locally
    double local_dot0 = X0_old * X0_old;
    double local_dot1 = X1_old * X1_old;

    // Reduce sums globally
    double global_dot0, global_dot1;
    MPI_Allreduce(&local_dot0, &global_dot0, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_dot1, &global_dot1, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    double err0 = sqrt(global_dot0);
    double err1 = sqrt(global_dot1);

    if (myid == 0) {
      cout << "  Iter " << iter + 1 << ": dX0 = " << err0 << ", dX1 = " << err1
           << "\n";
    }

    if (err0 < outer_tol && err1 < outer_tol) {
      if (myid == 0)
        cout << "Outer iteration converged in " << iter + 1 << " steps.\n";
      break;
    }
  }

  char vishost[] = "localhost";
  int visport = 19916;

  socketstream sol_sock_1(vishost, visport);
  sol_sock_1 << "parallel " << num_procs << " " << myid << "\n";
  sol_sock_1.precision(8);
  sol_sock_1 << "solution\n" << pmesh << x1 << flush;
  sol_sock_1 << "keys Rjlbc\n" << flush;

  socketstream sol_sock_2(vishost, visport);
  sol_sock_2 << "parallel " << num_procs << " " << myid << "\n";
  sol_sock_2.precision(8);
  sol_sock_2 << "solution\n" << pmesh << x10 << flush;
  sol_sock_2 << "keys Rjlbc\n" << flush;

  return 0;
}