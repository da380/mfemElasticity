/*********************************************************************************

Solves a static elastic  problem, with a constant boundary traction applied to
the mesh's external boundary. This code illustrates the use of the
RigidBodySolver class to project out the null space for the problem.

This is the parallel version of ex1 that makes use of MPI.

Options:

[-m, --mesh]: The mesh. Either 2D or 3D. Tractions are applied to its external
              boundary. Default is star.mesh in the data directory.

[-o, --order]: The polynomial order used in the calculations. Default is 1.

[-sr, --serial_refinement]: The number of times to refine the mesh in serial.
                            The default it 0.

[-pr, --parallel_refinement]: The number of times to refine the mesh in
                              parallel. The default it 0.

*********************************************************************************/

#include <cmath>
#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char* argv[]) {
  // 1. Initialize MPI and HYPRE.
  Mpi::Init();
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  // 2. Parse command-line options.
  const char* mesh_file = "../data/star.mesh";
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
    if (myid == 0) {
      args.PrintUsage(cout);
    }
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

  if (myid == 0) {
    mesh.attributes.Print(cout);
    mesh.bdr_attributes.Print(cout);
  }

  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  {
    for (int l = 0; l < parallel_refinement; l++) {
      pmesh.UniformRefinement();
    }
  }

  auto fec = H1_FECollection(order, dim);
  ParFiniteElementSpace fespace(&pmesh, &fec, dim);
  HYPRE_BigInt size = fespace.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  // Trivial list of essential boundary conditions.
  Array<int> ess_tdof_list;

  auto x = ParGridFunction(&fespace);
  x = 0;

  // Set up the linear form.
  auto b = ParLinearForm(&fespace);
  auto marker = mfemElasticity::ExternalBoundaryMarker(&pmesh);
  marker[0] = 0;
  marker[1] = 1;

  if (myid == 0) {
    marker.Print(cout);
  }

  auto sigma =
      FunctionCoefficient([](const Vector& pt) { return pt[0] * pt[1]; });
  b.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(sigma), marker);
  b.Assemble();

  // Set up the bilinear form
  auto lambda = ConstantCoefficient(1);
  auto mu = ConstantCoefficient(1);
  auto a = ParBilinearForm(&fespace);
  a.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
  a.Assemble();

  // Form the Linear system.
  HypreParMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  // Set the preconditioner
  auto prec = HypreBoomerAMG();
  prec.SetElasticityOptions(&fespace);

  // Set the solver.
  auto solver = CGSolver(MPI_COMM_WORLD);
  solver.SetPreconditioner(prec);
  solver.SetOperator(A);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  // Set the rigid body solver.
  auto rigidBodySolver =
      mfemElasticity::RigidBodySolver(MPI_COMM_WORLD, &fespace);
  rigidBodySolver.SetSolver(solver);

  // Solve and recover the solution.
  rigidBodySolver.Mult(B, X);
  a.RecoverFEMSolution(X, b, x);

  // =====================================================================
  // POST-PROCESSING: Calculate and remove the physical centroid shift
  // =====================================================================
  if (myid == 0) {
    cout << "Calculating physical centroid shift in parallel..." << endl;
  }

  ParBilinearForm m(&fespace);
  m.AddDomainIntegrator(new VectorMassIntegrator());
  m.Assemble();
  m.Finalize();

  // ParallelAssemble returns a HypreParMatrix representing true global physics
  HypreParMatrix* M = m.ParallelAssemble();

  Vector shift(dim);
  Vector x_true(fespace.GetTrueVSize());
  x.GetTrueDofs(x_true);  // Extract true DOFs safely across distributed memory

  for (int d = 0; d < dim; d++) {
    Vector dir(dim);
    dir = 0.0;
    dir(d) = 1.0;
    VectorConstantCoefficient dir_coeff(dir);

    ParGridFunction const_vec(&fespace);
    const_vec.ProjectCoefficient(dir_coeff);

    Vector const_vec_true(fespace.GetTrueVSize());
    const_vec.GetTrueDofs(const_vec_true);

    Vector M_x_true(fespace.GetTrueVSize());
    M->Mult(x_true, M_x_true);

    // MPI_COMM_WORLD ensures we sum the dot product properly across all
    // processors
    double int_x = mfem::InnerProduct(MPI_COMM_WORLD, const_vec_true, M_x_true);

    Vector M_const_true(fespace.GetTrueVSize());
    M->Mult(const_vec_true, M_const_true);
    double volume =
        mfem::InnerProduct(MPI_COMM_WORLD, const_vec_true, M_const_true);

    shift(d) = int_x / volume;
  }

  if (myid == 0) {
    cout << "Calculated Shift (x, y): " << shift(0) << ", " << shift(1) << endl;
  }

  // Subtract the calculated shift
  for (int d = 0; d < dim; d++) {
    Vector dir(dim);
    dir = 0.0;
    dir(d) = -shift(d);
    VectorConstantCoefficient shift_coeff(dir);

    ParGridFunction shift_gf(&fespace);
    shift_gf.ProjectCoefficient(shift_coeff);

    x += shift_gf;
  }

  delete M;  // Clean up the allocated HypreParMatrix
  // =====================================================================

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
    sol_sock << "keys Rjlvvvvvmm\n" << flush;
  } else {
    sol_sock << "keys m\n" << flush;
  }
}