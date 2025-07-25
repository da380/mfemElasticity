/******************************************************************************

Solves the Poisson equation associated with a given linearised displacement
using a multipole expansion to account for the exterior domain.

This is the parallel version of ex5 that makes use of MPI.

The mesh is required to have a spherical exterior boundary. It must have two
attributes. Attirbute 1 is an inner domain with spherically boundary on in which
the density is equal to 1. Attribute 2 is the remainder of the domain, and here
the density is equal to zero. The boundary between attirbutes 1 and 2 is
labelled 1, while the exterior boundary is lablled 2. It is on the exterior
boundary that the multipole exapansion is used to express the required Neumann
conditions. This results in a modified right hand side within the resulting
linear system.

The displacement is taken to be a rigid translation of attribute 1, in which
case the exact solution can be readily determined.

Note that the calculations are done in units for which G = 1.

[-m, --mesh]: The mesh. Either 2D or 3D, but must have the attributes as
              described above. Default is circular_offset.msh in the data
              directory.

[-o, --order]: The polynomial order used in the calculations. Default is 1.

[-sr, --serial_refinement]: The number of times to refine the mesh in serial.
                            The default it 0.

[-pr, --parallel_refinement]: The number of times to refine the mesh in
                              parallel. The default it 0.

[-deg, --degree]: The degree used for the DtN mapping. Default is 4.

[-res, --residual]: If equal to 1, the output is the residual between the
                    numerical solution and an exact one. Default is 0.

*******************************************************************************/

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
  // Initialise MPI and Hypre.
  Mpi::Init();
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

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
  if (myid == 0) {
    args.PrintOptions(cout);
  }

  // Read in mesh in serial.
  auto mesh = Mesh(mesh_file, 1, 1);
  auto dim = mesh.Dimension();
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

  assert(pmesh.attributes.Max() == 2);

  // Get centroid and radius for inner domain.
  auto c1 = MeshCentroid(&pmesh, Array<int>{1, 0});
  auto [found1, same1, r1] = BoundaryRadius(&pmesh, Array<int>{1, 0}, c1);
  assert(found1 == 1 && same1 == 1);

  // Get centroid and radius for combined domain.
  auto c2 = MeshCentroid(&pmesh, Array<int>{1, 1});
  auto [found2, same2, r2] = BoundaryRadius(&pmesh, Array<int>{0, 1}, c2);
  assert(found2 == 1 && same2 == 1);

  // Set up the scalar finite element space.
  auto H1 = H1_FECollection(order, dim);
  ParFiniteElementSpace fes(&pmesh, &H1);
  HYPRE_BigInt size = fes.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  // Set up the vector finite element space.
  auto L2 = L2_FECollection(order - 1, dim);
  auto vfes = ParFiniteElementSpace(&pmesh, &L2, dim);

  // Set the density coefficient.
  auto rho_coeff1 = ConstantCoefficient(1);
  auto rho_coeff2 = ConstantCoefficient(0);

  auto attr = Array<int>{1, 2};
  auto coeffs = Array<Coefficient *>{&rho_coeff1, &rho_coeff2};
  auto rho_coeff = PWCoefficient(attr, coeffs);

  // Set up the displacement
  auto uv = Vector(dim);
  uv = 0.0;
  uv[0] = 1;
  uv[1] = 1;
  if (dim == 3) {
    uv[2] = 1;
  }

  auto uCoeff1 = VectorConstantCoefficient(uv);
  auto uCoeff = PWVectorCoefficient(dim);
  uCoeff.UpdateCoefficient(1, uCoeff1);
  auto u = ParGridFunction(&vfes);
  u.ProjectCoefficient(uCoeff);

  // Set up the mixed bilinearform
  auto d = ParMixedBilinearForm(&fes, &vfes);
  d.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(rho_coeff));
  d.Assemble();

  // Act the transposed form to generate rhs.
  auto b = ParGridFunction(&fes);
  d.MultTranspose(u, b);

  // Assemble the binlinear form for Poisson's equation.
  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  // Assemble mass-shifted binlinear form for preconditioning.
  auto eps = ConstantCoefficient(0.01);
  auto as = ParBilinearForm(&fes, &a);
  as.AddDomainIntegrator(new MassIntegrator(eps));
  as.Assemble();

  // Set up the multipole operator.
  auto c =
      PoissonLinearisedMultipoleOperator(MPI_COMM_WORLD, &vfes, &fes, degree);
  c.AddMult(u, b, -1);

  // Scale the rhs.
  b *= -4 * pi;

  // Set up the linear system
  auto x = ParGridFunction(&fes);
  x = 0.0;
  Array<int> ess_tdof_list{};
  HypreParMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  // Form the preconditioner, using the mass-matrix to make it
  // positive-definite.
  HypreParMatrix As;
  as.FormSystemMatrix(ess_tdof_list, As);
  auto P = HypreBoomerAMG(As);

  // Set up the solver.
  auto solver = CGSolver(MPI_COMM_WORLD);
  solver.SetOperator(A);
  solver.SetPreconditioner(P);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  // Sovler the linear system.

  auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
  orthoSolver.SetSolver(solver);
  orthoSolver.Mult(B, X);
  a.RecoverFEMSolution(X, b, x);

  auto exact = UniformSphereSolution(dim, c1, r1);
  auto exact_coeff = exact.LinearisedCoefficient(uv);

  auto y = ParGridFunction(&fes);
  y.ProjectCoefficient(exact_coeff);

  //  Remove the mean from the solution.
  {
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
    auto py = l(y);
    y -= py;
  }

  if (residual == 1) {
    x -= y;
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
