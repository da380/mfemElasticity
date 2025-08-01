/******************************************************************************

Solves the Poisson equation on a whole space using one of the folowing methods:

1. A homogeneous Neumann condition is applied on the meshes exterior boundary.
This approach does not require the exterior boundary to take a specific form,
but it will only be accurate if the boundary is sufficiently far from regions
with non-zero density.

2. A Dirichlet-to-Neumann mapping is applied on the exterior boundary. This
results in an additional bilinear form being added to the weak form of the
equations along, for 2D problems, with a modification to the force term. This
method requires that the exterior boundary is spherical relative to the mesh's
centroid.

3. A multipole expansion is used to relate the density to the Neumann condition
on the exterior boundary. The result is a modification to the rhs. This method
again requires that the exterior boundary is spherical relative to the mesh's
centroid.

The force term for the equations can take one of two forms:

1. A uniform density (with units chosen such that it equals one) within the
meshes first attribute, and zero density in the rest of the mesh.
2. The density perturbation associated with a rigid translation of the density
structure in option 1.

For solution methods 1 and 2 there is no additional change needed to handle the
linearised problem, while for method 3 a modified form of the multipole
expansion is required.

The residual between the computed solution and an analytical solution can be
output, but in this case it is necessary for the mesh to comprise two
attributes, with attribute 1 being spherical about its centroid.

Note that the calculations are done in units for which G = 1.

This is the parallel version of ex2.

[-m, --mesh]: The mesh. Either 2D or 3D, but must have the attributes as
              described above. Default is circular_offset.msh in the data
              directory.

[-o, --order]: The polynomial order used in the calculations. Default is 1.

[-sr, --serial_refinement]: The number of times to refine the mesh in serial.
                            The default it 0.

[-pr, --parallel_refinement]: The number of times to refine the mesh in
                              parallel. The default it 0.

[-deg, --degree]: The degree used for the DtN mapping. Default is 8.

[-res, --residual]: If equal to 1, the output is the residual between the
                    numerical solution and an exact one. Default is 0.

[-mth, --method]: The solution method. 0 = homogeneous Neumann, 1 = DtN,
                  2 = Multipole. Default is 0.

[-lin, --linearised ]: Choice of force term. 0 = reference problem with uniform
                       density, 1 = linearised problem with rigid translation.
                       Default is 0.

*******************************************************************************/

/******************************************************************************

Solves the Poisson equation on a whole space through use of a Dirichlet to
Neumann mapping as implemented in the PoissonDtNOperator class.

This is the parallel version of ex2 that makes use of MPI.

The mesh is required to have a spherical exterior boundary. It must have two
attributes. Attirbute 1 is an inner domain with spherically boundary on in which
the density is equal to 1. Attribute 2 is the remainder of the domain, and here
the density is equal to zero. The boundary between attirbutes 1 and 2 is
labelled 1, while the exterior boundary is lablled 2. It is on the exterior
boundary that the DtN mapping acts, this being through the addition of a
bilinear form to the weak form of the equations.

Note that the calculations are done in units for which G = 1.

[-m, --mesh]: The mesh. Either 2D or 3D, but must have the attributes as
              described above. Default is circular_offset.msh in the data
              directory.

[-o, --order]: The polynomial order used in the calculations. Default is 1.



[-deg, --degree]: The degree used for the DtN mapping. Default is 4.

[-res, --residual]: If equal to 1, the output is the residual between the
                    numerical solution and an exact one. Default is 0.

*******************************************************************************/

#include <fstream>
#include <iostream>
#include <numbers>

#include "mfem.hpp"
#include "mfemElasticity.hpp"
#include "uniform_sphere.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

constexpr real_t pi = std::numbers::pi_v<real_t>;

int main(int argc, char *argv[]) {
  // Initialise MPI and Hypre.
  Mpi::Init();
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  // Set default options.
  const char *mesh_file = "../data/circular_offset.msh";

  int order = 1;
  int serial_refinement = 0;
  int parallel_refinement = 0;
  int degree = 4;
  int residual = 0;
  int method = 0;
  int linearised = 0;

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
  args.AddOption(&method, "-mth", "--method",
                 "Solution method: 0 = Neuman, 1 = DtN, 2 = multipole.");
  args.AddOption(&linearised, "-lin", "--linearised",
                 "Solve reference (0) or linearised (1) problem.");

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

  // Form the parallel mesh.
  auto pmesh = ParMesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  {
    for (int l = 0; l < parallel_refinement; l++) {
      pmesh.UniformRefinement();
    }
  }

  // Properties of the first attribute.
  auto dom_marker = Array<int>(pmesh.attributes.Max());
  dom_marker = 0;
  dom_marker[0] = 1;
  auto bdr_marker = Array<int>(pmesh.bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[0] = 1;
  auto c1 = MeshCentroid(&pmesh, dom_marker);
  auto [found1, same1, r1] = SphericalBoundaryRadius(&pmesh, bdr_marker, c1);

  // Properties of the full mesh.
  auto c2 = MeshCentroid(&pmesh);
  auto [found2, same2, r2] = SphericalBoundaryRadius(&pmesh, c2);

  // If residual from exact solution required, check mesh is appropriate.
  if (residual) {
    assert(found1 == 1 && same1 == 1);
  }

  // Set up the finite element spaces.
  auto L2 = L2_FECollection(order - 1, dim);
  auto H1 = H1_FECollection(order, dim);

  // Space for the potential.
  auto fes = ParFiniteElementSpace(&pmesh, &H1);
  HYPRE_BigInt size = fes.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  // For multipole method we need a discontinuous L2 space.
  std::unique_ptr<ParFiniteElementSpace> dfes;
  if (method == 2) {
    dfes = std::make_unique<ParFiniteElementSpace>(&pmesh, &L2);
  }

  // For the linearised problem, we need a discontinous vector L2 space.
  std::unique_ptr<ParFiniteElementSpace> vfes;
  if (linearised == 1) {
    vfes = std::make_unique<ParFiniteElementSpace>(&pmesh, &L2, dim);
  }

  // Assemble the binlinear form for Poisson's equation.
  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  // Assemble mass-shifted binlinear form for preconditioning.
  auto eps = ConstantCoefficient(0.001);
  auto as = ParBilinearForm(&fes);
  as.AddDomainIntegrator(new DiffusionIntegrator());
  as.AddDomainIntegrator(new MassIntegrator(eps));
  as.Assemble();

  // Set the density coefficient.
  auto rho_coeff1 = ConstantCoefficient(1);
  auto rho_coeff = PWCoefficient();
  rho_coeff.UpdateCoefficient(1, rho_coeff1);

  // Set gridfunction for the potential.
  auto x = ParGridFunction(&fes);

  // Set up the linear form for the rhs.
  ParLinearForm b(&fes);

  // Set the constant displacement vector for the linearised problem.
  auto uv = Vector(dim);
  uv = 1.0;

  // Project displacement to a gridfunction if necessary.
  std::unique_ptr<ParGridFunction> u;
  if (linearised == 1) {
    auto uCoeff1 = VectorConstantCoefficient(uv);
    auto uCoeff = PWVectorCoefficient(dim);
    uCoeff.UpdateCoefficient(1, uCoeff1);
    u = std::make_unique<ParGridFunction>(vfes.get());
    u->ProjectCoefficient(uCoeff);
  }

  if (linearised == 0) {
    // For reference problem set up the linear form.
    b.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff));
    b.Assemble();

    if (method == 1 && dim == 2) {
      x = 1.0;
      auto mass = b(x);
      auto l = ParLinearForm(&fes);
      auto one = ConstantCoefficient(1);
      auto boundary_marker = ExternalBoundaryMarker(&pmesh);
      l.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), boundary_marker);
      l.Assemble();
      auto length = l(x);
      b.Add(-mass / length, l);
    }
  } else {
    // For the linearised problem, form the mixed bilinear form and map the
    // displacement to the rhs.
    auto d = ParMixedBilinearForm(&fes, vfes.get());
    d.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(rho_coeff));
    d.Assemble();
    d.MultTranspose(*u, b);
  }

  // If using the multipole method, modify the rhs.
  if (method == 2) {
    if (linearised == 0) {
      auto c =
          PoissonMultipoleOperator(MPI_COMM_WORLD, dfes.get(), &fes, degree);
      auto rhof = GridFunction(dfes.get());
      rhof.ProjectCoefficient(rho_coeff);
      c.AddMult(rhof, b, -1);
    } else {
      auto c = PoissonLinearisedMultipoleOperator(MPI_COMM_WORLD, vfes.get(),
                                                  &fes, degree);
      c.AddMult(*u, b, -1);
    }
  }

  // Scale the linear form.
  b *= -4 * pi;

  // Set up the linear system
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
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  if (method == 1) {
    auto c = PoissonDtNOperator(MPI_COMM_WORLD, &fes, degree);
    auto C = c.RAP();
    auto D = SumOperator(&A, 1, &C, 1, false, false);
    solver.SetOperator(D);
    solver.SetPreconditioner(P);

    if (dim == 2) {
      auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
      orthoSolver.SetSolver(solver);
      orthoSolver.Mult(B, X);
    } else {
      solver.Mult(B, X);
    }
  } else {
    solver.SetOperator(A);
    solver.SetPreconditioner(P);
    auto orthoSolver = OrthoSolver(MPI_COMM_WORLD);
    orthoSolver.SetSolver(solver);
    orthoSolver.Mult(B, X);
  }

  a.RecoverFEMSolution(X, b, x);

  if (residual == 1) {
    // Subtract exact solution.
    auto y = ParGridFunction(&fes);
    auto exact = UniformSphereSolution(dim, c1, r1);
    auto exact_coeff =
        linearised == 0 ? exact.Coefficient() : exact.LinearisedCoefficient(uv);
    y.ProjectCoefficient(exact_coeff);
    x -= y;
  }

  // Remove the mean from the solution.
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
