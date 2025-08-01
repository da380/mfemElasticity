/******************************************************************************
## Poisson Solver for Whole-Space Problems

Solves the Poisson equation (∇²ϕ = -4πGρ, with G=1) on a whole space
by using a finite computational domain with transparent boundary conditions.

This is the parallel version of ex2.

---
### Boundary Condition Methods

1.  **Homogeneous Neumann:** A `∂ϕ/∂n = 0` condition is applied on the
    mesh's exterior boundary. This approach is general but is only
    accurate if the boundary is far from the source density.

2.  **Dirichlet-to-Neumann (DtN):** An exact DtN operator is added to the
    system matrix. For 2D problems, an additional correction is also
    applied to the right-hand-side vector. This method requires a
    spherical exterior boundary.

3.  **Multipole Expansion:** The right-hand-side vector is modified based
    on a multipole expansion of the interior density. This method also
    requires a spherical exterior boundary.

---
### Source Terms

1.  **Reference Problem:** A uniform density (ρ=1) within the mesh's
    first attribute and zero elsewhere.

2.  **Linearised Problem:** The density perturbation caused by a rigid
    translation of the reference density distribution.

---
### Command-Line Options

[-m, --mesh]:       Mesh file. Must have attributes as described above.
[-o, --order]:      Finite element polynomial order. Default is 1.
[-r, --refinement]: Number of uniform mesh refinements. Default is 0.
[-deg, --degree]:   Expansion degree for DtN/Multipole methods. Default is 8.
[-res, --residual]: Set to 1 to output the pointwise error against an exact
                    solution (requires a spherical source). Default is 0.
[-mth, --method]:   Solution method: 0=Neumann, 1=DtN, 2=Multipole.
                    Default is 0.
[-lin, --linearised]: Problem type: 0=Reference, 1=Linearised. Default is 0.

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
