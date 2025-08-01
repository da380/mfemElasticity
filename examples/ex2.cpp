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

[-m, --mesh]: The mesh. Either 2D or 3D, but must have the attributes as
              described above. Default is circular_offset.msh in the data
              directory.

[-o, --order]: The polynomial order used in the calculations. Default is 1.

[-r, --refinement]: The number of times to refine the mesh. Default it 0.

[-deg, --degree]: The degree used for the DtN mapping. Default is 8.

[-res, --residual]: If equal to 1, the output is the residual between the
                    numerical solution and an exact one. Default is 0.

[-mth, --method]: The solution method. 0 = homogeneous Neumann, 1 = DtN,
                  2 = Multipole. Default is 0.

[-lin, --linearised ]: Choice of force term. 0 = reference problem with uniform
                       density, 1 = linearised problem with rigid translation.
                       Default is 0.

*******************************************************************************/

#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>

#include "mfem.hpp"
#include "mfemElasticity.hpp"
#include "uniform_sphere.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

constexpr real_t pi = std::numbers::pi_v<mfem::real_t>;

int main(int argc, char *argv[]) {
  // Set default options.
  const char *mesh_file = "../data/circular_offset.msh";
  int order = 1;
  int refinement = 0;
  int degree = 8;
  int residual = 0;
  int method = 0;
  int linearised = 0;

  // Deal with options.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&refinement, "-r", "--refinement",
                 "number of  mesh refinements");
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
  args.PrintOptions(cout);

  // Read in mesh.
  auto mesh = Mesh(mesh_file, 1, 1);
  auto dim = mesh.Dimension();
  {
    for (int l = 0; l < refinement; l++) {
      mesh.UniformRefinement();
    }
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

  // Properties of the full mesh.
  auto c2 = MeshCentroid(&mesh);
  auto [found2, same2, r2] = SphericalBoundaryRadius(&mesh, c2);

  // If residual from exact solution required, check mesh is appropriate.
  if (residual) {
    assert(found1 == 1 && same1 == 1);
  }

  // Set up the finite element spaces.
  auto L2 = L2_FECollection(order - 1, dim);
  auto H1 = H1_FECollection(order, dim);

  // Space for the potential.
  auto fes = FiniteElementSpace(&mesh, &H1);
  cout << "Number of finite element unknowns: " << fes.GetTrueVSize() << endl;

  // For multipole method we need a discontinuous L2 space.
  std::unique_ptr<FiniteElementSpace> dfes;
  if (method == 2) {
    dfes = std::make_unique<FiniteElementSpace>(&mesh, &L2);
  }

  // For the linearised problem, we need a discontinous vector L2 space.
  std::unique_ptr<FiniteElementSpace> vfes;
  if (linearised == 1) {
    vfes = std::make_unique<FiniteElementSpace>(&mesh, &L2, dim);
  }

  // Assemble the binlinear form for Poisson's equation.
  auto a = BilinearForm(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  // Assemble mass-shifted binlinear form for preconditioning.
  auto eps = ConstantCoefficient(0.01);
  auto as = BilinearForm(&fes);
  as.AddDomainIntegrator(new DiffusionIntegrator());
  as.AddDomainIntegrator(new MassIntegrator(eps));
  as.Assemble();

  // Set the density coefficient.
  auto rho_coeff1 = ConstantCoefficient(1);
  auto rho_coeff = PWCoefficient();
  rho_coeff.UpdateCoefficient(1, rho_coeff1);

  // Set gridfunction for the potential.
  auto x = GridFunction(&fes);

  // Set up the linear form for the rhs.
  auto b = LinearForm(&fes);

  // Set the constant displacement vector for the linearised problem.
  auto uv = Vector(dim);
  uv = 1.0;

  // Project displacement to a gridfunction if necessary.
  std::unique_ptr<GridFunction> u;
  if (linearised == 1) {
    auto uCoeff1 = VectorConstantCoefficient(uv);
    auto uCoeff = PWVectorCoefficient(dim);
    uCoeff.UpdateCoefficient(1, uCoeff1);
    u = std::make_unique<GridFunction>(vfes.get());
    u->ProjectCoefficient(uCoeff);
  }

  if (linearised == 0) {
    // For reference problem set up the linear form.
    b.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff));
    b.Assemble();

    // For the DtN method in 2D add the additoinal term to the rhs.
    if (method == 1 and dim == 2) {
      x = 1.0;
      auto mass = b(x);
      auto l = LinearForm(&fes);
      auto one = ConstantCoefficient(1);
      auto boundary_marker = ExternalBoundaryMarker(&mesh);
      l.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), boundary_marker);
      l.Assemble();
      auto length = l(x);
      b.Add(-mass / length, l);
    }
  } else {
    // For the linearised problem, form the mixed bilinear form and map the
    // displacement to the rhs.
    auto d = MixedBilinearForm(&fes, vfes.get());
    d.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(rho_coeff));
    d.Assemble();
    d.MultTranspose(*u, b);
  }

  // If using the multipole method, modify the rhs.
  if (method == 2) {
    if (linearised == 0) {
      auto c = PoissonMultipoleOperator(dfes.get(), &fes, degree);
      auto rhof = GridFunction(dfes.get());
      rhof.ProjectCoefficient(rho_coeff);
      c.AddMult(rhof, b, -1);
    } else {
      auto c = PoissonLinearisedMultipoleOperator(vfes.get(), &fes, degree);
      c.AddMult(*u, b, -1);
    }
  }

  // Scale the linear form.
  b *= -4 * pi;

  // Set up the linear system
  x = 0.0;
  Array<int> ess_tdof_list{};
  SparseMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  // Set up the preconditioner.
  SparseMatrix As;
  as.FormSystemMatrix(ess_tdof_list, As);
  auto P = GSSmoother(As);

  auto solver = CGSolver();
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  if (method == 1) {
    auto c = PoissonDtNOperator(&fes, degree);
    auto D = SumOperator(&A, 1, &c, 1, false, false);
    solver.SetOperator(D);
    solver.SetPreconditioner(P);
    if (dim == 2) {
      auto orthoSolver = OrthoSolver();
      orthoSolver.SetSolver(solver);
      orthoSolver.Mult(B, X);
    } else {
      solver.Mult(B, X);
    }
  } else {
    solver.SetOperator(A);
    solver.SetPreconditioner(P);
    auto orthoSolver = OrthoSolver();
    orthoSolver.SetSolver(solver);
    orthoSolver.Mult(B, X);
  }

  if (residual == 1) {
    // Subtract exact solution.
    auto y = GridFunction(&fes);
    auto exact = UniformSphereSolution(dim, c1, r1);
    auto exact_coeff =
        linearised == 0 ? exact.Coefficient() : exact.LinearisedCoefficient(uv);
    y.ProjectCoefficient(exact_coeff);
    x -= y;
  }

  // Remove mean from the solution.
  {
    auto l = LinearForm(&fes);
    auto z = GridFunction(&fes);
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
  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);

  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  x.Save(sol_ofs);

  // Visualise if glvis is open.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock.precision(8);
  sol_sock << "solution\n" << mesh << x << flush;
  if (dim == 2) {
    sol_sock << "keys Rjlbc\n" << flush;
  } else {
    sol_sock << "keys RRRilmc\n" << flush;
  }
}
