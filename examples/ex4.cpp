#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>  // For std::numbers::pi_v

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;
using std::numbers::pi;

int main(int argc, char *argv[]) {
  // === Set default options and parse command-line arguments ===

  // Set the inner and outer radii
  real_t inner_radius = 1;
  real_t outer_radius = 2;

  // Default mesh files
  string mesh_file2 = "../data/disk2.msh";  // 2D unit disk
  string mesh_file3 = "../data/ball.msh";   // 3D unit ball
  // Default finite element polynomial order
  int order = 2;
  // Default number of uniform mesh refinements
  int refinement = 0;
  // Default problem dimension
  int dim = 2;
  // Set the mapped outer radius
  real_t mapped_outer_radius = 10;
  // Plot just the inner domain
  int plot_submesh = 0;
  // Plot the residual
  int res = 0;

  // Initialize MFEM's options parser
  OptionsParser args(argc, argv);
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&refinement, "-r", "--refinement",
                 "number of  mesh refinements");
  args.AddOption(&dim, "-d", "--dim", "dimension of problem (2 or 3)");
  args.AddOption(&mapped_outer_radius, "-c", "--c", "Mapped outer radius");
  args.AddOption(&plot_submesh, "-sub", "--submesh",
                 "Plot just the inner domain");
  args.AddOption(&res, "-res", "--residual",
                 "plot residual from extact solution");

  // Parse the arguments
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // === Load and refine the mesh ===

  // Read the mesh from file based on the dimension
  auto mesh = Mesh(dim == 2 ? mesh_file2 : mesh_file3, 1, 1);
  {
    // Apply uniform refinements if requested
    for (int l = 0; l < refinement; l++) {
      mesh.UniformRefinement();
    }
  }

  // === Define domain and boundary markers ===

  // Set up marker array for inner domain.
  auto inner_marker = Array<int>(mesh.attributes.Max());
  inner_marker = 0;  // Set all to inactive
  inner_marker[0] = 1;

  // Set up marker for outer domain.
  auto outer_marker = Array<int>(mesh.attributes.Max());
  outer_marker = 0;  // Set all to inactive
  outer_marker[1] = 1;

  // Set up marker for interior boundary.
  auto interior_bdr_marker = Array<int>(mesh.bdr_attributes.Max());
  interior_bdr_marker = 0;  // Set all to inactive
  interior_bdr_marker[0] = 1;

  // Set up marker for exterior boundary.
  auto exterior_bdr_marker = Array<int>(mesh.bdr_attributes.Max());
  exterior_bdr_marker = 0;  // Set all to inactive
  exterior_bdr_marker[1] = 1;

  // ===  Set up Finite Element spaces ===

  // H1_FECollection: Standard continuous "H^1" finite elements (e.g., for
  // potentials)
  auto H1 = H1_FECollection(order, dim);
  // Scalar FE space (for the solution 'phi' and 'zeta')
  auto fes = FiniteElementSpace(&mesh, &H1);
  // Vector FE space (will be used later to deform the mesh nodes)
  auto vfes = FiniteElementSpace(&mesh, &H1, dim);

  // ===  Set up the standard (untransformed) Laplace problem ===

  // Get the list of "true" degrees of freedom (T-Dofs) on the essential
  // boundary
  auto ess_tdof_list = Array<int>();

  auto xi = VectorFunctionCoefficient(
      dim, [inner_radius, outer_radius, mapped_outer_radius](const Vector &x,
                                                             Vector &y) {
        auto epsilon = inner_radius * (outer_radius - inner_radius) /
                       (mapped_outer_radius - inner_radius);
        auto r = x.Norml2();
        y = x;
        if (r > inner_radius) {
          auto fac = (outer_radius - inner_radius + epsilon) /
                     (outer_radius - r + epsilon);
          y *= fac * inner_radius / r;
        }
      });

  // Create a GridFunction 'phi' to hold the FE solution
  auto phi = GridFunction(&fes);
  phi = 0.0;

  // Set up the bilinear form  for the weak form of the Laplace equation:
  auto a = BilinearForm(&fes);
  a.AddDomainIntegrator(new TransformedDiffusionIntegrator(xi));
  a.Assemble();

  auto eps = ConstantCoefficient(0.01);
  auto as = BilinearForm(&fes);
  as.AddDomainIntegrator(new TransformedDiffusionIntegrator(xi));
  as.AddDomainIntegrator(new MassIntegrator(eps));
  as.Assemble();

  // Set up the linear form 'b' for the right-hand side (RHS)
  auto f =
      FunctionCoefficient([inner_radius](const Vector &x) { return x[0]; });
  auto b = LinearForm(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(f), inner_marker);
  b.Assemble();
  b *= -4 * pi;

  // Form the final linear system A*X = B
  SparseMatrix A;
  Vector B, X;
  // This function modifies A and B to incorporate the Dirichlet BCs
  a.FormLinearSystem(ess_tdof_list, phi, b, A, X, B);

  // Set up a preconditioner (Gauss-Seidel smoother)
  SparseMatrix As;
  as.FormSystemMatrix(ess_tdof_list, As);
  auto P = GSSmoother(As);

  // Set up the solver (Conjugate Gradient)
  auto solver = CGSolver();
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);  // Print solver progress
  solver.SetPreconditioner(P);
  solver.SetOperator(A);

  auto orthoSolver = OrthoSolver();
  orthoSolver.SetSolver(solver);
  orthoSolver.Mult(B, X);

  // Recover the final solution 'phi' (GridFunction) from the raw vector 'X'
  a.RecoverFEMSolution(X, b, phi);

  auto exact_solution = TransformedFunctionCoefficient(
      xi, [inner_radius, outer_radius, mapped_outer_radius](const Vector &x) {
        auto r = x.Norml2();
        auto a = inner_radius;
        if (r <= inner_radius) {
          return -0.5 * pi * x[0] * (2 * a * a - r * r) / a;
        } else {
          return -0.5 * pi * std::pow(a, 3) * x[0] / (r * r);
        }
      });

  auto l = LinearForm(&fes);
  auto z = GridFunction(&fes);
  z = 1.0;
  auto one = ConstantCoefficient(1);
  l.AddDomainIntegrator(new DomainLFIntegrator(one));
  l.Assemble();
  auto area = l(z);
  l /= area;
  auto pphi = l(phi);
  phi -= pphi;

  if (plot_submesh) {
    auto submesh = SubMesh::CreateFromDomain(mesh, inner_marker);

    auto fes_inner = FiniteElementSpace(&submesh, &H1);
    auto phi_inner = GridFunction(&fes_inner);
    submesh.Transfer(phi, phi_inner);

    if (res) {
      auto tmp = GridFunction(&fes_inner);
      tmp.ProjectCoefficient(exact_solution);
      phi_inner -= tmp;
    }

    // === Visualize the standard solution ===
    char vishost[] = "localhost";
    int visport = 19916;

    auto phi_sock = socketstream(vishost, visport);
    phi_sock.precision(8);
    // Send the mesh and the solution 'phi' to GLVis
    phi_sock << "solution\n"
             << submesh << phi_inner << "window_title 'phi'" << flush;
    if (dim == 2) {
      phi_sock << "keys Rjlcb\n" << flush;  // 2D viewing keys
    } else {
      phi_sock << "keys RRRjlci zZ\n" << flush;  // 3D viewing keys
    }

  } else {
    if (res) {
      auto tmp = GridFunction(&fes);
      tmp.ProjectCoefficient(exact_solution);
      phi -= tmp;
    }

    // === Visualize the standard solution ===
    char vishost[] = "localhost";
    int visport = 19916;

    auto phi_sock = socketstream(vishost, visport);
    phi_sock.precision(8);
    // Send the mesh and the solution 'phi' to GLVis
    phi_sock << "solution\n" << mesh << phi << "window_title 'phi'" << flush;
    if (dim == 2) {
      phi_sock << "keys Rjlcb\n" << flush;  // 2D viewing keys
    } else {
      phi_sock << "keys RRRjlci zZ\n" << flush;  // 3D viewing keys
    }
  }

  // === Deform the mesh and visualize the "pushed-forward" solution ===

  /*
  mesh.SetNodalFESpace(&vfes);
  auto *x = mesh.GetNodes();
  auto y = GridFunction(&vfes);
  y.ProjectCoefficient(xi);
  *x = y;

  auto phiT_sock = socketstream(vishost, visport);
  phiT_sock.precision(8);
  // Send the mesh and the solution 'phi' to GLVis
  phiT_sock << "solution\n" << mesh << phi << "window_title 'phi'" << flush;
  if (dim == 2) {
    phiT_sock << "keys Rjlbc\n" << flush;  // 2D viewing keys
  } else {
    phiT_sock << "keys RRRjlci zZ\n" << flush;  // 3D viewing keys
  }
  */
}