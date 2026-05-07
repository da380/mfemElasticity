#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>  // For std::numbers::pi_v

// Main MFEM library header
#include "mfem.hpp"
// Headers from a custom MFEM-based library (e.g., for elasticity,
// transformations)
#include "mfemElasticity.hpp"

// Use standard C++ and MFEM namespaces
using namespace std;
using namespace mfem;
using namespace mfemElasticity;

//------------------------------------------------------------------------------
//
// PURPOSE:
//
// This example (ex3) demonstrates how to solve a partial differential
// equation (PDE) on a deformed domain using the "transformed domain"
// or "pullback" approach.
//
// The core idea is to perform all calculations on a simple,
// undeformed *reference mesh* (a disk or a ball).
//
// The process is as follows:
//
// 1. **Solve Standard Problem:** First, it solves a standard Laplace
//    equation ($\nabla^2 \phi = 0$) on the reference mesh. This serves
//    as a baseline.
//
// 2. **Define Transformation:** It defines a smooth coordinate
//    transformation (a diffeomorphism, $q(x)$) that deforms the mesh.
//    In this case, it's an interior "twist" that leaves the boundary
//    and center fixed.
//
//
// 3. **Solve Transformed Problem:** It solves the *transformed*
//    Laplace equation on the *same reference mesh*. This is done
//    using the `TransformedDiffusionIntegrator`, which automatically
//    handles the geometric factors (Jacobians) from the
//    transformation $q(x)$. The resulting solution, $\zeta(x)$, is the
//    "pullback" of the physical solution $u$, meaning $\zeta(x) = u(q(x))$.
//
// 4. **Error Calculation:** It computes the L2 error for both the
//    standard solution ($\phi$) and the transformed solution ($\zeta$)
//    by comparing them to their respective exact solutions.
//
// 5. **Visualization:**
//    a. Visualizes $\phi$ on the reference mesh.
//    b. Visualizes $\zeta$ on the reference mesh (the pullback).
//    c. Physically deforms the mesh nodes using the transformation
//       $q(x)$ and visualizes $\zeta$ on this *new* deformed
//       mesh. This shows the final "push-forward" solution $u$ on
//       the actual, physical domain.
//
//------------------------------------------------------------------------------

// Define pi using the mfem::real_t type
constexpr real_t pi = std::numbers::pi_v<mfem::real_t>;

// Define the exact solution for the 2D problem: u(x, y) = x*y
// This will be used for setting Dirichlet boundary conditions and calculating
// error.
real_t exact_solution2(const Vector &x) { return x[0] * x[1]; }

// Define the exact solution for the 3D problem: u(x, y, z) = y*z
real_t exact_solution3(const Vector &x) { return x[1] * x[2]; }

int main(int argc, char *argv[]) {
  // === 1. Set default options and parse command-line arguments ===

  // Default mesh files
  string mesh_file2 = "../data/disk.msh";  // 2D unit disk
  string mesh_file3 = "../data/ball.msh";  // 3D unit ball
  // Default finite element polynomial order
  int order = 2;
  // Default number of uniform mesh refinements
  int refinement = 0;
  // Default rotation angle (in degrees) for the transformation
  int theta = 10;
  // Default problem dimension
  int dim = 2;

  // Initialize MFEM's options parser
  OptionsParser args(argc, argv);
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&refinement, "-r", "--refinement",
                 "number of  mesh refinements");
  args.AddOption(&theta, "-th", "--theta", "rotation angle in degrees");
  args.AddOption(&dim, "-d", "--dim", "dimension of problem (2 or 3)");

  // Parse the arguments
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // === 2. Load and refine the mesh ===

  // Read the mesh from file based on the dimension
  auto mesh = Mesh(dim == 2 ? mesh_file2 : mesh_file3, 1, 1);
  {
    // Apply uniform refinements if requested
    for (int l = 0; l < refinement; l++) {
      mesh.UniformRefinement();
    }
  }

  // === 3. Define domain and boundary markers ===
  // These markers are used to specify which parts of the mesh
  // to apply integrators or boundary conditions to.

  // Mark boundary attribute 1 as active (for Dirichlet BCs)
  auto bdr_marker = Array<int>(mesh.bdr_attributes.Max());
  bdr_marker = 0;     // Set all to inactive
  bdr_marker[0] = 1;  // Set attribute 1 to active

  // === 4. Set up Finite Element spaces ===

  // H1_FECollection: Standard continuous "H^1" finite elements (e.g., for
  // potentials)
  auto H1 = H1_FECollection(order, dim);
  // Scalar FE space (for the solution 'phi' and 'zeta')
  auto fes = FiniteElementSpace(&mesh, &H1);
  // Vector FE space (will be used later to deform the mesh nodes)
  auto vfes = FiniteElementSpace(&mesh, &H1, dim);

  // === 5. Set up the standard (untransformed) Laplace problem ===

  // Get the list of "true" degrees of freedom (T-Dofs) on the essential
  // boundary
  auto ess_tdof_list = Array<int>();
  fes.GetEssentialTrueDofs(bdr_marker, ess_tdof_list);

  // Define a coefficient for the boundary condition
  // This wraps the appropriate C++ function (exact_solution2 or
  // exact_solution3)
  auto g = FunctionCoefficient(
      std::function(dim == 2 ? exact_solution2 : exact_solution3));

  // Create a GridFunction 'phi' to hold the FE solution
  auto phi = GridFunction(&fes);
  // Project the exact solution 'g' onto 'phi'. This sets the values
  // for the Dirichlet boundary conditions.
  phi.ProjectCoefficient(g);

  // Set up the bilinear form 'a' for the weak form of the Laplace equation:
  // a(u, v) = \int_{\Omega} \nabla u \cdot \nabla v dx
  auto a = BilinearForm(&fes);
  // Add the standard diffusion integrator (gradient-gradient term)
  a.AddDomainIntegrator(new DiffusionIntegrator());
  // Assemble the stiffness matrix 'A'
  a.Assemble();

  // Set up the linear form 'b' for the right-hand side (RHS)
  // b(v) = \int_{\Omega} f * v dx. Here f = 0.
  auto b = LinearForm(&fes);
  b.Assemble();  // Assemble the load vector 'B' (will be all zeros)

  // Form the final linear system A*X = B
  SparseMatrix A;
  Vector B, X;
  // This function modifies A and B to incorporate the Dirichlet BCs
  a.FormLinearSystem(ess_tdof_list, phi, b, A, X, B);

  // Set up a preconditioner (Gauss-Seidel smoother)
  auto P = GSSmoother(A);

  // Set up the solver (Conjugate Gradient)
  auto solver = CGSolver();
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);  // Print solver progress
  solver.SetPreconditioner(P);
  solver.SetOperator(A);

  // Solve the system A*X = B
  solver.Mult(B, X);

  // Recover the final solution 'phi' (GridFunction) from the raw vector 'X'
  a.RecoverFEMSolution(X, b, phi);

  // === 6. Visualize the standard solution ===
  char vishost[] = "localhost";
  int visport = 19916;

  auto phi_sock = socketstream(vishost, visport);
  phi_sock.precision(8);
  // Send the mesh and the solution 'phi' to GLVis
  phi_sock << "solution\n" << mesh << phi << "window_title 'phi'" << flush;
  if (dim == 2) {
    phi_sock << "keys Rjlmmc\n" << flush;  // 2D viewing keys
  } else {
    phi_sock << "keys RRRjlci zZ\n" << flush;  // 3D viewing keys
  }

  // === 7. Set up the coordinate transformation (Diffeomorphism) ===
  // This defines a smooth, invertible map q(x) that deforms the domain.
  // We will solve the *transformed* PDE on the *original* (reference) mesh.

  auto qv =
      VectorFunctionCoefficient(dim, [theta, dim](const Vector &x, Vector &v) {
        using std::sin, std::cos;
        v.SetSize(x.Size());
        auto r = x.Norml2();  // Get radius |x|

        // Define a "bump" function: f(r) = 4*r^2*(1-r^2)
        // This function is 0 at r=0 (center) and r=1 (boundary).
        // It creates a transformation that only affects the *interior*.
        auto f = 4 * r * r * (1 - r * r);

        // Scale the rotation angle by this bump function.
        // The rotation (twist) will be maximum inside and zero at the boundary.
        auto theta_scaled = (pi / 180) * f * theta;

        if (dim == 2) {
          // Apply a 2D rotation matrix
          v[0] = cos(theta_scaled) * x[0] + sin(theta_scaled) * x[1];
          v[1] = -sin(theta_scaled) * x[0] + cos(theta_scaled) * x[1];
        } else {
          // Apply a 3D rotation matrix (around the x-axis)
          v[0] = x[0];
          v[1] = cos(theta_scaled) * x[1] + sin(theta_scaled) * x[2];
          v[2] = -sin(theta_scaled) * x[1] + cos(theta_scaled) * x[2];
        }
      });

  // === 8. Set up the transformed Laplace problem ===

  // Create a new GridFunction 'zeta' for the transformed solution
  auto zeta = GridFunction(&fes);
  // Project the *same* boundary condition 'g'. We are solving on the
  // reference domain, so the boundary values on the reference boundary
  // are still given by g(x).
  zeta.ProjectCoefficient(g);

  // Set up the transformed bilinear form 'at'
  auto at = BilinearForm(&fes);
  // This is the key: TransformedDiffusionIntegrator.
  // It automatically computes the "pullback" of the diffusion operator
  // from the deformed domain to the reference domain using the
  // Jacobian of the transformation.
  at.AddDomainIntegrator(new TransformedDiffusionIntegrator(qv));
  at.Assemble();  // Assemble the transformed stiffness matrix 'At'

  // Set up an empty linear form (RHS is still 0)
  auto bt = LinearForm(&fes);
  bt.Assemble();

  // Set up the linear system
  auto At = SparseMatrix();
  auto Bt = Vector();
  auto Xt = Vector();
  // Form the system, applying the *same* boundary DoFs
  at.FormLinearSystem(ess_tdof_list, zeta, bt, At, Xt, Bt);

  // Set up a preconditioner
  auto Pt = GSSmoother(At);

  // Set up the solver
  auto solverT = CGSolver();
  solverT.SetRelTol(1e-12);
  solverT.SetMaxIter(10000);
  solverT.SetPrintLevel(1);
  solverT.SetPreconditioner(Pt);
  solverT.SetOperator(At);

  // Solve the transformed system
  solverT.Mult(Bt, Xt);

  // Recover the transformed solution 'zeta'
  at.RecoverFEMSolution(Xt, bt, zeta);

  // === 9. Calculate L2 errors ===

  // Get L2 error for the standard problem
  // Compare computed solution 'phi' with exact solution 'g'
  auto phi_error = phi.ComputeL2Error(g);
  std::cout << "L2 error for potential = " << phi_error << std::endl;

  // Get L2 error for the transformed problem
  // We must compare 'zeta' (solution on reference domain) with the
  // "pullback" of the exact solution, h(x) = g(q(x)).
  // TransformedFunctionCoefficient computes this composition g(q(x)).
  auto h = TransformedFunctionCoefficient(
      qv, std::function(dim == 2 ? exact_solution2 : exact_solution3));

  // Compare computed transformed solution 'zeta' with exact transformed
  // solution 'h'
  auto zeta_error = zeta.ComputeL2Error(h);
  std::cout << "L2 error for transformed potential = " << zeta_error
            << std::endl;

  // === 10. Visualize the transformed solution ===

  // Visualize 'zeta' on the *original, undeformed* mesh.
  auto zeta_sock = socketstream(vishost, visport);
  zeta_sock.precision(8);
  zeta_sock << "solution\n" << mesh << zeta << "window_title 'zeta'" << flush;
  if (dim == 2) {
    zeta_sock << "keys Rjlmmc\n" << flush;
  } else {
    zeta_sock << "keys RRRjlci zZ\n" << flush;
  }

  // === 11. Deform the mesh and visualize the "pushed-forward" solution ===

  // Tell the mesh to use the vector FE space 'vfes' to manage its nodes
  mesh.SetNodalFESpace(&vfes);
  // Get a pointer to the mesh nodes (as a GridFunction)
  auto *x = mesh.GetNodes();

  // Create a new GridFunction 'y' to store the deformed node positions
  auto y = GridFunction(&vfes);
  // Project the transformation  onto 'y'
  y.ProjectCoefficient(qv);

  // Overwrite the original mesh nodes 'x' with the new positions 'y'
  *x = y;  // The mesh is now physically deformed

  // Visualize the *same* solution 'zeta' but on the *deformed* mesh.
  // GLVis will use the new node positions, showing the "push-forward"
  // of the solution onto the physical (deformed) domain.
  auto zetaT_sock = socketstream(vishost, visport);
  zetaT_sock.precision(8);
  zetaT_sock << "solution\n"
             << mesh << zeta << "window_title 'zeta pushed forward'" << flush;
  if (dim == 2) {
    zetaT_sock << "keys Rjlmmc\n" << flush;
  } else {
    zetaT_sock << "keys RRRjlci zZ\n" << flush;
  }
}
