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

  // Form the submesh
  auto sub_mesh = SubMesh::CreateFromDomain(mesh, Array<int>{1});

  auto marker = Array<int>{1, 0};

  // Set up the FE-spaces
  auto H1 = H1_FECollection(order, dim);
  auto L2 = L2_FECollection(order, dim);

  auto fes = FiniteElementSpace(&mesh, &H1);
  auto sub_fes = FiniteElementSpace(&sub_mesh, &H1);

  auto rho = ConstantCoefficient(1.0);

  auto a00 = BilinearForm(&sub_fes);
  a00.AddDomainIntegrator(new DiffusionIntegrator());
  a00.AddDomainIntegrator(new MassIntegrator(rho));
  a00.Assemble();

  auto a11 = BilinearForm(&fes);
  a11.AddDomainIntegrator(new DiffusionIntegrator());
  a11.AddDomainIntegrator(new MassIntegrator(rho));
  a11.Assemble();

  auto mu = ConstantCoefficient(0.5);
  auto a01 = BilinearForm(&sub_fes);
  a01.AddDomainIntegrator(new MassIntegrator(mu));
  a01.Assemble();

  auto a10 = BilinearForm(&fes);
  a10.AddDomainIntegrator(new MassIntegrator(mu), marker);
  a10.Assemble();

  auto f = FunctionCoefficient([&c1](const Vector& x) { return x[1] - c1[1]; });
  auto b0 = LinearForm(&sub_fes);
  b0.AddDomainIntegrator(new DomainLFIntegrator(f));
  b0.Assemble();

  auto b1 = LinearForm(&fes);
  b1.Assemble();

  // Set up GridFunctions
  auto x0 = GridFunction(&sub_fes);
  auto x1 = GridFunction(&fes);

  // Set up the linear systems
  x0 = 0.0;
  Array<int> ess_tdof_list{};
  SparseMatrix A00;
  Vector B0, X0;
  a00.FormLinearSystem(ess_tdof_list, x0, b0, A00, X0, B0);

  x1 = 0.0;
  SparseMatrix A11;
  Vector B1, X1;
  a11.FormLinearSystem(ess_tdof_list, x1, b1, A11, X1, B1);

  // Set up the solvers.
  auto P00 = GSSmoother(A00);
  auto solver00 = CGSolver();
  solver00.SetRelTol(1e-12);
  solver00.SetAbsTol(1e-12);
  solver00.SetMaxIter(10000);
  solver00.SetPrintLevel(1);
  solver00.SetOperator(A00);
  solver00.SetPreconditioner(P00);

  auto P11 = GSSmoother(A11);
  auto solver11 = CGSolver();
  solver11.SetRelTol(1e-12);
  solver11.SetAbsTol(1e-12);
  solver11.SetMaxIter(10000);
  solver11.SetPrintLevel(1);
  solver11.SetOperator(A11);
  solver11.SetPreconditioner(P11);

  solver00.iterative_mode = true;
  solver11.iterative_mode = true;

  // --- Automated Block Gauss-Seidel Outer Iterations ---

  // 1. Store the base right-hand side vectors.
  // This prevents accumulating the cross-coupling terms over multiple
  // iterations.
  Vector B0_base(B0);
  Vector B1_base(B1);

  // 2. Allocate vectors for the current iteration's RHS
  Vector B0_curr(B0.Size());
  Vector B1_curr(B1.Size());

  // 3. Allocate transfer buffers
  auto x10 = GridFunction(&fes);
  auto x01 = GridFunction(&sub_fes);

  int max_outer_iters = 100;
  double outer_tol = 1e-8;

  cout << "\n--- Starting Outer Iterations ---\n";

  // Store previous solutions to calculate the convergence residual
  Vector X0_old(X0);
  Vector X1_old(X1);

  for (int iter = 0; iter < max_outer_iters; ++iter) {
    // ==========================================
    // STEP 1: Solve Submesh (Domain 0)
    // ==========================================

    X0_old = X0;
    X1_old = X1;

    // Transfer the global solution (x1) down to the submesh (x01)
    sub_mesh.Transfer(x1, x01);

    // Update RHS: B0_curr = B0_base - a01 * x01
    B0_curr = B0_base;
    a01.AddMult(x01, B0_curr, -1.0);

    // Solve A00 * X0 = B0_curr
    solver00.Mult(B0_curr, X0);

    // Recover the GridFunction x0
    a00.RecoverFEMSolution(X0, b0, x0);

    // ==========================================
    // STEP 2: Solve Global Mesh (Domain 1)
    // ==========================================

    // Transfer the submesh solution (x0) up to the global mesh (x10)
    sub_mesh.Transfer(x0, x10);

    // Update RHS: B1_curr = B1_base - a10 * x10
    B1_curr = B1_base;
    a10.AddMult(x10, B1_curr, -1.0);

    // Solve A11 * X1 = B1_curr
    solver11.Mult(B1_curr, X1);

    // Recover the GridFunction x1
    a11.RecoverFEMSolution(X1, b1, x1);

    // ==========================================
    // Convergence Check
    // ==========================================
    X0_old -= X0;
    X1_old -= X1;

    double err0 = X0_old.Norml2();
    double err1 = X1_old.Norml2();

    cout << "  Iter " << iter + 1 << ": dX0 = " << err0 << ", dX1 = " << err1
         << "\n";

    if (err0 < outer_tol && err1 < outer_tol) {
      cout << "Outer iteration converged in " << iter + 1 << " steps.\n";
      break;
    }
  }

  // Visualise the two solutions.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock_0(vishost, visport);
  sol_sock_0.precision(8);
  sol_sock_0 << "solution\n" << sub_mesh << x0 << flush;
  sol_sock_0 << "keys Rjlbc\n" << flush;

  socketstream sol_sock_1(vishost, visport);
  sol_sock_1.precision(8);
  sol_sock_1 << "solution\n" << mesh << x1 << flush;
  sol_sock_1 << "keys Rjlbc\n" << flush;

  return 0;
}