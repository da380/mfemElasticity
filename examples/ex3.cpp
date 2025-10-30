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

real_t exact_solution(const Vector &x) { return x[0] * x[1]; }

int main(int argc, char *argv[]) {
  // Set default options.
  const char *mesh_file = "../data/disk.msh";
  int order = 2;
  int refinement = 0;
  int theta = 10;

  // Deal with options.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&refinement, "-r", "--refinement",
                 "number of  mesh refinements");
  args.AddOption(&theta, "-th", "--theta", "rotation angle in degrees");

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

  // Set up the finite element spaces.
  auto L2 = L2_FECollection(order - 1, dim);
  auto H1 = H1_FECollection(order, dim);

  // Space for the potential.
  auto fes = FiniteElementSpace(&mesh, &H1);

  // Vector space for the transformations.
  auto vfes = FiniteElementSpace(&mesh, &H1, dim);

  // Set up Dirichlet boundary conditions.
  auto ess_tdof_list = Array<int>();
  fes.GetEssentialTrueDofs(bdr_marker, ess_tdof_list);

  // Define the boundary conditions using a FunctionCoefficient
  // auto g = FunctionCoefficient(&exact_solution);
  auto g = FunctionCoefficient(std::function(exact_solution));

  // Set the gridfunction for the potential for the original problem
  auto phi = GridFunction(&fes);
  phi.ProjectCoefficient(g);

  // Set up the standard Laplace equation
  auto a = BilinearForm(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator());
  a.Assemble();

  // Set up an empty linearform
  auto b = LinearForm(&fes);
  b.Assemble();

  // Set up the linear system
  SparseMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, phi, b, A, X, B);

  // Set up a preconditioner
  auto P = GSSmoother(A);

  auto solver = CGSolver();
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);
  solver.SetPreconditioner(P);
  solver.SetOperator(A);
  solver.Mult(B, X);

  a.RecoverFEMSolution(X, b, phi);

  // Visualise if glvis is open.
  char vishost[] = "localhost";
  int visport = 19916;

  auto phi_sock = socketstream(vishost, visport);
  phi_sock.precision(8);
  phi_sock << "solution\n" << mesh << phi << "window_title 'phi'" << flush;
  phi_sock << "keys Rjlmmc\n" << flush;

  // Set up the diffeomorphism
  auto qv = VectorFunctionCoefficient(dim, [theta](const Vector &x, Vector &v) {
    using std::sin, std::cos;
    v.SetSize(x.Size());
    auto r = x.Norml2();
    auto f = 4 * r * r * (1 - r * r);
    auto theta_scaled = (pi / 180) * f * theta;
    v[0] = cos(theta_scaled) * x[0] + sin(theta_scaled) * x[1];
    v[1] = -sin(theta_scaled) * x[0] + cos(theta_scaled) * x[1];
  });
  auto xi = DiffeomorphismCoefficient(dim, qv);

  // Set up gridfunction for transformed potential
  auto zeta = GridFunction(&fes);
  zeta.ProjectCoefficient(g);

  // Set up the transformed Laplace problem
  auto at = BilinearForm(&fes);
  at.AddDomainIntegrator(new TransformedDiffusionIntegrator(xi));
  at.Assemble();

  // Set up empty linear form
  auto bt = LinearForm(&fes);
  bt.Assemble();

  // Set up the linear system
  auto At = SparseMatrix();
  auto Bt = Vector();
  auto Xt = Vector();
  at.FormLinearSystem(ess_tdof_list, zeta, bt, At, Xt, Bt);

  // Set up a preconditioner
  auto Pt = GSSmoother(At);

  auto solverT = CGSolver();
  solverT.SetRelTol(1e-12);
  solverT.SetMaxIter(10000);
  solverT.SetPrintLevel(1);
  solverT.SetPreconditioner(Pt);
  solverT.SetOperator(At);
  solverT.Mult(Bt, Xt);

  at.RecoverFEMSolution(Xt, bt, zeta);

  // Get L2 error for the reference problem
  auto phi_error = phi.ComputeL2Error(g);
  std::cout << "L2 error for potential = " << phi_error << std::endl;

  // Get L2 error for the transform problem
  auto h = TransformedFunctionCoefficient(xi, std::function(exact_solution));
  auto zeta_error = zeta.ComputeL2Error(h);
  std::cout << "L2 error for transformed potential = " << zeta_error
            << std::endl;

  // Visualise the transformed soluion
  auto zeta_sock = socketstream(vishost, visport);
  zeta_sock.precision(8);
  zeta_sock << "solution\n" << mesh << zeta << "window_title 'zeta'" << flush;
  zeta_sock << "keys Rjlmmc\n" << flush;

  // Now transform the mesh and push forward zeta
  mesh.SetNodalFESpace(&vfes);
  auto *x = mesh.GetNodes();
  auto y = GridFunction(&vfes);
  y.ProjectCoefficient(xi);
  *x = y;

  auto phiT_sock = socketstream(vishost, visport);
  phiT_sock.precision(8);
  phiT_sock << "solution\n"
            << mesh << zeta << "window_title 'zeta pushed forward'" << flush;
  phiT_sock << "keys Rjlmmc\n" << flush;
}
