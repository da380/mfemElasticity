// ============================================================================
// anisotropic_elasticity.cpp
//
// Static linear elasticity with a transversely isotropic material whose
// symmetry axis is the radial direction (radial anisotropy, as in PREM),
// assembled with mfemElasticity::ElasticTensorIntegrator. The boundary
// attribute 1 is clamped and a uniform body force is applied.
//
// With -iso the Love constants are set to their isotropic values
// (A = C = lambda + 2 mu, F = lambda, L = N = mu) and the solution is
// compared with one assembled by mfem::ElasticityIntegrator; the two agree
// to solver tolerance.
//
// Sample runs:
//    ./anisotropic_elasticity -m ../data/star.mesh -o 2
//    ./anisotropic_elasticity -m ../data/beam-tet.mesh -o 1 -iso
//    ./anisotropic_elasticity -m ../data/ball.msh -o 1 -A 3.0 -C 2.6 -F 1.0
// ============================================================================

#include <iostream>
#include <memory>

#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

int main(int argc, char* argv[]) {
  const char* mesh_file = "../data/star.mesh";
  int order = 1;
  int ref_levels = 0;
  real_t A = 3.1, C = 2.7, F = 1.1, L = 0.9, N = 1.2;
  bool isotropic = false;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "Number of uniform mesh refinements.");
  args.AddOption(&A, "-A", "--love-A", "Love constant A.");
  args.AddOption(&C, "-C", "--love-C", "Love constant C.");
  args.AddOption(&F, "-F", "--love-F", "Love constant F.");
  args.AddOption(&L, "-L", "--love-L", "Love constant L.");
  args.AddOption(&N, "-N", "--love-N", "Love constant N.");
  args.AddOption(&isotropic, "-iso", "--isotropic", "-no-iso", "--no-isotropic",
                 "Use isotropic Love constants (lambda = mu = 1) and compare "
                 "with mfem::ElasticityIntegrator.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "GLVis visualisation.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  Mesh mesh(mesh_file, 1, 1);
  const int dim = mesh.Dimension();
  for (int l = 0; l < ref_levels; l++) {
    mesh.UniformRefinement();
  }

  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);
  cout << "Displacement unknowns: " << fes.GetTrueVSize() << "\n";

  // Material: TI with the radial axis about the mesh centre.
  const real_t lambda = 1.0, mu = 1.0;
  if (isotropic) {
    A = C = lambda + 2.0 * mu;
    F = lambda;
    L = N = mu;
  }
  ConstantCoefficient cA(A), cC(C), cF(F), cL(L), cN(N);
  Vector centre(dim);
  {
    Vector lo, hi;
    mesh.GetBoundingBox(lo, hi);
    for (int i = 0; i < dim; i++) {
      centre[i] = 0.5 * (lo[i] + hi[i]);
    }
  }
  RadialUnitVectorCoefficient axis(dim, centre);
  TransverselyIsotropicElasticTensorCoefficient tensor(dim, cA, cC, cF, cL, cN,
                                                       axis);

  // Boundary conditions and load.
  Array<int> ess_bdr(mesh.bdr_attributes.Max()), ess_tdof_list;
  ess_bdr = 0;
  ess_bdr[0] = 1;
  fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  Vector g(dim);
  g = 0.0;
  g[dim - 1] = -0.1;
  VectorConstantCoefficient body(g);

  auto solve = [&](BilinearFormIntegrator* integ, GridFunction& u) {
    LinearForm b(&fes);
    b.AddDomainIntegrator(new VectorDomainLFIntegrator(body));
    b.Assemble();
    BilinearForm a(&fes);
    a.AddDomainIntegrator(integ);
    a.Assemble();
    u = 0.0;
    SparseMatrix Amat;
    Vector X, B;
    a.FormLinearSystem(ess_tdof_list, u, b, Amat, X, B);
    GSSmoother prec(Amat);
    CGSolver cg;
    cg.SetPreconditioner(prec);
    cg.SetOperator(Amat);
    cg.SetRelTol(1e-12);
    cg.SetMaxIter(10000);
    cg.SetPrintLevel(IterativeSolver::PrintLevel().Summary());
    cg.Mult(B, X);
    a.RecoverFEMSolution(X, b, u);
  };

  GridFunction u(&fes);
  solve(new ElasticTensorIntegrator(tensor), u);
  Vector zero(dim);
  zero = 0.0;
  VectorConstantCoefficient z(zero);
  cout << "||u||_L2 (anisotropic integrator) = " << u.ComputeL2Error(z) << "\n";

  if (isotropic) {
    ConstantCoefficient lam(lambda), m(mu);
    GridFunction u_ref(&fes);
    solve(new ElasticityIntegrator(lam, m), u_ref);
    u_ref -= u;
    cout << "||u - u_ref||_inf / ||u||_inf = "
         << u_ref.Normlinf() / u.Normlinf() << "\n";
    u_ref += u;
  }

  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n" << mesh << u << flush;
  }
  return 0;
}
