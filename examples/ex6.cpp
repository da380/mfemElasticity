// -----------------------------------------------------------------------------
// Coupled PDE system:
//
// On submesh Ω1 (sphere of radius 1):
//     Δψ1 + ψ2 = f1
//
// On full mesh Ω2 (sphere of radius 2):
//     Δψ2 + ψ1 = f2
//
// Boundary conditions:
//     ψ1 = 0 on ∂Ω1
//     ψ2 = 0 on ∂Ω2
//
// Notes:
// - The system is solved in the block manner.
// - To generate the mesh, run: build/concentric_spheres -r 1-2 -s 0.02-0.1 -out
// mesh/ex7.msh
// -----------------------------------------------------------------------------
#include <cmath>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

real_t psi1Exact(const Vector &x) {
  const real_t r = sqrt(x(0) * x(0) + x(1) * x(1) + x(2) * x(2));
  const real_t z = x(2);

  if (r < 1.0) {
    return (1.0 - r * r) * z;
  } else {
    return 0.0;
  }
}

real_t psi2Exact(const Vector &x) {
  const real_t r = sqrt(x(0) * x(0) + x(1) * x(1) + x(2) * x(2));
  const real_t z = x(2);

  if (r < 1.0) {
    return (19.0 / 7.0 - 12.0 / 7.0 * r * r) * z;
  } else {
    return (-1.0 / 7.0 + 8.0 / (7.0 * r * r * r)) * z;
  }
}

real_t f1Exact(const Vector &x) {
  const real_t r = sqrt(x(0) * x(0) + x(1) * x(1) + x(2) * x(2));
  const real_t z = x(2);

  if (r < 1.0) {
    return (-51.0 / 7.0 - 12.0 / 7.0 * r * r) * z;
  } else {
    return (-1.0 / 7.0 + 8.0 / (7.0 * r * r * r)) * z;
  }
}

real_t f2Exact(const Vector &x) {
  const real_t r = sqrt(x(0) * x(0) + x(1) * x(1) + x(2) * x(2));
  const real_t z = x(2);

  if (r < 1.0) {
    return (-113.0 / 7.0 - r * r) * z;
  } else {
    return 0.0;
  }
}

int main(int argc, char *argv[]) {
  StopWatch chrono;

  const char *mesh_file = "mesh/ex7.msh";
  real_t rel_tol = 1e-10;
  int order = 1;
  bool visualization = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Relative tolerance.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Enable or disable GLVis.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  Array<int> attr_cond;
  attr_cond.Append(1);
  SubMesh mesh_cond(SubMesh::CreateFromDomain(*mesh, attr_cond));

  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(mesh, &fec), fes_cond(&mesh_cond, &fec);

  cout << "Number of psi1-unknowns: " << fes_cond.GetVSize() << endl;
  cout << "Number of psi2-unknowns: " << fes.GetVSize() << endl;

  GridFunction psi1_gf(&fes_cond), psi2_gf(&fes), psi2_gf_cond(&fes_cond);
  psi1_gf = 0.0;
  psi2_gf = 0.0;
  psi2_gf_cond = 0.0;

  FunctionCoefficient psi1_exact_coeff(psi1Exact), psi2_exact_coeff(psi2Exact),
      f1_coeff(f1Exact), f2_coeff(f2Exact);

  ConstantCoefficient zero(0.0), one(1.0), minus_one(-1.0);

  Array<int> ess_tdof_list, ess_tdof_list_cond;

  Array<int> bdr_marker(mesh->bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[1] = 1;

  Array<int> bdr_marker_cond(mesh_cond.bdr_attributes.Max());
  bdr_marker_cond = 0;
  bdr_marker_cond[0] = 1;

  fes.GetEssentialTrueDofs(bdr_marker, ess_tdof_list);
  fes_cond.GetEssentialTrueDofs(bdr_marker_cond, ess_tdof_list_cond);

  psi1_gf.ProjectBdrCoefficient(zero, bdr_marker_cond);
  psi2_gf.ProjectBdrCoefficient(zero, bdr_marker);

  LinearForm *b1 = new LinearForm(&fes_cond);
  b1->AddDomainIntegrator(new DomainLFIntegrator(f1_coeff));
  b1->Assemble();
  *b1 *= -1.0;

  LinearForm *b2 = new LinearForm(&fes);
  b2->AddDomainIntegrator(new DomainLFIntegrator(f2_coeff));
  b2->Assemble();
  *b2 *= -1.0;

  BilinearForm *a11 = new BilinearForm(&fes_cond);
  BilinearForm *a22 = new BilinearForm(&fes);

  auto a12 = new MixedBilinearFormSubMesh(&fes, &fes_cond, &fes_cond, true);

  auto a21 = new MixedBilinearFormSubMesh(&fes_cond, &fes, &fes_cond, false);

  a11->AddDomainIntegrator(new DiffusionIntegrator(one));
  a11->Assemble();
  a11->Finalize();

  a22->AddDomainIntegrator(new DiffusionIntegrator(one));
  a22->Assemble();
  a22->Finalize();

  a12->AddDomainIntegrator(new MassIntegrator(minus_one));
  a12->Assemble();
  a12->Finalize();

  a21->AddDomainIntegrator(new MassIntegrator(minus_one));
  a21->Assemble();
  a21->Finalize();

  SparseMatrix A11, A22;
  Vector X1, B1, X2, B2;

  // B1 and B2 will need further reductions from off-diagonal contributions for
  // non-zero Dirichlet BCs
  a11->FormLinearSystem(ess_tdof_list_cond, psi1_gf, *b1, A11, X1, B1);
  a22->FormLinearSystem(ess_tdof_list, psi2_gf, *b2, A22, X2, B2);

  OperatorHandle A12_handle;
  a12->FormRectangularSystemMatrix(ess_tdof_list, ess_tdof_list_cond,
                                   A12_handle);

  OperatorHandle A21_handle;
  a21->FormRectangularSystemMatrix(ess_tdof_list_cond, ess_tdof_list,
                                   A21_handle);

  Array<int> block_offsets(3);
  block_offsets[0] = 0;
  block_offsets[1] = X1.Size();
  block_offsets[2] = X2.Size();
  block_offsets.PartialSum();

  BlockVector X(block_offsets), B(block_offsets);
  X = 0.0;
  B = 0.0;

  B.GetBlock(0) = B1;
  B.GetBlock(1) = B2;

  BlockOperator Op(block_offsets);
  Op.SetBlock(0, 0, &A11);
  Op.SetBlock(0, 1, A12_handle.Ptr());
  Op.SetBlock(1, 0, A21_handle.Ptr());
  Op.SetBlock(1, 1, &A22);

  BlockDiagonalPreconditioner Prec(block_offsets);
  DSmoother prec11(A11);
  DSmoother prec22(A22);
  Prec.SetDiagonalBlock(0, &prec11);
  Prec.SetDiagonalBlock(1, &prec22);

  CGSolver solver;
  solver.SetRelTol(rel_tol);
  solver.SetMaxIter(3000);
  solver.SetPrintLevel(1);
  solver.SetOperator(Op);
  solver.SetPreconditioner(Prec);

  chrono.Clear();
  chrono.Start();

  solver.Mult(B, X);

  if (solver.GetConverged()) {
    std::cout << "Converged in " << solver.GetNumIterations()
              << " iterations with a residual norm of " << solver.GetFinalNorm()
              << ".\n";
  } else {
    std::cout << "Did not converge in " << solver.GetNumIterations()
              << " iterations. Residual norm is " << solver.GetFinalNorm()
              << ".\n";
  }

  chrono.Stop();
  cout << "Solver time = " << chrono.RealTime() << " s." << endl;

  psi1_gf.SetFromTrueDofs(X.GetBlock(0));
  psi2_gf.SetFromTrueDofs(X.GetBlock(1));

  mesh_cond.Transfer(psi2_gf, psi2_gf_cond);

  int order_quad = max(2, 2 * order + 1);
  const IntegrationRule *irs[Geometry::NumGeom];
  for (int i = 0; i < Geometry::NumGeom; ++i) {
    irs[i] = &(IntRules.Get(i, order_quad));
  }

  real_t psi1_l2_err =
      psi1_gf.ComputeL2Error(psi1_exact_coeff, irs) /
      ComputeLpNorm(2.0, psi1_exact_coeff, *fes_cond.GetMesh(), irs);
  real_t psi2_l2_err =
      psi2_gf.ComputeL2Error(psi2_exact_coeff, irs) /
      ComputeLpNorm(2.0, psi2_exact_coeff, *fes.GetMesh(), irs);
  real_t psi2_cond_l2_err =
      psi2_gf_cond.ComputeL2Error(psi2_exact_coeff, irs) /
      ComputeLpNorm(2.0, psi2_exact_coeff, *fes_cond.GetMesh(), irs);

  cout << "\nErrors:" << endl;
  cout << "psi1 L2 error = " << psi1_l2_err << endl;
  cout << "psi2 L2 error = " << psi2_l2_err << endl;
  cout << "psi2 (submesh) L2 error = " << psi2_cond_l2_err << endl;

  if (visualization) {
    GridFunction psi1_exact_gf(&fes_cond), psi2_exact_gf(&fes),
        psi2_exact_gf_cond(&fes_cond);
    psi1_exact_gf.ProjectCoefficient(psi1_exact_coeff);
    psi2_exact_gf.ProjectCoefficient(psi2_exact_coeff);
    mesh_cond.Transfer(psi2_exact_gf, psi2_exact_gf_cond);

    char vishost[] = "localhost";
    int visport = 19916;

    socketstream psi1_sock(vishost, visport);
    psi1_sock.precision(8);
    psi1_sock << "solution\n"
              << mesh_cond << psi1_gf << "window_title 'psi1 numerical'"
              << endl;

    socketstream psi1e_sock(vishost, visport);
    psi1e_sock.precision(8);
    psi1e_sock << "solution\n"
               << mesh_cond << psi1_exact_gf << "window_title 'psi1 exact'"
               << endl;

    socketstream psi2_sock(vishost, visport);
    psi2_sock.precision(8);
    psi2_sock << "solution\n"
              << mesh_cond << psi2_gf_cond
              << "window_title 'psi2 numerical (submesh)'" << endl;

    socketstream psi2e_sock(vishost, visport);
    psi2e_sock.precision(8);
    psi2e_sock << "solution\n"
               << mesh_cond << psi2_exact_gf_cond
               << "window_title 'psi2 exact (submesh)'" << endl;
  }

  delete b1;
  delete b2;
  delete a11;
  delete a12;
  delete a22;
  delete mesh;

  return 0;
}
