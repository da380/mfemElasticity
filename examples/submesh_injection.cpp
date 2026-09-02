// -----------------------------------------------------------------------------
// A tour of SubMeshDofInjection (serial).
//
// Mesh: data/circular_offset.msh — a disk M (attribute 1) inside a larger
// offset disk Ω (attribute 2 is the surrounding annulus). Boundary attribute
// 1 is the internal circle ∂M, attribute 2 the outer circle ∂Ω.
//
// Part A. Moving fields between the parent mesh and the submesh.
//
//   The injection is built from a space on the parent mesh and its "shadow"
//   on the submesh (same FE collection object, vdim and ordering — made by
//   MakeShadowSpace). Its MultTranspose restricts a parent field to the
//   submesh (identical to SubMesh::Transfer, which we verify); its Mult
//   injects a submesh field back, extended by zero.
//
// Part B. A toy coupled problem, solved through the injection.
//
//   Find φ on Ω and u on M such that
//
//       ∫_Ω ∇φ·∇φ' + ∫_M u φ'  =  ∫_Ω f φ'   for all φ',   φ = 0 on ∂Ω,
//       ∫_M φ u'   - ∫_M u u'  =  0           for all u'.
//
//   The cross terms ∫_M u φ' and ∫_M φ u' are integrals over the submesh in
//   which one field lives on the parent mesh — exactly the structure of the
//   elastogravity coupling ∫_M ρ ∇φ·u'. With the injection they need no
//   custom assembly: if M_sub is the plain mass matrix on the submesh
//   (between the shadow space and itself), then
//
//       B  = P M_sub  = RemapRows(M_sub)     (parent rows × sub cols),
//       Bᵀ = M_sub Pᵀ = RemapColumns(M_sub)  (sub rows × parent cols),
//
//   and the block system reads
//
//       [ A   B ] [Φ]   [F]
//       [ Bᵀ -M ] [U] = [0],   solved here with MINRES.
//
//   The toy is chosen to be self-checking, in two independent ways:
//
//   1. The second equation says M U = Bᵀ Φ = M_sub (Pᵀ Φ), i.e. u is exactly
//      the dof-wise restriction of φ to the submesh: U = Pᵀ Φ.
//   2. Eliminating u gives A Φ + P M_sub Pᵀ Φ = F, which is precisely the
//      single-mesh problem "Poisson with a reaction term confined to M":
//      assemble it directly on the parent mesh with an attribute-restricted
//      MassIntegrator and the two solutions must agree to solver tolerance.
//
// Sample run:  ./submesh_injection -o 2
// -----------------------------------------------------------------------------

#include <cmath>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

int main(int argc, char *argv[]) {
  const char *mesh_file = "../data/circular_offset.msh";
  int order = 2;
  real_t rel_tol = 1e-12;
  bool visualization = false;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&rel_tol, "-rt", "--rel-tol", "Solver relative tolerance.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization", "Enable or disable GLVis.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // ---------------------------------------------------------------------------
  // Meshes and spaces. The u space *is* the shadow space: the restriction of
  // the parent φ space to the submesh.
  // ---------------------------------------------------------------------------
  Mesh mesh(mesh_file, 1, 1);
  const int dim = mesh.Dimension();

  Array<int> patch_attr({1});
  SubMesh patch(SubMesh::CreateFromDomain(mesh, patch_attr));

  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec);
  auto shadow = SubMeshDofInjection::MakeShadowSpace(fes, patch);

  const int m = fes.GetVSize();
  const int n = shadow->GetVSize();
  cout << "\nParent dofs: " << m << ",  submesh dofs: " << n << endl;

  auto injection = SubMeshDofInjection(*shadow, fes);

  // ---------------------------------------------------------------------------
  // Part A: field transfer, both directions.
  // ---------------------------------------------------------------------------
  auto g_coeff = FunctionCoefficient([](const Vector &x) {
    return sin(3.0 * x[0]) * cos(2.0 * x[1]) + 0.5 * x[0] * x[1];
  });

  GridFunction g(&fes);
  g.ProjectCoefficient(g_coeff);

  // Parent -> submesh: MultTranspose is an exact dof-wise restriction, and
  // agrees with MFEM's own SubMesh::Transfer.
  GridFunction g_sub(shadow.get()), g_sub_ref(shadow.get());
  injection.MultTranspose(g, g_sub);
  SubMesh::Transfer(g, g_sub_ref);
  g_sub_ref -= g_sub;
  cout << "\nPart A: field transfer" << endl;
  cout << "  ||P^T g - Transfer(g)||_inf     = " << g_sub_ref.Normlinf()
       << endl;

  // Submesh -> parent: Mult extends by zero. The round trip P^T P is the
  // identity on the submesh.
  GridFunction g_ext(&fes);
  injection.Mult(g_sub, g_ext);
  GridFunction g_round(shadow.get());
  injection.MultTranspose(g_ext, g_round);
  g_round -= g_sub;
  cout << "  ||P^T (P g_sub) - g_sub||_inf   = " << g_round.Normlinf() << endl;

  // ---------------------------------------------------------------------------
  // Part B: the coupled toy problem.
  // ---------------------------------------------------------------------------
  auto f_coeff = FunctionCoefficient([](const Vector &x) {
    const real_t dx = x[0] - 0.2, dy = x[1] + 0.3;
    return exp(-4.0 * (dx * dx + dy * dy));
  });
  ConstantCoefficient one(1.0);

  // Dirichlet condition on the outer circle (parent boundary attribute 2).
  Array<int> bdr_marker(mesh.bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[1] = 1;
  Array<int> ess_tdof_list;
  fes.GetEssentialTrueDofs(bdr_marker, ess_tdof_list);

  // The elimination of essential dofs below leaves B untouched, which is
  // only right if no essential parent dof lies in the closure of M. That
  // holds for this geometry; check it rather than assume it.
  {
    Array<int> ess_vdofs;
    fes.GetEssentialVDofs(bdr_marker, ess_vdofs);
    for (int i = 0; i < n; i++) {
      MFEM_VERIFY(ess_vdofs[injection.ParentVDofs()[i]] == 0,
                  "The submesh touches the essential boundary.");
    }
  }

  // A: stiffness on the parent, with essential elimination.
  BilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator(one));
  a.Assemble();

  LinearForm b(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
  b.Assemble();

  GridFunction phi(&fes);
  phi = 0.0;

  SparseMatrix A;
  Vector Phi, F;
  a.FormLinearSystem(ess_tdof_list, phi, b, A, Phi, F);

  // M: mass on the submesh. B and B^T then come from the injection by pure
  // re-indexing — no cross-mesh assembly anywhere.
  BilinearForm m_form(shadow.get());
  m_form.AddDomainIntegrator(new MassIntegrator(one));
  m_form.Assemble();
  m_form.Finalize();
  SparseMatrix &M = m_form.SpMat();

  auto B = injection.RemapRows(M);      // = P M   (parent × sub)
  auto Bt = injection.RemapColumns(M);  // = M P^T (sub × parent) = B^T

  // Block system and MINRES.
  Array<int> offsets({0, m, m + n});
  BlockOperator block_op(offsets);
  block_op.SetBlock(0, 0, &A);
  block_op.SetBlock(0, 1, B.get());
  block_op.SetBlock(1, 0, Bt.get());
  block_op.SetBlock(1, 1, &M, -1.0);

  DSmoother prec_A(A), prec_M(M);
  BlockDiagonalPreconditioner prec(offsets);
  prec.SetDiagonalBlock(0, &prec_A);
  prec.SetDiagonalBlock(1, &prec_M);

  BlockVector X(offsets), Rhs(offsets);
  X = 0.0;
  X.GetBlock(0) = Phi;
  Rhs = 0.0;
  Rhs.GetBlock(0) = F;

  MINRESSolver minres;
  minres.SetRelTol(rel_tol);
  minres.SetMaxIter(20000);
  minres.SetPrintLevel(0);
  minres.SetOperator(block_op);
  minres.SetPreconditioner(prec);
  minres.Mult(Rhs, X);

  cout << "\nPart B: block solve" << endl;
  cout << "  MINRES iterations               = " << minres.GetNumIterations()
       << (minres.GetConverged() ? "" : "  (NOT converged)") << endl;

  // Note: not RecoverFEMSolution here. In serial legacy assembly the X
  // returned by FormLinearSystem aliases phi's memory and RecoverFEMSolution
  // relies on that aliasing; our solution lives in a BlockVector instead, so
  // copy it back explicitly (conforming serial space: vdofs = tdofs, and the
  // eliminated boundary values were carried through the solve).
  GridFunction u(shadow.get());
  phi = X.GetBlock(0);
  u = X.GetBlock(1);

  // Check 1: the second block equation forces u to be the dof-wise
  // restriction of φ.
  GridFunction phi_restricted(shadow.get());
  injection.MultTranspose(phi, phi_restricted);
  phi_restricted -= u;
  cout << "  ||u - P^T phi||_inf             = " << phi_restricted.Normlinf()
       << "   (solver tolerance)" << endl;

  // Check 2: eliminating u gives the single-mesh problem with the reaction
  // term confined to the patch, assembled here directly on the parent mesh
  // with an attribute marker.
  GridFunction phi_mono(&fes);
  phi_mono = 0.0;
  {
    Array<int> patch_marker(mesh.attributes.Max());
    patch_marker = 0;
    patch_marker[0] = 1;

    BilinearForm a_mono(&fes);
    a_mono.AddDomainIntegrator(new DiffusionIntegrator(one));
    a_mono.AddDomainIntegrator(new MassIntegrator(one), patch_marker);
    a_mono.Assemble();

    LinearForm b_mono(&fes);
    b_mono.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
    b_mono.Assemble();

    SparseMatrix A_mono;
    Vector Phi_mono, F_mono;
    a_mono.FormLinearSystem(ess_tdof_list, phi_mono, b_mono, A_mono, Phi_mono,
                            F_mono);

    GSSmoother prec_mono(A_mono);
    CGSolver cg;
    cg.SetRelTol(rel_tol);
    cg.SetMaxIter(20000);
    cg.SetPrintLevel(0);
    cg.SetOperator(A_mono);
    cg.SetPreconditioner(prec_mono);
    cg.Mult(F_mono, Phi_mono);

    a_mono.RecoverFEMSolution(Phi_mono, b_mono, phi_mono);
  }

  GridFunction diff(phi_mono);
  diff -= phi;
  cout << "  ||phi - phi_monolithic||_inf    = " << diff.Normlinf()
       << "   (solver tolerance; ||phi||_inf = " << phi.Normlinf() << ")"
       << endl;

  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;

    socketstream phi_sock(vishost, visport);
    phi_sock.precision(8);
    phi_sock << "solution\n"
             << mesh << phi << "window_title 'phi on the parent mesh'" << endl;
    phi_sock << "keys Rjlbc\n" << flush;

    socketstream u_sock(vishost, visport);
    u_sock.precision(8);
    u_sock << "solution\n"
           << patch << u << "window_title 'u on the submesh'" << endl;
    u_sock << "keys Rjlbc\n" << flush;
  }

  return 0;
}
