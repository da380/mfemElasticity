// -----------------------------------------------------------------------------
// A tour of SubMeshDofInjection (parallel). Companion to submesh_injection.cpp
// — read that file first: the toy problem and its two self-checks are the
// same. This version shows what changes in parallel, which is exactly one
// thing: coupling blocks and field transfers act between *true-dof* vectors,
// through the HypreParMatrix injection
//
//     Pi = injection.NewTrueDofMatrix()   (parent true dofs × sub true dofs),
//
// a boolean (±1) matrix with Pi^T Pi = I. Pi^T is simultaneously the exact
// restriction parent → sub and the dual prolongation, so the coupling blocks
// of the toy system are pure products with the submesh mass matrix M̂:
//
//     B = Pi M̂ (parent × sub),   B^T = M̂ Pi^T,
//
// formed here with mfem::ParMult — no parallel-specific assembly, no
// communication code, and ranks holding no submesh elements just contribute
// empty blocks.
//
// Sample run:  mpirun -np 4 ./submesh_injection_p -o 2
// -----------------------------------------------------------------------------

#include <cmath>
#include <memory>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

int main(int argc, char *argv[]) {
  Mpi::Init(argc, argv);
  Hypre::Init();
  const int num_procs = Mpi::WorldSize();
  const int myid = Mpi::WorldRank();

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
    if (Mpi::Root()) {
      args.PrintUsage(cout);
    }
    return 1;
  }
  if (Mpi::Root()) {
    args.PrintOptions(cout);
  }

  // ---------------------------------------------------------------------------
  // Meshes, spaces, injection. ParSubMesh inherits the parent's partition, so
  // some ranks may hold no submesh elements at all; nothing below needs to
  // care.
  // ---------------------------------------------------------------------------
  Mesh smesh(mesh_file, 1, 1);
  const int dim = smesh.Dimension();
  ParMesh pmesh(MPI_COMM_WORLD, smesh);
  smesh.Clear();

  Array<int> patch_attr({1});
  ParSubMesh patch(ParSubMesh::CreateFromDomain(pmesh, patch_attr));

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes(&pmesh, &fec);
  auto shadow = SubMeshDofInjection::MakeShadowSpace(fes, patch);

  auto injection = SubMeshDofInjection(*shadow, fes);
  auto Pi = injection.NewTrueDofMatrix();

  const int mt = fes.GetTrueVSize();
  const int nt = shadow->GetTrueVSize();
  if (Mpi::Root()) {
    cout << "\nGlobal parent true dofs: " << fes.GlobalTrueVSize()
         << ",  global submesh true dofs: " << shadow->GlobalTrueVSize()
         << endl;
  }
  cout << "  rank " << myid << ": " << mt << " parent / " << nt
       << " submesh true dofs" << endl;

  // ---------------------------------------------------------------------------
  // Part A: true-dof field transfer, against ParSubMesh::Transfer (which
  // works on L-vectors with its own communication; Pi needs none).
  // ---------------------------------------------------------------------------
  auto g_coeff = FunctionCoefficient([](const Vector &x) {
    return sin(3.0 * x[0]) * cos(2.0 * x[1]) + 0.5 * x[0] * x[1];
  });

  ParGridFunction g(&fes);
  g.ProjectCoefficient(g_coeff);
  ParGridFunction g_sub_ref(shadow.get());
  g_sub_ref = 0.0;
  ParSubMesh::Transfer(g, g_sub_ref);

  Vector g_t(mt), g_sub_t(nt), g_sub_ref_t(nt);
  g.GetTrueDofs(g_t);
  g_sub_ref.GetTrueDofs(g_sub_ref_t);
  Pi->MultTranspose(g_t, g_sub_t);
  g_sub_ref_t -= g_sub_t;
  const auto transfer_err =
      GlobalLpNorm(infinity(), g_sub_ref_t.Normlinf(), MPI_COMM_WORLD);

  // Round trip: Pi^T Pi = I.
  Vector g_ext_t(mt), g_round_t(nt);
  Pi->Mult(g_sub_t, g_ext_t);
  Pi->MultTranspose(g_ext_t, g_round_t);
  g_round_t -= g_sub_t;
  const auto round_err =
      GlobalLpNorm(infinity(), g_round_t.Normlinf(), MPI_COMM_WORLD);

  if (Mpi::Root()) {
    cout << "\nPart A: true-dof field transfer" << endl;
    cout << "  ||Pi^T g - Transfer(g)||_inf    = " << transfer_err << endl;
    cout << "  ||Pi^T (Pi g_sub) - g_sub||_inf = " << round_err << endl;
  }

  // ---------------------------------------------------------------------------
  // Part B: the coupled toy problem at true-dof level.
  // ---------------------------------------------------------------------------
  auto f_coeff = FunctionCoefficient([](const Vector &x) {
    const real_t dx = x[0] - 0.2, dy = x[1] + 0.3;
    return exp(-4.0 * (dx * dx + dy * dy));
  });
  ConstantCoefficient one(1.0);

  Array<int> bdr_marker(pmesh.bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[1] = 1;
  Array<int> ess_tdof_list;
  fes.GetEssentialTrueDofs(bdr_marker, ess_tdof_list);

  // As in the serial example: elimination leaves B untouched, valid only if
  // no essential parent dof lies in the closure of the submesh.
  {
    Array<int> ess_vdofs;
    fes.GetEssentialVDofs(bdr_marker, ess_vdofs);
    for (int i = 0; i < injection.SubVSize(); i++) {
      MFEM_VERIFY(ess_vdofs[injection.ParentVDofs()[i]] == 0,
                  "The submesh touches the essential boundary.");
    }
  }

  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator(one));
  a.Assemble();

  ParLinearForm b(&fes);
  b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
  b.Assemble();

  ParGridFunction phi(&fes);
  phi = 0.0;

  HypreParMatrix A;
  Vector Phi, F;
  a.FormLinearSystem(ess_tdof_list, phi, b, A, Phi, F);

  // M̂: the true-dof mass matrix on the submesh; then the coupling blocks are
  // products with Pi.
  ParBilinearForm m_form(shadow.get());
  m_form.AddDomainIntegrator(new MassIntegrator(one));
  m_form.Assemble();
  m_form.Finalize();
  unique_ptr<HypreParMatrix> M(m_form.ParallelAssemble());

  unique_ptr<HypreParMatrix> B(ParMult(Pi.get(), M.get()));  // parent × sub
  unique_ptr<HypreParMatrix> Bt(B->Transpose());             // sub × parent

  Array<int> offsets({0, mt, mt + nt});
  BlockOperator block_op(offsets);
  block_op.SetBlock(0, 0, &A);
  block_op.SetBlock(0, 1, B.get());
  block_op.SetBlock(1, 0, Bt.get());
  block_op.SetBlock(1, 1, M.get(), -1.0);

  HypreBoomerAMG prec_A(A);
  prec_A.SetPrintLevel(0);
  HypreDiagScale prec_M(*M);
  BlockDiagonalPreconditioner prec(offsets);
  prec.SetDiagonalBlock(0, &prec_A);
  prec.SetDiagonalBlock(1, &prec_M);

  BlockVector X(offsets), Rhs(offsets);
  X = 0.0;
  X.GetBlock(0) = Phi;
  Rhs = 0.0;
  Rhs.GetBlock(0) = F;

  MINRESSolver minres(MPI_COMM_WORLD);
  minres.SetRelTol(rel_tol);
  minres.SetMaxIter(20000);
  minres.SetPrintLevel(1);
  minres.SetOperator(block_op);
  minres.SetPreconditioner(prec);
  minres.Mult(Rhs, X);

  if (Mpi::Root()) {
    cout << "\nPart B: block solve" << endl;
    cout << "  MINRES iterations               = " << minres.GetNumIterations()
         << (minres.GetConverged() ? "" : "  (NOT converged)") << endl;
  }

  a.RecoverFEMSolution(X.GetBlock(0), b, phi);
  ParGridFunction u(shadow.get());
  u.SetFromTrueDofs(X.GetBlock(1));

  // Check 1: u = Pi^T phi at true-dof level.
  Vector phi_t(mt), phi_restricted_t(nt);
  phi.GetTrueDofs(phi_t);
  Pi->MultTranspose(phi_t, phi_restricted_t);
  phi_restricted_t -= X.GetBlock(1);
  const auto u_err =
      GlobalLpNorm(infinity(), phi_restricted_t.Normlinf(), MPI_COMM_WORLD);
  if (Mpi::Root()) {
    cout << "  ||u - Pi^T phi||_inf            = " << u_err
         << "   (solver tolerance)" << endl;
  }

  // Check 2: agreement with the monolithic single-mesh problem.
  ParGridFunction phi_mono(&fes);
  phi_mono = 0.0;
  {
    Array<int> patch_marker(pmesh.attributes.Max());
    patch_marker = 0;
    patch_marker[0] = 1;

    ParBilinearForm a_mono(&fes);
    a_mono.AddDomainIntegrator(new DiffusionIntegrator(one));
    a_mono.AddDomainIntegrator(new MassIntegrator(one), patch_marker);
    a_mono.Assemble();

    ParLinearForm b_mono(&fes);
    b_mono.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
    b_mono.Assemble();

    HypreParMatrix A_mono;
    Vector Phi_mono, F_mono;
    a_mono.FormLinearSystem(ess_tdof_list, phi_mono, b_mono, A_mono, Phi_mono,
                            F_mono);

    HypreBoomerAMG prec_mono(A_mono);
    prec_mono.SetPrintLevel(0);
    CGSolver cg(MPI_COMM_WORLD);
    cg.SetRelTol(rel_tol);
    cg.SetMaxIter(20000);
    cg.SetPrintLevel(0);
    cg.SetOperator(A_mono);
    cg.SetPreconditioner(prec_mono);
    cg.Mult(F_mono, Phi_mono);

    a_mono.RecoverFEMSolution(Phi_mono, b_mono, phi_mono);
  }

  phi_mono -= phi;
  const auto mono_err =
      GlobalLpNorm(infinity(), phi_mono.Normlinf(), MPI_COMM_WORLD);
  const auto phi_norm =
      GlobalLpNorm(infinity(), phi.Normlinf(), MPI_COMM_WORLD);
  if (Mpi::Root()) {
    cout << "  ||phi - phi_monolithic||_inf    = " << mono_err
         << "   (solver tolerance; ||phi||_inf = " << phi_norm << ")" << endl;
  }

  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;

    socketstream phi_sock(vishost, visport);
    phi_sock << "parallel " << num_procs << " " << myid << "\n";
    phi_sock.precision(8);
    phi_sock << "solution\n"
             << pmesh << phi << "window_title 'phi on the parent mesh'"
             << flush;
    phi_sock << "keys Rjlbc\n" << flush;

    socketstream u_sock(vishost, visport);
    u_sock << "parallel " << num_procs << " " << myid << "\n";
    u_sock.precision(8);
    u_sock << "solution\n"
           << patch << u << "window_title 'u on the submesh'" << flush;
    u_sock << "keys Rjlbc\n" << flush;
  }

  return 0;
}
