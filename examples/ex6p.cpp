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

  Mpi::Init(argc, argv);
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();
  bool verbose = (myid == 0);

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
    if (verbose) {
      args.PrintUsage(cout);
    }
    return 1;
  }
  if (verbose) {
    args.PrintOptions(cout);
  }

  Mesh *mesh = new Mesh(mesh_file, 1, 1);

  int dim = mesh->Dimension();
  ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

  Array<int> attr_cond;
  attr_cond.Append(1);

  ParSubMesh pmesh_cond(ParSubMesh::CreateFromDomain(*pmesh, attr_cond));

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace pfes(pmesh, &fec), pfes_cond(&pmesh_cond, &fec);

  HYPRE_BigInt psi1_size = pfes_cond.GlobalTrueVSize();
  HYPRE_BigInt psi2_size = pfes.GlobalTrueVSize();

  if (verbose) {
    cout << "Number of psi1-unknowns: " << psi1_size << endl;
    cout << "Number of psi2-unknowns: " << psi2_size << endl;
  }

  ParGridFunction psi1_gf(&pfes_cond), psi2_gf(&pfes), psi2_gf_cond(&pfes_cond);
  psi1_gf = 0.0;
  psi2_gf = 0.0;
  psi2_gf_cond = 0.0;

  Array<int> block_trueOffsets(3);
  block_trueOffsets[0] = 0;
  block_trueOffsets[1] = pfes_cond.TrueVSize();
  block_trueOffsets[2] = pfes.TrueVSize();
  block_trueOffsets.PartialSum();

  FunctionCoefficient psi1_exact_coeff(psi1Exact);
  FunctionCoefficient psi2_exact_coeff(psi2Exact);
  FunctionCoefficient f1_coeff(f1Exact);
  FunctionCoefficient f2_coeff(f2Exact);

  ConstantCoefficient zero(0.0), one(1.0), minus_one(-1.0);

  Array<int> ess_tdof_list, ess_tdof_list_cond;

  Array<int> bdr_marker(pmesh->bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[1] = 1;

  Array<int> bdr_marker_cond(pmesh_cond.bdr_attributes.Max());
  bdr_marker_cond = 0;
  bdr_marker_cond[0] = 1;

  pfes.GetEssentialTrueDofs(bdr_marker, ess_tdof_list);
  pfes_cond.GetEssentialTrueDofs(bdr_marker_cond, ess_tdof_list_cond);

  psi1_gf.ProjectBdrCoefficient(zero, bdr_marker_cond);
  psi2_gf.ProjectBdrCoefficient(zero, bdr_marker);

  ParLinearForm *b1 = new ParLinearForm(&pfes_cond);
  b1->AddDomainIntegrator(new DomainLFIntegrator(f1_coeff));
  b1->Assemble();
  *b1 *= -1.0;

  ParLinearForm *b2 = new ParLinearForm(&pfes);
  b2->AddDomainIntegrator(new DomainLFIntegrator(f2_coeff));
  b2->Assemble();
  *b2 *= -1.0;

  ParBilinearForm *a11 = new ParBilinearForm(&pfes_cond);
  ParBilinearForm *a22 = new ParBilinearForm(&pfes);

  auto a12 =
      new ParMixedBilinearFormSubMesh(&pfes, &pfes_cond, &pfes_cond, true);
  auto a21 =
      new ParMixedBilinearFormSubMesh(&pfes_cond, &pfes, &pfes_cond, false);

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

  HypreParMatrix A11, A22;
  HypreParVector X1, B1, X2, B2;

  a11->FormLinearSystem(ess_tdof_list_cond, psi1_gf, *b1, A11, X1, B1);
  a22->FormLinearSystem(ess_tdof_list, psi2_gf, *b2, A22, X2, B2);

  OperatorHandle A12_handle(Operator::Hypre_ParCSR);
  OperatorHandle A21_handle(Operator::Hypre_ParCSR);

  a12->FormRectangularSystemMatrix(ess_tdof_list, ess_tdof_list_cond,
                                   A12_handle);

  a21->FormRectangularSystemMatrix(ess_tdof_list_cond, ess_tdof_list,
                                   A21_handle);

  BlockVector trueX(block_trueOffsets), trueB(block_trueOffsets);
  trueX = 0.0;
  trueB = 0.0;

  trueB.GetBlock(0) = B1;
  trueB.GetBlock(1) = B2;

  BlockOperator Op(block_trueOffsets);
  Op.SetBlock(0, 0, &A11);
  Op.SetBlock(0, 1, A12_handle.Ptr());
  Op.SetBlock(1, 0, A21_handle.Ptr());
  Op.SetBlock(1, 1, &A22);

  BlockDiagonalPreconditioner Prec(block_trueOffsets);
  HypreBoomerAMG prec11(A11);
  HypreBoomerAMG prec22(A22);
  Prec.SetDiagonalBlock(0, &prec11);
  Prec.SetDiagonalBlock(1, &prec22);

  CGSolver solver(MPI_COMM_WORLD);
  solver.SetRelTol(rel_tol);
  solver.SetAbsTol(0.0);
  solver.SetMaxIter(3000);
  solver.SetPrintLevel(verbose);
  solver.SetOperator(Op);
  solver.SetPreconditioner(Prec);

  chrono.Clear();
  chrono.Start();

  solver.Mult(trueB, trueX);

  if (solver.GetConverged()) {
    if (verbose) {
      std::cout << "Converged in " << solver.GetNumIterations()
                << " iterations with a residual norm of "
                << solver.GetFinalNorm() << ".\n";
    }
  } else {
    if (verbose) {
      std::cout << "Did not converge in " << solver.GetNumIterations()
                << " iterations. Residual norm is " << solver.GetFinalNorm()
                << ".\n";
    }
  }

  chrono.Stop();
  if (verbose) cout << "Solver time = " << chrono.RealTime() << " s." << endl;

  Vector X1_block(trueX.GetBlock(0));
  Vector X2_block(trueX.GetBlock(1));

  a11->RecoverFEMSolution(X1_block, *b1, psi1_gf);
  a22->RecoverFEMSolution(X2_block, *b2, psi2_gf);
  pmesh_cond.Transfer(psi2_gf, psi2_gf_cond);

  int order_quad = max(2, 2 * order + 1);
  const IntegrationRule *irs[Geometry::NumGeom];
  for (int i = 0; i < Geometry::NumGeom; ++i) {
    irs[i] = &(IntRules.Get(i, order_quad));
  }

  real_t psi1_l2_err =
      psi1_gf.ComputeL2Error(psi1_exact_coeff, irs) /
      ComputeGlobalLpNorm(2.0, psi1_exact_coeff, pmesh_cond, irs);
  real_t psi2_l2_err = psi2_gf.ComputeL2Error(psi2_exact_coeff, irs) /
                       ComputeGlobalLpNorm(2.0, psi2_exact_coeff, *pmesh, irs);
  real_t psi2_cond_l2_err =
      psi2_gf_cond.ComputeL2Error(psi2_exact_coeff, irs) /
      ComputeGlobalLpNorm(2.0, psi2_exact_coeff, pmesh_cond, irs);

  if (verbose) {
    cout << "\nErrors:" << endl;
    cout << "psi1 L2 error = " << psi1_l2_err << endl;
    cout << "psi2 L2 error = " << psi2_l2_err << endl;
    cout << "psi2 (submesh) L2 error = " << psi2_cond_l2_err << endl;
  }

  if (visualization) {
    ParGridFunction psi1_exact_gf(&pfes_cond), psi2_exact_gf(&pfes),
        psi2_exact_gf_cond(&pfes_cond);
    psi1_exact_gf.ProjectCoefficient(psi1_exact_coeff);
    psi2_exact_gf.ProjectCoefficient(psi2_exact_coeff);
    pmesh_cond.Transfer(psi2_exact_gf, psi2_exact_gf_cond);

    char vishost[] = "localhost";
    int visport = 19916;

    socketstream psi1_sock(vishost, visport);
    psi1_sock << "parallel " << num_procs << " " << myid << "\n";
    psi1_sock.precision(8);
    psi1_sock << "solution\n"
              << pmesh_cond << psi1_gf << "window_title 'psi1 numerical'"
              << endl;
    MPI_Barrier(pmesh_cond.GetComm());

    socketstream psi1e_sock(vishost, visport);
    psi1e_sock << "parallel " << num_procs << " " << myid << "\n";
    psi1e_sock.precision(8);
    psi1e_sock << "solution\n"
               << pmesh_cond << psi1_exact_gf << "window_title 'psi1 exact'"
               << endl;
    MPI_Barrier(pmesh_cond.GetComm());

    socketstream psi2_sock(vishost, visport);
    psi2_sock << "parallel " << num_procs << " " << myid << "\n";
    psi2_sock.precision(8);
    psi2_sock << "solution\n"
              << pmesh_cond << psi2_gf_cond
              << "window_title 'psi2 numerical (submesh)'" << endl;
    MPI_Barrier(pmesh_cond.GetComm());

    socketstream psi2e_sock(vishost, visport);
    psi2e_sock << "parallel " << num_procs << " " << myid << "\n";
    psi2e_sock.precision(8);
    psi2e_sock << "solution\n"
               << pmesh_cond << psi2_exact_gf_cond
               << "window_title 'psi2 exact (submesh)'" << endl;
  }

  delete b1;
  delete b2;
  delete a11;
  delete a12;
  delete a21;
  delete a22;
  delete pmesh;
  delete mesh;

  return 0;
}
