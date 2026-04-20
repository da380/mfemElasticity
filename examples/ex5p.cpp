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

  a12->ParallelAssemble(A12_handle);
  a21->ParallelAssemble(A21_handle);

  Operator *A12 = A12_handle.Ptr();
  Operator *A21 = A21_handle.Ptr();

  HypreBoomerAMG prec11(A11);
  HypreBoomerAMG prec22(A22);

  CGSolver solver1(MPI_COMM_WORLD), solver2(MPI_COMM_WORLD);
  solver1.SetRelTol(rel_tol);
  solver1.SetAbsTol(0.0);
  solver1.SetMaxIter(3000);
  solver1.SetPrintLevel(0);

  solver2.SetRelTol(rel_tol);
  solver2.SetAbsTol(0.0);
  solver2.SetMaxIter(3000);
  solver2.SetPrintLevel(0);

  solver1.SetOperator(A11);
  solver1.SetPreconditioner(prec11);

  solver2.SetOperator(A22);
  solver2.SetPreconditioner(prec22);

  Vector X1_iter(X1.Size()), X2_iter(X2.Size());
  Vector B1_iter(B1.Size()), B2_iter(B2.Size());

  Vector Psi1(X1.Size()), Psi2(X2.Size());
  Psi1 = 0.0;
  Psi2 = 0.0;

  Vector Psi2_temp(Psi2.Size()), Psi2_diff(Psi2.Size());
  Psi2_temp = 0.0;
  Psi2_diff = 0.0;

  int max_iter = 1000;
  int iter = 0;
  real_t rel_tol_coup = 1e-10;

  chrono.Clear();
  chrono.Start();
  for (int i = 0; i < max_iter; i++) {
    iter++;

    B1_iter = B1;
    B2_iter = B2;

    A12->AddMult(Psi2, B1_iter, -1.0);
    solver1.Mult(B1_iter, X1_iter);
    Psi1 = X1_iter;

    A21->AddMult(Psi1, B2_iter, -1.0);
    solver2.Mult(B2_iter, X2_iter);
    Psi2_temp = X2_iter;

    Psi2_diff = Psi2_temp;
    Psi2_diff -= Psi2;

    real_t local_num = Psi2_diff * Psi2_diff;
    real_t global_num = 0.0;
    MPI_Allreduce(&local_num, &global_num, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    real_t local_den = Psi2_temp * Psi2_temp;
    real_t global_den = 0.0;
    MPI_Allreduce(&local_den, &global_den, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    real_t res = sqrt(global_num) / sqrt(global_den);
    Psi2 = Psi2_temp;

    if (verbose) {
      cout << "Iteration " << iter << ", residual = " << res << endl;
    }

    if (res < rel_tol_coup) {
      chrono.Stop();
      if (verbose) {
        cout << "Converged at iteration " << iter << endl;
        cout << "Time = " << chrono.RealTime() << " s" << endl;
      }
      break;
    }

    if (i == max_iter - 1) {
      chrono.Stop();
      if (verbose) {
        cout << "Did not converge in " << max_iter << " iterations." << endl;
        cout << "Time = " << chrono.RealTime() << " s" << endl;
      }
    }
  }

  psi1_gf.Distribute(&Psi1);
  psi2_gf.Distribute(&Psi2);
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
