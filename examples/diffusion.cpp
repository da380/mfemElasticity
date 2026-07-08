#include <fstream>
#include <iostream>

#include "mfem.hpp"

using namespace std;
using namespace mfem;

class HeatOperator : public TimeDependentOperator {
 protected:
  ParFiniteElementSpace &fespace;
  Array<int> ess_tdof_list;

  ParBilinearForm *M;
  ParBilinearForm *K;

  HypreParMatrix Mmat;
  HypreParMatrix Kmat;
  HypreParMatrix *T;
  real_t current_dt;

  CGSolver M_solver;
  HypreSmoother M_prec;

  CGSolver T_solver;
  HypreSmoother T_prec;

  Coefficient &k;

  mutable Vector z;

 public:
  HeatOperator(ParFiniteElementSpace &f, Coefficient &k)
      : TimeDependentOperator(f.GetTrueVSize(), (real_t)0.0),
        fespace(f),
        k{k},
        M(nullptr),
        K(nullptr),
        T(nullptr),
        current_dt(0.0),
        M_solver(f.GetComm()),
        T_solver(f.GetComm()),
        z(height) {
    const real_t rel_tol = 1e-8;

    M = new ParBilinearForm(&fespace);
    M->AddDomainIntegrator(new MassIntegrator());
    M->Assemble(0);
    M->FormSystemMatrix(ess_tdof_list, Mmat);

    M_solver.iterative_mode = false;
    M_solver.SetRelTol(rel_tol);
    M_solver.SetAbsTol(0.0);
    M_solver.SetMaxIter(100);
    M_solver.SetPrintLevel(0);
    M_prec.SetType(HypreSmoother::Jacobi);
    M_solver.SetPreconditioner(M_prec);
    M_solver.SetOperator(Mmat);

    K = new ParBilinearForm(&fespace);
    K->AddDomainIntegrator(new DiffusionIntegrator(k));
    K->Assemble(0);
    K->FormSystemMatrix(ess_tdof_list, Kmat);

    T_solver.iterative_mode = false;
    T_solver.SetRelTol(rel_tol);
    T_solver.SetAbsTol(0.0);
    T_solver.SetMaxIter(100);
    T_solver.SetPrintLevel(0);
    T_solver.SetPreconditioner(T_prec);
  }

  void Mult(const Vector &u, Vector &du_dt) const {
    Kmat.Mult(u, z);
    z.Neg();
    M_solver.Mult(z, du_dt);
  }

  void ImplicitSolve(const real_t dt, const Vector &u, Vector &du_dt) {
    T = Add(1.0, Mmat, dt, Kmat);
    current_dt = dt;
    T_solver.SetOperator(*T);
    Kmat.Mult(u, z);
    z.Neg();
    T_solver.Mult(z, du_dt);
    delete T;
  }

  ~HeatOperator() {
    delete M;
    delete K;
  }
};

int main(int argc, char *argv[]) {
  Mpi::Init(argc, argv);
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  int ser_ref_levels = 0;
  int par_ref_levels = 0;
  int order = 1;
  int dimension = 2;
  real_t ratio = 1;

  // int ode_solver_type = 1;
  int ode_solver_type = 23;
  real_t t_final = 0.5;
  real_t dt = 1.0e-2;

  bool visualization = true;
  int vis_steps = 5;

  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                 "Number of times to refine the mesh uniformly in serial.");
  args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                 "Number of times to refine the mesh uniformly in parallel.");
  args.AddOption(&order, "-o", "--order",
                 "Order (degree) of the finite elements.");
  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                 ODESolver::Types.c_str());
  args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
  args.AddOption(&dt, "-dt", "--time-step", "Time step.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                 "Visualize every n-th timestep.");
  args.AddOption(&ratio, "-r", "--ratio", "Ratio of diffusion constants.");
  args.AddOption(&dimension, "-d", "--dimension", "Dimension of the model.");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }

  if (myid == 0) {
    args.PrintOptions(cout);
  }

  Mesh *mesh =
      dimension == 2
          ? new Mesh("/home/david/dev/meshing/examples/dyke2D.msh", 1, 1)
          : new Mesh("/home/david/dev/meshing/examples/dyke3D.msh", 1, 1);

  int dim = mesh->Dimension();

  unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

  for (int lev = 0; lev < ser_ref_levels; lev++) {
    mesh->UniformRefinement();
  }

  ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;
  for (int lev = 0; lev < par_ref_levels; lev++) {
    pmesh->UniformRefinement();
  }

  H1_FECollection fe_coll(order, dim);
  ParFiniteElementSpace fespace(pmesh, &fe_coll);

  HYPRE_BigInt fe_size = fespace.GlobalTrueVSize();
  if (myid == 0) {
    cout << "Number of temperature unknowns: " << fe_size << endl;
  }

  auto kd = 1;
  auto ks = ratio * kd;
  auto k1 = ConstantCoefficient(kd);
  auto k2 = ConstantCoefficient(ks);
  auto attr = Array<int>{1, 2};
  auto kptr = Array<Coefficient *>(2);
  kptr[0] = &k1;
  kptr[1] = &k2;
  auto k = PWCoefficient(attr, kptr);

  auto u01 = ConstantCoefficient(1);
  auto u02 = ConstantCoefficient(0);
  auto u0ptr = Array<Coefficient *>{&u01, &u02};
  auto u0 = PWCoefficient(attr, u0ptr);
  ParGridFunction u_gf(&fespace);
  u_gf.ProjectCoefficient(u0);
  Vector u;
  u_gf.GetTrueDofs(u);

  HeatOperator oper(fespace, k);

  u_gf.SetFromTrueDofs(u);
  {
    ostringstream mesh_name, sol_name;
    mesh_name << "dyke-mesh." << setfill('0') << setw(6) << myid;
    sol_name << "dyke-init." << setfill('0') << setw(6) << myid;
    ofstream omesh(mesh_name.str().c_str());
    omesh.precision(precision);
    pmesh->Print(omesh);
    ofstream osol(sol_name.str().c_str());
    osol.precision(precision);
    u_gf.Save(osol);
  }

  socketstream sout;
  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;
    sout.open(vishost, visport);
    sout << "parallel " << num_procs << " " << myid << endl;
    int good = sout.good(), all_good;
    MPI_Allreduce(&good, &all_good, 1, MPI_INT, MPI_MIN, pmesh->GetComm());
    if (!all_good) {
      sout.close();
      visualization = false;
      if (myid == 0) {
        cout << "Unable to connect to GLVis server at " << vishost << ':'
             << visport << endl;
        cout << "GLVis visualization disabled.\n";
      }
    } else {
      sout.precision(precision);
      sout << "solution\n" << *pmesh << u_gf;
      if (dim == 2) {
        sout << "keys Rjl\n";
      } else {
        sout << "keys ic\n";
      }
      sout << "valuerange 0 1\n";
      sout << "autoscale mesh\n";
      if (dim == 2) sout << "zoom 2\n";
      sout << "pause\n";
      sout << flush;
      if (myid == 0) {
        cout << "GLVis visualization paused."
             << " Press space (in the GLVis window) to resume it.\n";
      }
    }
  }

  ode_solver->Init(oper);
  real_t t = 0.0;

  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    if (t + dt >= t_final - dt / 2) {
      last_step = true;
    }

    ode_solver->Step(u, t, dt);

    if (last_step || (ti % vis_steps) == 0) {
      if (myid == 0) {
        cout << "step " << ti << ", t = " << t << endl;
      }

      u_gf.SetFromTrueDofs(u);
      if (visualization) {
        sout << "parallel " << num_procs << " " << myid << "\n";
        sout << "solution\n" << *pmesh << u_gf << flush;
      }
    }
  }

  {
    ostringstream sol_name;
    sol_name << "dyke-final." << setfill('0') << setw(6) << myid;
    ofstream osol(sol_name.str().c_str());
    osol.precision(precision);
    u_gf.Save(osol);
  }

  delete pmesh;

  return 0;
}