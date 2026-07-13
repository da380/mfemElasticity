#include <algorithm>
#include <cmath>
#include <iostream>

#include "mfem.hpp"

using namespace mfem;
using namespace std;

// ============================================================================
// 1. Initial Conditions (Vector-Valued)
// ============================================================================
void initial_deformation(const Vector &x, Vector &u) {
  u = 0.0;
  // Shifted left (x=0.5) and into the shallow surface waveguide (y=0.9)
  double r2 = (x(0) - 1.5) * (x(0) - 1.5) + (x(1) - 0.9) * (x(1) - 0.9);
  u(0) = exp(-10000 * r2);
}

void initial_velocity(const Vector &x, Vector &v) { v = 0.0; }

// ============================================================================
// 2. Material Property Functions (Waveguide Interface at y = 0.8)
// ============================================================================
double rho_func(const Vector &x) {
  if (x(1) > 0.8)
    return 1.0;  // Slow waveguide layer
  else
    return 2.0;  // Fast bedrock layer
}

double lambda_func(const Vector &x) {
  if (x(1) > 0.8)
    return 1.0;
  else
    return 4.0;
}

double mu_func(const Vector &x) {
  if (x(1) > 0.8)
    return 1.0;
  else
    return 4.0;
}

// ============================================================================
// 3. The Custom PDE Operator
// ============================================================================
class WaveOperator : public SecondOrderTimeDependentOperator {
 private:
  HypreParMatrix &M;
  HypreParMatrix &K;
  Array<int> &ess_tdof_list;

  CGSolver A_solver;
  CGSolver M_solver;

  HypreSmoother A_prec;
  HypreSmoother M_prec;

  mutable Vector z;

 public:
  WaveOperator(HypreParMatrix &M_, HypreParMatrix &K_, Array<int> &ess_bdr)
      : SecondOrderTimeDependentOperator(M_.Height()),
        M(M_),
        K(K_),
        ess_tdof_list(ess_bdr),
        z(M_.Height()),
        A_solver(M_.GetComm()),
        M_solver(M_.GetComm()) {
    A_prec.SetType(HypreSmoother::Jacobi);
    A_solver.iterative_mode = false;
    A_solver.SetRelTol(1e-8);
    A_solver.SetAbsTol(1e-12);
    A_solver.SetMaxIter(100);
    A_solver.SetPrintLevel(0);
    A_solver.SetPreconditioner(A_prec);

    M_prec.SetType(HypreSmoother::Jacobi);
    M_solver.iterative_mode = false;
    M_solver.SetRelTol(1e-8);
    M_solver.SetAbsTol(1e-12);
    M_solver.SetMaxIter(100);
    M_solver.SetPrintLevel(0);
    M_solver.SetPreconditioner(M_prec);
    M_solver.SetOperator(M);
  }

  virtual void ImplicitSolve(const double fac0, const double fac1,
                             const Vector &x, const Vector &dxdt,
                             Vector &k) override {
    K.Mult(x, z);
    z.Neg();
    for (int i = 0; i < ess_tdof_list.Size(); i++) {
      z[ess_tdof_list[i]] = 0.0;
    }

    HypreParMatrix *A = Add(1.0, M, fac0, K);
    A_solver.SetOperator(*A);
    k = 0.0;
    A_solver.Mult(z, k);
    delete A;
  }

  virtual void Mult(const Vector &x, const Vector &dxdt,
                    Vector &y) const override {
    K.Mult(x, z);
    z.Neg();
    for (int i = 0; i < ess_tdof_list.Size(); i++) {
      z[ess_tdof_list[i]] = 0.0;
    }
    y = 0.0;
    M_solver.Mult(z, y);
  }
};

// ============================================================================
// 4. Main Execution Loop
// ============================================================================
int main(int argc, char *argv[]) {
  Mpi::Init(argc, argv);
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();

  const char *mesh_file = "";
  int order = 1;
  int ref_levels = 1;
  double cfl = 0.5;
  int solver_type = 0;
  int vis_steps = 10;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "Number of mesh refinements");
  args.AddOption(&cfl, "-c", "--cfl", "CFL number.");
  args.AddOption(&solver_type, "-s", "--solver", "0 = Implicit, 1 = Explicit.");
  args.AddOption(&vis_steps, "-vs", "--vis-steps",
                 "Visualize every N-th timestep.");
  args.ParseCheck();

  Mesh *mesh;
  if (std::string(mesh_file) == "") {
    if (myid == 0)
      cout << "Generating internal 60x20 Cartesian waveguide mesh..." << endl;

    // --- NEW: Stretched 3.0 x 1.0 Mesh ---
    mesh = new Mesh(
        Mesh::MakeCartesian2D(60, 20, Element::QUADRILATERAL, true, 3.0, 1.0));
  } else {
    if (myid == 0) cout << "Loading mesh from file: " << mesh_file << endl;
    mesh = new Mesh(mesh_file, 1, 1);
  }

  for (int l = 0; l < ref_levels; l++) {
    mesh->UniformRefinement();
  }

  int dim = mesh->Dimension();

  ParMesh pmesh(MPI_COMM_WORLD, *mesh);
  delete mesh;

  double h_min = std::numeric_limits<double>::infinity();
  for (int i = 0; i < pmesh.GetNE(); i++) {
    h_min = std::min(h_min, pmesh.GetElementSize(i));
  }
  MPI_Allreduce(MPI_IN_PLACE, &h_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  H1_FECollection fec(order, dim);
  ParFiniteElementSpace pfespace(&pmesh, &fec, dim);

  Array<int> ess_bdr(pmesh.bdr_attributes.Max());
  ess_bdr = 1;
  Array<int> ess_tdof_list;

  FunctionCoefficient rho(rho_func);
  FunctionCoefficient lambda(lambda_func);
  FunctionCoefficient mu(mu_func);

  ParBilinearForm m(&pfespace);
  m.AddDomainIntegrator(new VectorMassIntegrator(rho));
  m.Assemble();
  HypreParMatrix M;
  m.FormSystemMatrix(ess_tdof_list, M);

  ParBilinearForm k_form(&pfespace);
  k_form.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
  k_form.Assemble();
  HypreParMatrix K;
  k_form.FormSystemMatrix(ess_tdof_list, K);

  ParGridFunction u(&pfespace);
  ParGridFunction v(&pfespace);

  VectorFunctionCoefficient u_0(dim, initial_deformation);
  VectorFunctionCoefficient v_0(dim, initial_velocity);

  u.ProjectCoefficient(u_0);
  v.ProjectCoefficient(v_0);

  Vector U(pfespace.GetTrueVSize());
  Vector V(pfespace.GetTrueVSize());
  u.GetTrueDofs(U);
  v.GetTrueDofs(V);

  WaveOperator wave_op(M, K, ess_tdof_list);
  SecondOrderODESolver *ode_solver = nullptr;

  if (solver_type == 0) {
    ode_solver = new AverageAccelerationSolver();
  } else {
    ode_solver = new CentralDifferenceSolver();
  }
  ode_solver->Init(wave_op);

  double p_wave_speed_max = sqrt((4.0 + 2.0 * 4.0) / 2.0);

  double t = 0.0;
  double t_final = 0.4;
  int eff_order = (order > 0) ? order : 1;
  double dt = cfl * h_min / (p_wave_speed_max * eff_order);

  if (myid == 0) cout << "Using time step dt: " << dt << endl;

  socketstream glvis_out;
  glvis_out.open("localhost", 19916);
  glvis_out.precision(8);

  if (glvis_out.is_open()) {
    glvis_out << "parallel " << num_procs << " " << myid << "\n";
    glvis_out << "solution\n" << pmesh << u;
    glvis_out << "keys Rjlvvvvv\n";
    glvis_out << "palette_name manga\n";

    // Adjusted valuerange to capture both positive and negative oscillations
    glvis_out << "valuerange -0.1 0.1\n";
    glvis_out << "autoscale off\n";
    glvis_out << "pause\n";
    glvis_out << flush;
  }

  int step = 0;

  while (t < t_final) {
    ode_solver->Step(U, V, t, dt);
    step++;

    if (step % vis_steps == 0) {
      if (myid == 0) cout << "Step: " << step << ", Time: " << t << endl;

      if (glvis_out.is_open()) {
        u.Distribute(U);
        glvis_out << "parallel " << num_procs << " " << myid << "\n";
        glvis_out << "solution\n" << pmesh << u << flush;
      }
    }
  }

  delete ode_solver;
  return 0;
}