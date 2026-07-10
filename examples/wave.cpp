#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>

#include "mfem.hpp"

using namespace mfem;
using namespace std;

// ============================================================================
// 1. Initial Conditions
// ============================================================================
// Initial displacement: A Gaussian pulse in the center
double initial_wave_shape(const Vector &x) {
  double r2 = (x(0) - 0.5) * (x(0) - 0.5) + (x(1) - 0.5) * (x(1) - 0.5);
  return exp(-40.0 * r2);
}

// Initial velocity: Starts from rest
double initial_velocity(const Vector &x) { return 0.0; }

// ============================================================================
// 2. The Custom Second-Order PDE Operator
// ============================================================================
class WaveOperator : public SecondOrderTimeDependentOperator {
 private:
  SparseMatrix &M;
  SparseMatrix &K;
  Array<int> &ess_tdof_list;

  CGSolver A_solver;  // Used for ImplicitSolve
  CGSolver M_solver;  // Used for Mult (Explicit)
  mutable Vector z;

 public:
  WaveOperator(SparseMatrix &M_, SparseMatrix &K_, Array<int> &ess_bdr)
      : SecondOrderTimeDependentOperator(M_.Height(), 0.0),
        M(M_),
        K(K_),
        ess_tdof_list(ess_bdr),
        z(M_.Height()) {
    // Setup Implicit Solver (A = M + fac0*K)
    A_solver.iterative_mode = false;
    A_solver.SetRelTol(1e-8);
    A_solver.SetAbsTol(1e-12);
    A_solver.SetMaxIter(100);
    A_solver.SetPrintLevel(0);

    // Setup Explicit Solver (M)
    M_solver.iterative_mode = false;
    M_solver.SetRelTol(1e-8);
    M_solver.SetAbsTol(1e-12);
    M_solver.SetMaxIter(100);
    M_solver.SetPrintLevel(0);
    M_solver.SetOperator(M);  // M never changes, so we set it once here
  }

  // -------------------------------------------------------------------------
  // The Implicit Evaluation: Solve (M + fac0 * K) * k = -K * x
  // -------------------------------------------------------------------------
  virtual void ImplicitSolve(const double fac0, const double fac1,
                             const Vector &x, const Vector &dxdt,
                             Vector &k) override {
    K.Mult(x, z);
    z.Neg();

    for (int i = 0; i < ess_tdof_list.Size(); i++) {
      z[ess_tdof_list[i]] = 0.0;
    }

    SparseMatrix *A = Add(1.0, M, fac0, K);
    A_solver.SetOperator(*A);

    k = 0.0;
    A_solver.Mult(z, k);

    delete A;
  }

  // -------------------------------------------------------------------------
  // The Explicit Evaluation: Solve M * y = -K * x
  // -------------------------------------------------------------------------
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
// 3. Main Execution Loop
// ============================================================================
int main(int argc, char *argv[]) {
  // 1. Command-line options parsing
  const char *mesh_file = "../data/star.mesh";
  int order = 1;
  int ref_levels = 0;
  double cfl = 1.0;
  int solver_type = 0;  // 0 = Implicit, 1 = Explicit

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "Number of mesh refinements");
  args.AddOption(&cfl, "-c", "--cfl",
                 "CFL number for time step determination.");
  args.AddOption(&solver_type, "-s", "--solver",
                 "0 = Implicit (Avg Accel), 1 = Explicit (Central Diff).");
  args.ParseCheck();

  // 2. Load and refine the mesh
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();

  for (int l = 0; l < ref_levels; l++) {
    mesh.UniformRefinement();
  }

  // Calculate minimum element size (h_min) for CFL
  double h_min = std::numeric_limits<double>::infinity();
  for (int i = 0; i < mesh.GetNE(); i++) {
    h_min = std::min(h_min, mesh.GetElementSize(i));
  }

  // 3. Define the Finite Element Space
  H1_FECollection fec(order, dim);
  FiniteElementSpace fespace(&mesh, &fec);

  // 4. Identify boundary degrees of freedom
  Array<int> ess_bdr(mesh.bdr_attributes.Max());
  ess_bdr = 1;
  Array<int> ess_tdof_list;
  fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  // 5. Assemble Mass Matrix (M)
  BilinearForm m(&fespace);
  m.AddDomainIntegrator(new MassIntegrator());
  m.Assemble();
  SparseMatrix M;
  m.FormSystemMatrix(ess_tdof_list, M);

  // 6. Assemble Stiffness Matrix (K)
  BilinearForm k_form(&fespace);
  k_form.AddDomainIntegrator(new DiffusionIntegrator());
  k_form.Assemble();
  SparseMatrix K;
  k_form.FormSystemMatrix(ess_tdof_list, K);

  // 7. Set up initial conditions
  GridFunction u(&fespace);
  GridFunction v(&fespace);
  FunctionCoefficient u_0(initial_wave_shape);
  FunctionCoefficient v_0(initial_velocity);

  u.ProjectCoefficient(u_0);
  v.ProjectCoefficient(v_0);

  for (int i = 0; i < ess_tdof_list.Size(); i++) {
    u[ess_tdof_list[i]] = 0.0;
    v[ess_tdof_list[i]] = 0.0;
  }

  // 8. Initialize Operator and Dynamic Solver Toggle
  WaveOperator wave_op(M, K, ess_tdof_list);
  SecondOrderODESolver *ode_solver = nullptr;

  if (solver_type == 0) {
    cout << "Using IMPLICIT Solver (Average Acceleration)." << endl;
    ode_solver = new AverageAccelerationSolver();
  } else {
    cout << "Using EXPLICIT Solver (Central Difference)." << endl;
    ode_solver = new CentralDifferenceSolver();
  }
  ode_solver->Init(wave_op);

  // 9. Time Stepping Setup (CFL)
  double t = 0.0;
  double t_final = 2.0;
  double wave_speed = 1.0;

  int eff_order = (order > 0) ? order : 1;
  double dt = cfl * h_min / (wave_speed * eff_order);

  cout << "Mesh Refinements: " << ref_levels << endl;
  cout << "Polynomial Order: " << order << endl;
  cout << "Calculated min h: " << h_min << endl;
  cout << "Using time step dt: " << dt << endl;

  // 10. GLVis Socket Setup
  socketstream glvis_out;
  glvis_out.open("localhost", 19916);
  glvis_out.precision(8);

  if (glvis_out.is_open()) {
    cout << "Connected to GLVis! Starting simulation..." << endl;
    glvis_out << "solution\n" << mesh << u;
    glvis_out << "window_title 'Wave Equation: t = 0'\n";
    glvis_out << "keys Rjl\n";
    glvis_out << "valuerange -0.5 0.5\n";
    glvis_out << "autoscale off\n";
    glvis_out << "pause\n";
    glvis_out << flush;
  }

  // 11. The Integration Loop
  while (t < t_final) {
    ode_solver->Step(u, v, t, dt);

    if (glvis_out.is_open()) {
      glvis_out << "solution\n" << mesh << u;
      glvis_out << "window_title 'Wave Equation: t = " << t << "'\n";
      glvis_out << flush;
    }
  }

  // 12. Cleanup
  delete ode_solver;
  return 0;
}