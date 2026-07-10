#include <iostream>

#include "mfem.hpp"

using namespace mfem;
using namespace std;

// ============================================================================
// 1. The Initial Condition Function
// ============================================================================
// Creates a "hot spot" centered at the origin (0,0)
double initial_temperature(const Vector &x) {
  double r2 = x(0) * x(0) + x(1) * x(1);
  if (x.Size() == 3) {
    r2 += x(2) * x(2);
  }  // Support 3D meshes too!
  return exp(-20.0 * r2);
}

// ============================================================================
// 2. The Custom PDE Operator (Unchanged from before)
// ============================================================================
class HeatOperator : public TimeDependentOperator {
 private:
  SparseMatrix &M;
  SparseMatrix &K;
  Array<int> &ess_tdof_list;

  CGSolver A_solver;
  mutable Vector z;

 public:
  HeatOperator(SparseMatrix &M_, SparseMatrix &K_, Array<int> &ess_bdr)
      : TimeDependentOperator(M_.Height(), 0.0,
                              TimeDependentOperator::IMPLICIT),
        M(M_),
        K(K_),
        ess_tdof_list(ess_bdr),
        z(M_.Height()) {
    A_solver.iterative_mode = false;
    A_solver.SetRelTol(1e-8);
    A_solver.SetAbsTol(1e-12);
    A_solver.SetMaxIter(100);
    A_solver.SetPrintLevel(0);
  }

  virtual void ImplicitSolve(const double dt, const Vector &x,
                             Vector &k) override {
    K.Mult(x, z);
    z.Neg();

    for (int i = 0; i < ess_tdof_list.Size(); i++) {
      z[ess_tdof_list[i]] = 0.0;
    }

    SparseMatrix *A = Add(1.0, M, dt, K);
    A_solver.SetOperator(*A);

    k = 0.0;
    A_solver.Mult(z, k);

    delete A;
  }

  virtual void Mult(const Vector &x, Vector &y) const override {}
};

// ============================================================================
// 3. Main Execution Loop
// ============================================================================
int main(int argc, char *argv[]) {
  // 1. Command-line options parsing
  const char *mesh_file =
      "../data/star.mesh";  // Default mesh if none is provided

  int order = 1;
  int ref_levels = 0;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "number of mesh refinements");
  args.ParseCheck();

  // 2. Load the mesh from the file
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();

  for (int l = 0; l < ref_levels; l++) {
    mesh.UniformRefinement();
  }

  // 3. Define the Finite Element Space (Continuous linear elements: H1, order
  // 1)
  H1_FECollection fec(order, dim);
  FiniteElementSpace fespace(&mesh, &fec);

  // 4. Identify the boundary degrees of freedom (for Dirichlet BCs)
  Array<int> ess_bdr(mesh.bdr_attributes.Max());
  ess_bdr = 1;
  Array<int> ess_tdof_list;
  fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  // 5. Assemble the Mass Matrix (M)
  BilinearForm m(&fespace);
  m.AddDomainIntegrator(new MassIntegrator());
  m.Assemble();
  SparseMatrix M;
  m.FormSystemMatrix(ess_tdof_list, M);

  // 6. Assemble the Stiffness Matrix (K)
  BilinearForm k(&fespace);
  k.AddDomainIntegrator(new DiffusionIntegrator());
  k.Assemble();
  SparseMatrix K;
  k.FormSystemMatrix(ess_tdof_list, K);

  // 7. Set up the initial condition
  GridFunction u(&fespace);
  FunctionCoefficient u_0(initial_temperature);
  u.ProjectCoefficient(u_0);
  for (int i = 0; i < ess_tdof_list.Size(); i++) {
    u[ess_tdof_list[i]] = 0.0;
  }

  // 8. Initialize Operator and ODE Solver
  HeatOperator heat_op(M, K, ess_tdof_list);
  BackwardEulerSolver ode_solver;
  ode_solver.Init(heat_op);

  // 9. Time Stepping Setup
  double t = 0.0;
  double dt = 0.01;
  double t_final = 3.0;

  // 10. GLVis Socket Setup
  socketstream glvis_out;
  glvis_out.open("localhost", 19916);
  glvis_out.precision(8);

  if (glvis_out.is_open()) {
    cout << "Connected to GLVis! Starting simulation..." << endl;
    glvis_out << "solution\n" << mesh << u;
    glvis_out << "window_title 'Heat Equation: t = 0'\n";
    glvis_out
        << "pause\n";  // Wait for user interaction in GLVis before starting
    glvis_out << flush;
  } else {
    cout << "GLVis not found. Running headlessly..." << endl;
  }

  // 11. The Integration Loop
  while (t < t_final) {
    ode_solver.Step(u, t, dt);

    // Send the updated solution to GLVis
    if (glvis_out.is_open()) {
      glvis_out << "solution\n" << mesh << u;
      glvis_out << "window_title 'Heat Equation: t = " << t << "'\n";
      glvis_out << flush;
    }
  }

  return 0;
}