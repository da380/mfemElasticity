#include <algorithm>
#include <cmath>
#include <iostream>

#include "mfem.hpp"

using namespace mfem;
using namespace std;

// ============================================================================
// 1. Initial Conditions (Now Vector-Valued!)
// ============================================================================
// Initial displacement: A Gaussian pulse pushing outward in the X-direction
void initial_deformation(const Vector &x, Vector &u) {
  u = 0.0;  // Initialize all components to zero
  double r2 = (x(0) - 0.5) * (x(0) - 0.5) + (x(1) - 0.5) * (x(1) - 0.5);
  u(0) = exp(-1000.0 * r2);  // P-wave style pulse in X
  // u(1) remains 0.0
}

void initial_velocity(const Vector &x, Vector &v) { v = 0.0; }

// ============================================================================
// 2. The Custom PDE Operator (UNCHANGED from scalar wave!)
// ============================================================================
class WaveOperator : public SecondOrderTimeDependentOperator {
 private:
  SparseMatrix &M;
  SparseMatrix &K;
  Array<int> &ess_tdof_list;

  CGSolver A_solver;
  CGSolver M_solver;
  mutable Vector z;

 public:
  WaveOperator(SparseMatrix &M_, SparseMatrix &K_, Array<int> &ess_bdr)
      : SecondOrderTimeDependentOperator(M_.Height(), 0.0),
        M(M_),
        K(K_),
        ess_tdof_list(ess_bdr),
        z(M_.Height()) {
    A_solver.iterative_mode = false;
    A_solver.SetRelTol(1e-8);
    A_solver.SetAbsTol(1e-12);
    A_solver.SetMaxIter(100);
    A_solver.SetPrintLevel(0);

    M_solver.iterative_mode = false;
    M_solver.SetRelTol(1e-8);
    M_solver.SetAbsTol(1e-12);
    M_solver.SetMaxIter(100);
    M_solver.SetPrintLevel(0);
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
    SparseMatrix *A = Add(1.0, M, fac0, K);
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
// 3. Main Execution Loop
// ============================================================================
int main(int argc, char *argv[]) {
  // 1. Command-line options parsing
  // Set default to an empty string to trigger internal generation
  const char *mesh_file = "";

  int order = 1;
  int ref_levels = 1;
  double cfl = 0.5;
  int solver_type = 0;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use (leave blank for internal 2D square).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "Number of mesh refinements");
  args.AddOption(&cfl, "-c", "--cfl", "CFL number.");
  args.AddOption(&solver_type, "-s", "--solver", "0 = Implicit, 1 = Explicit.");
  args.ParseCheck();

  // 2. Load OR Generate the Mesh
  Mesh *mesh;

  // If no mesh file is provided, generate a 20x20 square mesh [0,1]x[0,1]
  if (std::string(mesh_file) == "") {
    cout << "No mesh file provided. Generating internal 20x20 Cartesian mesh..."
         << endl;
    // Parameters: nx, ny, element_type, generate_edges, size_x, size_y
    mesh = new Mesh(
        Mesh::MakeCartesian2D(20, 20, Element::QUADRILATERAL, true, 1.0, 1.0));

  } else {
    cout << "Loading mesh from file: " << mesh_file << endl;
    mesh = new Mesh(mesh_file, 1, 1);
  }

  int dim = mesh->Dimension();

  // Refine the mesh
  for (int l = 0; l < ref_levels; l++) {
    mesh->UniformRefinement();
  }

  // Calculate minimum element size (h_min) for CFL
  double h_min = std::numeric_limits<double>::infinity();
  for (int i = 0; i < mesh->GetNE(); i++) {
    h_min = std::min(h_min, mesh->GetElementSize(i));
  }

  // 3. Define the Finite Element Space
  H1_FECollection fec(order, dim);
  // Note: We pass the pointer 'mesh' directly now, rather than '&mesh'
  FiniteElementSpace fespace(mesh, &fec, dim);

  // 4. Identify the boundary degrees of freedom
  Array<int> ess_bdr(mesh->bdr_attributes.Max());
  ess_bdr = 1;  // Clamp all boundaries
  Array<int> ess_tdof_list;
  fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  // --- NEW: Material Properties ---
  ConstantCoefficient rho(1.0);     // Density
  ConstantCoefficient lambda(1.0);  // Lamé first parameter
  ConstantCoefficient mu(1.0);      // Shear modulus

  // --- NEW: Vector Integrators ---
  BilinearForm m(&fespace);
  m.AddDomainIntegrator(new VectorMassIntegrator(rho));
  m.Assemble();
  SparseMatrix M;
  m.FormSystemMatrix(ess_tdof_list, M);

  BilinearForm k_form(&fespace);
  k_form.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
  k_form.Assemble();
  SparseMatrix K;
  k_form.FormSystemMatrix(ess_tdof_list, K);

  // --- NEW: Vector Initial Conditions ---
  GridFunction u(&fespace);
  GridFunction v(&fespace);

  VectorFunctionCoefficient u_0(dim, initial_deformation);
  VectorFunctionCoefficient v_0(dim, initial_velocity);

  u.ProjectCoefficient(u_0);
  v.ProjectCoefficient(v_0);

  for (int i = 0; i < ess_tdof_list.Size(); i++) {
    u[ess_tdof_list[i]] = 0.0;
    v[ess_tdof_list[i]] = 0.0;
  }

  WaveOperator wave_op(M, K, ess_tdof_list);
  SecondOrderODESolver *ode_solver = nullptr;

  if (solver_type == 0) {
    ode_solver = new AverageAccelerationSolver();
  } else {
    ode_solver = new CentralDifferenceSolver();
  }
  ode_solver->Init(wave_op);

  // --- NEW: P-Wave Speed for CFL ---
  // The fastest wave in elastic media is the P-wave: cp = sqrt((lambda + 2*mu)
  // / rho)
  double p_wave_speed = sqrt((1.0 + 2.0 * 1.0) / 1.0);

  double t = 0.0;
  double t_final = 10.0;
  int eff_order = (order > 0) ? order : 1;
  double dt = cfl * h_min / (p_wave_speed * eff_order);

  cout << "Using time step dt: " << dt << endl;

  socketstream glvis_out;
  glvis_out.open("localhost", 19916);
  glvis_out.precision(8);

  if (glvis_out.is_open()) {
    // GLVis treats vector GridFunctions specially. It will deform the mesh!
    glvis_out << "solution\n" << *mesh << u;
    glvis_out << "window_title 'Elastic Wave: t = 0'\n";
    glvis_out << "keys Rjlvvvvvnnpppppppppppp\n";
    glvis_out << "valuerange 0.0 0.5\n";
    glvis_out << "autoscale off\n";
    glvis_out << "pause\n";
    glvis_out << flush;
  }

  while (t < t_final) {
    ode_solver->Step(u, v, t, dt);

    if (glvis_out.is_open()) {
      glvis_out << "solution\n" << *mesh << u;
      glvis_out << "window_title 'Elastic Wave: t = " << t << "'\n";
      glvis_out << flush;
    }
  }

  delete ode_solver;
  return 0;
}
