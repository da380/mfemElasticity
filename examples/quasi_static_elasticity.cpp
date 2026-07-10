#include <memory>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

/**
 * @brief Virtual base class for quasi-static elastic problems.
 * * This interface defines the common interactions required by viscoelastic
 * solvers or time-stepping loops. It abstracts away the specific boundary
 * conditions and solver implementations.
 */
class QuasiStaticLinearElasticProblem {
 public:
  /**
   * @brief Access the underlying finite element space.
   * @return A reference to the MFEM FiniteElementSpace.
   */
  virtual mfem::FiniteElementSpace& GetFES() = 0;

  /**
   * @brief Access the current displacement solution.
   * @return A reference to the GridFunction representing the displacement
   * field.
   */
  virtual mfem::GridFunction& GetDisplacement() = 0;

  /**
   * @brief Update the right-hand side (RHS) of the problem for a specific time.
   * * This method should rebuild or update the linear form based on
   * time-dependent boundary conditions, tractions, or body forces.
   * * @param t The current physical time.
   */
  virtual void SetRHS(mfem::real_t t) = 0;

  /**
   * @brief Increment the current right-hand side vector.
   * * Useful for applying history terms or explicit viscoelastic stress
   * updates.
   * * @param v The vector to add to the existing RHS linear form.
   */
  virtual void IncrementRHS(const mfem::Vector& v) = 0;

  /**
   * @brief Solve the linear system for the current state.
   * * Updates the displacement GridFunction based on the current RHS.
   */
  virtual void Solve() = 0;

  /**
   * @brief Default virtual destructor.
   */
  virtual ~QuasiStaticLinearElasticProblem() = default;
};

/**
 * @brief Implementation of a pure traction (Neumann) quasi-static elastic
 * problem.
 * * @note Because pure traction problems lack Dirichlet boundary conditions,
 * the resulting stiffness matrix is singular (it contains a rigid body null
 * space). This class automatically sets up an `mfemElasticity::RigidBodySolver`
 * to project out zero-energy translational and rotational modes during the
 * solve step.
 */
class TractionProblem : public QuasiStaticLinearElasticProblem {
 private:
  /// Non-owning pointer to the mesh (memory managed externally)
  mfem::Mesh* _mesh;

  std::unique_ptr<mfem::FiniteElementCollection> _fec;
  std::unique_ptr<mfem::FiniteElementSpace> _fes;

  std::unique_ptr<mfem::Coefficient> _lambda;
  std::unique_ptr<mfem::Coefficient> _mu;
  std::unique_ptr<mfem::VectorCoefficient> _tc;

  std::unique_ptr<mfem::Array<int>> _marker;
  std::unique_ptr<mfem::LinearForm> _b;
  std::unique_ptr<mfem::BilinearForm> _a;
  mfem::GridFunction _u;

  std::unique_ptr<mfem::SparseMatrix> _A;
  std::unique_ptr<mfem::Vector> _B, _X;
  std::unique_ptr<mfem::GSSmoother> _M;
  std::unique_ptr<mfem::IterativeSolver> _solver;
  std::unique_ptr<mfemElasticity::RigidBodySolver> _rigidSolver;

 public:
  /**
   * @brief Construct a new Traction Problem.
   * * Initializes the finite element space, allocates memory for the linear
   * system, assembles the time-independent stiffness matrix, and configures the
   * solvers.
   * * @param mesh Pointer to the computational mesh.
   * @param order The polynomial order for the H1 finite element collection.
   */
  TractionProblem(mfem::Mesh* mesh, int order) : _mesh(mesh) {
    using namespace mfem;

    int dim = _mesh->Dimension();
    _fec = std::make_unique<H1_FECollection>(order, dim);
    _fes = std::make_unique<FiniteElementSpace>(_mesh, _fec.get(), dim);

    _lambda = std::make_unique<ConstantCoefficient>(1.0);
    _mu = std::make_unique<ConstantCoefficient>(1.0);

    _a = std::make_unique<BilinearForm>(_fes.get());
    _a->AddDomainIntegrator(new ElasticityIntegrator(*_lambda, *_mu));
    _a->Assemble();

    _b = std::make_unique<LinearForm>(_fes.get());
    _b->Assemble();

    _u.SetSpace(_fes.get());
    _u = 0.0;

    _A = std::make_unique<SparseMatrix>();
    _X = std::make_unique<Vector>();
    _B = std::make_unique<Vector>();

    Array<int> ess_tdof_list;
    _a->FormLinearSystem(ess_tdof_list, _u, *_b, *_A, *_X, *_B);

    _M = std::make_unique<GSSmoother>(*_A);

    auto cg = std::make_unique<CGSolver>();
    cg->SetPreconditioner(*_M);
    cg->SetOperator(*_A);
    cg->SetRelTol(1e-12);
    cg->SetMaxIter(10000);
    cg->SetPrintLevel(1);

    _solver = std::move(cg);

    _rigidSolver =
        std::make_unique<mfemElasticity::RigidBodySolver>(_fes.get());
    _rigidSolver->SetSolver(*_solver);
  }

  mfem::FiniteElementSpace& GetFES() override { return *_fes; }

  mfem::GridFunction& GetDisplacement() override { return _u; }

  /**
   * @brief Set the uniform unit traction right-hand side.
   * * @param t The current time.
   * @note Overwrites any existing boundary integrators on the linear form.
   */
  void SetRHS(mfem::real_t t) override {
    using namespace mfem;
    int dim = _mesh->Dimension();
    Vector tv(dim);
    tv = 0.0;
    tv[1] = 1 + t;

    _marker = std::make_unique<mfem::Array<int>>(_mesh->bdr_attributes.Max());
    *_marker = 0;
    _mesh->MarkExternalBoundaries(*_marker);

    _tc = std::make_unique<VectorConstantCoefficient>(tv);
    _b = std::make_unique<LinearForm>(_fes.get());

    _b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*_tc), *_marker);
    _b->Assemble();
  }

  void IncrementRHS(const mfem::Vector& v) override { *_b += v; }

  /**
   * @brief Solve the linear elasticity system.
   * * Uses the pre-configured RigidBodySolver to project out the null space
   * and conjugate gradient (CG) to solve the resulting constrained system.
   */
  void Solve() override {
    using namespace mfem;
    Array<int> ess_tdof_list;

    _a->FormLinearSystem(ess_tdof_list, _u, *_b, *_A, *_X, *_B);
    _rigidSolver->Mult(*_B, *_X);
    _a->RecoverFEMSolution(*_X, *_b, _u);
  }
};

using namespace std;
using namespace mfem;

int main(int argc, char* argv[]) {
  // Set the default options.
  const char* mesh_file = "../data/star.mesh";
  int order = 1;
  int ref_levels = 0;

  // Read in command line options and process.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "number of mesh refinements");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // Read in the mesh and refine if requested.
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();
  for (int l = 0; l < ref_levels; l++) {
    mesh.UniformRefinement();
  }

  auto problem = TractionProblem(&mesh, order);

  problem.SetRHS(0);
  problem.Solve();

  auto& u = problem.GetDisplacement();

  // Write solution to file.
  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);
  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  u.Save(sol_ofs);

  // Visualise if glvis is open.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock.precision(8);
  sol_sock << "solution\n" << mesh << u << flush;
  if (dim == 2) {
    sol_sock << "keys Rjlvvvvvmm\n" << flush;
  } else {
    sol_sock << "keys m\n" << flush;
  }

  return 0;
}