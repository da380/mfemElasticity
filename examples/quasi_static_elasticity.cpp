

#include <memory>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

class QuasiStaticElasticProblem {
  // Virtual base class for quasi-static elastic problems
  // whose purpose is to set a common interface for their
  // interaction with viscoelastic solvers.

 public:
  // Return a reference to the finite element space.
  virtual mfem::FiniteElementSpace& GetFES() = 0;

  // Return a reference to the GridFunction for the displacement.
  virtual mfem::GridFunction& GetDisplacement() = 0;

  // Set the RHS at the time t.
  // virtual void SetRHS(mfem::real_t t) = 0;

  // Increment the RHS by a given vector.
  // virtual void IncrementRHS(const mfem::Vector& v) = 0;

  // Solve the linear system.
  // virtual void Solve() = 0;
};

class TractionProblem : public QuasiStaticElasticProblem {
 private:
  mfem::Mesh* _mesh;
  mfem::FiniteElementCollection* _fec;
  mfem::FiniteElementSpace* _fes;

  mfem::LinearForm* _b;
  mfem::BilinearForm* _a;
  mfem::GridFunction _u;

 public:
  TractionProblem(mfem::Mesh* mesh, int order) : _mesh(mesh) {
    using namespace mfem;
    // Set up the FES
    auto dim = _mesh->Dimension();
    auto _fec = new H1_FECollection(order, dim);
    auto _fes = new FiniteElementSpace(_mesh, _fec, dim);

    // Set up the linear form
    auto tv = Vector(dim);
    tv = 0.0;
    tv[0] = 1;
    auto tc = VectorConstantCoefficient(tv);

    _b = new LinearForm(_fes);
    _b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(tc));
    _b->Assemble();

    // Set up the bilinear form
    auto lambda = ConstantCoefficient(1);
    auto mu = ConstantCoefficient(1);
    _a = new BilinearForm(_fes);
    _a->AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
    _a->Assemble();

    // Set up the GridFunction
    _u = GridFunction(_fes);
    _u = 0.0;
  }

  ~TractionProblem() {
    delete _fec;
    delete _fes;
    delete _b;
    delete _a;
  }

  mfem::FiniteElementSpace& GetFES() override { return *_fes; }

  mfem::GridFunction& GetDisplacement() override { return _u; }
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

  /*

  // Set up the finite element space.
  auto fec = H1_FECollection(order, dim);
  auto fes = FiniteElementSpace(&mesh, &fec, dim);
  cout << "Number of finite element unknowns: " << fes.GetTrueVSize() << endl;

  // Set up the constant traction vector coefficient.
  auto tv = Vector(dim);
  tv = 0.0;
  tv[0] = 1;
  auto tc = VectorConstantCoefficient(tv);


  // Set up the linear form.
  auto b = LinearForm(&fes);
  b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(tc));
  b.Assemble();

  // Set up the bilinear form
  auto lambda = ConstantCoefficient(1);
  auto mu = ConstantCoefficient(1);
  auto a = BilinearForm(&fes);
  a.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
  a.Assemble();

  // Set up the gridfunction.
  auto x = GridFunction(&fes);
  x = 0.0;

  // Set the linear system.
  Array<int> ess_tdof_list;
  SparseMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
  cout << "Size of linear system: " << A.Height() << endl;

  // Set the preconditioner.
  GSSmoother M(A);

  // Set the solver.
  auto solver = CGSolver();
  solver.SetPreconditioner(M);
  solver.SetOperator(A);
  solver.SetRelTol(1e-12);
  solver.SetMaxIter(10000);
  solver.SetPrintLevel(1);

  // Set up the rigid body solver.
  auto rigidSolver = mfemElasticity::RigidBodySolver(&fes);
  rigidSolver.SetSolver(solver);

  // Solve the equations.
  rigidSolver.Mult(B, X);
  a.RecoverFEMSolution(X, b, x);

  // Write solution to file.
  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);
  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  x.Save(sol_ofs);

  // Visualise if glvis is open.
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock.precision(8);
  sol_sock << "solution\n" << mesh << x << flush;
  if (dim == 2) {
    sol_sock << "keys Rjlvvvvvmm\n" << flush;
  } else {
    sol_sock << "keys m\n" << flush;
  }

  */
  return 0;
}
