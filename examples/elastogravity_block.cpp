#include <mfem.hpp>
#include <mfemElasticity.hpp>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace mfem;
constexpr real_t pi = 3.141592653589793238462643383279502884;
constexpr real_t G_const = 6.67430e-11;
constexpr real_t L_scale = 6371e3;
constexpr real_t rho_scale = 5000.0;

const real_t T_scale = 1.0 / sqrt(G_const * rho_scale);
const real_t gravity_scale = L_scale / (T_scale * T_scale);
const real_t potential_scale = L_scale * L_scale / (T_scale * T_scale);
const real_t stress_scale = rho_scale * L_scale * L_scale / (T_scale * T_scale);

real_t rho_func(const Vector &coord);
real_t mu_func(const Vector &coord);
real_t lamb_func(const Vector &coord);
real_t loading_func(const Vector &coord);

real_t azimuthal_func(const Vector &coord);
real_t polar_func(const Vector &coord);

class RigidTranslationCoefficient : public VectorCoefficient {
 private:
  int _component;

 public:
  RigidTranslationCoefficient(int dimension, int component)
      : VectorCoefficient(dimension), _component(component) {
    MFEM_ASSERT(component >= 0 && component < dimension,
                "component out of range");
  }

  void Eval(Vector &V, ElementTransformation &T,
            const IntegrationPoint &ip) override {
    V.SetSize(vdim);
    V = 0.0;
    V[_component] = 1.0;
  }
};

class RigidRotationCoefficient : public VectorCoefficient {
 private:
  int _component;
  Vector _x;

 public:
  RigidRotationCoefficient(int dimension, int component)
      : VectorCoefficient(dimension), _component(component) {
    MFEM_ASSERT(dimension == 2 || dimension == 3, "dimension must be 2 or 3");

    if (dimension == 2) {
      MFEM_ASSERT(component == 2, "in 2D only z-rotation is defined");
    } else {
      MFEM_ASSERT(component >= 0 && component < 3,
                  "rotation component out of range");
    }
  }

  void Eval(Vector &V, ElementTransformation &T,
            const IntegrationPoint &ip) override {
    V.SetSize(vdim);
    V = 0.0;

    _x.SetSize(vdim);
    T.Transform(ip, _x);

    if (_component == 0) {
      V[0] = 0.0;
      V[1] = -_x[2];
      V[2] = _x[1];
    } else if (_component == 1) {
      V[0] = _x[2];
      V[1] = 0.0;
      V[2] = -_x[0];
    } else {
      V[0] = -_x[1];
      V[1] = _x[0];

      if (vdim == 3) {
        V[2] = 0.0;
      }
    }
  }
};

class BlockRigidBodySolverLocal : public Solver {
 private:
  FiniteElementSpace *_fes_u;
  FiniteElementSpace *_fes_phi;
  Array<int> *_block_offsets;
  VectorCoefficient *_dphi0_coeff;

  std::vector<BlockVector *> _ns;

  Solver *_solver = nullptr;

  mutable BlockVector _b;
  mutable BlockVector _x;

  real_t Dot(const Vector &x, const Vector &y) const {
    return InnerProduct(x, y);
  }

  real_t Norm(const Vector &x) const { return std::sqrt(Dot(x, x)); }

  real_t BlockDot(const BlockVector &x, const BlockVector &y) const {
    const real_t alpha_phi = 0.0;

    const real_t u_norm_y = y.GetBlock(0).Norml2();

    if (u_norm_y < 1e-30) {
      return InnerProduct(x.GetBlock(1), y.GetBlock(1));
    }

    return InnerProduct(x.GetBlock(0), y.GetBlock(0)) +
           alpha_phi * InnerProduct(x.GetBlock(1), y.GetBlock(1));
  }

  real_t BlockNorm(const BlockVector &x) const {
    return std::sqrt(std::max(BlockDot(x, x), real_t(0.0)));
  }

  int ElasticRigidBodyDim() const {
    int dim = _fes_u->GetVDim();
    return dim * (dim + 1) / 2;
  }

  void AddCoupledRigidMode(VectorCoefficient &u_coeff) {
    GridFunction u_gf(_fes_u);
    GridFunction phi_gf(_fes_phi);

    u_gf = 0.0;
    phi_gf = 0.0;

    u_gf.ProjectCoefficient(u_coeff);

    InnerProductCoefficient phi_coeff(u_coeff, *_dphi0_coeff);

    phi_gf.ProjectCoefficient(phi_coeff);
    phi_gf.Neg();

    BlockVector *nv = new BlockVector(*_block_offsets);
    *nv = 0.0;

    u_gf.GetTrueDofs(nv->GetBlock(0));
    phi_gf.GetTrueDofs(nv->GetBlock(1));

    _ns.push_back(nv);
  }

  void AddPurePhiConstantMode() {
    GridFunction phi_gf(_fes_phi);
    phi_gf = 1.0;

    BlockVector *nv = new BlockVector(*_block_offsets);
    *nv = 0.0;

    phi_gf.GetTrueDofs(nv->GetBlock(1));

    _ns.push_back(nv);
  }

  void GramSchmidt() {
    for (int i = 0; i < (int)_ns.size(); i++) {
      BlockVector &nv1 = *_ns[i];

      for (int j = 0; j < i; j++) {
        BlockVector &nv2 = *_ns[j];
        // real_t product = Dot(nv1, nv2);
        real_t product = BlockDot(nv1, nv2);
        nv1.Add(-product, nv2);
      }

      // real_t norm = Norm(nv1);
      real_t norm = BlockNorm(nv1);

      MFEM_VERIFY(norm > 0.0,
                  "zero nullspace vector in BlockRigidBodySolverLocal");

      nv1 /= norm;
    }
  }

 public:
  /*void ProjectOrthogonalToNullspace(const Vector &x, Vector &y) const
  {
      y = x;

      for (int i = 0; i < (int)_ns.size(); i++)
      {
          const BlockVector &nv = *_ns[i];
          real_t product = Dot(y, nv);
          y.Add(-product, nv);
      }
  }*/

  void ProjectOrthogonalToNullspace(const Vector &x, Vector &y) const {
    y = x;

    Vector y_u(y.GetData() + (*_block_offsets)[0],
               (*_block_offsets)[1] - (*_block_offsets)[0]);

    Vector y_phi(y.GetData() + (*_block_offsets)[1],
                 (*_block_offsets)[2] - (*_block_offsets)[1]);

    for (int i = 0; i < (int)_ns.size(); i++) {
      const BlockVector &nv = *_ns[i];

      real_t product;

      if (nv.GetBlock(0).Norml2() < 1e-30) {
        product = InnerProduct(y_phi, nv.GetBlock(1));
      } else {
        product = InnerProduct(y_u, nv.GetBlock(0));
      }

      y.Add(-product, nv);
    }
  }

 public:
  BlockRigidBodySolverLocal(FiniteElementSpace *fes_u,
                            FiniteElementSpace *fes_phi,
                            Array<int> *block_offsets,
                            VectorCoefficient *dphi0_coeff)
      : Solver((*block_offsets)[2], false),
        _fes_u(fes_u),
        _fes_phi(fes_phi),
        _block_offsets(block_offsets),
        _dphi0_coeff(dphi0_coeff),
        _b(*block_offsets),
        _x(*block_offsets) {
    int dim = _fes_u->GetVDim();

    MFEM_ASSERT(dim == 2 || dim == 3, "dimensions must be two or three");

    height = (*_block_offsets)[2];
    width = height;

    for (int component = 0; component < dim; component++) {
      RigidTranslationCoefficient u_coeff(dim, component);
      AddCoupledRigidMode(u_coeff);
    }

    if (dim == 2) {
      RigidRotationCoefficient u_coeff(dim, 2);
      AddCoupledRigidMode(u_coeff);

      AddPurePhiConstantMode();
    } else {
      for (int component = 0; component < dim; component++) {
        RigidRotationCoefficient u_coeff(dim, component);
        AddCoupledRigidMode(u_coeff);
      }
    }

    GramSchmidt();

    cout << "Block nullspace dimension = " << _ns.size() << endl;
  }

  ~BlockRigidBodySolverLocal() {
    for (int i = 0; i < (int)_ns.size(); i++) {
      delete _ns[i];
    }
  }

  void SetSolver(Solver &solver) {
    _solver = &solver;

    height = _solver->Height();
    width = _solver->Width();

    MFEM_VERIFY(height == width, "solver must be square");
  }

  void SetOperator(const Operator &op) override {
    MFEM_VERIFY(_solver != nullptr, "SetSolver must be called first");

    _solver->SetOperator(op);

    height = _solver->Height();
    width = _solver->Width();

    MFEM_VERIFY(height == width, "solver must be square");
  }

  void Mult(const Vector &b, Vector &x) const override {
    MFEM_VERIFY(_solver != nullptr, "SetSolver must be called first");

    ProjectOrthogonalToNullspace(b, _b);

    _x = 0.0;
    _solver->Mult(_b, _x);

    ProjectOrthogonalToNullspace(_x, x);
  }
};

int main(int argc, char *argv[]) {
  StopWatch chrono;

  const char *mesh_file = "ex5_2d.msh";
  real_t rel_tol = 1e-10;
  int order_u = 1;
  int deg = 16;
  bool visualization = false;

  real_t shifting_factor = 1e-3;

  OptionsParser args(argc, argv);

  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");

  args.AddOption(&rel_tol, "-rt", "--rel-tol",
                 "Relative tolerance for linear solving.");

  args.AddOption(&order_u, "-o", "--order",
                 "Order degree of the finite elements.");

  args.AddOption(&deg, "-deg", "--degree",
                 "Truncation degree for the DtN map.");

  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");

  args.Parse();

  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }

  args.PrintOptions(cout);

  real_t G_nd = G_const * rho_scale * T_scale * T_scale;
  real_t poisson_rhs_factor = -4.0 * pi * G_nd;
  real_t phi_block_factor = 1.0 / (4.0 * pi * G_nd);
  real_t surface_load_scale = rho_scale * L_scale;

  cout << "\nDimensionless constants:\n";
  cout << "  G_nd                  = " << G_nd << endl;
  cout << "  -4*pi*G_nd            = " << poisson_rhs_factor << endl;
  cout << "  1/(4*pi*G_nd)         = " << phi_block_factor << endl;

  cout << "\nReference scales:\n";
  cout << "  Length scale L        = " << L_scale << " m\n";
  cout << "  Time scale T          = " << T_scale << " s\n";
  cout << "  Density scale rho0    = " << rho_scale << " kg/m^3\n";
  cout << "  Potential scale Phi0  = " << potential_scale << " m^2/s^2\n";
  cout << "  Gravity scale g0      = " << gravity_scale << " m/s^2\n";
  cout << "  Stress scale p0       = " << stress_scale << " Pa\n";
  cout << "  Surface load scale    = " << surface_load_scale << " kg/m^2\n";
  cout << endl;

  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  cout << "Mesh dimension: " << dim << endl;
  cout << "Domain attributes: ";
  mesh->attributes.Print(cout);
  cout << "Boundary attributes: ";
  mesh->bdr_attributes.Print(cout);

  Array<int> attr_cond(mesh->attributes.Max());
  attr_cond = 0;
  attr_cond[0] = 1;

  SubMesh mesh_cond(SubMesh::CreateFromDomain(*mesh, attr_cond));

  int order_phi = order_u;
  int order_dphi = order_phi - 1;

  H1_FECollection fec_u(order_u, dim);
  H1_FECollection fec_phi(order_phi, dim);
  L2_FECollection fec_dphi(order_dphi, dim);

  FiniteElementSpace fes_phi(mesh, &fec_phi);
  FiniteElementSpace fes_phi_cond(&mesh_cond, &fec_phi);
  FiniteElementSpace fes_dphi_cond(&mesh_cond, &fec_dphi, dim);
  FiniteElementSpace fes_u(&mesh_cond, &fec_u, dim);

  cout << "Number of u-unknowns: " << fes_u.GetVSize() << endl;
  cout << "Number of phi-unknowns: " << fes_phi.GetVSize() << endl;

  GridFunction u_gf(&fes_u);
  GridFunction phi_gf(&fes_phi);
  GridFunction phi_gf_cond(&fes_phi_cond);

  GridFunction phi0_gf(&fes_phi);
  GridFunction phi0_gf_cond(&fes_phi_cond);
  GridFunction dphi0_gf_cond(&fes_dphi_cond);

  u_gf = 0.0;
  phi_gf = 0.0;
  phi_gf_cond = 0.0;
  phi0_gf = 0.0;
  phi0_gf_cond = 0.0;
  dphi0_gf_cond = 0.0;

  FunctionCoefficient rho_coeff(rho_func);
  FunctionCoefficient mu_coeff(mu_func);
  FunctionCoefficient lamb_coeff(lamb_func);
  FunctionCoefficient loading_coeff(loading_func);

  Array<int> ess_tdof_list;

  Array<int> bdr_marker(mesh->bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[mesh->bdr_attributes.Max() - 2] = 1;

  Array<int> bdr_marker_outer(mesh->bdr_attributes.Max());
  bdr_marker_outer = 0;
  bdr_marker_outer[mesh->bdr_attributes.Max() - 1] = 1;

  Array<int> bdr_marker_cond(mesh_cond.bdr_attributes.Max());
  bdr_marker_cond = 0;
  bdr_marker_cond[mesh_cond.bdr_attributes.Max() - 1] = 1;

  auto DtN = mfemElasticity::PoissonDtNOperator(&fes_phi, deg);
  DtN.Assemble();

  ConstantCoefficient one(1.0);
  ProductCoefficient rhs_coeff(poisson_rhs_factor, rho_coeff);

  LinearForm b0(&fes_phi);
  b0.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
  b0.Assemble();

  if (dim == 2) {
    phi0_gf = 1.0;

    real_t mass = b0(phi0_gf);

    LinearForm l(&fes_phi);
    l.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), bdr_marker_outer);
    l.Assemble();

    real_t length = l(phi0_gf);

    b0.Add(-mass / length, l);
  }

  BilinearForm a0(&fes_phi);
  a0.AddDomainIntegrator(new DiffusionIntegrator(one));
  a0.Assemble();

  ConstantCoefficient eps0(shifting_factor);

  BilinearForm a0s(&fes_phi);
  a0s.AddDomainIntegrator(new DiffusionIntegrator(one));
  a0s.AddDomainIntegrator(new MassIntegrator(eps0));
  a0s.Assemble();
  a0s.Finalize();

  OperatorPtr A0;
  Vector B0, Phi0;

  a0.FormLinearSystem(ess_tdof_list, phi0_gf, b0, A0, Phi0, B0);

  cout << "Size of equilibrium linear system: " << A0->Height() << endl;

  auto S0 = SumOperator(A0.Ptr(), 1.0, &DtN, 1.0, false, false);

  SparseMatrix A0s;
  a0s.FormSystemMatrix(ess_tdof_list, A0s);

  GSSmoother M0(A0s);

  CGSolver solver0;
  solver0.SetOperator(S0);
  solver0.SetPreconditioner(M0);
  solver0.SetRelTol(rel_tol);
  solver0.SetMaxIter(3000);
  solver0.SetPrintLevel(0);

  if (dim == 2) {
    OrthoSolver ortho_solver0;
    ortho_solver0.SetSolver(solver0);
    ortho_solver0.Mult(B0, Phi0);
  } else {
    solver0.Mult(B0, Phi0);
  }

  a0.RecoverFEMSolution(Phi0, b0, phi0_gf);

  DiscreteLinearOperator Grad(&fes_phi_cond, &fes_dphi_cond);
  Grad.AddDomainInterpolator(new GradientInterpolator);
  Grad.Assemble();

  mesh_cond.Transfer(phi0_gf, phi0_gf_cond);
  Grad.Mult(phi0_gf_cond, dphi0_gf_cond);

  GradientGridFunctionCoefficient dphi0_coeff(&phi0_gf);
  VectorGridFunctionCoefficient dphi0_cond_coeff(&dphi0_gf_cond);

  ScalarVectorProductCoefficient dphi0_sig_cond_coeff(loading_coeff,
                                                      dphi0_cond_coeff);

  cout << "Equilibrium state computed." << endl;

  if (visualization) {
    GridFunction phi0_vis(phi0_gf);
    phi0_vis *= potential_scale;

    char vishost[] = "localhost";
    int visport = 19916;

    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);

    sol_sock << "solution\n"
             << *mesh << phi0_vis
             << "window_title 'Dimensional equilibrium potential [m^2/s^2]'"
             << flush;

    if (dim == 2) {
      sol_sock << "keys Rjlbc\n" << flush;
    } else {
      sol_sock << "keys RRRilmc\n" << flush;
    }
  }

  Array<int> block_offsets(3);
  block_offsets[0] = 0;
  block_offsets[1] = fes_u.GetVSize();
  block_offsets[2] = fes_phi.GetVSize();
  block_offsets.PartialSum();

  cout << "***********************************************************\n";
  cout << "dim(u)       = " << block_offsets[1] - block_offsets[0] << "\n";
  cout << "dim(phi)     = " << block_offsets[2] - block_offsets[1] << "\n";
  cout << "dim(u+phi)   = " << block_offsets.Last() << "\n";
  cout << "***********************************************************\n";

  BlockVector X(block_offsets);
  BlockVector Rhs(block_offsets);

  X = 0.0;
  Rhs = 0.0;

  LinearForm b1(&fes_u);
  b1.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(dphi0_sig_cond_coeff),
                           bdr_marker_cond);
  b1.Assemble();

  LinearForm b2(&fes_phi);
  b2.AddBoundaryIntegrator(new BoundaryLFIntegrator(loading_coeff), bdr_marker);
  b2.Assemble();

  if (dim == 2) {
    GridFunction one_phi(&fes_phi);
    one_phi = 1.0;

    LinearForm outer_l(&fes_phi);
    outer_l.AddBoundaryIntegrator(new BoundaryLFIntegrator(one),
                                  bdr_marker_outer);
    outer_l.Assemble();

    real_t mass = b2(one_phi);
    real_t outer_length = outer_l(one_phi);

    b2.Add(-mass / outer_length, outer_l);
  }

  Rhs.GetBlock(0) = b1;
  Rhs.GetBlock(1) = b2;

  BilinearForm *a11(new BilinearForm(&fes_u));
  BilinearForm *a22(new BilinearForm(&fes_phi));

  auto a12 = new mfemElasticity::MixedBilinearFormSubMesh(&fes_phi, &fes_u,
                                                          &fes_phi_cond, true);

  auto a21 = new mfemElasticity::MixedBilinearFormSubMesh(&fes_u, &fes_phi,
                                                          &fes_phi_cond, false);

  ConstantCoefficient c0(phi_block_factor);

  ProductCoefficient half_rho_coeff(0.5, rho_coeff);
  ProductCoefficient minus_half_rho_coeff(-0.5, rho_coeff);

  auto *a11_integ_0 = new ElasticityIntegrator(lamb_coeff, mu_coeff);

  auto *a11_integ_1 = new mfemElasticity::DomainVectorGradVectorIntegrator(
      dphi0_cond_coeff, half_rho_coeff);

  ScalarVectorProductCoefficient a11_integ_2_coeff(minus_half_rho_coeff,
                                                   dphi0_cond_coeff);

  auto *a11_integ_2 =
      new mfemElasticity::DomainVectorDivVectorIntegrator(a11_integ_2_coeff);

  auto *a11_integ_1_t = new TransposeIntegrator(a11_integ_1, 0);

  auto *a11_integ_2_t = new TransposeIntegrator(a11_integ_2, 0);

  a11->AddDomainIntegrator(a11_integ_0);
  a11->AddDomainIntegrator(a11_integ_1);
  a11->AddDomainIntegrator(a11_integ_2);
  a11->AddDomainIntegrator(a11_integ_1_t);
  a11->AddDomainIntegrator(a11_integ_2_t);
  a11->Assemble();
  a11->Finalize();

  a22->AddDomainIntegrator(new DiffusionIntegrator(c0));
  a22->Assemble();
  a22->Finalize();

  ConstantCoefficient eps22(shifting_factor * phi_block_factor);

  BilinearForm *a22s(new BilinearForm(&fes_phi));
  a22s->AddDomainIntegrator(new DiffusionIntegrator(c0));
  a22s->AddDomainIntegrator(new MassIntegrator(eps22));
  a22s->Assemble();
  a22s->Finalize();

  a12->AddDomainIntegrator(new GradientIntegrator(rho_coeff));
  a12->Assemble();
  a12->Finalize();

  a21->AddDomainIntegrator(
      new TransposeIntegrator(new GradientIntegrator(rho_coeff)));
  a21->Assemble();
  a21->Finalize();

  SparseMatrix &A11(a11->SpMat());
  SparseMatrix &A22_0(a22->SpMat());
  SparseMatrix &A22s(a22s->SpMat());
  SparseMatrix &A12(a12->SpMat());
  SparseMatrix &A21(a21->SpMat());

  auto A22 = SumOperator(&A22_0, 1.0, &DtN, phi_block_factor, false, false);

  cout << "Symmetry tests: A11 = " << A11.IsSymmetric()
       << ", A22_0 = " << A22_0.IsSymmetric() << endl;

  BlockOperator EGOp(block_offsets);

  EGOp.SetBlock(0, 0, &A11);
  EGOp.SetBlock(0, 1, &A12);
  EGOp.SetBlock(1, 0, &A21);
  EGOp.SetBlock(1, 1, &A22);

  GSSmoother prec11(A11);
  GSSmoother prec22(A22s);

  BlockDiagonalPreconditioner EGPrec(block_offsets);
  EGPrec.SetDiagonalBlock(0, &prec11);
  EGPrec.SetDiagonalBlock(1, &prec22);

  MINRESSolver solver;
  solver.SetRelTol(rel_tol);
  solver.SetAbsTol(0.0);
  solver.SetMaxIter(5000);
  solver.SetOperator(EGOp);
  solver.SetPreconditioner(EGPrec);
  solver.SetPrintLevel(1);

  BlockRigidBodySolverLocal rigid_solver(&fes_u, &fes_phi, &block_offsets,
                                         &dphi0_coeff);

  rigid_solver.SetSolver(solver);

  chrono.Clear();
  chrono.Start();

  rigid_solver.Mult(Rhs, X);

  chrono.Stop();

  if (solver.GetConverged()) {
    cout << "Block MINRES converged in " << solver.GetNumIterations()
         << " iterations with residual norm " << solver.GetFinalNorm() << "."
         << endl;
  } else {
    cout << "Block MINRES did not converge in " << solver.GetNumIterations()
         << " iterations. Residual norm is " << solver.GetFinalNorm() << "."
         << endl;
  }

  cout << "Block solve takes " << chrono.RealTime() << "s." << endl;

  u_gf.SetFromTrueDofs(X.GetBlock(0));
  phi_gf.SetFromTrueDofs(X.GetBlock(1));

  // ------------------------------------------------------------
  // Check remaining pure displacement rigid components and
  // pure phi constant component
  // ------------------------------------------------------------
  // Pure translations in u
  for (int c = 0; c < dim; c++) {
    RigidTranslationCoefficient trans(dim, c);

    GridFunction t_gf(&fes_u);
    t_gf = 0.0;
    t_gf.ProjectCoefficient(trans);

    Vector t_true(fes_u.GetVSize());
    t_gf.GetTrueDofs(t_true);

    real_t prod = InnerProduct(X.GetBlock(0), t_true);
    real_t rel =
        std::abs(prod) /
        std::max(X.GetBlock(0).Norml2() * t_true.Norml2(), real_t(1e-30));

    cout << "<u, pure translation " << c << "> / (||u||||t||) = " << rel
         << endl;
  }

  // Pure rotations in u
  if (dim == 2) {
    RigidRotationCoefficient rot(dim, 2);

    GridFunction r_gf(&fes_u);
    r_gf = 0.0;
    r_gf.ProjectCoefficient(rot);

    Vector r_true(fes_u.GetVSize());
    r_gf.GetTrueDofs(r_true);

    real_t prod = InnerProduct(X.GetBlock(0), r_true);
    real_t rel =
        std::abs(prod) /
        std::max(X.GetBlock(0).Norml2() * r_true.Norml2(), real_t(1e-30));

    cout << "<u, pure rotation z" << "> / (||u||||r||) = " << rel << endl;
  } else {
    for (int c = 0; c < dim; c++) {
      RigidRotationCoefficient rot(dim, c);

      GridFunction r_gf(&fes_u);
      r_gf = 0.0;
      r_gf.ProjectCoefficient(rot);

      Vector r_true(fes_u.GetVSize());
      r_gf.GetTrueDofs(r_true);

      real_t prod = InnerProduct(X.GetBlock(0), r_true);
      real_t rel =
          std::abs(prod) /
          std::max(X.GetBlock(0).Norml2() * r_true.Norml2(), real_t(1e-30));

      cout << "<u, pure rotation " << c << "> / (||u||||r||) = " << rel << endl;
    }
  }

  // Pure constant mode in phi
  {
    GridFunction one_phi_gf(&fes_phi);
    one_phi_gf = 1.0;

    Vector one_phi_true(fes_phi.GetVSize());
    one_phi_gf.GetTrueDofs(one_phi_true);

    real_t prod = InnerProduct(X.GetBlock(1), one_phi_true);
    real_t rel =
        std::abs(prod) /
        std::max(X.GetBlock(1).Norml2() * one_phi_true.Norml2(), real_t(1e-30));

    cout << "<phi, pure constant>" << " / (||phi||||1||) = " << rel << endl;
  }

  //
  mesh_cond.Transfer(phi_gf, phi_gf_cond);

  if (visualization) {
    GridFunction u_vis(u_gf);
    GridFunction phi_vis(phi_gf_cond);

    u_vis *= L_scale;
    phi_vis *= potential_scale;

    char vishost[] = "localhost";
    int visport = 19916;

    socketstream u_sock(vishost, visport);
    u_sock.precision(8);

    u_sock << "solution\n"
           << mesh_cond << u_vis << "window_title 'Dimensional deformation [m]'"
           << endl;

    if (dim == 2) {
      u_sock << "keys Rjlbc\n" << flush;
    } else {
      u_sock << "keys RRRilmc\n" << flush;
    }

    socketstream phi_sock(vishost, visport);
    phi_sock.precision(8);

    phi_sock
        << "solution\n"
        << mesh_cond << phi_vis
        << "window_title 'Dimensional gravity potential perturbation [m^2/s^2]'"
        << endl;

    if (dim == 2) {
      phi_sock << "keys Rjlbc\n" << flush;
    } else {
      phi_sock << "keys RRRilmc\n" << flush;
    }
  }

  delete a11;
  delete a12;
  delete a21;
  delete a22;
  delete a22s;
  delete mesh;

  return 0;
}

real_t azimuthal_func(const Vector &coord) {
  if (coord.Size() == 2) {
    return 0.0;
  }

  return sin(2.0 * atan2(coord[1], coord[0]));
}

real_t polar_func(const Vector &coord) {
  real_t r = coord.Norml2();

  if (r == 0.0) {
    return 0.0;
  }

  real_t theta;

  if (coord.Size() == 2) {
    theta = acos(coord[1] / r);
  } else {
    theta = acos(coord[2] / r);
  }

  return 0.015 * (1.0 + cos(2.0 * theta));
}

real_t rho_func(const Vector &coord) {
  real_t r = coord.Norml2();

  if (r > 1.0) {
    return 0.0;
  }

  real_t rho_surface = 2.6e3;
  real_t rho_center = 1.3e4;

  real_t rho_dim = rho_center + (rho_surface - rho_center) * r;

  return rho_dim / rho_scale;
}

real_t mu_func(const Vector &coord) {
  real_t r = coord.Norml2();

  real_t mu_surface = 70e9;
  real_t mu_center = 140e9;

  real_t mu_dim = mu_center + (mu_surface - mu_center) * r;

  real_t polar_perturb = polar_func(coord);
  real_t azimuthal_perturb = 0.05 * azimuthal_func(coord);

  mu_dim *= (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);

  return mu_dim / stress_scale;
}

real_t lamb_func(const Vector &coord) {
  real_t r = coord.Norml2();

  real_t lamb_surface = 100e9;
  real_t lamb_center = 300e9;

  real_t lamb_dim = lamb_center + (lamb_surface - lamb_center) * r;

  real_t polar_perturb = polar_func(coord);
  real_t azimuthal_perturb = 0.05 * azimuthal_func(coord);

  lamb_dim *= (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);

  return lamb_dim / stress_scale;
}

real_t loading_func(const Vector &coord) {
  real_t factor = 1e-1;

  real_t pressure_high = 10e6;
  real_t pressure_low = 1e6;

  real_t pressure_profile = 0.0;

  if (coord.Size() == 2) {
    real_t r = coord.Norml2();

    if (r == 0.0) {
      pressure_profile = pressure_high;
    } else {
      real_t theta = acos(coord[1] / r);

      pressure_profile =
          (pressure_low + pressure_high) / 2.0 +
          (pressure_high - pressure_low) / 2.0 * cos(2.0 * theta);
    }
  } else {
    real_t r = coord.Norml2();

    if (r == 0.0) {
      pressure_profile = pressure_high;
    } else {
      real_t theta = acos(coord[2] / r);

      pressure_profile =
          (pressure_low + pressure_high) / 2.0 +
          (pressure_high - pressure_low) / 2.0 * cos(2.0 * theta);
    }
  }

  real_t azimuthal_perturb = 0.2 * azimuthal_func(coord);

  real_t pressure_dim = -pressure_profile * (1.0 + azimuthal_perturb) * factor;

  real_t sigma_dim = pressure_dim / gravity_scale;

  return sigma_dim / (rho_scale * L_scale);
}
