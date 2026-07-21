#pragma once

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

class Nondimensionalisation {
 private:
  real_t L;    // Length scale [m]
  real_t T;    // Time scale [s]
  real_t RHO;  // Density scale [kg/m^3]

 public:
  Nondimensionalisation(real_t length_scale, real_t time_scale,
                        real_t density_scale)
      : L(length_scale), T(time_scale), RHO(density_scale) {}

  // Accessors
  real_t Length() const { return L; }
  real_t Time() const { return T; }
  real_t Density() const { return RHO; }

  // Derived scales
  real_t Velocity() const { return L / T; }
  real_t Acceleration() const { return L / (T * T); }
  real_t Pressure() const { return RHO * L * L / (T * T); }  // [Pa]
  real_t Gravity() const { return L / (T * T); }
  real_t Potential() const { return L * L / (T * T); }

  // Scaling functions for scalars
  real_t ScaleLength(real_t x) const { return x / L; }
  real_t UnscaleLength(real_t x_nd) const { return x_nd * L; }

  real_t ScaleDensity(real_t rho) const { return rho / RHO; }
  real_t UnscaleDensity(real_t rho_nd) const { return rho_nd * RHO; }

  real_t ScaleGravityPotential(real_t phi) const { return phi / Potential(); }
  real_t UnscaleGravityPotential(real_t phi_nd) const {
    return phi_nd * Potential();
  }

  real_t ScaleStress(real_t sigma) const { return sigma / Pressure(); }
  real_t UnscaleStress(real_t sigma_nd) const { return sigma_nd * Pressure(); }

  // Scaling for GridFunction fields
  void UnscaleGravityPotential(GridFunction &phi_gf) const {
    phi_gf *= Potential();
  }
  void UnscaleDisplacement(GridFunction &u_gf) const { u_gf *= L; }
  void UnscaleStress(GridFunction &sigma_gf) const { sigma_gf *= Pressure(); }

  // Create a scaled density coefficient from a dimensional one
  Coefficient *MakeScaledDensityCoefficient(Coefficient &rho_coeff) const {
    return new ProductCoefficient(1.0 / RHO, rho_coeff);
  }

  void Print() const {
    cout << "Scaling parameters:\n";
    cout << "  Length scale: " << L << " m\n";
    cout << "  Time scale: " << T << " s\n";
    cout << "  Density scale: " << RHO << " kg/m^3\n";
    cout << "  Gravity potential scale: " << Potential() << " m^2/s^2\n";
  }
};

struct Constants {
 public:
  static constexpr real_t G = 6.6743e-11;
  static constexpr real_t c = 2.99792458e8;
  static constexpr real_t h = 6.62607015e-34;
  static constexpr real_t _h = 1.054571817e-34;
  static constexpr real_t kB = 1.380649e-23;
  static constexpr real_t NA = 6.02214076e23;
  static constexpr real_t e = 1.602176634e-19;
  static constexpr real_t epi0 = 8.854187817e-12;
  static constexpr real_t mu0 = 1.25663706212e-6;

  static constexpr real_t R = 6371e3;
};

class RadialDerivativeCoefficient : public mfem::Coefficient {
 private:
  const mfem::GridFunction &u_gf;

 public:
  RadialDerivativeCoefficient(const mfem::GridFunction &u) : u_gf(u) {}

  mfem::real_t Eval(mfem::ElementTransformation &T,
                    const mfem::IntegrationPoint &ip) override {
    const int dim = T.GetDimension();

    mfem::Vector x(dim), grad_u(dim);

    T.SetIntPoint(&ip);
    T.Transform(ip, x);

    mfem::real_t r = x.Norml2();

    if (r < 1e-14) {
      r = 1e-14;
    }

    x /= r;

    u_gf.GetGradient(T, grad_u);

    return mfem::InnerProduct(x, grad_u);
  }
};

class NormCoefficient : public mfem::Coefficient {
 private:
  mfem::VectorCoefficient &v_coeff;

 public:
  NormCoefficient(mfem::VectorCoefficient &v) : v_coeff(v) {}

  mfem::real_t Eval(mfem::ElementTransformation &T,
                    const mfem::IntegrationPoint &ip) override {
    mfem::Vector v(v_coeff.GetVDim());

    T.SetIntPoint(&ip);

    v_coeff.Eval(v, T, ip);

    return v.Norml2();
  }
};

class BoundaryFluxIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient *Q;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector shape, normal, shape_projected;
#endif

 public:
  BoundaryFluxIntegrator(mfem::Coefficient &q) : Q(&q) {}

  void AssembleElementMatrix(const mfem::FiniteElement &el,
                             mfem::ElementTransformation &Tr,
                             mfem::DenseMatrix &elmat) override {
    const int nd = el.GetDof();
    const int dim = Tr.GetSpaceDim();

#ifdef MFEM_THREAD_SAFE
    mfem::Vector shape, normal, shape_projected;
#endif

    shape.SetSize(nd);
    normal.SetSize(dim);
    shape_projected.SetSize(dim * nd);

    elmat.SetSize(dim * nd);
    elmat = 0.0;

    const mfem::IntegrationRule *ir = IntRule;
    if (ir == nullptr) {
      const int order = 2 * el.GetOrder() + Tr.OrderW();
      ir = &mfem::IntRules.Get(el.GetGeomType(), order);
    }

    for (int q = 0; q < ir->GetNPoints(); q++) {
      const mfem::IntegrationPoint &ip = ir->IntPoint(q);

      Tr.SetIntPoint(&ip);
      el.CalcShape(ip, shape);

      mfem::CalcOrtho(Tr.Jacobian(), normal);

      const mfem::real_t normal_norm = normal.Norml2();
      if (normal_norm < 1e-30) {
        continue;
      }

      normal /= normal_norm;

      for (int d = 0; d < dim; d++) {
        for (int i = 0; i < nd; i++) {
          shape_projected[i + d * nd] = shape[i] * normal[d];
        }
      }

      mfem::real_t w = ip.weight * Tr.Weight();

      if (Q) {
        w *= Q->Eval(Tr, ip);
      }

      mfem::AddMult_a_VVt(w, shape_projected, elmat);
    }
  }
};

class BoundaryFluxMixedIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient *Q;
  mfem::real_t normal_sign;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector trial_shape, test_shape, normal, shape_projected;
#endif

 public:
  BoundaryFluxMixedIntegrator(mfem::Coefficient &q, mfem::real_t sign = 1.0)
      : Q(&q), normal_sign(sign) {}

  void AssembleElementMatrix2(const mfem::FiniteElement &trial_fe,
                              const mfem::FiniteElement &test_fe,
                              mfem::ElementTransformation &Tr,
                              mfem::DenseMatrix &elmat) override {
    const int tr_nd = trial_fe.GetDof();
    const int te_nd = test_fe.GetDof();
    const int dim = Tr.GetSpaceDim();

#ifdef MFEM_THREAD_SAFE
    mfem::Vector trial_shape, test_shape, normal, shape_projected;
#endif

    trial_shape.SetSize(tr_nd);
    test_shape.SetSize(te_nd);
    normal.SetSize(dim);
    shape_projected.SetSize(dim * te_nd);

    elmat.SetSize(dim * te_nd, tr_nd);
    elmat = 0.0;

    const mfem::IntegrationRule *ir = IntRule;
    if (ir == nullptr) {
      const int order = trial_fe.GetOrder() + test_fe.GetOrder() + Tr.OrderW();

      ir = &mfem::IntRules.Get(trial_fe.GetGeomType(), order);
    }

    for (int q = 0; q < ir->GetNPoints(); q++) {
      const mfem::IntegrationPoint &ip = ir->IntPoint(q);

      Tr.SetIntPoint(&ip);

      trial_fe.CalcShape(ip, trial_shape);
      test_fe.CalcShape(ip, test_shape);

      mfem::CalcOrtho(Tr.Jacobian(), normal);

      const mfem::real_t normal_norm = normal.Norml2();
      if (normal_norm < 1e-30) {
        continue;
      }

      normal *= normal_sign / normal_norm;

      for (int d = 0; d < dim; d++) {
        for (int i = 0; i < te_nd; i++) {
          shape_projected[i + d * te_nd] = test_shape[i] * normal[d];
        }
      }

      mfem::real_t w = ip.weight * Tr.Weight();

      if (Q) {
        w *= Q->Eval(Tr, ip);
      }

      shape_projected *= w;

      mfem::AddMultVWt(shape_projected, trial_shape, elmat);
    }
  }
};

class TwoBlockRigidBodySolver : public Solver {
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
    return InnerProduct(x.GetBlock(0), y.GetBlock(0)) +
           InnerProduct(x.GetBlock(1), y.GetBlock(1));
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

        real_t product = BlockDot(nv1, nv2);
        nv1.Add(-product, nv2);
      }

      real_t norm = BlockNorm(nv1);

      MFEM_VERIFY(norm > 0.0,
                  "zero nullspace vector in TwoBlockRigidBodySolver");

      nv1 /= norm;
    }
  }

 public:
  void ProjectOrthogonalToNullspace(const Vector &x, Vector &y) const {
    y = x;

    Vector y_u(y.GetData() + (*_block_offsets)[0],
               (*_block_offsets)[1] - (*_block_offsets)[0]);

    Vector y_phi(y.GetData() + (*_block_offsets)[1],
                 (*_block_offsets)[2] - (*_block_offsets)[1]);

    for (int i = 0; i < (int)_ns.size(); i++) {
      const BlockVector &nv = *_ns[i];

      real_t product = InnerProduct(y_u, nv.GetBlock(0)) +
                       InnerProduct(y_phi, nv.GetBlock(1));

      y.Add(-product, nv);
    }
  }

 public:
  TwoBlockRigidBodySolver(FiniteElementSpace *fes_u,
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
      RigidTranslation u_coeff(dim, component);
      AddCoupledRigidMode(u_coeff);
    }

    if (dim == 2) {
      RigidRotation u_coeff(dim, 2);
      AddCoupledRigidMode(u_coeff);

      AddPurePhiConstantMode();
    } else {
      for (int component = 0; component < dim; component++) {
        RigidRotation u_coeff(dim, component);
        AddCoupledRigidMode(u_coeff);
      }
    }

    GramSchmidt();

    cout << "Block nullspace dimension = " << _ns.size() << endl;
  }

  ~TwoBlockRigidBodySolver() {
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

class TwoBlockRigidBodySolverParallel : public Solver {
 private:
  MPI_Comm _comm;

  ParFiniteElementSpace *_fes_u;
  ParFiniteElementSpace *_fes_phi;

  Array<int> *_block_true_offsets;

  VectorCoefficient *_dphi0_coeff;

  std::vector<std::unique_ptr<BlockVector>> _ns;

  Solver *_solver = nullptr;

  mutable BlockVector _b;
  mutable BlockVector _x;

  bool _add_constant_phi_mode;

  real_t Dot(const Vector &x, const Vector &y) const {
    return InnerProduct(_comm, x, y);
  }

  real_t Norm(const Vector &x) const { return std::sqrt(Dot(x, x)); }

  real_t BlockDot(const BlockVector &x, const BlockVector &y) const {
    return InnerProduct(_comm, x.GetBlock(0), y.GetBlock(0)) +
           InnerProduct(_comm, x.GetBlock(1), y.GetBlock(1));
  }

  real_t BlockNorm(const BlockVector &x) const {
    return std::sqrt(std::max(BlockDot(x, x), real_t(0.0)));
  }

  void AddCoupledRigidMode(VectorCoefficient &u_coeff) {
    ParGridFunction u_gf(_fes_u);
    ParGridFunction phi_gf(_fes_phi);

    u_gf = 0.0;
    phi_gf = 0.0;

    u_gf.ProjectCoefficient(u_coeff);

    InnerProductCoefficient phi_coeff(u_coeff, *_dphi0_coeff);

    phi_gf.ProjectCoefficient(phi_coeff);
    phi_gf.Neg();

    auto nv = std::make_unique<BlockVector>(*_block_true_offsets);
    *nv = 0.0;

    u_gf.GetTrueDofs(nv->GetBlock(0));
    phi_gf.GetTrueDofs(nv->GetBlock(1));

    _ns.push_back(std::move(nv));
  }

  void AddPurePhiConstantMode() {
    ParGridFunction phi_gf(_fes_phi);
    phi_gf = 1.0;

    auto nv = std::make_unique<BlockVector>(*_block_true_offsets);
    *nv = 0.0;

    phi_gf.GetTrueDofs(nv->GetBlock(1));

    _ns.push_back(std::move(nv));
  }

  void GramSchmidt() {
    for (int i = 0; i < (int)_ns.size(); i++) {
      BlockVector &ni = *_ns[i];

      for (int j = 0; j < i; j++) {
        BlockVector &nj = *_ns[j];

        real_t product = BlockDot(ni, nj);
        ni.Add(-product, nj);
      }

      real_t norm = BlockNorm(ni);

      MFEM_VERIFY(norm > 0.0,
                  "zero nullspace vector in TwoBlockRigidBodySolverParallel");

      ni /= norm;
    }
  }

  void ProjectOrthogonalToNullspace(const Vector &x, Vector &y) const {
    y = x;

    Vector y_u(y.GetData() + (*_block_true_offsets)[0],
               (*_block_true_offsets)[1] - (*_block_true_offsets)[0]);

    Vector y_phi(y.GetData() + (*_block_true_offsets)[1],
                 (*_block_true_offsets)[2] - (*_block_true_offsets)[1]);

    for (int i = 0; i < (int)_ns.size(); i++) {
      const BlockVector &ni = *_ns[i];

      real_t product = InnerProduct(_comm, y_u, ni.GetBlock(0)) +
                       InnerProduct(_comm, y_phi, ni.GetBlock(1));

      y.Add(-product, ni);
    }
  }

 public:
  TwoBlockRigidBodySolverParallel(MPI_Comm comm, ParFiniteElementSpace *fes_u,
                                  ParFiniteElementSpace *fes_phi,
                                  Array<int> *block_true_offsets,
                                  VectorCoefficient *dphi0_coeff,
                                  bool add_constant_phi_mode)
      : Solver(0, false),
        _comm(comm),
        _fes_u(fes_u),
        _fes_phi(fes_phi),
        _block_true_offsets(block_true_offsets),
        _dphi0_coeff(dphi0_coeff),
        _b(*block_true_offsets),
        _x(*block_true_offsets),
        _add_constant_phi_mode(add_constant_phi_mode) {
    int dim = _fes_u->GetVDim();

    MFEM_ASSERT(dim == 2 || dim == 3, "dimension must be 2 or 3");

    height = (*_block_true_offsets)[2];
    width = height;

    for (int c = 0; c < dim; c++) {
      RigidTranslation u_coeff(dim, c);
      AddCoupledRigidMode(u_coeff);
    }

    if (dim == 2) {
      RigidRotation u_coeff(dim, 2);
      AddCoupledRigidMode(u_coeff);
    } else {
      for (int c = 0; c < dim; c++) {
        RigidRotation u_coeff(dim, c);
        AddCoupledRigidMode(u_coeff);
      }
    }

    if (_add_constant_phi_mode) {
      AddPurePhiConstantMode();
    }

    GramSchmidt();

    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank == 0) {
      cout << "Parallel block nullspace dimension = " << _ns.size() << endl;
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

    _solver->iterative_mode = iterative_mode;
    _solver->Mult(_b, _x);

    ProjectOrthogonalToNullspace(_x, x);
  }
};

class ThreeBlockRigidBodySolver : public Solver {
 private:
  FiniteElementSpace *_fes_u_ic;
  FiniteElementSpace *_fes_u_mantle;
  FiniteElementSpace *_fes_phi;

  Array<int> _block_offsets;

  const BlockOperator *_A;

  Solver *_A33_prec;
  MINRESSolver _A33_solver;

  Solver *_solver = nullptr;

  std::vector<std::unique_ptr<BlockVector>> _ns;
  std::vector<std::unique_ptr<BlockVector>> _clean;

  mutable BlockVector _b;
  mutable BlockVector _x;

  bool _add_constant_phi_mode;

  Vector _phi_constant;
  real_t _phi_constant_norm2 = 0.0;

  enum { U_IC_BLOCK = 0, U_MANTLE_BLOCK = 1, PHI_BLOCK = 2 };

  const Operator &B(int i, int j) const { return _A->GetBlock(i, j); }

  std::string AxisName(int c) const {
    if (c == 0) {
      return "x";
    }
    if (c == 1) {
      return "y";
    }
    if (c == 2) {
      return "z";
    }
    return "?";
  }

  void GetBlockView(const Vector &x, int block, Vector &xb) const {
    xb.SetDataAndSize(const_cast<real_t *>(x.GetData()) + _block_offsets[block],
                      _block_offsets[block + 1] - _block_offsets[block]);
  }

  real_t EuclideanDot(const BlockVector &x, const BlockVector &y) const {
    return InnerProduct(x.GetBlock(U_IC_BLOCK), y.GetBlock(U_IC_BLOCK)) +
           InnerProduct(x.GetBlock(U_MANTLE_BLOCK),
                        y.GetBlock(U_MANTLE_BLOCK)) +
           InnerProduct(x.GetBlock(PHI_BLOCK), y.GetBlock(PHI_BLOCK));
  }

  real_t EuclideanNorm(const BlockVector &x) const {
    return std::sqrt(std::max(EuclideanDot(x, x), real_t(0.0)));
  }

  real_t EuclideanVectorModeDot(const Vector &x,
                                const BlockVector &mode) const {
    Vector x_ic, x_mantle, x_phi;

    GetBlockView(x, U_IC_BLOCK, x_ic);
    GetBlockView(x, U_MANTLE_BLOCK, x_mantle);
    GetBlockView(x, PHI_BLOCK, x_phi);

    return InnerProduct(x_ic, mode.GetBlock(U_IC_BLOCK)) +
           InnerProduct(x_mantle, mode.GetBlock(U_MANTLE_BLOCK)) +
           InnerProduct(x_phi, mode.GetBlock(PHI_BLOCK));
  }

  void BuildPhiConstant() {
    _phi_constant.SetSize(_block_offsets[PHI_BLOCK + 1] -
                          _block_offsets[PHI_BLOCK]);

    _phi_constant = 0.0;

    GridFunction phi_gf(_fes_phi);
    phi_gf = 1.0;
    phi_gf.GetTrueDofs(_phi_constant);

    _phi_constant_norm2 = InnerProduct(_phi_constant, _phi_constant);
  }

  void ProjectPhiConstant(Vector &v) const {
    if (!_add_constant_phi_mode) {
      return;
    }

    if (_phi_constant_norm2 <= 0.0) {
      return;
    }

    real_t alpha = InnerProduct(v, _phi_constant) / _phi_constant_norm2;
    v.Add(-alpha, _phi_constant);
  }

  void ComputePhiMode(const Vector &q_ic, const Vector &q_mantle,
                      Vector &psi) const {
    Vector rhs(B(PHI_BLOCK, PHI_BLOCK).Height());

    rhs = 0.0;

    B(PHI_BLOCK, U_IC_BLOCK).AddMult(q_ic, rhs, -1.0);
    B(PHI_BLOCK, U_MANTLE_BLOCK).AddMult(q_mantle, rhs, -1.0);

    ProjectPhiConstant(rhs);

    psi.SetSize(B(PHI_BLOCK, PHI_BLOCK).Width());
    psi = 0.0;

    _A33_solver.Mult(rhs, psi);

    ProjectPhiConstant(psi);
  }

  void PrintModeResidual(const std::string &name, const BlockVector &z) const {
    BlockVector r(_block_offsets);

    r = 0.0;

    _A->Mult(z, r);

    real_t z_norm = EuclideanNorm(z);
    z_norm = std::max(z_norm, real_t(1e-30));

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Rigid mode residual [" << name << "]" << std::endl;
    std::cout << "  row 1 abs = " << r.GetBlock(U_IC_BLOCK).Norml2()
              << ", /mode = " << r.GetBlock(U_IC_BLOCK).Norml2() / z_norm
              << std::endl;
    std::cout << "  row 2 abs = " << r.GetBlock(U_MANTLE_BLOCK).Norml2()
              << ", /mode = " << r.GetBlock(U_MANTLE_BLOCK).Norml2() / z_norm
              << std::endl;
    std::cout << "  row 3 abs = " << r.GetBlock(PHI_BLOCK).Norml2()
              << ", /mode = " << r.GetBlock(PHI_BLOCK).Norml2() / z_norm
              << std::endl;
  }

  void AddGlobalCoupledRigidMode(VectorCoefficient &u_coeff,
                                 const std::string &name) {
    Vector q_ic(_block_offsets[U_IC_BLOCK + 1] - _block_offsets[U_IC_BLOCK]);

    Vector q_mantle(_block_offsets[U_MANTLE_BLOCK + 1] -
                    _block_offsets[U_MANTLE_BLOCK]);

    Vector psi(_block_offsets[PHI_BLOCK + 1] - _block_offsets[PHI_BLOCK]);

    q_ic = 0.0;
    q_mantle = 0.0;
    psi = 0.0;

    GridFunction u_ic_gf(_fes_u_ic);
    GridFunction u_mantle_gf(_fes_u_mantle);

    u_ic_gf = 0.0;
    u_mantle_gf = 0.0;

    u_ic_gf.ProjectCoefficient(u_coeff);
    u_mantle_gf.ProjectCoefficient(u_coeff);

    u_ic_gf.GetTrueDofs(q_ic);
    u_mantle_gf.GetTrueDofs(q_mantle);

    ComputePhiMode(q_ic, q_mantle, psi);

    auto nv = std::make_unique<BlockVector>(_block_offsets);
    auto cv = std::make_unique<BlockVector>(_block_offsets);

    *nv = 0.0;

    nv->GetBlock(U_IC_BLOCK) = q_ic;
    nv->GetBlock(U_MANTLE_BLOCK) = q_mantle;
    nv->GetBlock(PHI_BLOCK) = psi;

    PrintModeResidual(name, *nv);

    *cv = *nv;

    _ns.push_back(std::move(nv));
    _clean.push_back(std::move(cv));
  }

  void AddGlobalRigidModes(int dim) {
    for (int c = 0; c < dim; c++) {
      RigidTranslation u_coeff(dim, c);

      AddGlobalCoupledRigidMode(u_coeff, "global T" + AxisName(c));
    }

    if (dim == 2) {
      RigidRotation u_coeff(dim, 2);

      AddGlobalCoupledRigidMode(u_coeff, "global Rz");
    } else {
      for (int c = 0; c < dim; c++) {
        RigidRotation u_coeff(dim, c);

        AddGlobalCoupledRigidMode(u_coeff, "global R" + AxisName(c));
      }
    }
  }

  void AddPurePhiConstantMode() {
    auto nv = std::make_unique<BlockVector>(_block_offsets);
    auto cv = std::make_unique<BlockVector>(_block_offsets);

    *nv = 0.0;

    nv->GetBlock(PHI_BLOCK) = _phi_constant;

    *cv = *nv;

    _ns.push_back(std::move(nv));
    _clean.push_back(std::move(cv));
  }

  void GramSchmidt(std::vector<std::unique_ptr<BlockVector>> &modes) {
    std::vector<std::unique_ptr<BlockVector>> modes_orth;

    for (int i = 0; i < (int)modes.size(); i++) {
      auto ni = std::make_unique<BlockVector>(_block_offsets);

      *ni = *modes[i];

      for (int j = 0; j < (int)modes_orth.size(); j++) {
        BlockVector &nj = *modes_orth[j];

        real_t product = EuclideanDot(*ni, nj);

        ni->Add(-product, nj);
      }

      real_t norm = EuclideanNorm(*ni);

      if (norm <= 1e-24) {
        continue;
      }

      *ni /= norm;

      modes_orth.push_back(std::move(ni));
    }

    modes.swap(modes_orth);
  }

  void ProjectOrthogonalToModesEuclidean(
      const Vector &x, Vector &y,
      const std::vector<std::unique_ptr<BlockVector>> &modes) const {
    y = x;

    for (int i = 0; i < (int)modes.size(); i++) {
      const BlockVector &ni = *modes[i];

      real_t product = EuclideanVectorModeDot(y, ni);

      y.Add(-product, ni);
    }
  }

 public:
  ThreeBlockRigidBodySolver(FiniteElementSpace *fes_u_ic,
                            FiniteElementSpace *fes_u_mantle,
                            FiniteElementSpace *fes_phi,
                            Array<int> *block_offsets, const BlockOperator &A,
                            Solver &prec33, bool add_constant_phi_mode)
      : Solver(A.Width(), false),
        _fes_u_ic(fes_u_ic),
        _fes_u_mantle(fes_u_mantle),
        _fes_phi(fes_phi),
        _block_offsets(*block_offsets),
        _A(&A),
        _A33_prec(&prec33),
        _b(_block_offsets),
        _x(_block_offsets),
        _add_constant_phi_mode(add_constant_phi_mode) {
    int dim = _fes_u_ic->GetVDim();

    MFEM_ASSERT(dim == 2 || dim == 3, "dimension must be 2 or 3");

    MFEM_ASSERT(_fes_u_mantle->GetVDim() == dim,
                "solid displacement spaces must have the same vdim");

    MFEM_VERIFY(A.Height() == A.Width(),
                "BlockRigidBodySolverSLS requires a square block operator");

    MFEM_VERIFY(_block_offsets.Size() == 4,
                "BlockRigidBodySolverSLS expects three blocks");

    MFEM_VERIFY(
        _block_offsets[1] - _block_offsets[0] == _fes_u_ic->GetTrueVSize(),
        "inner-core block size mismatch");

    MFEM_VERIFY(
        _block_offsets[2] - _block_offsets[1] == _fes_u_mantle->GetTrueVSize(),
        "mantle block size mismatch");

    MFEM_VERIFY(
        _block_offsets[3] - _block_offsets[2] == _fes_phi->GetTrueVSize(),
        "phi block size mismatch");

    height = A.Height();
    width = A.Width();

    BuildPhiConstant();

    _A33_solver.SetRelTol(1e-10);
    _A33_solver.SetAbsTol(0.0);
    _A33_solver.SetMaxIter(5000);
    _A33_solver.SetPrintLevel(0);
    _A33_solver.SetOperator(B(PHI_BLOCK, PHI_BLOCK));
    _A33_solver.SetPreconditioner(*_A33_prec);
    _A33_solver.iterative_mode = false;

    AddGlobalRigidModes(dim);

    if (_add_constant_phi_mode) {
      AddPurePhiConstantMode();
    }

    GramSchmidt(_ns);
    GramSchmidt(_clean);

    MFEM_VERIFY(_ns.size() > 0, "empty nullspace in ThreeBlockRigidBodySolver");

    std::cout << "Block SLS1 nullspace dimension = " << _ns.size() << std::endl;

    std::cout << "Block SLS1 Euclidean cleaning dimension = " << _clean.size()
              << std::endl;
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

    ProjectOrthogonalToModesEuclidean(b, _b, _ns);

    _x = 0.0;

    _solver->iterative_mode = iterative_mode;
    _solver->Mult(_b, _x);

    ProjectOrthogonalToModesEuclidean(_x, x, _clean);
  }
};

class ThreeBlockRigidBodySolverParallel : public Solver {
 private:
  MPI_Comm _comm;

  ParFiniteElementSpace *_fes_u_ic;
  ParFiniteElementSpace *_fes_u_mantle;
  ParFiniteElementSpace *_fes_phi;

  Array<int> _block_true_offsets;

  const BlockOperator *_A;

  Solver *_A33_prec;
  MINRESSolver _A33_solver;

  Solver *_solver = nullptr;

  std::vector<std::unique_ptr<BlockVector>> _ns;
  std::vector<std::unique_ptr<BlockVector>> _clean;

  mutable BlockVector _b;
  mutable BlockVector _x;

  bool _add_constant_phi_mode;

  Vector _phi_constant;
  real_t _phi_constant_norm2 = 0.0;

  enum { U_IC_BLOCK = 0, U_MANTLE_BLOCK = 1, PHI_BLOCK = 2 };

  const Operator &B(int i, int j) const { return _A->GetBlock(i, j); }

  std::string AxisName(int c) const {
    if (c == 0) {
      return "x";
    }
    if (c == 1) {
      return "y";
    }
    if (c == 2) {
      return "z";
    }
    return "?";
  }

  void GetBlockView(const Vector &x, int block, Vector &xb) const {
    xb.SetDataAndSize(
        const_cast<real_t *>(x.GetData()) + _block_true_offsets[block],
        _block_true_offsets[block + 1] - _block_true_offsets[block]);
  }

  real_t Dot(const Vector &x, const Vector &y) const {
    return InnerProduct(_comm, x, y);
  }

  real_t Norm(const Vector &x) const {
    return std::sqrt(std::max(Dot(x, x), real_t(0.0)));
  }

  real_t EuclideanDot(const BlockVector &x, const BlockVector &y) const {
    return InnerProduct(_comm, x.GetBlock(U_IC_BLOCK), y.GetBlock(U_IC_BLOCK)) +
           InnerProduct(_comm, x.GetBlock(U_MANTLE_BLOCK),
                        y.GetBlock(U_MANTLE_BLOCK)) +
           InnerProduct(_comm, x.GetBlock(PHI_BLOCK), y.GetBlock(PHI_BLOCK));
  }

  real_t EuclideanNorm(const BlockVector &x) const {
    return std::sqrt(std::max(EuclideanDot(x, x), real_t(0.0)));
  }

  real_t EuclideanVectorModeDot(const Vector &x,
                                const BlockVector &mode) const {
    Vector x_ic, x_mantle, x_phi;

    GetBlockView(x, U_IC_BLOCK, x_ic);
    GetBlockView(x, U_MANTLE_BLOCK, x_mantle);
    GetBlockView(x, PHI_BLOCK, x_phi);

    return InnerProduct(_comm, x_ic, mode.GetBlock(U_IC_BLOCK)) +
           InnerProduct(_comm, x_mantle, mode.GetBlock(U_MANTLE_BLOCK)) +
           InnerProduct(_comm, x_phi, mode.GetBlock(PHI_BLOCK));
  }

  void BuildPhiConstant() {
    _phi_constant.SetSize(_block_true_offsets[PHI_BLOCK + 1] -
                          _block_true_offsets[PHI_BLOCK]);

    _phi_constant = 0.0;

    ParGridFunction phi_gf(_fes_phi);
    phi_gf = 1.0;
    phi_gf.GetTrueDofs(_phi_constant);

    _phi_constant_norm2 = InnerProduct(_comm, _phi_constant, _phi_constant);
  }

  void ProjectPhiConstant(Vector &v) const {
    if (!_add_constant_phi_mode) {
      return;
    }

    if (_phi_constant_norm2 <= 0.0) {
      return;
    }

    real_t alpha = InnerProduct(_comm, v, _phi_constant) / _phi_constant_norm2;
    v.Add(-alpha, _phi_constant);
  }

  void ComputePhiMode(const Vector &q_ic, const Vector &q_mantle,
                      Vector &psi) const {
    Vector rhs(B(PHI_BLOCK, PHI_BLOCK).Height());

    rhs = 0.0;

    B(PHI_BLOCK, U_IC_BLOCK).AddMult(q_ic, rhs, -1.0);
    B(PHI_BLOCK, U_MANTLE_BLOCK).AddMult(q_mantle, rhs, -1.0);

    ProjectPhiConstant(rhs);

    psi.SetSize(B(PHI_BLOCK, PHI_BLOCK).Width());
    psi = 0.0;

    _A33_solver.Mult(rhs, psi);

    ProjectPhiConstant(psi);
  }

  void PrintModeResidual(const std::string &name, const BlockVector &z) const {
    BlockVector r(_block_true_offsets);

    r = 0.0;

    _A->Mult(z, r);

    real_t z_norm = EuclideanNorm(z);
    z_norm = std::max(z_norm, real_t(1e-30));

    real_t row1_abs = Norm(r.GetBlock(U_IC_BLOCK));
    real_t row2_abs = Norm(r.GetBlock(U_MANTLE_BLOCK));
    real_t row3_abs = Norm(r.GetBlock(PHI_BLOCK));

    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank == 0) {
      std::cout << std::scientific << std::setprecision(6);
      std::cout << "Rigid mode residual [" << name << "]" << std::endl;
      std::cout << "  row 1 abs = " << row1_abs
                << ", /mode = " << row1_abs / z_norm << std::endl;
      std::cout << "  row 2 abs = " << row2_abs
                << ", /mode = " << row2_abs / z_norm << std::endl;
      std::cout << "  row 3 abs = " << row3_abs
                << ", /mode = " << row3_abs / z_norm << std::endl;
    }
  }

  void AddGlobalCoupledRigidMode(VectorCoefficient &u_coeff,
                                 const std::string &name) {
    Vector q_ic(_block_true_offsets[U_IC_BLOCK + 1] -
                _block_true_offsets[U_IC_BLOCK]);

    Vector q_mantle(_block_true_offsets[U_MANTLE_BLOCK + 1] -
                    _block_true_offsets[U_MANTLE_BLOCK]);

    Vector psi(_block_true_offsets[PHI_BLOCK + 1] -
               _block_true_offsets[PHI_BLOCK]);

    q_ic = 0.0;
    q_mantle = 0.0;
    psi = 0.0;

    ParGridFunction u_ic_gf(_fes_u_ic);
    ParGridFunction u_mantle_gf(_fes_u_mantle);

    u_ic_gf = 0.0;
    u_mantle_gf = 0.0;

    u_ic_gf.ProjectCoefficient(u_coeff);
    u_mantle_gf.ProjectCoefficient(u_coeff);

    u_ic_gf.GetTrueDofs(q_ic);
    u_mantle_gf.GetTrueDofs(q_mantle);

    ComputePhiMode(q_ic, q_mantle, psi);

    auto nv = std::make_unique<BlockVector>(_block_true_offsets);
    auto cv = std::make_unique<BlockVector>(_block_true_offsets);

    *nv = 0.0;

    nv->GetBlock(U_IC_BLOCK) = q_ic;
    nv->GetBlock(U_MANTLE_BLOCK) = q_mantle;
    nv->GetBlock(PHI_BLOCK) = psi;

    PrintModeResidual(name, *nv);

    *cv = *nv;

    _ns.push_back(std::move(nv));
    _clean.push_back(std::move(cv));
  }

  void AddGlobalRigidModes(int dim) {
    for (int c = 0; c < dim; c++) {
      RigidTranslation u_coeff(dim, c);

      AddGlobalCoupledRigidMode(u_coeff, "global T" + AxisName(c));
    }

    if (dim == 2) {
      RigidRotation u_coeff(dim, 2);

      AddGlobalCoupledRigidMode(u_coeff, "global Rz");
    } else {
      for (int c = 0; c < dim; c++) {
        RigidRotation u_coeff(dim, c);

        AddGlobalCoupledRigidMode(u_coeff, "global R" + AxisName(c));
      }
    }
  }

  void AddPurePhiConstantMode() {
    auto nv = std::make_unique<BlockVector>(_block_true_offsets);
    auto cv = std::make_unique<BlockVector>(_block_true_offsets);

    *nv = 0.0;

    nv->GetBlock(PHI_BLOCK) = _phi_constant;

    *cv = *nv;

    _ns.push_back(std::move(nv));
    _clean.push_back(std::move(cv));
  }

  void GramSchmidt(std::vector<std::unique_ptr<BlockVector>> &modes) {
    std::vector<std::unique_ptr<BlockVector>> modes_orth;

    for (int i = 0; i < (int)modes.size(); i++) {
      auto ni = std::make_unique<BlockVector>(_block_true_offsets);

      *ni = *modes[i];

      for (int j = 0; j < (int)modes_orth.size(); j++) {
        BlockVector &nj = *modes_orth[j];

        real_t product = EuclideanDot(*ni, nj);

        ni->Add(-product, nj);
      }

      real_t norm = EuclideanNorm(*ni);

      if (norm <= 1e-24) {
        continue;
      }

      *ni /= norm;

      modes_orth.push_back(std::move(ni));
    }

    modes.swap(modes_orth);
  }

  void ProjectOrthogonalToModesEuclidean(
      const Vector &x, Vector &y,
      const std::vector<std::unique_ptr<BlockVector>> &modes) const {
    y = x;

    for (int i = 0; i < (int)modes.size(); i++) {
      const BlockVector &ni = *modes[i];

      real_t product = EuclideanVectorModeDot(y, ni);

      y.Add(-product, ni);
    }
  }

 public:
  ThreeBlockRigidBodySolverParallel(MPI_Comm comm,
                                    ParFiniteElementSpace *fes_u_ic,
                                    ParFiniteElementSpace *fes_u_mantle,
                                    ParFiniteElementSpace *fes_phi,
                                    Array<int> *block_true_offsets,
                                    const BlockOperator &A, Solver &prec33,
                                    bool add_constant_phi_mode)
      : Solver(A.Width(), false),
        _comm(comm),
        _fes_u_ic(fes_u_ic),
        _fes_u_mantle(fes_u_mantle),
        _fes_phi(fes_phi),
        _block_true_offsets(*block_true_offsets),
        _A(&A),
        _A33_prec(&prec33),
        _A33_solver(comm),
        _b(_block_true_offsets),
        _x(_block_true_offsets),
        _add_constant_phi_mode(add_constant_phi_mode) {
    int dim = _fes_u_ic->GetVDim();

    MFEM_ASSERT(dim == 2 || dim == 3, "dimension must be 2 or 3");

    MFEM_ASSERT(_fes_u_mantle->GetVDim() == dim,
                "solid displacement spaces must have the same vdim");

    MFEM_VERIFY(
        A.Height() == A.Width(),
        "ThreeBlockRigidBodySolverParallel requires a square block operator");

    MFEM_VERIFY(_block_true_offsets.Size() == 4,
                "ThreeBlockRigidBodySolverParallel expects three blocks");

    MFEM_VERIFY(_block_true_offsets[1] - _block_true_offsets[0] ==
                    _fes_u_ic->TrueVSize(),
                "inner-core block size mismatch");

    MFEM_VERIFY(_block_true_offsets[2] - _block_true_offsets[1] ==
                    _fes_u_mantle->TrueVSize(),
                "mantle block size mismatch");

    MFEM_VERIFY(_block_true_offsets[3] - _block_true_offsets[2] ==
                    _fes_phi->TrueVSize(),
                "phi block size mismatch");

    height = A.Height();
    width = A.Width();

    BuildPhiConstant();

    _A33_solver.SetRelTol(1e-10);
    _A33_solver.SetAbsTol(0.0);
    _A33_solver.SetMaxIter(5000);
    _A33_solver.SetPrintLevel(0);
    _A33_solver.SetOperator(B(PHI_BLOCK, PHI_BLOCK));
    _A33_solver.SetPreconditioner(*_A33_prec);
    _A33_solver.iterative_mode = false;

    AddGlobalRigidModes(dim);

    if (_add_constant_phi_mode) {
      AddPurePhiConstantMode();
    }

    GramSchmidt(_ns);
    GramSchmidt(_clean);

    MFEM_VERIFY(_ns.size() > 0,
                "empty nullspace in ThreeBlockRigidBodySolverParallel");

    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank == 0) {
      std::cout << "Parallel block SLS1 nullspace dimension = " << _ns.size()
                << std::endl;

      std::cout << "Parallel block SLS1 Euclidean cleaning dimension = "
                << _clean.size() << std::endl;
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

    ProjectOrthogonalToModesEuclidean(b, _b, _ns);

    _x = 0.0;

    _solver->iterative_mode = iterative_mode;
    _solver->Mult(_b, _x);

    ProjectOrthogonalToModesEuclidean(_x, x, _clean);
  }
};

void SendParSolutionToGLVis(MPI_Comm world, int myid, int dim,
                            mfem::ParMesh &pm, mfem::ParGridFunction &gf,
                            const char *title) {
  const int active = (pm.GetNE() > 0) ? 1 : 0;

  int active_count = 0;
  MPI_Allreduce(&active, &active_count, 1, MPI_INT, MPI_SUM, world);

  if (active_count == 0) {
    if (myid == 0) {
      std::cout << "Skipping GLVis output for empty mesh: " << title
                << std::endl;
    }
    return;
  }

  MPI_Comm vis_comm;
  MPI_Comm_split(world, active ? 0 : MPI_UNDEFINED, myid, &vis_comm);

  if (!active) {
    MPI_Barrier(world);
    return;
  }

  int vis_rank = 0;
  int vis_size = 1;

  MPI_Comm_rank(vis_comm, &vis_rank);
  MPI_Comm_size(vis_comm, &vis_size);

  char vishost[] = "localhost";
  int visport = 19916;

  mfem::socketstream sock(vishost, visport);

  sock << "parallel " << vis_size << " " << vis_rank << "\n";

  sock.precision(8);

  sock << "solution\n"
       << pm << gf << "window_title '" << title << "'" << std::endl;

  if (dim == 2) {
    sock << "keys Rjlbc\n" << std::flush;
  } else {
    sock << "keys RRRilc\n" << std::flush;
  }

  MPI_Comm_free(&vis_comm);

  MPI_Barrier(world);
}
