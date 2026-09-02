#include "ElasticTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for ViscoelasticOperator (design doc doc/viscoelastic_design.md,
  section 5, tests 3-7).

  - Strain maps: Galerkin and interpolation maps both reproduce the exact
    deviatoric strain of a polynomial displacement at the internal nodes.
  - Creep: a Maxwell body under constant uniaxial stress has a deviatoric
    strain growing linearly at rate dev(sigma)/(2 eta); the exponential
    trapezoid step is exact for it at any dt, RK4 to high accuracy.
  - Relaxation: with a prescribed uniform strain, every branch relaxes as
    m_k = d (1 - exp(-t / tau_k)); exponential Euler and trapezoid are exact
    at any dt (two branches with tau ratio 100, dt >> tau_1), backward Euler
    gives its known discrete relaxation.
  - Temporal convergence orders against an RK4 reference for a
    time-varying load: ETD1 -> 1, BE -> 1, exponential trapezoid -> 2,
    SDIRK23 (L-stable variant) -> 2.
  - Long-time limit of a clamped generalised Maxwell body equals the elastic
    solution with mu = mu_inf.
  - A two-field mock problem: the block state layout and force routing
    reproduce two independent single-field operators.
*/

namespace {

using namespace elastic_test;

using Param = std::tuple<int, int, int>;  // (dim, elementType, order)

// Trace-free component index: lower triangle, column-major, last diagonal
// entry dropped (TraceFreeSymmetricMatrixIndex).
int TFIndex(int dim, int j, int k) {
  if (j < k) {
    std::swap(j, k);
  }
  return j + k * dim - k * (k + 1) / 2;
}

void ConstantUniaxial(const Vector& x, Vector& f) {
  UniaxialTraction(x, 0.0, f);
}

void ConstantPull(const Vector& x, Vector& f) { PullTraction(x, 0.0, f); }

double RelMaxDiff(const Vector& a, const Vector& b) {
  Vector d(a);
  d -= b;
  return d.Normlinf() / (b.Normlinf() + 1e-300);
}

// dev(sym(A)) components in the trace-free layout.
Vector DeviatoricPart(const DenseMatrix& A) {
  const int dim = A.Height();
  DenseMatrix S(dim);
  double tr = 0.0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      S(i, j) = 0.5 * (A(i, j) + A(j, i));
    }
    tr += A(i, i);
  }
  Vector d(dim * (dim + 1) / 2 - 1);
  for (int k = 0; k < dim; k++) {
    for (int j = k; j < dim; j++) {
      const int idx = TFIndex(dim, j, k);
      if (idx < d.Size()) {
        d[idx] = S(j, k) - (j == k ? tr / dim : 0.0);
      }
    }
  }
  return d;
}

class ViscoelasticTest : public testing::TestWithParam<Param> {
 protected:
  void SetUp() override {
    std::tie(dim, elementType, order) = GetParam();
    mesh = std::make_unique<Mesh>(MakeSmallMesh(dim, elementType));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes = std::make_unique<FiniteElementSpace>(mesh.get(), fec.get(), dim);
    kappa = std::make_unique<ConstantCoefficient>(Kappa(dim));
    mu = std::make_unique<ConstantCoefficient>(kMu);
    tau = std::make_unique<ConstantCoefficient>(kTau);
    x0_attr = BdrAttributeAt(*mesh, 0, 0.0);
    x1_attr = BdrAttributeAt(*mesh, 0, 1.0);
    nbdr = mesh->bdr_attributes.Max();
  }

  static constexpr double kTau = 1.0;

  int dim = 2, elementType = 0, order = 1, x0_attr = -1, x1_attr = -1, nbdr = 0;
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<FiniteElementCollection> fec;
  std::unique_ptr<FiniteElementSpace> fes;
  std::unique_ptr<ConstantCoefficient> kappa, mu, tau;
};

TEST_P(ViscoelasticTest, StrainMaps) {
  auto rheology = GeneralisedMaxwellRheology::Maxwell(dim, *kappa, *mu, *tau);
  auto marker = Marker(nbdr, {x0_attr, x1_attr});
  VectorFunctionCoefficient traction(dim, ConstantUniaxial);
  TractionProblem problem(fes.get(), rheology, traction, marker);

  // u_i = sum_j a_ij x_j^p (degree p, represented exactly).
  const int p = order;
  DenseMatrix a(dim);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      a(i, j) = 0.3 + 0.7 * i - 0.4 * j + 0.2 * i * j;
    }
  }
  VectorFunctionCoefficient u_coef(dim, [&](const Vector& x, Vector& u) {
    for (int i = 0; i < dim; i++) {
      u[i] = 0.0;
      for (int j = 0; j < dim; j++) {
        u[i] += a(i, j) * std::pow(x[j], p);
      }
    }
  });
  const int nc = dim * (dim + 1) / 2 - 1;
  VectorFunctionCoefficient d_coef(nc, [&](const Vector& x, Vector& d) {
    DenseMatrix G(dim);  // G(i, j) = d u_i / d x_j
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        G(i, j) = a(i, j) * p * std::pow(x[j], p - 1);
      }
    }
    d = DeviatoricPart(G);
  });
  GridFunction u(fes.get());
  u.ProjectCoefficient(u_coef);

  for (auto map : {ViscoelasticOperator::StrainMap::Galerkin,
                   ViscoelasticOperator::StrainMap::Interpolation}) {
    ViscoelasticOperator visco(problem, -1, map);
    EXPECT_EQ(visco.NumFields(), 1);
    EXPECT_EQ(visco.NumBranches(0), 1);
    EXPECT_EQ(visco.Height(), visco.BranchSize(0));
    GridFunction d_exact(&visco.InternalVariableSpace(0));
    d_exact.ProjectCoefficient(d_coef);
    Vector d;
    visco.ComputeStrain(0, u, d);
    ASSERT_EQ(d.Size(), d_exact.Size());
    EXPECT_LT(RelMaxDiff(d, d_exact), 1e-10);
  }
}

TEST_P(ViscoelasticTest, CreepUnderConstantStress) {
  auto rheology = GeneralisedMaxwellRheology::Maxwell(dim, *kappa, *mu, *tau);
  auto marker = Marker(nbdr, {x0_attr, x1_attr});
  VectorFunctionCoefficient traction(dim, ConstantUniaxial);
  TractionProblem problem(fes.get(), rheology, traction, marker);
  ViscoelasticOperator visco(problem);

  double exx0 = 0.0, eyy0 = 0.0;
  UniaxialStrain(dim, kSigma, exx0, eyy0);
  const double eta = kMu * kTau;
  // dev(sigma) = sigma diag(1 - 1/d, -1/d, ...); d(t) = d_0 + t dev(sigma)/(2
  // eta).
  auto exx_at = [&](double t) {
    return exx0 + kSigma * (1.0 - 1.0 / dim) * t / (2.0 * eta);
  };
  auto eyy_at = [&](double t) { return eyy0 - kSigma / dim * t / (2.0 * eta); };

  // Exponential trapezoid: exact for a strain linear in time, at dt >> tau.
  {
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0, dt = 2.0;
    for (int step = 0; step < 4; step++) {
      ode.Step(m, t, dt);
    }
    EXPECT_NEAR(t, 8.0, 1e-14);
    ASSERT_TRUE(visco.SolveElastic(m, t));
    EXPECT_LT(MaxStrainError(problem.Displacement(), exx_at(t), eyy_at(t)),
              1e-8 * exx0);
  }

  // RK4 at dt = 0.1 tau: high accuracy, and the same limit.
  {
    RK4Solver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0, dt = 0.1;
    for (int step = 0; step < 10; step++) {
      ode.Step(m, t, dt);
    }
    ASSERT_TRUE(visco.SolveElastic(m, t));
    EXPECT_LT(MaxStrainError(problem.Displacement(), exx_at(t), eyy_at(t)),
              1e-5 * exx0);
  }
}

TEST_P(ViscoelasticTest, RelaxationUnderPrescribedStrain) {
  ConstantCoefficient mu_inf(0.3), mu1(0.7), mu2(0.5), tau1(1.0), tau2(100.0);
  std::vector<MaxwellBranch> branches{{&mu1, &tau1}, {&mu2, &tau2}};
  GeneralisedMaxwellRheology rheology(dim, *kappa, mu_inf, branches);

  // Prescribed displacement u = A x on the whole boundary; no traction.
  DenseMatrix A(dim);
  A = 0.0;
  A(0, 0) = 0.01;
  A(0, 1) = 0.004;
  A(1, 0) = -0.002;
  A(1, 1) = -0.006;
  if (dim == 3) {
    A(2, 2) = 0.003;
    A(0, 2) = 0.005;
  }
  VectorFunctionCoefficient dirichlet(
      dim, [&](const Vector& x, Vector& u) { A.Mult(x, u); });
  Vector zero(dim);
  zero = 0.0;
  VectorConstantCoefficient no_traction(zero);
  Array<int> all(nbdr), none(nbdr);
  all = 1;
  none = 0;
  ClampedProblem problem(fes.get(), rheology, all, no_traction, none,
                         &dirichlet);
  ViscoelasticOperator visco(problem);
  ASSERT_EQ(visco.NumBranches(0), 2);
  EXPECT_NEAR(visco.MinRelaxationTime(), 1.0, 1e-14);

  const Vector d0 = DeviatoricPart(A);
  const int nd = visco.InternalScalarSpace(0).GetVSize();
  const double taus[2] = {1.0, 100.0};

  // m_k(t) at every node against the analytic value.
  const double d0_max = d0.Normlinf();
  auto check = [&](const Vector& m, double t, double tol, auto branch_factor) {
    for (int k = 0; k < 2; k++) {
      Vector mk = visco.Branch(m, 0, k);
      const double factor = branch_factor(k, t);
      for (int c = 0; c < d0.Size(); c++) {
        for (int p = 0; p < nd; p++) {
          EXPECT_NEAR(mk[c * nd + p], d0[c] * factor, tol * d0_max)
              << "branch " << k << " component " << c;
        }
      }
    }
  };
  auto exact = [&](int k, double t) { return 1.0 - std::exp(-t / taus[k]); };

  // Exponential Euler and trapezoid: exact at dt = 5 >> tau_1.
  for (int scheme = 0; scheme < 2; scheme++) {
    std::unique_ptr<ODESolver> ode;
    if (scheme == 0) {
      ode = std::make_unique<ExponentialEulerSolver>();
    } else {
      ode = std::make_unique<ExponentialTrapezoidSolver>();
    }
    ode->Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0, dt = 5.0;
    for (int step = 0; step < 6; step++) {
      ode->Step(m, t, dt);
    }
    check(m, t, 1e-9, exact);
  }

  // Backward Euler: m_n = d (1 - (1 + h)^{-n}) exactly.
  {
    BackwardEulerSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0, dt = 0.5;
    const int n = 4;
    for (int step = 0; step < n; step++) {
      ode.Step(m, t, dt);
    }
    check(m, t, 1e-9, [&](int k, double) {
      return 1.0 - std::pow(1.0 + dt / taus[k], -n);
    });
  }

  // Explicit RK4 at a stable step converges to the same curve.
  {
    RK4Solver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0, dt = 0.1;
    for (int step = 0; step < 10; step++) {
      ode.Step(m, t, dt);
    }
    check(m, t, 1e-6, exact);
  }
}

TEST_P(ViscoelasticTest, LongTimeLimit) {
  ConstantCoefficient mu_inf(0.5), mu1(1.0), tau1(1.0);
  std::vector<MaxwellBranch> branches{{&mu1, &tau1}};
  GeneralisedMaxwellRheology rheology(dim, *kappa, mu_inf, branches);
  auto ess_bdr = Marker(nbdr, {x0_attr});
  auto pull_marker = Marker(nbdr, {x1_attr});
  VectorFunctionCoefficient pull(dim, ConstantPull);

  ClampedProblem problem(fes.get(), rheology, ess_bdr, pull, pull_marker);
  ViscoelasticOperator visco(problem);
  ExponentialTrapezoidSolver ode;
  ode.Init(visco);
  Vector m(visco.Height());
  m = 0.0;
  double t = 0.0, dt = 5.0;
  for (int step = 0; step < 20; step++) {
    ode.Step(m, t, dt);
  }
  ASSERT_TRUE(visco.SolveElastic(m, t));

  auto relaxed = GeneralisedMaxwellRheology::Elastic(dim, *kappa, mu_inf);
  ClampedProblem limit(fes.get(), relaxed, ess_bdr, pull, pull_marker);
  limit.AssembleForce(t);
  ASSERT_TRUE(limit.Solve());
  EXPECT_LT(RelMaxDiff(problem.Displacement(), limit.Displacement()), 1e-6);

  // And the unrelaxed (t = 0) response differs from it.
  auto unrelaxed = GeneralisedMaxwellRheology::Elastic(
      dim, *kappa, rheology.UnrelaxedShearModulus());
  ClampedProblem initial(fes.get(), unrelaxed, ess_bdr, pull, pull_marker);
  initial.AssembleForce(0.0);
  ASSERT_TRUE(initial.Solve());
  EXPECT_GT(RelMaxDiff(initial.Displacement(), limit.Displacement()), 0.1);
}

INSTANTIATE_TEST_SUITE_P(Viscoelastic, ViscoelasticTest,
                         testing::Combine(testing::Values(2, 3),
                                          testing::Values(0, 1),
                                          testing::Values(1, 2)));

// ---------------------------------------------------------------------------

TEST(ViscoelasticConvergence, TemporalOrders) {
  const int dim = 2;
  Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
  H1_FECollection fec(1, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);
  ConstantCoefficient kappa(Kappa(dim)), mu(kMu), tau(1.0);
  auto rheology = GeneralisedMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  const int x0 = BdrAttributeAt(mesh, 0, 0.0),
            x1 = BdrAttributeAt(mesh, 0, 1.0);
  auto marker = Marker(mesh.bdr_attributes.Max(), {x0, x1});
  VectorFunctionCoefficient traction(
      dim, [](const Vector& x, real_t t, Vector& f) {
        f = 0.0;
        const double s = kSigma * (1.0 + 0.5 * std::sin(2.0 * t));
        if (x[0] > 1.0 - 1e-8) {
          f[0] = s;
        } else if (x[0] < 1e-8) {
          f[0] = -s;
        }
      });
  TractionProblem problem(&fes, rheology, traction, marker);
  ViscoelasticOperator visco(problem);
  const double t_final = 1.0;

  auto integrate = [&](ODESolver& ode, double dt) {
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0;
    const int n = static_cast<int>(std::round(t_final / dt));
    for (int step = 0; step < n; step++) {
      ode.Step(m, t, dt);
    }
    return m;
  };

  RK4Solver rk4;
  const Vector ref = integrate(rk4, 1.0 / 1024.0);
  ASSERT_GT(ref.Normlinf(), 0.0);

  struct Case {
    std::string name;
    std::unique_ptr<ODESolver> ode;
    double expected_order;
  };
  std::vector<Case> cases;
  cases.push_back({"ETD1", std::make_unique<ExponentialEulerSolver>(), 1.0});
  cases.push_back({"BE", std::make_unique<BackwardEulerSolver>(), 1.0});
  cases.push_back(
      {"ExpTrap", std::make_unique<ExponentialTrapezoidSolver>(), 2.0});
  // MFEM's default SDIRK23 variant is third order; use the second-order
  // L-stable one (gamma_opt = 2).
  cases.push_back({"SDIRK23", std::make_unique<SDIRK23Solver>(2), 2.0});

  for (auto& c : cases) {
    const double e1 = RelMaxDiff(integrate(*c.ode, 1.0 / 16.0), ref);
    const double e2 = RelMaxDiff(integrate(*c.ode, 1.0 / 32.0), ref);
    const double rate = std::log2(e1 / e2);
    EXPECT_NEAR(rate, c.expected_order, 0.25)
        << c.name << ": e(1/16) = " << e1 << ", e(1/32) = " << e2;
  }
}

// ---------------------------------------------------------------------------

// A two-field problem delegating to two independent single-field problems.
class TwoFieldProblem : public QuasiStaticLinearElasticProblem {
 public:
  TwoFieldProblem(ElasticProblemBase& a, ElasticProblemBase& b) : p_{&a, &b} {}
  int NumDisplacementFields() const override { return 2; }
  FiniteElementSpace& DisplacementSpace(int i) override {
    return p_[i]->DisplacementSpace();
  }
  const GridFunction& Displacement(int i) const override {
    return p_[i]->Displacement();
  }
  const GeneralisedMaxwellRheology& Rheology(int i) const override {
    return p_[i]->Rheology();
  }
  void AssembleForce(real_t t) override {
    for (auto* p : p_) {
      p->AssembleForce(t);
    }
  }
  void AddForce(int i, const Vector& f) override { p_[i]->AddForce(0, f); }
  using QuasiStaticLinearElasticProblem::AddForce;
  bool Solve() override { return p_[0]->Solve() && p_[1]->Solve(); }
  bool SupportsEffectiveShearModulus() const override { return true; }
  void SetEffectiveShearModulus(int i, Coefficient& c) override {
    p_[i]->SetEffectiveShearModulus(0, c);
  }
  void ClearEffectiveShearModulus() override {
    for (auto* p : p_) {
      p->ClearEffectiveShearModulus();
    }
  }
  void RegisterFields(DataCollection&) override {}

 private:
  ElasticProblemBase* p_[2];
};

TEST(ViscoelasticTwoField, RoutesForcesPerField) {
  const int dim = 2;
  Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
  H1_FECollection fec1(1, dim), fec2(2, dim);
  FiniteElementSpace fes1(&mesh, &fec1, dim), fes2(&mesh, &fec2, dim);
  ConstantCoefficient kappa(Kappa(dim)), mu(kMu), tau(1.0), mu_inf(0.4),
      tau2(3.0);
  auto rh1 = GeneralisedMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  std::vector<MaxwellBranch> branches{{&mu, &tau}, {&mu, &tau2}};
  GeneralisedMaxwellRheology rh2(dim, kappa, mu_inf, branches);
  const int x0 = BdrAttributeAt(mesh, 0, 0.0),
            x1 = BdrAttributeAt(mesh, 0, 1.0);
  auto ess = Marker(mesh.bdr_attributes.Max(), {x0});
  auto pull_marker = Marker(mesh.bdr_attributes.Max(), {x1});
  VectorFunctionCoefficient pull(dim, PullTraction);

  ClampedProblem a1(&fes1, rh1, ess, pull, pull_marker);
  ClampedProblem b1(&fes2, rh2, ess, pull, pull_marker);
  TwoFieldProblem two(a1, b1);
  ViscoelasticOperator visco_two(two);
  ASSERT_EQ(visco_two.NumFields(), 2);
  ASSERT_EQ(visco_two.NumBranches(0), 1);
  ASSERT_EQ(visco_two.NumBranches(1), 2);
  ASSERT_EQ(visco_two.Offsets().Size(), 4);
  EXPECT_EQ(visco_two.BranchOffset(1, 1),
            visco_two.BranchSize(0) + visco_two.BranchSize(1));
  EXPECT_EQ(visco_two.Height(),
            visco_two.BranchSize(0) + 2 * visco_two.BranchSize(1));

  ClampedProblem a2(&fes1, rh1, ess, pull, pull_marker);
  ClampedProblem b2(&fes2, rh2, ess, pull, pull_marker);
  ViscoelasticOperator visco_a(a2), visco_b(b2);

  for (int scheme = 0; scheme < 2; scheme++) {
    auto make = [scheme]() -> std::unique_ptr<ODESolver> {
      if (scheme == 0) {
        return std::make_unique<ExponentialTrapezoidSolver>();
      }
      return std::make_unique<RK4Solver>();
    };
    auto ode_two = make(), ode_a = make(), ode_b = make();
    ode_two->Init(visco_two);
    ode_a->Init(visco_a);
    ode_b->Init(visco_b);
    Vector m_two(visco_two.Height()), m_a(visco_a.Height()),
        m_b(visco_b.Height());
    m_two = 0.0;
    m_a = 0.0;
    m_b = 0.0;
    double t_two = 0.0, t_a = 0.0, t_b = 0.0, dt = 0.4;
    for (int step = 0; step < 3; step++) {
      ode_two->Step(m_two, t_two, dt);
      ode_a->Step(m_a, t_a, dt);
      ode_b->Step(m_b, t_b, dt);
    }
    EXPECT_LT(RelMaxDiff(visco_two.Branch(m_two, 0, 0), m_a), 1e-12);
    for (int k = 0; k < 2; k++) {
      EXPECT_LT(
          RelMaxDiff(visco_two.Branch(m_two, 1, k), visco_b.Branch(m_b, 0, k)),
          1e-12);
    }
    ASSERT_GT(m_a.Normlinf(), 0.0);
    ASSERT_GT(m_b.Normlinf(), 0.0);
  }
}

}  // namespace
