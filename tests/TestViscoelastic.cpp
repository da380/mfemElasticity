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
  - Anisotropic rheology (full symmetric internal variables, tensor branch
    moduli) with an isotropic tensor reproduces the isotropic operator to
    round-off for every scheme, displacement and internal variables alike.
  - A transversely isotropic Maxwell bar under constant uniaxial stress:
    homogeneous state, so the FE solution follows the 6 x 6 (3 x 3) ODE of
    the branch variable, integrated to high accuracy as the reference.
  - State-dependent relaxation times (doc/nonlinear_viscoelastic_design.md):
    a power law with gamma = 0 reproduces the linear body; under constant
    uniaxial stress the deviatoric stress, hence tau, is constant, so the
    creep is the linear one with tau(|T|) (exact for the trapezoid; rate
    ratio 2^n between two stress levels); under a prescribed strain tau
    grows as the stress relaxes, and the trapezoid with one corrector is
    second order, ETD1 and backward Euler first order, against a nodal ODE
    reference; the adaptive solver meets its tolerance with fewer steps.
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
  auto rheology = IsotropicMaxwellRheology::Maxwell(dim, *kappa, *mu, *tau);
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
  auto rheology = IsotropicMaxwellRheology::Maxwell(dim, *kappa, *mu, *tau);
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
  IsotropicMaxwellRheology rheology(dim, *kappa, mu_inf, branches);

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
  IsotropicMaxwellRheology rheology(dim, *kappa, mu_inf, branches);
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

  auto relaxed = rheology.LongTermElastic();
  ClampedProblem limit(fes.get(), relaxed, ess_bdr, pull, pull_marker);
  limit.AssembleForce(t);
  ASSERT_TRUE(limit.Solve());
  EXPECT_LT(RelMaxDiff(problem.Displacement(), limit.Displacement()), 1e-6);

  // And the unrelaxed (t = 0) response differs from it.
  auto unrelaxed = rheology.UnrelaxedElastic();
  ClampedProblem initial(fes.get(), unrelaxed, ess_bdr, pull, pull_marker);
  initial.AssembleForce(0.0);
  ASSERT_TRUE(initial.Solve());
  EXPECT_GT(RelMaxDiff(initial.Displacement(), limit.Displacement()), 0.1);
}

TEST_P(ViscoelasticTest, AnisotropicMatchesIsotropic) {
  auto marker = Marker(nbdr, {x0_attr, x1_attr});
  VectorFunctionCoefficient traction(dim, ConstantUniaxial);
  ConstantCoefficient mu_inf(0.3 * kMu), mu1(0.7 * kMu), tau1(1.0);
  std::vector<MaxwellBranch> branches{{&mu1, &tau1}};
  IsotropicMaxwellRheology iso(dim, *kappa, mu_inf, branches);
  // The same body as tensors: C_inf = kappa vol + 2 mu_inf dev, C_1 = 2 mu_1
  // dev.
  auto C_inf = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, *kappa,
                                                                  mu_inf);
  ConstantCoefficient zero(0.0);
  auto C_1 = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, zero, mu1);
  std::vector<AnisotropicBranch> abranches{{&C_1, &tau1}};
  AnisotropicMaxwellRheology aniso(dim, C_inf, abranches);

  TractionProblem pi(fes.get(), iso, traction, marker);
  TractionProblem pa(fes.get(), aniso, traction, marker);

  for (int scheme = 0; scheme < 4; scheme++) {
    ViscoelasticOperator vi(pi), va(pa);
    ASSERT_TRUE(vi.TraceFree(0));
    ASSERT_FALSE(va.TraceFree(0));
    ASSERT_EQ(va.NumComponents(0), vi.NumComponents(0) + 1);
    auto make = [scheme]() -> std::unique_ptr<ODESolver> {
      switch (scheme) {
        case 0:
          return std::make_unique<ExponentialTrapezoidSolver>();
        case 1:
          return std::make_unique<ExponentialEulerSolver>();
        case 2:
          return std::make_unique<BackwardEulerSolver>();
        default:
          return std::make_unique<RK4Solver>();
      }
    };
    auto oi = make(), oa = make();
    oi->Init(vi);
    oa->Init(va);
    Vector mi(vi.Height()), ma(va.Height());
    mi = 0.0;
    ma = 0.0;
    double ti = 0.0, ta = 0.0;
    double dt = scheme == 3 ? 0.1 : 0.5;
    for (int step = 0; step < 4; step++) {
      oi->Step(mi, ti, dt);
      oa->Step(ma, ta, dt);
    }
    ASSERT_TRUE(vi.SolveElastic(mi, ti));
    ASSERT_TRUE(va.SolveElastic(ma, ta));
    EXPECT_GT(pi.Displacement().Normlinf(), 0.0);
    EXPECT_LT(RelMaxDiff(pa.Displacement(), pi.Displacement()), 1e-10)
        << "scheme " << scheme;

    // The full internal variable's deviatoric part is the trace-free one.
    const int nd = vi.InternalScalarSpace(0).GetVSize();
    const int nc = vi.NumComponents(0);
    Vector mtf = vi.Branch(mi, 0, 0), mfull = va.Branch(ma, 0, 0);
    double err = 0.0;
    for (int p = 0; p < nd; p++) {
      double tr = 0.0;
      for (int j = 0; j < dim; j++) {
        tr += mfull[TFIndex(dim, j, j) * nd + p];
      }
      for (int c = 0; c < nc; c++) {
        int j, k;
        SymmetricTensorBasis::Component(dim, c, j, k);
        const double dev = mfull[c * nd + p] - (j == k ? tr / dim : 0.0);
        err = std::max(err, std::abs(dev - mtf[c * nd + p]));
      }
    }
    EXPECT_LT(err, 1e-10 * mtf.Normlinf()) << "scheme " << scheme;
  }
}

TEST_P(ViscoelasticTest, TransverselyIsotropicHomogeneousCreep) {
  // A transversely isotropic Maxwell bar (deviatoric part relaxing) under
  // constant uniaxial stress: the state is homogeneous and exactly
  // representable, so the FE strain follows the local ODE
  //   eps = C_U^{-1} (sigma_0 + C_1 m),  m' = (eps - m) / tau,
  // in Mandel form, integrated here by RK4 at a tiny step.
  auto marker = Marker(nbdr, {x0_attr, x1_attr});
  VectorFunctionCoefficient traction(dim, ConstantUniaxial);
  const int ns = SymmetricTensorBasis::Size(dim);
  ConstantCoefficient A(3.2), C(2.8), F(1.1), L(0.9), N(1.05), tau(1.0);
  Vector axis(dim);
  axis = 0.0;
  axis[0] = 1.0;
  axis[1] = 0.5;
  if (dim == 3) {
    axis[2] = 0.3;
  }
  axis /= axis.Norml2();
  VectorConstantCoefficient axis_coef(axis);
  TransverselyIsotropicElasticTensorCoefficient C_ti(dim, A, C, F, L, N,
                                                     axis_coef);
  auto rheology = AnisotropicMaxwellRheology::DeviatoricMaxwell(dim, C_ti, tau);
  TractionProblem problem(fes.get(), rheology, traction, marker);
  ViscoelasticOperator visco(problem);

  // The Mandel matrices and the ODE reference.
  DenseMatrix Cm(ns), P(ns), C1(ns), Cinv(ns), tmp(ns);
  TransverselyIsotropicElasticTensorCoefficient::Build(
      dim, 3.2, 2.8, 1.1, 0.9, 1.05, axis, Cm);
  SymmetricTensorBasis::DeviatoricProjector(dim, P);
  Mult(P, Cm, tmp);
  Mult(tmp, P, C1);
  DenseMatrixInverse inv(Cm);
  inv.GetInverseMatrix(Cinv);
  Vector sigma0(ns);
  sigma0 = 0.0;
  sigma0[SymmetricTensorBasis::Index(dim, 0, 0)] = kSigma;
  auto eps_of = [&](const Vector& m, Vector& eps) {
    Vector rhs(ns);
    C1.Mult(m, rhs);
    rhs += sigma0;
    Cinv.Mult(rhs, eps);
  };
  auto rate = [&](const Vector& m, Vector& k) {
    eps_of(m, k);
    k -= m;
  };
  const double t_final = 1.0;
  Vector m_ref(ns), k1(ns), k2(ns), k3(ns), k4(ns), tmpv(ns);
  m_ref = 0.0;
  {
    const int n = 20000;
    const double h = t_final / n;
    for (int i = 0; i < n; i++) {
      rate(m_ref, k1);
      add(m_ref, 0.5 * h, k1, tmpv);
      rate(tmpv, k2);
      add(m_ref, 0.5 * h, k2, tmpv);
      rate(tmpv, k3);
      add(m_ref, h, k3, tmpv);
      rate(tmpv, k4);
      for (int s = 0; s < ns; s++) {
        m_ref[s] += h / 6.0 * (k1[s] + 2.0 * k2[s] + 2.0 * k3[s] + k4[s]);
      }
    }
  }
  Vector eps_ref(ns);
  eps_of(m_ref, eps_ref);
  // Mandel -> tensor components.
  Vector eps_comp(ns);
  for (int s = 0; s < ns; s++) {
    int j, k;
    SymmetricTensorBasis::Component(dim, s, j, k);
    eps_comp[s] = eps_ref[s] / SymmetricTensorBasis::Scale(j, k);
  }
  ASSERT_GT(eps_comp.Normlinf(), 0.0);

  // Max over element centres of |eps_h - eps_ref|.
  auto strain_error = [&](const GridFunction& u) {
    double err = 0.0;
    DenseMatrix g(dim);
    for (int e = 0; e < fes->GetNE(); e++) {
      auto* T = fes->GetElementTransformation(e);
      T->SetIntPoint(&Geometries.GetCenter(T->GetGeometryType()));
      u.GetVectorGradient(*T, g);
      for (int s = 0; s < ns; s++) {
        int j, k;
        SymmetricTensorBasis::Component(dim, s, j, k);
        err = std::max(err, std::abs(0.5 * (g(j, k) + g(k, j)) - eps_comp[s]));
      }
    }
    return err / eps_comp.Normlinf();
  };

  // RK4 at dt = 0.05: fourth order, and the trapezoid at dt = 0.02.
  {
    RK4Solver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0, dt = 0.05;
    for (int step = 0; step < 20; step++) {
      ode.Step(m, t, dt);
    }
    ASSERT_TRUE(visco.SolveElastic(m, t));
    EXPECT_LT(strain_error(problem.Displacement()), 1e-6);
  }
  {
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0, dt = 0.02;
    for (int step = 0; step < 50; step++) {
      ode.Step(m, t, dt);
    }
    ASSERT_TRUE(visco.SolveElastic(m, t));
    EXPECT_LT(strain_error(problem.Displacement()), 1e-3);
  }
}

TEST_P(ViscoelasticTest, PowerLawLinearLimit) {
  auto marker = Marker(nbdr, {x0_attr, x1_attr});
  VectorFunctionCoefficient traction(dim, ConstantUniaxial);
  ConstantCoefficient gamma0(0.0), n3(3.0), mu0(kMu);
  PowerLawRelaxation law(gamma0, n3, mu0);
  auto linear = IsotropicMaxwellRheology::Maxwell(dim, *kappa, *mu, *tau);
  auto lawful = IsotropicMaxwellRheology::Maxwell(dim, *kappa, *mu, *tau, &law);
  EXPECT_TRUE(linear.IsLinear());
  EXPECT_FALSE(lawful.IsLinear());  // state-dependent by type, gamma = 0
  TractionProblem pl(fes.get(), linear, traction, marker);
  TractionProblem pn(fes.get(), lawful, traction, marker);
  for (int scheme = 0; scheme < 4; scheme++) {
    ViscoelasticOperator vl(pl), vn(pn);
    EXPECT_TRUE(vl.IsLinear());
    EXPECT_FALSE(vn.IsLinear());
    auto make = [scheme]() -> std::unique_ptr<ODESolver> {
      switch (scheme) {
        case 0:
          return std::make_unique<ExponentialTrapezoidSolver>();
        case 1:
          return std::make_unique<ExponentialEulerSolver>();
        case 2:
          return std::make_unique<BackwardEulerSolver>();
        default:
          return std::make_unique<RK4Solver>();
      }
    };
    auto ol = make(), on = make();
    ol->Init(vl);
    on->Init(vn);
    Vector ml(vl.Height()), mn(vn.Height());
    ml = 0.0;
    mn = 0.0;
    double tl = 0.0, tn = 0.0;
    double dt = scheme == 3 ? 0.1 : 0.5;
    for (int step = 0; step < 3; step++) {
      ol->Step(ml, tl, dt);
      on->Step(mn, tn, dt);
    }
    EXPECT_LT(RelMaxDiff(mn, ml), 1e-10) << "scheme " << scheme;
  }
}

TEST_P(ViscoelasticTest, PowerLawCreepUnderConstantStress) {
  // Stress control: T = dev sigma_0 is constant, so tau = tau0 F(|T|) is
  // constant in time and the creep is the linear one at that tau; the
  // trapezoid is exact at any dt. Doubling the stress multiplies the
  // long-time rate by 2^n in the power-law regime.
  auto marker = Marker(nbdr, {x0_attr, x1_attr});
  ConstantCoefficient gamma(3.0), n3(3.0), mu0(kMu);
  PowerLawRelaxation law(gamma, n3, mu0);
  auto rheology = IsotropicMaxwellRheology::Maxwell(dim, *kappa, *mu, *tau, &law);

  auto rate_at = [&](double sigma, double& exx_rate) {
    VectorFunctionCoefficient traction(dim, [sigma](const Vector& x, Vector& f) {
      f = 0.0;
      if (x[0] > 1.0 - 1e-8) {
        f[0] = sigma;
      } else if (x[0] < 1e-8) {
        f[0] = -sigma;
      }
    });
    TractionProblem problem(fes.get(), rheology, traction, marker);
    ViscoelasticOperator visco(problem);
    ExponentialTrapezoidSolver ode;
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0, dt = 2.0;
    for (int step = 0; step < 4; step++) {
      ode.Step(m, t, dt);
    }
    EXPECT_EQ(visco.LastCorrectorPasses(), 0);  // tau unchanged: stops early
    EXPECT_TRUE(visco.SolveElastic(m, t));
    // |dev sigma_0|^2 = sigma^2 (1 - 1/d).
    const double T = sigma * std::sqrt(1.0 - 1.0 / dim);
    const double F = 1.0 / (1.0 + 3.0 * std::pow(T / (2.0 * kMu), 2.0));
    const double eta = kMu * kTau * F;
    double exx0 = 0.0, eyy0 = 0.0;
    UniaxialStrain(dim, sigma, exx0, eyy0);
    const double exx = exx0 + sigma * (1.0 - 1.0 / dim) * t / (2.0 * eta);
    const double eyy = eyy0 - sigma / dim * t / (2.0 * eta);
    EXPECT_LT(MaxStrainError(problem.Displacement(), exx, eyy), 1e-8 * exx0);
    // The nodal times are tau0 F everywhere.
    const Vector& itau = visco.InverseRelaxationTimes(0, 0);
    for (int p = 0; p < itau.Size(); p++) {
      EXPECT_NEAR(itau[p] * kTau * F, 1.0, 1e-8);
    }
    exx_rate = (1.0 - 1.0 / dim) * sigma / (2.0 * eta);
  };
  double r1 = 0.0, r2 = 0.0;
  rate_at(kSigma, r1);
  rate_at(2.0 * kSigma, r2);
  // Between Newtonian (ratio 2) and power law (ratio 8): the composite law
  // gives 2 (1 + 3 x^2 4) / (1 + 3 x^2) with x = |T| / 2 mu.
  const double x = kSigma * std::sqrt(1.0 - 1.0 / dim) / (2.0 * kMu);
  const double expected = 2.0 * (1.0 + 12.0 * x * x) / (1.0 + 3.0 * x * x);
  EXPECT_NEAR(r2 / r1, expected, 1e-8 * expected);
}

TEST_P(ViscoelasticTest, PowerLawRelaxationOrders) {
  // Strain control: u = A x on the boundary, one Maxwell branch with the
  // power law. The state is homogeneous, T = 2 mu (d - m), and at every
  // node m obeys dm/dt = (d - m) (1 + gamma (|T| / 2 mu0)^(n-1)) / tau0,
  // integrated here by RK4 at a tiny step as the reference.
  if (order == 2 && elementType == 0) {
    return;  // the same physics; keep the run short
  }
  ConstantCoefficient gamma(4.0), n3(3.0), mu0(kMu);
  PowerLawRelaxation law(gamma, n3, mu0);
  auto rheology = IsotropicMaxwellRheology::Maxwell(dim, *kappa, *mu, *tau, &law);
  DenseMatrix A(dim);
  A = 0.0;
  A(0, 0) = 0.5;
  A(0, 1) = 0.2;
  A(1, 0) = -0.1;
  A(1, 1) = -0.3;
  if (dim == 3) {
    A(2, 2) = 0.15;
    A(0, 2) = 0.25;
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
  const Vector d0 = DeviatoricPart(A);  // trace-free components
  const int nc = d0.Size();

  // Nodal reference: full-tensor |T| from the trace-free components.
  auto dev_norm = [&](const Vector& r) {
    Vector full(dim * (dim + 1) / 2);
    for (int c = 0; c < nc; c++) {
      full[c] = r[c];
    }
    double tr = 0.0;
    for (int j = 0; j < dim - 1; j++) {
      tr += full[SymmetricTensorBasis::Index(dim, j, j)];
    }
    full[SymmetricTensorBasis::Index(dim, dim - 1, dim - 1)] = -tr;
    return LocalState::DeviatoricNorm(dim, full.GetData());
  };
  auto rate = [&](const Vector& m, Vector& k) {
    Vector r(d0);
    r -= m;
    const double T = 2.0 * kMu * dev_norm(r);
    const double g = (1.0 + 4.0 * std::pow(T / (2.0 * kMu), 2.0)) / kTau;
    k = r;
    k *= g;
  };
  const double t_final = 1.0;
  Vector m_ref(nc), k1(nc), k2(nc), k3(nc), k4(nc), tmp(nc);
  m_ref = 0.0;
  {
    const int n = 40000;
    const double h = t_final / n;
    for (int i = 0; i < n; i++) {
      rate(m_ref, k1);
      add(m_ref, 0.5 * h, k1, tmp);
      rate(tmp, k2);
      add(m_ref, 0.5 * h, k2, tmp);
      rate(tmp, k3);
      add(m_ref, h, k3, tmp);
      rate(tmp, k4);
      for (int c = 0; c < nc; c++) {
        m_ref[c] += h / 6.0 * (k1[c] + 2.0 * k2[c] + 2.0 * k3[c] + k4[c]);
      }
    }
  }
  ASSERT_GT(m_ref.Normlinf(), 0.0);
  const int nd = visco.InternalScalarSpace(0).GetVSize();
  auto error = [&](const Vector& m) {
    double e = 0.0;
    Vector mk = visco.Branch(m, 0, 0);
    for (int c = 0; c < nc; c++) {
      for (int p = 0; p < nd; p++) {
        e = std::max(e, std::abs(mk[c * nd + p] - m_ref[c]));
      }
    }
    return e / m_ref.Normlinf();
  };
  auto integrate = [&](ODESolver& ode, double dt) {
    ode.Init(visco);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0;
    const int n = static_cast<int>(std::round(t_final / dt));
    for (int step = 0; step < n; step++) {
      ode.Step(m, t, dt);
    }
    return error(m);
  };
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
  for (auto& c : cases) {
    const double e1 = integrate(*c.ode, 1.0 / 8.0);
    const double e2 = integrate(*c.ode, 1.0 / 16.0);
    const double e3 = integrate(*c.ode, 1.0 / 32.0);
    const double rate23 = std::log2(e2 / e3);
    EXPECT_NEAR(rate23, c.expected_order, 0.3)
        << c.name << ": e(1/8) = " << e1 << ", e(1/16) = " << e2
        << ", e(1/32) = " << e3;
    EXPECT_LT(e3, c.expected_order > 1.5 ? 2e-3 : 5e-2) << c.name;
  }

  // Converged corrector: still second order, smaller constant.
  {
    visco.SetCorrectorIterations(20, 1e-10);
    ExponentialTrapezoidSolver ode;
    const double e2 = integrate(ode, 1.0 / 16.0);
    const double e3 = integrate(ode, 1.0 / 32.0);
    EXPECT_NEAR(std::log2(e2 / e3), 2.0, 0.3) << "converged corrector";
    EXPECT_GT(visco.LastCorrectorPasses(), 0);
    visco.SetCorrectorIterations(1);
  }

  // Adaptive stepping: the estimate is that of the first-order ETD1
  // companion, so the tolerance is conservative for the propagated
  // trapezoid solution; at rtol = 1e-3 the run takes a few tens of steps
  // (starting from a step that is rejected) and lands well inside 2e-3.
  {
    AdaptiveExponentialTrapezoidSolver ode;
    ode.Init(visco);
    ode.SetTolerances(1e-3, 1e-8);
    Vector m(visco.Height());
    m = 0.0;
    double t = 0.0, dt = 0.5;
    const int steps = ode.Integrate(m, t, t_final, dt);
    EXPECT_NEAR(t, t_final, 1e-12);
    EXPECT_LT(error(m), 2e-3);
    EXPECT_GT(steps, 2);
    EXPECT_LT(steps, 80);
    EXPECT_GT(ode.NumRejectedSteps(), 0);
    EXPECT_EQ(ode.NumAcceptedSteps(), steps);
    EXPECT_LE(ode.LastErrorEstimate(), 1.0);
  }
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
  auto rheology = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
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
  TwoFieldProblem(LinearElasticProblemBase& a, LinearElasticProblemBase& b) : p_{&a, &b} {}
  int NumDisplacementFields() const override { return 2; }
  FiniteElementSpace& DisplacementSpace(int i) override {
    return p_[i]->DisplacementSpace();
  }
  const GridFunction& Displacement(int i) const override {
    return p_[i]->Displacement();
  }
  const mfemElasticity::Rheology& Rheology(int i) const override {
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
  bool SupportsRelaxationWeights() const override { return true; }
  void SetRelaxationWeights(int i,
                            const std::vector<Coefficient*>& beta) override {
    p_[i]->SetRelaxationWeights(0, beta);
  }
  void ClearRelaxationWeights() override {
    for (auto* p : p_) {
      p->ClearRelaxationWeights();
    }
  }
  void RegisterFields(DataCollection&) override {}

 private:
  LinearElasticProblemBase* p_[2];
};

TEST(ViscoelasticTwoField, RoutesForcesPerField) {
  const int dim = 2;
  Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
  H1_FECollection fec1(1, dim), fec2(2, dim);
  FiniteElementSpace fes1(&mesh, &fec1, dim), fes2(&mesh, &fec2, dim);
  ConstantCoefficient kappa(Kappa(dim)), mu(kMu), tau(1.0), mu_inf(0.4),
      tau2(3.0);
  auto rh1 = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  std::vector<MaxwellBranch> branches{{&mu, &tau}, {&mu, &tau2}};
  IsotropicMaxwellRheology rh2(dim, kappa, mu_inf, branches);
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

// An elastic field alongside a Maxwell one: the elastic field carries no
// internal state and its displacement is the static solution at every
// time, while the Maxwell field evolves as it would alone.
TEST(ViscoelasticTwoField, ElasticFieldHasNoState) {
  const int dim = 2;
  Mesh mesh = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
  H1_FECollection fec(1, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);
  ConstantCoefficient kappa(Kappa(dim)), mu(kMu), tau(1.0);
  IsotropicElasticRheology elastic(dim, kappa, mu);
  auto maxwell = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu, tau);
  const int x0 = BdrAttributeAt(mesh, 0, 0.0),
            x1 = BdrAttributeAt(mesh, 0, 1.0);
  auto ess = Marker(mesh.bdr_attributes.Max(), {x0});
  auto pull_marker = Marker(mesh.bdr_attributes.Max(), {x1});
  VectorFunctionCoefficient pull(dim, PullTraction);

  ClampedProblem a1(&fes, elastic, ess, pull, pull_marker);
  ClampedProblem b1(&fes, maxwell, ess, pull, pull_marker);
  TwoFieldProblem two(a1, b1);
  ViscoelasticOperator visco_two(two);
  ASSERT_EQ(visco_two.NumFields(), 2);
  ASSERT_EQ(visco_two.NumBranches(0), 0);
  ASSERT_EQ(visco_two.NumBranches(1), 1);
  ASSERT_EQ(visco_two.Height(), visco_two.BranchSize(1));
  EXPECT_EQ(visco_two.BranchOffset(1, 0), 0);
  EXPECT_TRUE(visco_two.IsLinear());

  ClampedProblem b2(&fes, maxwell, ess, pull, pull_marker);
  ViscoelasticOperator visco_b(b2);
  ClampedProblem static_problem(&fes, elastic, ess, pull, pull_marker);

  for (int scheme = 0; scheme < 2; scheme++) {
    auto make = [scheme]() -> std::unique_ptr<ODESolver> {
      if (scheme == 0) {
        return std::make_unique<ExponentialTrapezoidSolver>();
      }
      return std::make_unique<RK4Solver>();
    };
    auto ode_two = make(), ode_b = make();
    ode_two->Init(visco_two);
    ode_b->Init(visco_b);
    Vector m_two(visco_two.Height()), m_b(visco_b.Height());
    m_two = 0.0;
    m_b = 0.0;
    double t_two = 0.0, t_b = 0.0, dt = 0.4;
    for (int step = 0; step < 3; step++) {
      ode_two->Step(m_two, t_two, dt);
      ode_b->Step(m_b, t_b, dt);
    }
    EXPECT_LT(RelMaxDiff(visco_two.Branch(m_two, 1, 0), m_b), 1e-12);
    ASSERT_GT(m_b.Normlinf(), 0.0);

    ASSERT_TRUE(visco_two.SolveElastic(m_two, t_two));
    static_problem.AssembleForce(t_two);
    ASSERT_TRUE(static_problem.Solve());
    EXPECT_GT(static_problem.Displacement().Normlinf(), 0.0);
    EXPECT_LT(RelMaxDiff(a1.Displacement(), static_problem.Displacement()),
              1e-12);
  }
}

}  // namespace
