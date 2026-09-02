#include "ElasticTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for ElasticProblemBase, TractionProblem and ClampedProblem (design
  doc doc/viscoelastic_design.md, section 5, test 2 and the elastic parts
  of the interface contract).

  - The bulk/deviatoric split assembled by the base class equals MFEM's
    ElasticityIntegrator(lambda, mu) with lambda = kappa - 2 mu / d.
  - ClampedProblem reproduces a direct MFEM assembly and solve.
  - SetEffectiveShearModulus (constant and piecewise-constant L2 field)
    reproduces a direct assembly with that modulus; Clear restores mu_U.
  - TractionProblem under uniaxial stress gives the exact constant strain.
  - Loads scale with time through the registered coefficients, and
    AddForce superposes exactly like an integrator on the load.
*/

namespace {

using namespace elastic_test;

// (dim, elementType, order)
using Param = std::tuple<int, int, int>;

class ElasticProblemTest : public testing::TestWithParam<Param> {
 protected:
  void SetUp() override {
    std::tie(dim, elementType, order) = GetParam();
    mesh = std::make_unique<Mesh>(MakeSmallMesh(dim, elementType));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes = std::make_unique<FiniteElementSpace>(mesh.get(), fec.get(), dim);
    kappa = std::make_unique<ConstantCoefficient>(Kappa(dim));
    mu = std::make_unique<ConstantCoefficient>(kMu);
    lambda = std::make_unique<ConstantCoefficient>(kLambda);
    rheology = std::make_unique<GeneralisedMaxwellRheology>(
        GeneralisedMaxwellRheology::Elastic(dim, *kappa, *mu));
    x0_attr = BdrAttributeAt(*mesh, 0, 0.0);
    x1_attr = BdrAttributeAt(*mesh, 0, 1.0);
    nbdr = mesh->bdr_attributes.Max();
  }

  // Direct MFEM solve of the clamped problem with (lambda, mu) and the
  // load b at time t (b already assembled), to tight tolerance.
  GridFunction DirectClamped(Coefficient& lam, Coefficient& m,
                             const Array<int>& ess_bdr, LinearForm& b) {
    BilinearForm a(fes.get());
    a.AddDomainIntegrator(new ElasticityIntegrator(lam, m));
    a.Assemble();
    Array<int> ess_bdr_copy(ess_bdr), ess_tdof;
    fes->GetEssentialTrueDofs(ess_bdr_copy, ess_tdof);
    GridFunction u(fes.get());
    u = 0.0;
    SparseMatrix A;
    Vector X, B;
    a.FormLinearSystem(ess_tdof, u, b, A, X, B);
    GSSmoother prec(A);
    CGSolver cg;
    cg.SetPreconditioner(prec);
    cg.SetOperator(A);
    cg.SetRelTol(1e-14);
    cg.SetAbsTol(0.0);
    cg.SetMaxIter(20000);
    cg.SetPrintLevel(0);
    cg.Mult(B, X);
    EXPECT_TRUE(cg.GetConverged());
    a.RecoverFEMSolution(X, b, u);
    return u;
  }

  static double RelMaxDiff(const Vector& a, const Vector& b) {
    Vector d(a);
    d -= b;
    return d.Normlinf() / (b.Normlinf() + 1e-300);
  }

  int dim = 2, elementType = 0, order = 1, x0_attr = -1, x1_attr = -1, nbdr = 0;
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<FiniteElementCollection> fec;
  std::unique_ptr<FiniteElementSpace> fes;
  std::unique_ptr<ConstantCoefficient> kappa, mu, lambda;
  std::unique_ptr<GeneralisedMaxwellRheology> rheology;
};

TEST_P(ElasticProblemTest, SplitIdentity) {
  BilinearForm split(fes.get());
  split.AddDomainIntegrator(new ElasticityIntegrator(*kappa, 1.0, 0.0));
  split.AddDomainIntegrator(new ElasticityIntegrator(*mu, -2.0 / dim, 1.0));
  split.Assemble();
  split.Finalize();

  BilinearForm ref(fes.get());
  ref.AddDomainIntegrator(
      new ElasticityIntegrator(rheology->UnrelaxedLame(), *mu));
  ref.Assemble();
  ref.Finalize();

  EXPECT_LT(MaxDiff(split.SpMat(), ref.SpMat()), 1e-13 * ref.SpMat().MaxNorm());
}

TEST_P(ElasticProblemTest, ClampedMatchesDirect) {
  auto ess_bdr = Marker(nbdr, {x0_attr});
  auto pull_marker = Marker(nbdr, {x1_attr});
  VectorFunctionCoefficient traction(dim, PullTraction);

  ClampedProblem problem(fes.get(), *rheology, ess_bdr, traction, pull_marker);
  EXPECT_EQ(problem.NumDisplacementFields(), 1);
  EXPECT_EQ(&problem.DisplacementSpace(), fes.get());
  EXPECT_EQ(&problem.Rheology(), rheology.get());
  EXPECT_TRUE(problem.SupportsEffectiveShearModulus());
  EXPECT_FALSE(problem.IsParallel());

  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());

  LinearForm b(fes.get());
  traction.SetTime(0.0);
  b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction),
                          pull_marker);
  b.Assemble();
  auto u_ref = DirectClamped(*lambda, *mu, ess_bdr, b);
  ASSERT_GT(u_ref.Normlinf(), 0.0);
  EXPECT_LT(RelMaxDiff(problem.Displacement(), u_ref), 1e-8);

  // A second solve at the same time warm-starts and must return the same.
  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());
  EXPECT_LT(RelMaxDiff(problem.Displacement(), u_ref), 1e-8);
}

TEST_P(ElasticProblemTest, EffectiveShearModulus) {
  auto ess_bdr = Marker(nbdr, {x0_attr});
  auto pull_marker = Marker(nbdr, {x1_attr});
  VectorFunctionCoefficient traction(dim, PullTraction);
  ClampedProblem problem(fes.get(), *rheology, ess_bdr, traction, pull_marker);

  LinearForm b(fes.get());
  traction.SetTime(0.0);
  b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction),
                          pull_marker);
  b.Assemble();

  // Constant effective modulus.
  ConstantCoefficient mu_eff(0.4 * kMu);
  SumCoefficient lambda_eff(*kappa, mu_eff, 1.0, -2.0 / dim);
  problem.SetEffectiveShearModulus(0, mu_eff);
  EXPECT_EQ(&problem.CurrentShearModulus(), &mu_eff);
  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());
  auto u_ref = DirectClamped(lambda_eff, mu_eff, ess_bdr, b);
  EXPECT_LT(RelMaxDiff(problem.Displacement(), u_ref), 1e-8);

  // Piecewise-constant effective modulus given as a GridFunctionCoefficient
  // on an L2 space, as the viscoelastic layer will supply it.
  L2_FECollection l2fec(0, dim);
  FiniteElementSpace sfes(mesh.get(), &l2fec);
  GridFunction mu_field(&sfes);
  FunctionCoefficient mu_var(
      [](const Vector& x) { return kMu * (0.3 + 0.6 * x[0]); });
  mu_field.ProjectCoefficient(mu_var);
  GridFunctionCoefficient mu_eff_gf(&mu_field);
  SumCoefficient lambda_eff_gf(*kappa, mu_eff_gf, 1.0, -2.0 / dim);
  problem.SetEffectiveShearModulus(0, mu_eff_gf);
  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());
  auto u_ref_gf = DirectClamped(lambda_eff_gf, mu_eff_gf, ess_bdr, b);
  EXPECT_LT(RelMaxDiff(problem.Displacement(), u_ref_gf), 1e-8);
  EXPECT_GT(RelMaxDiff(u_ref_gf, u_ref), 1e-3);  // the two cases differ

  // Clear restores mu_U.
  problem.ClearEffectiveShearModulus();
  EXPECT_EQ(&problem.CurrentShearModulus(), &rheology->UnrelaxedShearModulus());
  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());
  auto u_ref_u = DirectClamped(*lambda, *mu, ess_bdr, b);
  EXPECT_LT(RelMaxDiff(problem.Displacement(), u_ref_u), 1e-8);
}

TEST_P(ElasticProblemTest, TractionUniaxialStrain) {
  auto marker = Marker(nbdr, {x0_attr, x1_attr});
  VectorFunctionCoefficient traction(dim, UniaxialTraction);
  TractionProblem problem(fes.get(), *rheology, traction, marker);

  double exx = 0.0, eyy = 0.0;
  UniaxialStrain(dim, kSigma, exx, eyy);

  problem.AssembleForce(0.0);
  ASSERT_TRUE(problem.Solve());
  EXPECT_LT(MaxStrainError(problem.Displacement(), exx, eyy), 1e-8 * exx);

  // Doubling the load (t = 1) doubles the strain, from a warm start.
  problem.AssembleForce(1.0);
  ASSERT_TRUE(problem.Solve());
  EXPECT_LT(MaxStrainError(problem.Displacement(), 2.0 * exx, 2.0 * eyy),
            1e-8 * exx);
}

TEST_P(ElasticProblemTest, AddForceSuperposes) {
  auto ess_bdr = Marker(nbdr, {x0_attr});
  auto pull_marker = Marker(nbdr, {x1_attr});
  VectorFunctionCoefficient traction(dim, PullTraction);
  Vector g(dim);
  g = 0.0;
  g[0] = 0.1;
  VectorConstantCoefficient body(g);

  // Problem 1: body force through AddForce.
  ClampedProblem p1(fes.get(), *rheology, ess_bdr, traction, pull_marker);
  LinearForm extra(fes.get());
  extra.AddDomainIntegrator(new VectorDomainLFIntegrator(body));
  extra.Assemble();
  p1.AssembleForce(0.5);
  p1.AddForce(extra);
  ASSERT_TRUE(p1.Solve());

  // Problem 2: the same body force as a load integrator.
  ClampedProblem p2(fes.get(), *rheology, ess_bdr, traction, pull_marker);
  p2.ExternalLoad().AddDomainIntegrator(new VectorDomainLFIntegrator(body));
  p2.AssembleForce(0.5);
  ASSERT_TRUE(p2.Solve());
  EXPECT_LT(RelMaxDiff(p1.Displacement(), p2.Displacement()), 1e-8);

  // AssembleForce clears the increment: p1 at t = 0.5 without the extra
  // force must differ from the above and equal p2 without it.
  p1.AssembleForce(0.5);
  ASSERT_TRUE(p1.Solve());
  EXPECT_GT(RelMaxDiff(p1.Displacement(), p2.Displacement()), 1e-3);
}

INSTANTIATE_TEST_SUITE_P(ElasticProblem, ElasticProblemTest,
                         testing::Combine(testing::Values(2, 3),
                                          testing::Values(0, 1),
                                          testing::Values(1, 2)));

}  // namespace
