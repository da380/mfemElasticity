#include "TestCommon.hpp"

/*
  Tests for IsotropicMaxwellRheology (design doc
  doc/viscoelastic_design.md, section 5, test 1): the derived coefficients
  evaluate to the right combinations of the inputs at quadrature points.
  And for AnisotropicMaxwellRheology (doc/fluid_solid_design.md section 7):
  the unrelaxed tensor, the branch moduli, and the stiffness objects of the
  two rheologies assembling the same matrix for an isotropic tensor, with
  and without relaxation weights.
*/

namespace {

// Evaluate c at every quadrature point of an order-3 rule on every element
// and apply `check(value, x)`.
template <class F>
void ForEachQuadraturePoint(Mesh& mesh, Coefficient& c, F check) {
  for (auto e = 0; e < mesh.GetNE(); e++) {
    auto* T = mesh.GetElementTransformation(e);
    const auto& ir = IntRules.Get(mesh.GetElementGeometry(e), 3);
    for (auto q = 0; q < ir.GetNPoints(); q++) {
      const auto& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      Vector x(mesh.Dimension());
      T->Transform(ip, x);
      check(c.Eval(*T, ip), x);
    }
  }
}

double Kappa(const Vector& x) { return 2.0 + x[0]; }
double MuInf(const Vector& x) { return 0.5 + 0.25 * x[x.Size() - 1]; }
double Mu1(const Vector& x) { return 1.0 + x[0] * x[0]; }
double Mu2(const Vector& x) { return 0.3 + x[1]; }

class RheologyTest : public testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    dim = GetParam();
    mesh = std::make_unique<Mesh>(
        dim == 2 ? Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL)
                 : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON));
  }
  int dim = 2;
  std::unique_ptr<Mesh> mesh;
  FunctionCoefficient kappa{Kappa}, mu_inf{MuInf}, mu1{Mu1}, mu2{Mu2};
  ConstantCoefficient tau1{1.0}, tau2{100.0};
};

TEST_P(RheologyTest, UnrelaxedModuli) {
  std::vector<MaxwellBranch> branches{{&mu1, &tau1}, {&mu2, &tau2}};
  IsotropicMaxwellRheology r(dim, kappa, mu_inf, branches);

  EXPECT_EQ(r.SpaceDim(), dim);
  EXPECT_EQ(r.NumBranches(), 2);
  EXPECT_EQ(r.Branch(0).mu, &mu1);
  EXPECT_EQ(r.Branch(1).tau, &tau2);
  EXPECT_EQ(&r.BulkModulus(), &kappa);
  EXPECT_EQ(&r.LongTermShearModulus(), &mu_inf);

  ForEachQuadraturePoint(*mesh, r.UnrelaxedShearModulus(),
                         [](double v, const Vector& x) {
                           EXPECT_NEAR(v, MuInf(x) + Mu1(x) + Mu2(x), 1e-14);
                         });
  ForEachQuadraturePoint(*mesh, r.UnrelaxedLame(),
                         [this](double v, const Vector& x) {
                           const auto mu_u = MuInf(x) + Mu1(x) + Mu2(x);
                           EXPECT_NEAR(v, Kappa(x) - 2.0 * mu_u / dim, 1e-14);
                         });
}

TEST_P(RheologyTest, ElasticFactory) {
  auto r = IsotropicMaxwellRheology::Elastic(dim, kappa, mu1);
  EXPECT_EQ(r.NumBranches(), 0);
  ForEachQuadraturePoint(
      *mesh, r.UnrelaxedShearModulus(),
      [](double v, const Vector& x) { EXPECT_NEAR(v, Mu1(x), 1e-14); });
}

TEST_P(RheologyTest, MaxwellFactory) {
  auto r = IsotropicMaxwellRheology::Maxwell(dim, kappa, mu1, tau1);
  EXPECT_EQ(r.NumBranches(), 1);
  EXPECT_EQ(r.Branch(0).mu, &mu1);
  EXPECT_EQ(r.Branch(0).tau, &tau1);
  // mu_inf = 0, so mu_U = mu1 (the factory's owned zero must survive the
  // move out of the factory).
  ForEachQuadraturePoint(*mesh, r.LongTermShearModulus(),
                         [](double v, const Vector&) { EXPECT_EQ(v, 0.0); });
  ForEachQuadraturePoint(
      *mesh, r.UnrelaxedShearModulus(),
      [](double v, const Vector& x) { EXPECT_NEAR(v, Mu1(x), 1e-14); });
}

TEST_P(RheologyTest, AnisotropicTensors) {
  auto C_inf = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa,
                                                                  mu_inf);
  ConstantCoefficient zero(0.0);
  auto C_1 = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, zero, mu1);
  auto C_2 = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, zero, mu2);
  std::vector<AnisotropicBranch> branches{{&C_1, &tau1}, {&C_2, &tau2}};
  AnisotropicMaxwellRheology r(dim, C_inf, branches);
  EXPECT_EQ(r.SpaceDim(), dim);
  EXPECT_EQ(r.NumBranches(), 2);
  EXPECT_FALSE(r.TraceFreeInternalVariables());
  EXPECT_EQ(&r.RelaxationTime(1), &tau2);
  EXPECT_EQ(&r.LongTermTensor(), &C_inf);

  // C_U = C_inf + C_1 + C_2 = the isotropic tensor with mu_U.
  std::vector<MaxwellBranch> ibranches{{&mu1, &tau1}, {&mu2, &tau2}};
  IsotropicMaxwellRheology iso(dim, kappa, mu_inf, ibranches);
  auto C_u = IsotropicElasticTensorCoefficient::FromBulkModulus(
      dim, kappa, iso.UnrelaxedShearModulus());
  const int ns = SymmetricTensorBasis::Size(dim);
  DenseMatrix Ca(ns), Cb(ns), P(ns);
  SymmetricTensorBasis::DeviatoricProjector(dim, P);
  for (auto e = 0; e < mesh->GetNE(); e++) {
    auto* T = mesh->GetElementTransformation(e);
    const auto& ir = IntRules.Get(mesh->GetElementGeometry(e), 3);
    for (auto q = 0; q < ir.GetNPoints(); q++) {
      const auto& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      r.UnrelaxedTensor().Eval(Ca, *T, ip);
      C_u.Eval(Cb, *T, ip);
      Ca -= Cb;
      EXPECT_LT(Ca.MaxMaxNorm(), 1e-13 * Cb.MaxMaxNorm());
      // Branch moduli: the anisotropic C_1 and the isotropic 2 mu_1 P_dev.
      r.BranchModulus(0, *T, ip, Ca);
      iso.BranchModulus(0, *T, ip, Cb);
      Ca -= Cb;
      EXPECT_LT(Ca.MaxMaxNorm(), 1e-13 * Cb.MaxMaxNorm());
      Vector x(dim);
      T->Transform(ip, x);
      iso.BranchModulus(1, *T, ip, Cb);
      Cb.Add(-2.0 * Mu2(x), P);
      EXPECT_LT(Cb.MaxMaxNorm(), 1e-13);
    }
  }
}

TEST_P(RheologyTest, StiffnessObjectsAgree) {
  H1_FECollection fec(1, dim);
  FiniteElementSpace fes(mesh.get(), &fec, dim);
  std::vector<MaxwellBranch> ibranches{{&mu1, &tau1}, {&mu2, &tau2}};
  IsotropicMaxwellRheology iso(dim, kappa, mu_inf, ibranches);
  auto C_inf = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa,
                                                                  mu_inf);
  ConstantCoefficient zero(0.0);
  auto C_1 = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, zero, mu1);
  auto C_2 = IsotropicElasticTensorCoefficient::FromBulkModulus(dim, zero, mu2);
  std::vector<AnisotropicBranch> branches{{&C_1, &tau1}, {&C_2, &tau2}};
  AnisotropicMaxwellRheology aniso(dim, C_inf, branches);

  auto si = iso.MakeStiffness();
  auto sa = aniso.MakeStiffness();
  // Template forms holding the integrators; each assembly borrows them.
  BilinearForm ti(&fes), ta(&fes);
  si->AddIntegrators(ti);
  sa->AddIntegrators(ta);
  double norm_i = 0.0;
  auto assemble = [&]() {
    BilinearForm ai(&fes, &ti), aa(&fes, &ta);
    ai.Assemble();
    aa.Assemble();
    ai.Finalize();
    aa.Finalize();
    norm_i = ai.SpMat().MaxNorm();
    return MaxDiff(ai.SpMat(), aa.SpMat()) / norm_i;
  };
  EXPECT_FALSE(si->IsRelaxed());
  EXPECT_LT(assemble(), 1e-13);
  const double norm_u = norm_i;

  FunctionCoefficient b1([](const Vector& x) { return 0.2 + 0.5 * x[0]; });
  ConstantCoefficient b2(0.7);
  si->SetRelaxationWeights({&b1, &b2});
  sa->SetRelaxationWeights({&b1, &b2});
  EXPECT_TRUE(si->IsRelaxed());
  EXPECT_TRUE(sa->IsRelaxed());
  EXPECT_LT(assemble(), 1e-13);
  EXPECT_GT(std::abs(norm_i - norm_u), 1e-3 * norm_u);

  si->ClearRelaxationWeights();
  sa->ClearRelaxationWeights();
  EXPECT_LT(assemble(), 1e-13);
  EXPECT_NEAR(norm_i, norm_u, 1e-13 * norm_u);
}

INSTANTIATE_TEST_SUITE_P(Rheology, RheologyTest, testing::Values(2, 3));

}  // namespace
