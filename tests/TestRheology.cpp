#include "TestCommon.hpp"

/*
  Tests for GeneralisedMaxwellRheology (design doc
  doc/viscoelastic_design.md, section 5, test 1): the derived coefficients
  evaluate to the right combinations of the inputs at quadrature points.
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
  GeneralisedMaxwellRheology r(dim, kappa, mu_inf, branches);

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
  auto r = GeneralisedMaxwellRheology::Elastic(dim, kappa, mu1);
  EXPECT_EQ(r.NumBranches(), 0);
  ForEachQuadraturePoint(
      *mesh, r.UnrelaxedShearModulus(),
      [](double v, const Vector& x) { EXPECT_NEAR(v, Mu1(x), 1e-14); });
}

TEST_P(RheologyTest, MaxwellFactory) {
  auto r = GeneralisedMaxwellRheology::Maxwell(dim, kappa, mu1, tau1);
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

INSTANTIATE_TEST_SUITE_P(Rheology, RheologyTest, testing::Values(2, 3));

}  // namespace
