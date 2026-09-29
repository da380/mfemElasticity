#include "SelfGravitatingTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for the background-state module (background.hpp): the radial
  hydrostatic balance against analytic profiles, and the generated
  coefficient chains against hand-rolled references. The end-to-end
  acceptance — a relabelled background reproducing the base solution —
  is TestReferentialProblem.RelabelledEquilibrium2D, which runs on this
  module.
*/

namespace {

using namespace self_grav_test;

constexpr double kPi = std::numbers::pi;

}  // namespace

TEST(Background, UniformStateMatchesAnalytic) {
  for (int dim : {2, 3}) {
    RadialHydrostaticState state(
        dim, [](double) { return kRho; }, kG, 1.0);
    for (double r : {0.0, 0.25, 0.5, 0.75, 1.0}) {
      const double g = dim == 2 ? 2.0 * kPi * kG * kRho * r
                                : 4.0 * kPi * kG * kRho * r / 3.0;
      const double p = dim == 2
                           ? kPi * kG * kRho * kRho * (1.0 - r * r)
                           : 2.0 * kPi * kG * kRho * kRho * (1.0 - r * r) / 3.0;
      EXPECT_NEAR(state.Gravity(r), g, 1e-6);
      EXPECT_NEAR(state.Pressure(r), p, 1e-6);
    }
    // Exterior: the field of the total mass, zero pressure.
    const double g_ext = dim == 2 ? 2.0 * kPi * kG * kRho / 1.5
                                  : 4.0 * kPi * kG * kRho / (3.0 * 1.5 * 1.5);
    EXPECT_NEAR(state.Gravity(1.5), g_ext, 1e-6);
    EXPECT_EQ(state.Pressure(1.5), 0.0);
    EXPECT_EQ(state.Density(1.5), 0.0);
  }
}

TEST(Background, StratifiedStateMatchesAnalytic) {
  // rho = rho0 (1 - r^2/2), R = 1, 3-D:
  // g = 4 pi G rho0 (r/3 - r^3/10),
  // p = 4 pi G rho0^2 (F(1) - F(r)), F(s) = s^2/6 - s^4/15 + s^6/120.
  const double rho0 = 1.3;
  RadialHydrostaticState state(
      3, [rho0](double r) { return rho0 * (1.0 - 0.5 * r * r); }, kG, 1.0);
  auto F = [](double s) {
    const double s2 = s * s;
    return s2 / 6.0 - s2 * s2 / 15.0 + s2 * s2 * s2 / 120.0;
  };
  for (double r : {0.1, 0.4, 0.7, 0.95}) {
    const double g = 4.0 * kPi * kG * rho0 * (r / 3.0 - r * r * r / 10.0);
    const double p = 4.0 * kPi * kG * rho0 * rho0 * (F(1.0) - F(r));
    EXPECT_NEAR(state.Gravity(r), g, 1e-6);
    EXPECT_NEAR(state.Pressure(r), p, 1e-6);
  }
}

// The generated coefficients equal the hand-rolled chain on the uniform
// disc: p0 = pi G rho^2 (1 - r^2), C = Bare(C_eff, p0), S_e = -p0 1.
TEST(Background, HydrostaticCoefficientsMatchHandRolled) {
  const int dim = 2;
  Mesh mesh(MeshFile(dim).c_str(), 1, 1);

  RadialHydrostaticBackground bg(
      dim, [](double) { return kRho; }, [](double) { return kKappa; },
      [](double) { return kMu; }, kG, 1.0);

  ConstantCoefficient kappa(kKappa), mu(kMu);
  auto C_eff =
      IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);
  FunctionCoefficient p0([](const Vector& x) {
    return kPi * kG * kRho * kRho * (1.0 - (x * x));
  });
  BareElasticTensorCoefficient C_ref(dim, C_eff, p0);

  int n_body = 0;
  for (int e = 0; e < mesh.GetNE(); e++) {
    if (mesh.GetAttribute(e) != 1) {
      continue;
    }
    n_body++;
    auto* T = mesh.GetElementTransformation(e);
    const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(e));
    T->SetIntPoint(&ip);
    Vector x(dim);
    T->Transform(ip, x);

    const double p_ref = kPi * kG * kRho * kRho * (1.0 - (x * x));
    EXPECT_NEAR(bg.Pressure().Eval(*T, ip), p_ref, 1e-6);
    EXPECT_NEAR(bg.Density().Eval(*T, ip), kRho, 1e-14);

    DenseMatrix C1, C2;
    bg.ElasticTensor().Eval(C1, *T, ip);
    C_ref.Eval(C2, *T, ip);
    C1 -= C2;
    EXPECT_LT(C1.MaxMaxNorm(), 1e-6);

    DenseMatrix S;
    bg.EquilibriumStress().Eval(S, *T, ip);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        EXPECT_NEAR(S(i, j), i == j ? -p_ref : 0.0, 1e-6);
      }
    }
  }
  ASSERT_GT(n_body, 0);

  // The equilibrium mapping is the identity, exactly.
  auto* T = mesh.GetElementTransformation(0);
  const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(0));
  T->SetIntPoint(&ip);
  Vector x(dim), y(dim);
  T->Transform(ip, x);
  bg.EquilibriumMapping().Eval(y, *T, ip);
  y -= x;
  EXPECT_LT(y.Norml2(), 1e-15);
  DenseMatrix F;
  bg.EquilibriumMapping().EvalGradient(F, *T, ip);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      EXPECT_NEAR(F(i, j), i == j ? 1.0 : 0.0, 1e-15);
    }
  }
}

// The relabelled chain equals the hand-rolled transformation laws at a
// point, for a non-trivial interior map.
TEST(Background, RelabelledCoefficientsMatchHandRolled) {
  const int dim = 2;
  Mesh mesh(MeshFile(dim).c_str(), 1, 1);

  RadialHydrostaticBackground base(
      dim, [](double) { return kRho; }, [](double) { return kKappa; },
      [](double) { return kMu; }, kG, 1.0);

  // xi = x (1 + c (r (1 - r))^2) inside r < 1, identity beyond.
  const double c = 0.3;
  auto h = [](double r) {
    if (r >= 1.0) {
      return 0.0;
    }
    const double q = r * (1.0 - r);
    return q * q;
  };
  CallableDiffeomorphism xi(
      dim,
      [c, h](const Vector& x, Vector& y) {
        y = x;
        y *= 1.0 + c * h(x.Norml2());
      },
      [c, h](const Vector& x, DenseMatrix& F) {
        const double r = x.Norml2();
        F = 0.0;
        const double f = 1.0 + c * h(r);
        for (int i = 0; i < x.Size(); i++) {
          F(i, i) = f;
        }
        if (r > 0.0 && r < 1.0) {
          const double dh = 2.0 * r * (1.0 - r) * (1.0 - 2.0 * r);
          for (int i = 0; i < x.Size(); i++) {
            for (int j = 0; j < x.Size(); j++) {
              F(i, j) += c * dh / r * x(i) * x(j);
            }
          }
        }
      });
  RelabelledBackground rel(base, xi);

  // Hand-rolled chain, as tier (ii) originally built it.
  auto pressure = [&base](const Vector& y) {
    return base.PressureAt(y.Norml2());
  };
  TransformedFunctionCoefficient p0_xi(xi, pressure);
  ConstantCoefficient kappa(kKappa), mu(kMu);
  auto C_eff =
      IsotropicElasticTensorCoefficient::FromBulkModulus(dim, kappa, mu);
  BareElasticTensorCoefficient C_comp(dim, C_eff, p0_xi);
  RelabelledElasticTensorCoefficient C_ref(dim, C_comp, xi);
  TransformedMatrixFunctionCoefficient S_comp(
      dim, xi, [&pressure](const Vector& y, DenseMatrix& S) {
        S.SetSize(y.Size());
        S = 0.0;
        const double p = pressure(y);
        for (int i = 0; i < y.Size(); i++) {
          S(i, i) = -p;
        }
      });
  PullbackStressCoefficient S_ref(dim, S_comp, xi);
  JacobianCoefficient jac(xi);

  int n_body = 0;
  for (int e = 0; e < mesh.GetNE(); e += 3) {
    if (mesh.GetAttribute(e) != 1) {
      continue;
    }
    n_body++;
    auto* T = mesh.GetElementTransformation(e);
    const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(e));
    T->SetIntPoint(&ip);

    DenseMatrix A, B;
    rel.ElasticTensor().Eval(A, *T, ip);
    C_ref.Eval(B, *T, ip);
    A -= B;
    EXPECT_LT(A.MaxMaxNorm(), 1e-12);

    rel.EquilibriumStress().Eval(A, *T, ip);
    S_ref.Eval(B, *T, ip);
    A -= B;
    EXPECT_LT(A.MaxMaxNorm(), 1e-12);

    const double rho_ref = kRho * jac.Eval(*T, ip);
    EXPECT_NEAR(rel.Density().Eval(*T, ip), rho_ref, 1e-12);
  }
  ASSERT_GT(n_body, 0);
}
