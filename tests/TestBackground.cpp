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

// The analytic buffer-taper rule: phi = x + t(r)(xi - x) with the cubic
// smoothstep. Exact equality with xi inside, the exact identity outside,
// the closed-form blend in between, and an independent differentiation
// path (the gradient of the nodal interpolant) agreeing at
// interpolation level.
TEST(Background, TaperedDiffeomorphismBlends) {
  const int dim = 2;
  Mesh mesh(MeshFile(dim).c_str(), 1, 1);

  const double c = 0.08;
  auto f = [c](double r) { return 1.0 + c * std::exp(-r * r); };
  auto df = [c](double r) { return -2.0 * r * c * std::exp(-r * r); };
  RadialDiffeomorphism xi(dim, f, df);
  const double r0 = 0.9, r1 = 1.15;
  TaperedDiffeomorphism phi(xi, r0, r1);

  int n_in = 0, n_mid = 0, n_out = 0;
  Vector x(dim), v, xiv(dim);
  DenseMatrix F, E(dim);
  for (int e = 0; e < mesh.GetNE(); e++) {
    auto* T = mesh.GetElementTransformation(e);
    const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(e));
    T->SetIntPoint(&ip);
    T->Transform(ip, x);
    const double r = x.Norml2();

    phi.Eval(v, *T, ip);
    phi.EvalGradient(F, *T, ip);
    EXPECT_GT(F.Det(), 0.0);

    // The independently coded blend.
    double t = 1.0, dt = 0.0;
    if (r >= r1) {
      t = 0.0;
    } else if (r > r0) {
      const double s = (r - r0) / (r1 - r0);
      t = 1.0 - s * s * (3.0 - 2.0 * s);
      dt = -6.0 * s * (1.0 - s) / (r1 - r0);
    }
    xiv = x;
    xiv *= f(r);
    for (int i = 0; i < dim; i++) {
      EXPECT_NEAR(v(i), x(i) + t * (xiv(i) - x(i)), 1e-13);
      for (int j = 0; j < dim; j++) {
        const double Fxi =
            (i == j ? f(r) : 0.0) + df(r) / r * x(i) * x(j);
        E(i, j) = (i == j ? 1.0 : 0.0) + t * (Fxi - (i == j ? 1.0 : 0.0)) +
                  dt / r * (xiv(i) - x(i)) * x(j);
        EXPECT_NEAR(F(i, j), E(i, j), 1e-13);
      }
    }
    (r < r0 ? n_in : (r > r1 ? n_out : n_mid))++;
  }
  ASSERT_GT(n_in, 0);
  ASSERT_GT(n_mid, 0);
  ASSERT_GT(n_out, 0);

  // Independent differentiation: the interpolant's discrete gradient.
  auto gd = Interpolate(phi, mesh);
  double max_dF = 0.0;
  for (int e = 0; e < mesh.GetNE(); e++) {
    auto* T = mesh.GetElementTransformation(e);
    const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(e));
    T->SetIntPoint(&ip);
    DenseMatrix Fa, Fd;
    phi.EvalGradient(Fa, *T, ip);
    gd.EvalGradient(Fd, *T, ip);
    Fa -= Fd;
    max_dF = std::max(max_dF, Fa.MaxMaxNorm());
  }
  EXPECT_LT(max_dF, 2e-2);
}

// The elliptic buffer-taper rule: a mapping non-trivial on the physical
// surface, extended harmonically through the buffer. Body values are the
// interpolant of xi, the buffer obeys the maximum principle, the
// displacement dies towards the DtN sphere, and the result is a
// diffeomorphism.
TEST(Background, HarmonicExtensionMapping) {
  const int dim = 2;
  Mesh mesh(MeshFile(dim).c_str(), 1, 1);
  Vector bb_min, bb_max;
  mesh.GetBoundingBox(bb_min, bb_max);
  const double r_out = bb_max.Normlinf();

  // Non-identity ON the surface r = 1 (|h| = c q(1)^2 there).
  const double c = 0.05;
  auto q = [](double r) { return 1.0 - r * r / 1.44; };
  auto f = [c, q](double r) { return 1.0 + c * q(r) * q(r); };
  auto df = [c, q](double r) {
    return c * 2.0 * q(r) * (-2.0 * r / 1.44);
  };
  RadialDiffeomorphism xi(dim, f, df);
  const double trace = c * q(1.0) * q(1.0);  // |h| at r = 1

  Array<int> body_attr({1}), buffer_attr({2});
  auto phi = NewHarmonicExtensionMapping(mesh, 2, xi, body_attr, buffer_attr);
  const GridFunction& h = phi.Displacement();

  double body_err = 0.0, buffer_max = 0.0, inner_buffer_max = 0.0;
  Vector x(dim), hv;
  for (int e = 0; e < mesh.GetNE(); e++) {
    auto* T = mesh.GetElementTransformation(e);
    const auto& ip = Geometries.GetCenter(mesh.GetElementGeometry(e));
    T->SetIntPoint(&ip);
    T->Transform(ip, x);
    const double r = x.Norml2();
    h.GetVectorValue(*T, ip, hv);

    DenseMatrix F;
    phi.EvalGradient(F, *T, ip);
    EXPECT_GT(F.Det(), 0.0);

    if (mesh.GetAttribute(e) == 1) {
      // h = (f(r) - 1) x on the body, to interpolation error.
      for (int i = 0; i < dim; i++) {
        body_err = std::max(body_err,
                            std::abs(hv(i) - (f(r) - 1.0) * x(i)));
      }
    } else {
      buffer_max = std::max(buffer_max, hv.Norml2());
      if (r < 0.5 * (1.0 + r_out)) {
        inner_buffer_max = std::max(inner_buffer_max, hv.Norml2());
      }
    }
  }
  EXPECT_LT(body_err, 1e-4);
  EXPECT_LT(buffer_max, 1.05 * trace);   // maximum principle
  EXPECT_GT(inner_buffer_max, 0.1 * trace);  // non-trivial extension

  // The displacement dies towards the DtN sphere.
  DenseMatrix pts(dim, 8);
  for (int i = 0; i < 8; i++) {
    const double th = 2.0 * kPi * i / 8.0 + 0.05;
    pts(0, i) = 0.999 * r_out * std::cos(th);
    pts(1, i) = 0.999 * r_out * std::sin(th);
  }
  Array<int> elem;
  Array<IntegrationPoint> ips;
  mesh.FindPoints(pts, elem, ips, false);
  int n_found = 0;
  for (int i = 0; i < 8; i++) {
    if (elem[i] < 0) {
      continue;
    }
    n_found++;
    h.GetVectorValue(elem[i], ips[i], hv);
    EXPECT_LT(hv.Norml2(), 0.05 * trace);
  }
  ASSERT_GT(n_found, 4);
}
