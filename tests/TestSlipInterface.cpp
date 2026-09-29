#include <array>
#include <cmath>
#include <functional>
#include <vector>

#include "gtest/gtest.h"

/*
  Finite-difference verification of the slip-interface second variation
  (doc/slip_interface.tex): the interface bilinear form

      B_Sigma(v, v) = - oint_Sigma pi nu . grad_Sigma(v_s + v_f)[s] dS,
      s = -F_e^{-1} [v],   nu = cof(F_e) N,

  is pinned (sign, factors, polarisation) against the EXACT energy of a
  constrained broken motion. No finite elements anywhere: the slip
  family is exact at finite epsilon (rotations of the circular
  referential interface), all fields are analytic, and every integral
  is high-order quadrature, so the comparison is limited only by the
  FD step and quadrature resolution.

  Construction (2-D, gravity-free, constant-referential-pressure
  equilibria -- exact by the Piola identity div cof F = 0):
    - referential interface Sigma = circle radius rc; fluid inside,
      solid outside with fields supported below r = 0.9 (no outer
      boundary terms);
    - solid energy W = S_e : E_rel + (1/2) E_rel : C : E_rel with
      E_rel = (F^T F - F_e^T F_e)/2 and S_e = -pi J_e C_e^{-1} (the
      pullback of -pi 1), so the equilibrium stress is -pi cof F_e,
      divergence-free, with interface traction -pi nu;
    - fluid energy V(x, J) = -pi (J - J_e) + (kappa/2)(J - J_e)^2;
    - slip family sigma_eps = rotation of Sigma by the angle
      eps a(theta); extended into the fluid by R_eps = rotation by
      eps a(theta) g(r) with g = (r/rc)^3 (analytic in x), and
      phi_f = phi_s o R_eps: the attachment holds EXACTLY at finite
      eps, and the family's own curvature reproduces the sigma_2
      normal term of the derivation.

  The test asserts d^2E/deps^2 (numerical) = Q_vol(v_s, v_f) + B_Sigma
  to FD accuracy, that B_Sigma is genuinely load-bearing (dropping or
  flipping it breaks the match by orders of magnitude), and the
  polarised identity via a mixed second difference of a two-parameter
  family. Case 1: phi_e = id. Case 2: a curved radial phi_e
  (nu != N, exercising the full Nanson and trace-derivative content).
*/

namespace {

using Vec2 = std::array<double, 2>;
using Mat2 = std::array<std::array<double, 2>, 2>;

constexpr double kPi = 3.14159265358979323846;
constexpr double rc = 0.6;      // referential interface radius
constexpr double rOut = 0.9;    // support limit of the solid field
constexpr double piP = 0.3;     // referential pressure
constexpr double lam = 1.1, mu = 0.7, kap = 2.0;

Mat2 MatMul(const Mat2& A, const Mat2& B) {
  Mat2 C{};
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j];
    }
  }
  return C;
}
Mat2 MatT(const Mat2& A) { return {{{A[0][0], A[1][0]}, {A[0][1], A[1][1]}}}; }
double Det(const Mat2& A) { return A[0][0] * A[1][1] - A[0][1] * A[1][0]; }
double Tr(const Mat2& A) { return A[0][0] + A[1][1]; }
Mat2 Inv(const Mat2& A) {
  const double d = Det(A);
  return {{{A[1][1] / d, -A[0][1] / d}, {-A[1][0] / d, A[0][0] / d}}};
}
Mat2 Add(const Mat2& A, const Mat2& B, double b = 1.0) {
  Mat2 C{};
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      C[i][j] = A[i][j] + b * B[i][j];
    }
  }
  return C;
}
double Dot(const Mat2& A, const Mat2& B) {
  double s = 0.0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      s += A[i][j] * B[i][j];
    }
  }
  return s;
}
constexpr Mat2 kI{{{1.0, 0.0}, {0.0, 1.0}}};

// --- The equilibrium mapping (case-dependent) -----------------------------
// phi_e = f(r) x with f = f0 - f2 r^2 (f2 = 0 gives the identity).
struct EquilibriumMap {
  double f0 = 1.0, f2 = 0.0;
  Vec2 Map(const Vec2& x) const {
    const double f = f0 - f2 * (x[0] * x[0] + x[1] * x[1]);
    return {f * x[0], f * x[1]};
  }
  Mat2 Grad(const Vec2& x) const {
    const double f = f0 - f2 * (x[0] * x[0] + x[1] * x[1]);
    Mat2 F = {{{f, 0.0}, {0.0, f}}};
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        F[i][j] += -2.0 * f2 * x[i] * x[j];
      }
    }
    return F;
  }
};

// --- The independent solid-side field V (global analytic) -----------------
// V = h(r) b(x), h = ((rOut^2 - r^2)/(rOut^2 - rc^2))^2 (polynomial in x).
struct Field {
  double c1, c2, c3, c4, c5;  // generic coefficients of b(x)
  Vec2 Eval(const Vec2& x) const {
    const double q = (rOut * rOut - x[0] * x[0] - x[1] * x[1]) /
                     (rOut * rOut - rc * rc);
    if (q <= 0.0) {
      return {0.0, 0.0};
    }
    const double h = q * q;
    return {h * (c1 + c2 * x[1] + c3 * x[0] * x[0]),
            h * (c4 * x[0] + c5 * x[0] * x[1])};
  }
  Mat2 Grad(const Vec2& x) const {
    const double denom = rOut * rOut - rc * rc;
    const double q = (rOut * rOut - x[0] * x[0] - x[1] * x[1]) / denom;
    if (q <= 0.0) {
      return Mat2{};
    }
    const double h = q * q;
    const Vec2 dh = {-4.0 * q * x[0] / denom, -4.0 * q * x[1] / denom};
    const Vec2 b = {c1 + c2 * x[1] + c3 * x[0] * x[0],
                    c4 * x[0] + c5 * x[0] * x[1]};
    const Mat2 db = {{{2.0 * c3 * x[0], c2}, {c4 + c5 * x[1], c5 * x[0]}}};
    Mat2 G{};
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        G[i][j] = dh[j] * b[i] + h * db[i][j];
      }
    }
    return G;
  }
};

// --- The slip flow --------------------------------------------------------
// Rotation by the angle chi(x) = A(x), A = (a1 cos + a2 sin 2theta) (r/rc)^3
// as a polynomial-in-x expression: A = [a1 x r^2 + a2 (2 x y) r] / rc^3.
// (r = |x|; the a2 term uses sin(2theta) r^3 = 2 x y r, smooth away from 0
// and O(r^3) at 0.)
struct SlipFlow {
  double a1, a2;
  double A(const Vec2& x) const {
    const double r2 = x[0] * x[0] + x[1] * x[1];
    const double r = std::sqrt(r2);
    return (a1 * x[0] * r2 + a2 * 2.0 * x[0] * x[1] * r) / (rc * rc * rc);
  }
  Vec2 GradA(const Vec2& x) const {
    const double r2 = x[0] * x[0] + x[1] * x[1];
    const double r = std::sqrt(r2);
    Vec2 g = {a1 * (3.0 * x[0] * x[0] + x[1] * x[1]),
              a1 * 2.0 * x[0] * x[1]};
    if (r > 1e-14) {
      g[0] += a2 * (2.0 * x[1] * r + 2.0 * x[0] * x[0] * x[1] / r);
      g[1] += a2 * (2.0 * x[0] * r + 2.0 * x[0] * x[1] * x[1] / r);
    }
    g[0] /= rc * rc * rc;
    g[1] /= rc * rc * rc;
    return g;
  }
  // R_eps(x) = Rot(eps A(x)) x and its gradient.
  Vec2 Map(double eps, const Vec2& x) const {
    const double c = std::cos(eps * A(x)), s = std::sin(eps * A(x));
    return {c * x[0] - s * x[1], s * x[0] + c * x[1]};
  }
  Mat2 Grad(double eps, const Vec2& x) const {
    const double chi = eps * A(x);
    const double c = std::cos(chi), s = std::sin(chi);
    const Mat2 R = {{{c, -s}, {s, c}}};
    const Vec2 g = GradA(x);
    const Vec2 zx = {-x[1], x[0]};  // z cross x
    Mat2 M = kI;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        M[i][j] += zx[i] * eps * g[j];
      }
    }
    return MatMul(R, M);
  }
};

// --- Energies -------------------------------------------------------------
struct Model {
  EquilibriumMap phi;

  double SolidDensity(const Vec2& x, const Mat2& F) const {
    const Mat2 Fe = phi.Grad(x);
    const Mat2 Ce = MatMul(MatT(Fe), Fe);
    const double Je = Det(Fe);
    const Mat2 E = Add(MatMul(MatT(F), F), Ce, -1.0);  // 2 E_rel
    // W = S_e : E_rel + (lam/2) tr(E)^2 + mu |E|^2, S_e = -pi Je Ce^{-1}.
    const Mat2 Se = Inv(Ce);
    const double SeE = -piP * Je * 0.5 * Dot(Se, E);
    const double trE = 0.5 * Tr(E);
    double EE = 0.0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        EE += 0.25 * E[i][j] * E[i][j];
      }
    }
    return SeE + 0.5 * lam * trE * trE + mu * EE;
  }
  double FluidDensity(const Vec2& x, const Mat2& F) const {
    const double J = Det(F);
    const double Je = Det(phi.Grad(x));
    return -piP * (J - Je) + 0.5 * kap * (J - Je) * (J - Je);
  }

  // Exact energy of the constrained two-parameter family.
  double Energy(double e1, double e2, const Field& V1, const Field& V2,
                const SlipFlow& S1, const SlipFlow& S2, int Nr,
                int Nt) const {
    // Combined solid path phi_s = phi_e + e1 V1 + e2 V2; combined slip
    // rotation angle e1 A1 + e2 A2 (a valid exact family).
    auto rot = [&](const Vec2& x) -> Vec2 {
      const double chi = e1 * S1.A(x) + e2 * S2.A(x);
      const double c = std::cos(chi), s = std::sin(chi);
      return {c * x[0] - s * x[1], s * x[0] + c * x[1]};
    };
    auto drot = [&](const Vec2& x) -> Mat2 {
      const double chi = e1 * S1.A(x) + e2 * S2.A(x);
      const double c = std::cos(chi), s = std::sin(chi);
      const Mat2 R = {{{c, -s}, {s, c}}};
      const Vec2 g1 = S1.GradA(x), g2 = S2.GradA(x);
      const Vec2 zx = {-x[1], x[0]};
      Mat2 M = kI;
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          M[i][j] += zx[i] * (e1 * g1[j] + e2 * g2[j]);
        }
      }
      return MatMul(R, M);
    };

    auto Fs = [&](const Vec2& x) -> Mat2 {
      return Add(Add(phi.Grad(x), V1.Grad(x), e1), V2.Grad(x), e2);
    };

    double E = 0.0;
    // Solid annulus rc..rOut.
    for (int i = 0; i < Nr; i++) {
      const double r = rc + (rOut - rc) * (i + 0.5) / Nr;
      const double wr = (rOut - rc) / Nr * r;
      for (int j = 0; j < Nt; j++) {
        const double th = 2.0 * kPi * (j + 0.5) / Nt;
        const Vec2 x = {r * std::cos(th), r * std::sin(th)};
        E += wr * (2.0 * kPi / Nt) * SolidDensity(x, Fs(x));
      }
    }
    // Fluid disc 0..rc: phi_f = phi_s o R.
    for (int i = 0; i < Nr; i++) {
      const double r = rc * (i + 0.5) / Nr;
      const double wr = rc / Nr * r;
      for (int j = 0; j < Nt; j++) {
        const double th = 2.0 * kPi * (j + 0.5) / Nt;
        const Vec2 x = {r * std::cos(th), r * std::sin(th)};
        const Vec2 y = rot(x);
        const Mat2 Ff = MatMul(Fs(y), drot(x));
        E += wr * (2.0 * kPi / Nt) * FluidDensity(x, Ff);
      }
    }
    return E;
  }

  // Predicted volume Hessian (the Section-3 operators, per region, with
  // each region's own first-order field).
  double SolidHess(const Vec2& x, const Mat2& Dv) const {
    const Mat2 Fe = phi.Grad(x);
    const Mat2 Ce = MatMul(MatT(Fe), Fe);
    const double Je = Det(Fe);
    // material: lam tr(M)^2 + 2 mu |M|^2, M = sym(Fe^T Dv)
    const Mat2 FtD = MatMul(MatT(Fe), Dv);
    const Mat2 M = {{{FtD[0][0], 0.5 * (FtD[0][1] + FtD[1][0])},
                     {0.5 * (FtD[0][1] + FtD[1][0]), FtD[1][1]}}};
    const double mat = lam * Tr(M) * Tr(M) + 2.0 * mu * Dot(M, M);
    // geometric: S_e : (Dv^T Dv) = -pi Je tr(Ce^{-1} Dv^T Dv)
    const Mat2 G = MatMul(MatT(Dv), Dv);
    const double geo = -piP * Je * Dot(Inv(Ce), G);
    return mat + geo;
  }
  double FluidHess(const Vec2& x, const Mat2& Dv) const {
    const Mat2 Fe = phi.Grad(x);
    const double Je = Det(Fe);
    const Mat2 H = MatMul(Inv(Fe), Dv);
    const double trH = Tr(H);
    // tr(H^2) = sum_ij H[i][j] H[j][i]
    const double trHH =
        H[0][0] * H[0][0] + 2.0 * H[0][1] * H[1][0] + H[1][1] * H[1][1];
    return kap * Je * Je * trH * trH - piP * Je * (trH * trH - trHH);
  }
};

// The first-order fields of the family: v_s = V; v_f = V + F_e s_ext with
// s_ext the slip-flow generator, s_ext = A(x) (z cross x).
Vec2 SExt(const SlipFlow& S, const Vec2& x) {
  const double A = S.A(x);
  return {-A * x[1], A * x[0]};
}
Mat2 DSExt(const SlipFlow& S, const Vec2& x) {
  const Vec2 g = S.GradA(x);
  const double A = S.A(x);
  Mat2 G = {{{0.0, -A}, {A, 0.0}}};
  const Vec2 zx = {-x[1], x[0]};
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      G[i][j] += zx[i] * g[j];
    }
  }
  return G;
}

// v_f and its gradient (chain rule at eps = 0: v_f = V + Dphi_e[s_ext]
// evaluated referentially; its gradient D(V + F_e s_ext)).
Vec2 VFluid(const Model& m, const Field& V, const SlipFlow& S,
            const Vec2& x) {
  const Vec2 v = V.Eval(x);
  const Vec2 se = SExt(S, x);
  const Mat2 Fe = m.phi.Grad(x);
  return {v[0] + Fe[0][0] * se[0] + Fe[0][1] * se[1],
          v[1] + Fe[1][0] * se[0] + Fe[1][1] * se[1]};
}
Mat2 DVFluid(const Model& m, const Field& V, const SlipFlow& S,
             const Vec2& x, double h = 1e-6) {
  // D(F_e s_ext) by an accurate central difference of the analytic
  // product (avoids hand-coding D^2 phi_e; h-error ~1e-12).
  Mat2 G = V.Grad(x);
  for (int j = 0; j < 2; j++) {
    Vec2 xp = x, xm = x;
    xp[j] += h;
    xm[j] -= h;
    const Vec2 sp = SExt(S, xp), sm = SExt(S, xm);
    const Mat2 Fp = m.phi.Grad(xp), Fm = m.phi.Grad(xm);
    for (int i = 0; i < 2; i++) {
      const double fp = Fp[i][0] * sp[0] + Fp[i][1] * sp[1];
      const double fm = Fm[i][0] * sm[0] + Fm[i][1] * sm[1];
      G[i][j] += (fp - fm) / (2.0 * h);
    }
  }
  return G;
}

// Predicted quadratic forms.
double QVol(const Model& m, const Field& Va, const SlipFlow& Sa,
            const Field& Vb, const SlipFlow& Sb, int Nr, int Nt) {
  // Polarised volume Hessian via the quadratic-form parallelogram.
  auto quad = [&](const Field& V, const SlipFlow& S, double sgnb,
                  const Field& V2, const SlipFlow& S2) {
    double E = 0.0;
    for (int i = 0; i < Nr; i++) {
      const double r = rc + (rOut - rc) * (i + 0.5) / Nr;
      const double wr = (rOut - rc) / Nr * r;
      for (int j = 0; j < Nt; j++) {
        const double th = 2.0 * kPi * (j + 0.5) / Nt;
        const Vec2 x = {r * std::cos(th), r * std::sin(th)};
        Mat2 D = Add(V.Grad(x), V2.Grad(x), sgnb);
        E += wr * (2.0 * kPi / Nt) * m.SolidHess(x, D);
      }
    }
    for (int i = 0; i < Nr; i++) {
      const double r = rc * (i + 0.5) / Nr;
      const double wr = rc / Nr * r;
      for (int j = 0; j < Nt; j++) {
        const double th = 2.0 * kPi * (j + 0.5) / Nt;
        const Vec2 x = {r * std::cos(th), r * std::sin(th)};
        Mat2 D = Add(DVFluid(m, V, S, x), DVFluid(m, V2, S2, x), sgnb);
        E += wr * (2.0 * kPi / Nt) * m.FluidHess(x, D);
      }
    }
    return E;
  };
  // B(a,b) = [Q(a+b) - Q(a-b)]/4.
  return 0.25 * (quad(Va, Sa, 1.0, Vb, Sb) - quad(Va, Sa, -1.0, Vb, Sb));
}

double BSigma(const Model& m, const Field& Va, const SlipFlow& Sa,
              const Field& Vb, const SlipFlow& Sb, int Nt) {
  // B_Sigma(a, b) = -1/2 oint pi { nu . d/dth(vs_a + vf_a) * s_b(th)
  //                              + nu . d/dth(vs_b + vf_b) * s_a(th) } dth,
  // with the tangential slip parameter along theta: s = a(th) d(pos)/dth,
  // so grad_Sigma(.)[s] = a(th) d(.)/dth, and a(th) = A(x(th)) since
  // g(rc) = 1.
  auto sum_trace = [&](const Field& V, const SlipFlow& S,
                       double th) -> Vec2 {
    const Vec2 x = {rc * std::cos(th), rc * std::sin(th)};
    const Vec2 vs = V.Eval(x);
    const Vec2 vf = VFluid(m, V, S, x);
    return {vs[0] + vf[0], vs[1] + vf[1]};
  };
  double B = 0.0;
  const int N = Nt;
  const double dth = 2.0 * kPi / N;
  for (int j = 0; j < N; j++) {
    const double th = j * dth;
    const Vec2 x = {rc * std::cos(th), rc * std::sin(th)};
    // nu = cof(F_e) N, N = outward radial unit.
    const Mat2 Fe = m.phi.Grad(x);
    const Mat2 cof = {{{Fe[1][1], -Fe[1][0]}, {-Fe[0][1], Fe[0][0]}}};
    const Vec2 Nref = {std::cos(th), std::sin(th)};
    const Vec2 nu = {cof[0][0] * Nref[0] + cof[0][1] * Nref[1],
                     cof[1][0] * Nref[0] + cof[1][1] * Nref[1]};
    // centered d/dth of the trace sums: h independent of the grid (the
    // traces are analytic in theta), so the derivative is ~1e-10 exact.
    auto ddth = [&](const Field& V, const SlipFlow& S) -> Vec2 {
      const double h = 1e-5;
      const Vec2 p = sum_trace(V, S, th + h), q = sum_trace(V, S, th - h);
      return {(p[0] - q[0]) / (2.0 * h), (p[1] - q[1]) / (2.0 * h)};
    };
    const Vec2 da = ddth(Va, Sa), db = ddth(Vb, Sb);
    const double aa = Sa.A(x), ab = Sb.A(x);
    // dS = rc dtheta on the referential circle.
    B += -0.5 * piP *
         ((nu[0] * da[0] + nu[1] * da[1]) * ab +
          (nu[0] * db[0] + nu[1] * db[1]) * aa) *
         rc * dth;
  }
  return B;
}

void RunCase(const EquilibriumMap& phi, const char* label) {
  Model m{phi};
  const Field V1{0.31, 0.62, -0.41, 0.53, 0.27};
  const Field V2{-0.22, 0.35, 0.18, -0.47, 0.39};
  const SlipFlow S1{0.8, 0.0};
  const SlipFlow S2{0.0, 0.6};

  const int Nr = 400, Nt = 600;
  const double eps = 2e-4;

  // Diagonal: d^2/deps^2 of the exact energy vs Q_vol + B_Sigma.
  const double Ep = m.Energy(eps, 0.0, V1, V2, S1, S2, Nr, Nt);
  const double E0 = m.Energy(0.0, 0.0, V1, V2, S1, S2, Nr, Nt);
  const double Em = m.Energy(-eps, 0.0, V1, V2, S1, S2, Nr, Nt);
  const double d2 = (Ep - 2.0 * E0 + Em) / (eps * eps);

  const double qvol = QVol(m, V1, S1, V1, S1, Nr, Nt);
  const double bsig = BSigma(m, V1, S1, V1, S1, Nt);
  const double scale = std::abs(qvol) + std::abs(bsig) + 1e-30;

  SCOPED_TRACE(label);
  // The match, and that B_Sigma is load-bearing.
  EXPECT_NEAR(d2, qvol + bsig, 2e-4 * scale)
      << "d2=" << d2 << " qvol=" << qvol << " B=" << bsig;
  EXPECT_GT(std::abs(bsig), 50.0 * std::abs(d2 - qvol - bsig))
      << "interface term not resolved: B=" << bsig
      << " residual=" << d2 - qvol - bsig;

  // Polarisation: mixed second difference vs the polarised forms.
  const double Epp = m.Energy(eps, eps, V1, V2, S1, S2, Nr, Nt);
  const double Epm = m.Energy(eps, -eps, V1, V2, S1, S2, Nr, Nt);
  const double Emp = m.Energy(-eps, eps, V1, V2, S1, S2, Nr, Nt);
  const double Emm = m.Energy(-eps, -eps, V1, V2, S1, S2, Nr, Nt);
  const double d2mix = (Epp - Epm - Emp + Emm) / (4.0 * eps * eps);
  // The mixed second derivative of E equals the POLARISED Hessian
  // directly (E is quadratic to leading order), so no factor of two.
  const double qmix = QVol(m, V1, S1, V2, S2, Nr, Nt);
  const double bmix = BSigma(m, V1, S1, V2, S2, Nt);
  const double scale2 = std::abs(qmix) + std::abs(bmix) + 1e-30;
  EXPECT_NEAR(d2mix, qmix + bmix, 3e-4 * scale2)
      << "d2mix=" << d2mix << " qmix=" << qmix << " Bmix=" << bmix;
}

}  // namespace

TEST(SlipInterface, SecondVariationIdentity) {
  RunCase(EquilibriumMap{1.0, 0.0}, "phi_e = id");
}

TEST(SlipInterface, SecondVariationMapped) {
  // f = 1.15 - (0.15/0.81) r^2: curved radial map, f(0.9) = 1 - ish,
  // J_e and nu genuinely non-trivial on Sigma.
  RunCase(EquilibriumMap{1.15, 0.15 / 0.81}, "phi_e radial");
}
