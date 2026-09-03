#include "SelfGravitatingTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for spherical_harmonics.hpp: SurfaceHarmonics,
  HarmonicExpansionCoefficient and BoundaryHarmonicCoefficients, on the
  surface of the body SubMesh of the self-gravitating test meshes (order-2
  gmsh spheres/circles) and on the parent mesh's outer boundary.

  - Index map: Index(l, m) inverts Degree/Order; 2-D and 3-D sizes.
  - Orthonormality: the coefficients of the synthesised harmonic Y_i on the
    curved surface are e_i to the geometry's accuracy.
  - Against PoissonDtNOperator::HarmonicCoefficients on the outer boundary
    (index permutation; 2-D DtN coefficients are those of the unnormalised
    Fourier basis).
  - Radial component: u = Y_i n gives e_i, from a VectorCoefficient and from
    its interpolant on the vector space.
  - LoadVector against a BoundaryLFIntegrator of the same expansion.
  - Interior harmonic continuation carries the factor (r/R)^l.
*/

namespace {

using namespace self_grav_test;

int MaxDegree(int dim) { return dim == 2 ? 6 : 3; }
// Tolerances on the order-2 curved test meshes: boundary integrals of
// exact fields (geometry only), and of their order-2 interpolants.
double GeomTol(int dim) { return dim == 2 ? 1e-6 : 3e-3; }
double InterpTol(int dim) { return dim == 2 ? 1e-3 : 5e-2; }

struct HarmonicCase {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> body;
  H1_FECollection fec;
  std::unique_ptr<FiniteElementSpace> scalar, vector, phi;
  Array<int> surface, outer;
  int dim, L;

  explicit HarmonicCase(int d)
      : fec(2, d), dim(d), L(MaxDegree(d)) {
    parent = std::make_unique<Mesh>(MeshFile(dim).c_str(), 1, 1);
    body = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, BodyMarker(*parent)));
    scalar = std::make_unique<FiniteElementSpace>(body.get(), &fec);
    vector = std::make_unique<FiniteElementSpace>(body.get(), &fec, dim);
    phi = std::make_unique<FiniteElementSpace>(parent.get(), &fec);
    surface = SurfaceMarker(*body);
    outer = ExternalBoundaryMarker(parent.get());
  }
};

Vector Unit(int n, int i) {
  Vector e(n);
  e = 0.0;
  e[i] = 1.0;
  return e;
}

class SphericalHarmonicsTest : public testing::TestWithParam<int> {};

TEST_P(SphericalHarmonicsTest, IndexMap) {
  const int dim = GetParam();
  SurfaceHarmonics basis(dim, 4);
  EXPECT_EQ(basis.Size(), dim == 2 ? 9 : 25);
  for (int i = 0; i < basis.Size(); i++) {
    EXPECT_EQ(basis.Index(basis.Degree(i), basis.Order(i)), i);
    EXPECT_LE(std::abs(basis.Order(i)), basis.Degree(i));
  }
  Vector x(dim), Y;
  x = 0.3;
  x[dim - 1] = 0.9;
  basis.Eval(x, Y);
  EXPECT_EQ(Y.Size(), basis.Size());
  EXPECT_NEAR(Y[0], 1.0 / std::sqrt(dim == 2 ? 2.0 * M_PI : 4.0 * M_PI),
              1e-14);
  // Direction only.
  Vector Y2;
  x *= 3.0;
  basis.Eval(x, Y2);
  Y2 -= Y;
  EXPECT_LT(Y2.Normlinf(), 1e-13);
}

TEST_P(SphericalHarmonicsTest, Orthonormality) {
  HarmonicCase s(GetParam());
  BoundaryHarmonicCoefficients bhc(*s.scalar, s.surface, s.L,
                                   BoundaryHarmonicCoefficients::Component::Scalar);
  EXPECT_NEAR(bhc.Radius(), 1.0, 1e-3);
  const int n = bhc.Size();
  double err = 0.0;
  for (int i = 0; i < n; i++) {
    auto f = bhc.Expansion(Unit(n, i));
    Vector c;
    bhc.Coefficients(*f, c);
    for (int j = 0; j < n; j++) {
      err = std::max(err, std::abs(c[j] - (i == j ? 1.0 : 0.0)));
    }
    // Through the interpolant on the order-2 space: interpolation error on
    // top, worst for the highest degree.
    GridFunction g(s.scalar.get());
    g.ProjectCoefficient(*f);
    bhc.Coefficients(g, c);
    EXPECT_NEAR(c[i], 1.0, InterpTol(s.dim)) << "harmonic " << i;
  }
  EXPECT_LT(err, GeomTol(s.dim));
}

TEST_P(SphericalHarmonicsTest, MatchesDtNOperator) {
  HarmonicCase s(GetParam());
  const int dim = s.dim, L = s.L;
  PoissonDtNOperator dtn(s.phi.get(), L);
  dtn.Assemble();
  BoundaryHarmonicCoefficients bhc(*s.phi, s.outer, L,
                                   BoundaryHarmonicCoefficients::Component::Scalar,
                                   dtn.Centroid());
  // The DtN radius comes from the vertices, ours from the quadrature points.
  EXPECT_NEAR(bhc.Radius(), dtn.BoundaryRadius(), 1e-4);

  FunctionCoefficient f([dim](const Vector& x) {
    return std::exp(0.5 * x[0]) * (1.0 + 0.3 * x[1]) +
           (dim == 3 ? 0.7 * x[2] * x[0] : 0.0);
  });
  GridFunction g(s.phi.get());
  g.ProjectCoefficient(f);
  Vector c_dtn, c;
  dtn.HarmonicCoefficients(g, c_dtn);
  bhc.Coefficients(g, c);
  double scale = 0.0;
  for (int i = 0; i < c.Size(); i++) {
    scale = std::max(scale, std::abs(c[i]));
  }
  ASSERT_GT(scale, 0.0);
  // The DtN operator weights each quadrature point by its own radius; the
  // difference is the radius spread of the curved faces.
  const double tol = (dim == 2 ? 1e-6 : 1e-4) * scale;
  const auto& basis = bhc.Basis();
  if (dim == 2) {
    for (int k = 1; k <= L; k++) {
      EXPECT_NEAR(c_dtn[2 * (k - 1)] * std::sqrt(M_PI), c[basis.Index(k, k)],
                  tol);
      EXPECT_NEAR(c_dtn[2 * (k - 1) + 1] * std::sqrt(M_PI),
                  c[basis.Index(k, -k)], tol);
    }
  } else {
    EXPECT_NEAR(c_dtn[0], c[0], tol);
    for (int l = 1; l <= L; l++) {
      EXPECT_NEAR(c_dtn[l * l], c[basis.Index(l, 0)], tol);
      for (int m = 1; m <= l; m++) {
        EXPECT_NEAR(c_dtn[l * l + 2 * m - 1], c[basis.Index(l, m)],
                    tol);
        EXPECT_NEAR(c_dtn[l * l + 2 * m], c[basis.Index(l, -m)],
                    tol);
      }
    }
  }
}

TEST_P(SphericalHarmonicsTest, RadialComponent) {
  HarmonicCase s(GetParam());
  BoundaryHarmonicCoefficients bhc(*s.vector, s.surface, s.L,
                                   BoundaryHarmonicCoefficients::Component::Radial);
  const int n = bhc.Size(), dim = s.dim;
  const auto& basis = bhc.Basis();
  double err = 0.0, err_gf = 0.0;
  for (int i = 0; i < n; i++) {
    // u = Y_i(x^) x^ + a tangential part that must not contribute.
    VectorFunctionCoefficient u(dim, [&](const Vector& x, Vector& v) {
      Vector Y;
      basis.Eval(x, Y);
      const double r = x.Norml2();
      v = x;
      v *= Y[i] / r;
      // tangential: rotate x in the (0,1) plane.
      v[0] += -x[1] / r * 0.4;
      v[1] += x[0] / r * 0.4;
    });
    Vector c;
    bhc.Coefficients(u, c);
    for (int j = 0; j < n; j++) {
      err = std::max(err, std::abs(c[j] - (i == j ? 1.0 : 0.0)));
    }
    GridFunction g(s.vector.get());
    g.ProjectCoefficient(u);
    bhc.Coefficients(g, c);
    err_gf = std::max(err_gf, std::abs(c[i] - 1.0));
  }
  EXPECT_LT(err, GeomTol(dim));
  EXPECT_LT(err_gf, InterpTol(dim));
}

TEST_P(SphericalHarmonicsTest, LoadVectorMatchesLinearForm) {
  HarmonicCase s(GetParam());
  BoundaryHarmonicCoefficients bhc(*s.scalar, s.surface, s.L,
                                   BoundaryHarmonicCoefficients::Component::Scalar);
  const int n = bhc.Size();
  Vector c(n);
  for (int i = 0; i < n; i++) {
    c[i] = std::cos(1.3 * i + 0.2);
  }
  Vector b;
  bhc.LoadVector(c, b);
  auto f = bhc.Expansion(c);
  LinearForm lf(s.scalar.get());
  lf.AddBoundaryIntegrator(new BoundaryLFIntegrator(*f, 4, 4), s.surface);
  lf.Assemble();
  EXPECT_EQ(b.Size(), lf.Size());
  Vector d(b);
  d -= lf;
  EXPECT_GT(lf.Normlinf(), 0.0);
  EXPECT_LT(d.Normlinf(), 1e-7 * lf.Normlinf());  // different rules

  // Duality: g . b = R^{d-1} c . coefficients(g).
  GridFunction g(s.scalar.get());
  FunctionCoefficient gc([](const Vector& x) { return 1.0 + x[0] * x[1]; });
  g.ProjectCoefficient(gc);
  Vector cg;
  bhc.Coefficients(g, cg);
  EXPECT_NEAR(g * b, std::pow(bhc.Radius(), s.dim - 1) * (c * cg),
              1e-12 * std::abs(g * b));
}

TEST_P(SphericalHarmonicsTest, InteriorHarmonic) {
  HarmonicCase s(GetParam());
  const int L = s.L;
  SurfaceHarmonics basis(s.dim, L);
  Vector c(basis.Size());
  for (int i = 0; i < c.Size(); i++) {
    c[i] = 0.1 * (i + 1);
  }
  Vector centre(s.dim);
  centre = 0.0;
  HarmonicExpansionCoefficient surface(basis, c, centre, 1.0, false);
  HarmonicExpansionCoefficient interior(basis, c, centre, 1.0, true);
  FunctionCoefficient exact([&](const Vector& x) {
    Vector Y;
    basis.Eval(x, Y);
    const double r = x.Norml2();
    double f = 0.0;
    for (int i = 0; i < c.Size(); i++) {
      f += c[i] * Y[i] * std::pow(r, basis.Degree(i));
    }
    return f;
  });
  // At the quadrature points of the body's elements.
  double err = 0.0, err_surface = 0.0;
  for (int e = 0; e < s.body->GetNE(); e++) {
    auto* T = s.body->GetElementTransformation(e);
    const auto& ir = IntRules.Get(T->GetGeometryType(), 2);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const auto& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      err = std::max(err, std::abs(interior.Eval(*T, ip) - exact.Eval(*T, ip)));
      Vector x;
      T->Transform(ip, x);
      Vector Y;
      basis.Eval(x, Y);
      err_surface = std::max(err_surface, std::abs(surface.Eval(*T, ip) - (c * Y)));
    }
  }
  EXPECT_LT(err, 1e-13);
  EXPECT_LT(err_surface, 1e-13);
}

INSTANTIATE_TEST_SUITE_P(Harmonics, SphericalHarmonicsTest,
                         testing::Values(2, 3));

}  // namespace
