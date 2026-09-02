// Tests for TransformedDiffusionIntegrator (poisson.hpp): the pull-back of the
// Laplace form under a diffeomorphism xi(x). Two families of checks:
//
//  1. The three ways of specifying the mapping agree. For a radial map
//     xi = f(x) x with f affine, the scalar path (f given) and the matrix path
//     (a = J F^{-1} F^{-T} given analytically) must coincide to round-off at
//     every order; the vector path (xi given) joins them once xi = f x lies in
//     the trial space (order >= 2). This is the test that catches the wrong
//     index in the scalar-path Jacobian F(j,k) = x_j d_k f.
//
//  2. For an affine map the transformed form on the reference mesh equals the
//     ordinary DiffusionIntegrator assembled on the mapped mesh.
#include "TestCommon.hpp"

namespace {

Mesh SmallMesh(int dim, int elementType) {
  if (dim == 2) {
    return Mesh::MakeCartesian2D(
        6, 6, elementType == 0 ? Element::TRIANGLE : Element::QUADRILATERAL);
  }
  return Mesh::MakeCartesian3D(
      4, 4, 4, elementType == 0 ? Element::TETRAHEDRON : Element::HEXAHEDRON);
}

// Affine radial scale f(x) = f0 + g . x.
struct AffineRadialMap {
  real_t f0;
  Vector g;
  real_t f(const Vector& x) const { return f0 + (g * x); }
  // F = f I + x g^T, then a = det(F) F^{-1} F^{-T}.
  void a(const Vector& x, DenseMatrix& A) const {
    const int dim = x.Size();
    DenseMatrix F(dim);
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        F(j, k) = x(j) * g(k);
      }
      F(j, j) += f(x);
    }
    const real_t J = F.Det();
    F.Invert();
    A.SetSize(dim);
    MultABt(F, F, A);
    A *= J;
  }
};

class AffineRadialMatrixCoefficient : public MatrixCoefficient {
 public:
  AffineRadialMatrixCoefficient(int dim, const AffineRadialMap& map)
      : MatrixCoefficient(dim), map_(map), x_(dim) {}
  void Eval(DenseMatrix& K, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    T.Transform(ip, x_);
    map_.a(x_, K);
  }

 private:
  AffineRadialMap map_;
  Vector x_;
};

std::unique_ptr<SparseMatrix> Assemble(FiniteElementSpace& fes,
                                       BilinearFormIntegrator* integ) {
  BilinearForm a(&fes);
  a.AddDomainIntegrator(integ);
  a.Assemble();
  a.Finalize();
  return std::make_unique<SparseMatrix>(a.SpMat());
}

}  // namespace

class TransformedDiffusionTest
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

// 1. Scalar / vector / matrix specifications of the same radial map agree.
TEST_P(TransformedDiffusionTest, MappingPathsAgree) {
  const auto [dim, order, elementType] = GetParam();
  auto mesh = SmallMesh(dim, elementType);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec);

  AffineRadialMap map;
  map.f0 = 1.0;
  map.g.SetSize(dim);
  map.g(0) = 0.3;
  map.g(1) = -0.2;
  if (dim == 3) map.g(2) = 0.15;

  FunctionCoefficient f_coeff(
      [map](const Vector& x) -> real_t { return map.f(x); });
  RadialDiffeomorphismCoefficient xi_coeff(dim, f_coeff);
  AffineRadialMatrixCoefficient a_coeff(dim, map);

  auto A_scalar = Assemble(fes, new TransformedDiffusionIntegrator(f_coeff));
  auto A_vector = Assemble(fes, new TransformedDiffusionIntegrator(xi_coeff));
  auto A_matrix = Assemble(fes, new TransformedDiffusionIntegrator(a_coeff));

  const real_t scale = A_matrix->MaxNorm();
  ASSERT_GT(scale, 0.0);
  const real_t tol = 1e-12 * scale;

  // f is affine, so its nodal interpolant is exact at every order: the scalar
  // path must reproduce the analytic a(x) to round-off.
  EXPECT_LT(MaxDiff(*A_scalar, *A_matrix), tol);

  // xi = f x is quadratic; its nodal interpolant is exact for order >= 2.
  if (order >= 2) {
    EXPECT_LT(MaxDiff(*A_vector, *A_matrix), tol);
  }

  // The mapped Laplacian must still be symmetric.
  std::unique_ptr<SparseMatrix> At(Transpose(*A_scalar));
  EXPECT_LT(MaxDiff(*A_scalar, *At), tol);
}

// 2. Affine maps: pull-back on the reference mesh == Laplacian on the mapped
//    mesh. Uses a uniform scaling through the scalar path and a general
//    affine map through the vector path.
TEST_P(TransformedDiffusionTest, MatchesMappedMesh) {
  const auto [dim, order, elementType] = GetParam();
  H1_FECollection fec(order, dim);

  // (a) xi = c x through the scalar path.
  {
    const real_t c = 1.7;
    auto mesh = SmallMesh(dim, elementType);
    FiniteElementSpace fes(&mesh, &fec);
    ConstantCoefficient c_coeff(c);
    auto A_ref = Assemble(fes, new TransformedDiffusionIntegrator(c_coeff));

    auto mapped = SmallMesh(dim, elementType);
    mapped.SetCurvature(order);
    VectorFunctionCoefficient scale(dim, [c](const Vector& x, Vector& y) {
      y = x;
      y *= c;
    });
    mapped.Transform(scale);
    FiniteElementSpace fes_mapped(&mapped, &fec);
    ConstantCoefficient one(1.0);
    auto A_mapped = Assemble(fes_mapped, new DiffusionIntegrator(one));

    EXPECT_LT(MaxDiff(*A_ref, *A_mapped), 1e-12 * A_mapped->MaxNorm());
  }

  // (b) xi = M x + b through the vector path, M a fixed non-symmetric matrix
  //     with positive determinant.
  {
    DenseMatrix M(dim);
    M = 0.0;
    for (int i = 0; i < dim; i++) M(i, i) = 1.0 + 0.2 * i;
    M(0, 1) = 0.4;
    M(1, 0) = -0.1;
    if (dim == 3) {
      M(0, 2) = 0.25;
      M(2, 1) = 0.3;
    }
    ASSERT_GT(M.Det(), 0.0);
    Vector b(dim);
    b = 0.5;

    VectorFunctionCoefficient xi(dim, [M, b](const Vector& x, Vector& y) {
      y.SetSize(x.Size());
      M.Mult(x, y);
      y += b;
    });

    auto mesh = SmallMesh(dim, elementType);
    FiniteElementSpace fes(&mesh, &fec);
    auto A_ref = Assemble(fes, new TransformedDiffusionIntegrator(xi));

    auto mapped = SmallMesh(dim, elementType);
    mapped.SetCurvature(order);
    mapped.Transform(xi);
    FiniteElementSpace fes_mapped(&mapped, &fec);
    ConstantCoefficient one(1.0);
    auto A_mapped = Assemble(fes_mapped, new DiffusionIntegrator(one));

    EXPECT_LT(MaxDiff(*A_ref, *A_mapped), 1e-12 * A_mapped->MaxNorm());
  }
}

INSTANTIATE_TEST_SUITE_P(
    TransformedDiffusion, TransformedDiffusionTest,
    ::testing::Combine(::testing::Values(2, 3), ::testing::Values(1, 2, 3),
                       ::testing::Values(0, 1)));
