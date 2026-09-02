#include "TestCommon.hpp"

/*
  Element- and matrix-level tests for ElasticTensorIntegrator (design doc
  doc/anisotropic_elasticity_design.md, section 4, tests 5-9).

  - Isotropic tensor: element matrices equal mfem::ElasticityIntegrator's on
    a non-affine mesh, every element type and order.
  - Symmetry, non-negativity, and the rigid modes in the null space of the
    assembled matrix for a TI material with a spatially varying axis.
  - Energy patch test: for u = G x on an affine mesh with a constant TI
    tensor, u^T A u / 2 = |Omega| eps^ C eps^ / 2.
  - Rotation covariance: rotating mesh and axis transforms element
    matrices by the block rotation.
  - 2-D/3-D consistency: a plane-strain 2-D problem and an extruded 3-D
    slab give the same in-plane energy per unit thickness.
*/

namespace {

using Param = std::tuple<int, int, int>;  // (dim, elementType, order)

Mesh SmallMesh(int dim, int elementType, int n = 3) {
  return dim == 2
             ? Mesh::MakeCartesian2D(n, n,
                                     elementType == 0 ? Element::TRIANGLE
                                                      : Element::QUADRILATERAL)
             : Mesh::MakeCartesian3D(n, n, n,
                                     elementType == 0 ? Element::TETRAHEDRON
                                                      : Element::HEXAHEDRON);
}

void Jiggle(const Vector& x, Vector& y) {
  y = x;
  const double s = 0.08;
  y[0] += s * std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
  y[1] += s * std::cos(0.5 * M_PI * x[0]) * std::sin(M_PI * x[1]);
  if (x.Size() == 3) {
    y[2] += s * std::sin(M_PI * x[2]) * std::cos(M_PI * x[0]);
  }
}

double MaxDiff(const DenseMatrix& A, const DenseMatrix& B) {
  if (A.Height() != B.Height() || A.Width() != B.Width() || A.Height() == 0) {
    return std::numeric_limits<double>::infinity();
  }
  DenseMatrix D(A);
  D -= B;
  return D.MaxMaxNorm();
}

class ElasticTensorIntegratorTest : public testing::TestWithParam<Param> {
 protected:
  void SetUp() override {
    std::tie(dim, elementType, order) = GetParam();
    n = SymmetricTensorBasis::Size(dim);
  }
  int dim = 2, elementType = 0, order = 1, n = 3;
};

TEST_P(ElasticTensorIntegratorTest, IsotropicMatchesMfem) {
  auto mesh = SmallMesh(dim, elementType);
  mesh.SetCurvature(2);
  mesh.Transform(Jiggle);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);

  FunctionCoefficient lambda([](const Vector& x) { return 1.0 + 0.5 * x[0]; });
  FunctionCoefficient mu(
      [](const Vector& x) { return 0.7 + 0.3 * x[x.Size() - 1]; });
  IsotropicElasticTensorCoefficient iso(dim, lambda, mu);

  ElasticityIntegrator ref(lambda, mu);
  ElasticTensorIntegrator ours(iso);
  DenseMatrix A, B;
  for (int e = 0; e < mesh.GetNE(); e++) {
    auto* T = mesh.GetElementTransformation(e);
    ref.AssembleElementMatrix(*fes.GetFE(e), *T, A);
    ours.AssembleElementMatrix(*fes.GetFE(e), *T, B);
    ASSERT_EQ(A.Height(), B.Height());
    EXPECT_LT(MaxDiff(A, B), 1e-13 * A.MaxMaxNorm());
  }
}

TEST_P(ElasticTensorIntegratorTest, SymmetryAndRigidModes) {
  // Isoparametric geometry, so that the rigid rotation omega x X(xi) is
  // representable in the displacement space on the curved elements.
  auto mesh = SmallMesh(dim, elementType);
  mesh.SetCurvature(order);
  mesh.Transform(Jiggle);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);

  ConstantCoefficient A(3.1), C(2.7), F(1.1), L(0.9), N(1.2);
  Vector x0(dim);
  x0 = -0.3;
  RadialUnitVectorCoefficient axis(dim, x0);
  TransverselyIsotropicElasticTensorCoefficient ti(dim, A, C, F, L, N, axis);

  BilinearForm a(&fes);
  a.AddDomainIntegrator(new ElasticTensorIntegrator(ti));
  a.Assemble();
  a.Finalize();
  std::unique_ptr<SparseMatrix> Kt(Transpose(a.SpMat()));
  std::unique_ptr<SparseMatrix> D(Add(1.0, a.SpMat(), -1.0, *Kt));
  const double scale = a.SpMat().MaxNorm();
  EXPECT_LT(D->MaxNorm(), 1e-12 * scale);

  // Non-negative quadratic form.
  for (int trial = 0; trial < 5; trial++) {
    auto x = RandomVector(fes.GetVSize());
    Vector y(x.Size());
    a.SpMat().Mult(x, y);
    EXPECT_GT(x * y, -1e-12 * scale * (x * x));
  }

  // Rigid modes in the null space.
  const int nrot = dim == 2 ? 1 : 3;
  for (int c = 0; c < dim + nrot; c++) {
    GridFunction g(&fes);
    if (c < dim) {
      RigidTranslation t(dim, c);
      g.ProjectCoefficient(t);
    } else {
      // RigidRotation's 2-D rotation is component 2 (about z).
      RigidRotation r(dim, dim == 2 ? 2 : c - dim);
      g.ProjectCoefficient(r);
    }
    Vector y(g.Size());
    a.SpMat().Mult(g, y);
    EXPECT_LT(y.Normlinf(), 1e-11 * scale * g.Normlinf()) << "mode " << c;
  }
}

TEST_P(ElasticTensorIntegratorTest, EnergyPatchTest) {
  auto mesh = SmallMesh(dim, elementType);  // affine, |Omega| = 1
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);

  ConstantCoefficient A(3.1), C(2.7), F(1.1), L(0.9), N(1.2);
  auto axis_v = RandomVector(dim);
  VectorConstantCoefficient axis(axis_v);
  TransverselyIsotropicElasticTensorCoefficient ti(dim, A, C, F, L, N, axis);
  BilinearForm a(&fes);
  a.AddDomainIntegrator(new ElasticTensorIntegrator(ti));
  a.Assemble();
  a.Finalize();

  auto G = RandomMatrix(dim);
  VectorFunctionCoefficient u_coef(
      dim, [&](const Vector& x, Vector& u) { G.Mult(x, u); });
  GridFunction u(&fes);
  u.ProjectCoefficient(u_coef);
  Vector Au(u.Size());
  a.SpMat().Mult(u, Au);
  const double energy = 0.5 * (u * Au);

  // eps^ C eps^ / 2 with eps = sym(G).
  Vector eps_hat(n);
  for (int s = 0; s < n; s++) {
    int j, k;
    SymmetricTensorBasis::Component(dim, s, j, k);
    eps_hat[s] = SymmetricTensorBasis::Scale(j, k) * 0.5 * (G(j, k) + G(k, j));
  }
  Vector nrm(axis_v);
  nrm /= nrm.Norml2();
  DenseMatrix Cm;
  TransverselyIsotropicElasticTensorCoefficient::Build(dim, 3.1, 2.7, 1.1, 0.9,
                                                       1.2, nrm, Cm);
  Vector Ce(n);
  Cm.Mult(eps_hat, Ce);
  const double exact = 0.5 * (eps_hat * Ce);
  EXPECT_NEAR(energy, exact, 1e-12 * std::abs(exact));
}

TEST_P(ElasticTensorIntegratorTest, RotationCovariance) {
  auto mesh = SmallMesh(dim, elementType, 2);
  mesh.SetCurvature(2);
  mesh.Transform(Jiggle);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes(&mesh, &fec, dim);

  // A rotation R (about z in 3-D, by a fixed angle).
  DenseMatrix R(dim);
  R = 0.0;
  const double th = 0.7;
  R(0, 0) = std::cos(th);
  R(0, 1) = -std::sin(th);
  R(1, 0) = std::sin(th);
  R(1, 1) = std::cos(th);
  if (dim == 3) {
    R(2, 2) = 1.0;
  }

  Mesh rmesh(mesh);
  rmesh.Transform([&](const Vector& x, Vector& y) { R.Mult(x, y); });
  FiniteElementSpace rfes(&rmesh, &fec, dim);

  ConstantCoefficient A(3.1), C(2.7), F(1.1), L(0.9), N(1.2);
  auto axis_v = RandomVector(dim);
  Vector raxis_v(dim);
  R.Mult(axis_v, raxis_v);
  VectorConstantCoefficient axis(axis_v), raxis(raxis_v);
  TransverselyIsotropicElasticTensorCoefficient ti(dim, A, C, F, L, N, axis),
      rti(dim, A, C, F, L, N, raxis);
  ElasticTensorIntegrator integ(ti), rinteg(rti);

  DenseMatrix E, rE, P, tmp, PEPt;
  for (int e = 0; e < mesh.GetNE(); e++) {
    integ.AssembleElementMatrix(*fes.GetFE(e),
                                *mesh.GetElementTransformation(e), E);
    rinteg.AssembleElementMatrix(*rfes.GetFE(e),
                                 *rmesh.GetElementTransformation(e), rE);
    const int dof = E.Height() / dim;
    P.SetSize(E.Height());
    tmp.SetSize(E.Height());
    PEPt.SetSize(E.Height());
    P = 0.0;
    for (int c = 0; c < dim; c++) {
      for (int cp = 0; cp < dim; cp++) {
        for (int i = 0; i < dof; i++) {
          P(dof * c + i, dof * cp + i) = R(c, cp);
        }
      }
    }
    Mult(P, E, tmp);
    MultABt(tmp, P, PEPt);
    EXPECT_LT(MaxDiff(rE, PEPt), 1e-12 * E.MaxMaxNorm());
  }
}

INSTANTIATE_TEST_SUITE_P(ElasticTensorIntegrator, ElasticTensorIntegratorTest,
                         testing::Combine(testing::Values(2, 3),
                                          testing::Values(0, 1),
                                          testing::Values(1, 2, 3)));

// Plane strain: a 2-D quad mesh with u = G x versus a hex slab of thickness
// h with u = (G x, 0): energies per unit thickness agree.
TEST(ElasticTensorPlaneStrain, SlabEnergy) {
  ConstantCoefficient A(3.1), C(2.7), F(1.1), L(0.9), N(1.2);
  Vector n2(2), n3(3);
  n2[0] = 0.6;
  n2[1] = 0.8;
  n3[0] = 0.6;
  n3[1] = 0.8;
  n3[2] = 0.0;
  VectorConstantCoefficient axis2(n2), axis3(n3);
  TransverselyIsotropicElasticTensorCoefficient ti2(2, A, C, F, L, N, axis2),
      ti3(3, A, C, F, L, N, axis3);
  auto G = RandomMatrix(2);

  Mesh mesh2 = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL);
  const double h = 0.25;
  Mesh mesh3 = Mesh::MakeCartesian3D(3, 3, 1, Element::HEXAHEDRON, 1.0, 1.0, h);
  H1_FECollection fec(2, 2), fec3(2, 3);
  FiniteElementSpace fes2(&mesh2, &fec, 2), fes3(&mesh3, &fec3, 3);

  auto energy = [&](FiniteElementSpace& fes, MatrixCoefficient& c,
                    VectorCoefficient& uc) {
    BilinearForm a(&fes);
    a.AddDomainIntegrator(new ElasticTensorIntegrator(c));
    a.Assemble();
    a.Finalize();
    GridFunction u(&fes);
    u.ProjectCoefficient(uc);
    Vector Au(u.Size());
    a.SpMat().Mult(u, Au);
    return 0.5 * (u * Au);
  };
  VectorFunctionCoefficient u2(
      2, [&](const Vector& x, Vector& u) { G.Mult(x, u); });
  VectorFunctionCoefficient u3(3, [&](const Vector& x, Vector& u) {
    Vector xy(2), uv(2);
    xy[0] = x[0];
    xy[1] = x[1];
    G.Mult(xy, uv);
    u[0] = uv[0];
    u[1] = uv[1];
    u[2] = 0.0;
  });
  const double e2 = energy(fes2, ti2, u2);
  const double e3 = energy(fes3, ti3, u3);
  EXPECT_NEAR(e3 / h, e2, 1e-12 * std::abs(e2));
}

}  // namespace
