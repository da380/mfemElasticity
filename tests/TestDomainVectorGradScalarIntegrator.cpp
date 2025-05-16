#include "TestCommon.hpp"

class BilinearFormIntegratorTests
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

TEST_P(BilinearFormIntegratorTests,
       DomainVectorGradScalarIntegratorScalarCoefficient) {
  const auto& current_tuple = GetParam();

  auto dim = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);
  auto elementType = std::get<2>(current_tuple);

  auto mesh = MakeMesh(dim, elementType);

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &H1);
  auto vector_fes = FiniteElementSpace(&mesh, &L2, dim);

  auto q = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });

  auto b = MixedBilinearForm(&scalar_fes, &vector_fes);
  b.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(q));
  b.Assemble();

  auto u = GridFunction(&scalar_fes);
  auto uF = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });
  u.ProjectCoefficient(uF);

  auto v = GridFunction(&vector_fes);
  auto vF = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) {
    y.SetSize(x.Size());
    y = 1.;
  });
  v.ProjectCoefficient(vF);

  auto w = GridFunction(&vector_fes);
  b.Mult(u, w);
  auto value1 = v * w;

  auto zF = ScalarVectorProductCoefficient(q, vF);
  auto l = LinearForm(&scalar_fes);
  l.AddDomainIntegrator(new DomainLFGradIntegrator(zF));
  l.Assemble();
  auto value2 = l * u;

  EXPECT_NEAR(value1, value2, 1.e-6 * std::abs(value1));
}

TEST_P(BilinearFormIntegratorTests,
       DomainVectorGradScalarIntegratorVectorCoefficient) {
  const auto& current_tuple = GetParam();

  auto dim = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);
  auto elementType = std::get<2>(current_tuple);

  auto mesh = MakeMesh(dim, elementType);

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &H1);
  auto vector_fes = FiniteElementSpace(&mesh, &L2, dim);

  auto a = RandomVector(dim);
  auto qv = VectorConstantCoefficient(a);

  auto b = MixedBilinearForm(&scalar_fes, &vector_fes);
  b.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(qv));
  b.Assemble();

  auto u = GridFunction(&scalar_fes);
  auto uF = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });
  u.ProjectCoefficient(uF);

  auto v = GridFunction(&vector_fes);
  auto vF = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) {
    y.SetSize(x.Size());
    y = x;
  });
  v.ProjectCoefficient(vF);

  auto w = GridFunction(&vector_fes);
  b.Mult(u, w);
  auto value1 = v * w;

  auto zF = VectorFunctionCoefficient(dim, [a](const Vector& x, Vector& y) {
    auto dim = x.Size();
    y.SetSize(dim);
    for (auto i = 0; i < dim; i++) {
      y(i) = a(i) * x(i);
    }
  });

  auto l = mfem::LinearForm(&scalar_fes);
  l.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(zF));
  l.Assemble();
  auto value2 = l * u;

  EXPECT_NEAR(value1, value2, 1.e-6 * std::abs(value1));
}

TEST_P(BilinearFormIntegratorTests,
       DomainVectorGradScalarIntegratorMatrixCoefficient) {
  const auto& current_tuple = GetParam();

  auto dim = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);
  auto elementType = std::get<2>(current_tuple);

  auto mesh = MakeMesh(dim, elementType);

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &H1);
  auto vector_fes = FiniteElementSpace(&mesh, &L2, dim);

  auto A = RandomMatrix(dim);
  auto qm = MatrixConstantCoefficient(A);

  auto b = MixedBilinearForm(&scalar_fes, &vector_fes);
  b.AddDomainIntegrator(new DomainVectorGradScalarIntegrator(qm));
  b.Assemble();

  auto u = mfem::GridFunction(&scalar_fes);
  auto uF = mfem::FunctionCoefficient(
      [](const mfem::Vector& x) { return x.Norml2(); });
  u.ProjectCoefficient(uF);

  auto v = mfem::GridFunction(&vector_fes);
  auto vF = mfem::VectorFunctionCoefficient(
      dim, [](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(x.Size());
        y = x;
      });
  v.ProjectCoefficient(vF);

  auto w = mfem::GridFunction(&vector_fes);
  b.Mult(u, w);
  auto value1 = v * w;

  auto qmt = TransposeMatrixCoefficient(qm);
  auto zF = MatrixVectorProductCoefficient(qmt, vF);

  auto l = mfem::LinearForm(&scalar_fes);
  l.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(zF));
  l.Assemble();
  auto value2 = l * u;

  EXPECT_NEAR(value1, value2, 1.e-6 * std::abs(value1));
}

INSTANTIATE_TEST_SUITE_P(DimensionOrderElementType, BilinearFormIntegratorTests,
                         ::testing::Values(std::make_tuple(1, 2, 0),
                                           std::make_tuple(2, 2, 0),
                                           std::make_tuple(2, 2, 1),
                                           std::make_tuple(3, 2, 0),
                                           std::make_tuple(3, 2, 1)));

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

/*
TEST_P(InterpolatorTest, DomainVectorGradScalarIntegratorScalarCoefficient) {
  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &H1);
  auto vector_fes = FiniteElementSpace(&mesh, &L2, dim);

  auto b = MixedBilinearForm(&scalar_fes, &vector_fes);
  b.AddDomainIntegrator(new mfemElasticity::DomainVectorGradScalarIntegrator());
  b.Assemble();

  auto u = GridFunction(&scalar_fes);
  auto uF = FunctionCoefficient(
      [](const Vector& x) { return x(0) * x(1) * x(1); });
  u.ProjectCoefficient(uF);

  auto v = GridFunction(&vector_fes);
  auto vF = VectorFunctionCoefficient(
      dim, [](const Vector& x, Vector& y) {
        y.SetSize(x.Size());
        y = 1.;
      });
  v.ProjectCoefficient(vF);

  auto w = GridFunction(&vector_fes);
  b.Mult(u, w);
  auto value1 = v * w;

  auto l = mfem::LinearForm(&scalar_fes);
  l.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(vF));
  l.Assemble();
  auto value2 = l * u;

  auto error = std::abs(value2 - value1) / std::abs(value1);
  EXPECT_TRUE(error < 1.e-6);
}

TEST_P(InterpolatorTest, DomainVectorGradScalarIntegratorVectorCoefficient) {
  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &H1);
  auto vector_fes = FiniteElementSpace(&mesh, &L2, dim);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> distrib(0, 1);

  auto a = Vector(dim);
  for (auto j = 0; j < dim; j++) {
    a(j) = distrib(gen);
  }
  auto qv = VectorConstantCoefficient(a);

  auto b = mfem::MixedBilinearForm(&scalar_fes, &vector_fes);
  b.AddDomainIntegrator(
      new mfemElasticity::DomainVectorGradScalarIntegrator(qv));
  b.Assemble();

  auto u = mfem::GridFunction(&scalar_fes);
  auto uF = mfem::FunctionCoefficient(
      [](const mfem::Vector& x) { return x.Norml2(); });
  u.ProjectCoefficient(uF);

  auto v = mfem::GridFunction(&vector_fes);
  auto vF = mfem::VectorFunctionCoefficient(
      dim, [](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(x.Size());
        y = x;
      });
  v.ProjectCoefficient(vF);

  auto w = mfem::GridFunction(&vector_fes);
  b.Mult(u, w);
  auto value1 = v * w;

  auto l = mfem::LinearForm(&scalar_fes);
  l.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(vF));
  l.Assemble();
  auto value2 = l * u;

  auto error = std::abs(value2 - value1) / std::abs(value1);
  EXPECT_TRUE(error < 1.e-6);
}

TEST_P(InterpolatorTest, DomainVectorGradScalarIntegratorMatrixCoefficient) {
  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &H1);
  auto vector_fes = FiniteElementSpace(&mesh, &L2, dim);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> distrib(0, 1);

  auto A = DenseMatrix(dim);
  for (auto j = 0; j < dim; j++) {
    for (auto i = 0; i < dim; i++) {
      A(i, j) = distrib(gen);
    }
  }
  auto qm = MatrixConstantCoefficient(A);

  auto b = mfem::MixedBilinearForm(&scalar_fes, &vector_fes);
  b.AddDomainIntegrator(
      new mfemElasticity::DomainVectorGradScalarIntegrator(qm));
  b.Assemble();

  auto u = mfem::GridFunction(&scalar_fes);
  auto uF = mfem::FunctionCoefficient(
      [](const mfem::Vector& x) { return x(0) * x(1) * x(1); });
  u.ProjectCoefficient(uF);

  auto v = mfem::GridFunction(&vector_fes);
  auto vF = mfem::VectorFunctionCoefficient(
      dim, [](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(x.Size());
        y = 1.;
      });
  v.ProjectCoefficient(vF);

  auto w = mfem::GridFunction(&vector_fes);
  b.Mult(u, w);
  auto value1 = v * w;

  auto l = mfem::LinearForm(&scalar_fes);
  l.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(vF));
  l.Assemble();
  auto value2 = l * u;

  auto error = std::abs(value2 - value1) / std::abs(value1);
  EXPECT_TRUE(error < 1.e-6);
}
*/