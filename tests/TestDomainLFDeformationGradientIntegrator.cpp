
#include "TestCommon.hpp"

class LinearFormIntegratorTests
    : public ::testing::TestWithParam<DimOrderTypeTuple> {};

TEST_P(LinearFormIntegratorTests, DomainLFDeformationGradientIntegrator) {
  const auto& current_tuple = GetParam();

  auto dim = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);
  auto elementType = std::get<2>(current_tuple);

  auto mesh = MakeMesh(dim, elementType);

  auto H1 = H1_FECollection(order, dim);
  auto vector_fes = FiniteElementSpace(&mesh, &H1, dim);

  auto m = MatrixFunctionCoefficient(dim, [](const Vector& x, DenseMatrix& m) {
    auto dim = x.Size();
    m.SetSize(dim);
    for (auto j = 0; j < dim; j++) {
      for (auto i = 0; i < dim; i++) {
        m(i, j) = (i + 1) * (j + 2) * x(i) * x(j);
      }
    }
  });

  auto b = LinearForm(&vector_fes);
  b.AddDomainIntegrator(
      new mfemElasticity::DomainLFDeformationGradientIntegrator(m));
  b.Assemble();

  auto f = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) {
    y = x;
    y *= x * x;
  });

  auto x = GridFunction(&vector_fes);
  x.ProjectCoefficient(f);
  auto value1 = b(x);

  auto L2 = L2_FECollection(order, dim);
  auto scalar_fes = FiniteElementSpace(&mesh, &L2);

  auto c = LinearForm(&scalar_fes);
  auto one = ConstantCoefficient(1);
  c.AddDomainIntegrator(new DomainLFIntegrator(one));
  c.Assemble();

  auto h = FunctionCoefficient([](const Vector& x) {
    auto dim = x.Size();
    auto f = x * x;
    auto sum = real_t{0};
    for (auto j = 0; j < dim; j++) {
      for (auto i = 0; i < dim; i++) {
        sum += (i + 1) * (j + 2) * x(i) * x(j) *
               (2 * x(i) * x(j) + f * (i == j ? 1 : 0));
      }
    }
    return sum;
  });

  auto y = GridFunction(&scalar_fes);
  y.ProjectCoefficient(h);
  auto value2 = c(y);

  EXPECT_NEAR(value1, value2, 1.e-6 * std::abs(value1));
}

INSTANTIATE_TEST_SUITE_P(DimensionOrderElementType, LinearFormIntegratorTests,
                         ::testing::Values(std::make_tuple(1, 2, 0),
                                           std::make_tuple(2, 2, 0),
                                           std::make_tuple(2, 2, 1),
                                           std::make_tuple(3, 2, 0),
                                           std::make_tuple(3, 2, 1)));

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}