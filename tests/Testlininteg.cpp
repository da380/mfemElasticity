#include <gtest/gtest.h>

#include <string>
#include <tuple>

#include "mfemElasticity.hpp"

using namespace mfem;

// Define a test fixture that is parameterized with a tuple
using ParamTuple = std::tuple<std::string, int>;
class linintegTest : public ::testing::TestWithParam<ParamTuple> {};

// Define the test case that uses the tuple parameters
TEST_P(linintegTest, TestTupleElements) {
  const auto& current_tuple = GetParam();

  auto mesh_file = std::get<0>(current_tuple);
  int order = std::get<1>(current_tuple);

  auto mesh = mfem::Mesh(mesh_file, 1, 1);

  auto dim = mesh.Dimension();
  {
    int ref_levels = (int)floor(log(1000. / mesh.GetNE()) / log(2.) / dim);
    for (int l = 0; l < ref_levels; l++) {
      mesh.UniformRefinement();
    }
  }

  // Define the vector finite element space.
  auto H1 = H1_FECollection(order + 1, dim);
  auto fes = FiniteElementSpace(&mesh, &H1, dim);

  // Set up a matrix coefficient function.
  auto m = MatrixFunctionCoefficient(dim, [](const Vector& x, DenseMatrix& m) {
    auto dim = x.Size();
    m.SetSize(dim);
    for (auto j = 0; j < dim; j++) {
      for (auto i = 0; i < dim; i++) {
        m(i, j) = (i + 1) * (j + 2);
      }
    }
  });

  // Set up the linearform.
  auto b = LinearForm(&fes);
  b.AddDomainIntegrator(
      new mfemElasticity::DomainLFDeformationGradientIntegrator(m));
  b.Assemble();

  // Set up a vector field.
  auto f = VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) {
    y = x;
    y *= x * x;
  });

  auto x = GridFunction(&fes);
  x.ProjectCoefficient(f);
  auto value1 = b(x);

  // Define scalar finite element space.

  auto L2 = L2_FECollection(order, dim);
  auto scalarFES = FiniteElementSpace(&mesh, &L2);

  // Set up the associated linear form.
  auto c = LinearForm(&scalarFES);
  auto one = ConstantCoefficient(1);
  c.AddDomainIntegrator(new DomainLFIntegrator(one));
  c.Assemble();

  // Set up scalar field as the divergence of the first.
  auto h = FunctionCoefficient([](const Vector& x) {
    auto dim = x.Size();
    auto f = x * x;
    auto sum = real_t{0};
    for (auto j = 0; j < dim; j++) {
      for (auto i = 0; i < dim; i++) {
        sum += (i + 1) * (j + 2) * (2 * x(i) * x(j) + f * (i == j ? 1 : 0));
      }
    }
    return sum;
  });

  auto y = GridFunction(&scalarFES);
  y.ProjectCoefficient(h);
  auto value2 = c(y);

  EXPECT_FLOAT_EQ(value1, value2);
}

// Instantiate the test suite with different tuple values
INSTANTIATE_TEST_SUITE_P(
    , linintegTest,
    ::testing::Values(std::make_tuple("../data/star.mesh", 1),
                      std::make_tuple("../data/star.mesh", 2),
                      std::make_tuple("../data/fichera.mesh", 1),
                      std::make_tuple("../data/fichera.mesh", 2)));

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}