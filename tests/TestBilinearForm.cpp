#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <tuple>

#include "mfemElasticity.hpp"

using namespace mfem;

using ParamTuple = std::tuple<std::string, int>;
class InterpolatorTest : public testing::TestWithParam<ParamTuple> {
 protected:
  void SetUp() {
    const auto& current_tuple = GetParam();

    auto mesh_file = std::get<0>(current_tuple);
    order = std::get<1>(current_tuple);

    mesh = Mesh(mesh_file, 1, 1);
    dim = mesh.Dimension();
    {
      int ref_levels = (int)floor(log(10000. / mesh.GetNE()) / log(2.) / dim);
      for (int l = 0; l < ref_levels; l++) {
        mesh.UniformRefinement();
      }
    }

    L2 = std::make_unique<L2_FECollection>(order, dim);
    H1 = std::make_unique<H1_FECollection>(order + 1, dim);

    scalar_fes = std::make_unique<FiniteElementSpace>(&mesh, H1.get());
    vector_fes = std::make_unique<FiniteElementSpace>(&mesh, H1.get(), dim);
    matrix_fes =
        std::make_unique<FiniteElementSpace>(&mesh, L2.get(), dim * dim);
    strain_fes = std::make_unique<FiniteElementSpace>(&mesh, L2.get(),
                                                      dim * (dim + 1) / 2);
    deviatoric_strain_fes = std::make_unique<FiniteElementSpace>(
        &mesh, L2.get(), dim * (dim + 1) / 2 - 1);
  }

  int order, dim;
  DenseMatrix A;
  Mesh mesh;
  std::unique_ptr<FiniteElementCollection> L2, H1;
  std::unique_ptr<FiniteElementSpace> scalar_fes, vector_fes, matrix_fes,
      strain_fes, deviatoric_strain_fes;
};

TEST_P(InterpolatorTest, DomainVectorScalarIntegrator) {
  auto qv = mfem::VectorFunctionCoefficient(
      dim, [](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(x.Size());
        y = x;
      });

  auto b = mfem::MixedBilinearForm(scalar_fes.get(), vector_fes.get());
  b.AddDomainIntegrator(new mfemElasticity::DomainVectorScalarIntegrator(qv));
  b.Assemble();

  auto u = mfem::GridFunction(scalar_fes.get());
  auto uF = mfem::FunctionCoefficient(
      [](const mfem::Vector& x) { return x.Norml2(); });
  u.ProjectCoefficient(uF);

  auto v = mfem::GridFunction(vector_fes.get());
  auto vF = mfem::VectorFunctionCoefficient(
      dim, [](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(x.Size());
        y = x;
      });
  v.ProjectCoefficient(vF);

  auto w = mfem::GridFunction(vector_fes.get());

  b.Mult(u, w);
  auto value1 = v * w;

  auto rv = ScalarVectorProductCoefficient(uF, qv);
  auto l = LinearForm(vector_fes.get());
  l.AddDomainIntegrator(new VectorDomainLFIntegrator(rv));
  l.Assemble();

  auto value2 = l * v;

  auto error = std::abs(value2 - value1) / std::abs(value1);
  std::cout << value1 << std::endl;
  std::cout << value2 << std::endl;

  EXPECT_TRUE(error < 1.e-5);
}

INSTANTIATE_TEST_SUITE_P(
    , InterpolatorTest,
    ::testing::Values(std::make_tuple("../data/star.mesh", 1),
                      std::make_tuple("../data/star.mesh", 2),
                      std::make_tuple("../data/fichera.mesh", 1),
                      std::make_tuple("../data/fichera.mesh", 2),
                      std::make_tuple("../data/beam-tet.mesh", 1),
                      std::make_tuple("../data/beam-tet.mesh", 2)));

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
