#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <tuple>

#include "mfemElasticity.hpp"

using namespace mfem;

using ParamTuple = std::tuple<std::string, int>;
class DomainVectorScalarIntegrator : public testing::TestWithParam<ParamTuple> {
};

TEST_P(DomainVectorScalarIntegrator, ScalarCoefficient) {
  const auto& current_tuple = GetParam();

  auto mesh_file = std::get<0>(current_tuple);
  auto order = std::get<1>(current_tuple);

  auto mesh = Mesh(mesh_file, 1, 1);
  auto dim = mesh.Dimension();
  for (int l = 0; l < 3; l++) {
    mesh.UniformRefinement();
  }

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &H1);
  auto vector_fes = FiniteElementSpace(&mesh, &L2, dim);

  auto qv = mfem::VectorFunctionCoefficient(
      dim, [](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(x.Size());
        y = x;
      });

  auto b = mfem::MixedBilinearForm(&scalar_fes, &vector_fes);
  b.AddDomainIntegrator(new mfemElasticity::DomainVectorScalarIntegrator(qv));
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

  auto rv = ScalarVectorProductCoefficient(uF, qv);
  auto l = LinearForm(&vector_fes);
  l.AddDomainIntegrator(new VectorDomainLFIntegrator(rv));
  l.Assemble();

  auto value2 = l * v;

  auto error = std::abs(value2 - value1) / std::abs(value1);
  EXPECT_TRUE(error < 1.e-6);
}

INSTANTIATE_TEST_SUITE_P(
    bilininteg, DomainVectorScalarIntegrator,
    ::testing::Values(std::make_tuple("../data/star.mesh", 2),
                      std::make_tuple("../data/star.mesh", 3),
                      std::make_tuple("../data/fichera.mesh", 2),
                      std::make_tuple("../data/fichera.mesh", 3),
                      std::make_tuple("../data/beam-tet.mesh", 2),
                      std::make_tuple("../data/beam-tet.mesh", 3)));

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
