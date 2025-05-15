#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <string>
#include <tuple>

#include "mfemElasticity.hpp"

using namespace mfem;

using ParamTuple = std::tuple<std::string, int>;
class bilinintegTest : public ::testing::TestWithParam<ParamTuple> {
 protected:
  void SetUp() {
    const auto& current_tuple = GetParam();

    auto mesh_file = std::get<0>(current_tuple);
    order = std::get<1>(current_tuple);

    mesh = Mesh(mesh_file, 1, 1);
    dim = mesh.Dimension();
    {
      int ref_levels = (int)floor(log(1000. / mesh.GetNE()) / log(2.) / dim);
      for (int l = 0; l < ref_levels; l++) {
        mesh.UniformRefinement();
      }
    }

    L2 = std::make_unique<L2_FECollection>(order, dim);
    H1 = std::make_unique<H1_FECollection>(order + 1, dim);

    vector_fes = std::make_unique<FiniteElementSpace>(&mesh, H1.get(), dim);
    matrix_fes =
        std::make_unique<FiniteElementSpace>(&mesh, L2.get(), dim * dim);
    strain_fes = std::make_unique<FiniteElementSpace>(&mesh, L2.get(),
                                                      dim * (dim + 1) / 2);
    deviatoric_strain_fes = std::make_unique<FiniteElementSpace>(
        &mesh, L2.get(), dim * (dim + 1) / 2 - 1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> distrib(0, 1);

    A.SetSize(dim);
    for (auto j = 0; j < dim; j++) {
      for (auto i = 0; i < dim; i++) {
        A(i, j) = distrib(gen);
      }
    }

    uF = std::make_unique<VectorFunctionCoefficient>(
        dim, [this](const Vector& x, Vector& u) {
          u.SetSize(x.Size());
          A.Mult(x, u);
        });

    FF = std::make_unique<VectorFunctionCoefficient>(
        dim * dim, [this](const Vector& x, Vector& m) {
          auto dim = x.Size();
          m.SetSize(dim * dim);
          auto k = 0;
          for (auto j = 0; j < dim; j++) {
            for (auto i = 0; i < dim; i++) {
              m(k++) = A(i, j);
            }
          }
        });

    EF = std::make_unique<VectorFunctionCoefficient>(
        dim * (dim + 1) / 2, [this](const Vector& x, Vector& m) {
          auto dim = x.Size();
          m.SetSize(dim * (dim + 1) / 2);
          auto k = 0;
          for (auto j = 0; j < dim; j++) {
            for (auto i = j; i < dim; i++) {
              m(k++) = 0.5 * (A(i, j) + A(j, i));
            }
          }
        });

    DF = std::make_unique<VectorFunctionCoefficient>(
        dim * (dim + 1) / 2 - 1, [this](const Vector& x, Vector& m) {
          auto dim = x.Size();
          m.SetSize(dim * (dim + 1) / 2 - 1);
          auto trace = A.Trace();
          auto k = 0;
          for (auto j = 0; j < dim - 1; j++) {
            for (auto i = j; i < dim; i++) {
              m(k) = 0.5 * (A(i, j) + A(j, i));
              if (i == j) {
                m(k) -= trace / dim;
              }
              k++;
            }
          }
        });
  }

  int order, dim;
  DenseMatrix A;
  Mesh mesh;
  std::unique_ptr<FiniteElementCollection> L2, H1;
  std::unique_ptr<FiniteElementSpace> vector_fes, matrix_fes, strain_fes,
      deviatoric_strain_fes;
  std::unique_ptr<VectorFunctionCoefficient> uF, FF, EF, DF;
};

TEST_P(bilinintegTest, DeformationGradientInterpolator) {
  auto u = GridFunction(vector_fes.get());
  u.ProjectCoefficient(*uF);

  auto b = DiscreteLinearOperator(vector_fes.get(), matrix_fes.get());
  b.AddDomainInterpolator(
      new mfemElasticity::DeformationGradientInterpolator());
  b.Assemble();

  auto F = GridFunction(matrix_fes.get());
  b.Mult(u, F);

  auto error = F.ComputeL2Error(*FF);

  EXPECT_TRUE(error < 1.e-8);
}

TEST_P(bilinintegTest, StrainInterpolator) {
  auto u = GridFunction(vector_fes.get());
  u.ProjectCoefficient(*uF);

  auto b = DiscreteLinearOperator(vector_fes.get(), strain_fes.get());
  b.AddDomainInterpolator(new mfemElasticity::StrainInterpolator());
  b.Assemble();

  auto E = GridFunction(strain_fes.get());
  b.Mult(u, E);

  auto error = E.ComputeL2Error(*EF);

  EXPECT_TRUE(error < 1.e-8);
}

TEST_P(bilinintegTest, DeviatoricStrainInterpolator) {
  auto u = GridFunction(vector_fes.get());
  u.ProjectCoefficient(*uF);

  auto b =
      DiscreteLinearOperator(vector_fes.get(), deviatoric_strain_fes.get());
  b.AddDomainInterpolator(new mfemElasticity::DeviatoricStrainInterpolator());
  b.Assemble();

  auto D = GridFunction(deviatoric_strain_fes.get());
  b.Mult(u, D);

  auto error = D.ComputeL2Error(*DF);

  EXPECT_TRUE(error < 1.e-8);
}

INSTANTIATE_TEST_SUITE_P(
    , bilinintegTest,
    ::testing::Values(std::make_tuple("../data/star.mesh", 0),
                      std::make_tuple("../data/star.mesh", 1),
                      std::make_tuple("../data/fichera.mesh", 0),
                      std::make_tuple("../data/fichera.mesh", 1),
                      std::make_tuple("../data/beam-tet.mesh", 0),
                      std::make_tuple("../data/beam-tet.mesh", 1)));

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
