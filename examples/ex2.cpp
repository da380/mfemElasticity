
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace mfem;
using namespace mfemElasticity;
using namespace std;

DenseMatrix RandomMatrix(int dim) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> distrib(0, 1);

  auto A = DenseMatrix(dim);
  for (auto j = 0; j < dim; j++) {
    for (auto i = 0; i < dim; i++) {
      A(i, j) = distrib(gen);
    }
  }
  return A;
}

int main(int argc, char* argv[]) {
  // Parse command-line options.
  auto mesh_file = std::string("../data/star.mesh");
  int order = 1;

  auto args = mfem::OptionsParser(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(std::cout);
    return 1;
  }
  args.PrintOptions(std::cout);

  // Read the mesh from the given mesh file.
  auto mesh = Mesh(mesh_file, 1, 1);

  auto dim = mesh.Dimension();
  {
    int ref_levels = (int)floor(log(500. / mesh.GetNE()) / log(2.) / dim);
    for (int l = 0; l < ref_levels; l++) {
      mesh.UniformRefinement();
    }
  }

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order + 1, dim);

  auto vector_fes = FiniteElementSpace(&mesh, &H1, dim);
  auto deviatoric_strain_fes =
      FiniteElementSpace(&mesh, &L2, dim * (dim + 1) / 2 - 1);

  auto q = FunctionCoefficient([](const Vector& x) { return x.Norml2(); });

  auto b = MixedBilinearForm(&vector_fes, &deviatoric_strain_fes);
  b.AddDomainIntegrator(
      new DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator(q));
  b.Assemble();

  auto B = RandomMatrix(dim);
  auto uF = VectorFunctionCoefficient(
      dim, [B](const Vector& x, Vector& y) { B.Mult(x, y); });
  auto u = GridFunction(&vector_fes);
  u.ProjectCoefficient(uF);

  auto A = RandomMatrix(dim);
  A.Symmetrize();
  auto trace = A.Trace();
  for (auto i = 0; i < dim; i++) {
    A(i, i) -= trace / dim;
  }

  auto a = Vector(dim * (dim + 1) / 2 - 1);
  auto k = 0;
  for (auto j = 0; j < dim - 1; j++) {
    for (auto i = j; i < dim; i++) {
      a(k++) = A(i, j);
    }
  }
  auto vF = VectorConstantCoefficient(a);

  auto v = GridFunction(&deviatoric_strain_fes);
  v.ProjectCoefficient(vF);

  auto w = GridFunction(&deviatoric_strain_fes);
  b.Mult(u, w);

  auto value1 = v * w;

  auto AF = MatrixConstantCoefficient(A);
  auto MF = ScalarMatrixProductCoefficient(q, AF);
  auto l = LinearForm(&vector_fes);
  l.AddDomainIntegrator(new DomainLFDeformationGradientIntegrator(MF));
  l.Assemble();

  auto value2 = l * u;

  cout << value1 << " " << value2 << endl;
}