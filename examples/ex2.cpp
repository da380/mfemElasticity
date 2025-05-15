
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace mfem;
using namespace std;

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

  cout << v * w << endl;

  auto qmt = TransposeMatrixCoefficient(qm);
  auto zF = MatrixVectorProductCoefficient(qmt, vF);

  auto l = mfem::LinearForm(&scalar_fes);
  l.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(zF));
  l.Assemble();

  cout << l * u << endl;
}