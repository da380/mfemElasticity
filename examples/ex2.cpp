
#include <cmath>
#include <fstream>
#include <iostream>
#include <stack>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace mfem;
using namespace std;

int main(int argc, char* argv[]) {
  // Parse command-line options.
  auto mesh_file = std::string("../data/star.mesh");
  int order = 0;

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
  auto mesh = mfem::Mesh(mesh_file, 1, 1);

  auto dim = mesh.Dimension();
  {
    int ref_levels = (int)floor(log(500. / mesh.GetNE()) / log(2.) / dim);
    for (int l = 0; l < ref_levels; l++) {
      mesh.UniformRefinement();
    }
  }

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order + 1, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &L2);
  auto vector_fes = FiniteElementSpace(&mesh, &H1, dim);

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

  cout << v * w << endl;

  /*
  auto kappa = mfem::ConstantCoefficient(dim);
  auto l = mfem::LinearForm(&scalar_fes);
  l.AddDomainIntegrator(new mfem::DomainLFIntegrator(kappa));
  l.Assemble();
  cout << l * u << endl;
  */

  auto rv = ScalarVectorProductCoefficient(uF, qv);
  auto l = LinearForm(&vector_fes);
  l.AddDomainIntegrator(new VectorDomainLFIntegrator(rv));
  l.Assemble();

  cout << l * v << endl;
}