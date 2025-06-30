
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace mfem;
using namespace mfemElasticity;
using namespace std;

Mesh MakeMesh(int dim, int elementType) {
  if (dim == 1) {
    return Mesh::MakeCartesian1D(20);
  } else if (dim == 2) {
    return Mesh::MakeCartesian2D(
        20, 20, elementType == 0 ? Element::TRIANGLE : Element::QUADRILATERAL);
  } else {
    return Mesh::MakeCartesian3D(
        20, 20, 20,
        elementType == 0 ? Element::TETRAHEDRON : Element::HEXAHEDRON);
  }
}

int main(int argc, char* argv[]) {
  // Parse command-line options.
  // auto mesh_file = std::string("../data/star.mesh");
  int order = 1;

  auto args = mfem::OptionsParser(argc, argv);
  // args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
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
  // auto mesh = Mesh(mesh_file, 1, 1);

  auto mesh = MakeMesh(2, 0);

  auto dim = mesh.Dimension();
  {
    int ref_levels = (int)floor(log(500. / mesh.GetNE()) / log(2.) / dim);
    for (int l = 0; l < ref_levels; l++) {
      mesh.UniformRefinement();
    }
  }

  auto L2 = L2_FECollection(order, dim);
  auto H1 = H1_FECollection(order + 1, dim);

  auto vector_fes0 = FiniteElementSpace(&mesh, &H1, dim);
  auto vector_fes1 = FiniteElementSpace(&mesh, &L2, dim);
  auto scalar_fes = FiniteElementSpace(&mesh, &L2);

  auto q =
      VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) { y = x; });

  auto b = MixedBilinearForm(&vector_fes0, &vector_fes1);
  b.AddDomainIntegrator(new DomainVectorDivVectorIntegrator(q));
  b.Assemble();

  auto uF =
      VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) { y = x; });
  auto u = GridFunction(&vector_fes0);
  u.ProjectCoefficient(uF);

  auto vF =
      VectorFunctionCoefficient(dim, [](const Vector& x, Vector& y) { y = x; });
  auto v = GridFunction(&vector_fes1);
  v.ProjectCoefficient(vF);

  auto z = GridFunction(&vector_fes1);
  b.Mult(u, z);
  cout << z * v << endl;

  auto wF = InnerProductCoefficient(q, vF);
  auto w = GridFunction(&scalar_fes);
  w.ProjectCoefficient(wF);

  auto c = MixedBilinearForm(&scalar_fes, &vector_fes0);
  c.AddDomainIntegrator(new DomainDivVectorScalarIntegrator());
  c.Assemble();

  c.Mult(w, z);

  cout << u * z << endl;
}