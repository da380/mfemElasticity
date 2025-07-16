#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *mesh_file =
      "/home/david/dev/meshing/examples/circular_offset.msh";
  int order = 1;
  int ref_levels = 0;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&ref_levels, "-r", "--refine", "Number of mesh refinements");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();

  {
    for (int l = 0; l < ref_levels; l++) {
      mesh.UniformRefinement();
    }
  }

  auto x0 = DomainCentroid(&mesh, 1);
  // x0(0) = 0.5;
  // x0(1) = 0.5;

  auto check_radius = BoundaryRadius(&mesh, 2, x0);
  if (check_radius) {
    cout << check_radius.value() << endl;
  } else {
    cout << "Boundary is not spherical about given point\n";
  }

  /*
  auto bdr_attributes = Array<int>{2};
  auto subMesh = SubMesh::CreateFromBoundary(mesh, bdr_attributes);

  auto *nodes = subMesh.GetNodes();

  auto ofs = ofstream("nodes.txt");

  auto x = Vector(dim);
  for (auto i = 0; i < subMesh.GetNE(); i++) {
    for (auto j = 0; j < subMesh.SpaceDimension(); j++) {
      x(j) = (*nodes)(j + dim * i);
    }
    x.Print(ofs);
    cout << x.Norml2() << endl;
  }
*/
}