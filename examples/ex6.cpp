#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace std;
using namespace mfem;
using namespace mfemElasticity;

bool SphericalBoundary(const Mesh &mesh, const Vector &x0) {
  auto bdr_marker = Array<int>(mesh.bdr_attributes.Max());
  bdr_marker = 0;
  mesh.MarkExternalBoundaries(bdr_marker);
  auto count = 0;
  for (auto mark : bdr_marker) {
    if (mark == 1) {
      count++;
    }
  }

  return true;
}

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