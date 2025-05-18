
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

using namespace mfem;
using namespace mfemElasticity;
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
}