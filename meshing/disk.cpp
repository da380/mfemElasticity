#include <algorithm>  // For std::min
#include <cmath>      // For std::sqrt, std::abs
#include <iostream>
#include <vector>

// Include the main Gmsh C++ API header
#include <gmsh.h>

#include "common.hpp"

const double a = 1.0;
const double x_0 = 0.0;
const double y_0 = 0.0;
const double small = 0.05;
const double big = 0.1;
const double fac = 0.2;

double meshSizeCallback(int dim, int tag, double x, double y, double z,
                        double lc) {
  double r0 = std::sqrt(std::pow(x - x_0, 2) + std::pow(y - y_0, 2));
  double d0 = std::abs(r0 - a);

  return d0 < fac * a ? small + (big - small) * d0 / (fac * a) : big;
}

int main(int argc, char **argv) {
  gmsh::initialize(argc, argv);
  gmsh::option::setNumber("General.Terminal", 1);

  gmsh::option::setNumber("Mesh.Nodes", 1);
  gmsh::option::setNumber("Mesh.VolumeFaces", 1);
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

  gmsh::model::add("circular_offset");

  gmsh::model::mesh::setSizeCallback(meshSizeCallback);

  const double lc = 0.1;

  auto circle1_info = createCircle(x_0, y_0, a, lc);
  int l1 = circle1_info.first;
  std::vector<int> b1 = circle1_info.second;

  int v1 = gmsh::model::geo::addPlaneSurface({l1});
  gmsh::model::geo::synchronize();

  gmsh::model::addPhysicalGroup(2, {v1}, 1);

  gmsh::model::addPhysicalGroup(1, b1, 1);

  gmsh::option::setNumber("Mesh.ElementOrder", 3);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);

  gmsh::model::mesh::generate(2);

  gmsh::write("disk.msh");

  bool no_popup = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-nopopup") {
      no_popup = true;
      break;
    }
  }
  if (!no_popup) {
    gmsh::fltk::run();
  }

  gmsh::finalize();

  return 0;
}