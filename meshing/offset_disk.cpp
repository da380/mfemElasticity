#include <gmsh.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"

// Define geometry parameters in a single place to avoid mismatches
// between the meshing callback and the geometry generation.
namespace {
const double r_inner = 1.0;
const double r_outer = 1.75;
const double x_inner = 0.5;
const double y_inner = 0.5;
const double x_outer = 0.0;
const double y_outer = 0.0;
const double lc_init = 0.1;
}  // namespace

double meshSizeCallback(int dim, int tag, double x, double y, double z,
                        double lc) {
  const double small = 0.01;
  const double big = 0.1;
  const double fac = 0.3;

  double r0 = std::sqrt(std::pow(x - x_inner, 2) + std::pow(y - y_inner, 2));
  double r1 = std::sqrt(std::pow(x - x_outer, 2) + std::pow(y - y_outer, 2));

  double d0 = std::abs(r0 - r_inner);
  double d1 = std::abs(r1 - r_outer);

  double size = big;

  if (d0 < fac * r_inner) {
    size = small + (big - small) * d0 / (fac * r_inner);
  }

  if (d1 < fac * r_outer) {
    size = std::min(size, (r_outer / r_inner) *
                              (small + (big - small) * d1 / (fac * r_outer)));
  }

  return size;
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

  auto circle1_info = createCircle(x_inner, y_inner, r_inner, lc_init);
  int l1 = circle1_info.first;
  std::vector<int> b1 = circle1_info.second;

  auto circle2_info = createCircle(x_outer, y_outer, r_outer, lc_init);
  int l2 = circle2_info.first;
  std::vector<int> b2 = circle2_info.second;

  int v1 = gmsh::model::geo::addPlaneSurface({l1});
  int v2 = gmsh::model::geo::addPlaneSurface({l2, l1});

  gmsh::model::geo::synchronize();

  gmsh::model::addPhysicalGroup(2, {v1}, 1);
  gmsh::model::addPhysicalGroup(2, {v2}, 2);
  gmsh::model::addPhysicalGroup(1, b1, 1);
  gmsh::model::addPhysicalGroup(1, b2, 2);

  gmsh::option::setNumber("Mesh.ElementOrder", 3);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);

  gmsh::model::mesh::generate(2);
  gmsh::write("circular_offset.msh");

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