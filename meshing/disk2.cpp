

#include <algorithm>  // For std::min
#include <cmath>      // For std::sqrt, std::abs
#include <iostream>
#include <vector>

// Include the main Gmsh C++ API header
#include <gmsh.h>

#include "common.hpp"

const double a = 1.0;
const double b = 2.0;
const double small = 0.025;
const double big = 0.1;
const double fac = 0.2;

double meshSizeCallback(int dim, int tag, double x, double y, double z,
                        double lc) {
  auto r0 = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
  auto d0 = std::abs(r0 - a);
  auto d1 = std::abs(r0 - b);

  auto size = big;

  if (d0 < fac * a) {
    size = small + (big - small) * d0 / (fac * a);
  }

  if (d1 < fac * b) {
    size = std::min(size, (b / a) * (small + (big - small) * d1 / (fac * b)));
  }

  return size;
}

int main(int argc, char** argv) {
  gmsh::initialize(argc, argv);
  gmsh::option::setNumber("General.Terminal", 1);

  gmsh::option::setNumber("Mesh.Nodes", 1);
  gmsh::option::setNumber("Mesh.VolumeFaces", 1);
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);
  gmsh::option::setNumber("Mesh.ElementOrder", 3);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);

  gmsh::model::add("circular_offset");

  gmsh::model::mesh::setSizeCallback(meshSizeCallback);

  const double lc = 0.1;

  auto d1 = gmsh::model::occ::addDisk(0, 0, 0, a, a);
  auto d2 = gmsh::model::occ::addDisk(0, 0, 0, b, b);

  std::vector<std::pair<int, int>> outDimTags;
  std::vector<std::vector<std::pair<int, int>>> outDimTagsMap;

  gmsh::model::occ::fragment({{2, d2}}, {{2, d1}}, outDimTags, outDimTagsMap);

  gmsh::model::occ::synchronize();

  // 4. Identify the new surface and curve tags

  // The logic here is identical to the 'fuse' example.
  // outDimTagsMap[0] -> maps for the object(s) (d2)
  // outDimTagsMap[1] -> maps for the tool(s) (d1)

  // The tool (d1) becomes the new inner surface.
  // Its map is in outDimTagsMap[1].
  int inner_surf_tag = outDimTagsMap[1][0].second;

  // The object (d2) was split into two surfaces. Its map (outDimTagsMap[0])
  // will contain both new surfaces. We find the one that is NOT
  // the inner surface.
  int annulus_surf_tag = -1;
  for (const auto& pair : outDimTagsMap[0]) {
    if (pair.second != inner_surf_tag) {
      annulus_surf_tag = pair.second;
      break;
    }
  }
  if (annulus_surf_tag == -1) {
    throw std::runtime_error("Could not find annulus surface tag.");
  }

  // --- Find boundaries (Identical logic) ---

  // Get the boundary of the inner surface (should be one curve)
  std::vector<std::pair<int, int>> inner_bnd_pairs;
  gmsh::model::getBoundary({{2, inner_surf_tag}}, inner_bnd_pairs, false, false,
                           false);

  if (inner_bnd_pairs.empty()) {
    throw std::runtime_error("Inner surface has no boundary.");
  }
  int inner_curve_tag = inner_bnd_pairs[0].second;

  // Get the boundary of the annulus (should be two curves)
  std::vector<std::pair<int, int>> annulus_bnd_pairs;
  gmsh::model::getBoundary({{2, annulus_surf_tag}}, annulus_bnd_pairs, false,
                           false, false);

  // One of these curves is the inner_curve_tag. The other is the outer.
  int outer_curve_tag = -1;
  for (const auto& pair : annulus_bnd_pairs) {
    if (pair.second != inner_curve_tag) {
      outer_curve_tag = pair.second;
      break;
    }
  }
  if (outer_curve_tag == -1) {
    throw std::runtime_error("Could not find outer curve tag.");
  }

  // 5. Create Physical Groups (Identical logic)

  // Surfaces (Dimension 2)
  gmsh::model::addPhysicalGroup(2, {inner_surf_tag}, 1);
  gmsh::model::addPhysicalGroup(2, {annulus_surf_tag}, 2);

  // Boundaries (Dimension 1)
  gmsh::model::addPhysicalGroup(1, {inner_curve_tag}, 1);
  gmsh::model::addPhysicalGroup(1, {outer_curve_tag}, 2);

  // --- Generate mesh and visualize ---
  gmsh::model::mesh::generate(2);

  gmsh::write("disk2.msh");

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