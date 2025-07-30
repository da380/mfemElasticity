#include <algorithm>  // For std::min
#include <cmath>      // For std::sqrt, std::pow, std::abs
#include <iostream>
#include <string>  // For std::string comparison in argument parsing
#include <vector>

// Include the main Gmsh C++ API header
#include <gmsh.h>

#include "common.hpp"

// Global parameters for the mesh size callback
// In a larger application, these might be passed via a struct or class
// or managed differently, but for a direct translation, global constants work.
const double G_x0 = 0.0;
const double G_y0 = 0.25;
const double G_z0 = 0.0;
const double G_a = 0.5;
const double G_b = 1.0;
const double G_small = 0.025;
const double G_big = 0.05;
const double G_fac = 0.2;

// Custom mesh size callback function
// The signature must match gmsh::model::mesh::setSizeCallback's expectation
double meshSizeCallback(int dim, int tag, double x, double y, double z,
                        double lc) {
  double r0 = std::sqrt(std::pow(x - G_x0, 2) + std::pow(y - G_y0, 2) +
                        std::pow(z - G_z0, 2));
  double r1 = std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));

  double d0 = std::abs(r0 - G_a);
  double d1 = std::abs(r1 - G_b);

  double size = G_big;

  if (d0 < G_fac * G_a) {
    size = G_small + (G_big - G_small) * d0 / (G_fac * G_a);
  }

  if (d1 < G_fac * G_b) {
    // This line directly translates the Python: size = 2 * big - big * d1 /
    // (fac * b) If the original Python had `min(size, ...)` uncommented, use
    // std::min here.
    size = std::min(size, 2 * G_big - G_big * d1 / (G_fac * G_b));
  }

  return size;
}

int main(int argc, char **argv) {
  gmsh::initialize(argc, argv);
  gmsh::option::setNumber("General.Terminal", 1);  // Print info to terminal

  gmsh::model::add("spherical_offset");

  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

  // Set the custom mesh size callback
  gmsh::model::mesh::setSizeCallback(meshSizeCallback);

  // Initial characteristic length for point creation
  const double lc_val = 0.1;

  // Create the two spheres
  auto sphere1_info = createSphere(G_x0, G_y0, G_z0, G_a, lc_val);
  int sl1 = sphere1_info.first;
  std::vector<int> s_tags1 =
      sphere1_info.second;  // Surface tags of inner sphere

  auto sphere2_info = createSphere(0, 0, 0, G_b, lc_val);
  int sl2 = sphere2_info.first;
  std::vector<int> s_tags2 =
      sphere2_info.second;  // Surface tags of outer sphere

  // Create volumes
  // v1 is the inner sphere volume
  int v1 = gmsh::model::geo::addVolume({sl1});
  // v2 is the volume between the outer sphere and the inner sphere
  int v2 = gmsh::model::geo::addVolume({sl2, sl1});

  // Remove duplicates (e.g., points, curves, surfaces that might be shared)
  gmsh::model::occ::removeAllDuplicates();  // Use OCC's removeAllDuplicates if
                                            // using OCC kernel
  gmsh::model::geo::synchronize();  // Synchronize the CAD kernel with Gmsh's
                                    // model

  // Add Physical Groups for volumes and surfaces
  gmsh::model::addPhysicalGroup(3, {v1}, 1);  // Physical Volume 1: Inner sphere
  gmsh::model::addPhysicalGroup(
      3, {v2}, 2);  // Physical Volume 2: Volume between spheres

  // Physical surfaces for the boundaries
  gmsh::model::addPhysicalGroup(
      2, s_tags1, 1);  // Physical Surface 1: Inner sphere boundary
  gmsh::model::addPhysicalGroup(
      2, s_tags2, 2);  // Physical Surface 2: Outer sphere boundary

  // Set meshing options
  gmsh::option::setNumber("Mesh.ElementOrder", 2);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);

  // Generate the 3D mesh
  gmsh::model::mesh::generate(3);

  // Write the mesh to a file
  gmsh::write("spherical_offset.msh");

  // Launch the GUI to see the results (if not running with -nopopup)
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