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
const double x_0 = 0.0;
const double y_0 = 0.0;
const double z_0 = 0.0;
const double a = 1.0;
const double small = 0.05;
const double big = 0.1;
const double fac = 0.2;

// Custom mesh size callback function
// The signature must match gmsh::model::mesh::setSizeCallback's expectation
double meshSizeCallback(int dim, int tag, double x, double y, double z,
                        double lc) {
  double r0 = std::sqrt(std::pow(x - x_0, 2) + std::pow(y - y_0, 2) +
                        std::pow(z - z_0, 2));

  double d0 = std::abs(r0 - a);

  return d0 < fac * a ? small + (big - small) * d0 / (fac * a) : big;
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
  auto sphere1_info = createSphere(x_0, y_0, z_0, a, lc_val);
  int sl1 = sphere1_info.first;
  std::vector<int> s_tags1 =
      sphere1_info.second;  // Surface tags of inner sphere

  // Create volumes
  // v1 is the inner sphere volume
  int v1 = gmsh::model::geo::addVolume({sl1});

  gmsh::model::geo::synchronize();  // Synchronize the CAD kernel with Gmsh's
                                    // model

  // Add Physical Groups for volumes and surfaces
  gmsh::model::addPhysicalGroup(3, {v1}, 1);  // Physical Volume 1: Inner sphere

  // Physical surfaces for the boundaries
  gmsh::model::addPhysicalGroup(
      2, s_tags1, 1);  // Physical Surface 1: Inner sphere boundary

  // Set meshing options
  gmsh::option::setNumber("Mesh.ElementOrder", 2);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);

  // Generate the 3D mesh
  gmsh::model::mesh::generate(3);

  // Write the mesh to a file
  gmsh::write("ball.msh");

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