#include <algorithm>  // For std::min
#include <cmath>      // For std::sqrt, std::pow, std::abs
#include <iostream>
#include <string>  // For std::string comparison in argument parsing
#include <vector>

// Include the main Gmsh C++ API header
#include <gmsh.h>

#include "common.hpp"

int main(int argc, char **argv) {
  gmsh::initialize(argc, argv);
  gmsh::option::setNumber("General.Terminal", 1);  // Print info to terminal

  gmsh::model::add("spherical_offset");

  gmsh::option::setNumber("Mesh.CharacteristicLengthMin", 0.1);
  gmsh::option::setNumber("Mesh.CharacteristicLengthMax", 0.1);

  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

  // Initial characteristic length for point creation
  const double lc_val = 0.01;

  gmsh::model::occ::addSphere(0, 0, 0, 1);

  gmsh::model::occ::synchronize();

  // Set meshing options
  gmsh::option::setNumber("Mesh.ElementOrder", 3);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);

  gmsh::model::mesh::generate(3);

  /*
      // Create the two spheres
      auto sphere_info = createSphere(0, 0, 0, 1, lc_val);
  int sl1 = sphere_info.first;
  std::vector<int> s_tags = sphere_info.second;

  int v1 = gmsh::model::geo::addVolume({sl1});

  // Remove duplicates (e.g., points, curves, surfaces that might be shared)
  gmsh::model::occ::removeAllDuplicates();  // Use OCC's removeAllDuplicates if
                                            // using OCC kernel
  gmsh::model::geo::synchronize();  // Synchronize the CAD kernel with Gmsh's
                                    // model

  // Physical surfaces for the boundaries
  gmsh::model::addPhysicalGroup(3, {v1}, 1);  // Physical Volume 1: Inner sphere
  gmsh::model::addPhysicalGroup(2, s_tags, 1);

  // Set meshing options
  gmsh::option::setNumber("Mesh.ElementOrder", 2);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);

  gmsh::model::mesh::generate(3);

  // Generate the 2D mesh
  // gmsh::model::mesh::generate(2);
  */

  std::vector<size_t> nodeTags;
  std::vector<double> coords;
  std::vector<double> parametricCoords;  // Not used
  gmsh::model::mesh::getNodes(nodeTags, coords, parametricCoords);

  // The `coords` vector is a flat list: [x1, y1, z1, x2, y2, z2, ...].
  // We will create a new vector for the updated coordinates.
  std::vector<double> newCoords(coords.size());

  for (size_t i = 0; i < nodeTags.size(); ++i) {
    // Get the coordinates for the current node
    double x = coords[i * 3 + 0];
    double y = coords[i * 3 + 1];
    double z = coords[i * 3 + 2];

    // Convert Cartesian to Spherical to get the angles.
    // Since our base mesh is a unit sphere, r = 1.
    double r = 1.0;
    double theta = acos(z / r);  // polar angle
    double phi = atan2(y, x);    // azimuthal angle

    // Get the new radius for this direction from our target function
    double new_r = r * (1 + 0.1 * r * r * sin(2 * theta) * sin(2 * phi));

    double new_x = new_r * x / r;
    double new_y = new_r * y / r;
    double new_z = new_r * z / r;

    // Update the coordinates for this node in Gmsh
    gmsh::model::mesh::setNode(nodeTags[i], {new_x, new_y, new_z}, {});
  }

  // Write the mesh to a file
  gmsh::write("sphere.msh");

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