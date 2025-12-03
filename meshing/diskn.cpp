#include <gmsh.h>

#include <algorithm>   // For std::sort, std::min
#include <cmath>       // For std::sqrt, std::abs
#include <functional>  // For std::function
#include <iostream>
#include <set>        // For std::set
#include <stdexcept>  // For std::runtime_error
#include <string>     // For std::string
#include <vector>

// Your meshing parameters, now as global constants
const double small = 0.01;
const double big = 0.1;
const double fac = 0.2;

int main(int argc, char **argv) {
  gmsh::initialize(argc, argv);

  // --- All your options are preserved ---
  gmsh::option::setNumber("General.Terminal", 1);
  gmsh::option::setNumber("Mesh.Nodes", 1);
  gmsh::option::setNumber("Mesh.VolumeFaces", 1);
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);
  gmsh::option::setNumber("Mesh.ElementOrder", 3);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);

  gmsh::model::add("nested_disks_n");

  // --- 1. Define Radii and Generalized Mesh Callback ---

  // Define your list of radii. Can be in any order.
  std::vector<double> radii = {2.0, 4.0, 1.0, 3.0, 10};

  // Sort the radii: smallest to largest. This is crucial for the logic.
  std::sort(radii.begin(), radii.end());

  int n_layers = radii.size();
  if (n_layers < 1) {  // 1 layer is valid, it's just one disk
    throw std::runtime_error("Need at least 1 disk.");
  }

  // Define the mesh size callback as a C++ lambda function
  // This allows it to "capture" the radii vector from main.
  std::function<double(int, int, double, double, double, double)>
      meshSizeCallback = [&radii](int dim, int tag, double x, double y,
                                  double z, double lc) -> double {
    double r0 = std::sqrt(x * x + y * y);
    double size = big;
    double r_min = radii[0];  // Innermost radius for scaling (your 'a')

    // Loop over all boundary radii
    for (double radius : radii) {
      double d = std::abs(r0 - radius);
      // If we are close to this boundary
      if (d < fac * radius) {
        // This generalizes your (b/a) scaling factor.
        // For r=a, (a/a) = 1. For r=b, (b/a). For r=c, (c/a).
        double scaling_factor = radius / r_min;
        double refined_size =
            scaling_factor * (small + (big - small) * d / (fac * radius));

        // Take the smallest size if we are near multiple boundaries
        size = std::min(size, refined_size);
      }
    }
    return size;
  };

  // Set the C++ std::function as the callback
  gmsh::model::mesh::setSizeCallback(meshSizeCallback);

  // --- 2. Create Geometry ---

  double x = 0, y = 0, z = 0;

  // The "Object" is the single largest disk
  double largest_radius = radii.back();
  int object_tag =
      gmsh::model::occ::addDisk(x, y, z, largest_radius, largest_radius);
  std::pair<int, int> object_entity = {2, object_tag};

  // The "Tools" are all the other smaller disks
  std::vector<std::pair<int, int>> tool_entities;
  for (size_t i = 0; i < n_layers - 1; ++i) {
    int tool_tag = gmsh::model::occ::addDisk(x, y, z, radii[i], radii[i]);
    tool_entities.push_back({2, tool_tag});
  }

  // --- 3. Fragment the Disks ---
  std::vector<std::pair<int, int>> outDimTags;
  std::vector<std::vector<std::pair<int, int>>> outDimTagsMap;

  // Handle n=1 case (single disk, no fragment)
  if (n_layers > 1) {
    gmsh::model::occ::fragment({object_entity}, tool_entities, outDimTags,
                               outDimTagsMap);
  }

  gmsh::model::occ::synchronize();

  // --- 4. Identify New Surface and Curve Tags ---

  // We will have n_layers surfaces (1 inner disk + (n-1) annuli)
  // We will have n_layers curves (boundaries)
  std::vector<int> surface_tags(n_layers);
  std::vector<int> curve_tags(n_layers);

  if (n_layers == 1) {
    // Simple case: just one disk
    surface_tags[0] = object_tag;
  } else {
    // General case: Use the map
    std::set<int> surfaces_found_so_far;

    // The innermost disk (Surface 0) is from the map of the smallest tool
    surface_tags[0] = outDimTagsMap[1][0].second;
    surfaces_found_so_far.insert(surface_tags[0]);

    // Find annuli 1 to n-2
    for (size_t i = 1; i < n_layers - 1; ++i) {
      for (const auto &pair : outDimTagsMap[i + 1]) {
        if (surfaces_found_so_far.find(pair.second) ==
            surfaces_found_so_far.end()) {
          surface_tags[i] = pair.second;
          surfaces_found_so_far.insert(pair.second);
          break;
        }
      }
    }

    // Find the outermost annulus (Surface n-1)
    for (const auto &pair : outDimTagsMap[0]) {
      if (surfaces_found_so_far.find(pair.second) ==
          surfaces_found_so_far.end()) {
        surface_tags[n_layers - 1] = pair.second;
        break;
      }
    }
  }

  // --- Find Boundaries ---
  std::set<int> curves_found_so_far;

  // Curve 0 is the boundary of Surface 0 (innermost disk)
  std::vector<std::pair<int, int>> bnd_pairs;
  gmsh::model::getBoundary({{2, surface_tags[0]}}, bnd_pairs, false, false,
                           false);

  // Handle case of n=1 (center point is a boundary)
  if (n_layers == 1 && radii[0] == 0) {
    // Special case: disk at r=0, boundary is just a point
    // Not creating a curve group. Or we could, depends on need.
    // Let's assume radii are always > 0 for this problem.
  }

  if (bnd_pairs.empty()) {
    throw std::runtime_error("Innermost surface has no boundary.");
  }
  curve_tags[0] = bnd_pairs[0].second;
  curves_found_so_far.insert(curve_tags[0]);

  // Annulus `i` (surface_tags[i]) is bounded by Curve `i-1` and Curve `i`
  for (size_t i = 1; i < n_layers; ++i) {
    std::vector<std::pair<int, int>> current_bnd_pairs;
    gmsh::model::getBoundary({{2, surface_tags[i]}}, current_bnd_pairs, false,
                             false, false);

    for (const auto &pair : current_bnd_pairs) {
      if (curves_found_so_far.find(pair.second) == curves_found_so_far.end()) {
        curve_tags[i] = pair.second;
        curves_found_so_far.insert(pair.second);
        break;
      }
    }
  }

  // --- 5. Create Physical Groups (1-based) ---
  std::cout << "Creating Physical Groups...\n";

  // Create Physical Surfaces (Dim 2)
  for (size_t i = 0; i < n_layers; ++i) {
    int physical_tag = i + 1;  // 1-based tag (1, 2, 3, ...)
    gmsh::model::addPhysicalGroup(2, {surface_tags[i]}, physical_tag);
    std::cout << " - Surface Physical Tag: " << physical_tag
              << " (for geometric surface " << surface_tags[i] << ")\n";
  }

  // Create Physical Curves (Dim 1)
  for (size_t i = 0; i < n_layers; ++i) {
    int physical_tag = i + 1;  // 1-based tag (1, 2, 3, ...)
    gmsh::model::addPhysicalGroup(1, {curve_tags[i]}, physical_tag);
    std::cout << " - Curve Physical Tag:   " << physical_tag
              << " (for geometric curve " << curve_tags[i] << ")\n";
  }

  // --- Generate mesh and visualize ---
  gmsh::model::mesh::generate(2);

  gmsh::write("nested_disks_n.msh");

  // Your -nopopup logic is preserved
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