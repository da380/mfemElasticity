#include <algorithm>  // For std::min
#include <cmath>      // For std::sqrt, std::pow, std::abs
#include <iostream>
#include <string>  // For std::string comparison in argument parsing
#include <vector>

// Include the main Gmsh C++ API header
#include <gmsh.h>

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

// Helper function to create a sphere geometry
// Returns a pair: first is the surface loop tag, second is a vector of surface
// tags
std::pair<int, std::vector<int>> createSphere(double x, double y, double z,
                                              double r, double lc_val) {
  // Define points on the sphere and center
  int p1 = gmsh::model::geo::addPoint(x, y, z, lc_val);      // Center
  int p2 = gmsh::model::geo::addPoint(x + r, y, z, lc_val);  // +X
  int p3 = gmsh::model::geo::addPoint(x, y + r, z, lc_val);  // +Y
  int p4 = gmsh::model::geo::addPoint(x, y, z + r, lc_val);  // +Z
  int p5 = gmsh::model::geo::addPoint(x - r, y, z, lc_val);  // -X
  int p6 = gmsh::model::geo::addPoint(x, y - r, z, lc_val);  // -Y
  int p7 = gmsh::model::geo::addPoint(x, y, z - r, lc_val);  // -Z

  // Define circular arcs (edges of the sphere's "patches")
  int c1 = gmsh::model::geo::addCircleArc(p2, p1, p7);
  int c2 = gmsh::model::geo::addCircleArc(p7, p1, p5);
  int c3 = gmsh::model::geo::addCircleArc(p5, p1, p4);
  int c4 = gmsh::model::geo::addCircleArc(p4, p1, p2);
  int c5 = gmsh::model::geo::addCircleArc(p2, p1, p3);
  int c6 = gmsh::model::geo::addCircleArc(p3, p1, p5);
  int c7 = gmsh::model::geo::addCircleArc(p5, p1, p6);
  int c8 = gmsh::model::geo::addCircleArc(p6, p1, p2);
  int c9 = gmsh::model::geo::addCircleArc(p7, p1, p3);
  int c10 = gmsh::model::geo::addCircleArc(p3, p1, p4);
  int c11 = gmsh::model::geo::addCircleArc(p4, p1, p6);
  int c12 = gmsh::model::geo::addCircleArc(p6, p1, p7);

  // Define curve loops for each "patch" of the sphere
  int l1 = gmsh::model::geo::addCurveLoop({c5, c10, c4});
  int l2 = gmsh::model::geo::addCurveLoop({c9, -c5, c1});
  int l3 = gmsh::model::geo::addCurveLoop({c12, -c8, -c1});
  int l4 = gmsh::model::geo::addCurveLoop({c8, -c4, c11});
  int l5 = gmsh::model::geo::addCurveLoop({-c10, c6, c3});
  int l6 = gmsh::model::geo::addCurveLoop({-c11, -c3, c7});
  int l7 = gmsh::model::geo::addCurveLoop({-c2, -c7, -c12});
  int l8 = gmsh::model::geo::addCurveLoop({-c6, -c9, c2});

  // Create surfaces from the curve loops using SurfaceFilling
  int s1 = gmsh::model::geo::addSurfaceFilling({l1});
  int s2 = gmsh::model::geo::addSurfaceFilling({l2});
  int s3 = gmsh::model::geo::addSurfaceFilling({l3});
  int s4 = gmsh::model::geo::addSurfaceFilling({l4});
  int s5 = gmsh::model::geo::addSurfaceFilling({l5});
  int s6 = gmsh::model::geo::addSurfaceFilling({l6});
  int s7 = gmsh::model::geo::addSurfaceFilling({l7});
  int s8 = gmsh::model::geo::addSurfaceFilling({l8});

  std::vector<int> surface_tags = {s1, s2, s3, s4, s5, s6, s7, s8};
  int surface_loop_tag = gmsh::model::geo::addSurfaceLoop(surface_tags);

  return {surface_loop_tag, surface_tags};
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