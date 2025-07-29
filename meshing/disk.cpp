#include <algorithm>  // For std::min
#include <cmath>      // For std::sqrt, std::abs
#include <iostream>
#include <vector>

// Include the main Gmsh C++ API header
#include <gmsh.h>

// Custom mesh size callback function
double meshSizeCallback(int dim, int tag, double x, double y, double z,
                        double lc) {
  // Parameters from the Python script
  const double a = 1.0;
  const double b = 1.75;
  const double x0 = 0.5;
  const double y0 = 0.5;
  const double x1 = 0.3;
  const double y1 = 0.0;
  const double small = 0.01;
  const double big = 0.1;
  const double fac = 0.3;

  double r0 = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2));
  double r1 = std::sqrt(std::pow(x - x1, 2) + std::pow(y - y1, 2));

  double d0 = std::abs(r0 - a);
  double d1 = std::abs(r1 - b);

  double size = big;

  if (d0 < fac * a) {
    size = small + (big - small) * d0 / (fac * a);
  }

  if (d1 < fac * b) {
    size = std::min(size, (b / a) * (small + (big - small) * d1 / (fac * b)));
  }

  return size;
}

// Helper function to create a circle geometry
// Returns a pair: first is the curve loop tag, second is a vector of curve tags
std::pair<int, std::vector<int>> createCircle(double x, double y, double r,
                                              double lc_val) {
  int p1 = gmsh::model::geo::addPoint(x, y, 0, lc_val);
  int p2 = gmsh::model::geo::addPoint(x + r, y, 0, lc_val);
  int p3 = gmsh::model::geo::addPoint(x, y + r, 0, lc_val);
  int p4 = gmsh::model::geo::addPoint(x - r, y, 0, lc_val);
  int p5 = gmsh::model::geo::addPoint(x, y - r, 0, lc_val);

  int c1 = gmsh::model::geo::addCircleArc(p2, p1, p3);
  int c2 = gmsh::model::geo::addCircleArc(p3, p1, p4);
  int c3 = gmsh::model::geo::addCircleArc(p4, p1, p5);
  int c4 = gmsh::model::geo::addCircleArc(p5, p1, p2);

  std::vector<int> curve_tags = {c1, c2, c3, c4};
  int curve_loop_tag = gmsh::model::geo::addCurveLoop(curve_tags);

  return {curve_loop_tag, curve_tags};
}

int main(int argc, char **argv) {
  gmsh::initialize(argc, argv);
  gmsh::option::setNumber(
      "General.Terminal",
      1);  // Equivalent to Python's setNumber("General.Terminal", 1)

  gmsh::option::setNumber("Mesh.Nodes", 1);
  gmsh::option::setNumber("Mesh.VolumeFaces", 1);
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

  gmsh::model::add("circular_offset");

  // Set the custom mesh size callback
  // Note: In C++, you pass a function pointer
  gmsh::model::mesh::setSizeCallback(meshSizeCallback);

  // Parameters for circles
  const double a = 1.0;
  const double b = 1.75;
  const double x0 = 0.5;
  const double y0 = 0.5;
  const double x1 = 0.3;
  const double y1 = 0.0;
  const double lc = 0.1;  // This initial lc value is used for point creation

  // Create the two circles
  auto circle1_info = createCircle(x0, y0, a, lc);
  int l1 = circle1_info.first;
  std::vector<int> b1 = circle1_info.second;  // Boundary curves of circle 1

  auto circle2_info = createCircle(x1, y1, b, lc);
  int l2 = circle2_info.first;
  std::vector<int> b2 = circle2_info.second;  // Boundary curves of circle 2

  // Create plane surfaces
  // The Python script implies a 'difference' in v2's creation due to [l1, l2]
  // In Gmsh API, [l1, l2] implies a surface bounded by both loops.
  // If you explicitly wanted the difference (inner circle cut from outer),
  // you'd use a boolean operation, but for this specific setup, it's correct.
  int v1 = gmsh::model::geo::addPlaneSurface({l1});
  int v2 = gmsh::model::geo::addPlaneSurface(
      {l1, l2});  // Outer loop (l1) and inner loop (l2) define a surface

  gmsh::model::geo::synchronize();

  // Add Physical Groups
  gmsh::model::addPhysicalGroup(2, {v1},
                                1);  // Physical Surface 1: Inner circle
  gmsh::model::addPhysicalGroup(2, {v2},
                                2);  // Physical Surface 2: Area between circles

  // Note: For physical curves, you pass vectors directly
  gmsh::model::addPhysicalGroup(
      1, b1, 1);  // Physical Curve 1: Boundary of inner circle
  gmsh::model::addPhysicalGroup(
      1, b2, 2);  // Physical Curve 2: Boundary of outer circle

  // Set meshing options
  gmsh::option::setNumber("Mesh.ElementOrder", 3);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);

  // Generate the mesh (2D for surfaces)
  // The Python script had generate(3) which would try to make a volume mesh.
  // For 2D surfaces, you typically generate(2).
  // If you intend to extrude later for a volume, 3 is fine, but for just these
  // surfaces, 2 is correct.
  gmsh::model::mesh::generate(2);  // Generate 2D mesh on surfaces

  // Write the mesh to a file
  gmsh::write("circular_offset.msh");

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