#include <gmsh.h>

#include <iostream>
#include <vector>

#include "CircularMesh.hpp"

int main(int argc, char **argv) {
  gmsh::initialize(argc, argv);

  gmsh::model::add("example");

  gmsh::option::setNumber("Mesh.CharacteristicLengthMin", 0.1);
  gmsh::option::setNumber("Mesh.CharacteristicLengthMax", 0.1);

  gmsh::option::setNumber("Mesh.ElementOrder", 3);
  gmsh::option::setNumber("Mesh.MeshOnlyVisible", 1);

  gmsh::option::setNumber("Mesh.Nodes", 1);
  gmsh::option::setNumber("Mesh.VolumeFaces", 1);
  gmsh::option::setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0);

  auto circle1 =
      Circle(1, [](auto theta) { return 0.2 * std::sin(6 * theta); });
  auto circle2 = Circle(2);

  auto c1 = circle1.AddCurveLoop();
  auto c2 = circle2.AddCurveLoop();

  auto s1 = gmsh::model::occ::addPlaneSurface({c1});
  auto s2 = gmsh::model::occ::addPlaneSurface({c1, c2});

  gmsh::model::occ::synchronize();

  // Add Physical Groups
  gmsh::model::addPhysicalGroup(2, {s1},
                                1);  // Physical Surface 1: Inner circle
  gmsh::model::addPhysicalGroup(2, {s2},
                                2);  // Physical Surface 2: Area between circles

  // Note: For physical curves, you pass vectors directly
  gmsh::model::addPhysicalGroup(
      1, {c1}, 1);  // Physical Curve 1: Boundary of inner circle
  gmsh::model::addPhysicalGroup(
      1, {c2}, 2);  // Physical Curve 2: Boundary of outer circle

  gmsh::model::mesh::generate(2);

  gmsh::write("occ_curves.msh");

  gmsh::fltk::run();

  gmsh::finalize();
  return 0;

  /*

  gmsh::model::add("occ_curves_example");  // Add a new model

  double lc = 0.01;  // Characteristic length for meshing

  // --- Straight Lines (connected in sequence) ---
  // Define points using OCC for straight lines
  int p1_occ_tag = gmsh::model::occ::addPoint(0, 0, 0, lc);
  int p2_occ_tag = gmsh::model::occ::addPoint(1, 0, 0, lc);
  int p3_occ_tag = gmsh::model::occ::addPoint(1, 1, 0, lc);
  int p4_occ_tag = gmsh::model::occ::addPoint(0, 1, 0, lc);

  int p5_occ_tag = gmsh::model::occ::addPoint(0.5, 0.5, 0.1, lc);
  int p6_occ_tag = gmsh::model::occ::addPoint(0.2, 0.5, -0.2, lc);

  // Create straight lines between consecutive points
  int l1_occ_tag = gmsh::model::occ::addLine(p1_occ_tag, p2_occ_tag);
  int l2_occ_tag = gmsh::model::occ::addLine(p2_occ_tag, p3_occ_tag);
  int l3_occ_tag = gmsh::model::occ::addLine(p3_occ_tag, p4_occ_tag);
  int l4_occ_tag =
      gmsh::model::occ::addLine(p4_occ_tag, p1_occ_tag);  // Close the square

  // You can now define a curve loop and surface from the straight lines
  std::vector<int> square_lines = {l1_occ_tag, l2_occ_tag, l3_occ_tag,
                                   l4_occ_tag};
  int square_curve_loop = gmsh::model::occ::addCurveLoop(
      square_lines);  // Note: using geo namespace for curve loops/surfaces
  // gmsh::model::geo::addPlaneSurface({square_curve_loop});

  gmsh::model::occ::addSurfaceFilling(square_curve_loop, -1,
                                      {p5_occ_tag, p6_occ_tag});

  // --- Essential Step: Synchronize OCC entities with Gmsh's model ---
  gmsh::model::occ::synchronize();

  // --- Final steps ---
  gmsh::model::geo::synchronize();  // Synchronize again after adding geo
                                    // entities (if any)

  // Optional: Generate 2D mesh to visualize the curves
  gmsh::model::mesh::generate(2);

  // Save the mesh file (e.g., as .msh or .geo for geometry only)
  gmsh::write("occ_curves.msh");

  // To visualize the result directly, uncomment the following:
  gmsh::fltk::run();

  gmsh::finalize();
  return 0;
  */
}