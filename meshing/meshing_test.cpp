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

  /*
  auto circle1 =
      Circle(1, [](auto theta) { return 0.3 * std::sin(6 * theta); });
  auto circle2 = Circle(2);
  auto circle3 = Circle(4);
  auto circle4 = Circle(5);
  */

  auto circles = Circles();

  auto circle1 = Circle(1);
  circles.AddCircle(circle1);
  auto circle2 = Circle(20);
  circles.AddCircle(circle2);

  auto [bdr, dom] = circles.AddSurface();

  gmsh::model::occ::synchronize();

  for (auto i = 0; i < bdr.size(); i++) {
    gmsh::model::addPhysicalGroup(2, {dom[i]}, i + 1);
    gmsh::model::addPhysicalGroup(1, {bdr[i]}, i + 1);
  }

  gmsh::model::mesh::generate(2);

  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::write("occ_curves.msh");

  gmsh::fltk::run();

  gmsh::finalize();
  return 0;
}