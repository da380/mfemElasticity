#include <gmsh.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void createConcentricSphericalLayers(const std::vector<double> &radii,
                                     double meshSizeMin, double meshSizeMax,
                                     int elementOrder, int algorithm,
                                     const std::string &outputFileName) {
  int numLayers = radii.size();

  if (numLayers < 1) {
    std::cerr << "Error: There should be at least one layer." << std::endl;
    return;
  }

  gmsh::initialize();
  gmsh::model::add("ConcentricSphericalLayers");

  gmsh::option::setNumber("Mesh.MeshSizeMin", meshSizeMin);
  gmsh::option::setNumber("Mesh.MeshSizeMax", meshSizeMax);

  for (int i = 0; i < numLayers; ++i) {
    gmsh::model::occ::addSphere(0, 0, 0, radii[i]);
  }
  gmsh::model::occ::synchronize();
  // Innest layer
  int layerTag = 1;
  int surfaceTag = 1;
  gmsh::model::addPhysicalGroup(3, {1}, layerTag);
  gmsh::model::setPhysicalName(3, layerTag, "layer_1");
  std::vector<std::pair<int, int>> surfaceEntities;
  gmsh::model::getBoundary({{3, 1}}, surfaceEntities, false, false,
                           false);  // combined - oriented - recursive
  std::pair<int, int> surface = surfaceEntities[0];
  gmsh::model::addPhysicalGroup(2, {surface.second}, surfaceTag);
  gmsh::model::setPhysicalName(2, surfaceTag, "surface_" + std::to_string(1));
  // Other layers
  for (int i = 1; i < numLayers; ++i) {
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int>>> ovv;

    gmsh::model::occ::cut(
        {{3, i + 1}}, {{3, i}}, ov, ovv, -1, false,
        false);  // auto-assigns tags - removeObject - removeTool
    gmsh::model::occ::synchronize();

    std::vector<int> volumeTags;
    for (const auto &entity : ov) {
      volumeTags.push_back(entity.second);
    }
    ++layerTag;
    gmsh::model::addPhysicalGroup(3, volumeTags, layerTag);
    gmsh::model::setPhysicalName(3, layerTag, "layer_" + std::to_string(i + 1));
    for (const auto &volumeTag : volumeTags) {
      std::vector<std::pair<int, int>> surfaceEntities;
      gmsh::model::getBoundary({{3, volumeTag}}, surfaceEntities, false, false,
                               false);
      std::pair<int, int> surface =
          surfaceEntities[0];  // Only take the inner surface
      ++surfaceTag;
      gmsh::model::addPhysicalGroup(2, {surface.second}, surfaceTag);
      gmsh::model::setPhysicalName(2, surfaceTag,
                                   "surface_" + std::to_string(i + 1));
    }
  }

  for (int i = 1; i < numLayers; ++i) {
    gmsh::model::occ::remove({{3, i + 1}});
  }
  gmsh::model::occ::synchronize();

  std::vector<double> facesList(numLayers);
  for (int i = 0; i < numLayers; ++i) facesList[i] = i + 1;

  gmsh::model::mesh::field::add("Distance", 1);
  gmsh::model::mesh::field::setNumbers(1, "FacesList", facesList);

  gmsh::model::mesh::field::add("Threshold", 2);
  gmsh::model::mesh::field::setNumber(2, "InField", 1);
  gmsh::model::mesh::field::setNumber(2, "SizeMin", meshSizeMin);
  gmsh::model::mesh::field::setNumber(2, "SizeMax", meshSizeMax);
  gmsh::model::mesh::field::setNumber(2, "DistMin", 0.0);
  double fac = 10.0;
  gmsh::model::mesh::field::setNumber(2, "DistMax", meshSizeMin * fac);
  gmsh::model::mesh::field::setAsBackgroundMesh(2);

  gmsh::option::setNumber(
      "Mesh.Algorithm3D",
      algorithm);  // 1-Delaunay, 4-Frontal, 7-MMG3D, 9-R-tree Delaunay, 10-HXT
                   // (Frontal-Delaunay), 11-Automatic
  gmsh::option::setNumber("Mesh.ElementOrder", elementOrder);
  gmsh::option::setNumber("Mesh.HighOrderOptimize", 1);
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::model::mesh::generate(3);
  gmsh::write(outputFileName);

  gmsh::finalize();
}

std::vector<double> parseString(const std::string &string_arg) {
  std::vector<double> entries;
  std::istringstream iss(string_arg);
  std::string token;

  while (std::getline(iss, token, '-')) {
    token.erase(std::remove_if(token.begin(), token.end(), ::isspace),
                token.end());
    entries.push_back(std::stod(token));
  }

  return entries;
}

int main(int argc, char **argv) {
  std::vector<double> radii = {1, 2};
  double meshSizeMin = 0.02;
  double meshSizeMax = 0.06;
  int algorithm = 1;
  int elementOrder = 2;
  std::string outputFileName = "mesh/concentric_spherical_layers.msh";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-r" && i + 1 < argc) {
      radii = parseString(argv[++i]);
    } else if (arg == "-s" && i + 1 < argc) {
      std::string meshSizeStr = argv[++i];
      auto meshSizes = parseString(meshSizeStr);
      if (meshSizes.size() == 2) {
        meshSizeMin = meshSizes[0];
        meshSizeMax = meshSizes[1];
      } else {
        std::cerr << "Error: mesh sizes should have two values.\n";
        return 1;
      }
    } else if (arg == "-o" && i + 1 < argc) {
      elementOrder = std::stoi(argv[++i]);
    } else if (arg == "-ma" && i + 1 < argc) {
      algorithm = std::stod(argv[++i]);
    } else if (arg == "-out" && i + 1 < argc) {
      outputFileName = argv[++i];
    }
  }

  createConcentricSphericalLayers(radii, meshSizeMin, meshSizeMax, elementOrder,
                                  algorithm, outputFileName);

  return 0;
}
