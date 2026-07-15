#include <gmsh.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<double> parseString(const std::string &string_arg)
{
    std::vector<double> entries;
    std::istringstream iss(string_arg);
    std::string token;

    while (std::getline(iss, token, '-'))
    {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace),
                    token.end());
        entries.push_back(std::stod(token));
    }

    return entries;
}

void createConcentricCircularLayers(const std::vector<double> &input_radii,
                                    double meshSizeMin,
                                    double meshSizeMax,
                                    int elementOrder,
                                    int algorithm,
                                    const std::string &outputFileName)
{
    if (input_radii.size() < 1)
    {
        std::cerr << "Error: there should be at least one radius." << std::endl;
        return;
    }

    std::vector<double> radii = input_radii;
    std::sort(radii.begin(), radii.end());

    for (double r : radii)
    {
        if (r <= 0.0)
        {
            std::cerr << "Error: all radii must be positive." << std::endl;
            return;
        }
    }

    gmsh::initialize();
    gmsh::model::add("ConcentricCircularLayers");

    gmsh::option::setNumber("Mesh.MeshSizeMin", meshSizeMin);
    gmsh::option::setNumber("Mesh.MeshSizeMax", meshSizeMax);
    gmsh::option::setNumber("Mesh.Algorithm", algorithm);
    gmsh::option::setNumber("Mesh.ElementOrder", elementOrder);
    gmsh::option::setNumber("Mesh.HighOrderOptimize", 1);
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);

    int center = gmsh::model::geo::addPoint(0.0, 0.0, 0.0, meshSizeMin);

    std::vector<int> curveLoops;
    std::vector<std::vector<int>> circleCurves;

    for (int i = 0; i < static_cast<int>(radii.size()); i++)
    {
        double r = radii[i];

        int p1 = gmsh::model::geo::addPoint( r,  0.0, 0.0, meshSizeMin);
        int p2 = gmsh::model::geo::addPoint(0.0,  r,  0.0, meshSizeMin);
        int p3 = gmsh::model::geo::addPoint(-r, 0.0, 0.0, meshSizeMin);
        int p4 = gmsh::model::geo::addPoint(0.0, -r, 0.0, meshSizeMin);

        int c1 = gmsh::model::geo::addCircleArc(p1, center, p2);
        int c2 = gmsh::model::geo::addCircleArc(p2, center, p3);
        int c3 = gmsh::model::geo::addCircleArc(p3, center, p4);
        int c4 = gmsh::model::geo::addCircleArc(p4, center, p1);

        std::vector<int> curves = {c1, c2, c3, c4};
        circleCurves.push_back(curves);

        int loop = gmsh::model::geo::addCurveLoop(curves);
        curveLoops.push_back(loop);
    }

    std::vector<int> surfaceTags;

    int innerSurface = gmsh::model::geo::addPlaneSurface({curveLoops[0]});
    surfaceTags.push_back(innerSurface);

    for (int i = 1; i < static_cast<int>(radii.size()); i++)
    {
        int annulusSurface =
            gmsh::model::geo::addPlaneSurface({curveLoops[i], curveLoops[i - 1]});
        surfaceTags.push_back(annulusSurface);
    }

    gmsh::model::geo::synchronize();

    for (int i = 0; i < static_cast<int>(surfaceTags.size()); i++)
    {
        int attr = i + 1;
        gmsh::model::addPhysicalGroup(2, {surfaceTags[i]}, attr);
        gmsh::model::setPhysicalName(2, attr,
                                     "layer_" + std::to_string(attr));
    }

    for (int i = 0; i < static_cast<int>(circleCurves.size()); i++)
    {
        int attr = i + 1;
        gmsh::model::addPhysicalGroup(1, circleCurves[i], attr);
        gmsh::model::setPhysicalName(1, attr,
                                     "circle_" + std::to_string(attr));
    }

    std::vector<double> curvesList;
    for (const auto &curves : circleCurves)
    {
        for (int c : curves)
        {
            curvesList.push_back(static_cast<double>(c));
        }
    }

    gmsh::model::mesh::field::add("Distance", 1);
    gmsh::model::mesh::field::setNumbers(1, "CurvesList", curvesList);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", meshSizeMin);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", meshSizeMax);
    gmsh::model::mesh::field::setNumber(2, "DistMin", 0.0);
    gmsh::model::mesh::field::setNumber(2, "DistMax", 10.0 * meshSizeMin);
    gmsh::model::mesh::field::setAsBackgroundMesh(2);

    gmsh::model::mesh::generate(2);
    gmsh::write(outputFileName);

    gmsh::finalize();
}

int main(int argc, char **argv)
{
    std::vector<double> radii = {1.0, 1.2};
    double meshSizeMin = 0.02;
    double meshSizeMax = 0.06;
    int algorithm = 6;
    int elementOrder = 2;
    std::string outputFileName = "mesh/concentric_circular_layers.msh";

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-r" && i + 1 < argc)
        {
            radii = parseString(argv[++i]);
        }
        else if (arg == "-s" && i + 1 < argc)
        {
            std::string meshSizeStr = argv[++i];
            auto meshSizes = parseString(meshSizeStr);

            if (meshSizes.size() == 2)
            {
                meshSizeMin = meshSizes[0];
                meshSizeMax = meshSizes[1];
            }
            else
            {
                std::cerr << "Error: mesh sizes should have two values."
                          << std::endl;
                return 1;
            }
        }
        else if (arg == "-o" && i + 1 < argc)
        {
            elementOrder = std::stoi(argv[++i]);
        }
        else if (arg == "-ma" && i + 1 < argc)
        {
            algorithm = std::stoi(argv[++i]);
        }
        else if (arg == "-out" && i + 1 < argc)
        {
            outputFileName = argv[++i];
        }
    }

    createConcentricCircularLayers(radii,
                                   meshSizeMin,
                                   meshSizeMax,
                                   elementOrder,
                                   algorithm,
                                   outputFileName);

    return 0;
}
