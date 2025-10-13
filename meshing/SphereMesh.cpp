#include "SphereMesh.hpp"

double Sphere::DistanceTo(double x, double y, double z) const {
  return std::sqrt((x - x_) * (x - x_) + (y - y_) * (y - y_) +
                   (z - z_) * (z - z_));
}

int Sphere::AddArc(double theta1, double theta2, double phi1, double phi2,
                   int tag, int np) const {
  using std::sin, std::cos;
  auto points = std::vector<int>(np);
  auto dtheta = (theta2 - theta1) / (np - 1);
  auto dphi = (phi2 - phi1) / (np - 1);
  for (auto i = 0; i < np; i++) {
    auto theta = theta1 + i * dtheta;
    auto phi = phi1 + i * dphi;
    auto r = f_ ? r_ + f_(theta, phi) : r_;
    auto x = x_ + r * sin(theta) * cos(phi);
    auto y = y_ + r * sin(theta) * sin(phi);
    auto z = z_ + r * cos(theta);
    points[i] = gmsh::model::occ::addPoint(x, y, z);
  }
  return gmsh::model::occ::addBSpline(points);
}

void Sphere::AppendPointsAlongArc(double theta1, double theta2, double phi1,
                                  double phi2, std::vector<int>& points,
                                  int np) const {
  auto dtheta = (theta2 - theta1) / np;
  auto dphi = (phi2 - phi1) / np;
  for (auto i = 1; i < np; i++) {
    auto theta = theta1 + i * dtheta;
    auto phi = phi1 + i * dphi;
    auto r = f_ ? r_ + f_(theta, phi) : r_;
    auto x = x_ + r * sin(theta) * cos(phi);
    auto y = y_ + r * sin(theta) * sin(phi);
    auto z = z_ + r * cos(theta);
    points.push_back(gmsh::model::occ::addPoint(x, y, z));
  }
}

std::vector<int> Sphere::SetPointsWithinPatch(double theta1, double theta2,
                                              double phi1, double phi2,
                                              int np) const {
  auto points = std::vector<int>();
  auto dtheta = (theta2 - theta1) / np;
  for (auto i = 0; i < np; i++) {
    auto theta = theta1 + i * dtheta;
    AppendPointsAlongArc(theta, theta, phi1, phi2, points, i + 1);
  }
  return points;
}

std::pair<int, std::vector<int>> Sphere::AddSurface(int tag, int np) const {
  std::vector<int> surface_tags;
  int surface_loop_tag;

  // Build the necessary arcs.
  auto c1 = AddArc(0, pi / 2, 0, 0, -1, np);
  auto c2 = AddArc(0, pi / 2, pi / 2, pi / 2, -1, np);
  auto c3 = AddArc(0, pi / 2, pi, pi, -1, np);
  auto c4 = AddArc(0, pi / 2, 3 * pi / 2, 3 * pi / 2, -1, np);
  auto c5 = AddArc(pi / 2, pi / 2, 0, pi / 2, -1, np);
  auto c6 = AddArc(pi / 2, pi / 2, pi / 2, pi, -1, np);
  auto c7 = AddArc(pi / 2, pi / 2, pi, 3 * pi / 2, -1, np);
  auto c8 = AddArc(pi / 2, pi / 2, 3 * pi / 2, 2 * pi, -1, np);
  auto c9 = AddArc(pi, pi / 2, 0, 0, -1, np);
  auto c10 = AddArc(pi, pi / 2, pi / 2, pi / 2, -1, np);
  auto c11 = AddArc(pi, pi / 2, pi, pi, -1, np);
  auto c12 = AddArc(pi, pi / 2, 3 * pi / 2, 3 * pi / 2, -1, np);

  // Define curve loops for each "patch" of the sphere
  auto loops = std::vector<int>();
  loops.push_back(gmsh::model::occ::addCurveLoop({c1, c5, -c2}));
  loops.push_back(gmsh::model::occ::addCurveLoop({c2, c6, -c3}));
  loops.push_back(gmsh::model::occ::addCurveLoop({c3, c7, -c4}));
  loops.push_back(gmsh::model::occ::addCurveLoop({c4, c8, -c1}));
  loops.push_back(gmsh::model::occ::addCurveLoop({c9, c5, -c10}));
  loops.push_back(gmsh::model::occ::addCurveLoop({c10, c6, -c11}));
  loops.push_back(gmsh::model::occ::addCurveLoop({c11, c7, -c12}));
  loops.push_back(gmsh::model::occ::addCurveLoop({c12, c8, -c9}));

  // Create surfaces from the curve loops using SurfaceFilling
  for (auto i = 0; i < 8; i++) {
    auto theta1 = i < 4 ? 0 : pi;
    auto theta2 = pi / 2;
    auto j = i % 4;
    auto phi1 = j * pi / 2;
    auto phi2 = (j + 1) * pi / 2;
    auto points = SetPointsWithinPatch(theta1, theta2, phi1, phi2, np);

    surface_tags.push_back(
        gmsh::model::occ::addSurfaceFilling({loops[i]}, -1, points));
    for (auto point : points) {
      gmsh::model::occ::remove({{0, point}});
    }
  }
  surface_loop_tag = gmsh::model::occ::addSurfaceLoop(surface_tags);

  return {surface_loop_tag, surface_tags};
}