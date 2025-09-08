#include "CircularMesh.hpp"

double Circle::DistanceTo(double x, double y) const {
  auto r = std::hypot(x - x_, y - y_);
  return std::abs(r - r_);
}

int Circle::AddCurve(int tag, int np) const {
  if (f_) {
    auto dtheta = 2 * std::numbers::pi / np;
    std::vector<int> points;
    for (auto i = 0; i < np; i++) {
      auto theta = i * dtheta;
      auto r = r_ + f_(theta);
      auto x = x_ + r * std::cos(theta);
      auto y = y_ + r * std::sin(theta);
      auto p = gmsh::model::occ::addPoint(x, y, 0);
      points.push_back(p);
    }
    points.push_back(points.front());
    return gmsh::model::occ::addBSpline(points);
  } else {
    return gmsh::model::occ::addCircle(x_, y_, 0, r_, tag);
  }
}

int Circle::AddCurveLoop(int tag, int np) const {
  return gmsh::model::occ::addCurveLoop({AddCurve(tag, np)});
}

std::pair<int, int> Circle::AddSurface(int tag, int np) const {
  auto c = AddCurve(-1, np);
  auto cl = gmsh::model::occ::addCurveLoop({c});
  auto s = gmsh::model::occ::addPlaneSurface({cl}, tag);
  return {cl, s};
}

std::pair<std::vector<int>, std::vector<int>> Circles::AddSurface() const {
  auto bdr = std::vector<int>();
  auto dom = std::vector<int>();

  for (auto circle : circles_) {
    auto c = circle.AddCurveLoop();
    bdr.push_back(c);
  }

  auto s = gmsh::model::occ::addPlaneSurface({bdr[0]});
  dom.push_back(s);
  for (auto i = 1; i < circles_.size(); i++) {
    auto s = gmsh::model::occ::addPlaneSurface({bdr[i - 1], bdr[i]});
    dom.push_back(s);
  }

  return {bdr, dom};
}