
#pragma once

#include <gmsh.h>

#include <cmath>
#include <functional>
#include <numbers>

class Circle {
 private:
  const double x_ = 0;
  const double y_ = 0;
  const double r_ = 1;
  std::function<double(double)> f_ = nullptr;

 public:
  Circle() = default;

  Circle(double r) : r_{r} {}

  Circle(double r, std::function<double(double)> f) : r_{r}, f_{f} {}

  Circle(double x, double y, double r, std::function<double(double)> f)
      : x_{x}, y_{y}, r_{r}, f_{std::move(f)} {}

  Circle(double x, double y, double r) : x_{x}, y_{y}, r_{r} {}

  Circle(std::function<double(double)> f) : f_{std::move(f)} {}

  double DistanceTo(double x, double y) const {
    auto r = std::hypot(x - x_, y - y_);
    return std::abs(r - r_);
  }

  double Topo(double theta) const {
    if (f_) {
      return f_(theta);
    } else {
      return 0;
    }
  }

  int AddCurve(int tag = -1, int np = 90) const {
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

  int AddCurveLoop(int tag = -1, int np = 90) const {
    return gmsh::model::occ::addCurveLoop({AddCurve(tag, np)});
  }

  std::pair<int, int> AddSurface(int tag = -1, int np = 90) const {
    auto c = AddCurve(-1, np);
    auto cl = gmsh::model::occ::addCurveLoop({c});
    auto s = gmsh::model::occ::addPlaneSurface({cl}, tag);
    return {cl, s};
  }
};