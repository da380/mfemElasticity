
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

  static const int np_ = 180;

 public:
  Circle() = default;

  Circle(double r) : r_{r} {}

  Circle(double r, std::function<double(double)> f) : r_{r}, f_{f} {}

  Circle(double x, double y, double r, std::function<double(double)> f)
      : x_{x}, y_{y}, r_{r}, f_{std::move(f)} {}

  Circle(double x, double y, double r) : x_{x}, y_{y}, r_{r} {}

  Circle(std::function<double(double)> f) : f_{std::move(f)} {}

  double DistanceTo(double x, double y) const;

  int AddCurve(int tag = -1, int np = np_) const;

  int AddCurveLoop(int tag = -1, int np = np_) const;

  std::pair<int, int> AddSurface(int tag = -1, int np = np_) const;
};

class Circles {
 private:
  std::vector<Circle> circles_;

 public:
  Circles() = default;

  Circles(std::vector<Circle> circles) : circles_{circles} {}

  void AddCircle(Circle circle) { circles_.push_back(circle); }

  std::pair<std::vector<int>, std::vector<int>> AddSurface() const;
};
