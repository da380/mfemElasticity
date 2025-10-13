#pragma once

#include <gmsh.h>

#include <cmath>
#include <functional>
#include <numbers>

class Sphere {
 private:
  static constexpr double pi = std::numbers::pi;
  const double x_ = 0;
  const double y_ = 0;
  const double z_ = 0;
  const double r_ = 1;
  std::function<double(double, double)> f_ = nullptr;

  static const int np_ = 30;

 public:
  Sphere() = default;

  Sphere(double r) : r_{r} {}

  Sphere(double r, std::function<double(double, double)> f) : r_{r}, f_{f} {}

  Sphere(double x, double y, double z, double r,
         std::function<double(double, double)> f)
      : x_{x}, y_{y}, z_{z}, r_{r}, f_{std::move(f)} {}

  Sphere(double x, double y, double z, double r) : x_{x}, y_{y}, z_{z}, r_{r} {}

  Sphere(std::function<double(double, double)> f) : f_{std::move(f)} {}

  double DistanceTo(double x, double y, double z) const;

  int AddArc(double theta1, double theta2, double phi1, double phi2,
             int tag = -1, int np = np_) const;

  void AppendPointsAlongArc(double theta1, double theta2, double phi1,
                            double phi2, std::vector<int>& points,
                            int np = np_) const;

  std::vector<int> SetPointsWithinPatch(int patch, int np = np_) const;

  std::vector<int> SetPointsWithinPatch(double theta1, double theta2,
                                        double phi1, double phi2,
                                        int np = np_) const;

  std::pair<int, std::vector<int>> AddSurface(int tag = -1, int np = np_) const;
};