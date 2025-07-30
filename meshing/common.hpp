#pragma once

#include <gmsh.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <ranges>
#include <vector>

// Simple struct for circles.
struct Circle {
  const double x0;
  const double y0;
  const double r;

  Circle(double x0, double y0, double r) : x0{x0}, y0{y0}, r{r} {}

  double DistanceTo(double x, double y) const {
    return std::hypot(x - x0, y - y0);
  }
};

class Circles {
 private:
  double _small;
  double _big;
  std::vector<Circle> _circles;

  void SetDefaultSizes() {
    _small = std::ranges::min(
        std::ranges::views::transform(_circles, [](auto c) { return c.r; }));
    _big = std::ranges::max(
        std::ranges::views::transform(_circles, [](auto c) { return c.r; }));
  }

 public:
  Circles(const std::vector<Circle>& circles) : _circles{circles} {
    SetDefaultSizes();
  }
  Circles(std::vector<Circle>&& circles) : _circles{circles} {
    SetDefaultSizes();
  }

  Circles(const std::vector<Circle>& circles, double small, double big)
      : _circles{circles}, _small{small}, _big{big} {}
  Circles(std::vector<Circle>&& circles, double small, double big)
      : _circles{circles}, _small{small}, _big{big} {}

  void SetSmall(double small) { _small = small; }
  void SetBig(double big) { _big = big; }

  double Small() const { return _small; }
  double Big() const { return _big; }

  double MeshSize(int dim, int tag, double x, double y, double z,
                  double size) const {
    auto distances = std::ranges::views::transform(
        _circles, [x, y](auto c) { return c.DistanceTo(x, y); });

    return size;
  }
};

// Simple stuct for spheres.
struct Sphere {
  double x0;
  double y0;
  double z0;
  double r;

  Sphere(double x0, double y0, double z0, double r)
      : x0{x0}, y0{y0}, z0{z0}, r{r} {}
};

// Helper function to create a circle geometry
// Returns a pair: first is the curve loop tag, second is a vector of curve tags
std::pair<int, std::vector<int>> createCircle(Circle&& circle, double size);

// Helper function to create a circle geometry
// Returns a pair: first is the curve loop tag, second is a vector of curve tags
std::pair<int, std::vector<int>> createCircle(double x, double y, double r,
                                              double lc_val);

// Helper function to create a sphere geometry
// Returns a pair: first is the surface loop tag, second is a vector of surface
// tags
std::pair<int, std::vector<int>> createSphere(double x, double y, double z,
                                              double r, double lc_val);