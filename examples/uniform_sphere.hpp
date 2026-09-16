#pragma once

#include <cmath>
#include <numbers>

#include "mfem.hpp"

class UniformSphereSolution {
 private:
  static constexpr mfem::real_t pi = std::numbers::pi_v<mfem::real_t>;

  int dim_;
  mfem::real_t r_;

  mfem::Vector x_;

 public:
  UniformSphereSolution(int dim, const mfem::Vector& x, mfem::real_t r)
      : dim_{dim}, x_{x}, r_{r} {}

  mfem::FunctionCoefficient Coefficient() const {
    using namespace mfem;
    if (dim_ == 2) {
      return FunctionCoefficient([this](const Vector& x) {
        auto r = x.DistanceTo(x_);
        if (r <= r_) {
          return pi * r * r;
        } else {
          return 2 * pi * r_ * log(r / r_) + pi * r_ * r_;
        }
      });
    } else {
      return FunctionCoefficient([this](const Vector& x) {
        auto r = x.DistanceTo(x_);
        if (r <= r_) {
          return -2 * pi * (3 * r_ * r_ - r * r) / 3;
        } else {
          return -4 * pi * pow(r_, 3) / (3 * r);
        }
      });
    }
  }

  mfem::FunctionCoefficient LinearisedCoefficient(const mfem::Vector& a) const {
    using namespace mfem;

    if (dim_ == 2) {
      return FunctionCoefficient([this, a](const Vector& x) {
        auto dim = x.Size();
        auto r = x.DistanceTo(x_);
        auto dr = x;
        dr -= x_;
        dr /= r;
        if (r <= r_) {
          return -2 * pi * r * (dr * a);
        } else {
          return -2 * pi * (dr * a) / r;
        }
      });
    } else {
      return FunctionCoefficient([this, &a](const Vector& x) {
        auto dim = x.Size();
        auto r = x.DistanceTo(x_);
        auto dr = x;
        dr -= x_;
        dr /= r;
        if (r <= r_) {
          return -4 * pi * r * (dr * a) / 3;
        } else {
          return -4 * pi * std::pow(r_, 3) * (dr * a) / (3 * r * r);
        }
      });
    }
  }
};
