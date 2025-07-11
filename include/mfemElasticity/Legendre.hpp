#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "mfem.hpp"

namespace mfemElasticity {

struct LegendreHelper {
  static constexpr mfem::real_t pi = std::atan(1) * 4;
  static constexpr mfem::real_t invSqrtFourPi = 1 / std::sqrt(4 * pi);
  static constexpr mfem::real_t logSqrtPi = std::log(std::sqrt(pi));
  static constexpr mfem::real_t log2 = std::log(static_cast<mfem::real_t>(2));

  mfem::Vector _sqrt;
  mfem::Vector _isqrt;

  void SetSquareRoots(int lMax);

  int MinusOnePower(int m) const { return m % 2 ? -1 : 1; }

  // Returns log(m!)
  mfem::real_t LogFactorial(int m) const;

  // Returns log[(2m-1)!!]
  mfem::real_t LogDoubleFactorial(int m) const;

  // Returns P_{ll}(x)
  mfem::real_t Pll(int l, mfem::real_t x) const;

  // Returns three-point recursion coefficients for given degree and order
  std::pair<mfem::real_t, mfem::real_t> RecursionCoefficients(int l,
                                                              int m) const {
    auto alpha =
        _sqrt[2 * l + 1] * _sqrt[2 * l - 1] * _isqrt[l + m] * _isqrt[l - m];
    auto beta = _sqrt[l - 1 + m] * _sqrt[l - 1 - m] * _isqrt[2 * (l - 1) + 1] *
                _isqrt[2 * (l - 1) - 1];
    return {alpha, beta};
  }
};

}  // namespace mfemElasticity
