#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "mfem.hpp"

namespace mfemElasticity {

/*
  Helper struct for computation of Associated Legendre polynomials
  normalised following the conventions in Dahlen & Tromp (1998).
*/
struct LegendreHelper {
  static constexpr mfem::real_t pi = std::atan(1) * 4;
  static constexpr mfem::real_t sqrtPi = std::sqrt(pi);
  static constexpr mfem::real_t invSqrtFourPi = 1 / std::sqrt(4 * pi);
  static constexpr mfem::real_t logSqrtPi = std::log(std::sqrt(pi));
  static constexpr mfem::real_t log2 = std::log(static_cast<mfem::real_t>(2));

  mfem::Vector _sqrt;
  mfem::Vector _isqrt;

  // Precompute integer square roots up to 2*lMax+1.
  void SetSquareRoots(int lMax);

  // Precompute necessary square roots based on dimension and degree.
  void SetSquareRoots(int dim, int degree);

  int MinusOnePower(int m) const { return m % 2 ? -1 : 1; }

  // Returns log(m!)
  mfem::real_t LogFactorial(int m) const;

  // Returns log[(2m-1)!!]
  mfem::real_t LogDoubleFactorial(int m) const;

  // Returns p_{ll}(x)
  mfem::real_t Pll(int l, mfem::real_t x) const;

  /*
    Returns three-point recursion coefficients such that:
    p_{l+1,m} = alpha * (x * p_{l,m} - beta * p_{l-1,m})
  */
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
