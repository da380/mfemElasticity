#include "mfemElasticity/legendre.hpp"

namespace mfemElasticity {

void LegendreHelper::SetSquareRoots(int lMax) {
  _sqrt.SetSize(2 * lMax + 2);
  _isqrt.SetSize(2 * lMax + 2);
  for (auto l = 0; l <= 2 * lMax + 1; l++) {
    _sqrt(l) = std::sqrt(static_cast<mfem::real_t>(l));
  }
  for (auto l = 1; l <= 2 * lMax + 1; l++) {
    _isqrt(l) = 1 / _sqrt(l);
  }
}

void LegendreHelper::SetSquareRoots(int dim, int degree) {
  if (dim == 3) {
    _sqrt.SetSize(2 * degree + 2);
    _isqrt.SetSize(2 * degree + 2);
    for (auto l = 0; l <= 2 * degree + 1; l++) {
      _sqrt(l) = std::sqrt(static_cast<mfem::real_t>(l));
    }
    for (auto l = 1; l <= 2 * degree + 1; l++) {
      _isqrt(l) = 1 / _sqrt(l);
    }
  } else {
    _sqrt.SetSize(degree + 1);
    for (auto k = 0; k <= degree; k++) {
      _sqrt(k) = std::sqrt(static_cast<mfem::real_t>(k));
    }
  }
}

mfem::real_t LegendreHelper::LogFactorial(int m) const {
  return std::lgamma(static_cast<mfem::real_t>(m + 1));
}

mfem::real_t LegendreHelper::LogDoubleFactorial(int m) const {
  return -logSqrtPi + m * log2 +
         std::lgamma(static_cast<mfem::real_t>(m + 0.5));
}

mfem::real_t LegendreHelper::Pll(int l, mfem::real_t x) const {
  using namespace mfem;
  if (l == 0) return invSqrtFourPi;
  auto sin2 = 1 - x * x;
  if (std::abs(sin2) < std::numeric_limits<real_t>::min()) return 0;
  auto logValue =
      0.5 * (std::log(static_cast<real_t>(2 * l + 1)) - LogFactorial(2 * l)) +
      LogDoubleFactorial(l) + 0.5 * l * std::log(sin2);
  return MinusOnePower(l) * invSqrtFourPi * std::exp(logValue);
}

std::pair<mfem::real_t, mfem::real_t> LegendreHelper::RecursionCoefficients(
    int l, int m) const {
  auto alpha =
      _sqrt[2 * l + 1] * _sqrt[2 * l - 1] * _isqrt[l + m] * _isqrt[l - m];
  auto beta = _sqrt[l - 1 + m] * _sqrt[l - 1 - m] * _isqrt[2 * (l - 1) + 1] *
              _isqrt[2 * (l - 1) - 1];
  return {alpha, beta};
}

}  // namespace mfemElasticity