#include "mfemElasticity/Legendre.hpp"

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

}  // namespace mfemElasticity