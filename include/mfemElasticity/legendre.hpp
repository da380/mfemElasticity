#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "mfem.hpp"

namespace mfemElasticity {

/**
 * @brief Helper struct for the computation of Associated Legendre polynomials.
 *
 * This struct provides methods and precomputed values to efficiently
 * compute Associated Legendre polynomials. The normalization follows
 * the conventions described in Dahlen & Tromp (1998), specifically
 * equation (C.1) for the general form $P_l^m(\cos\theta)$, and
 * equation (C.2) for the special case $P_l^l(\cos\theta)$.
 *
 * The polynomials are defined such that:
 * $$ P_l^m(x) = (-1)^m (1-x^2)^{m/2} \frac{d^m}{dx^m} P_l(x) $$
 * and normalized as:
 * $$ \int_{-1}^{1} [P_l^m(x)]^2 dx = \frac{2}{2l+1} \frac{(l+m)!}{(l-m)!} $$
 * The implementation uses recurrence relations for efficiency and precomputes
 * square roots to speed up calculations of recursion coefficients.
 */
struct LegendreHelper {
  /** @brief Value of pi, computed as $4 \times \text{atan}(1)$. */
  static constexpr mfem::real_t pi = std::atan(1) * 4;
  /** @brief Value of $\sqrt{\pi}$. */
  static constexpr mfem::real_t sqrtPi = std::sqrt(pi);
  /** @brief Value of $1/\sqrt{4\pi}$. Used in spherical harmonic normalization.
   */
  static constexpr mfem::real_t invSqrtFourPi = 1 / std::sqrt(4 * pi);
  /** @brief Value of $\log(\sqrt{\pi})$. */
  static constexpr mfem::real_t logSqrtPi = std::log(std::sqrt(pi));
  /** @brief Value of $\log(2)$. */
  static constexpr mfem::real_t log2 = std::log(static_cast<mfem::real_t>(2));

  mfem::Vector
      _sqrt; /**< Precomputed square roots of integers: `_sqrt[k] = sqrt(k)`. */
  mfem::Vector _isqrt; /**< Precomputed inverse square roots of integers:
                          `_isqrt[k] = 1/sqrt(k)`. */

  /**
   * @brief Precomputes integer square roots and inverse square roots up to $2
   * \times \text{lMax} + 1$.
   * @param lMax The maximum degree $l$ for which polynomials will be computed.
   * The precomputation range is determined by the maximum index needed in
   * recursion.
   */
  void SetSquareRoots(int lMax);

  /**
   * @brief Precomputes necessary square roots based on spatial dimension and
   * polynomial degree. This overload might be used for convenience when the
   * required maximum index depends on these parameters. Internally calls the
   * other `SetSquareRoots` with an appropriate `lMax`.
   * @param dim The spatial dimension (e.g., 2 or 3).
   * @param degree The maximum polynomial degree.
   */
  void SetSquareRoots(int dim, int degree);

  /**
   * @brief Calculates $(-1)^m$.
   * @param m The integer exponent.
   * @return 1 if $m$ is even, -1 if $m$ is odd.
   */
  int MinusOnePower(int m) const { return m % 2 ? -1 : 1; }

  /**
   * @brief Returns $\log(m!)$ using `std::lgamma` or precomputed values.
   * @param m The integer for which to compute the logarithm of factorial.
   * @return The value of $\log(m!)$.
   */
  mfem::real_t LogFactorial(int m) const;

  /**
   * @brief Returns $\log[(2m-1)!!]$, where $!!$ denotes the double factorial.
   *
   * The double factorial $(2m-1)!! = (2m-1)(2m-3)\dots 1$.
   * Can be expressed as $\frac{(2m)!}{2^m m!}$.
   * @param m The integer for which to compute the logarithm of the double
   * factorial.
   * @return The value of $\log[(2m-1)!!]$.
   */
  mfem::real_t LogDoubleFactorial(int m) const;

  /**
   * @brief Computes the Associated Legendre polynomial $P_l^l(x)$.
   *
   * This is the initial value for the recursion relations, specifically derived
   * from equation (C.2) in Dahlen & Tromp (1998):
   * $$ P_l^l(x) = (-1)^l (2l-1)!! (1-x^2)^{l/2} $$
   * where $(2l-1)!! = (2l-1)(2l-3)\dots 1$.
   * @param l The degree $l$.
   * @param x The argument $x$ (typically $\cos\theta$).
   * @return The value of $P_l^l(x)$.
   */
  mfem::real_t Pll(int l, mfem::real_t x) const;

  /**
   * @brief Returns the three-point recursion coefficients for $P_{l+1,m}(x)$.
   *
   * The recurrence relation is given by:
   * $$ P_{l+1}^m(x) = \alpha \cdot (x \cdot P_l^m(x) - \beta \cdot
   * P_{l-1}^m(x)) $$ The coefficients $\alpha$ and $\beta$ are derived from
   * equation (C.4) in Dahlen & Tromp (1998):
   * $$ \alpha = \sqrt{\frac{(2l+1)(2l-1)}{(l+m)(l-m)}} $$
   * $$ \beta = \sqrt{\frac{(l-1+m)(l-1-m)}{(2l-1)(2l-3)}} $$
   * @param l The current degree $l$.
   * @param m The order $m$.
   * @return A `std::pair` where the first element is $\alpha$ and the second is
   * $\beta$.
   */
  std::pair<mfem::real_t, mfem::real_t> RecursionCoefficients(int l,
                                                              int m) const;
};

}  // namespace mfemElasticity
