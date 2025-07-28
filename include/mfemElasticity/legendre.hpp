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
 * the conventions described in Dahlen & Tromp (1998) - *Theoretical Global
 * Seismology*.
 *
 * The implementation uses recurrence relations for efficiency and precomputes
 * square roots to speed up calculations of recursion coefficients.
 */
struct LegendreHelper {
  /** @brief Value of pi, computed as \f$4 \times \text{atan}(1)\f$. */
  static constexpr mfem::real_t pi = std::atan(1) * 4;
  /** @brief Value of \f$\sqrt{\pi}\f$. */
  static constexpr mfem::real_t sqrtPi = std::sqrt(pi);
  /** @brief Value of \f$1/\sqrt{4\pi}\f$. Used in spherical harmonic
   * normalization. */
  static constexpr mfem::real_t invSqrtFourPi = 1 / std::sqrt(4 * pi);
  /** @brief Value of \f$\log(\sqrt{\pi})\f$. */
  static constexpr mfem::real_t logSqrtPi = std::log(std::sqrt(pi));
  /** @brief Value of \f$\log(2)\f$. */
  static constexpr mfem::real_t log2 = std::log(static_cast<mfem::real_t>(2));

  mfem::Vector
      _sqrt; /**< Precomputed square roots of integers: `_sqrt[k] = sqrt(k)`. */
  mfem::Vector _isqrt; /**< Precomputed inverse square roots of integers:
                            `_isqrt[k] = 1/sqrt(k)`. */

  /**
   * @brief Precomputes integer square roots and inverse square roots up to \f$2
   * \times \text{lMax} + 1\f$.
   * @param lMax The maximum degree \f$l\f$ for which polynomials will be
   * computed. The precomputation range is determined by the maximum index
   * needed in recurrence relations.
   */
  void SetSquareRoots(int lMax);

  /**
   * @brief Precomputes necessary square roots based on spatial dimension and
   * polynomial degree.
   * This overload is provided for convenience when the required maximum index
   * depends on these parameters. Internally, it calls the
   * other `SetSquareRoots` method with an appropriately determined `lMax`.
   * @param dim The spatial dimension (e.g., 2 or 3).
   * @param degree The maximum polynomial degree.
   */
  void SetSquareRoots(int dim, int degree);

  /**
   * @brief Calculates \f$(-1)^m\f$.
   * @param m The integer exponent.
   * @return \f$1\f$ if \f$m\f$ is even, \f$-1\f$ if \f$m\f$ is odd.
   */
  int MinusOnePower(int m) const { return m % 2 ? -1 : 1; }

  /**
   * @brief Returns \f$\log(m!)\f$ using `std::lgamma` (logarithm of the Gamma
   * function).
   * @param m The integer for which to compute the logarithm of factorial.
   * @return The value of \f$\log(m!)\f$.
   */
  mfem::real_t LogFactorial(int m) const;

  /**
   * @brief Returns \f$\log[(2m-1)!!]\f$, where \f$!!\f$ denotes the double
   * factorial.
   *
   * The double factorial \f$(2m-1)!! = (2m-1)(2m-3)\dots 1\f$.
   * This can also be expressed as \f$\frac{(2m)!}{2^m m!}\f$.
   * @param m The integer for which to compute the logarithm of the double
   * factorial.
   * @return The value of \f$\log[(2m-1)!!]\f$.
   */
  mfem::real_t LogDoubleFactorial(int m) const;

  /**
   * @brief Computes the Associated Legendre polynomial \f$P_{ll}(x)\f$.
   *
   * @param l The degree \f$l\f$.
   * @param x The argument \f$x\f$ (typically \f$\cos\theta\f$).
   * @return The value of \f$P_{ll}(x)\f$.
   */
  mfem::real_t Pll(int l, mfem::real_t x) const;

  /**
   * @brief Returns the three-point recursion coefficients for
   * \f$P_{l+1,m}(x)\f$.
   *
   * The recurrence relation is given by:
   * \f[
   * P_{l+1 \, m}(x) = \alpha  [x \cdot P_{l\,m}(x) - \beta \cdot P_{l-1\,m}(x)]
   * \f]
   *
   * @param l The current degree \f$l\f$.
   * @param m The order \f$m\f$.
   * @return A `std::pair` where the first element is \f$\alpha\f$ and the
   * second is \f$\beta\f$.
   */
  std::pair<mfem::real_t, mfem::real_t> RecursionCoefficients(int l,
                                                              int m) const;
};

}  // namespace mfemElasticity
