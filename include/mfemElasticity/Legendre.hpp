#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

template <typename Real>
class Legendre {
 private:
  using Int = std::ptrdiff_t;
  const Real invSqrtFourPi =
      0.28209479177387814347403972578038629292202531466449942842204286085532123422;
  const Real logSqrtPi =
      0.5723649429247000870717136756765293558236474064576557857568115357360;
  const Real log2 =
      0.69314718055994530941723212145817656807550013436025525412068000949339362196;

  Int _l;
  Real _x;

  std::vector<Real> _current;
  std::vector<Real> _previous;

  // Returns (-1)^m
  Int MinusOnePower(Int m) const { return m % 2 ? -1 : 1; }

  // Returns log[m!]
  Real LogFactorial(Int m) const {
    return std::lgamma(static_cast<Real>(m + 1));
  }

  // Returns log[(2m-1)!!]
  Real LogDoubleFactorial(Int m) const {
    return -logSqrtPi + m * log2 + std::lgamma(static_cast<Real>(m + 0.5));
  }

  // Returns P_{ll} at the stored argument.
  Real Pll(Int l) const {
    if (l == 0) return invSqrtFourPi;
    auto sin2 = 1 - _x * _x;
    if (std::abs(sin2) < std::numeric_limits<Real>::min()) return 0;
    auto logValue =
        0.5 * (std::log(static_cast<Real>(2 * l + 1)) - LogFactorial(2 * l)) +
        LogDoubleFactorial(l) + 0.5 * l * std::log(sin2);
    return MinusOnePower(l) * invSqrtFourPi * std::exp(logValue);
  }

 public:
  Legendre(Real x) {
    _l = 0;
    _x = x;
    _previous.push_back(0);
    _current.push_back(Pll(0));
  }

  Legendre(Real x, Int l) : Legendre(x) {
    for (auto n = 0; n < l; n++) {
      ++(*this);
    }
  }

  // Return the current degree.
  Int Degree() const { return _l; }

  // Return the value at a given order.
  Real operator()(Int m) const {
    assert(std::abs(m) <= _l);
    return m >= 0 ? _current[m] : MinusOnePower(m) * _current[-m];
  }

  // Constant iterators to the data.
  auto begin() const { return _current.cbegin(); }
  auto end() const { return _current.cend(); }

  // Return the current size.
  auto size() const { return _current.size(); }

  // Increment the degree by one.
  auto& operator++() {
    _l++;
    for (auto m = 0; m < _l; m++) {
      const auto alpha = std::sqrt(static_cast<Real>(4 * _l * _l - 1) /
                                   static_cast<Real>(_l * _l - m * m));
      const auto beta =
          std::sqrt(static_cast<Real>((_l - 1) * (_l - 1) - m * m) /
                    static_cast<Real>(4 * (_l - 1) * (_l - 1) - 1));
      _previous[m] = alpha * (_x * _current[m] - beta * _previous[m]);
    }
    _previous.push_back(Pll(_l));
    _current.push_back(0);
    std::swap(_current, _previous);
    return *this;
  }
};