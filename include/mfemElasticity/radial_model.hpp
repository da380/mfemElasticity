#pragma once
#pragma once

#include <complex>
#include <functional>
#include <iostream>

#include "mfem.hpp"

namespace RadialModel {

class RadialModelCoefficient : public mfem::Coefficient {
 private:
  std::function<mfem::real_t(mfem::real_t, int)> f_;

 public:
  RadialModelCoefficient() = default;
  RadialModelCoefficient(std::function<mfem::real_t(mfem::real_t, int)> &&f)
      : f_{f} {}

  mfem::real_t Eval(mfem::ElementTransformation &T,
                    const mfem::IntegrationPoint &ip) override;
};

}  // namespace RadialModel