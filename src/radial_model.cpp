#include "mfemElasticity/radial_model.hpp"

namespace RadialModel {

mfem::real_t RadialModelCoefficient::Eval(mfem::ElementTransformation &T,
                                          const mfem::IntegrationPoint &ip) {
  mfem::real_t data[3];
  auto x = mfem::Vector(data, 3);
  T.Transform(ip, x);

  auto r = x.Norml2();
  auto attribute = T.Attribute;
  return f_(r, attribute);
}

}  // namespace RadialModel