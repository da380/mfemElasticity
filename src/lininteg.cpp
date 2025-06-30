#include "mfemElasticity/lininteg.hpp"

namespace mfemElasticity {

void DomainLFDeformationGradientIntegrator::AssembleRHSElementVect(
    const mfem::FiniteElement& el, mfem::ElementTransformation& Trans,
    mfem::Vector& elvect) {
  using namespace mfem;

  auto dof = el.GetDof();
  auto space_dim = Trans.GetSpaceDim();
  MFEM_ASSERT(_M.GetHeight() == space_dim && _M.GetWidth() == space_dim,
              "Width of matrix coefficient must equal spatial dimension");

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape, m;
  Vector v;
#endif
  dshape.SetSize(dof, space_dim);
  m.SetSize(space_dim, space_dim);
  elvect.SetSize(dof * space_dim);

  elvect = 0.0;
  v.SetSize(dof * space_dim);
  auto vm = DenseMatrix(v.GetData(), dof, space_dim);

  const auto* ir = GetIntegrationRule(el, Trans);
  if (ir == nullptr) {
    int intorder = 2 * el.GetOrder() + Trans.OrderW();
    ir = &IntRules.Get(el.GetGeomType(), intorder);
  }

  for (int i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto factor = Trans.Weight() * ip.weight;
    el.CalcPhysDShape(Trans, dshape);
    _M.Eval(m, Trans, ip);
    MultABt(dshape, m, vm);
    elvect.Add(factor, v);
  }
}

}  // namespace mfemElasticity
