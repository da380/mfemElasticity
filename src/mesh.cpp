#include "mfemElasticity/mesh.hpp"

namespace mfemElasticity {

mfem::Array<int> ExternalBoundaryMarker(mfem::Mesh* mesh) {
  auto bdr_marker = mfem::Array<int>(mesh->bdr_attributes.Max());
  bdr_marker = 0;
  mesh->MarkExternalBoundaries(bdr_marker);
  return bdr_marker;
}

mfem::Array<int> AllDomainsMarker(mfem::Mesh* mesh) {
  auto dom_marker = mfem::Array<int>(mesh->attributes.Max());
  dom_marker = 1;
  return dom_marker;
}

mfem::Array<int> AllBoundariesMarker(mfem::Mesh* mesh) {
  auto bdr_marker = mfem::Array<int>(mesh->bdr_attributes.Max());
  bdr_marker = 1;
  return bdr_marker;
}

mfem::real_t ExternalBoundaryRadius(mfem::Mesh* mesh) {
  using namespace mfem;
  auto dim = mesh->Dimension();
  auto bdr_marker = ExternalBoundaryMarker(mesh);
  auto bdr_mesh = SubMesh::CreateFromBoundary(*mesh, bdr_marker);
  bdr_mesh.EnsureNodes();
  auto* nodes = bdr_mesh.GetNodes();

  /*
  using namespace mfem;
  auto dim = mesh->Dimension();
  auto bdr_marker = ExternalBoundaryMarker(mesh);
  auto r = real_t{0};
  auto x = Vector(dim);

  for (auto i = 0; i < _te_fes->GetNBE(); i++) {
    const auto elm_attr = mesh->GetBdrAttribute(i);
    if (bdr_marker[elm_attr - 1] == 1) {
      const auto* fe = _te_fes->GetBE(i);
      auto* Trans = _te_fes->GetBdrElementTransformation(i);
      const auto ir = fe->GetNodes();
      const IntegrationPoint& ip = ir.IntPoint(0);
      Trans->SetIntPoint(&ip);
      Trans->Transform(ip, x);
      r = x.Norml2();
      break;
    }
  }
  return r;
  */
  return 1.0;
}

}  // namespace mfemElasticity