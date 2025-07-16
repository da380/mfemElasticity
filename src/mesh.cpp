
#include "mfemElasticity/mesh.hpp"

#include <optional>

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

std::optional<mfem::real_t> BoundaryRadius(mfem::Mesh* mesh, int bdr_attr,
                                           const mfem::Vector& x0) {
  using namespace mfem;

  const auto rtol = 1e-6;

  auto dim = mesh->Dimension();
  auto fec = H1_FECollection(1, dim);
  auto fes = FiniteElementSpace(mesh, &fec);

  auto radius = real_t{-1};
  auto x = Vector(dim);
  auto different = false;
  for (auto i = 0; i < mesh->GetNBE(); i++) {
    if (different) break;
    const auto attr = mesh->GetBdrAttribute(i);

    if (attr == bdr_attr) {
      const auto* fe = fes.GetBE(i);
      auto* Trans = fes.GetBdrElementTransformation(i);
      const auto& ir = fe->GetNodes();
      for (auto j = 0; j < ir.GetNPoints(); j++) {
        const auto& ip = ir.IntPoint(j);
        Trans->SetIntPoint(&ip);
        Trans->Transform(ip, x);
        auto d = x.DistanceTo(x0);
        if (radius < 0) {
          radius = d;
        } else {
          different = std::abs(radius - d) > rtol * radius;
          if (different) break;
        }
      }
    }
  }
  if (different) {
    return std::nullopt;
  } else {
    return radius;
  }
}

mfem::Vector DomainCentroid(mfem::Mesh* mesh, int dom_attr, int order) {
  using namespace mfem;
  auto dim = mesh->Dimension();
  auto x0 = Vector(dim);
  x0 = 0.0;

  return x0;
}

}  // namespace mfemElasticity