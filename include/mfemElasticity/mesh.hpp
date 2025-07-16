#pragma once

#include <optional>

#include "mfem.hpp"

namespace mfemElasticity {

// Given a mesh pointer, returns a marker array for its
// external boundary.
mfem::Array<int> ExternalBoundaryMarker(mfem::Mesh* mesh);

// Given a mesh pointer, returns a marker array for all its attributes.
mfem::Array<int> AllDomainsMarker(mfem::Mesh* mesh);

// Given a mesh pointer, returns a marker array for all its boundary attributes.
mfem::Array<int> AllBoundariesMarker(mfem::Mesh* mesh);

// Given a serial mesh pointer, returns the external boundary radius.
mfem::real_t ExternalBoundaryRadius(mfem::Mesh* mesh);

// Given a parallel mesh pointer, returns the external boundary radius.
mfem::real_t ExternalBoundaryRadius(mfem::ParMesh* mesh);

std::optional<mfem::real_t> BoundaryRadius(mfem::Mesh* mesh, int bdr_attr,
                                           const mfem::Vector& x0);

mfem::Vector DomainCentroid(mfem::Mesh* mesh, int dom_attr, int order = 1);

}  // namespace mfemElasticity