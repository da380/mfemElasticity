#pragma once

#include "mfem.hpp"

namespace mfemElasticity {

mfem::Array<int> ExternalBoundaryMarker(mfem::Mesh* mesh);

mfem::Array<int> AllDomainsMarker(mfem::Mesh* mesh);

mfem::Array<int> AllBoundariesMarker(mfem::Mesh* mesh);

}  // namespace mfemElasticity