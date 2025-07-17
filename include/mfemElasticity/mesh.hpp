#pragma once

#include <optional>
#include <tuple>

#include "mfem.hpp"

namespace mfemElasticity {

// Given a mesh pointer, returns a marker array for its
// external boundary.
mfem::Array<int> ExternalBoundaryMarker(mfem::Mesh* mesh);

// Given a mesh pointer, returns a marker array for all its attributes.
mfem::Array<int> AllDomainsMarker(mfem::Mesh* mesh);

// Given a mesh pointer, returns a marker array for all its boundary attributes.
mfem::Array<int> AllBoundariesMarker(mfem::Mesh* mesh);

std::tuple<int, int, mfem::real_t> BoundaryRadius(
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0);

std::tuple<int, int, mfem::real_t> BoundaryRadius(mfem::Mesh* mesh,
                                                  mfem::Array<int>&& bdr_marker,
                                                  const mfem::Vector& x0);

std::tuple<int, int, mfem::real_t> BoundaryRadius(
    mfem::Mesh* mesh, const mfem::Array<int>& bdr_marker);

std::tuple<int, int, mfem::real_t> BoundaryRadius(
    mfem::Mesh* mesh, mfem::Array<int>&& bdr_marker);

std::tuple<int, int, mfem::real_t> BoundaryRadius(mfem::Mesh* mesh,
                                                  const mfem::Vector& x0);

std::tuple<int, int, mfem::real_t> BoundaryRadius(mfem::Mesh* mesh);

#ifdef MFEM_USE_MPI
std::tuple<int, int, mfem::real_t> BoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker,
    const mfem::Vector& x0);

std::tuple<int, int, mfem::real_t> BoundaryRadius(mfem::ParMesh* mesh,
                                                  mfem::Array<int>&& bdr_marker,
                                                  const mfem::Vector& x0);

std::tuple<int, int, mfem::real_t> BoundaryRadius(
    mfem::ParMesh* mesh, const mfem::Array<int>& bdr_marker);

std::tuple<int, int, mfem::real_t> BoundaryRadius(
    mfem::ParMesh* mesh, mfem::Array<int>&& bdr_marker);

std::tuple<int, int, mfem::real_t> BoundaryRadius(mfem::ParMesh* mesh,
                                                  const mfem::Vector& x0);

std::tuple<int, int, mfem::real_t> BoundaryRadius(mfem::ParMesh* mesh);
#endif

mfem::Vector MeshCentroid(mfem::Mesh* mesh, mfem::Array<int>& dom_marker,
                          int order = 1);

mfem::Vector MeshCentroid(mfem::Mesh* mesh, mfem::Array<int>&& dom_marker,
                          int order = 1);

mfem::Vector MeshCentroid(mfem::Mesh* mesh, int order = 1);

#ifdef MFEM_USE_MPI

mfem::Vector MeshCentroid(mfem::ParMesh* mesh, mfem::Array<int>& dom_marker,
                          int order = 1);

mfem::Vector MeshCentroid(mfem::ParMesh* mesh, mfem::Array<int>&& dom_marker,
                          int order = 1);

mfem::Vector MeshCentroid(mfem::ParMesh* mesh, int order = 1);
#endif

}  // namespace mfemElasticity