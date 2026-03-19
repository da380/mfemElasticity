/**
 * @file submesh.hpp
 * @brief Defines the SubMeshProlongationMatrix for mapping between MFEM
 * SubMeshes and their parent meshes.
 */

#pragma once

#include "mfem.hpp"

namespace mfemElasticity {

/**
 * @brief Factory function to build an optimized prolongation matrix mapping a
 * SubMesh space to its parent space.
 *
 * This function generates a sparse matrix acting as a prolongation operator
 * $P$, mapping local degrees of freedom (DoFs) from the submesh space to the
 * global parent mesh space ($u_{global} = P u_{local}$). It calculates and
 * populates the Compressed Sparse Row (CSR) arrays in a highly optimized $O(N)$
 * pass.
 *
 * @note The caller takes ownership of the returned SparseMatrix pointer and is
 * responsible for deleting it. The underlying CSR arrays (`I`, `J`, and `Data`)
 * are automatically managed and will be freed by the SparseMatrix destructor.
 *
 * @param sub_fes The finite element space defined on the SubMesh.
 * @param parent_fes The finite element space defined on the exact parent Mesh
 * of the SubMesh.
 * @return mfem::SparseMatrix* A dynamically allocated SparseMatrix representing
 * the prolongation operator.
 */
mfem::SparseMatrix *SubMeshProlongationMatrix(
    const mfem::FiniteElementSpace &sub_fes,
    const mfem::FiniteElementSpace &parent_fes);

}  // namespace mfemElasticity