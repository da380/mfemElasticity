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





/**
 * @brief Mixed bilinear form assembly between a parent-mesh finite element
 * space and a corresponding SubMesh finite element space.
 *
 * This class assembles mixed operators in which one finite element space is
 * defined on a SubMesh and the other on its parent mesh. Local element
 * matrices are assembled on the SubMesh and inserted into the global mixed
 * matrix through an automatically constructed SubMesh-to-parent DoF mapping.
 *
 * If @p extended_trial is true, the trial space is defined on the parent mesh
 * and restricted to the SubMesh during assembly. Otherwise, the test space is
 * defined on the parent mesh and restricted to the SubMesh.
 *
 * @note The space supplied through @p sub_fes must be defined on the SubMesh
 * and represent the restriction of the parent-space field to the SubMesh.
 */
class MixedBilinearFormSubMesh : public mfem::MixedBilinearForm
{
protected:
    mfem::FiniteElementSpace *sub_fes = nullptr;
    mfem::Array<int> *vdof_to_vdof_map = nullptr;
    bool extended_trial;

public:
    MixedBilinearFormSubMesh(mfem::FiniteElementSpace *tr_fes,
                             mfem::FiniteElementSpace *te_fes,
                             mfem::FiniteElementSpace *sub_fes_,
                             bool extended_trial_);

    void Assemble(int skip_zeros = 1);

    ~MixedBilinearFormSubMesh() { delete vdof_to_vdof_map; }
};

#ifdef MFEM_USE_MPI
/**
 * @brief Parallel version of MixedBilinearFormSubMesh.
 *
 * Provides the same functionality as MixedBilinearFormSubMesh for
 * ParFiniteElementSpace objects defined on a ParSubMesh and its parent mesh.
 */
class ParMixedBilinearFormSubMesh : public mfem::ParMixedBilinearForm
{
protected:
    mfem::ParFiniteElementSpace *sub_pfes = nullptr;
    mfem::Array<int> *vdof_to_vdof_map = nullptr;
    bool extended_trial;

public:
    ParMixedBilinearFormSubMesh(mfem::ParFiniteElementSpace *tr_pfes,
                                mfem::ParFiniteElementSpace *te_pfes,
                                mfem::ParFiniteElementSpace *sub_pfes_,
                                bool extended_trial_);

    void Assemble(int skip_zeros = 1);

    ~ParMixedBilinearFormSubMesh() { delete vdof_to_vdof_map; }
};
#endif


}  // namespace mfemElasticity
