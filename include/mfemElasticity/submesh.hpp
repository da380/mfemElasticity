/**
 * @file submesh.hpp
 * @brief Defines the SubMeshProlongationMatrix for mapping between MFEM
 * SubMeshes and their parent meshes.
 */

#pragma once

#include <memory>

#include "mfem.hpp"

namespace mfemElasticity {

/**
 * @brief Signed injection of the vdofs of a space on a (Par)SubMesh into the
 * vdofs of the corresponding space on the parent mesh.
 *
 * Built from mfem::SubMeshUtils::BuildVdofToVdofMap; valid for both
 * SubMesh::From::Domain and SubMesh::From::Boundary submeshes, in serial and
 * (when the spaces are ParFiniteElementSpaces on a ParSubMesh) in parallel.
 *
 * The sub space must be the *shadow* of the parent space: same
 * FiniteElementCollection object, vdim and ordering — use MakeShadowSpace().
 * Restricted to conforming meshes and H1/L2 collections of fixed order.
 *
 * As an mfem::Operator this acts between L-vectors (local vdofs):
 * Mult() is the prolongation-by-zero sub → parent, MultTranspose() the exact
 * restriction parent → sub. NewTrueDofMatrix() provides the corresponding
 * true-dof injection Π for parallel block operators.
 */
class SubMeshDofInjection : public mfem::Operator {
 public:
  /**
   * @brief Build the injection for an existing shadow space.
   *
   * @param sub_fes Space on the SubMesh/ParSubMesh; must share the parent
   * space's FiniteElementCollection object, vdim and ordering.
   * @param parent_fes Space on the exact parent mesh of the submesh.
   */
  SubMeshDofInjection(const mfem::FiniteElementSpace& sub_fes,
                      const mfem::FiniteElementSpace& parent_fes);

  /**
   * @brief Construct the shadow of @p parent_fes on @p submesh (the space this
   * class expects as its sub space). The FiniteElementCollection object is
   * shared with the parent space, which must therefore outlive the result.
   */
  static std::unique_ptr<mfem::FiniteElementSpace> MakeShadowSpace(
      const mfem::FiniteElementSpace& parent_fes, mfem::SubMesh& submesh);

#ifdef MFEM_USE_MPI
  /** @brief Parallel overload of MakeShadowSpace. */
  static std::unique_ptr<mfem::ParFiniteElementSpace> MakeShadowSpace(
      const mfem::ParFiniteElementSpace& parent_fes,
      mfem::ParSubMesh& submesh);
#endif

  /** @brief Number of sub vdofs ( = Width()). */
  int SubVSize() const { return width; }

  /** @brief Number of parent vdofs ( = Height()). */
  int ParentVSize() const { return height; }

  /** @brief Parent vdof paired with each sub vdof (decoded, always >= 0). */
  const mfem::Array<int>& ParentVDofs() const { return parent_vdof_; }

  /** @brief Sign (+1/-1) relating each sub vdof to its parent vdof. */
  const mfem::Array<mfem::real_t>& Signs() const { return sign_; }

  /** @brief y = P x: scatter sub vdofs into parent vdofs, zero elsewhere. */
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  /** @brief y = P^T x: exact restriction of parent vdofs to sub vdofs. */
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

  /**
   * @brief P as an explicit sparse matrix (ParentVSize() x SubVSize(),
   * one +-1 entry per column, at most one per row).
   */
  std::unique_ptr<mfem::SparseMatrix> NewSparseMatrix() const;

  /**
   * @brief Given a finalized M with SubVSize() rows, return P M: row i of M
   * moved to row ParentVDofs()[i] and scaled by Signs()[i]; parent rows
   * outside the submesh are empty. O(nnz), no sparse product.
   */
  std::unique_ptr<mfem::SparseMatrix> RemapRows(
      const mfem::SparseMatrix& M) const;

  /**
   * @brief Given a finalized M with SubVSize() columns, return M P^T: column
   * indices remapped to parent vdofs and entries scaled by the signs.
   * O(nnz), no sparse product.
   */
  std::unique_ptr<mfem::SparseMatrix> RemapColumns(
      const mfem::SparseMatrix& M) const;

#ifdef MFEM_USE_MPI
  /**
   * @brief The true-dof injection Pi (parent true dofs x sub true dofs) as a
   * HypreParMatrix. Requires both spaces to be ParFiniteElementSpaces.
   *
   * Pi is a boolean (+-1) injection: one entry per column, at most one per
   * row, so Pi^T Pi = I and Pi^T is at once the exact primal restriction
   * parent -> sub and the dual prolongation.
   *
   * Built row-by-row over *owned* sub true dofs through
   * ParFiniteElementSpace::GetGlobalTDofNumber. (Note: the seemingly simpler
   * R_parent * P_loc * P_sub product is wrong when a shared parent dof on the
   * submesh boundary is owned by a rank whose local elements there all lie
   * outside the submesh: that rank's P_loc row is empty and the entry is
   * silently lost. The owner of a *sub* true dof always has the submesh
   * element, so this construction has no such case.)
   */
  std::unique_ptr<mfem::HypreParMatrix> NewTrueDofMatrix() const;
#endif

 private:
  const mfem::FiniteElementSpace* sub_fes_;
  const mfem::FiniteElementSpace* parent_fes_;
  mfem::Array<int> parent_vdof_;
  mfem::Array<mfem::real_t> sign_;
};

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
