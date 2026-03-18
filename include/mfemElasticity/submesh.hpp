/**
 * @file submesh.hpp
 * @brief Defines the SubMeshProlongationMatrix for mapping between MFEM
 * SubMeshes and their parent meshes.
 */

#pragma once

#include "mfem.hpp"

namespace mfemElasticity {

/**
 * @class SubMeshProlongationMatrix
 * @brief A specialized SparseMatrix for prolonging and restricting finite
 * element fields between a SubMesh and its parent Mesh.
 *
 * This matrix acts as a prolongation operator $P$, mapping local degrees of
 * freedom (DoFs) from the submesh space to the global parent mesh space
 * ($u_{global} = P u_{local}$).
 * * It can be used for:
 * - **Prolongation/Extension**: Mapping a local SubMesh Vector/LinearForm to
 * the parent space using `Mult()`.
 * - **Restriction**: Mapping a global parent GridFunction down to the SubMesh
 * space using `MultTranspose()`.
 */
class SubMeshProlongationMatrix : public mfem::SparseMatrix {
 private:
  /**
   * @struct CSRData
   * @brief Helper structure to safely package raw Compressed Sparse Row (CSR)
   * arrays during matrix construction.
   */
  struct CSRData {
    int *I;        ///< Array of row pointers.
    int *J;        ///< Array of column indices.
    double *Data;  ///< Array of non-zero matrix values.
    int m;         ///< Number of rows (Parent DoFs).
    int n;         ///< Number of columns (SubMesh DoFs).
  };

  /**
   * @brief Pre-calculates and allocates the CSR arrays for the prolongation
   * matrix.
   *
   * @details This function calculates the exact required size for the CSR
   * arrays in an $O(N)$ pass, avoiding the dynamic reallocation overhead of
   * standard MFEM SparseMatrix assembly.
   * * @param sub_fes The finite element space defined on the SubMesh.
   * @param parent_fes The finite element space defined on the parent Mesh.
   * @return CSRData A structure containing the fully populated and allocated
   * CSR arrays.
   */
  static CSRData BuildCSR(const mfem::FiniteElementSpace &sub_fes,
                          const mfem::FiniteElementSpace &parent_fes);

  /**
   * @brief Private constructor that initializes the base SparseMatrix and takes
   * ownership of the raw CSR data.
   *
   * @param d The CSR array data built by the BuildCSR() routine.
   */
  SubMeshProlongationMatrix(CSRData d)
      : mfem::SparseMatrix(d.I, d.J, d.Data, d.m, d.n) {
    // The base SparseMatrix takes ownership of the arrays and will delete[]
    // them automatically upon destruction.
  }

 public:
  /**
   * @brief Constructs an optimized prolongation matrix mapping the SubMesh
   * space to the parent space.
   *
   * @param sub_fes The finite element space defined on the SubMesh.
   * @param parent_fes The finite element space defined on the exact parent Mesh
   * of the SubMesh.
   */
  SubMeshProlongationMatrix(const mfem::FiniteElementSpace &sub_fes,
                            const mfem::FiniteElementSpace &parent_fes)
      : SubMeshProlongationMatrix(BuildCSR(sub_fes, parent_fes)) {}
};

}  // namespace mfemElasticity