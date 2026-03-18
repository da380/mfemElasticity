/**
 * @file submesh.cpp
 * @brief Implementation of the SubMeshProlongationMatrix CSR builder.
 */

#include "mfemElasticity/submesh.hpp"

namespace mfemElasticity {

SubMeshProlongationMatrix::CSRData SubMeshProlongationMatrix::BuildCSR(
    const mfem::FiniteElementSpace &sub_fes,
    const mfem::FiniteElementSpace &parent_fes) {
  using namespace mfem;
  int m = parent_fes.GetVSize();  // Rows: Parent DoFs
  int n = sub_fes.GetVSize();     // Cols: SubMesh DoFs

  // 1. Verify submesh exists and is topologically compatible
  const SubMesh *submesh = dynamic_cast<const SubMesh *>(sub_fes.GetMesh());
  MFEM_VERIFY(submesh != nullptr, "The sub_fes must be defined on a SubMesh.");
  MFEM_VERIFY(submesh->GetParent() == parent_fes.GetMesh(),
              "Mismatch! The parent_fes must be defined on the exact "
              "parent Mesh of the provided SubMesh.");

  // 2. Extract local DoF mapping
  Array<int> vdof_map;
  SubMeshUtils::BuildVdofToVdofMap(sub_fes, parent_fes, submesh->GetFrom(),
                                   submesh->GetParentElementIDMap(), vdof_map);

  // Decode the mapping and store it temporarily
  Array<int> p_vdofs(n);
  Array<double> p_signs(n);

  // Allocate the I array (size m + 1) and zero-initialize it
  int *I = new int[m + 1]();

  // Step A: Count the number of non-zeros per parent row (will be exactly 0 or
  // 1)
  for (int i = 0; i < n; i++) {
    double sign = 1.0;
    p_vdofs[i] = FiniteElementSpace::DecodeDof(vdof_map[i], sign);
    p_signs[i] = sign;

    I[p_vdofs[i] + 1]++;  // Increment row count
  }

  // Step B: Prefix sum to generate actual row offsets
  for (int i = 0; i < m; i++) {
    I[i + 1] += I[i];
  }

  // Step C: Allocate J and Data, and populate them directly
  // Since each SubMesh DoF connects to exactly one Parent DoF, nnz = n.
  int *J = new int[n];
  double *Data = new double[n];

  for (int i = 0; i < n; i++) {
    int row = p_vdofs[i];

    // Because the mapping is injective, this row will only ever receive one
    // entry. I[row] gives the exact and only starting position for this row's
    // data.
    int pos = I[row];

    J[pos] = i;              // Column is the submesh DoF
    Data[pos] = p_signs[i];  // Value is the orientation sign
  }

  return {I, J, Data, m, n};
}

}  // namespace mfemElasticity