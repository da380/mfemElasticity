/**
 * @file submesh.cpp
 * @brief Implementation of the SubMeshProlongationMatrix function.
 */

#include "mfemElasticity/submesh.hpp"

namespace mfemElasticity {

// Factory function to build the serial SubMesh prolongation matrix
mfem::SparseMatrix *SubMeshProlongationMatrix(
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

  // --- Build the SparseMatrix ---
  // Note: The MFEM SparseMatrix constructor taking (I, J, data) takes
  // ownership of the memory by default (own_ij = true, own_a = true).
  // Therefore, unlike the Hypre constructor, we DO NOT delete them here!
  bool own_ij = true;
  bool own_a = true;
  bool is_sorted = false;

  SparseMatrix *P =
      new SparseMatrix(I, J, Data, m, n, own_ij, own_a, is_sorted);

  return P;
}


using namespace mfem;
MixedBilinearFormSubMesh::MixedBilinearFormSubMesh(FiniteElementSpace *tr_fes,
                                                   FiniteElementSpace *te_fes,
                                                   FiniteElementSpace *sub_fes_,
                                                   bool extended_trial_)
    : MixedBilinearForm(tr_fes, te_fes),
      sub_fes(sub_fes_),
      extended_trial(extended_trial_)
{
    SubMesh *submesh = static_cast<SubMesh *>(sub_fes->GetMesh());

    vdof_to_vdof_map = new Array<int>();

    SubMesh::From from = submesh->GetFrom();
    Array<int> parent_element_ids = submesh->GetParentElementIDMap();

    if (extended_trial)
    {
        SubMeshUtils::BuildVdofToVdofMap(*sub_fes, *trial_fes,
                                         from, parent_element_ids,
                                         *vdof_to_vdof_map);
    }
    else
    {
        SubMeshUtils::BuildVdofToVdofMap(*sub_fes, *test_fes,
                                         from, parent_element_ids,
                                         *vdof_to_vdof_map);
    }
}

void MixedBilinearFormSubMesh::Assemble(int skip_zeros)
{
    ElementTransformation *eltrans;
    DenseMatrix elmat;

    Mesh *mesh = sub_fes->GetMesh();

    if (mat == NULL)
    {
        mat = new SparseMatrix(height, width);
    }

    // currently only supports domain integrators
    if (domain_integs.Size())
    {
        for (int k = 0; k < domain_integs.Size(); k++)
        {
            if (domain_integs_marker[k] != NULL)
            {
                MFEM_VERIFY(domain_integs_marker[k]->Size() ==
                            (mesh->attributes.Size() ? mesh->attributes.Max() : 0),
                            "invalid element marker for domain integrator #"
                            << k << ", counting from zero");
            }
        }

        DofTransformation dom_dof_trans, ran_dof_trans;
        for (int i = 0; i < sub_fes->GetNE(); i++)
        {
            const int elem_attr = mesh->GetAttribute(i);

            if (extended_trial)
            {
                sub_fes->GetElementVDofs(i, trial_vdofs, dom_dof_trans);
                test_fes->GetElementVDofs(i, test_vdofs, ran_dof_trans);
                eltrans = sub_fes->GetElementTransformation(i);

                elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
                elmat = 0.0;
                for (int k = 0; k < domain_integs.Size(); k++)
                {
                    if (domain_integs_marker[k] == NULL ||
                        (*(domain_integs_marker[k]))[elem_attr - 1] == 1)
                    {
                        domain_integs[k]->AssembleElementMatrix2(*sub_fes->GetFE(i),
                                                                 *test_fes->GetFE(i),
                                                                 *eltrans, elemmat);
                        elmat += elemmat;
                    }
                }
                TransformDual(ran_dof_trans, dom_dof_trans, elmat);

                Array<int> trial_vdofs_ext(trial_vdofs.Size());
                for (int l = 0; l < trial_vdofs.Size(); l++)
                {
                    real_t s1 = 1.0;
                    int sub_vdof = FiniteElementSpace::DecodeDof(trial_vdofs[l], s1);

                    real_t s2 = 1.0;
                    int parent_vdof = FiniteElementSpace::DecodeDof((*vdof_to_vdof_map)[sub_vdof], s2);

                    real_t s = s1 * s2;
                    trial_vdofs_ext[l] = (s > 0.0) ? parent_vdof : (-1 - parent_vdof);
                }

                mat->AddSubMatrix(test_vdofs, trial_vdofs_ext, elmat, skip_zeros);
            }
            else
            {
                trial_fes->GetElementVDofs(i, trial_vdofs, dom_dof_trans);
                sub_fes->GetElementVDofs(i, test_vdofs, ran_dof_trans);
                eltrans = sub_fes->GetElementTransformation(i);

                elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
                elmat = 0.0;
                for (int k = 0; k < domain_integs.Size(); k++)
                {
                    if (domain_integs_marker[k] == NULL ||
                        (*(domain_integs_marker[k]))[elem_attr - 1] == 1)
                    {
                        domain_integs[k]->AssembleElementMatrix2(*trial_fes->GetFE(i),
                                                                 *sub_fes->GetFE(i),
                                                                 *eltrans, elemmat);
                        elmat += elemmat;
                    }
                }
                TransformDual(ran_dof_trans, dom_dof_trans, elmat);

                Array<int> test_vdofs_ext(test_vdofs.Size());
                for (int l = 0; l < test_vdofs.Size(); l++)
                {
                    real_t s1 = 1.0;
                    int sub_vdof = FiniteElementSpace::DecodeDof(test_vdofs[l], s1);

                    real_t s2 = 1.0;
                    int parent_vdof = FiniteElementSpace::DecodeDof((*vdof_to_vdof_map)[sub_vdof], s2);

                    real_t s = s1 * s2;
                    test_vdofs_ext[l] = (s > 0.0) ? parent_vdof : (-1 - parent_vdof);
                }

                mat->AddSubMatrix(test_vdofs_ext, trial_vdofs, elmat, skip_zeros);
            }
        }
    }
}

#ifdef MFEM_USE_MPI
ParMixedBilinearFormSubMesh::ParMixedBilinearFormSubMesh(ParFiniteElementSpace *tr_pfes,
                                                         ParFiniteElementSpace *te_pfes,
                                                         ParFiniteElementSpace *sub_pfes_,
                                                         bool extended_trial_)
    : ParMixedBilinearForm(tr_pfes, te_pfes),
      sub_pfes(sub_pfes_),
      extended_trial(extended_trial_)
{
    ParSubMesh *psubmesh = static_cast<ParSubMesh *>(sub_pfes->GetMesh());

    vdof_to_vdof_map = new Array<int>();

    SubMesh::From from = psubmesh->GetFrom();
    Array<int> parent_element_ids = psubmesh->GetParentElementIDMap();

    if (extended_trial)
    {
        SubMeshUtils::BuildVdofToVdofMap(*sub_pfes, *trial_pfes,
                                         from, parent_element_ids,
                                         *vdof_to_vdof_map);
    }
    else
    {
        SubMeshUtils::BuildVdofToVdofMap(*sub_pfes, *test_pfes,
                                         from, parent_element_ids,
                                         *vdof_to_vdof_map);
    }
}

void ParMixedBilinearFormSubMesh::Assemble(int skip_zeros)
{
    ElementTransformation *eltrans;
    DenseMatrix elmat;

    Mesh *mesh = sub_pfes->GetMesh();

    if (mat == NULL)
    {
        mat = new SparseMatrix(height, width);
    }

    // currently only supports domain integrators
    if (domain_integs.Size())
    {
        for (int k = 0; k < domain_integs.Size(); k++)
        {
            if (domain_integs_marker[k] != NULL)
            {
                MFEM_VERIFY(domain_integs_marker[k]->Size() ==
                            (mesh->attributes.Size() ? mesh->attributes.Max() : 0),
                            "invalid element marker for domain integrator #"
                            << k << ", counting from zero");
            }
        }

        DofTransformation dom_dof_trans, ran_dof_trans;
        for (int i = 0; i < sub_pfes->GetNE(); i++)
        {
            const int elem_attr = mesh->GetAttribute(i);

            if (extended_trial)
            {
                sub_pfes->GetElementVDofs(i, trial_vdofs, dom_dof_trans);
                test_pfes->GetElementVDofs(i, test_vdofs, ran_dof_trans);
                eltrans = sub_pfes->GetElementTransformation(i);

                elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
                elmat = 0.0;
                for (int k = 0; k < domain_integs.Size(); k++)
                {
                    if (domain_integs_marker[k] == NULL ||
                        (*(domain_integs_marker[k]))[elem_attr - 1] == 1)
                    {
                        domain_integs[k]->AssembleElementMatrix2(*sub_pfes->GetFE(i),
                                                                 *test_pfes->GetFE(i),
                                                                 *eltrans, elemmat);
                        elmat += elemmat;
                    }
                }
                TransformDual(ran_dof_trans, dom_dof_trans, elmat);

                Array<int> trial_vdofs_ext(trial_vdofs.Size());
                for (int l = 0; l < trial_vdofs.Size(); l++)
                {
                    real_t s1 = 1.0;
                    int sub_vdof = FiniteElementSpace::DecodeDof(trial_vdofs[l], s1);

                    real_t s2 = 1.0;
                    int parent_vdof = FiniteElementSpace::DecodeDof((*vdof_to_vdof_map)[sub_vdof], s2);

                    real_t s = s1 * s2;
                    trial_vdofs_ext[l] = (s > 0.0) ? parent_vdof : (-1 - parent_vdof);
                }

                mat->AddSubMatrix(test_vdofs, trial_vdofs_ext, elmat, skip_zeros);
            }
            else
            {
                trial_pfes->GetElementVDofs(i, trial_vdofs, dom_dof_trans);
                sub_pfes->GetElementVDofs(i, test_vdofs, ran_dof_trans);
                eltrans = sub_pfes->GetElementTransformation(i);

                elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
                elmat = 0.0;
                for (int k = 0; k < domain_integs.Size(); k++)
                {
                    if (domain_integs_marker[k] == NULL ||
                        (*(domain_integs_marker[k]))[elem_attr - 1] == 1)
                    {
                        domain_integs[k]->AssembleElementMatrix2(*trial_pfes->GetFE(i),
                                                                 *sub_pfes->GetFE(i),
                                                                 *eltrans, elemmat);
                        elmat += elemmat;
                    }
                }
                TransformDual(ran_dof_trans, dom_dof_trans, elmat);

                Array<int> test_vdofs_ext(test_vdofs.Size());
                for (int l = 0; l < test_vdofs.Size(); l++)
                {
                    real_t s1 = 1.0;
                    int sub_vdof = FiniteElementSpace::DecodeDof(test_vdofs[l], s1);

                    real_t s2 = 1.0;
                    int parent_vdof = FiniteElementSpace::DecodeDof((*vdof_to_vdof_map)[sub_vdof], s2);

                    real_t s = s1 * s2;
                    test_vdofs_ext[l] = (s > 0.0) ? parent_vdof : (-1 - parent_vdof);
                }

                mat->AddSubMatrix(test_vdofs_ext, trial_vdofs, elmat, skip_zeros);
            }
        }
    }
}
#endif

}  // namespace mfemElasticity
