#include "mfemElasticity/Moments.hpp"

namespace mfemElasticity {

int RowDim(mfem::Mesh* mesh) {
  auto dim = mesh->Dimension();
  return 1 + dim + dim * (dim + 1) / 2;
}

MomentsOperator::MomentsOperator(mfem::FiniteElementSpace* fes,
                                 const mfem::Array<int>& dom_marker)
    : mfem::Operator(RowDim(fes->GetMesh()), fes->GetVSize()),
      _fes{fes},
      _dom_marker{dom_marker},
      _mat(RowDim(fes->GetMesh()), fes->GetVSize()) {}

#ifdef MFEM_USE_MPI
MomentsOperator::MomentsOperator(MPI_Comm comm,
                                 mfem::ParFiniteElementSpace* fes,
                                 const mfem::Array<int>& dom_marker)
    : mfem::Operator(RowDim(fes->GetMesh()), fes->GetVSize()),
      _parallel{true},
      _comm{comm},
      _pfes{fes},
      _fes{fes},
      _dom_marker{dom_marker},
      _mat(RowDim(fes->GetMesh()), fes->GetVSize()) {}

#endif

void MomentsOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

  y.SetSize(height);
  _mat.Mult(x, y);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, y.GetData(), height, MFEM_MPI_REAL_T, MPI_SUM,
                  _comm);
  }
#endif
}

void MomentsOperator::MultTranspose(const mfem::Vector& x,
                                    mfem::Vector& y) const {
  using namespace mfem;

  y.SetSize(width);
  _mat.MultTranspose(x, y);
}

void MomentsOperator::Assemble() {
  using namespace mfem;
  auto* mesh = _fes->GetMesh();

  auto elmat = DenseMatrix();
  auto vdofs = Array<int>();
  auto rows = Array<int>(width);
  for (auto i = 0; i < width; i++) {
    rows[i] = i;
  }

  for (auto i = 0; i < _fes->GetNBE(); i++) {
    const auto elm_attr = mesh->GetAttribute(i);
    if (_dom_marker[elm_attr - 1] == 1) {
      _fes->GetElementVDofs(i, vdofs);
      const auto* fe = _fes->GetFE(i);
      auto* Trans = _fes->GetElementTransformation(i);

      AssembleElementMatrix(*fe, *Trans, elmat);

      _mat.AddSubMatrix(rows, vdofs, elmat);
    }
  }

  _mat.Finalize();
}

}  // namespace mfemElasticity