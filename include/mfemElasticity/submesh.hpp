#pragma once

#include "mfem.hpp"

namespace mfemElasticity {

class SubMeshProlongationMatrix : public mfem::SparseMatrix {
 private:
  struct CSRData {
    int *I;
    int *J;
    double *Data;
    int m, n;
  };

  static CSRData BuildCSR(const mfem::FiniteElementSpace &sub_fes,
                          const mfem::FiniteElementSpace &parent_fes);

  SubMeshProlongationMatrix(CSRData d)
      : mfem::SparseMatrix(d.I, d.J, d.Data, d.m, d.n) {}

 public:
  SubMeshProlongationMatrix(const mfem::FiniteElementSpace &sub_fes,
                            const mfem::FiniteElementSpace &parent_fes)
      : SubMeshProlongationMatrix(BuildCSR(sub_fes, parent_fes)) {}
};

}  // namespace mfemElasticity