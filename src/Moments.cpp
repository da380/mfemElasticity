#include "mfemElasticity/Moments.hpp"

namespace mfemElasticity {

int MomentsOperator::RowDim(mfem::Mesh* mesh) {
  auto dim = mesh->Dimension();
  return 1 + dim + dim * (dim + 1) / 2;
}

MomentsOperator::MomentsOperator(mfem::FiniteElementSpace* fes,
                                 const mfem::Array<int>& dom_marker)
    : mfem::Operator(RowDim(fes->GetMesh()), fes->GetVSize()),
      _moments_dim{RowDim(fes->GetMesh())},
      _fes{fes},
      _dom_marker{dom_marker},
      _mat(fes->GetVSize(), _moments_dim) {
#ifndef MFEM_THREAD_SAFE
  _c.SetSize(_moments_dim);
  _x.SetSize(_fes->GetMesh()->Dimension());
#endif
}

#ifdef MFEM_USE_MPI
MomentsOperator::MomentsOperator(MPI_Comm comm,
                                 mfem::ParFiniteElementSpace* fes,
                                 const mfem::Array<int>& dom_marker)
    : mfem::Operator(RowDim(fes->GetMesh()), fes->GetVSize()),
      _parallel{true},
      _comm{comm},
      _pfes{fes},
      _moments_dim{RowDim(fes->GetMesh())},
      _fes{fes},
      _dom_marker{dom_marker},
      _mat(fes->GetVSize(), _moments_dim) {
#ifndef MFEM_THREAD_SAFE
  _c.SetSize(_moments_dim);
  _x.SetSize(_fes->GetMesh()->Dimension());
#endif
}

#endif

void MomentsOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

  y.SetSize(_moments_dim);
  _mat.MultTranspose(x, y);

#ifdef MFEM_USE_MPI
  if (_parallel) {
    MPI_Allreduce(MPI_IN_PLACE, y.GetData(), _moments_dim, MFEM_MPI_REAL_T,
                  MPI_SUM, _comm);
  }
#endif
}

void MomentsOperator::MultTranspose(const mfem::Vector& x,
                                    mfem::Vector& y) const {
  using namespace mfem;

  y.SetSize(width);
  _mat.Mult(x, y);
}

void MomentsOperator::Assemble() {
  using namespace mfem;
  auto* mesh = _fes->GetMesh();

  auto elmat = DenseMatrix();
  auto vdofs = Array<int>();
  auto rows = Array<int>(_moments_dim);
  for (auto i = 0; i < _moments_dim; i++) {
    rows[i] = i;
  }

  for (auto i = 0; i < _fes->GetNE(); i++) {
    const auto elm_attr = mesh->GetAttribute(i);
    if (_dom_marker[elm_attr - 1] == 1) {
      _fes->GetElementVDofs(i, vdofs);
      const auto* fe = _fes->GetFE(i);
      auto* Trans = _fes->GetElementTransformation(i);

      AssembleElementMatrix(*fe, *Trans, elmat);

      _mat.AddSubMatrix(vdofs, rows, elmat);
    }
  }

  _mat.Finalize();
}

void MomentsOperator::AssembleElementMatrix(const mfem::FiniteElement& fe,
                                            mfem::ElementTransformation& Trans,
                                            mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dim = _fes->GetMesh()->Dimension();
  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector _c, _x, shape;
  _x.SetSize(dim);
  _c.SetSize(_moments_dim);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, _moments_dim);
  elmat = 0.0;

  const auto* ir = GetIntegrationRule(fe, Trans);
  if (ir == nullptr) {
    int intorder = fe.GetOrder() + Trans.OrderW();
    ir = &IntRules.Get(fe.GetGeomType(), intorder);
  }

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, _x);

    // Degree zero term.
    auto i = 0;
    _c(i++) = 1;

    // Degree one terms.
    for (auto k = 0; k < dim; k++) {
      _c(i++) = _x(k);
    }

    // Degree two terms.
    for (auto l = 0; l < dim; l++) {
      for (auto k = l; k < dim; k++) {
        _c(i++) = _x(k) * _x(l);
      }
    }

    fe.CalcShape(ip, shape);
    auto w = Trans.Weight() * ip.weight;

    AddMult_a_VWt(w, shape, _c, elmat);
  }
}

void MomentsOperator::Centroid(const mfem::Vector& moments_vector,
                               mfem::Vector& centroid) const {
  auto dim = _fes->GetMesh()->Dimension();
  centroid.SetSize(dim);
  for (auto i = 0; i < dim; i++) {
    centroid(i) = moments_vector(i + 1) / moments_vector(0);
  }
}

mfem::Vector MomentsOperator::Centroid(
    const mfem::Vector& moments_vector) const {
  auto centroid = mfem::Vector();
  Centroid(moments_vector, centroid);
  return centroid;
}

void MomentsOperator::InertiaTensor(const mfem::Vector& moments_vector,
                                    mfem::DenseMatrix& inertia_tensor) const {
  using namespace mfem;
  auto dim = _fes->GetMesh()->Dimension();
  inertia_tensor.SetSize(dim, dim);

  auto centroid = Centroid(moments_vector);

  auto* it = &moments_vector[1 + dim];
  for (auto j = 0; j < dim; j++) {
    for (auto i = j; i < dim; i++) {
      inertia_tensor(i, j) = -(*it++);
      if (i != j) {
        inertia_tensor(j, i) = inertia_tensor(i, j);
      }
    }
  }
  auto trace = -inertia_tensor.Trace();
  for (auto i = 0; i < dim; i++) {
    inertia_tensor(i, i) += trace;
  }
}

mfem::DenseMatrix MomentsOperator::InertiaTensor(
    const mfem::Vector& moments_vector) const {
  auto inertia_tensor = mfem::DenseMatrix();
  InertiaTensor(moments_vector, inertia_tensor);
  return inertia_tensor;
}

}  // namespace mfemElasticity