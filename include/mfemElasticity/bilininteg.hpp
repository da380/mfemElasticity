#pragma once

#include <array>

#include "mfem.hpp"

namespace mfemElasticity {

// Base class for indexing vector, matrix and tensor fields.
class Index {
 private:
  int _dim;
  int _dof;

 public:
  Index(int dim, int dof) : _dim{dim}, _dof{dof} {}

  // Return dimension
  int Dim() const { return _dim; }

  // Return degrees of freedom
  int Dof() const { return _dof; }

  // Return the number of components
  virtual int ComponentSize() const = 0;

  // Return the size of the field.
  int Size() const { return _dof * ComponentSize(); }
};

// Class for indexing vector fields.
class VectorIndex : public Index {
 public:
  VectorIndex(int dim, int dof) : Index(dim, dof) {}

  // Returns the offset to the jth block.
  int Offset(int j) const { return j * Dof(); }

  // Returns the index of the jth component and the ith node.
  int operator()(int i, int j) const { return i + Offset(j); }

  int ComponentSize() const override { return Dim(); }
};

// Class for indexing matrix fields
class MatrixIndex : public Index {
 public:
  MatrixIndex(int dim, int dof) : Index(dim, dof) {}

  virtual int ComponentOffset(int j, int k) const { return j + Dim() * k; }

  // Offset to (j,k)th block.
  int Offset(int j, int k) const { return ComponentOffset(j, k) * Dof(); }

  // Index for (j,k)th componet at ith node.
  int operator()(int i, int j, int k) const { return i + Offset(j, k); }

  // Return the component size.
  int ComponentSize() const override { return Dim() * Dim(); }
};

// Class for indexing symmetric matrix fields.
class SymmetricMatrixIndex : public MatrixIndex {
 public:
  SymmetricMatrixIndex(int dim, int dof) : MatrixIndex(dim, dof) {}

  // Returns the offset for the (j,k)th block.
  int ComponentOffset(int j, int k) const override {
    if (j < k) {
      return ComponentOffset(k, j);
    } else {
      return (j + k * Dim() - k * (k + 1) / 2);
    }
  }

  // Return size of the matrix field.
  int ComponentSize() const override { return Dim() * (Dim() + 1) / 2; }
};

// Class for indexing trace-free symmetric matrices. Note that the indexing
// here is identical to that for symmetric matrices except for the final
// diagonal element being removed. This is not checked for in calls
// to the indexing or offset functions.
class TraceFreeSymmetricMatrixIndex : public SymmetricMatrixIndex {
 public:
  TraceFreeSymmetricMatrixIndex(int dim, int dof)
      : SymmetricMatrixIndex(dim, dof) {}

  // Return size of the matrix field.
  int ComponentSize() const override {
    return SymmetricMatrixIndex::ComponentSize() - 1;
  }
};

/*
BilinearformIntegrator that acts on a test vector field, v, and a trial
scalar field, u, according to:

(v,u) \mapsto \int_{\Omega} q_{i} v_{i} u dx

where \Omega is the domain and q a vector coefficient.

It is assumed that the vector field is defined on a finite element space formed
from the product of a scalar nodal space.

*/
class DomainVectorScalarIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::VectorCoefficient* QV;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector trial_shape, test_shape, qv;
  mfem::DenseMatrix part_elmat;
#endif

 public:
  /*
    To define an instance a vector coefficient is provided along, optionally,
    with an integration rule. The vector coefficient must return values with
    size equal to the spatial dimension as the finite-element space.
  */
  DomainVectorScalarIntegrator(mfem::VectorCoefficient& qv,
                               const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /*
  Set the default integration rule. The orders of the trial space, test space,
  and element transformation are taken into account Variations in the
  coefficient are not considered.
  */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /*
    Implementation of the element level assembly.
  */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

 protected:
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/*
BilinearFormIntegrator that acts on a test vector field, v, and a trial scalar
field, u, according to:

(v,u) \mapsto \int_{\Omega}  v_{i} q_{ij} u_{,j} dx

where \Omega is the domain and q a matrix coefficient.

The coefficient can be set as a scalar, q, in which case the matrix coefficient
is proportional to the identity matrix. It can also be set as a vector, this
corresponding to the matrix coefficient being diagonal.

It is assumed that the vector field is defined on a finite element space formed
from the product of a scalar nodal space. The scalar field must be defined on a
nodel space on which the gradient operator is defined.
*/
class DomainVectorGradScalarIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr;
  mfem::VectorCoefficient* QV = nullptr;
  mfem::MatrixCoefficient* QM = nullptr;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector test_shape, qv;
  mfem::DenseMatrix trial_dshape, part_elmat, qm, tm;
#endif

 public:
  /*
  Construct the bilinearform integrator from, optionally, an integration rule.
  In this case, the matrix coefficient is the identity matrix.
  */
  DomainVectorGradScalarIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /*
  Construct the bilinearform integrator from a scalar coefficient and,
  optionally, an integration rule. In this case, the matrix coefficient is
  proportional to the identity matrix.
  */
  DomainVectorGradScalarIntegrator(mfem::Coefficient& q,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /*
  Construct the bilinearform integrator from a vector coefficient and,
  optionally, and integration rule. In this case, the matrix coefficient is
  diagonal.
  */
  DomainVectorGradScalarIntegrator(mfem::VectorCoefficient& qv,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /*
  Construct the bilinearform integrator from a matrix coefficient and,
  optionally, an integration rule.
  */
  DomainVectorGradScalarIntegrator(mfem::MatrixCoefficient& qm,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QM{&qm} {}

  /*
  Set the default integration rule. The orders of the trial space, test space,
  and element transformation are taken into account, with one order removed to
  account for the spatial derivatives. Variations in the matrix coefficient are
  not considered.
  */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /*
  Implementation of the element level calculations.
  */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

 protected:
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/*
BilinearFormIntegrator that acts on a test vector field, v, and a trial scalar
field, u, according to:

(v,u) \mapsto \int_{\Omega} q v_{i,i} u dx

where \Omega is the domain and q a scalar coefficient.

It is assumed that the vector field is defined on a finite element space formed
from the product of a scalar nodal space on which the gradient operator is
defined.

*/
class DomainDivVectorScalarIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q;

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix test_dshape, part_elmat;
  mfem::Vector trial_shape;
#endif

 public:
  /*
  Construct the bilinearform integrator from, optionally, an integration rule.
  In this case the scalar coefficient is taken equal to the constant 1.
  */
  DomainDivVectorScalarIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{nullptr} {}

  /*
  Construct the bilinearform integrator from a scalar coefficient and,
  optionally, an integration rule.
  */
  DomainDivVectorScalarIntegrator(mfem::Coefficient& q,
                                  const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /*
  Set the default integration rule. The orders of the trial space, test space,
  and element transformation are taken into account, with one order removed to
  account for the spatial derivatives. Variations in the coefficient are
  not considered.
  */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /*
  Implementation of element level calculations.
  */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

 protected:
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/*
BilinearFormIntegrator that acts on a test vector field, v, and a trial vector
field, u, according to:

(v,u) \mapsto \int_{\Omega} q v_{i,i} u_{j,j} dx

where \Omega is the domain and q a scalar coefficient.

It is assumed that the vector fields are defined on finite element spaces
formed from the product of  scalar nodal spaces on which the gradient operator
are defined.

*/
class DomainDivVectorDivVectorIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr;

#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix trial_dshape, test_dshape;
#endif

 public:
  /*
  Construct bilinearform integrator from, optionally, an integration rule. In
  this case the coefficient is taken to be the constant 1.
  */
  DomainDivVectorDivVectorIntegrator(const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /*
  Construct the bilinear form integrator from a scalar coefficient and,
  optionally, and integration rule.
  */
  DomainDivVectorDivVectorIntegrator(mfem::Coefficient& q,
                                     const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /*
  Set the default integration rule. The orders of the trial space, test space,
  and element transformation are taken into account, with two order removed to
  account for the spatial derivatives. Variations in the coefficient are
  not considered.
  */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /*
  Implementation of the element level calculations.
  */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat);

  /*
  Assembly when the trial and test spaces are equal.
  */
  void AssembleElementMatrix(const mfem::FiniteElement& fe,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override {
    AssembleElementMatrix2(fe, fe, Trans, elmat);
  }

 protected:
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/*
BilinearFormIntegrator acting on test vector field, v, and trial vector field,
u, according to:

(v,u) \mapsto  \int_{\Omega} q v_{i} (w_{j} u_{j))_{,i} dx

where q is a scalar coefficient and w a vector coefficient.

It is assumed that the vector fields are defined on finite element spaces
formed from the product of  scalar nodal spaces. On the test space the gradient
operator must be defined. The vector coefficient should return vectors with size
equal to the mesh's spatial dimension.
*/
class DomainVectorGradVectorIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr;
  mfem::VectorCoefficient* QV;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector qv, test_shape;
  mfem::DenseMatrix trial_dshape, left_elmat, rigth_elmat_trans, part_elmat;
#endif

 public:
  /*
  Construct bilinearform integrator from vector coefficient and, optionally, an
  integration rule. In this case, the scalar coefficient is taken equal to the
  constant, 1.
  */
  DomainVectorGradVectorIntegrator(mfem::VectorCoefficient& qv,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /*
  Construct bilinearform integrator from the vector coefficient and scalar
  coefficient along, optionally, with an integration rule.
  */
  DomainVectorGradVectorIntegrator(mfem::VectorCoefficient& qv,
                                   mfem::Coefficient& q,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q}, QV{&qv} {}

  /*
  Set the default integration rule. The orders of the trial space, test space,
  and element transformation are taken into account, with one order removed to
  account for the spatial derivative. Variations in the coefficients are
  not considered.
  */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /*
  Implementation of element level calculations.
  */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

  /*
  Assembly when the trail and test spaces are equal.
  */
  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override {
    AssembleElementMatrix2(el, el, Trans, elmat);
  }

 protected:
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/*
BilinearFormIntegrator acting on a test vector field, v, and a
trial vector field, u, according to:

(v,u) \mapstp \int_{\Omega} q_{i} v_{i} u_{j,j} dx,

where \Omega is the domain and q a vector coefficient.
*/
class DomainVectorDivVectorIntegrator : public mfem::BilinearFormIntegrator {
 private:
  mfem::VectorCoefficient* QV = nullptr;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector qv, test_shape;
  mfem::DenseMatrix trial_dshape, part_elmat;
#endif

 public:
  /*
  Construct bilinearform integrator from vector coefficient and, optionally, an
  integration rule. In this case, the scalar coefficient is taken equal to the
  constant, 1.
  */
  DomainVectorDivVectorIntegrator(mfem::VectorCoefficient& qv,
                                  const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), QV{&qv} {}

  /*
  Set the default integration rule. The orders of the trial space, test space,
  and element transformation are taken into account, with one order removed to
  account for the spatial derivative. Variations in the coefficients are
  not considered.
  */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /*
  Implementation of element level calculations.
  */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

  /*
  Assembly when the trail and test spaces are equal.
  */
  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override {
    AssembleElementMatrix2(el, el, Trans, elmat);
  }

 protected:
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/*
BilinearFormIntegrator acting on a test matrix field, v, and a trial
vector field, u, according to:

(v,u) \mapsto \int_{\Omega} q v_{ij} u_{i,j} dx,

where \Omega is the domain and q a scalar coefficient.

The matrix field must be defined on a nodal finite element space formed from
the product of a scalar space. The ordering of the matrix components
corresponds to a dense matrix using column-major storage (i.e., v_{00},
v_{10}, v_{20}, v_{01}, ...). The vector field must be defined on a nodel
finite element space formed from the product of a scalar space for which the
gradient operator is defined. The vector and matrix fields need to have
compatible dimensions.
*/
class DomainMatrixDeformationGradientIntegrator
    : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q = nullptr;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector test_shape;
  mfem::DenseMatrix trial_dshape, part_elmat;
#endif

 public:
  /*
  Construct bilinearform integrator from, optionally, an integration rule. In
  this case, the scalar coefficient is taken equal to the constant, 1.
  */
  DomainMatrixDeformationGradientIntegrator(
      const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  /*
  Construct the bilinearform integrator from the scalar coefficient and,
  optionally, an integration rule.
  */
  DomainMatrixDeformationGradientIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /*
  Set the default integration rule. The orders of the trial space, test space,
  and element transformation are taken into account, with one order removed to
  account for the spatial derivative. Variations in the coefficients are
  not considered.
  */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /*
  Implementation of element-level calculations.
  */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

 protected:
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/*
BilinearFormIntegrator acting on a test symmetrix matrix field, v, and a trial
vector field, u, according to:

(v,u) \mapsto \frac{1}{2}\int_{\Omega} q v_{ij} (u_{i,j} + u_{j,i}) dx,

where \Omega is the domain and q a scalar coefficient.

The matrix field must be defined on a nodal finite element space formed from
the product of a scalar space. The ordering of the matrix components corresponds
to a dense matrix using column-major storage but storing only the lower
triangle. The vector field must be defined on a nodel finite element space
formed from the product of a scalar space for which the gradient operator is
defined. The vector and matrix fields need to have compatible dimensions.
*/
class DomainSymmetricMatrixStrainIntegrator
    : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector test_shape;
  mfem::DenseMatrix part_elmat, trial_dshape;
#endif

 public:
  DomainSymmetricMatrixStrainIntegrator(
      const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  DomainSymmetricMatrixStrainIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /*
  Set the default integration rule. The orders of the trial space, test space,
  and element transformation are taken into account, with one order removed to
  account for the spatial derivative. Variations in the coefficients are
  not considered.
  */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /*
  Implementation of element-level calculations.
  */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

 protected:
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/*
BilinearFormIntegrator acting on a test trace-free symmetrix matrix field, v,
and a trial vector field, u, according to:

(v,u) \mapsto \frac{1}{2}\int_{\Omega} q v_{ij} (u_{i,j} + u_{j,i} - (2/dim)
u_{k,k}\delta_{ij}) dx,

where \Omega is the domain and q a scalar coefficient.

The matrix field must be defined on a nodal finite element space formed from
the product of a scalar space. The ordering of the matrix components corresponds
to a dense matrix using column-major storage but storing only the lower
triangle and skipping the final elment. The vector field must be defined on a
nodel finite element space formed from the product of a scalar space for which
the gradient operator is defined. The vector and matrix fields need to have
compatible dimensions.
*/
class DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator
    : public mfem::BilinearFormIntegrator {
 private:
  mfem::Coefficient* Q;

#ifndef MFEM_THREAD_SAFE
  mfem::Vector test_shape;
  mfem::DenseMatrix part_elmat, trial_dshape;
#endif

 public:
  DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator(
      const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir) {}

  DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator(
      mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q{&q} {}

  /*
  Set the default integration rule. The orders of the trial space, test space,
  and element transformation are taken into account, with one order removed to
  account for the spatial derivative. Variations in the coefficients are
  not considered.
  */
  static const mfem::IntegrationRule& GetRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& Trans);

  /*
  Implementation of element-level calculations.
  */
  void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                              const mfem::FiniteElement& test_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;

 protected:
  const mfem::IntegrationRule* GetDefaultIntegrationRule(
      const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
      const mfem::ElementTransformation& trans) const override {
    return &GetRule(trial_fe, test_fe, trans);
  }
};

/*
DiscreteInterpolator that acts on a vector field, u, to return the matrix
field, v_{ij} = u_{i,j}. The matrix field components are stored using the
column-major format.

The vector field must be defined on a nodal finite element space formed
from the product of a scalar space on which the gradient operator is
defined. The matrix field must be defined on a nodal finite element space
formed from the product of a scalar space.
*/
class DeformationGradientInterpolator : public mfem::DiscreteInterpolator {
 private:
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix dshape;
#endif

 public:
  DeformationGradientInterpolator() {}

  void AssembleElementMatrix2(const mfem::FiniteElement& in_fe,
                              const mfem::FiniteElement& out_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/*
DiscreteInterpolator that acts on a vector field, u, to return the symmetric
matrix field, v_{ij} = (u_{i,j} + u_{j,i})/2. The matrix field components are
stored in the column-major format but only keeping elements in the lower
triangle. In 3D spaces this means: the ordering:

v_{00}, v_{10}, v_{20}, v_{11}, v_{21}, v_{22}

while in 2D we have

v_{00}, v_{10}, v_{11}
*/
class StrainInterpolator : public mfem::DiscreteInterpolator {
 private:
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix dshape;
#endif
 public:
  StrainInterpolator() {}

  void AssembleElementMatrix2(const mfem::FiniteElement& in_fe,
                              const mfem::FiniteElement& out_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

/*
DiscreteInterpolator that maps a vector field, u, into trace-free symmetric
matrix field, v, with components

v_{ij} = (u_{i,j} + u_{j,i})/2 - (1/dim) * u_{k,k} \delta_ij

*/
class DeviatoricStrainInterpolator : public mfem::DiscreteInterpolator {
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix dshape;
#endif
 public:
  DeviatoricStrainInterpolator() {}

  void AssembleElementMatrix2(const mfem::FiniteElement& in_fe,
                              const mfem::FiniteElement& out_fe,
                              mfem::ElementTransformation& Trans,
                              mfem::DenseMatrix& elmat) override;
};

}  // namespace mfemElasticity