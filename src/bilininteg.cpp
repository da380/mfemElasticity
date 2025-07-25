
#include "mfemElasticity/bilininteg.hpp"

#include <algorithm>
#include <cstddef>

namespace mfemElasticity {

const mfem::IntegrationRule& DomainVectorScalarIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainVectorScalarIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;
  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  auto same_shape = &trial_fe == &test_fe;

  elmat.SetSize(space_dim * test_dof, trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector trial_shape, test_shape, qv;
  DenseMatrix part_elmat;
#endif
  trial_shape.SetSize(trial_dof);
  qv.SetSize(space_dim);
  part_elmat.SetSize(test_dof, trial_dof);

  if (same_shape) {
    test_shape.NewDataAndSize(trial_shape.GetData(), test_dof);
  } else {
    test_shape.SetSize(test_dof);
  }
  const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    trial_fe.CalcShape(ip, trial_shape);
    if (!same_shape) {
      test_fe.CalcShape(ip, test_shape);
    }

    QV->Eval(qv, Trans, ip);
    MultVWt(test_shape, trial_shape, part_elmat);
    for (auto j = 0; j < space_dim; j++) {
      elmat.AddMatrix(w * qv(j), part_elmat, test_dof * j, 0);
    }
  }
}

const mfem::IntegrationRule& DomainVectorGradScalarIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainVectorGradScalarIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;
  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  elmat.SetSize(space_dim * test_dof, trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector test_shape, qv;
  DenseMatrix trial_dshape, part_elmat, qm, tm;
#endif
  test_shape.SetSize(test_dof);
  trial_dshape.SetSize(trial_dof, space_dim);
  part_elmat.SetSize(test_dof, trial_dof);

  if (QM) {
    qm.SetSize(space_dim);
    tm.SetSize(trial_dof, space_dim);
  } else if (QV) {
    qv.SetSize(space_dim);
  }

  const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    test_fe.CalcShape(ip, test_shape);
    trial_fe.CalcPhysDShape(Trans, trial_dshape);

    if (QM) {
      QM->Eval(qm, Trans, ip);
      MultABt(trial_dshape, qm, tm);
      for (auto j = 0; j < space_dim; j++) {
        auto tm_column = Vector(tm.GetColumn(j), trial_dof);
        MultVWt(test_shape, tm_column, part_elmat);
        elmat.AddMatrix(w, part_elmat, j * test_dof, 0);
      }
    } else if (QV) {
      QV->Eval(qv, Trans, ip);
      qv *= w;
      for (auto j = 0; j < space_dim; j++) {
        auto trial_dshape_column = Vector(trial_dshape.GetColumn(j), trial_dof);
        MultVWt(test_shape, trial_dshape_column, part_elmat);
        elmat.AddMatrix(qv(j), part_elmat, j * test_dof, 0);
      }
    } else {
      if (Q) {
        w *= Q->Eval(Trans, ip);
      }
      for (auto j = 0; j < space_dim; j++) {
        auto trial_dshape_column = Vector(trial_dshape.GetColumn(j), trial_dof);
        MultVWt(test_shape, trial_dshape_column, part_elmat);
        elmat.AddMatrix(w, part_elmat, j * test_dof, 0);
      }
    }
  }
}

const mfem::IntegrationRule& DomainDivVectorScalarIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainDivVectorScalarIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  elmat.SetSize(space_dim * test_dof, trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  auto test_dshape = mfem::DenseMatrix();
  auto trial_shape = mfem::Vector();
  auto part_elmat = mfem::DenseMatrix();
#endif
  test_dshape.SetSize(test_dof, space_dim);
  trial_shape.SetSize(trial_dof);
  part_elmat.SetSize(test_dof, trial_dof);

  const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);

    test_fe.CalcPhysDShape(Trans, test_dshape);
    trial_fe.CalcShape(ip, trial_shape);

    auto w = Trans.Weight() * ip.weight;
    if (Q) {
      w *= Q->Eval(Trans, ip);
    }

    for (auto j = 0; j < space_dim; j++) {
      auto test_dshape_column =
          mfem::Vector(test_dshape.GetColumn(j), test_dof);
      mfem::MultVWt(test_dshape_column, trial_shape, part_elmat);
      elmat.AddMatrix(w, part_elmat, j * test_dof, 0);
    }
  }
}

const mfem::IntegrationRule& DomainDivVectorDivVectorIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 2;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainDivVectorDivVectorIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  auto same_spaces = &test_fe == &trial_fe;

  elmat.SetSize(space_dim * test_dof, space_dim * trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  auto trial_dshape = DenseMatrix();
  auto test_dshape = DenseMatrix();
#endif
  trial_dshape.SetSize(trial_dof, space_dim);

  if (same_spaces) {
    test_dshape.Reset(trial_dshape.GetData(), test_dof, space_dim);
  } else {
    test_dshape.SetSize(test_dof, space_dim);
  }

  const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);

    trial_fe.CalcPhysDShape(Trans, trial_dshape);
    if (!same_spaces) {
      test_fe.CalcPhysDShape(Trans, test_dshape);
    }

    auto w = Trans.Weight() * ip.weight;
    if (Q) {
      w *= Q->Eval(Trans, ip);
    }

    auto test_dshape_vector =
        Vector(test_dshape.GetData(), space_dim * test_dof);
    auto trial_dshape_vector =
        Vector(trial_dshape.GetData(), space_dim * trial_dof);
    AddMult_a_VWt(w, test_dshape_vector, trial_dshape_vector, elmat);
  }
}

const mfem::IntegrationRule& DomainVectorGradVectorIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainVectorGradVectorIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  elmat.SetSize(space_dim * test_dof, space_dim * trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector qv(), test_shape();
  DenseMatrix left_elmat(), rigth_elmat_trans(), part_elmat();
#endif

  qv.SetSize(space_dim);
  rigth_elmat_trans.SetSize(space_dim * trial_dof, trial_dof);
  rigth_elmat_trans = 0.;

  const auto& trial_nodes = trial_fe.GetNodes();
  for (auto i = 0; i < trial_dof; i++) {
    const auto& ip = trial_nodes.IntPoint(i);
    Trans.SetIntPoint(&ip);
    QV->Eval(qv, Trans, ip);
    for (auto j = 0; j < space_dim; j++) {
      rigth_elmat_trans(i + trial_dof * j, i) = qv(j);
    }
  }

  part_elmat.SetSize(test_dof, trial_dof);
  left_elmat.SetSize(space_dim * test_dof, trial_dof);
  left_elmat = 0.;

  trial_dshape.SetSize(trial_dof, space_dim);
  test_shape.SetSize(test_dof);

  const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    trial_fe.CalcPhysDShape(Trans, trial_dshape);
    test_fe.CalcShape(ip, test_shape);

    if (Q) {
      w *= Q->Eval(Trans, ip);
    }

    for (auto j = 0; j < space_dim; j++) {
      auto trial_dshape_column = Vector(trial_dshape.GetColumn(j), trial_dof);
      MultVWt(test_shape, trial_dshape_column, part_elmat);
      left_elmat.AddMatrix(w, part_elmat, test_dof * j, 0);
    }
  }
  MultABt(left_elmat, rigth_elmat_trans, elmat);
}

const mfem::IntegrationRule& DomainVectorDivVectorIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainVectorDivVectorIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  elmat.SetSize(space_dim * test_dof, space_dim * trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector qv(), test_shape();
  DenseMatrix part_elmat();
#endif

  qv.SetSize(space_dim);
  part_elmat.SetSize(test_dof, trial_dof);

  trial_dshape.SetSize(trial_dof, space_dim);
  test_shape.SetSize(test_dof);

  const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    QV->Eval(qv, Trans, ip);
    qv *= w;

    test_fe.CalcShape(ip, test_shape);
    trial_fe.CalcPhysDShape(Trans, trial_dshape);

    for (auto k = 0; k < space_dim; k++) {
      auto trial_dshape_column = Vector(trial_dshape.GetColumn(k), trial_dof);
      MultVWt(test_shape, trial_dshape_column, part_elmat);
      for (auto j = 0; j < space_dim; j++) {
        elmat.AddMatrix(qv(j), part_elmat, j * test_dof, k * trial_dof);
      }
    }
  }
}

const mfem::IntegrationRule& DomainMatrixDeformationGradientIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainMatrixDeformationGradientIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  auto vectorIndex = VectorIndex(space_dim, trial_dof);
  auto matrixIndex = MatrixIndex(space_dim, test_dof);

  elmat.SetSize(matrixIndex.Size(), vectorIndex.Size());
  elmat = 0.;

  const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

#ifdef MFEM_THREAD_SAFE
  Vector test_shape();
  DenseMatrix trial_dshape(), part_elmat();
#endif
  test_shape.SetSize(test_dof);
  trial_dshape.SetSize(trial_dof, space_dim);
  part_elmat.SetSize(test_dof, trial_dof);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    trial_fe.CalcPhysDShape(Trans, trial_dshape);
    test_fe.CalcShape(ip, test_shape);

    if (Q) {
      w *= Q->Eval(Trans, ip);
    }

    for (auto k = 0; k < space_dim; k++) {
      auto trial_dshape_column = Vector(trial_dshape.GetColumn(k), trial_dof);
      MultVWt(test_shape, trial_dshape_column, part_elmat);
      for (auto j = 0; j < space_dim; j++) {
        elmat.AddMatrix(w, part_elmat, matrixIndex.Offset(j, k),
                        vectorIndex.Offset(j));
      }
    }
  }
}

const mfem::IntegrationRule& DomainSymmetricMatrixStrainIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainSymmetricMatrixStrainIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  auto vectorIndex = VectorIndex(space_dim, trial_dof);
  auto matrixIndex = SymmetricMatrixIndex(space_dim, test_dof);

  elmat.SetSize(matrixIndex.Size(), vectorIndex.Size());
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector test_shape();
  DenseMatrix trial_dshape(), part_elmat();
#endif
  test_shape.SetSize(test_dof);
  trial_dshape.SetSize(trial_dof, space_dim);
  part_elmat.SetSize(test_dof, trial_dof);

  const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    if (Q) {
      w *= Q->Eval(Trans, ip);
    }

    trial_fe.CalcPhysDShape(Trans, trial_dshape);
    test_fe.CalcShape(ip, test_shape);

    for (auto k = 0; k < space_dim; k++) {
      auto trial_dshape_column = Vector(trial_dshape.GetColumn(k), trial_dof);
      MultVWt(test_shape, trial_dshape_column, part_elmat);
      for (auto j = 0; j < space_dim; j++) {
        elmat.AddMatrix(w, part_elmat, matrixIndex.Offset(j, k),
                        vectorIndex.Offset(j));
      }
    }
  }
}

const mfem::IntegrationRule&
DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order =
      trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator::
    AssembleElementMatrix2(const mfem::FiniteElement& trial_fe,
                           const mfem::FiniteElement& test_fe,
                           mfem::ElementTransformation& Trans,
                           mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  auto vectorIndex = VectorIndex(space_dim, trial_dof);
  auto matrixIndex = TraceFreeSymmetricMatrixIndex(space_dim, test_dof);

  elmat.SetSize(matrixIndex.Size(), vectorIndex.Size());
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector test_shape();
  DenseMatrix trial_dshape(), part_elmat();
#endif
  test_shape.SetSize(test_dof);
  trial_dshape.SetSize(trial_dof, space_dim);
  part_elmat.SetSize(test_dof, trial_dof);

  const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);
    auto w = Trans.Weight() * ip.weight;

    if (Q) {
      w *= Q->Eval(Trans, ip);
    }

    trial_fe.CalcPhysDShape(Trans, trial_dshape);
    test_fe.CalcShape(ip, test_shape);

    for (auto k = 0; k < space_dim - 1; k++) {
      auto trial_dshape_column = Vector(trial_dshape.GetColumn(k), trial_dof);
      MultVWt(test_shape, trial_dshape_column, part_elmat);

      for (auto j = 0; j < space_dim; j++) {
        elmat.AddMatrix(w, part_elmat, matrixIndex.Offset(j, k),
                        vectorIndex.Offset(j));
      }
    }

    auto k = space_dim - 1;
    auto trial_dshape_column = Vector(trial_dshape.GetColumn(k), trial_dof);
    MultVWt(test_shape, trial_dshape_column, part_elmat);

    for (auto j = 0; j < space_dim - 1; j++) {
      elmat.AddMatrix(w, part_elmat, matrixIndex.Offset(j, k),
                      vectorIndex.Offset(j));
      elmat.AddMatrix(-w, part_elmat, matrixIndex.Offset(j, j),
                      vectorIndex.Offset(k));
    }
  }
}

void DeformationGradientInterpolator::AssembleElementMatrix2(
    const mfem::FiniteElement& in_fe, const mfem::FiniteElement& out_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = in_fe.GetDim();
  auto in_dof = in_fe.GetDof();
  auto out_dof = out_fe.GetDof();

  auto vectorIndex = VectorIndex(space_dim, in_dof);
  auto matrixIndex = MatrixIndex(space_dim, out_dof);

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape();
#endif
  dshape.SetSize(in_dof, space_dim);
  elmat.SetSize(matrixIndex.Size(), vectorIndex.Size());
  elmat = 0.;

  const auto& nodes = out_fe.GetNodes();

  for (auto i = 0; i < out_dof; i++) {
    const auto& ip = nodes.IntPoint(i);
    Trans.SetIntPoint(&ip);
    in_fe.CalcPhysDShape(Trans, dshape);
    for (auto l = 0; l < space_dim; l++) {
      for (auto j = 0; j < in_dof; j++) {
        for (auto k = 0; k < space_dim; k++) {
          elmat(matrixIndex(i, k, l), vectorIndex(j, k)) += dshape(j, l);
        }
      }
    }
  }
}

void StrainInterpolator::AssembleElementMatrix2(
    const mfem::FiniteElement& in_fe, const mfem::FiniteElement& out_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  constexpr auto half = static_cast<real_t>(1) / static_cast<real_t>(2);

  auto space_dim = in_fe.GetDim();
  auto in_dof = in_fe.GetDof();
  auto out_dof = out_fe.GetDof();

  auto vectorIndex = VectorIndex(space_dim, in_dof);
  auto matrixIndex = SymmetricMatrixIndex(space_dim, out_dof);

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape();
#endif
  dshape.SetSize(in_dof, space_dim);
  elmat.SetSize(matrixIndex.Size(), vectorIndex.Size());
  elmat = 0.;

  const auto& nodes = out_fe.GetNodes();

  for (auto i = 0; i < out_dof; i++) {
    const IntegrationPoint& ip = nodes.IntPoint(i);
    Trans.SetIntPoint(&ip);
    in_fe.CalcPhysDShape(Trans, dshape);

    for (auto l = 0; l < space_dim; l++) {
      for (auto j = 0; j < in_dof; j++) {
        for (auto k = l; k < space_dim; k++) {
          elmat(matrixIndex(i, k, l), vectorIndex(j, l)) += half * dshape(j, k);
          elmat(matrixIndex(i, k, l), vectorIndex(j, k)) += half * dshape(j, l);
        }
      }
    }
  }
}

void DeviatoricStrainInterpolator::AssembleElementMatrix2(
    const mfem::FiniteElement& in_fe, const mfem::FiniteElement& out_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = in_fe.GetDim();
  auto in_dof = in_fe.GetDof();
  auto out_dof = out_fe.GetDof();

  auto vectorIndex = VectorIndex(space_dim, in_dof);
  auto matrixIndex = TraceFreeSymmetricMatrixIndex(space_dim, out_dof);

#ifdef MFEM_THREAD_SAFE
  auto dshape = DenseMatrix();
#endif
  dshape.SetSize(in_dof, space_dim);
  elmat.SetSize(matrixIndex.Size(), vectorIndex.Size());
  elmat = 0.;

  constexpr auto half = static_cast<real_t>(1) / static_cast<real_t>(2);
  const auto space_dim_inverse = static_cast<real_t>(1) / space_dim;

  const auto& nodes = out_fe.GetNodes();

  for (auto i = 0; i < out_dof; i++) {
    const IntegrationPoint& ip = nodes.IntPoint(i);
    Trans.SetIntPoint(&ip);
    in_fe.CalcPhysDShape(Trans, dshape);

    for (auto l = 0; l < space_dim - 1; l++) {
      for (auto j = 0; j < in_dof; j++) {
        for (auto k = l; k < space_dim; k++) {
          elmat(matrixIndex(i, k, l), vectorIndex(j, k)) += half * dshape(j, l);
          elmat(matrixIndex(i, k, l), vectorIndex(j, l)) += half * dshape(j, k);
        }
        for (auto k = 0; k < space_dim; k++) {
          elmat(matrixIndex(i, l, l), vectorIndex(j, k)) -=
              space_dim_inverse * dshape(j, k);
        }
      }
    }
  }
}

const mfem::IntegrationRule& TransformedLaplaceIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void TransformedLaplaceIntegrator::AssembleElementMatrix2(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dim = Trans.GetSpaceDim();
  auto trial_dof = trial_fe.GetDof();
  auto test_dof = test_fe.GetDof();

  auto same_spaces = &test_fe == &trial_fe;

  elmat.SetSize(test_dof, trial_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  DenseMatrix trial_dshape, test_dshape, xi, F, A, B;
#endif
  trial_dshape.SetSize(trial_dof, dim);

  if (same_spaces) {
    test_dshape.Reset(trial_dshape.GetData(), test_dof, dim);
  } else {
    test_dshape.SetSize(test_dof, dim);
  }

  if (Q || QV || QM) {
    F.SetSize(dim, dim);
    A.SetSize(dim, dim);
    B.SetSize(test_dof, dim);
  }

  if (Q) {
    const auto& ir = test_fe.GetNodes();
  }

  if (QV) {
    const auto& ir = test_fe.GetNodes();
    QV->Eval(xi, Trans, ir);
  }

  const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);

    trial_fe.CalcPhysDShape(Trans, trial_dshape);
    if (!same_spaces) {
      test_fe.CalcPhysDShape(Trans, test_dshape);
    }

    auto w = Trans.Weight() * ip.weight;

    if (Q) {
    }

    if (QV) {
      Mult(xi, test_dshape, F);
      auto J = F.Det();
      F.Invert();
      MultABt(F, F, A);
      A *= J;
    }

    if (QM) {
      QM->Eval(A, Trans, ip);
    }

    if (Q || QV || QM) {
      MultABt(test_dshape, A, B);
      AddMult_a_ABt(w, test_dshape, B, elmat);
    } else {
      AddMult_a_ABt(w, test_dshape, trial_dshape, elmat);
    }
  }
}

}  // namespace mfemElasticity