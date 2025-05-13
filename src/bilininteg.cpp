
#include "mfemElasticity/bilininteg.hpp"

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
  auto trial_shape = Vector(test_dof);
  auto test_shape = Vector();
  auto qv = Vector(space_dim);
  auto partElmat = DenseMatrix(test_dof, trial_dof);
#else
  trial_shape.SetSize(trial_dof);

  qv.SetSize(space_dim);
  partElmat.SetSize(test_dof, trial_dof);
#endif

  const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

  if (same_shape) {
    test_shape.NewDataAndSize(trial_shape.GetData(), test_dof);
  } else {
    test_shape.SetSize(test_dof);
  }

  for (auto i = 0; i < ir->GetNPoints(); i++) {
    const auto& ip = ir->IntPoint(i);
    Trans.SetIntPoint(&ip);

    auto w = Trans.Weight() * ip.weight;

    trial_fe.CalcShape(ip, trial_shape);
    if (!same_shape) {
      test_fe.CalcShape(ip, test_shape);
    }

    QV->Eval(qv, Trans, ip);
    MultVWt(test_shape, trial_shape, partElmat);
    for (auto j = 0; j < space_dim; j++) {
      elmat.AddMatrix(w * qv(j), partElmat, trial_dof * j, 0);
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
  auto test_shape = Vector(test_dof);
  auto trial_dshape = DenseMatrix(trial_dof, space_dim);
  auto partElmat = DenseMatrix(test_dof, trial_dof);
  auto qm = DenseMatrix();
  auto tm = DenseMatrix();
  auto qv = Vector();
#else
  test_shape.SetSize(test_dof);
  trial_dshape.SetSize(trial_dof, space_dim);
  partElmat.SetSize(test_dof, trial_dof);
#endif

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
        MultVWt(test_shape, tm_column, partElmat);
        elmat.AddMatrix(w, partElmat, j * test_dof, 0);
      }
    } else if (QV) {
      QV->Eval(qv, Trans, ip);
      qv *= w;
      for (auto j = 0; j < space_dim; j++) {
        auto trial_dshape_column = Vector(trial_dshape.GetColumn(j), trial_dof);
        MultVWt(test_shape, trial_dshape_column, partElmat);
        elmat.AddMatrix(w, partElmat, j * test_dof, 0);
      }
    } else {
      if (Q) {
        w *= Q->Eval(Trans, ip);
      }
      for (auto j = 0; j < space_dim; j++) {
        auto trial_dshape_column = Vector(trial_dshape.GetColumn(j), trial_dof);
        MultVWt(test_shape, trial_dshape_column, partElmat);
        elmat.AddMatrix(w, partElmat, j * test_dof, 0);
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
  auto partElmat = mfem::DenseMatrix();
#endif
  test_dshape.SetSize(test_dof, space_dim);
  trial_shape.SetSize(trial_dof);
  partElmat.SetSize(test_dof, trial_dof);

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
      mfem::MultVWt(test_dshape_column, trial_shape, partElmat);
      elmat.AddMatrix(w, partElmat, j * test_dof, 0);
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
  DenseMatrix leftElmat(), rightElmatTrans(), partElmat();
#endif

  qv.SetSize(space_dim);
  rightElmatTrans.SetSize(space_dim * trial_dof, trial_dof);
  rightElmatTrans = 0.;

  const auto& trial_nodes = trial_fe.GetNodes();
  for (auto i = 0; i < trial_dof; i++) {
    const auto& ip = trial_nodes.IntPoint(i);
    Trans.SetIntPoint(&ip);
    QV->Eval(qv, Trans, ip);
    for (auto j = 0; j < space_dim; j++) {
      rightElmatTrans(i + trial_dof * j, i) = qv(j);
    }
  }

  partElmat.SetSize(test_dof, trial_dof);
  leftElmat.SetSize(space_dim * test_dof, trial_dof);
  leftElmat = 0.;

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
      MultVWt(test_shape, trial_dshape_column, partElmat);
      leftElmat.AddMatrix(w, partElmat, test_dof * j, 0);
    }
  }
  MultABt(leftElmat, rightElmatTrans, elmat);
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

  elmat.SetSize(space_dim * space_dim * test_dof, space_dim * trial_dof);
  elmat = 0.;

  const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

#ifdef MFEM_THREAD_SAFE
  Vector test_shape();
  DenseMatrix trial_dshape(), partElmat();
#endif
  test_shape.SetSize(test_dof);
  trial_dshape.SetSize(trial_dof, space_dim);
  partElmat.SetSize(test_dof, trial_dof);

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
      MultVWt(test_shape, trial_dshape_column, partElmat);
      for (auto k = 0; k < space_dim; k++) {
        elmat.AddMatrix(w, partElmat, test_dof * (k + space_dim * j),
                        trial_dof * k);
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

  auto strain_dim = space_dim * (space_dim + 1) / 2;
  elmat.SetSize(strain_dim * trial_dof, space_dim * test_dof);
  elmat = 0.;

#ifdef MFEM_THREAD_SAFE
  Vector test_shape();
  DenseMatrix trial_dshape(), partElmat();
#endif
  test_shape.SetSize(test_dof);
  trial_dshape.SetSize(trial_dof, space_dim);
  partElmat.SetSize(test_dof, trial_dof);

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

    auto it = space_dim == 2 ? index2.begin() : index3.begin();
    for (auto k = 0; k < space_dim; k++) {
      auto trial_dshape_column = Vector(trial_dshape.GetColumn(k), trial_dof);
      MultVWt(test_shape, trial_dshape_column, partElmat);

      for (auto j = 0; j < space_dim; j++) {
        elmat.AddMatrix(w, partElmat, *it++ * test_dof, trial_dof * k);
      }
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

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape();
#endif
  dshape.SetSize(in_dof, space_dim);
  elmat.SetSize(space_dim * space_dim * out_dof, space_dim * in_dof);
  elmat = 0.;

  const auto& nodes = out_fe.GetNodes();
  for (auto i = 0; i < out_dof; i++) {
    const auto& ip = nodes.IntPoint(i);
    Trans.SetIntPoint(&ip);
    in_fe.CalcPhysDShape(Trans, dshape);
    for (auto q = 0; q < space_dim; q++) {
      for (auto j = 0; j < in_dof; j++) {
        for (auto p = 0; p < space_dim; p++) {
          elmat(i + out_dof * (p + q * space_dim), j + in_dof * p) =
              dshape(j, q);
        }
      }
    }
  }
}

void StrainInterpolator::AssembleElementMatrix2(
    const mfem::FiniteElement& in_fe, const mfem::FiniteElement& out_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto space_dim = in_fe.GetDim();
  auto in_dof = in_fe.GetDof();
  auto out_dof = out_fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape();
#endif
  dshape.SetSize(in_dof, space_dim);
  elmat.SetSize(space_dim * (space_dim + 1) * out_dof / 2, space_dim * in_dof);
  elmat = 0.;

  const auto& nodes = out_fe.GetNodes();
  for (auto i = 0; i < out_dof; i++) {
    const IntegrationPoint& ip = nodes.IntPoint(i);
    Trans.SetIntPoint(&ip);
    in_fe.CalcPhysDShape(Trans, dshape);

    constexpr auto half = static_cast<real_t>(1) / static_cast<real_t>(2);

    if (space_dim == 2) {
      for (auto j = 0; j < in_dof; j++) {
        // e_{00} = u_{0,0}
        elmat(i, j) = dshape(j, 0);

        // e_{10} = 0.5 * (u_{0,1} + u_{1,0})
        elmat(i + out_dof, j) = half * dshape(j, 1);
        elmat(i + out_dof, j + in_dof) = half * dshape(j, 0);

        // e_{11} = u_{1,1}
        elmat(i + 2 * out_dof, j + in_dof) = dshape(j, 1);
      }
    } else {
      for (auto j = 0; j < in_dof; j++) {
        // e_{00} = u_{0,0}
        elmat(i, j) = dshape(j, 0);

        // e_{10} = 0.5 * (u_{0,1} + u_{1,0})
        elmat(i + out_dof, j) = half * dshape(j, 1);
        elmat(i + out_dof, j + in_dof) = half * dshape(j, 0);

        // e_{20} = 0.5 * (u_{0,2} + u_{2,0})
        elmat(i + 2 * out_dof, j) = 0.5 * dshape(j, 2);
        elmat(i + 2 * out_dof, j + 2 * in_dof) = 0.5 * dshape(j, 0);

        // e_{11} = u_{1,1}
        elmat(i + 3 * out_dof, j + in_dof) = dshape(j, 1);

        // e_{21} = 0.5 *( (u_{1,2} + u_{2,1})
        elmat(i + 4 * out_dof, j + in_dof) = 0.5 * dshape(j, 2);
        elmat(i + 4 * out_dof, j + 2 * in_dof) = 0.5 * dshape(j, 1);

        // e_{22} = u_{2,2}
        elmat(i + 5 * out_dof, j + 2 * in_dof) = dshape(j, 2);
      }
    }
  }
}

void DeviatoricStrainInterpolator::AssembleElementMatrix2(
    const mfem::FiniteElement& in_fe, const mfem::FiniteElement& out_fe,
    mfem::ElementTransformation& Trans, mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dim = in_fe.GetDim();
  auto in_dof = in_fe.GetDof();
  auto out_dof = out_fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  DenseMatrix dshape();
#endif
  dshape.SetSize(in_dof, dim);
  elmat.SetSize(dim * (dim + 1) * out_dof / 2 - 1, dim * in_dof);
  elmat = 0.;

  const auto& nodes = out_fe.GetNodes();
  for (auto i = 0; i < out_dof; i++) {
    const IntegrationPoint& ip = nodes.IntPoint(i);
    Trans.SetIntPoint(&ip);
    in_fe.CalcPhysDShape(Trans, dshape);

    constexpr auto half = static_cast<real_t>(1) / static_cast<real_t>(2);
    constexpr auto third = static_cast<real_t>(1) / static_cast<real_t>(3);
    constexpr auto twoThirds = static_cast<real_t>(2) / static_cast<real_t>(3);
    if (dim == 2) {
      for (auto j = 0; j < in_dof; j++) {
        // d_{00} =   0.5 *(u_{0,0} - u_{1,1})
        elmat(i, j) = half * dshape(j, 0);
        elmat(i, j + in_dof) = -half * dshape(j, 1);

        // d_{10} = 0.5 *(u_{1,0} + u_{0,1})
        elmat(i + out_dof, j) = half * dshape(j, 1);
        elmat(i + out_dof, j + in_dof) = half * dshape(j, 0);
      }
    } else {
      for (auto j = 0; j < in_dof; j++) {
        // d_{00} =   (2/3) * u_{0,0} - (1/3) * (u_{1,1} + u_{2,2})
        elmat(i, j) = twoThirds * dshape(j, 0);
        elmat(i, j + in_dof) = -third * dshape(j, 1);
        elmat(i, j + 2 * in_dof) = -third * dshape(j, 2);

        // d_{10} = 0.5 *(u_{1,0} + u_{0,1})
        elmat(i + out_dof, j) = half * dshape(j, 1);
        elmat(i + out_dof, j + in_dof) = half * dshape(j, 0);

        // d_{20} = 0.5 *(u_{2,0} + u_{0,2})
        elmat(i + 2 * out_dof, j) = half * dshape(j, 2);
        elmat(i + 2 * out_dof, j + 2 * in_dof) = half * dshape(j, 0);

        // d_{11} = (2/3) * u_{1,1} - (1/3) * (u_{0,0} + u_{2,2})
        elmat(i + 3 * out_dof, j) = -third * dshape(j, 0);
        elmat(i + 3 * out_dof, j + in_dof) = twoThirds * dshape(j, 1);
        elmat(i + 3 * out_dof, j + 2 * in_dof) = -third * dshape(j, 2);

        // d_{21} = 0.5 *(u_{2,1} + u_{1,2})
        elmat(i + 4 * out_dof, j + in_dof) = half * dshape(j, 2);
        elmat(i + 4 * out_dof, j + 2 * in_dof) = half * dshape(j, 1);
      }
    }
  }
}

}  // namespace mfemElasticity