#include "mfemElasticity/poisson.hpp"

namespace mfemElasticity {

/*****************************************************************
******************************************************************
******************************************************************
*****************************************************************/

int PoissonDtNOperator::CoeffDim() const {
  return fes_->GetMesh()->Dimension() == 2 ? 2 * degree_
                                           : (degree_ + 1) * (degree_ + 1);
}

void PoissonDtNOperator::SetUp() {
  assert(dim_ == 2 || dim_ == 3);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    auto* pmesh = pfes_->GetParMesh();

    SetBoundaryMarker(pmesh);

    auto [comm, has_bdr, root_rank] =
        SplitBoundaryCommunicator(pfes_->GetParMesh(), bdr_marker_);
    bdr_comm_ = comm;
    has_boundary_ = has_bdr;
    bdr_root_rank_ = root_rank;

  } else {
    SetBoundaryMarker(fes_->GetMesh());
  }
#else
  SetBoundaryMarker(fes_->GetMesh());
#endif

#ifndef MFEM_THREAD_SAFE
  c_.SetSize(coeff_dim_);
  x_.SetSize(dim_);
  if (dim_ == 3) {
    sin_.SetSize(degree_ + 1);
    cos_.SetSize(degree_ + 1);
    p_.SetSize(degree_ + 1);
    pm1_.SetSize(degree_ + 1);
  }
#endif
  SetSquareRoots(dim_, degree_);
}

PoissonDtNOperator::PoissonDtNOperator(mfem::FiniteElementSpace* fes,
                                       int degree)
    : mfem::Operator(fes->GetVSize()),
      fes_{fes},
      dim_{fes->GetMesh()->Dimension()},
      degree_{degree},
      coeff_dim_{CoeffDim()},
      mat_(fes->GetVSize(), coeff_dim_) {
  SetUp();
}

#ifdef MFEM_USE_MPI
PoissonDtNOperator::PoissonDtNOperator(MPI_Comm comm,
                                       mfem::ParFiniteElementSpace* fes,
                                       int degree)
    : mfem::Operator(fes->GetVSize()),
      parallel_{true},
      comm_{comm},
      pfes_{fes},
      fes_{fes},
      dim_{fes->GetMesh()->Dimension()},
      degree_{degree},
      coeff_dim_{CoeffDim()},
      mat_(fes->GetVSize(), coeff_dim_) {
  SetUp();
}
#endif

void PoissonDtNOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector c_(coeff_dim_);
#endif

  mat_.MultTranspose(x, c_);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    if (has_boundary_) {
      MPI_Allreduce(MPI_IN_PLACE, c_.GetData(), coeff_dim_, MFEM_MPI_REAL_T,
                    MPI_SUM, bdr_comm_);
    }
  }
#endif

  y.SetSize(x.Size());
  mat_.Mult(c_, y);
}

void PoissonDtNOperator::HarmonicCoefficients(const mfem::Vector& x,
                                              mfem::Vector& y) const {
  using namespace mfem;

  y.SetSize(coeff_dim_);

  mat_.MultTranspose(x, y);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, y.GetData(), coeff_dim_, MFEM_MPI_REAL_T,
                  MPI_SUM, comm_);
  }
#endif

  // Scale the result to form the harmonic coefficients.
  if (dim_ == 2) {
    auto i = 0;
    for (auto k = 1; k <= degree_; k++) {
      const auto fac = 1 / (sqrtPi * sqrt_(k));
      y(i++) *= fac;
      y(i++) *= fac;
    }
  } else {
    auto rfac = 1 / std::sqrt(bdr_radius_);
    y(0) *= rfac;
    auto i = 1;
    for (auto l = 1; l <= degree_; l++) {
      auto fac = rfac / sqrt_[l + 1];
      for (auto m = -l; m <= l; m++) {
        y(i++) *= fac;
      }
    }
  }
}

void PoissonDtNOperator::Assemble() {
  using namespace mfem;
  auto* mesh = fes_->GetMesh();

  auto elmat = DenseMatrix();
  auto vdofs = Array<int>();
  auto rows = Array<int>(coeff_dim_);
  for (auto i = 0; i < coeff_dim_; i++) {
    rows[i] = i;
  }

  for (auto i = 0; i < fes_->GetNBE(); i++) {
    const auto elm_attr = mesh->GetBdrAttribute(i);
    if (bdr_marker_[elm_attr - 1] == 1) {
      fes_->GetBdrElementVDofs(i, vdofs);
      const auto* fe = fes_->GetBE(i);
      auto* Trans = fes_->GetBdrElementTransformation(i);

      if (dim_ == 2) {
        AssembleElementMatrix2D(*fe, *Trans, elmat);
      } else {
        AssembleElementMatrix3D(*fe, *Trans, elmat);
      }

      mat_.AddSubMatrix(vdofs, rows, elmat);
    }
  }

  mat_.Finalize();
}

#ifdef MFEM_USE_MPI
mfem::RAPOperator PoissonDtNOperator::RAP() const {
  auto* P = fes_->GetProlongationMatrix();
  return mfem::RAPOperator(*P, *this, *P);
}
#endif

void PoissonDtNOperator::AssembleElementMatrix2D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;
  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector c_, shape, x_;
  x_.SetSize(2);
  c_.SetSize(coeff_dim_);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, coeff_dim_);
  elmat = 0.0;

  int intorder = fe.GetOrder() + Trans.OrderW();
  auto ir = &IntRules.Get(fe.GetGeomType(), intorder);

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x_);
    x_ -= x0_;

    fe.CalcShape(ip, shape);

    const auto ri = 1 / x_.Norml2();
    const auto sin = x_[1] * ri;
    const auto cos = x_[0] * ri;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    auto i = 0;
    for (auto k = 1; k <= degree_; k++) {
      const auto fac = sqrt_(k);
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      c_(i++) = fac * cos_k;
      c_(i++) = fac * sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    auto w = ri * Trans.Weight() * ip.weight / sqrtPi;
    AddMult_a_VWt(w, shape, c_, elmat);
  }
}

void PoissonDtNOperator::AssembleElementMatrix3D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector c_, shape, x_, sin_, cos_, p_, pm1_;
  x_.SetSize(3);
  c_.SetSize(coeff_dim_);
  sin_.SetSize(degree_ + 1);
  cos_.SetSize(degree_ + 1);
  p_.SetSize(degree_ + 1);
  pm1_.SetSize(degree_ + 1);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, coeff_dim_);
  elmat = 0.0;

  int intorder = fe.GetOrder() + Trans.OrderW();
  auto ir = &IntRules.Get(fe.GetGeomType(), intorder);

  sin_(0) = 0.0;
  cos_(0) = 1.0;

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x_);
    x_ -= x0_;

    const auto ri = 1 / x_.Norml2();
    const auto cos_theta = x_(2) * ri;
    const auto rxy = std::sqrt(x_(0) * x_(0) + x_(1) * x_(1));
    const auto cos = rxy > 0 ? x_(0) / rxy : real_t{1};
    const auto sin = rxy > 0 ? x_(1) / rxy : real_t{0};

    pm1_(0) = 0.0;
    p_(0) = Pll(0, cos_theta);

    auto rfac = std::sqrt(ri) * ri;
    c_(0) = rfac * p_(0);

    auto i = 1;
    for (auto l = 1; l <= degree_; l++) {
      auto fac = rfac * sqrt_[l + 1];

      sin_(l) = rxy > 0 ? sin_(l - 1) * cos + cos_(l - 1) * sin : 0.0;
      cos_(l) = cos_(l - 1) * cos - sin_(l - 1) * sin;

      for (auto m = 0; m < l; m++) {
        const auto [alpha, beta] = RecursionCoefficients(l, m);
        pm1_(m) = alpha * (cos_theta * p_(m) - beta * pm1_(m));
      }
      pm1_(l) = Pll(l, cos_theta);
      p_(l) = 0.0;
      std::swap(p_, pm1_);

      c_(i++) = fac * p_(0);

      fac *= sqrt_[2];
      for (auto m = 1; m <= l; m++) {
        c_(i++) = fac * p_[m] * cos_(m);
        c_(i++) = rxy > 0 ? fac * p_[m] * sin_(m) : 0.0;
      }
    }

    fe.CalcShape(ip, shape);
    auto w = Trans.Weight() * ip.weight;

    AddMult_a_VWt(w, shape, c_, elmat);
  }
}

/*****************************************************************
******************************************************************
******************************************************************
*****************************************************************/

int PoissonMultipoleOperator::CoeffDim() const {
  auto dim = tr_fes_->GetMesh()->Dimension();
  auto vDim = tr_fes_->GetVDim();
  return dim == 2 ? 2 * degree_ + 1 : (degree_ + 1) * (degree_ + 1);
}

void PoissonMultipoleOperator::SetUp() {
  assert(tr_fes_->GetMesh() == te_fes_->GetMesh());
#ifdef MFEM_USE_MPI
  if (parallel_) {
    SetBoundaryMarker(tr_pfes_->GetParMesh());
  } else {
    SetBoundaryMarker(tr_fes_->GetMesh());
  }
#else
  SetBoundaryMarker(tr_fes_->GetMesh());
#endif

#ifndef MFEM_THREAD_SAFE
  c_.SetSize(coeff_dim_);
  x_.SetSize(dim_);
  if (dim_ == 3) {
    sin_.SetSize(degree_ + 1);
    cos_.SetSize(degree_ + 1);
    p_.SetSize(degree_ + 1);
    pm1_.SetSize(degree_ + 1);
  }
#endif
  SetSquareRoots(dim_, degree_);
}

PoissonMultipoleOperator::PoissonMultipoleOperator(
    mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
    int degree, const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      tr_fes_{tr_fes},
      te_fes_{te_fes},
      dim_{tr_fes->GetMesh()->Dimension()},
      degree_{degree},
      coeff_dim_{CoeffDim()},
      dom_marker_{dom_marker},
      lmat_(te_fes->GetVSize(), coeff_dim_),
      rmat_(tr_fes->GetVSize(), coeff_dim_) {
  SetUp();
}

#ifdef MFEM_USE_MPI
PoissonMultipoleOperator::PoissonMultipoleOperator(
    MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
    mfem::ParFiniteElementSpace* te_fes, int degree,
    const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      parallel_{true},
      comm_{comm},
      tr_fes_{tr_fes},
      te_fes_{te_fes},
      tr_pfes_{tr_fes},
      te_pfes_{te_fes},
      dim_{tr_fes->GetMesh()->Dimension()},
      degree_{degree},
      coeff_dim_{CoeffDim()},
      dom_marker_{dom_marker},
      lmat_(te_fes->GetVSize(), coeff_dim_),
      rmat_(tr_fes->GetVSize(), coeff_dim_) {
  SetUp();
}
#endif

void PoissonMultipoleOperator::Mult(const mfem::Vector& x,
                                    mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector c_(coeff_dim_);
#endif

  rmat_.MultTranspose(x, c_);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, c_.GetData(), coeff_dim_, MFEM_MPI_REAL_T,
                  MPI_SUM, comm_);
  }
#endif

  y.SetSize(lmat_.Height());
  lmat_.Mult(c_, y);
}

void PoissonMultipoleOperator::MultTranspose(const mfem::Vector& x,
                                             mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector c_(coeff_dim_);
#endif

  lmat_.MultTranspose(x, c_);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, c_.GetData(), coeff_dim_, MFEM_MPI_REAL_T,
                  MPI_SUM, comm_);
  }
#endif

  y.SetSize(rmat_.Width());
  rmat_.Mult(c_, y);
}

void PoissonMultipoleOperator::Assemble() {
  auto* mesh = tr_fes_->GetMesh();

  auto elmat = mfem::DenseMatrix();
  auto vdofs = mfem::Array<int>();
  auto cdofs = mfem::Array<int>(coeff_dim_);
  for (auto i = 0; i < coeff_dim_; i++) {
    cdofs[i] = i;
  }

  for (auto i = 0; i < te_fes_->GetNBE(); i++) {
    const auto elm_attr = mesh->GetBdrAttribute(i);
    if (bdr_marker_[elm_attr - 1] == 1) {
      te_fes_->GetBdrElementVDofs(i, vdofs);
      const auto* fe = te_fes_->GetBE(i);
      auto* Trans = te_fes_->GetBdrElementTransformation(i);

      if (dim_ == 2) {
        AssembleLeftElementMatrix2D(*fe, *Trans, elmat);
      } else {
        AssembleLeftElementMatrix3D(*fe, *Trans, elmat);
      }

      lmat_.AddSubMatrix(vdofs, cdofs, elmat);
    }
  }

  lmat_.Finalize();

  for (auto i = 0; i < tr_fes_->GetNE(); i++) {
    const auto elm_attr = mesh->GetAttribute(i);
    if (dom_marker_[elm_attr - 1] == 1) {
      tr_fes_->GetElementVDofs(i, vdofs);
      const auto* fe = tr_fes_->GetFE(i);
      auto* Trans = tr_fes_->GetElementTransformation(i);

      if (dim_ == 2) {
        AssembleRightElementMatrix2D(*fe, *Trans, elmat);
      } else {
        AssembleRightElementMatrix3D(*fe, *Trans, elmat);
      }

      rmat_.AddSubMatrix(vdofs, cdofs, elmat);
    }
  }

  rmat_.Finalize();
}

#ifdef MFEM_USE_MPI
mfem::RAPOperator PoissonMultipoleOperator::RAP() const {
  auto* P_te = te_fes_->GetProlongationMatrix();
  auto* P_tr = tr_fes_->GetProlongationMatrix();
  return mfem::RAPOperator(*P_te, *this, *P_tr);
}
#endif

void PoissonMultipoleOperator::AssembleRightElementMatrix2D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector c_, shape, x_;
  c_.SetSize(coeff_dim_);
  x_.SetSize(2);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, coeff_dim_);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  const auto fac = 1 / (2 * pi * bdr_radius_);

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x_);
    x_ -= x0_;

    fe.CalcShape(ip, shape);

    auto radius = x_.Norml2();
    auto inverse_radius = radius > 0 ? 1 / radius : real_t{0};

    auto sin = x_[1] * inverse_radius;
    auto cos = x_[0] * inverse_radius;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    c_(0) = 1.0;

    auto ratio = radius / bdr_radius_;
    auto rfac = real_t{1.0};

    auto i = 1;
    for (auto k = 1; k <= degree_; k++) {
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      rfac *= ratio;
      c_(i++) = rfac * cos_k;
      c_(i++) = rfac * sin_k;

      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    auto w = fac * Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, c_, elmat);
  }
}

void PoissonMultipoleOperator::AssembleLeftElementMatrix2D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector c_, shape, x_;
  c_.SetSize(coeff_dim_);
  x_.SetSize(2);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, coeff_dim_);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x_);
    x_ -= x0_;

    fe.CalcShape(ip, shape);

    auto inverse_radius = 1 / bdr_radius_;

    auto sin = x_[1] * inverse_radius;
    auto cos = x_[0] * inverse_radius;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    c_(0) = 1.;

    auto i = 1;
    for (auto k = 1; k <= degree_; k++) {
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      c_(i++) = cos_k;
      c_(i++) = sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    auto w = Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, c_, elmat);
  }
}

void PoissonMultipoleOperator::AssembleRightElementMatrix3D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector c_, shape, x_, sin_, cos_, p_, pm1_;
  x_.SetSize(3);
  c_.SetSize(coeff_dim_);
  sin_.SetSize(degree_ + 1);
  cos_.SetSize(degree_ + 1);
  p_.SetSize(degree_ + 1);
  pm1_.SetSize(degree_ + 1);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, coeff_dim_);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  sin_(0) = 0.0;
  cos_(0) = 1.0;

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x_);
    x_ -= x0_;

    const auto r = x_.Norml2();
    const auto ri = 1 / r;
    const auto cos_theta = x_(2) * ri;
    const auto rxy = std::sqrt(x_(0) * x_(0) + x_(1) * x_(1));
    const auto cos = rxy > 0 ? x_(0) / rxy : real_t{1};
    const auto sin = rxy > 0 ? x_(1) / rxy : real_t{0};

    pm1_(0) = 0.0;
    p_(0) = Pll(0, cos_theta);

    const auto ratio = r / bdr_radius_;
    auto rfac = 1 / (bdr_radius_ * bdr_radius_);
    c_(0) = rfac * p_(0);

    auto i = 1;
    for (auto l = 1; l <= degree_; l++) {
      rfac *= ratio;
      auto fac = rfac * (l + 1) / (2 * l + 1);

      sin_(l) = rxy > 0 ? sin_(l - 1) * cos + cos_(l - 1) * sin : 0.0;
      cos_(l) = cos_(l - 1) * cos - sin_(l - 1) * sin;

      for (auto m = 0; m < l; m++) {
        const auto [alpha, beta] = RecursionCoefficients(l, m);
        pm1_(m) = alpha * (cos_theta * p_(m) - beta * pm1_(m));
      }
      pm1_(l) = Pll(l, cos_theta);
      p_(l) = 0.0;
      std::swap(p_, pm1_);

      c_(i++) = fac * p_(0);

      fac *= sqrt_[2];
      for (auto m = 1; m <= l; m++) {
        c_(i++) = fac * p_[m] * cos_(m);
        c_(i++) = rxy > 0 ? fac * p_[m] * sin_(m) : 0.0;
      }
    }

    fe.CalcShape(ip, shape);
    auto w = Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, c_, elmat);
  }
}

void PoissonMultipoleOperator::AssembleLeftElementMatrix3D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector c_, shape, x_, sin_, cos_, p_, pm1_;
  x_.SetSize(3);
  c_.SetSize(coeff_dim_);
  sin_.SetSize(degree_ + 1);
  cos_.SetSize(degree_ + 1);
  p_.SetSize(degree_ + 1);
  pm1_.SetSize(degree_ + 1);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, coeff_dim_);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  sin_(0) = 0.0;
  cos_(0) = 1.0;

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x_);
    x_ -= x0_;

    const auto r = x_.Norml2();
    const auto ri = 1 / r;
    const auto cos_theta = x_(2) * ri;
    const auto rxy = std::sqrt(x_(0) * x_(0) + x_(1) * x_(1));
    const auto cos = rxy > 0 ? x_(0) / rxy : real_t{1};
    const auto sin = rxy > 0 ? x_(1) / rxy : real_t{0};

    pm1_(0) = 0.0;
    p_(0) = Pll(0, cos_theta);

    c_(0) = p_(0);

    auto i = 1;
    for (auto l = 1; l <= degree_; l++) {
      sin_(l) = rxy > 0 ? sin_(l - 1) * cos + cos_(l - 1) * sin : 0.0;
      cos_(l) = cos_(l - 1) * cos - sin_(l - 1) * sin;

      for (auto m = 0; m < l; m++) {
        const auto [alpha, beta] = RecursionCoefficients(l, m);
        pm1_(m) = alpha * (cos_theta * p_(m) - beta * pm1_(m));
      }
      pm1_(l) = Pll(l, cos_theta);
      p_(l) = 0.0;
      std::swap(p_, pm1_);

      c_(i++) = p_(0);

      for (auto m = 1; m <= l; m++) {
        c_(i++) = sqrt_[2] * p_[m] * cos_(m);
        c_(i++) = rxy > 0 ? sqrt_[2] * p_[m] * sin_(m) : 0.0;
      }
    }

    fe.CalcShape(ip, shape);
    auto w = Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, c_, elmat);
  }
}

/*****************************************************************
******************************************************************
******************************************************************
*****************************************************************/

int PoissonLinearisedMultipoleOperator::CoeffDim() const {
  auto dim = tr_fes_->GetMesh()->Dimension();
  auto vDim = tr_fes_->GetVDim();
  return dim == 2 ? 2 * degree_ : (degree_ + 1) * (degree_ + 1) - 1;
}

void PoissonLinearisedMultipoleOperator::SetUp() {
  assert(tr_fes_->GetMesh() == te_fes_->GetMesh());
  assert(tr_fes_->GetMesh()->Dimension() == tr_fes_->GetVDim());
#ifdef MFEM_USE_MPI
  if (parallel_) {
    SetBoundaryMarker(tr_pfes_->GetParMesh());
  } else {
    SetBoundaryMarker(tr_fes_->GetMesh());
  }
#else
  SetBoundaryMarker(tr_fes_->GetMesh());
#endif

#ifndef MFEM_THREAD_SAFE
  c0_.SetSize(coeff_dim_);
  c1_.SetSize(coeff_dim_);
  x_.SetSize(dim_);
  if (dim_ == 3) {
    sin_.SetSize(degree_ + 1);
    cos_.SetSize(degree_ + 1);
    p_.SetSize(degree_ + 1);
    pm1_.SetSize(degree_ + 1);
    c2_.SetSize(coeff_dim_);
  }
#endif
  SetSquareRoots(dim_, degree_);
}

PoissonLinearisedMultipoleOperator::PoissonLinearisedMultipoleOperator(
    mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
    mfem::Coefficient& density, int degree, const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      tr_fes_{tr_fes},
      te_fes_{te_fes},
      density_{&density},
      dim_{tr_fes->GetMesh()->Dimension()},
      degree_{degree},
      coeff_dim_{CoeffDim()},
      dom_marker_{dom_marker},
      lmat_(te_fes->GetVSize(), coeff_dim_),
      rmat_(tr_fes->GetVSize(), coeff_dim_) {
  SetUp();
}

PoissonLinearisedMultipoleOperator::PoissonLinearisedMultipoleOperator(
    mfem::FiniteElementSpace* tr_fes, mfem::FiniteElementSpace* te_fes,
    int degree, const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      tr_fes_{tr_fes},
      te_fes_{te_fes},
      dim_{tr_fes->GetMesh()->Dimension()},
      degree_{degree},
      coeff_dim_{CoeffDim()},
      dom_marker_{dom_marker},
      lmat_(te_fes->GetVSize(), coeff_dim_),
      rmat_(tr_fes->GetVSize(), coeff_dim_) {
  SetUp();
}

#ifdef MFEM_USE_MPI

PoissonLinearisedMultipoleOperator::PoissonLinearisedMultipoleOperator(
    MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
    mfem::ParFiniteElementSpace* te_fes, mfem::Coefficient& density, int degree,
    const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      parallel_{true},
      comm_{comm},
      tr_fes_{tr_fes},
      te_fes_{te_fes},
      tr_pfes_{tr_fes},
      te_pfes_{te_fes},
      density_{&density},
      dim_{tr_fes->GetMesh()->Dimension()},
      degree_{degree},
      coeff_dim_{CoeffDim()},
      dom_marker_{dom_marker},
      lmat_(te_fes->GetVSize(), coeff_dim_),
      rmat_(tr_fes->GetVSize(), coeff_dim_) {
  SetUp();
}

PoissonLinearisedMultipoleOperator::PoissonLinearisedMultipoleOperator(
    MPI_Comm comm, mfem::ParFiniteElementSpace* tr_fes,
    mfem::ParFiniteElementSpace* te_fes, int degree,
    const mfem::Array<int>& dom_marker)
    : mfem::Operator(te_fes->GetVSize(), tr_fes->GetVSize()),
      parallel_{true},
      comm_{comm},
      tr_fes_{tr_fes},
      te_fes_{te_fes},
      tr_pfes_{tr_fes},
      te_pfes_{te_fes},
      dim_{tr_fes->GetMesh()->Dimension()},
      degree_{degree},
      coeff_dim_{CoeffDim()},
      dom_marker_{dom_marker},
      lmat_(te_fes->GetVSize(), coeff_dim_),
      rmat_(tr_fes->GetVSize(), coeff_dim_) {
  SetUp();
}
#endif

void PoissonLinearisedMultipoleOperator::Mult(const mfem::Vector& x,
                                              mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector c0_(coeff_dim_);
#endif

  rmat_.MultTranspose(x, c0_);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, c0_.GetData(), coeff_dim_, MFEM_MPI_REAL_T,
                  MPI_SUM, comm_);
  }
#endif

  y.SetSize(lmat_.Height());
  lmat_.Mult(c0_, y);
}

void PoissonLinearisedMultipoleOperator::MultTranspose(const mfem::Vector& x,
                                                       mfem::Vector& y) const {
  using namespace mfem;

#ifdef MFEM_THREAD_SAFE
  Vector c0_(coeff_dim_);
#endif

  lmat_.MultTranspose(x, c0_);

#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, c0_.GetData(), coeff_dim_, MFEM_MPI_REAL_T,
                  MPI_SUM, comm_);
  }
#endif

  y.SetSize(rmat_.Width());
  rmat_.Mult(c0_, y);
}

void PoissonLinearisedMultipoleOperator::Assemble() {
  auto* mesh = tr_fes_->GetMesh();

  auto elmat = mfem::DenseMatrix();
  auto vdofs = mfem::Array<int>();
  auto cdofs = mfem::Array<int>(coeff_dim_);
  for (auto i = 0; i < coeff_dim_; i++) {
    cdofs[i] = i;
  }

  for (auto i = 0; i < te_fes_->GetNBE(); i++) {
    const auto elm_attr = mesh->GetBdrAttribute(i);
    if (bdr_marker_[elm_attr - 1] == 1) {
      te_fes_->GetBdrElementVDofs(i, vdofs);
      const auto* fe = te_fes_->GetBE(i);
      auto* Trans = te_fes_->GetBdrElementTransformation(i);

      if (dim_ == 2) {
        AssembleLeftElementMatrix2D(*fe, *Trans, elmat);
      } else {
        AssembleLeftElementMatrix3D(*fe, *Trans, elmat);
      }

      lmat_.AddSubMatrix(vdofs, cdofs, elmat);
    }
  }

  lmat_.Finalize();

  for (auto i = 0; i < tr_fes_->GetNE(); i++) {
    const auto elm_attr = mesh->GetAttribute(i);
    if (dom_marker_[elm_attr - 1] == 1) {
      tr_fes_->GetElementVDofs(i, vdofs);
      const auto* fe = tr_fes_->GetFE(i);
      auto* Trans = tr_fes_->GetElementTransformation(i);

      if (dim_ == 2) {
        AssembleRightElementMatrix2D(*fe, *Trans, elmat);
      } else {
        AssembleRightElementMatrix3D(*fe, *Trans, elmat);
      }

      rmat_.AddSubMatrix(vdofs, cdofs, elmat);
    }
  }

  rmat_.Finalize();
}

#ifdef MFEM_USE_MPI
mfem::RAPOperator PoissonLinearisedMultipoleOperator::RAP() const {
  auto* P_te = te_fes_->GetProlongationMatrix();
  auto* P_tr = tr_fes_->GetProlongationMatrix();
  return mfem::RAPOperator(*P_te, *this, *P_tr);
}
#endif

void PoissonLinearisedMultipoleOperator::AssembleRightElementMatrix2D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();
  auto dim = Trans.GetSpaceDim();

#ifdef MFEM_THREAD_SAFE
  Vector c0_(coeff_dim_), c1_(coeff_dim_), shape(coeff_dim_), x_(2);
  DenseMatrix part_elmat;
#endif

  shape.SetSize(dof);
  part_elmat.SetSize(dof, coeff_dim_);

  elmat.SetSize(dim * dof, coeff_dim_);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  const auto fac = 1 / (2 * pi * bdr_radius_ * bdr_radius_);

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x_);
    x_ -= x0_;

    fe.CalcShape(ip, shape);

    auto r = x_.Norml2();
    auto ir = r > 0 ? 1 / r : real_t{0};

    auto sin = x_[1] * ir;
    auto cos = x_[0] * ir;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    auto ratio = r / bdr_radius_;
    auto rfac = real_t{1.0};

    auto i = 0;
    for (auto k = 1; k <= degree_; k++) {
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;

      c0_(i) = k * rfac * cos_k;
      c1_(i++) = -k * rfac * sin_k;

      c0_(i) = k * rfac * sin_k;
      c1_(i++) = k * rfac * cos_k;

      rfac *= ratio;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    auto w = fac * Trans.Weight() * ip.weight;

    if (density_) {
      w *= density_->Eval(Trans, ip);
    }

    MultVWt(shape, c0_, part_elmat);
    elmat.AddMatrix(w * cos, part_elmat, 0, 0);
    elmat.AddMatrix(w * sin, part_elmat, dof, 0);

    MultVWt(shape, c1_, part_elmat);
    elmat.AddMatrix(-w * sin, part_elmat, 0, 0);
    elmat.AddMatrix(w * cos, part_elmat, dof, 0);
  }
}

void PoissonLinearisedMultipoleOperator::AssembleLeftElementMatrix2D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector c0_(coeff_dim_), shape(dof), x_(2);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, coeff_dim_);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x_);
    x_ -= x0_;

    fe.CalcShape(ip, shape);

    auto inverse_radius = 1 / bdr_radius_;

    auto sin = x_[1] * inverse_radius;
    auto cos = x_[0] * inverse_radius;

    auto sin_k_m = 0.0;
    auto cos_k_m = 1.0;

    auto i = 0;
    for (auto k = 1; k <= degree_; k++) {
      auto sin_k = sin_k_m * cos + cos_k_m * sin;
      auto cos_k = cos_k_m * cos - sin_k_m * sin;
      c0_(i++) = cos_k;
      c0_(i++) = sin_k;
      sin_k_m = sin_k;
      cos_k_m = cos_k;
    }

    auto w = Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, c0_, elmat);
  }
}

void PoissonLinearisedMultipoleOperator::AssembleRightElementMatrix3D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();
  auto dim = Trans.GetSpaceDim();

#ifdef MFEM_THREAD_SAFE
  Vector c0_, c1_, c2_, shape, x_, sin_, cos_, p_, pm1_;
  x_.SetSize(3);
  c0_.SetSize(coeff_dim_);
  c1_.SetSize(coeff_dim_);
  c2_.SetSize(coeff_dim_);
  sin_.SetSize(degree_ + 1);
  cos_.SetSize(degree_ + 1);
  p_.SetSize(degree_ + 1);
  pm1_.SetSize(degree_ + 1);
  DenseMatrix part_elmat;
#endif

  shape.SetSize(dof);
  part_elmat.SetSize(dof, coeff_dim_);

  elmat.SetSize(dim * dof, coeff_dim_);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  sin_(0) = 0.0;
  cos_(0) = 1.0;

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x_);
    x_ -= x0_;

    const auto r = x_.Norml2();
    const auto ri = 1 / r;
    const auto rxy = std::sqrt(x_(0) * x_(0) + x_(1) * x_(1));
    const auto cos_theta = x_(2) * ri;
    const auto sin_theta = rxy * ri;
    const auto cosec_theta = rxy > 0 ? 1 / sin_theta : 0;
    const auto cos = rxy > 0 ? x_(0) / rxy : real_t{1};
    const auto sin = rxy > 0 ? x_(1) / rxy : real_t{0};

    pm1_(0) = 0.0;
    p_(0) = Pll(0, cos_theta);

    const auto ratio = r / bdr_radius_;
    auto rfac = std::pow(bdr_radius_, -3);

    auto i = 0;
    for (auto l = 1; l <= degree_; l++) {
      auto fac = rfac * (l + 1) / (2 * l + 1);

      sin_(l) = rxy > 0 ? sin_(l - 1) * cos + cos_(l - 1) * sin : 0.0;
      cos_(l) = cos_(l - 1) * cos - sin_(l - 1) * sin;

      for (auto m = 0; m < l; m++) {
        const auto [alpha, beta] = RecursionCoefficients(l, m);
        pm1_(m) = alpha * (cos_theta * p_(m) - beta * pm1_(m));
      }
      pm1_(l) = Pll(l, cos_theta);
      p_(l) = 0.0;
      std::swap(p_, pm1_);

      auto p_th_ = sqrt_[l] * sqrt_[l + 1] * p_(1);
      c0_(i) = fac * l * p_(0);
      c1_(i) = fac * p_th_;
      c2_(i++) = 0.0;

      fac *= sqrt_[2];
      for (auto m = 1; m < l; m++) {
        p_th_ = 0.5 * sqrt_[l - m] * sqrt_[l + m + 1] * p_[m + 1] -
                0.5 * sqrt_[l + m] * sqrt_[l - m + 1] * p_[m - 1];

        c0_(i) = fac * l * p_(m) * cos_(m);
        c1_(i) = fac * p_th_ * cos_(m);
        c2_(i++) = -fac * m * cosec_theta * p_(m) * sin_(m);

        c0_(i) = fac * l * p_(m) * sin_(m);
        c1_(i) = fac * p_th_ * sin_(m);
        c2_(i++) = fac * m * cosec_theta * p_(m) * cos_(m);
      }

      p_th_ = -0.5 * sqrt_[2 * l] * p_(l - 1);
      c0_(i) = fac * l * p_(l) * cos_(l);
      c1_(i) = fac * p_th_ * cos_(l);
      c2_(i++) = -fac * l * cosec_theta * p_(l) * sin_(l);

      c0_(i) = fac * l * p_(l) * sin_(l);
      c1_(i) = fac * p_th_ * sin_(l);
      c2_(i++) = fac * l * cosec_theta * p_(l) * cos_(l);

      rfac *= ratio;
    }

    fe.CalcShape(ip, shape);
    auto w = Trans.Weight() * ip.weight;

    if (density_) {
      w *= density_->Eval(Trans, ip);
    }

    MultVWt(shape, c0_, part_elmat);
    elmat.AddMatrix(w * sin_theta * cos, part_elmat, 0, 0);
    elmat.AddMatrix(w * sin_theta * sin, part_elmat, dof, 0);
    elmat.AddMatrix(w * cos_theta, part_elmat, 2 * dof, 0);

    MultVWt(shape, c1_, part_elmat);
    elmat.AddMatrix(w * cos_theta * cos, part_elmat, 0, 0);
    elmat.AddMatrix(w * cos_theta * sin, part_elmat, dof, 0);
    elmat.AddMatrix(-w * sin_theta, part_elmat, 2 * dof, 0);

    MultVWt(shape, c2_, part_elmat);
    elmat.AddMatrix(-w * sin, part_elmat, 0, 0);
    elmat.AddMatrix(w * cos, part_elmat, dof, 0);
  }
}

void PoissonLinearisedMultipoleOperator::AssembleLeftElementMatrix3D(
    const mfem::FiniteElement& fe, mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat) {
  using namespace mfem;

  auto dof = fe.GetDof();

#ifdef MFEM_THREAD_SAFE
  Vector c0_, shape, x_, sin_, cos_, p_, pm1_;
  x_.SetSize(3);
  c0_.SetSize(coeff_dim_);
  sin_.SetSize(degree_ + 1);
  cos_.SetSize(degree_ + 1);
  p_.SetSize(degree_ + 1);
  pm1_.SetSize(degree_ + 1);
#endif

  shape.SetSize(dof);
  elmat.SetSize(dof, coeff_dim_);
  elmat = 0.0;

  auto intorder = fe.GetOrder() + Trans.OrderW();
  auto* ir = &IntRules.Get(fe.GetGeomType(), intorder);

  sin_(0) = 0.0;
  cos_(0) = 1.0;

  for (auto j = 0; j < ir->GetNPoints(); j++) {
    const auto& ip = ir->IntPoint(j);
    Trans.SetIntPoint(&ip);
    Trans.Transform(ip, x_);
    x_ -= x0_;

    const auto r = x_.Norml2();
    const auto ri = 1 / r;
    const auto cos_theta = x_(2) * ri;
    const auto rxy = std::sqrt(x_(0) * x_(0) + x_(1) * x_(1));
    const auto cos = rxy > 0 ? x_(0) / rxy : real_t{1};
    const auto sin = rxy > 0 ? x_(1) / rxy : real_t{0};

    pm1_(0) = 0.0;
    p_(0) = Pll(0, cos_theta);

    auto i = 0;
    for (auto l = 1; l <= degree_; l++) {
      sin_(l) = rxy > 0 ? sin_(l - 1) * cos + cos_(l - 1) * sin : 0.0;
      cos_(l) = cos_(l - 1) * cos - sin_(l - 1) * sin;

      for (auto m = 0; m < l; m++) {
        const auto [alpha, beta] = RecursionCoefficients(l, m);
        pm1_(m) = alpha * (cos_theta * p_(m) - beta * pm1_(m));
      }
      pm1_(l) = Pll(l, cos_theta);
      p_(l) = 0.0;
      std::swap(p_, pm1_);

      c0_(i++) = p_(0);

      for (auto m = 1; m <= l; m++) {
        c0_(i++) = sqrt_[2] * p_[m] * cos_(m);
        c0_(i++) = rxy > 0 ? sqrt_[2] * p_[m] * sin_(m) : 0.0;
      }
    }

    fe.CalcShape(ip, shape);
    auto w = Trans.Weight() * ip.weight;
    AddMult_a_VWt(w, shape, c0_, elmat);
  }
}

/*****************************************************************
******************************************************************
******************************************************************
*****************************************************************/

const mfem::IntegrationRule& TransformedDiffusionIntegrator::GetRule(
    const mfem::FiniteElement& trial_fe, const mfem::FiniteElement& test_fe,
    const mfem::ElementTransformation& Trans) {
  const auto order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
  return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void TransformedDiffusionIntegrator::AssembleElementMatrix2(
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
  Vector fs, df, x;
  DenseMatrix trial_dshape, test_dshape, xis, F, a, trial_dshape_trans;
#endif
  trial_dshape.SetSize(trial_dof, dim);

  if (same_spaces) {
    test_dshape.Reset(trial_dshape.GetData(), test_dof, dim);
  } else {
    test_dshape.SetSize(test_dof, dim);
  }

  if (Q || QV) {
    F.SetSize(dim, dim);
  }

  if (Q || QV || QM) {
    a.SetSize(dim, dim);
    trial_dshape_trans.SetSize(trial_dof, dim);
  }

  if (Q) {
    // Evaluate radial function at trial nodes.
    x.SetSize(dim);
    df.SetSize(dim);
    fs.SetSize(trial_dof);
    const auto& ir = trial_fe.GetNodes();
    for (auto i = 0; i < ir.GetNPoints(); i++) {
      const auto& ip = ir.IntPoint(i);
      Trans.SetIntPoint(&ip);
      fs(i) = Q->Eval(Trans, ip);
    }
  }

  if (QV) {
    // Evaluate mapping at all trial nodes.
    const auto& ir = trial_fe.GetNodes();
    QV->Eval(xis, Trans, ir);
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
      // Compute F at the integration point from the radial mapping.
      Trans.Transform(ip, x);
      auto f = Q->Eval(Trans, ip);
      trial_dshape.MultTranspose(fs, df);
      for (auto k = 0; k < dim; k++) {
        for (auto j = 0; j < dim; j++) {
          F(j, k) = x(j) * df(k);
        }
        F(k, k) += f;
      }
    }

    if (QV) {
      // Compute F at the integration point from the mapping.
      Mult(xis, trial_dshape, F);
    }

    if (Q || QV) {
      // Form the matrix a = J F^{-1} F^{-T}
      auto J = F.Det();
      F.Invert();
      MultABt(F, F, a);
      a *= J;
    }

    if (QM) {
      // Evaluate a.
      QM->Eval(a, Trans, ip);
    }

    // Form the contribution to the local element matrix.
    if (Q || QV || QM) {
      Mult(trial_dshape, a, trial_dshape_trans);
      AddMult_a_ABt(w, test_dshape, trial_dshape_trans, elmat);
    } else {
      AddMult_a_ABt(w, test_dshape, trial_dshape, elmat);
    }
  }
}

RadialDiffeomorphismCoefficient::RadialDiffeomorphismCoefficient(
    int dim, mfem::Coefficient& Q)
    : mfem::VectorCoefficient(dim), Q_{&Q} {}

void RadialDiffeomorphismCoefficient::Eval(mfem::Vector& V,
                                           mfem::ElementTransformation& T,
                                           const mfem::IntegrationPoint& ip) {
  V.SetSize(vdim);
  T.Transform(ip, V);
  V *= Q_->Eval(T, ip);
}

mfem::real_t TransformedFunctionCoefficient::Eval(
    mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) {
  using namespace mfem;

  real_t data[3];
  Vector y(data, 3);
  xi_->Eval(y, T, ip);
  return f_(y);
}

}  // namespace mfemElasticity
