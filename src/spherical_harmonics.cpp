/**
 * @file spherical_harmonics.cpp
 * @brief Implementation of spherical_harmonics.hpp.
 */

#include "mfemElasticity/spherical_harmonics.hpp"

#include <cmath>

namespace mfemElasticity {

using namespace mfem;

namespace {
constexpr real_t kPi = 3.141592653589793238462643383279502884;
}

// ---------------------------------------------------------------------------
// SurfaceHarmonics

SurfaceHarmonics::SurfaceHarmonics(int dim, int max_degree)
    : dim_(dim), lmax_(max_degree) {
  MFEM_VERIFY(dim == 2 || dim == 3, "SurfaceHarmonics: dim must be 2 or 3.");
  MFEM_VERIFY(max_degree >= 0, "SurfaceHarmonics: negative degree.");
  size_ = dim == 2 ? 2 * lmax_ + 1 : (lmax_ + 1) * (lmax_ + 1);
  degree_.resize(size_);
  order_.resize(size_);
  if (dim == 2) {
    degree_[0] = 0;
    order_[0] = 0;
    for (int k = 1; k <= lmax_; k++) {
      degree_[2 * k - 1] = k;
      order_[2 * k - 1] = k;
      degree_[2 * k] = k;
      order_[2 * k] = -k;
    }
  } else {
    for (int l = 0; l <= lmax_; l++) {
      for (int m = -l; m <= l; m++) {
        degree_[l * l + l + m] = l;
        order_[l * l + l + m] = m;
      }
    }
    SetSquareRoots(3, lmax_);
#ifndef MFEM_THREAD_SAFE
    p_.SetSize(lmax_ + 1);
    pm1_.SetSize(lmax_ + 1);
    cos_.SetSize(lmax_ + 1);
    sin_.SetSize(lmax_ + 1);
#endif
  }
}

int SurfaceHarmonics::Index(int l, int m) const {
  MFEM_VERIFY(l >= 0 && l <= lmax_ && std::abs(m) <= l,
              "SurfaceHarmonics::Index: (l, m) out of range.");
  if (dim_ == 2) {
    MFEM_VERIFY(l == 0 || std::abs(m) == l,
                "SurfaceHarmonics::Index: in 2-D m = +l (cos) or -l (sin).");
    return l == 0 ? 0 : (m > 0 ? 2 * l - 1 : 2 * l);
  }
  return l * l + l + m;
}

void SurfaceHarmonics::Eval(const Vector& x, Vector& Y) const {
  Y.SetSize(size_);
  if (dim_ == 2) {
    const real_t r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    const real_t c = r > 0 ? x[0] / r : 1.0, s = r > 0 ? x[1] / r : 0.0;
    Y[0] = 1.0 / std::sqrt(2.0 * kPi);
    const real_t fac = 1.0 / std::sqrt(kPi);
    real_t ck = 1.0, sk = 0.0;  // cos k theta, sin k theta
    for (int k = 1; k <= lmax_; k++) {
      const real_t ck1 = ck * c - sk * s, sk1 = sk * c + ck * s;
      ck = ck1;
      sk = sk1;
      Y[2 * k - 1] = fac * ck;
      Y[2 * k] = fac * sk;
    }
    return;
  }

#ifdef MFEM_THREAD_SAFE
  Vector p_(lmax_ + 1), pm1_(lmax_ + 1), cos_(lmax_ + 1), sin_(lmax_ + 1);
#endif
  const real_t r = x.Norml2();
  MFEM_ASSERT(r > 0.0, "SurfaceHarmonics::Eval: zero vector.");
  const real_t cos_theta = x[2] / r;
  const real_t rxy = std::sqrt(x[0] * x[0] + x[1] * x[1]);
  const real_t c = rxy > 0 ? x[0] / rxy : 1.0;
  const real_t s = rxy > 0 ? x[1] / rxy : 0.0;

  cos_[0] = 1.0;
  sin_[0] = 0.0;
  pm1_ = 0.0;
  p_ = 0.0;
  p_[0] = Pll(0, cos_theta);
  Y[0] = p_[0];
  const real_t sqrt2 = std::sqrt(2.0);
  for (int l = 1; l <= lmax_; l++) {
    cos_[l] = cos_[l - 1] * c - sin_[l - 1] * s;
    sin_[l] = sin_[l - 1] * c + cos_[l - 1] * s;
    // X_{l,m} from X_{l-1,m} (p_) and X_{l-2,m} (pm1_), then X_{l,l}.
    for (int m = 0; m < l; m++) {
      const auto [alpha, beta] = RecursionCoefficients(l, m);
      pm1_[m] = alpha * (cos_theta * p_[m] - beta * pm1_[m]);
    }
    pm1_[l] = Pll(l, cos_theta);
    p_[l] = 0.0;
    std::swap(p_, pm1_);
    const int base = l * l + l;
    Y[base] = p_[0];
    for (int m = 1; m <= l; m++) {
      Y[base + m] = sqrt2 * p_[m] * cos_[m];
      Y[base - m] = sqrt2 * p_[m] * sin_[m];
    }
  }
}

// ---------------------------------------------------------------------------
// HarmonicExpansionCoefficient

HarmonicExpansionCoefficient::HarmonicExpansionCoefficient(
    const SurfaceHarmonics& basis, const Vector& coefficients,
    const Vector& centre, real_t radius, bool interior_harmonic)
    : basis_(&basis),
      c_(coefficients),
      x0_(centre),
      R_(radius),
      interior_(interior_harmonic) {
  MFEM_VERIFY(c_.Size() == basis_->Size(),
              "HarmonicExpansionCoefficient: coefficient vector size.");
  if (x0_.Size() == 0) {
    x0_.SetSize(basis_->Dim());
    x0_ = 0.0;
  }
  MFEM_VERIFY(x0_.Size() == basis_->Dim(),
              "HarmonicExpansionCoefficient: centre dimension.");
  MFEM_VERIFY(R_ > 0.0, "HarmonicExpansionCoefficient: radius.");
  x_.SetSize(basis_->Dim());
}

void HarmonicExpansionCoefficient::SetCoefficients(const Vector& c) {
  MFEM_VERIFY(c.Size() == basis_->Size(),
              "HarmonicExpansionCoefficient: coefficient vector size.");
  c_ = c;
}

real_t HarmonicExpansionCoefficient::Eval(ElementTransformation& T,
                                          const IntegrationPoint& ip) {
  T.Transform(ip, x_);
  x_ -= x0_;
  const real_t r = x_.Norml2();
  if (!(r > 0.0)) {
    // Only the constant survives at the centre (interior); a surface
    // field is undefined there, take the constant too.
    return c_[0] * (basis_->Dim() == 2 ? 1.0 / std::sqrt(2.0 * kPi)
                                       : 1.0 / std::sqrt(4.0 * kPi));
  }
  basis_->Eval(x_, Y_);
  real_t f = 0.0;
  if (interior_) {
    const real_t q = r / R_;
    real_t ql = 1.0;
    int l_prev = 0;
    for (int i = 0; i < c_.Size(); i++) {
      const int l = basis_->Degree(i);
      while (l_prev < l) {
        ql *= q;
        l_prev++;
      }
      f += c_[i] * Y_[i] * ql;
    }
  } else {
    f = c_ * Y_;
  }
  return f;
}

// ---------------------------------------------------------------------------
// BoundaryHarmonicCoefficients

BoundaryHarmonicCoefficients::BoundaryHarmonicCoefficients(
    FiniteElementSpace& fes, const Array<int>& bdr_marker, int max_degree,
    Component component, const Vector& centre, real_t radius_tolerance)
    : fes_(&fes),
      dim_(fes.GetMesh()->Dimension()),
      component_(component),
      marker_(bdr_marker),
      x0_(centre),
      basis_(fes.GetMesh()->Dimension(), max_degree) {
  MFEM_VERIFY(marker_.Size() == fes_->GetMesh()->bdr_attributes.Max(),
              "BoundaryHarmonicCoefficients: the marker must be sized to "
              "the mesh's bdr_attributes.Max().");
  if (component_ == Component::Scalar) {
    MFEM_VERIFY(fes_->GetVDim() == 1,
                "BoundaryHarmonicCoefficients: Scalar needs vdim 1.");
  } else {
    MFEM_VERIFY(fes_->GetVDim() == dim_,
                "BoundaryHarmonicCoefficients: Radial needs vdim = dim.");
  }
  if (x0_.Size() == 0) {
    x0_.SetSize(dim_);
    x0_ = 0.0;
  }
  MFEM_VERIFY(x0_.Size() == dim_, "BoundaryHarmonicCoefficients: centre.");
#ifdef MFEM_USE_MPI
  if (auto* pfes = dynamic_cast<ParFiniteElementSpace*>(fes_)) {
    parallel_ = true;
    comm_ = pfes->GetComm();
  }
#endif
  MeasureRadius(radius_tolerance);
  Assemble();
}

template <class F>
void BoundaryHarmonicCoefficients::ForEachQuadraturePoint(F visit) const {
  Mesh* mesh = fes_->GetMesh();
  Vector x(dim_), Y;
  const real_t scale = std::pow(R_, 1 - dim_);
  for (int b = 0; b < fes_->GetNBE(); b++) {
    if (!marker_[mesh->GetBdrAttribute(b) - 1]) {
      continue;
    }
    const FiniteElement* fe = fes_->GetBE(b);
    ElementTransformation* T = fes_->GetBdrElementTransformation(b);
    const int order = 2 * fe->GetOrder() + T->OrderW() + basis_.MaxDegree();
    const IntegrationRule& ir = IntRules.Get(fe->GetGeomType(), order);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      x -= x0_;
      basis_.Eval(x, Y);
      visit(b, *fe, *T, ip, x, Y, ip.weight * T->Weight() * scale);
    }
  }
}

void BoundaryHarmonicCoefficients::MeasureRadius(real_t tolerance) {
  // Area-weighted mean radius and the spread about it.
  Mesh* mesh = fes_->GetMesh();
  Vector x(dim_);
  real_t sums[2] = {0.0, 0.0};
  real_t rmin = INFINITY, rmax = 0.0;
  for (int b = 0; b < fes_->GetNBE(); b++) {
    if (!marker_[mesh->GetBdrAttribute(b) - 1]) {
      continue;
    }
    ElementTransformation* T = fes_->GetBdrElementTransformation(b);
    const IntegrationRule& ir =
        IntRules.Get(T->GetGeometryType(), 2 * T->OrderW() + 2);
    for (int q = 0; q < ir.GetNPoints(); q++) {
      const IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, x);
      x -= x0_;
      const real_t r = x.Norml2(), w = ip.weight * T->Weight();
      sums[0] += w;
      sums[1] += w * r;
      rmin = std::min(rmin, r);
      rmax = std::max(rmax, r);
    }
  }
#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPITypeMap<real_t>::mpi_type,
                  MPI_SUM, comm_);
    MPI_Allreduce(MPI_IN_PLACE, &rmin, 1, MPITypeMap<real_t>::mpi_type,
                  MPI_MIN, comm_);
    MPI_Allreduce(MPI_IN_PLACE, &rmax, 1, MPITypeMap<real_t>::mpi_type,
                  MPI_MAX, comm_);
  }
#endif
  MFEM_VERIFY(sums[0] > 0.0,
              "BoundaryHarmonicCoefficients: the marked boundary is empty.");
  R_ = sums[1] / sums[0];
  MFEM_VERIFY(rmax - rmin <= tolerance * R_,
              "BoundaryHarmonicCoefficients: the marked boundary is not a "
              "sphere about the centre (radii from "
                  << rmin << " to " << rmax << ").");
}

void BoundaryHarmonicCoefficients::Assemble() {
  const int n = basis_.Size();
  M_ = SparseMatrix(fes_->GetVSize(), n);
  Array<int> vdofs, cols(n);
  for (int i = 0; i < n; i++) {
    cols[i] = i;
  }
  Vector shape;
  DenseMatrix elmat;
  int current = -1;
  auto flush = [&]() {
    if (current >= 0) {
      M_.AddSubMatrix(vdofs, cols, elmat);
    }
  };
  ForEachQuadraturePoint([&](int b, const FiniteElement& fe,
                             ElementTransformation& T,
                             const IntegrationPoint& ip, const Vector& x,
                             const Vector& Y, real_t w) {
    if (b != current) {
      flush();
      current = b;
      fes_->GetBdrElementVDofs(b, vdofs);
      elmat.SetSize(vdofs.Size(), n);
      elmat = 0.0;
      shape.SetSize(fe.GetDof());
    }
    fe.CalcShape(ip, shape);
    const int dof = fe.GetDof();
    if (component_ == Component::Scalar) {
      for (int j = 0; j < dof; j++) {
        for (int i = 0; i < n; i++) {
          elmat(j, i) += w * shape[j] * Y[i];
        }
      }
    } else {
      const real_t r = x.Norml2();
      for (int c = 0; c < dim_; c++) {
        const real_t nc = x[c] / r;
        for (int j = 0; j < dof; j++) {
          for (int i = 0; i < n; i++) {
            elmat(c * dof + j, i) += w * nc * shape[j] * Y[i];
          }
        }
      }
    }
  });
  flush();
  M_.Finalize();
}

void BoundaryHarmonicCoefficients::Reduce(Vector& c) const {
#ifdef MFEM_USE_MPI
  if (parallel_) {
    MPI_Allreduce(MPI_IN_PLACE, c.GetData(), c.Size(),
                  MPITypeMap<real_t>::mpi_type, MPI_SUM, comm_);
  }
#endif
}

void BoundaryHarmonicCoefficients::Coefficients(const GridFunction& f,
                                                Vector& c) const {
  MFEM_VERIFY(f.Size() == M_.Height(),
              "BoundaryHarmonicCoefficients: the field is not on the space.");
  c.SetSize(Size());
  M_.MultTranspose(f, c);
  Reduce(c);
}

void BoundaryHarmonicCoefficients::Coefficients(Coefficient& f,
                                                Vector& c) const {
  MFEM_VERIFY(component_ == Component::Scalar,
              "BoundaryHarmonicCoefficients: scalar coefficient on a Radial "
              "operator.");
  c.SetSize(Size());
  c = 0.0;
  ForEachQuadraturePoint([&](int, const FiniteElement&,
                             ElementTransformation& T,
                             const IntegrationPoint& ip, const Vector&,
                             const Vector& Y, real_t w) {
    c.Add(w * f.Eval(T, ip), Y);
  });
  Reduce(c);
}

void BoundaryHarmonicCoefficients::Coefficients(VectorCoefficient& f,
                                                Vector& c) const {
  MFEM_VERIFY(component_ == Component::Radial,
              "BoundaryHarmonicCoefficients: vector coefficient on a Scalar "
              "operator.");
  c.SetSize(Size());
  c = 0.0;
  Vector v(dim_);
  ForEachQuadraturePoint([&](int, const FiniteElement&,
                             ElementTransformation& T,
                             const IntegrationPoint& ip, const Vector& x,
                             const Vector& Y, real_t w) {
    f.Eval(v, T, ip);
    c.Add(w * (v * x) / x.Norml2(), Y);
  });
  Reduce(c);
}

void BoundaryHarmonicCoefficients::LoadVector(const Vector& c,
                                              Vector& b) const {
  MFEM_VERIFY(c.Size() == Size(), "BoundaryHarmonicCoefficients: size.");
  b.SetSize(M_.Height());
  M_.Mult(c, b);
  b *= std::pow(R_, dim_ - 1);
}

std::unique_ptr<HarmonicExpansionCoefficient>
BoundaryHarmonicCoefficients::Expansion(const Vector& c,
                                        bool interior_harmonic) const {
  return std::make_unique<HarmonicExpansionCoefficient>(basis_, c, x0_, R_,
                                                        interior_harmonic);
}

}  // namespace mfemElasticity
