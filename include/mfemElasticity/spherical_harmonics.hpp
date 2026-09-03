/**
 * @file spherical_harmonics.hpp
 * @brief Real orthonormal harmonics on a circle or sphere (Fourier modes in
 * 2-D, the real spherical harmonics of Dahlen & Tromp (1998), Appendix B, in
 * 3-D), the synthesis of a field from a coefficient vector, and the analysis
 * of a finite-element field on a spherical boundary into coefficients.
 *
 * The analysis operator is the boundary counterpart of the coefficient
 * matrix inside PoissonDtNOperator, made available for any spherical
 * boundary of any mesh (the surface of the body on its SubMesh, say) and for
 * the radial component of a vector field as well as a scalar. With it, a
 * load or potential set from a coefficient vector and the harmonic
 * coefficients of the solution read off afterwards give Love numbers from
 * one solve per degree (examples/love_numbers.cpp).
 */

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"
#include "mfemElasticity/legendre.hpp"

namespace mfemElasticity {

/**
 * @brief Real orthonormal harmonics up to degree @f$L@f$ on the unit circle
 * (2-D) or sphere (3-D), @f$\int Y_i Y_j\,d\Omega = \delta_{ij}@f$.
 *
 * 3-D: @f$Y_{l0} = X_{l0}@f$, @f$Y_{lm} = \sqrt 2 X_{lm}\cos m\phi@f$ for
 * @f$m > 0@f$ and @f$Y_{l,-m} = \sqrt 2 X_{lm}\sin m\phi@f$, with
 * @f$X_{lm}@f$ the normalised associated Legendre functions (Dahlen & Tromp
 * B.6/B.8); coefficient index @f$i = l^2 + l + m@f$, @f$m = -l, \dots, l@f$,
 * so @f$(L+1)^2@f$ coefficients. Polar angle from the last coordinate axis.
 *
 * 2-D: @f$Y_0 = 1/\sqrt{2\pi}@f$, @f$Y_{k,+k} = \cos k\theta/\sqrt\pi@f$,
 * @f$Y_{k,-k} = \sin k\theta/\sqrt\pi@f$ with @f$\theta@f$ from the
 * @f$x@f$ axis; index @f$2k - 1@f$ for the cosine and @f$2k@f$ for the sine,
 * so @f$2L + 1@f$ coefficients. Only @f$m = \pm l@f$ (and @f$m = 0@f$ for
 * @f$l = 0@f$) exist.
 */
class SurfaceHarmonics : protected LegendreHelper {
 public:
  SurfaceHarmonics(int dim, int max_degree);

  int Dim() const { return dim_; }
  int MaxDegree() const { return lmax_; }

  /** @brief Number of coefficients up to MaxDegree(). */
  int Size() const { return size_; }

  /** @brief Coefficient index of @f$(l, m)@f$ (see the class description). */
  int Index(int l, int m) const;
  int Degree(int i) const { return degree_[i]; }
  int Order(int i) const { return order_[i]; }

  /** @brief All harmonics at the direction of @p x (any nonzero length,
   * relative to the centre), in @p Y (resized to Size()). */
  void Eval(const mfem::Vector& x, mfem::Vector& Y) const;

 private:
  int dim_, lmax_, size_;
  std::vector<int> degree_, order_;
#ifndef MFEM_THREAD_SAFE
  mutable mfem::Vector p_, pm1_, cos_, sin_;
#endif
};

/**
 * @brief @f$f(x) = \sum_i c_i Y_i(\hat x)\,(r/R)^{p_i}@f$ with @f$p_i = 0@f$
 * (a field on the sphere of radius @f$R@f$, extended constantly along rays,
 * for surface loads) or @f$p_i = l_i@f$ (the interior harmonic
 * continuation, for tidal potentials). Position relative to @p centre.
 * Holds a copy of the coefficients; SetCoefficients() replaces them.
 */
class HarmonicExpansionCoefficient : public mfem::Coefficient {
 public:
  HarmonicExpansionCoefficient(const SurfaceHarmonics& basis,
                               const mfem::Vector& coefficients,
                               const mfem::Vector& centre, mfem::real_t radius,
                               bool interior_harmonic = false);

  void SetCoefficients(const mfem::Vector& c);
  const mfem::Vector& Coefficients() const { return c_; }

  mfem::real_t Eval(mfem::ElementTransformation& T,
                    const mfem::IntegrationPoint& ip) override;

 private:
  const SurfaceHarmonics* basis_;
  mfem::Vector c_, x0_;
  mfem::real_t R_;
  bool interior_;
  mfem::Vector x_, Y_;
};

/**
 * @brief Harmonic coefficients of a finite-element field on a spherical
 * boundary of its mesh: @f$c_i = R^{1-d}\int_S f\,Y_i\,dS@f$ for a scalar
 * field, or of the radial component @f$f = u\cdot\hat x@f$ of a vector
 * field, so that @f$f|_S \approx \sum_i c_i Y_i@f$. Serial or parallel
 * (the space's communicator; every boundary element is integrated once).
 *
 * The matrix @f$M_{ji} = R^{1-d}\int_S \varphi_j (\hat x_c) Y_i\,dS@f$ over
 * the (v)dofs of the marked boundary elements is assembled once; analysis
 * is @f$c = M^T f@f$. The transpose direction, LoadVector(), gives the dual
 * vector @f$\int_S (\sum_i c_i Y_i)\,\varphi_j\,dS@f$, i.e. the load vector
 * of the surface field with those coefficients (a BoundaryLFIntegrator of
 * the corresponding HarmonicExpansionCoefficient does the same through the
 * problem classes).
 *
 * The boundary must be a sphere (circle) about @p centre; its radius is
 * measured and checked.
 */
class BoundaryHarmonicCoefficients {
 public:
  enum class Component { Scalar, Radial };

  /**
   * @param fes The field's space (vdim 1 for Scalar, dim for Radial).
   * @param bdr_marker Boundary attributes of the sphere (sized to the mesh's
   * bdr_attributes.Max(); copied).
   * @param max_degree Highest harmonic degree.
   * @param centre Centre of the sphere (empty: the origin).
   * @param radius_tolerance Largest relative spread of the radius over the
   * boundary quadrature points accepted (curved elements interpolate the
   * sphere between their nodes; 1e-3 on a coarse order-2 mesh).
   */
  BoundaryHarmonicCoefficients(mfem::FiniteElementSpace& fes,
                               const mfem::Array<int>& bdr_marker,
                               int max_degree, Component component,
                               const mfem::Vector& centre = mfem::Vector(),
                               mfem::real_t radius_tolerance = 1e-2);

  const SurfaceHarmonics& Basis() const { return basis_; }
  mfem::real_t Radius() const { return R_; }
  const mfem::Vector& Centre() const { return x0_; }
  const mfem::Array<int>& Marker() const { return marker_; }
  int Size() const { return basis_.Size(); }

  /** @brief Coefficients of the (Par)GridFunction @p f on the space. */
  void Coefficients(const mfem::GridFunction& f, mfem::Vector& c) const;

  /** @brief Coefficients of a coefficient evaluated directly at the
   * boundary quadrature points (Scalar component). */
  void Coefficients(mfem::Coefficient& f, mfem::Vector& c) const;

  /** @brief Coefficients of the radial component of @p f (Radial). */
  void Coefficients(mfem::VectorCoefficient& f, mfem::Vector& c) const;

  /** @brief @f$b = R^{d-1} M c@f$: the load vector of the surface field
   * with coefficients @p c, on the local (v)dofs. */
  void LoadVector(const mfem::Vector& c, mfem::Vector& b) const;

  /** @brief A synthesis coefficient on this sphere with the given
   * coefficients (surface field or interior harmonic). */
  std::unique_ptr<HarmonicExpansionCoefficient> Expansion(
      const mfem::Vector& c, bool interior_harmonic = false) const;

  const mfem::SparseMatrix& Matrix() const { return M_; }

 private:
  void MeasureRadius(mfem::real_t tolerance);
  void Assemble();
  /** Loop over the marked boundary elements' quadrature points: visit(T,
   * ip, x (relative), Y, w = weight R^{1-d}). */
  template <class F>
  void ForEachQuadraturePoint(F visit) const;
  void Reduce(mfem::Vector& c) const;

  mfem::FiniteElementSpace* fes_;
  int dim_;
  Component component_;
  mfem::Array<int> marker_;
  mfem::Vector x0_;
  mfem::real_t R_ = 0.0;
  SurfaceHarmonics basis_;
  mfem::SparseMatrix M_;
#ifdef MFEM_USE_MPI
  MPI_Comm comm_ = MPI_COMM_NULL;
  bool parallel_ = false;
#endif
};

}  // namespace mfemElasticity
