/**
 * @file elastic_tensor.hpp
 * @brief Anisotropic linear elasticity: the reduced (Mandel) representation
 * of symmetric tensors in the library's component ordering, a family of
 * MatrixCoefficients producing elasticity tensors in that representation
 * (isotropic, transversely isotropic, general Voigt input, rotated frames,
 * deviatoric splits), and the ElasticTensorIntegrator consuming them.
 */

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"

namespace mfemElasticity {

/**
 * @brief The reduced basis for symmetric second-order tensors used
 * throughout the library, and conversions to and from Voigt form.
 *
 * Components are ordered as in SymmetricMatrixIndex: lower triangle,
 * column-major, i.e. (11, 12, 13, 22, 23, 33) in 3-D and (11, 12, 22) in
 * 2-D. The reduced vectors and matrices use the **Mandel** (orthonormal)
 * scaling, @f$\hat\varepsilon_s = a_s \varepsilon_{jk}@f$ with @f$a_s =
 * 1@f$ on the diagonal and @f$\sqrt 2@f$ off it, so that
 * @f$\varepsilon:\sigma = \hat\varepsilon\cdot\hat\sigma@f$ and the
 * elasticity tensor becomes a symmetric @f$n_s \times n_s@f$ matrix
 * @f$\hat C_{st} = a_s a_t C_{(jk)(lm)}@f$ whose eigenvalues are the
 * eigen-stiffnesses and on which rotations act orthogonally.
 *
 * Voigt form (ordering 11, 22, 33, 23, 13, 12 and engineering shear
 * strains, i.e. @f$C_{\mathrm{Voigt}} = C_{(jk)(lm)}@f$ unscaled) is the
 * input convention of VoigtElasticTensorCoefficient only.
 */
struct SymmetricTensorBasis {
  /// Number of reduced components, d(d+1)/2.
  static int Size(int dim) { return dim * (dim + 1) / 2; }

  /// Reduced index of component (j, k) (either order).
  static int Index(int dim, int j, int k) {
    if (j < k) {
      const int t = j;
      j = k;
      k = t;
    }
    return j + k * dim - k * (k + 1) / 2;
  }

  /// The component (j >= k) of reduced index s.
  static void Component(int dim, int s, int& j, int& k);

  /// Mandel scale: 1 on the diagonal, sqrt(2) off it.
  static mfem::real_t Scale(int j, int k);

  /// Voigt index of component (j, k): 11, 22, 33, 23, 13, 12 (2-D: 11, 22,
  /// 12).
  static int VoigtIndex(int dim, int j, int k);

  /// Mandel matrix in library ordering from a Voigt matrix.
  static void FromVoigt(int dim, const mfem::DenseMatrix& Cv,
                        mfem::DenseMatrix& Cm);

  /// Voigt matrix from a Mandel matrix in library ordering.
  static void ToVoigt(int dim, const mfem::DenseMatrix& Cm,
                      mfem::DenseMatrix& Cv);

  /// Mandel matrix from the full tensor C_ijkl stored with index
  /// ((i d + j) d + k) d + l.
  static void Pack(int dim, const mfem::real_t* Cijkl, mfem::DenseMatrix& Cm);

  /// Full tensor (as for Pack) from a Mandel matrix.
  static void Unpack(int dim, const mfem::DenseMatrix& Cm, mfem::real_t* Cijkl);

  /// sigma_jk = C_jklm eps_lm for tensor-component vectors (unscaled, in
  /// library ordering) and a Mandel matrix.
  static void Apply(const mfem::DenseMatrix& Cm, const mfem::Vector& eps,
                    mfem::Vector& sig);

  /// The orthogonal projector onto volumetric tensors, (1 1^T)/d in Mandel
  /// form.
  static void VolumetricProjector(int dim, mfem::DenseMatrix& P);

  /// The orthogonal projector onto deviatoric tensors, I - (1 1^T)/d.
  static void DeviatoricProjector(int dim, mfem::DenseMatrix& P);

  /// Mandel matrix of the rotation eps -> R eps R^T, Q such that
  /// eps'^ = Q eps^ (orthogonal).
  static void RotationMatrix(int dim, const mfem::DenseMatrix& R,
                             mfem::DenseMatrix& Q);
};

/**
 * @brief A MatrixCoefficient whose Eval() returns an elasticity tensor as an
 * n_s x n_s Mandel matrix in SymmetricTensorBasis ordering.
 */
class ElasticTensorCoefficient : public mfem::MatrixCoefficient {
 public:
  explicit ElasticTensorCoefficient(int dim)
      : mfem::MatrixCoefficient(SymmetricTensorBasis::Size(dim)), dim_(dim) {}

  int SpaceDim() const { return dim_; }

 protected:
  int dim_;
};

/**
 * @brief Isotropic tensor @f$\hat C = \lambda \hat 1 \hat 1^T + 2\mu I@f$.
 */
class IsotropicElasticTensorCoefficient : public ElasticTensorCoefficient {
 public:
  IsotropicElasticTensorCoefficient(int dim, mfem::Coefficient& lambda,
                                    mfem::Coefficient& mu);

  /// From the bulk modulus: lambda = kappa - 2 mu / d.
  static IsotropicElasticTensorCoefficient FromBulkModulus(
      int dim, mfem::Coefficient& kappa, mfem::Coefficient& mu);

  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  mfem::Coefficient* lambda_or_kappa_;
  mfem::Coefficient* mu_;
  bool use_bulk_ = false;
};

/**
 * @brief Transversely isotropic tensor from Love's constants (A, C, F, L, N;
 * Dziewonski & Anderson 1981) and a symmetry-axis field.
 *
 * @f[
 *  C_{ijkl} = (A - 2N)\delta_{ij}\delta_{kl}
 *   + N(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})
 *   + (F - A + 2N)(\delta_{ij} n_k n_l + n_i n_j \delta_{kl})
 *   + (L - N)(\delta_{ik} n_j n_l + \delta_{il} n_j n_k
 *             + \delta_{jk} n_i n_l + \delta_{jl} n_i n_k)
 *   + (A + C - 2F - 4L)\, n_i n_j n_k n_l
 * @f]
 * With @f$n = e_3@f$: @f$C_{11} = A, C_{33} = C, C_{13} = F, C_{44} = L,
 * C_{66} = N, C_{12} = A - 2N@f$ (Voigt). The axis is normalised inside
 * Eval(). In 2-D with an in-plane axis the formula yields the plane-strain
 * restriction of the 3-D tensor; plane stress is not supported.
 *
 * Moduli and axis are non-owning unless created by FromVelocities().
 */
class TransverselyIsotropicElasticTensorCoefficient
    : public ElasticTensorCoefficient {
 public:
  TransverselyIsotropicElasticTensorCoefficient(int dim, mfem::Coefficient& A,
                                                mfem::Coefficient& C,
                                                mfem::Coefficient& F,
                                                mfem::Coefficient& L,
                                                mfem::Coefficient& N,
                                                mfem::VectorCoefficient& axis);

  /// PREM-style input: A = rho vph^2, C = rho vpv^2, N = rho vsh^2,
  /// L = rho vsv^2, F = eta (A - 2L). The derived coefficients are owned.
  static TransverselyIsotropicElasticTensorCoefficient FromVelocities(
      int dim, mfem::Coefficient& rho, mfem::Coefficient& vpv,
      mfem::Coefficient& vph, mfem::Coefficient& vsv, mfem::Coefficient& vsh,
      mfem::Coefficient& eta, mfem::VectorCoefficient& axis);

  TransverselyIsotropicElasticTensorCoefficient(
      TransverselyIsotropicElasticTensorCoefficient&&) = default;

  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

  /// The Mandel matrix for given moduli and unit axis (no coefficients).
  static void Build(int dim, mfem::real_t A, mfem::real_t C, mfem::real_t F,
                    mfem::real_t L, mfem::real_t N, const mfem::Vector& n,
                    mfem::DenseMatrix& Cm);

 private:
  mfem::Coefficient *A_, *C_, *F_, *L_, *N_;
  mfem::VectorCoefficient* axis_;
  std::vector<std::unique_ptr<mfem::Coefficient>> owned_;
  mfem::Vector n_;
  std::vector<mfem::real_t> full_;
};

/**
 * @brief General anisotropy from any MatrixCoefficient in Voigt convention
 * and ordering (6 x 6 in 3-D, 3 x 3 in 2-D).
 */
class VoigtElasticTensorCoefficient : public ElasticTensorCoefficient {
 public:
  VoigtElasticTensorCoefficient(int dim, mfem::MatrixCoefficient& C_voigt);

  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  mfem::MatrixCoefficient* Cv_;
  mfem::DenseMatrix tmp_;
};

/**
 * @brief A tensor given in a local material frame, rotated by a rotation
 * field: C'_ijkl = R_ia R_jb R_kc R_ld C_abcd.
 */
class RotatedElasticTensorCoefficient : public ElasticTensorCoefficient {
 public:
  /// @param C_local n_s x n_s Mandel tensor in the material frame.
  /// @param R d x d rotation field (columns = material axes in space).
  RotatedElasticTensorCoefficient(mfem::MatrixCoefficient& C_local,
                                  mfem::MatrixCoefficient& R);

  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  mfem::MatrixCoefficient* C_;
  mfem::MatrixCoefficient* R_;
  mfem::DenseMatrix Cl_, Rm_, Q_, tmp_;
};

/**
 * @brief The deviatoric part @f$P_{dev} C P_{dev}@f$ of a tensor, or its
 * complement @f$C - P_{dev} C P_{dev}@f$ (the relaxation split of the
 * viscoelastic layer). For an isotropic C the two parts are exactly
 * @f$2\mu@f$ dev-dev and @f$\kappa@f$ div-div.
 */
class DeviatoricProjectionElasticTensorCoefficient
    : public ElasticTensorCoefficient {
 public:
  /// @param deviatoric_part true: P C P; false: C - P C P.
  DeviatoricProjectionElasticTensorCoefficient(int dim,
                                               mfem::MatrixCoefficient& C,
                                               bool deviatoric_part = true);

  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  mfem::MatrixCoefficient* C_;
  bool deviatoric_part_;
  mfem::DenseMatrix P_, Cq_, tmp_;
};

/**
 * @brief The unit radial vector (x - x0)/|x - x0|; e_d at x = x0.
 */
class RadialUnitVectorCoefficient : public mfem::VectorCoefficient {
 public:
  explicit RadialUnitVectorCoefficient(int dim);
  RadialUnitVectorCoefficient(int dim, const mfem::Vector& x0);

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  mfem::Vector x0_, x_;
};

/**
 * @brief @f$(u, v) \mapsto \int_\Omega \varepsilon(v) : C : \varepsilon(u)@f$
 * for an elasticity tensor supplied as an n_s x n_s MatrixCoefficient in
 * Mandel form and SymmetricTensorBasis ordering (as produced by the
 * ElasticTensorCoefficient classes; sums and scalar products of them through
 * MFEM's matrix-coefficient algebra are fine too).
 *
 * Vector H1 space with Ordering::byNODES, element matrix layout
 * elmat(dof c + i, dof c' + i') and default quadrature order
 * 2 OrderGrad(el), as for mfem::ElasticityIntegrator. Per quadrature point
 * the reduced strain-displacement matrix B (eps^ = B u) is formed and
 * w B^T C B added.
 */
class ElasticTensorIntegrator : public mfem::BilinearFormIntegrator {
 public:
  explicit ElasticTensorIntegrator(mfem::MatrixCoefficient& C,
                                   const mfem::IntegrationRule* ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), C_(&C) {}

  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             mfem::ElementTransformation& Trans,
                             mfem::DenseMatrix& elmat) override;

  /// Build B (n_s x d dof) from the physical gradients gshape (dof x d).
  static void StrainDisplacementMatrix(int dim, const mfem::DenseMatrix& gshape,
                                       mfem::DenseMatrix& B);

 private:
  mfem::MatrixCoefficient* C_;
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix dshape_, gshape_, B_, Cq_, CB_;
#endif
};

}  // namespace mfemElasticity
