#include "mfemElasticity/background.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <utility>

namespace mfemElasticity {

using namespace mfem;

RadialHydrostaticState::RadialHydrostaticState(
    int dim, std::function<real_t(real_t)> rho, real_t G, real_t radius,
    int samples)
    : dim_(dim), rho_(std::move(rho)), G_(G), R_(radius) {
  MFEM_VERIFY(dim == 2 || dim == 3, "dimension must be 2 or 3");
  MFEM_VERIFY(radius > 0.0 && samples > 1, "invalid radius or sample count");
  h_ = R_ / samples;
  const real_t four_pi_G = 4.0 * std::numbers::pi * G_;

  // Cumulative trapezoid for m(r) = int_0^r rho s^{d-1} ds, then
  // g = 4 pi G m / r^{d-1}.
  g_.assign(samples + 1, 0.0);
  real_t m = 0.0;
  auto integrand = [&](real_t r) {
    return rho_(r) * (dim_ == 2 ? r : r * r);
  };
  real_t prev = integrand(0.0);
  for (int i = 1; i <= samples; i++) {
    const real_t r = i * h_;
    const real_t cur = integrand(r);
    m += 0.5 * h_ * (prev + cur);
    prev = cur;
    g_[i] = four_pi_G * m / (dim_ == 2 ? r : r * r);
  }

  // Cumulative trapezoid inward for p(r) = int_r^R rho g ds, p(R) = 0.
  p_.assign(samples + 1, 0.0);
  for (int i = samples - 1; i >= 0; i--) {
    const real_t f0 = rho_(i * h_) * g_[i];
    const real_t f1 = rho_((i + 1) * h_) * g_[i + 1];
    p_[i] = p_[i + 1] + 0.5 * h_ * (f0 + f1);
  }
}

real_t RadialHydrostaticState::Gravity(real_t r) const {
  if (r >= R_) {
    // Exterior field of the total mass.
    const real_t g_R = g_.back();
    return dim_ == 2 ? g_R * (R_ / r) : g_R * (R_ / r) * (R_ / r);
  }
  const real_t s = std::max(real_t{0}, r) / h_;
  const int i =
      std::min(static_cast<int>(s), static_cast<int>(g_.size()) - 2);
  const real_t w = s - i;
  return (1.0 - w) * g_[i] + w * g_[i + 1];
}

real_t RadialHydrostaticState::Pressure(real_t r) const {
  if (r >= R_) {
    return 0.0;
  }
  const real_t s = std::max(real_t{0}, r) / h_;
  const int i =
      std::min(static_cast<int>(s), static_cast<int>(p_.size()) - 2);
  const real_t w = s - i;
  return (1.0 - w) * p_[i] + w * p_[i + 1];
}

real_t RadialHydrostaticState::Density(real_t r) const {
  return r > R_ ? 0.0 : rho_(std::max(real_t{0}, r));
}

RadialHydrostaticBackground::RadialHydrostaticBackground(
    int dim, RadialFunc rho, RadialFunc kappa, RadialFunc mu, real_t G,
    real_t radius, int samples)
    : dim_(dim),
      kappa_fn_(std::move(kappa)),
      mu_fn_(std::move(mu)),
      state_(dim, std::move(rho), G, radius, samples),
      rho_coef_([this](const Vector& x) { return state_.Density(x.Norml2()); }),
      kappa_coef_([this](const Vector& x) { return kappa_fn_(x.Norml2()); }),
      mu_coef_([this](const Vector& x) { return mu_fn_(x.Norml2()); }),
      p0_coef_([this](const Vector& x) { return state_.Pressure(x.Norml2()); }),
      C_eff_(IsotropicElasticTensorCoefficient::FromBulkModulus(
          dim, kappa_coef_, mu_coef_)),
      C_(dim, C_eff_, p0_coef_),
      minus_p0_(-1.0, p0_coef_),
      identity_(dim),
      S_(minus_p0_, identity_),
      phi_e_(dim),
      rheology_(dim, C_, S_, phi_e_) {}

RelabelledBackground::RelabelledBackground(RadialHydrostaticBackground& base,
                                           Diffeomorphism& xi)
    : base_(&base),
      xi_(&xi),
      rho_xi_(xi,
              [this](const Vector& y) { return base_->DensityAt(y.Norml2()); }),
      kappa_xi_(xi,
                [this](const Vector& y) { return base_->KappaAt(y.Norml2()); }),
      mu_xi_(xi, [this](const Vector& y) { return base_->MuAt(y.Norml2()); }),
      p0_xi_(xi,
             [this](const Vector& y) { return base_->PressureAt(y.Norml2()); }),
      C_eff_xi_(IsotropicElasticTensorCoefficient::FromBulkModulus(
          base.SpaceDim(), kappa_xi_, mu_xi_)),
      C_comp_(base.SpaceDim(), C_eff_xi_, p0_xi_),
      C_rel_(base.SpaceDim(), C_comp_, xi),
      S_comp_(base.SpaceDim(), xi,
              [this](const Vector& y, DenseMatrix& S) {
                S.SetSize(y.Size());
                S = 0.0;
                const real_t p = base_->PressureAt(y.Norml2());
                for (int i = 0; i < y.Size(); i++) {
                  S(i, i) = -p;
                }
              }),
      S_rel_(base.SpaceDim(), S_comp_, xi),
      jac_(xi),
      rho_rel_(rho_xi_, jac_),
      rheology_(base.SpaceDim(), C_rel_, S_rel_, xi) {}

GridFunctionDiffeomorphism NewHarmonicExtensionMapping(
    Mesh& parent, int order, Diffeomorphism& xi,
    const Array<int>& body_attributes, const Array<int>& buffer_attributes) {
  const int dim = parent.SpaceDimension();
  MFEM_VERIFY(xi.GetVDim() == dim && order >= 1,
              "NewHarmonicExtensionMapping: dimension mismatch or bad order");

  auto fec = std::make_unique<H1_FECollection>(order, dim);
  auto fes = std::make_unique<FiniteElementSpace>(&parent, fec.get(), dim);
  auto h = std::make_unique<GridFunction>(fes.get());
  *h = 0.0;

  // Body: the nodal interpolant of the displacement xi - id.
  Array<int> battrs(body_attributes);
  SubMesh body(SubMesh::CreateFromDomain(parent, battrs));
  FiniteElementSpace fes_body(&body, fec.get(), dim);
  GridFunction h_body(&fes_body);
  VectorFunctionCoefficient pos(dim, [](const Vector& x, Vector& y) { y = x; });
  VectorSumCoefficient disp(xi, pos, 1.0, -1.0);
  h_body.ProjectCoefficient(disp);
  SubMesh::Transfer(h_body, *h);

  // Buffer: harmonic, Dirichlet on every buffer boundary (the body trace
  // on the shared surface, zero on the outer sphere).
  Array<int> vattrs(buffer_attributes);
  SubMesh buffer(SubMesh::CreateFromDomain(parent, vattrs));
  FiniteElementSpace fes_buffer(&buffer, fec.get(), dim);
  GridFunction h_buffer(&fes_buffer);
  h_buffer = 0.0;
  SubMesh::Transfer(*h, h_buffer);

  Array<int> ess_bdr(buffer.bdr_attributes.Size() ? buffer.bdr_attributes.Max()
                                                  : 0);
  ess_bdr = 1;
  Array<int> ess_tdof;
  fes_buffer.GetEssentialTrueDofs(ess_bdr, ess_tdof);
  MFEM_VERIFY(ess_tdof.Size() > 0,
              "NewHarmonicExtensionMapping: the buffer has no boundary dofs");

  ConstantCoefficient one(1.0);
  BilinearForm a(&fes_buffer);
  a.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
  a.Assemble();
  LinearForm b(&fes_buffer);
  b.Assemble();
  OperatorPtr A;
  Vector X, B;
  a.FormLinearSystem(ess_tdof, h_buffer, b, A, X, B);
  GSSmoother prec(*A.As<SparseMatrix>());
  CGSolver cg;
  cg.SetOperator(*A);
  cg.SetPreconditioner(prec);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(2000);
  cg.SetPrintLevel(0);
  cg.Mult(B, X);
  a.RecoverFEMSolution(X, b, h_buffer);
  SubMesh::Transfer(h_buffer, *h);

  return GridFunctionDiffeomorphism(std::move(fec), std::move(fes),
                                    std::move(h));
}

#ifdef MFEM_USE_MPI
GridFunctionDiffeomorphism NewHarmonicExtensionMapping(
    ParMesh& parent, int order, Diffeomorphism& xi,
    const Array<int>& body_attributes, const Array<int>& buffer_attributes) {
  const int dim = parent.SpaceDimension();
  MFEM_VERIFY(xi.GetVDim() == dim && order >= 1,
              "NewHarmonicExtensionMapping: dimension mismatch or bad order");

  auto fec = std::make_unique<H1_FECollection>(order, dim);
  auto fes = std::make_unique<ParFiniteElementSpace>(&parent, fec.get(), dim);
  auto h = std::make_unique<ParGridFunction>(fes.get());
  *h = 0.0;

  Array<int> battrs(body_attributes);
  ParSubMesh body(ParSubMesh::CreateFromDomain(parent, battrs));
  ParFiniteElementSpace fes_body(&body, fec.get(), dim);
  ParGridFunction h_body(&fes_body);
  VectorFunctionCoefficient pos(dim, [](const Vector& x, Vector& y) { y = x; });
  VectorSumCoefficient disp(xi, pos, 1.0, -1.0);
  h_body.ProjectCoefficient(disp);
  ParSubMesh::Transfer(h_body, *h);

  Array<int> vattrs(buffer_attributes);
  ParSubMesh buffer(ParSubMesh::CreateFromDomain(parent, vattrs));
  ParFiniteElementSpace fes_buffer(&buffer, fec.get(), dim);
  ParGridFunction h_buffer(&fes_buffer);
  h_buffer = 0.0;
  ParSubMesh::Transfer(*h, h_buffer);

  Array<int> ess_bdr(buffer.bdr_attributes.Size() ? buffer.bdr_attributes.Max()
                                                  : 0);
  ess_bdr = 1;
  Array<int> ess_tdof;
  fes_buffer.GetEssentialTrueDofs(ess_bdr, ess_tdof);

  ConstantCoefficient one(1.0);
  ParBilinearForm a(&fes_buffer);
  a.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
  a.Assemble();
  ParLinearForm b(&fes_buffer);
  b.Assemble();
  OperatorPtr A;
  Vector X, B;
  a.FormLinearSystem(ess_tdof, h_buffer, b, A, X, B);
  HypreBoomerAMG prec(*A.As<HypreParMatrix>());
  prec.SetPrintLevel(0);
  prec.SetSystemsOptions(dim);
  CGSolver cg(parent.GetComm());
  cg.SetOperator(*A);
  cg.SetPreconditioner(prec);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(2000);
  cg.SetPrintLevel(0);
  cg.Mult(B, X);
  a.RecoverFEMSolution(X, b, h_buffer);
  ParSubMesh::Transfer(h_buffer, *h);

  return GridFunctionDiffeomorphism(std::move(fec), std::move(fes),
                                    std::move(h));
}
#endif

}  // namespace mfemElasticity
