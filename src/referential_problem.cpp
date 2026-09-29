/**
 * @file referential_problem.cpp
 * @brief Implementation of ReferentialElasticRheology and
 * LinearQuasiStaticReferentialProblem.
 */

#include "mfemElasticity/referential_problem.hpp"

#include <cmath>
#include <numbers>

#include "mfemElasticity/detail/fem_factory.hpp"
#include "mfemElasticity/mesh.hpp"

namespace mfemElasticity {

using namespace mfem;

namespace {
constexpr real_t kPi = std::numbers::pi_v<real_t>;

/// Stiffness of the referential rheology: the total-Lagrangian split,
/// fixed integrators, weights no-ops.
class ReferentialStiffness : public ElasticStiffness {
 public:
  ReferentialStiffness(MatrixCoefficient& C, MatrixCoefficient& S,
                       Diffeomorphism& map)
      : C_(&C), S_(&S), map_(&map) {}

  void AddIntegrators(BilinearForm& form, Array<int>* marker) override {
    auto add = [&](BilinearFormIntegrator* i) {
      if (marker) {
        form.AddDomainIntegrator(i, *marker);
      } else {
        form.AddDomainIntegrator(i);
      }
    };
    add(new MaterialStiffnessIntegrator(*C_, *map_));
    add(new GeometricStiffnessIntegrator(*S_));
  }

  void SetRelaxationWeights(const std::vector<Coefficient*>&) override {}
  void ClearRelaxationWeights() override {}
  bool IsRelaxed() const override { return false; }

 private:
  MatrixCoefficient* C_;
  MatrixCoefficient* S_;
  Diffeomorphism* map_;
};

/// The rigid rotation of the mapped positions, W phi_e(x), about axis
/// `component` (2-D: the single in-plane rotation).
class MappedRotation : public VectorCoefficient {
 public:
  MappedRotation(Diffeomorphism& map, int component)
      : VectorCoefficient(map.GetVDim()), map_(&map), c_(component) {}

  void Eval(Vector& V, ElementTransformation& T,
            const IntegrationPoint& ip) override {
    map_->Eval(y_, T, ip);
    V.SetSize(vdim);
    if (vdim == 2) {
      V(0) = -y_(1);
      V(1) = y_(0);
    } else {
      // V = e_c x y: V_a = -y_b, V_b = y_a with (c, a, b) cyclic.
      const int a = (c_ + 1) % 3, b = (c_ + 2) % 3;
      V(c_) = 0.0;
      V(a) = -y_(b);
      V(b) = y_(a);
    }
  }

 private:
  Diffeomorphism* map_;
  int c_;
  Vector y_;
};

}  // namespace


std::unique_ptr<mfem::SparseMatrix> NewRadialVacuumExtension(
    FiniteElementSpace& body_fes, FiniteElementSpace& buffer_fes,
    real_t r_body, real_t r_outer, real_t taper_power, real_t pullback) {
  MFEM_VERIFY(body_fes.FEColl() == buffer_fes.FEColl() &&
                  body_fes.GetVDim() == buffer_fes.GetVDim(),
              "NewRadialVacuumExtension: the spaces must share a "
              "collection and vdim.");
  Mesh* body = body_fes.GetMesh();
  Mesh* buffer = buffer_fes.GetMesh();
  const int dim = body->Dimension();

  // The shared parent and the trace pairing.
  auto* body_sub = dynamic_cast<SubMesh*>(body);
  auto* buffer_sub = dynamic_cast<SubMesh*>(buffer);
  MFEM_VERIFY(body_sub && buffer_sub &&
                  body_sub->GetParent() == buffer_sub->GetParent(),
              "NewRadialVacuumExtension: both spaces must live on SubMeshes "
              "of one parent.");
  FiniteElementSpace parent_fes(
      const_cast<Mesh*>(static_cast<const Mesh*>(body_sub->GetParent())),
      const_cast<FiniteElementCollection*>(body_fes.FEColl()),
      body_fes.GetVDim(), body_fes.GetOrdering());
  SubMeshDofInjection inj_body(body_fes, parent_fes);
  SubMeshDofInjection inj_buffer(buffer_fes, parent_fes);
  auto J = NewSubMeshPairingMatrix(inj_buffer, inj_body);  // buffer x body

  // Scalar nodal coordinates of the buffer space.
  const int ns_buf = buffer_fes.GetNDofs();
  DenseMatrix coords(dim, ns_buf);
  {
    Array<int> dofs;
    Vector x(dim);
    for (int e = 0; e < buffer->GetNE(); e++) {
      const auto* fe = buffer_fes.GetFE(e);
      auto* T = buffer->GetElementTransformation(e);
      buffer_fes.GetElementDofs(e, dofs);
      const auto& nodes = fe->GetNodes();
      for (int i = 0; i < dofs.Size(); i++) {
        T->Transform(nodes.IntPoint(i), x);
        for (int d = 0; d < dim; d++) {
          coords(d, dofs[i]) = x(d);
        }
      }
    }
  }

  // Which buffer scalar dofs are paired (the trace): from the vdof
  // pairing's rows (component 0 suffices for byNODES ordering).
  std::vector<char> paired(ns_buf, 0);
  for (int sb = 0; sb < ns_buf; sb++) {
    if (J->RowSize(buffer_fes.DofToVDof(sb, 0)) > 0) {
      paired[sb] = 1;
    }
  }

  // Locate the interior nodes' surface projections in the body mesh.
  std::vector<int> interior;
  for (int sb = 0; sb < ns_buf; sb++) {
    if (!paired[sb]) {
      interior.push_back(sb);
    }
  }
  DenseMatrix pts(dim, static_cast<int>(interior.size()));
  for (std::size_t i = 0; i < interior.size(); i++) {
    real_t r = 0.0;
    for (int d = 0; d < dim; d++) {
      r += coords(d, interior[i]) * coords(d, interior[i]);
    }
    r = std::sqrt(r);
    const real_t scale = pullback * r_body / std::max(r, real_t(1e-30));
    for (int d = 0; d < dim; d++) {
      pts(d, i) = scale * coords(d, interior[i]);
    }
  }
  Array<int> elem;
  Array<IntegrationPoint> ips;
  if (pts.Width() > 0) {
    body->FindPoints(pts, elem, ips);
    // Retry unfound projections deeper inside the body (the curved
    // discrete boundary can dip below the nominal radius); the interior
    // rule is a gauge choice, so the retreat costs nothing.
    for (real_t factor : {0.99, 0.97, 0.9}) {
      int missing = 0;
      for (int i = 0; i < elem.Size(); i++) {
        if (elem[i] < 0) {
          missing++;
        }
      }
      if (missing == 0) {
        break;
      }
      DenseMatrix retry(dim, missing);
      std::vector<int> which;
      for (int i = 0; i < elem.Size(); i++) {
        if (elem[i] < 0) {
          for (int d = 0; d < dim; d++) {
            retry(d, static_cast<int>(which.size())) =
                pts(d, i) * factor / pullback;
          }
          which.push_back(i);
        }
      }
      Array<int> elem2;
      Array<IntegrationPoint> ips2;
      body->FindPoints(retry, elem2, ips2);
      for (std::size_t j = 0; j < which.size(); j++) {
        if (elem2[j] >= 0) {
          elem[which[j]] = elem2[j];
          ips[which[j]] = ips2[j];
        }
      }
    }
  }

  auto E = std::make_unique<SparseMatrix>(buffer_fes.GetVSize(),
                                          body_fes.GetVSize());
  const int vdim = body_fes.GetVDim();
  // Trace rows: copy the body values exactly.
  {
    Array<int> cols;
    Vector vals;
    for (int sb = 0; sb < ns_buf; sb++) {
      if (!paired[sb]) {
        continue;
      }
      for (int k = 0; k < vdim; k++) {
        const int row = buffer_fes.DofToVDof(sb, k);
        J->GetRow(row, cols, vals);
        for (int j = 0; j < cols.Size(); j++) {
          E->Set(row, cols[j], vals[j]);
        }
      }
    }
  }
  // Interior rows: tapered radial interpolation.
  {
    Array<int> dofs;
    Vector shape;
    for (std::size_t i = 0; i < interior.size(); i++) {
      const int sb = interior[i];
      MFEM_VERIFY(elem[i] >= 0,
                  "NewRadialVacuumExtension: surface projection not found "
                  "in the body mesh; reduce `pullback`.");
      real_t r = 0.0;
      for (int d = 0; d < dim; d++) {
        r += coords(d, sb) * coords(d, sb);
      }
      r = std::sqrt(r);
      real_t t = (r_outer - r) / (r_outer - r_body);
      t = std::min(real_t(1), std::max(real_t(0), t));
      t = std::pow(t, taper_power);
      if (t == 0.0) {
        continue;
      }
      const auto* fe = body_fes.GetFE(elem[i]);
      shape.SetSize(fe->GetDof());
      fe->CalcShape(ips[i], shape);
      body_fes.GetElementDofs(elem[i], dofs);
      for (int k = 0; k < vdim; k++) {
        const int row = buffer_fes.DofToVDof(sb, k);
        for (int a = 0; a < dofs.Size(); a++) {
          E->Set(row, body_fes.DofToVDof(dofs[a], k), t * shape(a));
        }
      }
    }
  }
  E->Finalize();
  return E;
}

// ---------------------------------------------------------------------------
// ReferentialElasticRheology

ReferentialElasticRheology::ReferentialElasticRheology(int dim,
                                                       MatrixCoefficient& C,
                                                       MatrixCoefficient& S_e,
                                                       Diffeomorphism& phi_e)
    : dim_(dim), C_(&C), S_(&S_e), map_(&phi_e) {
  MFEM_VERIFY(dim == 2 || dim == 3,
              "ReferentialElasticRheology: dim must be 2 or 3.");
  const int ns = SymmetricTensorBasis::Size(dim);
  MFEM_VERIFY(C.GetHeight() == ns && C.GetWidth() == ns,
              "ReferentialElasticRheology: C must be n_s x n_s.");
  MFEM_VERIFY(S_e.GetHeight() == dim && S_e.GetWidth() == dim,
              "ReferentialElasticRheology: S_e must be d x d.");
}

Coefficient& ReferentialElasticRheology::RelaxationTime(int) const {
  MFEM_ABORT("ReferentialElasticRheology has no branches.");
}

const RelaxationLaw* ReferentialElasticRheology::Law(int) const {
  MFEM_ABORT("ReferentialElasticRheology has no branches.");
}

void ReferentialElasticRheology::BranchModulus(int, ElementTransformation&,
                                               const IntegrationPoint&,
                                               DenseMatrix&) const {
  MFEM_ABORT("ReferentialElasticRheology has no branches.");
}

void ReferentialElasticRheology::UnrelaxedModulus(ElementTransformation& T,
                                                  const IntegrationPoint& ip,
                                                  DenseMatrix& CU) const {
  C_->Eval(CU, T, ip);
}

std::unique_ptr<ElasticStiffness> ReferentialElasticRheology::MakeStiffness()
    const {
  return std::make_unique<ReferentialStiffness>(*C_, *S_, *map_);
}

// ---------------------------------------------------------------------------
// Construction

LinearQuasiStaticReferentialProblem::LinearQuasiStaticReferentialProblem(
    FiniteElementSpace* fes_u, FiniteElementSpace* fes_zeta,
    const ReferentialElasticRheology& rheology, Coefficient& density,
    real_t gravitational_constant, int dtn_degree,
    Coefficient* background_zeta0)
    : LinearQuasiStaticProblemBase(fes_u, rheology),
      dim_(fes_u->GetMesh()->Dimension()),
      fes_zeta_(fes_zeta),
      ref_rheology_(&rheology),
      rho_(&density),
      G_(gravitational_constant),
      four_pi_G_(4.0 * kPi * gravitational_constant),
      dtn_degree_(dtn_degree),
      one_(1.0),
      inv_four_pi_G_(1.0 / (4.0 * kPi * gravitational_constant)),
      shift_coef_(shift_ / (4.0 * kPi * gravitational_constant)),
      K_zeta_(Operator::MFEM_SPARSEMAT),
      K_shift_(Operator::MFEM_SPARSEMAT),
      C_(Operator::MFEM_SPARSEMAT) {
  MFEM_VERIFY(G_ > 0.0,
              "LinearQuasiStaticReferentialProblem: G must be positive.");
  MFEM_VERIFY(fes_zeta_->GetVDim() == 1,
              "LinearQuasiStaticReferentialProblem: the potential space must "
              "be scalar.");
  MFEM_VERIFY(fes_zeta_->GetMesh()->Dimension() == dim_,
              "LinearQuasiStaticReferentialProblem: mesh dimensions differ.");

  ball_wide_ = (fes_u->GetMesh() == fes_zeta_->GetMesh());
#ifdef MFEM_USE_MPI
  pfes_zeta_ = dynamic_cast<ParFiniteElementSpace*>(fes_zeta_);
  MFEM_VERIFY(
      (pfes_ != nullptr) == (pfes_zeta_ != nullptr),
      "LinearQuasiStaticReferentialProblem: the displacement and potential "
      "spaces must both be serial or both be parallel.");
  if (pfes_zeta_) {
    K_zeta_.SetType(Operator::Hypre_ParCSR);
    K_shift_.SetType(Operator::Hypre_ParCSR);
    C_.SetType(Operator::Hypre_ParCSR);
  }
#endif
  if (!ball_wide_) {
#ifdef MFEM_USE_MPI
    if (pfes_zeta_) {
      auto* psub = dynamic_cast<ParSubMesh*>(pfes_->GetParMesh());
      MFEM_VERIFY(psub && psub->GetParent() == pfes_zeta_->GetParMesh(),
                  "LinearQuasiStaticReferentialProblem: the displacement "
                  "space must live on a ParSubMesh of the potential space's "
                  "mesh, or on that mesh itself.");
      shadow_zeta_ = SubMeshDofInjection::MakeShadowSpace(*pfes_zeta_, *psub);
    } else
#endif
    {
      auto* sub = dynamic_cast<SubMesh*>(fes_->GetMesh());
      MFEM_VERIFY(sub && sub->GetParent() == fes_zeta_->GetMesh(),
                  "LinearQuasiStaticReferentialProblem: the displacement "
                  "space must live on a SubMesh of the potential space's "
                  "mesh, or on that mesh itself.");
      shadow_zeta_ = SubMeshDofInjection::MakeShadowSpace(*fes_zeta_, *sub);
    }
    injection_ =
        std::make_unique<SubMeshDofInjection>(*shadow_zeta_, *fes_zeta_);
  }

  // Potential fields.
  zeta0_ = detail::MakeGridFunction(fes_zeta_);
  zeta_ = detail::MakeGridFunction(fes_zeta_);
  *zeta0_ = 0.0;
  *zeta_ = 0.0;
  if (!ball_wide_) {
    zeta0_shadow_ = detail::MakeGridFunction(shadow_zeta_.get());
    zeta_shadow_ = detail::MakeGridFunction(shadow_zeta_.get());
    *zeta0_shadow_ = 0.0;
    *zeta_shadow_ = 0.0;
  }
  grad_zeta0_shadow_ = std::make_unique<GradientGridFunctionCoefficient>(
      ball_wide_ ? zeta0_.get() : zeta0_shadow_.get());
  Zeta_true_.SetSize(fes_zeta_->GetTrueVSize());
  Zeta_true_ = 0.0;

  // The DtN operator on the outer boundary: the mapping is the identity
  // there, so the plain closure applies (doc/gravitating_elasticity.md §3).
#ifdef MFEM_USE_MPI
  if (pfes_zeta_) {
    dtn_ = std::make_unique<PoissonDtNOperator>(pfes_zeta_->GetComm(),
                                                pfes_zeta_, dtn_degree_);
    dtn_->Assemble();
    dtn_rap_ = std::make_unique<RAPOperator>(dtn_->RAP());
    dtn_op_ = dtn_rap_.get();
  } else
#endif
  {
    dtn_ = std::make_unique<PoissonDtNOperator>(fes_zeta_, dtn_degree_);
    dtn_->Assemble();
    dtn_op_ = dtn_.get();
  }

  // 2-D compatibility data, as in the Eulerian class.
  if (dim_ == 2) {
    auto ones = detail::MakeGridFunction(fes_zeta_);
    *ones = 1.0;
    ones->GetTrueDofs(ones_);
    auto marker = ExternalBoundaryMarker(fes_zeta_->GetMesh());
    auto l = detail::MakeLinearForm(fes_zeta_);
    l->AddBoundaryIntegrator(new BoundaryLFIntegrator(one_), marker);
    l->Assemble();
    ToTrueDofs(*fes_zeta_, *l, L_outer_);
    outer_length_ = Dot(L_outer_, ones_);
    MFEM_VERIFY(outer_length_ > 0.0,
                "LinearQuasiStaticReferentialProblem: empty outer boundary.");
  }

  SetupPotentialOperators();
  ComputeBackgroundPotential(background_zeta0);
  SetupCoupling();
  SetupGravityIntegrators();
  SetupRigidModes();

  b_zeta_ = detail::MakeLinearForm(ball_wide_ ? fes_zeta_
                                              : shadow_zeta_.get());
  B_zeta_.SetSize(fes_zeta_->GetTrueVSize());
  B_zeta_ = 0.0;
}

bool LinearQuasiStaticReferentialProblem::ParallelPotential() const {
#ifdef MFEM_USE_MPI
  return pfes_zeta_ != nullptr;
#else
  return false;
#endif
}

void LinearQuasiStaticReferentialProblem::ToTrueDofs(
    const FiniteElementSpace& fes, const Vector& L, Vector& T) const {
  const Operator* P = fes.GetProlongationMatrix();
  if (P) {
    T.SetSize(P->Width());
    P->MultTranspose(L, T);
  } else {
    T = L;
  }
}

void LinearQuasiStaticReferentialProblem::MakeCompatible(
    Vector& B_zeta) const {
  if (dim_ != 2) {
    return;
  }
  const real_t mass = Dot(B_zeta, ones_);
  B_zeta.Add(-mass / outer_length_, L_outer_);
}

void LinearQuasiStaticReferentialProblem::SetupPotentialOperators() {
  Array<int> empty;
  auto& map = ref_rheology_->EquilibriumMapping();

  // K_a = <a_e grad zeta, grad chi> on the ball (the mapped Laplacian);
  // A_zeta = (K_a + DtN) / 4 pi G.
  k_zeta_form_ = detail::MakeBilinearForm(fes_zeta_);
  k_zeta_form_->AddDomainIntegrator(new TransformedDiffusionIntegrator(map));
  k_zeta_form_->Assemble();
  k_zeta_form_->FormSystemMatrix(empty, K_zeta_);
  const real_t c = 1.0 / four_pi_G_;
  A_zeta_op_ = std::make_unique<SumOperator>(K_zeta_.Ptr(), c, dtn_op_, c,
                                             false, false);
  A_zeta_ = A_zeta_op_.get();

  // Preconditioner: the shifted mapped Laplacian (K_a + eps M) / 4 pi G.
  shift_coef_.constant = shift_ / four_pi_G_;
  k_shift_form_ = detail::MakeBilinearForm(fes_zeta_);
  auto* tdi = new TransformedDiffusionIntegrator(map);
  k_shift_form_->AddDomainIntegrator(tdi);
  k_shift_form_->AddDomainIntegrator(new MassIntegrator(shift_coef_));
  k_shift_form_->Assemble();
  k_shift_form_->FormSystemMatrix(empty, K_shift_);
  {
    // Scale K_shift by 1/4piG through the operator: assemble unscaled and
    // wrap; AMG wants the matrix, so scale the matrix itself instead.
#ifdef MFEM_USE_MPI
    if (pfes_zeta_) {
      *K_shift_.As<HypreParMatrix>() *= c;
      auto amg =
          std::make_unique<HypreBoomerAMG>(*K_shift_.As<HypreParMatrix>());
      amg->SetPrintLevel(0);
      prec_zeta_ = std::move(amg);
    } else
#endif
    {
      *K_shift_.As<SparseMatrix>() *= c;
      prec_zeta_ = std::make_unique<GSSmoother>(*K_shift_.As<SparseMatrix>());
    }
  }

  // CG on A_zeta for the background solve; in 2-D the constant is
  // projected from both sides.
#ifdef MFEM_USE_MPI
  if (pfes_zeta_) {
    cg_zeta_ = std::make_unique<CGSolver>(pfes_zeta_->GetComm());
    if (dim_ == 2) {
      projector_c_ = std::make_unique<NullSpaceProjector>(pfes_zeta_->GetComm());
    }
  } else
#endif
  {
    cg_zeta_ = std::make_unique<CGSolver>();
    if (dim_ == 2) {
      projector_c_ = std::make_unique<NullSpaceProjector>();
    }
  }
  if (dim_ == 2) {
    projector_c_->Add(ones_);
    projected_zeta_op_ =
        std::make_unique<ProjectedOperator>(*A_zeta_, *projector_c_);
    cg_zeta_->SetOperator(*projected_zeta_op_);
    projected_prec_zeta_ = std::make_unique<ProjectedSolver>(*projector_c_);
    projected_prec_zeta_->SetSolver(*prec_zeta_);
    cg_zeta_->SetPreconditioner(*projected_prec_zeta_);
  } else {
    cg_zeta_->SetOperator(*A_zeta_);
    cg_zeta_->SetPreconditioner(*prec_zeta_);
  }
  cg_zeta_->SetRelTol(1e-12);
  cg_zeta_->SetAbsTol(0.0);
  cg_zeta_->SetMaxIter(10000);
  cg_zeta_->iterative_mode = false;
  if (dim_ == 2) {
    projected_zeta_ = std::make_unique<ProjectedSolver>(*projector_c_);
    projected_zeta_->SetSolver(*cg_zeta_);
    projected_zeta_->iterative_mode = false;
    zeta_solver_ = projected_zeta_.get();
  } else {
    zeta_solver_ = cg_zeta_.get();
  }
}

void LinearQuasiStaticReferentialProblem::ComputeBackgroundPotential(
    Coefficient* zeta0) {
  if (zeta0) {
    zeta0_->ProjectCoefficient(*zeta0);
  } else {
    // (K_a + DtN) Zeta0 / 4 pi G = -(rho, chi)_B with the referential
    // density integrated on the SubMesh and injected into the ball.
    Vector bL(fes_zeta_->GetVSize()), B;
    if (ball_wide_) {
      auto rho_form = detail::MakeLinearForm(fes_zeta_);
      rho_form->AddDomainIntegrator(new DomainLFIntegrator(*rho_));
      rho_form->Assemble();
      bL = *rho_form;
    } else {
      auto rho_form = detail::MakeLinearForm(shadow_zeta_.get());
      rho_form->AddDomainIntegrator(new DomainLFIntegrator(*rho_));
      rho_form->Assemble();
      injection_->Mult(*rho_form, bL);
    }
    ToTrueDofs(*fes_zeta_, bL, B);
    B *= -1.0;
    MakeCompatible(B);
    Vector Zeta0(B.Size());
    Zeta0 = 0.0;
    zeta_solver_->Mult(B, Zeta0);
    MFEM_VERIFY(cg_zeta_->GetConverged(),
                "LinearQuasiStaticReferentialProblem: the background "
                "potential solve did not converge.");
    zeta0_->SetFromTrueDofs(Zeta0);
  }
  if (!ball_wide_) {
    injection_->MultTranspose(*zeta0_, *zeta0_shadow_);
  }
}

void LinearQuasiStaticReferentialProblem::SetupCoupling() {
  // c(zeta, v) = (1/4piG) int_B <a'(v) g0, grad zeta>: trial zeta on the
  // ball, test v on the body; C^T by transposition.
  Array<int> empty;
  auto& map = ref_rheology_->EquilibriumMapping();
  const real_t c = 1.0 / four_pi_G_;
  if (ball_wide_) {
#ifdef MFEM_USE_MPI
    if (pfes_zeta_) {
      auto form = std::make_unique<ParMixedBilinearForm>(pfes_zeta_, pfes_);
      form->AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
          map, *grad_zeta0_shadow_, c));
      form->Assemble();
      form->Finalize();
      form->FormRectangularSystemMatrix(empty, empty, C_);
      c_form_ = std::move(form);
      Ct_owned_.reset(C_.As<HypreParMatrix>()->Transpose());
    } else
#endif
    {
      auto form = std::make_unique<MixedBilinearForm>(fes_zeta_, fes_);
      form->AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
          map, *grad_zeta0_shadow_, c));
      form->Assemble();
      form->Finalize();
      form->FormRectangularSystemMatrix(empty, empty, C_);
      c_form_ = std::move(form);
      Ct_owned_.reset(Transpose(*C_.As<SparseMatrix>()));
    }
    C_op_ = C_.Ptr();
    Ct_op_ = Ct_owned_.get();
    return;
  }
#ifdef MFEM_USE_MPI
  if (pfes_zeta_) {
    auto form = std::make_unique<ParSubMeshMixedBilinearForm>(pfes_zeta_, pfes_);
    form->AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
        map, *grad_zeta0_shadow_, c));
    form->Assemble();
    form->FormRectangularSystemMatrix(empty, empty, C_);
    c_form_ = std::move(form);
    Ct_owned_.reset(C_.As<HypreParMatrix>()->Transpose());
  } else
#endif
  {
    auto form = std::make_unique<SubMeshMixedBilinearForm>(fes_zeta_, fes_);
    form->AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
        map, *grad_zeta0_shadow_, c));
    form->Assemble();
    form->FormRectangularSystemMatrix(empty, empty, C_);
    c_form_ = std::move(form);
    Ct_owned_.reset(Transpose(*C_.As<SparseMatrix>()));
  }
  C_op_ = C_.Ptr();
  Ct_op_ = Ct_owned_.get();
}

void LinearQuasiStaticReferentialProblem::SetupGravityIntegrators() {
  auto& map = ref_rheology_->EquilibriumMapping();
  StiffnessIntegrators().AddDomainIntegrator(new ReferentialGravityIntegrator(
      map, *grad_zeta0_shadow_, 1.0 / (2.0 * four_pi_G_)));
}

void LinearQuasiStaticReferentialProblem::SetupRigidModes() {
#ifdef MFEM_USE_MPI
  if (pfes_) {
    projector_u_ = std::make_unique<NullSpaceProjector>(pfes_->GetComm());
  } else
#endif
  {
    projector_u_ = std::make_unique<NullSpaceProjector>();
  }
  auto gf = detail::MakeGridFunction(fes_);
  Vector t;
  auto add = [&](VectorCoefficient& c) {
    gf->ProjectCoefficient(c);
    gf->GetTrueDofs(t);
    projector_u_->Add(t);
  };
  for (int c = 0; c < dim_; c++) {
    Vector e(dim_);
    e = 0.0;
    e[c] = 1.0;
    VectorConstantCoefficient tc(e);
    add(tc);
  }
  auto& map = ref_rheology_->EquilibriumMapping();
  if (dim_ == 2) {
    MappedRotation rot(map, 2);
    add(rot);
  } else {
    for (int c = 0; c < 3; c++) {
      MappedRotation rot(map, c);
      add(rot);
    }
  }

  // The block projector: displacement modes with zero potential partners
  // (doc/gravitating_elasticity.md §3.1), plus the constant in 2-D.
#ifdef MFEM_USE_MPI
  if (pfes_) {
    projector_block_ = std::make_unique<NullSpaceProjector>(pfes_->GetComm());
  } else
#endif
  {
    projector_block_ = std::make_unique<NullSpaceProjector>();
  }
  offsets_.SetSize(3);
  offsets_[0] = 0;
  offsets_[1] = fes_->GetTrueVSize();
  offsets_[2] = fes_zeta_->GetTrueVSize();
  offsets_.PartialSum();
  BlockVector n(offsets_);
  for (int i = 0; i < projector_u_->Size(); i++) {
    n.GetBlock(0) = projector_u_->Basis(i);
    n.GetBlock(1) = 0.0;
    projector_block_->Add(n);
  }
  if (dim_ == 2) {
    n.GetBlock(0) = 0.0;
    n.GetBlock(1) = ones_;
    projector_block_->Add(n);
  }
}

// ---------------------------------------------------------------------------
// Loads

void LinearQuasiStaticReferentialProblem::SetSurfaceLoad(
    Coefficient& sigma, const Array<int>& bdr_marker) {
  MFEM_VERIFY(bdr_marker.Size() == fes_->GetMesh()->bdr_attributes.Max(),
              "SetSurfaceLoad: the marker must be sized to the SubMesh's "
              "bdr_attributes.Max().");
  RegisterTimeDependent(sigma);
  load_markers_.push_back(bdr_marker);
  auto& marker = load_markers_.back();
  auto minus_sigma = std::make_unique<ProductCoefficient>(-1.0, sigma);
  // Potential row only: the u-row load of the Eulerian form is absorbed
  // by the change of variables (doc/gravitating_elasticity.md §3.1).
  b_zeta_->AddBoundaryIntegrator(new BoundaryLFIntegrator(*minus_sigma),
                                 marker);
  load_coefs_.push_back(std::move(minus_sigma));
}


void LinearQuasiStaticReferentialProblem::SetPrescribedVacuumExtension(
    FiniteElementSpace& fes_buffer, const SparseMatrix& E) {
  MFEM_VERIFY(!ball_wide_,
              "SetPrescribedVacuumExtension: for the SubMesh mode (the "
              "ball-wide mode carries its own extension field).");
  MFEM_VERIFY(!ParallelPotential(),
              "SetPrescribedVacuumExtension: serial only at present.");
  MFEM_VERIFY(E.Height() == fes_buffer.GetVSize() &&
                  E.Width() == fes_->GetVSize(),
              "SetPrescribedVacuumExtension: E must map body vdofs to "
              "buffer vdofs.");
  MFEM_VERIFY(!ext_EtGE_, "SetPrescribedVacuumExtension: already set.");

  auto& map = ref_rheology_->EquilibriumMapping();
  auto* buffer_sub = dynamic_cast<SubMesh*>(fes_buffer.GetMesh());
  MFEM_VERIFY(buffer_sub && buffer_sub->GetParent() == fes_zeta_->GetMesh(),
              "SetPrescribedVacuumExtension: the buffer space must live on "
              "a SubMesh of the ball.");

  // zeta0 and its gradient on the buffer.
  shadow_zeta_buffer_ =
      SubMeshDofInjection::MakeShadowSpace(*fes_zeta_, *buffer_sub);
  SubMeshDofInjection inj(*shadow_zeta_buffer_, *fes_zeta_);
  zeta0_buffer_ = detail::MakeGridFunction(shadow_zeta_buffer_.get());
  inj.MultTranspose(*zeta0_, *zeta0_buffer_);
  grad_zeta0_buffer_ =
      std::make_unique<GradientGridFunctionCoefficient>(zeta0_buffer_.get());

  // The buffer's gravity-gravity block, folded: E^T G_V E.
  BilinearForm g_form(&fes_buffer);
  g_form.AddDomainIntegrator(new ReferentialGravityIntegrator(
      map, *grad_zeta0_buffer_, 1.0 / (2.0 * four_pi_G_)));
  g_form.Assemble();
  g_form.Finalize();
  std::unique_ptr<SparseMatrix> Et(Transpose(E));
  std::unique_ptr<SparseMatrix> GE(mfem::Mult(g_form.SpMat(), E));
  ext_EtGE_.reset(mfem::Mult(*Et, *GE));

  // The buffer's coupling, folded onto the body rows: C_total = C + E^T C_V.
  SubMeshMixedBilinearForm c_form(fes_zeta_, &fes_buffer);
  c_form.AddDomainIntegrator(new ReferentialGravityCouplingIntegrator(
      map, *grad_zeta0_buffer_, 1.0 / four_pi_G_));
  c_form.Assemble();
  std::unique_ptr<SparseMatrix> EtCv(mfem::Mult(*Et, c_form.SpMat()));
  ext_C_total_.reset(Add(*C_.As<SparseMatrix>(), *EtCv));
  ext_Ct_total_.reset(Transpose(*ext_C_total_));
  C_op_ = ext_C_total_.get();
  Ct_op_ = ext_Ct_total_.get();
  operator_dirty_ = true;
}

void LinearQuasiStaticReferentialProblem::SetVacuumExtension(
    const Array<int>& buffer_marker, Coefficient& mu_gauge, real_t epsilon,
    int refinements) {
  MFEM_VERIFY(ball_wide_,
              "SetVacuumExtension: only for a ball-wide displacement.");
  SetGaugedFluid(buffer_marker, mu_gauge, epsilon, refinements,
                 GaugePenalty::Harmonic);
}

bool LinearQuasiStaticReferentialProblem::GaugeRefine(Vector& X) {
  // As for the gauged fluid's coupled refinement: the physical residual
  // after an exact regularised solve is [eps Q delta_u; 0].
  gauge_residuals_.clear();
  Vector B_zeta_saved(B_zeta_);
  B_zeta_ = 0.0;
  Vector Zeta_acc(Zeta_true_);
  Vector r(X.Size()), d(X.Size()), prev;
  bool ok = true;
  int outer = outer_its_;
  for (int k = 0; k < gauge_refinements_; ++k) {
    Q_.Ptr()->Mult(k == 0 ? X : prev, r);
    gauge_residuals_.push_back(std::sqrt(Dot(r, r)));
    d = 0.0;
    if (X_block_) {
      *X_block_ = 0.0;
    }
    ok = SolveLinearSystem(r, d) && ok;
    outer += outer_its_;
    X += d;
    Zeta_acc += Zeta_true_;
    prev = d;
  }
  B_zeta_ = B_zeta_saved;
  Zeta_true_ = Zeta_acc;
  outer_its_ = outer;
  if (X_block_) {
    X_block_->GetBlock(0) = X;
    X_block_->GetBlock(1) = Zeta_true_;
  }
  DistributePotential(Zeta_true_);
  return ok;
}

void LinearQuasiStaticReferentialProblem::AssembleForce(real_t t) {
  LinearQuasiStaticProblemBase::AssembleForce(t);
  b_zeta_->Assemble();
  if (ball_wide_) {
    ToTrueDofs(*fes_zeta_, *b_zeta_, B_zeta_);
  } else {
    Vector bL(fes_zeta_->GetVSize());
    injection_->Mult(*b_zeta_, bL);
    ToTrueDofs(*fes_zeta_, bL, B_zeta_);
  }
  MakeCompatible(B_zeta_);
}

void LinearQuasiStaticReferentialProblem::RegisterFields(DataCollection& dc) {
  LinearQuasiStaticProblemBase::RegisterFields(dc);
  dc.RegisterField("potential",
                   ball_wide_ ? zeta_.get() : zeta_shadow_.get());
  dc.RegisterField("background_potential",
                   ball_wide_ ? zeta0_.get() : zeta0_shadow_.get());
}

// ---------------------------------------------------------------------------
// Solver

void LinearQuasiStaticReferentialProblem::SetupSolver(OperatorHandle& A) {
  Operator* A_uu = A.Ptr();
  if (ext_EtGE_) {
    A_aug_.Clear();
    A_aug_.Reset(Add(*A.As<SparseMatrix>(), *ext_EtGE_), true);
    A_uu = A_aug_.Ptr();
    SetupDefaultPreconditioner(A_aug_);
  } else {
    SetupDefaultPreconditioner(A);
  }

  block_op_ = std::make_unique<BlockOperator>(offsets_);
  block_op_->SetBlock(0, 0, A_uu);
  block_op_->SetBlock(0, 1, const_cast<Operator*>(C_op_));
  block_op_->SetBlock(1, 0, const_cast<Operator*>(Ct_op_));
  block_op_->SetBlock(1, 1, const_cast<Operator*>(A_zeta_));

  block_prec_ = std::make_unique<BlockDiagonalPreconditioner>(offsets_);
  block_prec_->SetDiagonalBlock(0, prec_.get());
  block_prec_->SetDiagonalBlock(1, prec_zeta_.get());

#ifdef MFEM_USE_MPI
  if (pfes_) {
    minres_ = std::make_unique<MINRESSolver>(pfes_->GetComm());
  } else
#endif
  {
    minres_ = std::make_unique<MINRESSolver>();
  }
  projected_op_ =
      std::make_unique<ProjectedOperator>(*block_op_, *projector_block_);
  minres_->SetOperator(*projected_op_);
  projected_prec_ = std::make_unique<ProjectedSolver>(*projector_block_);
  projected_prec_->SetSolver(*block_prec_);
  minres_->SetPreconditioner(*projected_prec_);
  minres_->SetRelTol(rel_tol_);
  minres_->SetAbsTol(0.0);
  minres_->SetMaxIter(10000);
  minres_->SetPrintLevel(print_level_);
  minres_->iterative_mode = true;

  projected_ = std::make_unique<ProjectedSolver>(*projector_block_);
  projected_->SetSolver(*minres_);
  projected_->iterative_mode = true;

  if (!X_block_ || X_block_->Size() != offsets_.Last()) {
    X_block_ = std::make_unique<BlockVector>(offsets_);
    *X_block_ = 0.0;
  }
  B_block_ = std::make_unique<BlockVector>(offsets_);
}

bool LinearQuasiStaticReferentialProblem::SolveLinearSystem(const Vector& B,
                                                            Vector& X) {
  B_block_->GetBlock(0) = B;
  B_block_->GetBlock(1) = B_zeta_;
  bool ok = true;
  if (!SetWarmStartTolerance(*minres_, *projected_prec_, *B_block_)) {
    X = 0.0;
    Zeta_true_ = 0.0;
    *X_block_ = 0.0;
  } else {
    projected_->Mult(*B_block_, *X_block_);
    ok = minres_->GetConverged();
    outer_its_ = minres_->GetNumIterations();
    NoteIterations(outer_its_);
    X = X_block_->GetBlock(0);
    projector_u_->Project(X);
    Zeta_true_ = X_block_->GetBlock(1);
  }
  DistributePotential(Zeta_true_);
  return ok;
}

void LinearQuasiStaticReferentialProblem::DistributePotential(
    const Vector& Z) {
  zeta_->SetFromTrueDofs(Z);
  if (!ball_wide_) {
    injection_->MultTranspose(*zeta_, *zeta_shadow_);
  }
}

// ---------------------------------------------------------------------------
// Diagnostics

std::vector<real_t> LinearQuasiStaticReferentialProblem::RigidPairResiduals() {
  EnsureOperator();
  real_t a_max = 0.0;
#ifdef MFEM_USE_MPI
  if (pfes_) {
    auto* hyp = A_.As<HypreParMatrix>();
    SparseMatrix diag, offd;
    HYPRE_BigInt* cmap = nullptr;
    hyp->GetDiag(diag);
    hyp->GetOffd(offd, cmap);
    real_t local = std::max(diag.MaxNorm(), offd.MaxNorm());
    MPI_Allreduce(&local, &a_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                  pfes_->GetComm());
  } else
#endif
  {
    a_max = A_.As<SparseMatrix>()->MaxNorm();
  }
  std::vector<real_t> out;
  BlockVector n(offsets_), r(offsets_);
  for (int i = 0; i < projector_u_->Size(); i++) {
    n.GetBlock(0) = projector_u_->Basis(i);
    n.GetBlock(1) = 0.0;
    block_op_->Mult(n, r);
    out.push_back(std::sqrt(Dot(r, r)) / a_max);
  }
  return out;
}

}  // namespace mfemElasticity
