#include "SelfGravitatingTestCommon.hpp"
#include "TestCommon.hpp"

/*
  Tests for LinearQuasiStaticSlipReferentialProblem (the three-block
  slip-interface solver of doc/slip_interface.tex, "the collected
  operator") on the two-layer disc: fluid core (attribute 1), solid
  mantle (2), buffer shell (3), DtN sphere.

  - Barotropic cross-check, the acceptance test of the assembly: on the
    two-layer hydrostatic background the broken (slipping) formulation
    must reproduce the welded gauged solution of the same physical
    problem. Tangential continuity is an admissible gauge for a
    barotropic fluid, so the solid displacement (modulo rigid modes) and
    the potential agree at the two discretisations' common accuracy.
    Unlike the old gravity-free sliding tests, pi(r_cmb) > 0 here, so
    B_Sigma, the mismatch gravity pieces and all the extension folds are
    load-bearing.
  - The AL iterations contract the normal jump far below the tangential
    jump (the slip the constraint must leave free).
  - Extension invariance: fluid extensions of taper powers 2 and 3 agree
    on the observables -- the built-in gauge test of the design.
  - Rigid pairs (common translations, independent shell/core rotations)
    are near-null under the physical block operator.
*/

namespace {

using namespace self_grav_test;

constexpr double kRc = 3483.0 / 6371.0;
constexpr double kEps = 1.0e-2;
constexpr double kTheta = 1.0e2;
constexpr int kALIterations = 8;

// Marker for the boundary attributes at radius in (r_min, r_max).
Array<int> RadialBdrMarker(Mesh& mesh, double r_min, double r_max) {
  Array<int> marker(mesh.bdr_attributes.Max());
  marker = 0;
  for (int i = 0; i < mesh.GetNBE(); i++) {
    auto* tr = mesh.GetBdrElementTransformation(i);
    Vector c(mesh.Dimension());
    tr->Transform(Geometries.GetCenter(mesh.GetBdrElementGeometry(i)), c);
    const double r = c.Norml2();
    if (r > r_min && r < r_max) {
      marker[mesh.GetBdrAttribute(i) - 1] = 1;
    }
  }
  return marker;
}

// The two-layer setting shared by the tests: parent ball, solid/fluid/
// buffer SubMeshes, spaces on a common collection, and the background.
struct Setting {
  std::unique_ptr<Mesh> parent;
  std::unique_ptr<SubMesh> solid, fluid, buffer, body;
  std::unique_ptr<H1_FECollection> fec;
  std::unique_ptr<FiniteElementSpace> fes_s, fes_f, fes_buffer, fes_zeta,
      fes_body, fes_zeta_ref;
  std::unique_ptr<RadialHydrostaticBackground> bg;
  double r_out = 0.0;

  explicit Setting(int order = 2) {
    parent = std::make_unique<Mesh>("../data/elastogravity_two_layer_2d.msh",
                                    1, 1);
    const int dim = parent->Dimension();
    Array<int> fluid_attr({1}), solid_attr({2}), buffer_attr({3}),
        body_attr({1, 2});
    solid = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, solid_attr));
    fluid = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, fluid_attr));
    buffer = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, buffer_attr));
    body = std::make_unique<SubMesh>(
        SubMesh::CreateFromDomain(*parent, body_attr));
    fec = std::make_unique<H1_FECollection>(order, dim);
    fes_s = std::make_unique<FiniteElementSpace>(solid.get(), fec.get(), dim);
    fes_f = std::make_unique<FiniteElementSpace>(fluid.get(), fec.get(), dim);
    fes_buffer =
        std::make_unique<FiniteElementSpace>(buffer.get(), fec.get(), dim);
    fes_zeta = std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    fes_body = std::make_unique<FiniteElementSpace>(body.get(), fec.get(), dim);
    fes_zeta_ref =
        std::make_unique<FiniteElementSpace>(parent.get(), fec.get());
    Vector bb_min, bb_max;
    parent->GetBoundingBox(bb_min, bb_max);
    r_out = bb_max.Normlinf();
    bg = std::make_unique<RadialHydrostaticBackground>(
        dim, [](double) { return kRho; }, [](double) { return kKappa; },
        [](double r) { return r < kRc ? 0.0 : kMu; }, kG, 1.0);
  }
};

double SurfaceSigma(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[1] / r;
  return 0.02 * (1.0 + (2.0 * c * c - 1.0));
}

// Pure degree-2 surface load (no degree-0 part): the Dahlen reduction
// must avoid l = 0, where the Dahlen fluid treatment has its known gap
// (the gauged family, and hence the slip solver, sits on the other side
// of it).
double SurfaceSigmaDegree2(const Vector& x) {
  const double r = x.Norml2();
  const double c = x[1] / r;
  return 0.02 * (2.0 * c * c - 1.0);
}

// Radial pressure traction of the gravity-free sliding tests.
void PressureTraction(const Vector& x, Vector& f) {
  const double r = x.Norml2();
  const double c = x[1] / r;
  f = x;
  f *= -0.01 * (1.0 + (2.0 * c * c - 1.0)) / r;
}

}  // namespace

TEST(SlipProblem, TwoLayerBarotropicCrossCheck) {
  const int order = 2;
  Setting s(order);
  const int dim = 2;

  auto interface = RadialBdrMarker(*s.solid, 0.9 * kRc, 1.1 * kRc);
  auto surface_s = RadialBdrMarker(*s.solid, 0.9, 1.1);
  FunctionCoefficient sigma(SurfaceSigma);
  ConstantCoefficient mu_gauge(kKappa);

  // --- The slipping three-block problem.
  LinearQuasiStaticSlipReferentialProblem slip(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
      s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
  auto Evac = NewRadialVacuumExtension(*s.fes_s, *s.fes_buffer, 1.0, s.r_out);
  slip.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  auto Ef = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc);
  slip.SetFluidExtension(*Ef);
  slip.SetFluidGauge(mu_gauge, kEps);
  slip.SetConstraint(kTheta, kALIterations);
  slip.SetSurfaceLoad(sigma, surface_s);
  slip.SetRelTol(1e-10);
  slip.AssembleForce(0.0);
  ASSERT_TRUE(slip.Solve());

  // The AL iterations contract the normal jump.
  const auto& jumps = slip.NormalJumpHistory();
  ASSERT_EQ(static_cast<int>(jumps.size()), kALIterations);
  EXPECT_LT(jumps.back(), 0.05 * jumps.front());
  std::cout << "normal jump: first " << jumps.front() << ", last "
            << jumps.back() << "\n";

  // The tangential jump stays free: well above the normal jump.
  {
    FiniteElementSpace parent_v(s.parent.get(), s.fec.get(), dim);
    SubMeshDofInjection inj_s(*s.fes_s, parent_v), inj_f(*s.fes_f, parent_v);
    auto J = NewSubMeshPairingMatrix(inj_s, inj_f);
    Vector js(s.fes_s->GetVSize());
    J->Mult(slip.FluidDisplacement(), js);
    js -= slip.Displacement();
    ConstantCoefficient one(1.0);
    BilinearForm bn(s.fes_s.get()), mm(s.fes_s.get());
    bn.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(one),
                             interface);
    mm.AddBoundaryIntegrator(new VectorMassIntegrator(one), interface);
    bn.Assemble();
    bn.Finalize();
    mm.Assemble();
    mm.Finalize();
    Vector t(js.Size());
    bn.SpMat().Mult(js, t);
    const double normal2 = std::abs(InnerProduct(js, t));
    mm.SpMat().Mult(js, t);
    const double full2 = std::abs(InnerProduct(js, t));
    EXPECT_GT(full2 - normal2, 1e2 * normal2);
  }

  // --- The welded gauged reference: the same physical problem through
  // the verified general class (continuous space over both layers).
  LinearQuasiStaticReferentialProblem welded(
      s.fes_body.get(), s.fes_zeta_ref.get(), s.bg->Rheology(),
      s.bg->Density(), kG, kDtNDegree);
  auto Evac_b =
      NewRadialVacuumExtension(*s.fes_body, *s.fes_buffer, 1.0, s.r_out);
  welded.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac_b);
  Array<int> fluid_marker(s.body->attributes.Max());
  fluid_marker = 0;
  fluid_marker[0] = 1;
  welded.SetGaugedFluid(fluid_marker, mu_gauge, kEps, 3);
  auto surface_b = RadialBdrMarker(*s.body, 0.9, 1.1);
  welded.SetSurfaceLoad(sigma, surface_b);
  welded.SetRelTol(1e-10);
  welded.AssembleForce(0.0);
  ASSERT_TRUE(welded.Solve());

  // Solid (mantle) displacement, modulo rigid modes: transfer the welded
  // solution through the parent onto the solid SubMesh.
  {
    FiniteElementSpace parent_v(s.parent.get(), s.fec.get(), dim);
    GridFunction on_parent(&parent_v), on_solid(s.fes_s.get());
    on_parent = 0.0;
    SubMesh::Transfer(welded.Displacement(), on_parent);
    SubMesh::Transfer(on_parent, on_solid);
    Vector d(slip.Displacement());
    d -= on_solid;
    auto proj = MakeRigidModeProjector(*s.fes_s);
    proj->Project(d);
    Vector ref(on_solid);
    proj->Project(ref);
    const double rel = d.Norml2() / ref.Norml2();
    std::cout << "solid displacement, slip vs welded: " << rel << "\n";
    EXPECT_LT(rel, 3e-2);
  }

  // Potential on the SOLID region, modulo the 2-D constant. Off the
  // solid, zeta1 = phi1 + vtil . grad zeta0 carries the extension field
  // vtil, which differs between the two formulations (and between two
  // fluid extensions): zeta1 is an observable only where the motion is,
  // i.e. on the solid.
  auto solid_potential = [&](const GridFunction& zeta_ball) {
    auto shadow = SubMeshDofInjection::MakeShadowSpace(*s.fes_zeta, *s.solid);
    GridFunction z(shadow.get());
    SubMesh::Transfer(zeta_ball, z);
    Vector v(z);
    v -= v.Sum() / v.Size();
    return v;
  };
  {
    Vector zs = solid_potential(slip.Potential());
    Vector zw = solid_potential(welded.Potential());
    Vector d(zs);
    d -= zw;
    const double rel = d.Norml2() / zw.Norml2();
    std::cout << "potential (solid region), slip vs welded: " << rel << "\n";
    EXPECT_LT(rel, 3e-2);
  }

  // --- Extension invariance: a taper-power-3 fluid extension must give
  // the same observables (the interior rule is gauge).
  {
    LinearQuasiStaticSlipReferentialProblem slip3(
        s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
        s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
    slip3.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
    auto Ef3 = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc, 3.0);
    slip3.SetFluidExtension(*Ef3);
    slip3.SetFluidGauge(mu_gauge, kEps);
    slip3.SetConstraint(kTheta, kALIterations);
    slip3.SetSurfaceLoad(sigma, surface_s);
    slip3.SetRelTol(1e-10);
    slip3.AssembleForce(0.0);
    ASSERT_TRUE(slip3.Solve());

    Vector du(slip.Displacement());
    du -= slip3.Displacement();
    const double rel_u = du.Norml2() / slip.Displacement().Norml2();
    Vector z2 = solid_potential(slip.Potential());
    Vector z3 = solid_potential(slip3.Potential());
    Vector dz(z2);
    dz -= z3;
    const double rel_z = dz.Norml2() / z2.Norml2();
    std::cout << "extension invariance: u " << rel_u << ", zeta " << rel_z
              << "\n";
    EXPECT_LT(rel_u, 1e-2);
    EXPECT_LT(rel_z, 1e-2);
  }
}

// Unit A3, reduction (a): on the spherical hydrostatic two-layer
// background the slip solver must reproduce the Eulerian Dahlen-class
// solution (fluid displacement eliminated, F1-F3 interface terms) under
// the change of variables zeta1 = phi1 + u.grad Phi0. A pure degree-2
// load keeps the comparison away from the Dahlen degree-0 gap. This is
// a genuinely two-sided check: the two formulations share no interface
// machinery.
TEST(SlipProblem, ReducesToDahlenOnSphere) {
  const int order = 2;
  Setting s(order);
  const int dim = 2;

  auto interface = RadialBdrMarker(*s.solid, 0.9 * kRc, 1.1 * kRc);
  auto surface_s = RadialBdrMarker(*s.solid, 0.9, 1.1);
  FunctionCoefficient sigma(SurfaceSigmaDegree2);
  ConstantCoefficient mu_gauge(kKappa);

  // The slipping three-block problem.
  LinearQuasiStaticSlipReferentialProblem slip(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
      s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
  auto Evac = NewRadialVacuumExtension(*s.fes_s, *s.fes_buffer, 1.0, s.r_out);
  slip.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  auto Ef = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc);
  slip.SetFluidExtension(*Ef);
  slip.SetFluidGauge(mu_gauge, kEps);
  slip.SetConstraint(kTheta, kALIterations);
  slip.SetSurfaceLoad(sigma, surface_s);
  slip.SetRelTol(1e-10);
  slip.AssembleForce(0.0);
  ASSERT_TRUE(slip.Solve());

  // The Dahlen path: displacement on the mantle only, the fluid core as
  // a FluidRegion (uniform density, zero gradient), seismological
  // moduli.
  FiniteElementSpace fes_u_d(s.solid.get(), s.fec.get(), dim);
  FiniteElementSpace fes_phi(s.parent.get(), s.fec.get());
  ConstantCoefficient kappa_c(kKappa), mu_c(kMu), rho_c(kRho), zero(0.0);
  IsotropicElasticRheology e_rheology(dim, kappa_c, mu_c);
  FluidRegion core;
  core.attributes = Array<int>({1});
  core.density = &rho_c;
  core.density_gradient = &zero;
  core.interface_marker = interface;
  std::vector<FluidRegion> fluids{core};
  LinearQuasiStaticSelfGravitatingProblem dahlen(
      &fes_u_d, &fes_phi, e_rheology, rho_c, kG, kDtNDegree, nullptr, fluids);
  FunctionCoefficient sigma_d(SurfaceSigmaDegree2);
  dahlen.SetSurfaceLoad(sigma_d, surface_s);
  dahlen.SetRelTol(1e-10);
  dahlen.AssembleForce(0.0);
  ASSERT_TRUE(dahlen.Solve());

  // Solid displacement, modulo rigid modes (same space layout).
  {
    Vector d(slip.Displacement());
    d -= dahlen.Displacement();
    auto proj = MakeRigidModeProjector(*s.fes_s);
    proj->Project(d);
    Vector ref(dahlen.Displacement());
    proj->Project(ref);
    const double rel = d.Norml2() / ref.Norml2();
    std::cout << "solid displacement, slip vs Dahlen: " << rel << "\n";
    EXPECT_LT(rel, 3e-2);
  }

  // Potential on the solid through the change of variables, modulo the
  // 2-D constant.
  {
    VectorGridFunctionCoefficient u_c(&dahlen.Displacement());
    InnerProductCoefficient advect(u_c, dahlen.BackgroundGravity());
    GridFunctionCoefficient phi1(&dahlen.PotentialOnBody());
    SumCoefficient zeta_expected(phi1, advect);
    GridFunction d(slip.PotentialOnBody());
    GridFunction z(d);
    z.ProjectCoefficient(zeta_expected);
    d -= z;
    d -= d.Sum() / d.Size();
    Vector ref(z);
    ref -= ref.Sum() / ref.Size();
    const double rel = d.Norml2() / ref.Norml2();
    std::cout << "potential (change of variables), slip vs Dahlen: " << rel
              << "\n";
    EXPECT_LT(rel, 3e-2);
  }
}

// Unit A3, reduction (b): with a massless background (pi = 0, gravity
// off) the slip solver reduces to the gravity-free sliding interface,
// whose solid solution has the condensed rank-one cavity form (the
// TestSlidingInterface reference): the fluid enters only through its
// bulk modulus against the interface volume change. The potential
// decouples and stays zero.
TEST(SlipProblem, PressureFreeReducesToSlidingCavity) {
  const int order = 2;
  Setting s(order);
  const int dim = 2;

  auto interface = RadialBdrMarker(*s.solid, 0.9 * kRc, 1.1 * kRc);
  auto surface_s = RadialBdrMarker(*s.solid, 0.9, 1.1);
  ConstantCoefficient mu_gauge(kKappa);
  VectorFunctionCoefficient traction(dim, PressureTraction);

  // Massless background: p0 = 0, zeta0 = 0, bare = seismological.
  RadialHydrostaticBackground bg0(
      dim, [](double) { return 0.0; }, [](double) { return kKappa; },
      [](double r) { return r < kRc ? 0.0 : kMu; }, kG, 1.0);

  LinearQuasiStaticSlipReferentialProblem slip(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), bg0.Rheology(),
      bg0.Density(), bg0.Pressure(), interface, kG, kDtNDegree);
  auto Evac = NewRadialVacuumExtension(*s.fes_s, *s.fes_buffer, 1.0, s.r_out);
  slip.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  auto Ef = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc);
  slip.SetFluidExtension(*Ef);
  slip.SetFluidGauge(mu_gauge, kEps);
  slip.SetConstraint(kTheta, kALIterations);
  slip.ExternalLoad().AddBoundaryIntegrator(
      new VectorBoundaryLFIntegrator(traction), surface_s);
  slip.SetRelTol(1e-10);
  slip.AssembleForce(0.0);
  ASSERT_TRUE(slip.Solve());

  // The potential decouples (no density anywhere) and stays zero.
  {
    Vector z(slip.Potential());
    const double uscale = slip.Displacement().Norml2();
    EXPECT_LT(z.Norml2(), 1e-8 * uscale);
  }

  // Condensed rank-one cavity reference on the solid space (the
  // TestSlidingInterface construction): cavity solve plus the fluid's
  // bulk reaction kappa_f / V against the interface volume change.
  {
    ConstantCoefficient kappa_c(kKappa), mu_c(kMu), one(1.0);
    IsotropicElasticRheology rheology(dim, kappa_c, mu_c);
    LinearQuasiStaticTractionProblem cond(s.fes_s.get(), rheology, traction,
                                          surface_s);
    LinearForm b_lf(s.fes_s.get());
    b_lf.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(one),
                               interface);
    b_lf.Assemble();
    double V = 0.0;
    for (int i = 0; i < s.fluid->GetNE(); i++) {
      V += s.fluid->GetElementVolume(i);
    }
    const double sig = kKappa / V;
    cond.AssembleForce(0.0);
    ASSERT_TRUE(cond.Solve());
    GridFunction u_a(cond.Displacement());
    cond.AssembleForce(0.0);
    Vector delta(b_lf);
    delta -= cond.ExternalLoad();
    cond.AddForce(delta);
    ASSERT_TRUE(cond.Solve());
    GridFunction u_b(cond.Displacement());
    GridFunction u_cond(u_a);
    u_cond.Add(-sig * b_lf(u_a) / (1.0 + sig * b_lf(u_b)), u_b);

    Vector d(slip.Displacement());
    d -= u_cond;
    auto proj = MakeRigidModeProjector(*s.fes_s);
    proj->Project(d);
    Vector ref(u_cond);
    proj->Project(ref);
    const double rel = d.Norml2() / ref.Norml2();
    std::cout << "solid displacement, pi = 0 slip vs condensed cavity: "
              << rel << "\n";
    EXPECT_LT(rel, 1e-2);
  }
}

// The broken-zeta organisation head to head against the single-valued
// one (doc/slip_interface.tex, sec:brokenzeta and reduction (d)): the
// same physical problem, same meshes and spaces, but the two
// organisations share NO gravity-interface machinery — mismatch volume
// terms + fluid extension on one side, G_Sigma + the scalar-jump
// constraint (and no fluid extension at all) on the other. Agreement of
// the solid displacement and the solid-region potential is the
// designed self-benchmark of both. Both constraints' AL iterations
// must contract their jumps.
TEST(SlipProblem, BrokenZetaHeadToHead) {
  const int order = 2;
  Setting s(order);
  const int dim = 2;

  auto interface = RadialBdrMarker(*s.solid, 0.9 * kRc, 1.1 * kRc);
  auto surface_s = RadialBdrMarker(*s.solid, 0.9, 1.1);
  FunctionCoefficient sigma(SurfaceSigma);
  ConstantCoefficient mu_gauge(kKappa);

  // --- The single-valued reference (the verified path).
  LinearQuasiStaticSlipReferentialProblem sv(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
      s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
  auto Evac = NewRadialVacuumExtension(*s.fes_s, *s.fes_buffer, 1.0, s.r_out);
  sv.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  auto Ef = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc);
  sv.SetFluidExtension(*Ef);
  sv.SetFluidGauge(mu_gauge, kEps);
  sv.SetConstraint(kTheta, kALIterations);
  sv.SetSurfaceLoad(sigma, surface_s);
  sv.SetRelTol(1e-10);
  sv.AssembleForce(0.0);
  ASSERT_TRUE(sv.Solve());

  // --- The broken-zeta problem: outer (solid + buffer) scalar region,
  // no fluid extension anywhere.
  Array<int> outer_attr({2, 3});
  SubMesh outer(SubMesh::CreateFromDomain(*s.parent, outer_attr));
  auto fes_zo = SubMeshDofInjection::MakeShadowSpace(*s.fes_zeta, outer);

  LinearQuasiStaticSlipReferentialProblem bz(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
      s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
  bz.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  bz.SetFluidGauge(mu_gauge, kEps);
  bz.SetConstraint(kTheta, kALIterations);
  bz.EnableBrokenZeta(fes_zo.get(), kTheta);
  bz.SetSurfaceLoad(sigma, surface_s);
  bz.SetRelTol(1e-10);
  bz.AssembleForce(0.0);
  ASSERT_TRUE(bz.Solve());

  // Both constraints contract under the AL iterations.
  {
    const auto& jn = bz.NormalJumpHistory();
    const auto& jz = bz.ZetaJumpHistory();
    ASSERT_EQ(static_cast<int>(jn.size()), kALIterations);
    ASSERT_EQ(static_cast<int>(jz.size()), kALIterations);
    EXPECT_LT(jn.back(), 0.05 * jn.front());
    EXPECT_LT(jz.back(), 0.2 * jz.front());
    std::cout << "broken-zeta normal jump: first " << jn.front() << ", last "
              << jn.back() << "\n";
    std::cout << "broken-zeta scalar jump: first " << jz.front() << ", last "
              << jz.back() << "\n";
  }

  // Solid displacement, modulo rigid modes.
  {
    Vector d(bz.Displacement());
    d -= sv.Displacement();
    auto proj = MakeRigidModeProjector(*s.fes_s);
    proj->Project(d);
    Vector ref(sv.Displacement());
    proj->Project(ref);
    const double rel = d.Norml2() / ref.Norml2();
    std::cout << "solid displacement, broken vs single-valued: " << rel
              << "\n";
    EXPECT_LT(rel, 3e-2);
  }

  // Potential on the solid region, modulo the 2-D constant.
  {
    auto solid_potential = [&](const GridFunction& zeta_ball) {
      auto shadow =
          SubMeshDofInjection::MakeShadowSpace(*s.fes_zeta, *s.solid);
      GridFunction z(shadow.get());
      SubMesh::Transfer(zeta_ball, z);
      Vector v(z);
      v -= v.Sum() / v.Size();
      return v;
    };
    Vector zb = solid_potential(bz.Potential());
    Vector zs = solid_potential(sv.Potential());
    Vector d(zb);
    d -= zs;
    const double rel = d.Norml2() / zs.Norml2();
    std::cout << "potential (solid region), broken vs single-valued: " << rel
              << "\n";
    EXPECT_LT(rel, 3e-2);
  }
}

TEST(SlipProblem, RigidPairsNearNull) {
  Setting s(2);
  auto interface = RadialBdrMarker(*s.solid, 0.9 * kRc, 1.1 * kRc);
  ConstantCoefficient mu_gauge(kKappa);

  LinearQuasiStaticSlipReferentialProblem slip(
      s.fes_s.get(), s.fes_f.get(), s.fes_zeta.get(), s.bg->Rheology(),
      s.bg->Density(), s.bg->Pressure(), interface, kG, kDtNDegree);
  auto Evac = NewRadialVacuumExtension(*s.fes_s, *s.fes_buffer, 1.0, s.r_out);
  slip.SetPrescribedVacuumExtension(*s.fes_buffer, *Evac);
  auto Ef = NewRadialFluidExtension(*s.fes_s, *s.fes_f, kRc);
  slip.SetFluidExtension(*Ef);
  slip.SetFluidGauge(mu_gauge, kEps);
  slip.SetConstraint(kTheta, kALIterations);

  const auto res = slip.SlipRigidPairResiduals();
  ASSERT_EQ(static_cast<int>(res.size()), 4);  // 2 translations, 2 rotations
  for (size_t i = 0; i < res.size(); i++) {
    std::cout << "slip rigid pair " << i << ": residual " << res[i] << "\n";
    EXPECT_LT(res[i], 5e-2);
  }
}
