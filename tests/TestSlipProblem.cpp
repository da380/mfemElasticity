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
