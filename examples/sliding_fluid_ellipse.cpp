// ============================================================================
// sliding_fluid_ellipse.cpp
//
// The sliding fluid-solid interface (displacement discontinuity) on the
// geometry where it is genuinely needed: an ELLIPTICAL body with a fluid
// core, purely elastic, no gravity. Companion to gauged_fluid_cavity.cpp
// (which welds the interface: the admissible gauge for a barotropic fluid)
// and to doc/gauged_fluid.md §5 / doc/gauge_penalty_iteration.tex §4.
//
// Two displacement fields live on the solid and fluid SubMeshes of one
// parent; the pairing J = Pi_s^T Pi_f identifies the two interface traces
// nodally, and continuity of the NORMAL component alone is enforced by a
// penalty on the normal jump,
//
//     theta [ B, -B J; -J^T B, J^T B J ],   B = normal-normal form,
//
// driven to zero by augmented-Lagrangian iterations at moderate theta,
// interleaved with the Tikhonov refinement of the fluid's gauge penalty.
// The tangential jump stays free: the slip.
//
// Why the ellipse: on a circular interface a hydrostatic fluid can always
// be relabelled so that the displacement is continuous (slip along the
// interface is itself a relabelling), and under a uniform load the
// response is conformal with no slip at all. Neither holds on an ellipse:
//
//   1. even the UNIFORM radial load drives a genuine tangential slip
//      (run with -e 0 and again with the default to see the tangential
//      jump appear at O(ellipticity));
//   2. the frictionless interface no longer decouples the rotations:
//      a circular interface transmits no torque, so shell and core rotate
//      independently (two extra null modes, projected when -e 0); an
//      elliptical interface transmits torque through the normal forces
//      alone, and the relative rotation becomes a stiff physical mode.
//
// The geometry is made by deforming the canned two-layer disc with the
// area-preserving map (x, y) -> (a x, y / a), a = 1 + e: the interface
// and surface become concentric ellipses, and every piece of machinery
// (pairing, boundary normals, loads) simply acts on the deformed mesh.
//
// Printed diagnostics: AL contraction of the normal jump, the final
// normal and tangential jumps against the interface trace, and the fluid
// pressure spread (p = -kappa div u must still be uniform without
// gravity - the gauge-invariant observable of the fluid, ellipse or not).
//
// Serial (single rank): the parallel pairing exists
// (NewSubMeshPairingTrueDofMatrix), but the demonstration stays serial
// for clarity.
//
// Sample runs:
//    ./sliding_fluid_ellipse
//    ./sliding_fluid_ellipse -e 0
//    ./sliding_fluid_ellipse -e 0.3 -P2 0.01
//    ./sliding_fluid_ellipse -theta 1e3 -nal 12
// ============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "mfemElasticity.hpp"

using namespace mfem;
using namespace mfemElasticity;

namespace {

constexpr double kKappaSolid = 2.0;
constexpr double kMuSolid = 1.0;
constexpr double kKappaFluid = 1.0;
constexpr double kRCmb = 3483.0 / 6371.0;

double P0 = 0.01;  // uniform radial traction amplitude
double P2 = 0.0;   // degree-2 pattern
double A = 1.2;    // ellipse semi-axis a (b = 1 / a)

// Radial surface traction f = -P(x) x/|x| (torque-free about the origin
// for any pattern, so the common rotation stays a null mode).
void RadialTraction(const Vector& x, Vector& f) {
  const double r = x.Norml2();
  const double c = x[1] / r;
  f = x;
  f *= -(P0 + P2 * (2.0 * c * c - 1.0)) / r;
}

// Boundary attributes whose elements sit at scaled radius
// s = sqrt((x/a)^2 + (b-scaled)) in (s_min, s_max): the level sets of the
// undeformed radius, which the map carries onto concentric ellipses.
Array<int> ScaledRadialBdrMarker(Mesh& mesh, double a, double s_min,
                                 double s_max) {
  Array<int> marker(mesh.bdr_attributes.Max());
  marker = 0;
  for (int i = 0; i < mesh.GetNBE(); i++) {
    auto* tr = mesh.GetBdrElementTransformation(i);
    Vector c(mesh.Dimension());
    tr->Transform(Geometries.GetCenter(mesh.GetBdrElementGeometry(i)), c);
    const double sx = c[0] / a, sy = c[1] * a;
    const double s = std::sqrt(sx * sx + sy * sy);
    if (s > s_min && s < s_max) {
      marker[mesh.GetBdrAttribute(i) - 1] = 1;
    }
  }
  return marker;
}

void Show(Mesh& mesh, const GridFunction& f, const char* title) {
  char vishost[] = "localhost";
  socketstream sock(vishost, 19916);
  sock.precision(8);
  sock << "solution\n"
       << mesh << f << "window_title '" << title << "'"
       << "\nkeys Rjlbc\n"
       << std::flush;
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef MFEM_USE_MPI
  Mpi::Init(argc, argv);
  Hypre::Init();
  if (Mpi::WorldSize() > 1) {
    if (Mpi::Root()) {
      std::cout << "sliding_fluid_ellipse is a serial example: run it on "
                   "one rank.\n";
    }
    return 1;
  }
#endif

  const char* mesh_file = "../data/elastogravity_two_layer_2d.msh";
  int order = 2;
  double ellipticity = 0.2;
  double eps = 1.0e-2;
  double theta = 1.0e2;
  int n_al = 8;
  bool visualization = true;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use (2-D).");
  args.AddOption(&order, "-o", "--order", "Finite element order.");
  args.AddOption(&ellipticity, "-e", "--ellipticity",
                 "Semi-axis a = 1 + e, b = 1/a (0 recovers the disc).");
  args.AddOption(&eps, "-eps", "--epsilon", "Gauge penalty factor epsilon.");
  args.AddOption(&theta, "-theta", "--theta",
                 "Normal-jump penalty parameter.");
  args.AddOption(&n_al, "-nal", "--al-iterations",
                 "Augmented-Lagrangian iterations.");
  args.AddOption(&P0, "-P0", "--traction", "Uniform radial traction.");
  args.AddOption(&P2, "-P2", "--traction-degree2",
                 "Degree-2 traction pattern.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "GLVis visualisation (solid and fluid displacement, fluid "
                 "pressure).");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(std::cout);
    return 1;
  }
  A = 1.0 + ellipticity;
  const bool circular = ellipticity == 0.0;

  // The elliptical body: deform the whole canned mesh by the
  // area-preserving map (x, y) -> (a x, y/a).
  Mesh mesh(mesh_file, 1, 1);
  MFEM_VERIFY(mesh.Dimension() == 2, "a 2-D example");
  const int dim = 2;
  mesh.Transform([](const Vector& x, Vector& y) {
    y.SetSize(2);
    y[0] = A * x[0];
    y[1] = x[1] / A;
  });

  // Two independent displacement spaces on the solid and fluid SubMeshes
  // (shadow spaces of one parent space, as the dof injections require),
  // and the interface pairing J.
  Array<int> fluid_attr({1}), solid_attr({2});
  auto solid = SubMesh::CreateFromDomain(mesh, solid_attr);
  auto fluid = SubMesh::CreateFromDomain(mesh, fluid_attr);
  H1_FECollection fec(order, dim);
  FiniteElementSpace fes_parent(&mesh, &fec, dim);
  auto fes_s = SubMeshDofInjection::MakeShadowSpace(fes_parent, solid);
  auto fes_f = SubMeshDofInjection::MakeShadowSpace(fes_parent, fluid);
  SubMeshDofInjection inj_s(*fes_s, fes_parent), inj_f(*fes_f, fes_parent);
  auto J = NewSubMeshPairingMatrix(inj_s, inj_f);
  const int ns = fes_s->GetVSize(), nf = fes_f->GetVSize();

  // Stiffness: solid elasticity; fluid bulk modulus plus the eps
  // deviatoric gauge penalty (assembled separately too, for the gauge
  // source of the refinement).
  ConstantCoefficient kappa_s(kKappaSolid), mu_s(kMuSolid);
  ConstantCoefficient kappa_f(kKappaFluid), one(1.0);
  ConstantCoefficient eps_mu(eps * kKappaFluid);
  BilinearForm a_s(fes_s.get());
  a_s.AddDomainIntegrator(new ElasticityIntegrator(kappa_s, 1.0, 0.0));
  a_s.AddDomainIntegrator(new ElasticityIntegrator(mu_s, -2.0 / dim, 1.0));
  a_s.Assemble();
  a_s.Finalize();
  BilinearForm a_f(fes_f.get());
  a_f.AddDomainIntegrator(new ElasticityIntegrator(kappa_f, 1.0, 0.0));
  a_f.AddDomainIntegrator(new ElasticityIntegrator(eps_mu, -2.0 / dim, 1.0));
  a_f.Assemble();
  a_f.Finalize();
  BilinearForm q_f(fes_f.get());
  q_f.AddDomainIntegrator(new ElasticityIntegrator(eps_mu, -2.0 / dim, 1.0));
  q_f.Assemble();
  q_f.Finalize();

  // Interface forms, assembled ONCE on the solid side: B for the normal
  // jump, M for the full jump (tangential = difference).
  auto cmb = ScaledRadialBdrMarker(solid, A, 0.9 * kRCmb, 1.1 * kRCmb);
  BilinearForm b_form(fes_s.get());
  b_form.AddBoundaryIntegrator(new BoundaryNormalNormalIntegrator(one), cmb);
  b_form.Assemble();
  b_form.Finalize();
  const SparseMatrix& B = b_form.SpMat();
  BilinearForm m_form(fes_s.get());
  m_form.AddBoundaryIntegrator(new VectorMassIntegrator(one), cmb);
  m_form.Assemble();
  m_form.Finalize();

  // The penalty blocks theta [B, -BJ; -J^T B, J^T B J], folded into one
  // monolithic regularised matrix.
  std::unique_ptr<SparseMatrix> Jt(Transpose(*J));
  std::unique_ptr<SparseMatrix> BJ(mfem::Mult(B, *J));
  std::unique_ptr<SparseMatrix> JtB(mfem::Mult(*Jt, B));
  std::unique_ptr<SparseMatrix> JtBJ(mfem::Mult(*JtB, *J));
  Array<int> offsets({0, ns, nf});
  offsets.PartialSum();
  BlockMatrix blocks(offsets);
  std::unique_ptr<SparseMatrix> A00(Add(1.0, a_s.SpMat(), theta, B));
  std::unique_ptr<SparseMatrix> A11(Add(1.0, a_f.SpMat(), theta, *JtBJ));
  std::unique_ptr<SparseMatrix> A01(new SparseMatrix(*BJ));
  *A01 *= -theta;
  std::unique_ptr<SparseMatrix> A10(new SparseMatrix(*JtB));
  *A10 *= -theta;
  blocks.SetBlock(0, 0, A00.get());
  blocks.SetBlock(0, 1, A01.get());
  blocks.SetBlock(1, 0, A10.get());
  blocks.SetBlock(1, 1, A11.get());
  std::unique_ptr<SparseMatrix> R(blocks.CreateMonolithic());

  // Null space: common translations and the common rotation always. The
  // INDEPENDENT rotations of shell and core are null only for a circular
  // interface (it transmits no torque); an elliptical interface locks
  // them through the normal forces, and the relative rotation is a
  // physical mode.
  NullSpaceProjector P;
  {
    BlockVector n(offsets);
    GridFunction gs(fes_s.get()), gf(fes_f.get());
    for (int c = 0; c < dim; c++) {
      Vector e(dim);
      e = 0.0;
      e[c] = 1.0;
      VectorConstantCoefficient t(e);
      gs.ProjectCoefficient(t);
      gf.ProjectCoefficient(t);
      n.GetBlock(0) = gs;
      n.GetBlock(1) = gf;
      P.Add(n);
    }
    RigidRotation rot(dim, 2);
    gs.ProjectCoefficient(rot);
    gf.ProjectCoefficient(rot);
    if (circular) {
      n.GetBlock(0) = gs;
      n.GetBlock(1) = 0.0;
      P.Add(n);
      n.GetBlock(0) = 0.0;
      n.GetBlock(1) = gf;
      P.Add(n);
    } else {
      n.GetBlock(0) = gs;
      n.GetBlock(1) = gf;
      P.Add(n);
    }
  }

  // The load: radial traction on the outer (elliptical) surface.
  auto surf = ScaledRadialBdrMarker(solid, A, 0.9, 1.1);
  VectorFunctionCoefficient traction(dim, RadialTraction);
  LinearForm lf(fes_s.get());
  lf.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(traction), surf);
  lf.Assemble();
  BlockVector F(offsets);
  F.GetBlock(0) = lf;
  F.GetBlock(1) = 0.0;

  // Solver on the projected regularised operator.
  GSSmoother gs_prec(*R);
  ProjectedSolver prec_p(P);
  prec_p.SetSolver(gs_prec);
  ProjectedOperator op_p(*R, P);
  CGSolver cg;
  cg.SetOperator(op_p);
  cg.SetPreconditioner(prec_p);
  cg.SetRelTol(1e-12);
  cg.SetAbsTol(0.0);
  cg.SetMaxIter(20000);
  cg.iterative_mode = true;
  ProjectedSolver solver(P);
  solver.SetSolver(cg);
  solver.iterative_mode = true;

  // The augmented-Lagrangian loop, interleaved with the gauge source:
  //   R U_{k+1} = f - w_k + eps Q u_f,k ;  w_{k+1} = w_k + theta P U_{k+1}.
  // At the fixed point the normal jump vanishes at finite theta and the
  // eps shear drops out of the observables
  // (doc/gauge_penalty_iteration.tex §4.4).
  BlockVector U(offsets), rhs(offsets), w(offsets), t(offsets);
  U = 0.0;
  w = 0.0;
  auto normal_jump2 = [&](const BlockVector& X) {
    Vector js(ns);
    J->Mult(X.GetBlock(1), js);
    js -= X.GetBlock(0);
    Vector Bj(ns);
    B.Mult(js, Bj);
    return InnerProduct(js, Bj);
  };
  std::vector<double> jumps;
  for (int k = 0; k < n_al; k++) {
    rhs = F;
    rhs -= w;
    Vector qf(nf);
    q_f.SpMat().Mult(U.GetBlock(1), qf);
    rhs.GetBlock(1) += qf;
    solver.Mult(rhs, U);
    if (!cg.GetConverged()) {
      std::cout << "inner CG did not converge at AL iteration " << k << "\n";
      return 1;
    }
    Vector js(ns);
    J->Mult(U.GetBlock(1), js);
    js -= U.GetBlock(0);
    Vector Bj(ns);
    B.Mult(js, Bj);
    t = 0.0;
    t.GetBlock(0) -= Bj;
    Jt->AddMult(Bj, t.GetBlock(1));
    t *= theta;
    w += t;
    jumps.push_back(std::sqrt(normal_jump2(U)));
  }

  // Diagnostics: jump norms against the interface trace.
  GridFunction u_s(fes_s.get()), u_f(fes_f.get());
  u_s = U.GetBlock(0);
  u_f = U.GetBlock(1);
  {
    Vector Bu(ns);
    B.Mult(U.GetBlock(0), Bu);
    const double trace = std::sqrt(InnerProduct(U.GetBlock(0), Bu));
    Vector js(ns);
    J->Mult(U.GetBlock(1), js);
    js -= U.GetBlock(0);
    Vector Mj(ns);
    m_form.SpMat().Mult(js, Mj);
    const double full = std::sqrt(InnerProduct(js, Mj));
    const double normal = jumps.back();
    const double tangential = std::sqrt(std::max(0.0, full * full -
                                                          normal * normal));
    std::cout << "ellipse a = " << A << ", b = " << 1.0 / A << " (e = "
              << ellipticity << "), theta = " << theta << ", " << n_al
              << " AL iterations, eps = " << eps << "\n"
              << "normal-jump decay: ";
    for (auto j : jumps) {
      std::cout << std::setprecision(3) << j << "  ";
    }
    std::cout << "\nnormal jump / normal trace:     "
              << std::setprecision(3) << normal / trace
              << "  (constrained away)\n"
              << "tangential jump / normal trace: " << tangential / trace
              << "  (the slip; O(e) under the uniform load, 0 on the "
                 "disc)\n"
              << "null modes projected: " << P.Size()
              << (circular ? "  (disc: independent rotations)\n"
                           : "  (ellipse: rotations locked by the normal "
                             "forces)\n");
  }

  // The fluid pressure p = -kappa div u must be uniform without gravity,
  // ellipse or not: its spread is discretisation error, and it is the
  // gauge-invariant observable of the fluid.
  {
    DivergenceGridFunctionCoefficient div_u(&u_f);
    double p_int = 0.0, vol = 0.0;
    double p_max = -std::numeric_limits<double>::infinity();
    double p_min = std::numeric_limits<double>::infinity();
    for (int i = 0; i < fluid.GetNE(); i++) {
      auto* T = fluid.GetElementTransformation(i);
      const auto& ir = IntRules.Get(fluid.GetElementGeometry(i), 2 * order);
      for (int q = 0; q < ir.GetNPoints(); q++) {
        const auto& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        const double wq = ip.weight * T->Weight();
        const double p = -kKappaFluid * div_u.Eval(*T, ip);
        p_int += wq * p;
        vol += wq;
        p_max = std::max(p_max, p);
        p_min = std::min(p_min, p);
      }
    }
    const double p_mean = p_int / vol;
    std::cout << "fluid pressure: mean " << std::setprecision(6) << p_mean
              << ", relative spread " << (p_max - p_min) / std::abs(p_mean)
              << " (uniform in the continuum)\n";
  }

  if (visualization) {
    Show(solid, u_s, "Solid displacement");
    Show(fluid, u_f, "Fluid displacement (gauge; its trace slips)");
    L2_FECollection pfec(order - 1, dim);
    FiniteElementSpace pfes(&fluid, &pfec);
    GridFunction p_gf(&pfes);
    DivergenceGridFunctionCoefficient div_u(&u_f);
    ProductCoefficient minus_kappa_div(-kKappaFluid, div_u);
    p_gf.ProjectCoefficient(minus_kappa_div);
    Show(fluid, p_gf, "Fluid pressure");
  }

  return 0;
}
