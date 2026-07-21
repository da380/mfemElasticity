// Serial code for solving elastogravity problem for a solid-liquid-solid Earth model
// To generate the 2d mesh:
// ./concentric_circles -r 0.1931-0.5467-1.0-1.2 -s 0.02-0.06 -o 2 -ma 6 -out mesh/ex12_2d.msh
// To generate the 3d mesh:
// ./concentric_spheres -r 0.1931-0.5467-1.0-1.2 -s 0.02-0.06 -o 2 -ma 10 -out mesh/ex12_3d.msh
#include "common.hpp"
#include <mfem.hpp>
#include <mfemElasticity.hpp>

#include <cmath>

using namespace std;
using namespace mfem;

const int INNER_CORE_ATTR = 1;
const int OUTER_CORE_ATTR = 2;
const int MANTLE_ATTR = 3;
const int VACUUM_ATTR = 4;

const int ICB_BDR_ATTR = 1;
const int CMB_BDR_ATTR = 2;
const int SURFACE_BDR_ATTR = 3;
const int OUTER_BDR_ATTR = 4;

const real_t R_ICB = 1230.0 / 6371.0;
const real_t R_CMB = 3483.0 / 6371.0;
const real_t R_SURFACE = 1.0;

Nondimensionalisation ND(6371e3, 1.0 / sqrt(Constants::G * 5000.0), 5000.0);

real_t rho_func(const Vector &coord);
real_t mu_func(const Vector &coord);
real_t lamb_func(const Vector &coord);
real_t loading_func(const Vector &coord);

real_t drho_dr_func(const Vector &coord);
real_t rho_liquid_func(const Vector &coord);

real_t azimuthal_func(const Vector &coord);
real_t polar_func(const Vector &coord);

int main(int argc, char *argv[]) {
  StopWatch chrono;

  const char *mesh_file = "mesh/ex11_2d.msh";
  real_t rel_tol = 1e-10;
  int order_u = 1;
  int deg = 16;
  bool visualization = false;

  real_t shifting_factor = 1e-3;

  OptionsParser args(argc, argv);

  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");

  args.AddOption(&rel_tol, "-rt", "--rel-tol",
                 "Relative tolerance for linear solving.");

  args.AddOption(&order_u, "-o", "--order",
                 "Order degree of the finite elements.");

  args.AddOption(&deg, "-deg", "--degree",
                 "Truncation degree for the DtN map.");

  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");

  args.Parse();
  if (!args.Good()) {

    args.PrintUsage(cout);

    return 1;
  }

  args.PrintOptions(cout);

  const real_t G_nd = Constants::G * ND.Density() * ND.Time() * ND.Time();
  const real_t poisson_rhs_factor = -4.0 * M_PI * G_nd;
  const real_t phi_block_factor = 1.0 / (4.0 * M_PI * G_nd);
  const real_t surface_load_scale = ND.Density() * ND.Length();

  ND.Print();

  cout << "\nDimensionless constants:\n";
  cout << "  G_nd                  = " << G_nd << endl;
  cout << "  -4*pi*G_nd            = " << poisson_rhs_factor << endl;
  cout << "  1/(4*pi*G_nd)         = " << phi_block_factor << endl;

  cout << "\nReference scales:\n";
  cout << "  Length scale L        = " << ND.Length() << " m\n";
  cout << "  Time scale T          = " << ND.Time() << " s\n";
  cout << "  Density scale rho0    = " << ND.Density() << " kg/m^3\n";
  cout << "  Potential scale Phi0  = " << ND.Potential() << " m^2/s^2\n";
  cout << "  Gravity scale g0      = " << ND.Gravity() << " m/s^2\n";
  cout << "  Stress scale p0       = " << ND.Pressure() << " Pa\n";
  cout << "  Surface load scale    = " << surface_load_scale << " kg/m^2\n";
  cout << endl;

  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  cout << "Mesh dimension: " << dim << endl;
  cout << "Domain attributes: ";
  mesh->attributes.Print(cout);
  cout << "Boundary attributes: ";
  mesh->bdr_attributes.Print(cout);

  Array<int> attr_earth;
  attr_earth.Append(INNER_CORE_ATTR);
  attr_earth.Append(OUTER_CORE_ATTR);
  attr_earth.Append(MANTLE_ATTR);

  Array<int> attr_ic;
  attr_ic.Append(INNER_CORE_ATTR);

  Array<int> attr_mantle;
  attr_mantle.Append(MANTLE_ATTR);

  SubMesh mesh_ic(SubMesh::CreateFromDomain(*mesh, attr_ic));
  SubMesh mesh_mantle(SubMesh::CreateFromDomain(*mesh, attr_mantle));
  SubMesh mesh_earth(SubMesh::CreateFromDomain(*mesh, attr_earth));

  cout << "Inner-core submesh domain attributes: ";
  mesh_ic.attributes.Print(cout);
  cout << "Inner-core submesh boundary attributes: ";
  mesh_ic.bdr_attributes.Print(cout);

  cout << "Mantle submesh domain attributes: ";
  mesh_mantle.attributes.Print(cout);
  cout << "Mantle submesh boundary attributes: ";
  mesh_mantle.bdr_attributes.Print(cout);

  cout << "Earth submesh domain attributes: ";
  mesh_earth.attributes.Print(cout);
  cout << "Earth submesh boundary attributes: ";
  mesh_earth.bdr_attributes.Print(cout);

  int order_phi = order_u;
  int order_dphi = order_phi - 1;
  int order_prop = order_phi;

  H1_FECollection fec_u(order_u, dim);
  H1_FECollection fec_phi(order_phi, dim);
  L2_FECollection fec_dphi(order_dphi, dim);
  L2_FECollection fec_prop(order_prop, dim);

  FiniteElementSpace fes_phi(mesh, &fec_phi);
  FiniteElementSpace fes_phi_ic(&mesh_ic, &fec_phi);
  FiniteElementSpace fes_phi_mantle(&mesh_mantle, &fec_phi);
  FiniteElementSpace fes_phi_earth(&mesh_earth, &fec_phi);
  FiniteElementSpace fes_dphi_ic(&mesh_ic, &fec_dphi, dim);
  FiniteElementSpace fes_dphi_mantle(&mesh_mantle, &fec_dphi, dim);
  FiniteElementSpace fes_u_ic(&mesh_ic, &fec_u, dim);
  FiniteElementSpace fes_u_mantle(&mesh_mantle, &fec_u, dim);
  FiniteElementSpace fes_prop(mesh, &fec_prop);

  int u_ic_size = fes_u_ic.GetVSize();
  int u_mantle_size = fes_u_mantle.GetVSize();
  int phi_size = fes_phi.GetVSize();

  cout << "Number of inner-core u-unknowns: " << u_ic_size << endl;

  cout << "Number of mantle u-unknowns: " << u_mantle_size << endl;

  cout << "Number of phi-unknowns: " << phi_size << endl;

  GridFunction u_ic_gf(&fes_u_ic);
  GridFunction u_mantle_gf(&fes_u_mantle);
  GridFunction phi_gf(&fes_phi);
  GridFunction phi_earth_gf(&fes_phi_earth);

  GridFunction phi0_gf(&fes_phi);
  GridFunction phi0_ic_gf(&fes_phi_ic);
  GridFunction phi0_mantle_gf(&fes_phi_mantle);
  GridFunction dphi0_ic_gf(&fes_dphi_ic);
  GridFunction dphi0_mantle_gf(&fes_dphi_mantle);

  u_ic_gf = 0.0;
  u_mantle_gf = 0.0;
  phi_gf = 0.0;
  phi_earth_gf = 0.0;

  phi0_gf = 0.0;
  phi0_ic_gf = 0.0;
  phi0_mantle_gf = 0.0;
  dphi0_ic_gf = 0.0;
  dphi0_mantle_gf = 0.0;

  FunctionCoefficient rho_coeff(rho_func);
  FunctionCoefficient mu_coeff(mu_func);
  FunctionCoefficient lamb_coeff(lamb_func);
  FunctionCoefficient loading_coeff(loading_func);

  FunctionCoefficient rho_liquid_coeff(rho_liquid_func);
  ProductCoefficient minus_rho_liquid_coeff(-1.0, rho_liquid_coeff);

  Array<int> ess_tdof_list;

  Array<int> bdr_marker_surface(mesh->bdr_attributes.Max());
  bdr_marker_surface = 0;
  bdr_marker_surface[SURFACE_BDR_ATTR - 1] = 1;

  Array<int> bdr_marker_outer(mesh->bdr_attributes.Max());
  bdr_marker_outer = 0;
  bdr_marker_outer[OUTER_BDR_ATTR - 1] = 1;

  Array<int> bdr_marker_icb_ic(mesh_ic.bdr_attributes.Max());
  bdr_marker_icb_ic = 0;
  bdr_marker_icb_ic[ICB_BDR_ATTR - 1] = 1;

  Array<int> bdr_marker_cmb_mantle(mesh_mantle.bdr_attributes.Max());
  bdr_marker_cmb_mantle = 0;
  bdr_marker_cmb_mantle[CMB_BDR_ATTR - 1] = 1;

  Array<int> bdr_marker_surface_mantle(mesh_mantle.bdr_attributes.Max());
  bdr_marker_surface_mantle = 0;
  bdr_marker_surface_mantle[SURFACE_BDR_ATTR - 1] = 1;

  Array<int> domain_marker_liquid(mesh->attributes.Max());
  domain_marker_liquid = 0;
  domain_marker_liquid[OUTER_CORE_ATTR - 1] = 1;

  auto DtN = mfemElasticity::PoissonDtNOperator(&fes_phi, deg);
  DtN.Assemble();

  ConstantCoefficient one(1.0);
  ProductCoefficient rhs_coeff(poisson_rhs_factor, rho_coeff);

  LinearForm b0(&fes_phi);
  b0.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
  b0.Assemble();

  if (dim == 2) {
    phi0_gf = 1.0;

    real_t mass = b0(phi0_gf);

    LinearForm l(&fes_phi);
    l.AddBoundaryIntegrator(new BoundaryLFIntegrator(one), bdr_marker_outer);
    l.Assemble();

    real_t length = l(phi0_gf);

    b0.Add(-mass / length, l);
  }

  BilinearForm a0(&fes_phi);
  a0.AddDomainIntegrator(new DiffusionIntegrator(one));
  a0.Assemble();

  ConstantCoefficient eps0(shifting_factor);

  BilinearForm a0s(&fes_phi);
  a0s.AddDomainIntegrator(new DiffusionIntegrator(one));
  a0s.AddDomainIntegrator(new MassIntegrator(eps0));
  a0s.Assemble();

  OperatorPtr A0;
  Vector B0;
  Vector Phi0;

  a0.FormLinearSystem(ess_tdof_list, phi0_gf, b0, A0, Phi0, B0);

  cout << "Size of equilibrium linear system: " << A0->Height() << endl;

  auto S0 = SumOperator(A0.Ptr(), 1.0, &DtN, 1.0, false, false);

  SparseMatrix A0s;
  a0s.FormSystemMatrix(ess_tdof_list, A0s);

  GSSmoother M0(A0s);

  CGSolver solver0;
  solver0.SetOperator(S0);
  solver0.SetPreconditioner(M0);
  solver0.SetRelTol(rel_tol);
  solver0.SetMaxIter(3000);
  solver0.SetPrintLevel(0);

  if (dim == 2) {
    OrthoSolver ortho_solver0;
    ortho_solver0.SetSolver(solver0);
    ortho_solver0.Mult(B0, Phi0);
  } else {
    solver0.Mult(B0, Phi0);
  }

  a0.RecoverFEMSolution(Phi0, b0, phi0_gf);

  DiscreteLinearOperator Grad_ic(&fes_phi_ic, &fes_dphi_ic);
  Grad_ic.AddDomainInterpolator(new GradientInterpolator);
  Grad_ic.Assemble();

  DiscreteLinearOperator Grad_mantle(&fes_phi_mantle, &fes_dphi_mantle);
  Grad_mantle.AddDomainInterpolator(new GradientInterpolator);
  Grad_mantle.Assemble();

  mesh_ic.Transfer(phi0_gf, phi0_ic_gf);
  mesh_mantle.Transfer(phi0_gf, phi0_mantle_gf);

  Grad_ic.Mult(phi0_ic_gf, dphi0_ic_gf);
  Grad_mantle.Mult(phi0_mantle_gf, dphi0_mantle_gf);

  GradientGridFunctionCoefficient dphi0_coeff(&phi0_gf);
  VectorGridFunctionCoefficient dphi0_ic_coeff(&dphi0_ic_gf);
  VectorGridFunctionCoefficient dphi0_mantle_coeff(&dphi0_mantle_gf);
  ScalarVectorProductCoefficient dphi0_sig_mantle_coeff(loading_coeff,
                                                        dphi0_mantle_coeff);

  cout << "Equilibrium state computed." << endl;

  if (visualization) {
    GridFunction phi0_vis(&fes_phi);
    phi0_vis = phi0_gf;

    ND.UnscaleGravityPotential(phi0_vis);

    char vishost[] = "localhost";
    int visport = 19916;

    socketstream sol_sock(vishost, visport);

    sol_sock.precision(8);

    sol_sock << "solution\n"
             << *mesh << phi0_vis
             << "window_title 'Dimensional equilibrium potential [m^2/s^2]'"
             << flush;

    if (dim == 2) {
      sol_sock << "keys Rjlbc\n" << flush;
    } else {
      sol_sock << "keys RRRilc\n" << flush;
    }
  }

  RadialDerivativeCoefficient g_liquid_coeff(phi0_gf);
  GridFunction rho_gf(&fes_prop);
  rho_gf.ProjectCoefficient(rho_coeff);
  RadialDerivativeCoefficient dr_rho_coeff(rho_gf);
  RatioCoefficient invg_dr_rho_coeff(dr_rho_coeff, g_liquid_coeff);

  NormCoefficient g_ic_coeff(dphi0_ic_coeff);
  ProductCoefficient rho_liquid_g_ic_coeff(rho_liquid_coeff, g_ic_coeff);
  ProductCoefficient minus_rho_liquid_g_ic_coeff(-1.0, rho_liquid_g_ic_coeff);

  NormCoefficient g_mantle_coeff(dphi0_mantle_coeff);
  ProductCoefficient rho_liquid_g_mantle_coeff(rho_liquid_coeff,
                                               g_mantle_coeff);
  ProductCoefficient minus_rho_liquid_g_mantle_coeff(-1.0,
                                                     rho_liquid_g_mantle_coeff);

  Array<int> block_offsets(4);
  block_offsets[0] = 0;
  block_offsets[1] = fes_u_ic.GetVSize();
  block_offsets[2] = fes_u_mantle.GetVSize();
  block_offsets[3] = fes_phi.GetVSize();
  block_offsets.PartialSum();

  cout << "***********************************************************\n";
  cout << "global dim(u_ic)       = " << u_ic_size << "\n";
  cout << "global dim(u_mantle)   = " << u_mantle_size << "\n";
  cout << "global dim(phi)        = " << phi_size << "\n";
  cout << "global dim(total)      = " << u_ic_size + u_mantle_size + phi_size
       << "\n";
  cout << "***********************************************************\n";

  BlockVector X(block_offsets);
  BlockVector Rhs(block_offsets);

  X = 0.0;
  Rhs = 0.0;

  LinearForm b2(&fes_u_mantle);

  b2.AddBoundaryIntegrator(
      new VectorBoundaryLFIntegrator(dphi0_sig_mantle_coeff),
      bdr_marker_surface_mantle);

  b2.Assemble();

  LinearForm b3(&fes_phi);

  b3.AddBoundaryIntegrator(new BoundaryLFIntegrator(loading_coeff),
                           bdr_marker_surface);

  b3.Assemble();

  if (dim == 2) {
    GridFunction one_phi_gf(&fes_phi);
    one_phi_gf = 1.0;

    LinearForm outer_l_form(&fes_phi);
    outer_l_form.AddBoundaryIntegrator(new BoundaryLFIntegrator(one),
                                       bdr_marker_outer);

    outer_l_form.Assemble();

    real_t mass = b3(one_phi_gf);
    real_t outer_length = outer_l_form(one_phi_gf);

    b3.Add(-mass / outer_length, outer_l_form);
  }

  Rhs.GetBlock(1) = b2;
  Rhs.GetBlock(2) = b3;

  BilinearForm *a11(new BilinearForm(&fes_u_ic));
  BilinearForm *a22(new BilinearForm(&fes_u_mantle));
  BilinearForm *a33(new BilinearForm(&fes_phi));

  auto a13 = new mfemElasticity::MixedBilinearFormSubMesh(&fes_phi, &fes_u_ic,
                                                          &fes_phi_ic, true);

  auto a31 = new mfemElasticity::MixedBilinearFormSubMesh(&fes_u_ic, &fes_phi,
                                                          &fes_phi_ic, false);

  auto a23 = new mfemElasticity::MixedBilinearFormSubMesh(
      &fes_phi, &fes_u_mantle, &fes_phi_mantle, true);

  auto a32 = new mfemElasticity::MixedBilinearFormSubMesh(
      &fes_u_mantle, &fes_phi, &fes_phi_mantle, false);

  ConstantCoefficient c0(phi_block_factor);

  ProductCoefficient half_rho_coeff(0.5, rho_coeff);
  ProductCoefficient minus_half_rho_coeff(-0.5, rho_coeff);

  auto *a11_integ_0 = new ElasticityIntegrator(lamb_coeff, mu_coeff);

  auto *a11_integ_1 = new mfemElasticity::DomainVectorGradVectorIntegrator(
      dphi0_ic_coeff, half_rho_coeff);

  ScalarVectorProductCoefficient a11_integ_2_coeff(minus_half_rho_coeff,
                                                   dphi0_ic_coeff);

  auto *a11_integ_2 =
      new mfemElasticity::DomainVectorDivVectorIntegrator(a11_integ_2_coeff);

  auto *a11_integ_1_t = new TransposeIntegrator(a11_integ_1, 0);
  auto *a11_integ_2_t = new TransposeIntegrator(a11_integ_2, 0);

  a11->AddDomainIntegrator(a11_integ_0);
  a11->AddDomainIntegrator(a11_integ_1);
  a11->AddDomainIntegrator(a11_integ_2);
  a11->AddDomainIntegrator(a11_integ_1_t);
  a11->AddDomainIntegrator(a11_integ_2_t);
  a11->AddBoundaryIntegrator(
      new BoundaryFluxIntegrator(minus_rho_liquid_g_ic_coeff),
      bdr_marker_icb_ic);
  a11->Assemble();
  a11->Finalize();

  auto *a22_integ_0 = new ElasticityIntegrator(lamb_coeff, mu_coeff);

  auto *a22_integ_1 = new mfemElasticity::DomainVectorGradVectorIntegrator(
      dphi0_mantle_coeff, half_rho_coeff);

  ScalarVectorProductCoefficient a22_integ_2_coeff(minus_half_rho_coeff,
                                                   dphi0_mantle_coeff);

  auto *a22_integ_2 =
      new mfemElasticity::DomainVectorDivVectorIntegrator(a22_integ_2_coeff);

  auto *a22_integ_1_t = new TransposeIntegrator(a22_integ_1, 0);
  auto *a22_integ_2_t = new TransposeIntegrator(a22_integ_2, 0);

  a22->AddDomainIntegrator(a22_integ_0);
  a22->AddDomainIntegrator(a22_integ_1);
  a22->AddDomainIntegrator(a22_integ_2);
  a22->AddDomainIntegrator(a22_integ_1_t);
  a22->AddDomainIntegrator(a22_integ_2_t);
  a22->AddBoundaryIntegrator(
      new BoundaryFluxIntegrator(rho_liquid_g_mantle_coeff),
      bdr_marker_cmb_mantle);
  a22->Assemble();
  a22->Finalize();

  a33->AddDomainIntegrator(new DiffusionIntegrator(c0));
  a33->AddDomainIntegrator(new MassIntegrator(invg_dr_rho_coeff),
                           domain_marker_liquid);
  a33->Assemble();
  a33->Finalize();

  ConstantCoefficient eps22(shifting_factor * phi_block_factor);

  BilinearForm *a33s(new BilinearForm(&fes_phi));
  a33s->AddDomainIntegrator(new DiffusionIntegrator(c0));
  // a33s->AddDomainIntegrator(new MassIntegrator(invg_dr_rho_coeff),
  // domain_marker_liquid);
  a33s->AddDomainIntegrator(new MassIntegrator(eps22));
  a33s->Assemble();
  a33s->Finalize();

  a13->AddDomainIntegrator(new GradientIntegrator(rho_coeff));
  a13->AddBoundaryIntegrator(
      new BoundaryFluxMixedIntegrator(minus_rho_liquid_coeff, 1.0),
      bdr_marker_icb_ic);
  a13->Assemble();
  a13->Finalize();

  a31->AddDomainIntegrator(
      new TransposeIntegrator(new GradientIntegrator(rho_coeff)));
  a31->AddBoundaryIntegrator(
      new TransposeIntegrator(
          new BoundaryFluxMixedIntegrator(minus_rho_liquid_coeff, 1.0)),
      bdr_marker_icb_ic);
  a31->Assemble();
  a31->Finalize();

  a23->AddDomainIntegrator(new GradientIntegrator(rho_coeff));
  a23->AddBoundaryIntegrator(
      new BoundaryFluxMixedIntegrator(rho_liquid_coeff, -1.0),
      bdr_marker_cmb_mantle);
  a23->Assemble();
  a23->Finalize();

  a32->AddDomainIntegrator(
      new TransposeIntegrator(new GradientIntegrator(rho_coeff)));
  a32->AddBoundaryIntegrator(
      new TransposeIntegrator(
          new BoundaryFluxMixedIntegrator(rho_liquid_coeff, -1.0)),
      bdr_marker_cmb_mantle);
  a32->Assemble();
  a32->Finalize();

  SparseMatrix &A11(a11->SpMat());
  SparseMatrix &A22(a22->SpMat());
  SparseMatrix &A33_0(a33->SpMat());
  SparseMatrix &A33s(a33s->SpMat());
  SparseMatrix &A13(a13->SpMat());
  SparseMatrix &A31(a31->SpMat());
  SparseMatrix &A23(a23->SpMat());
  SparseMatrix &A32(a32->SpMat());

  auto A33 = SumOperator(&A33_0, 1.0, &DtN, phi_block_factor, false, false);

  BlockOperator EGOp(block_offsets);

  EGOp.SetBlock(0, 0, &A11);
  EGOp.SetBlock(0, 2, &A13);
  EGOp.SetBlock(1, 1, &A22);
  EGOp.SetBlock(1, 2, &A23);
  EGOp.SetBlock(2, 0, &A31);
  EGOp.SetBlock(2, 1, &A32);
  EGOp.SetBlock(2, 2, &A33);

  GSSmoother prec11(A11);

  GSSmoother prec22(A22);

  GSSmoother prec33(A33s);

  BlockDiagonalPreconditioner EGPrec(block_offsets);
  EGPrec.SetDiagonalBlock(0, &prec11);
  EGPrec.SetDiagonalBlock(1, &prec22);
  EGPrec.SetDiagonalBlock(2, &prec33);

  MINRESSolver solver;
  solver.SetRelTol(rel_tol);
  solver.SetAbsTol(0.0);
  solver.SetMaxIter(10000);
  solver.SetOperator(EGOp);
  solver.SetPreconditioner(EGPrec);
  solver.SetPrintLevel(1);

  ThreeBlockRigidBodySolver rigid_solver(&fes_u_ic, &fes_u_mantle, &fes_phi,
                                         &block_offsets, EGOp, prec33,
                                         dim == 2);

  rigid_solver.SetSolver(solver);

  chrono.Clear();
  chrono.Start();

  rigid_solver.Mult(Rhs, X);

  chrono.Stop();

  if (solver.GetConverged()) {
    cout << "Block MINRES converged in " << solver.GetNumIterations()
         << " iterations with residual norm " << solver.GetFinalNorm() << "."
         << endl;
  } else {
    cout << "Block MINRES did not converge in " << solver.GetNumIterations()
         << " iterations. Residual norm = " << solver.GetFinalNorm() << "."
         << endl;
  }

  cout << "Block solve time = " << chrono.RealTime() << " s" << endl;

  u_ic_gf.SetFromTrueDofs(X.GetBlock(0));
  u_mantle_gf.SetFromTrueDofs(X.GetBlock(1));
  phi_gf.SetFromTrueDofs(X.GetBlock(2));

  mesh_earth.Transfer(phi_gf, phi_earth_gf);
  if (visualization) {
    GridFunction u_ic_vis(&fes_u_ic);
    GridFunction u_mantle_vis(&fes_u_mantle);
    GridFunction phi_vis(&fes_phi_earth);

    u_ic_vis = u_ic_gf;
    u_mantle_vis = u_mantle_gf;
    phi_vis = phi_earth_gf;

    ND.UnscaleDisplacement(u_ic_vis);
    ND.UnscaleDisplacement(u_mantle_vis);
    ND.UnscaleGravityPotential(phi_vis);

    char vishost[] = "localhost";
    int visport = 19916;

    socketstream u_ic_sock(vishost, visport);

    u_ic_sock.precision(8);

    u_ic_sock << "solution\n"
              << mesh_ic << u_ic_vis
              << "window_title 'Dimensional inner-core deformation [m]'"
              << endl;

    if (dim == 2) {
      u_ic_sock << "keys Rjlbc\n" << flush;
    } else {
      u_ic_sock << "keys RRRilc\n" << flush;
    }

    socketstream u_mantle_sock(vishost, visport);

    u_mantle_sock.precision(8);

    u_mantle_sock << "solution\n"
                  << mesh_mantle << u_mantle_vis
                  << "window_title 'Dimensional mantle deformation [m]'"
                  << endl;

    if (dim == 2) {
      u_mantle_sock << "keys Rjlbc\n" << flush;
    } else {
      u_mantle_sock << "keys RRRilc\n" << flush;
    }

    socketstream phi_sock(vishost, visport);

    phi_sock.precision(8);

    phi_sock
        << "solution\n"
        << mesh_earth << phi_vis
        << "window_title 'Dimensional gravity potential perturbation [m^2/s^2]'"
        << endl;

    if (dim == 2) {
      phi_sock << "keys Rjlbc\n" << flush;
    } else {
      phi_sock << "keys RRRilc\n" << flush;
    }
  }

  delete a11;
  delete a22;
  delete a13;
  delete a31;
  delete a23;
  delete a32;
  delete a33;
  delete a33s;

  return 0;
}

real_t azimuthal_func(const Vector &coord) {
  if (coord.Size() == 2) {
    return 0.0;
  }

  return sin(2.0 * atan2(coord[1], coord[0]));
}

real_t polar_func(const Vector &coord) {
  real_t r = coord.Norml2();

  if (r == 0.0) {
    return 0.0;
  }

  real_t theta;

  if (coord.Size() == 2) {
    theta = acos(coord[1] / r);
  } else {
    theta = acos(coord[2] / r);
  }

  return 0.015 * (1.0 + cos(2.0 * theta));
}

real_t rho_func(const Vector &coord) {
  const real_t r = coord.Norml2();

  if (r > R_SURFACE) {
    return 0.0;
  }

  real_t rho_dim;

  if (r < R_ICB) {
    const real_t rho_center = 13.1e3;
    const real_t rho_icb_inner = 12.8e3;

    rho_dim = rho_center + (rho_icb_inner - rho_center) * (r / R_ICB);
  } else if (r < R_CMB) {
    const real_t rho_icb_outer = 12.2e3;
    const real_t rho_cmb_outer = 9.9e3;

    const real_t s = (r - R_ICB) / (R_CMB - R_ICB);
    rho_dim = rho_icb_outer + (rho_cmb_outer - rho_icb_outer) * s;
  } else {
    const real_t rho_cmb_mantle = 5.6e3;
    const real_t rho_surface = 3.3e3;

    const real_t s = (r - R_CMB) / (R_SURFACE - R_CMB);
    rho_dim = rho_cmb_mantle + (rho_surface - rho_cmb_mantle) * s;
  }

  return ND.ScaleDensity(rho_dim);
}

real_t rho_liquid_func(const Vector &coord) {
  real_t r = coord.Norml2();

  const real_t rho_icb_outer = 12.2e3;
  const real_t rho_cmb_outer = 9.9e3;

  if (r < R_ICB) {
    r = R_ICB;
  } else if (r > R_CMB) {
    r = R_CMB;
  }

  const real_t s = (r - R_ICB) / (R_CMB - R_ICB);
  const real_t rho_dim = rho_icb_outer + (rho_cmb_outer - rho_icb_outer) * s;

  return ND.ScaleDensity(rho_dim);
}

real_t mu_func(const Vector &coord) {
  const real_t r = coord.Norml2();

  if (r > R_SURFACE) {
    return 0.0;
  }

  real_t mu_dim;

  if (r < R_ICB) {
    const real_t s = r / R_ICB;

    const real_t mu_center = 176e9;
    const real_t mu_icb = 156e9;

    mu_dim = mu_center + (mu_icb - mu_center) * s;
  } else if (r < R_CMB) {
    return 0.0;
  } else {
    const real_t s = (r - R_CMB) / (R_SURFACE - R_CMB);

    const real_t mu_cmb = 294e9;
    const real_t mu_surface = 68e9;

    mu_dim = mu_cmb + (mu_surface - mu_cmb) * s;

    const real_t polar_perturb = polar_func(coord);
    const real_t azimuthal_perturb = 0.05 * azimuthal_func(coord);

    mu_dim *= (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
  }

  return ND.ScaleStress(mu_dim);
}

real_t lamb_func(const Vector &coord) {
  const real_t r = coord.Norml2();

  if (r > R_SURFACE) {
    return 0.0;
  }

  real_t lamb_dim;

  if (r < R_ICB) {
    const real_t s = r / R_ICB;

    const real_t lamb_center = 1.31e12;
    const real_t lamb_icb = 1.24e12;

    lamb_dim = lamb_center + (lamb_icb - lamb_center) * s;
  } else if (r < R_CMB) {
    return 0.0;
  } else {
    const real_t s = (r - R_CMB) / (R_SURFACE - R_CMB);

    const real_t lamb_cmb = 461e9;
    const real_t lamb_surface = 86e9;

    lamb_dim = lamb_cmb + (lamb_surface - lamb_cmb) * s;

    const real_t polar_perturb = polar_func(coord);
    const real_t azimuthal_perturb = 0.05 * azimuthal_func(coord);

    lamb_dim *= (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
  }

  return ND.ScaleStress(lamb_dim);
}

real_t loading_func(const Vector &coord) {
  real_t factor = 1e-1;

  real_t pressure_high = 10e6;
  real_t pressure_low = 1e6;

  real_t pressure_profile = 0.0;

  real_t r = coord.Norml2();

  if (coord.Size() == 2) {
    real_t theta = acos(coord[1] / r);

    pressure_profile = (pressure_low + pressure_high) / 2.0 +
                       (pressure_high - pressure_low) / 2.0 * cos(2.0 * theta);
  } else {
    real_t theta = acos(coord[2] / r);

    pressure_profile = (pressure_low + pressure_high) / 2.0 +
                       (pressure_high - pressure_low) / 2.0 * cos(2.0 * theta);
  }

  real_t azimuthal_perturb = 0.2 * azimuthal_func(coord);

  real_t pressure_dim = -pressure_profile * (1.0 + azimuthal_perturb) * factor;

  real_t sigma_dim = pressure_dim / ND.Gravity();

  return sigma_dim / (ND.Density() * ND.Length());
}
