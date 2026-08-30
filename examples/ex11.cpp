// Serial code for solving elastogravity problem for a liquid-solid Earth model
// To generate the 2d mesh:
// ./concentric_circles -r 0.5467-1.0-1.2 -s 0.02-0.06 -o 2 -ma 6 -out mesh/ex11_2d.msh
// To generate the 3d mesh:
// ./concentric_spheres -r 0.5467-1.0-1.2 -s 0.02-0.06 -o 2 -ma 10 -out mesh/ex11_3d.msh
#include "common.hpp"
#include <cmath>
#include <mfem.hpp>
#include <mfemElasticity.hpp>

using namespace std;
using namespace mfem;

const int LIQUID_ATTR = 1;
const int SOLID_ATTR = 2;
const int VACUUM_ATTR = 3;

const int LS_BDR_ATTR = 1;
const int SURFACE_BDR_ATTR = 2;
const int OUTER_BDR_ATTR = 3;

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

  const char *mesh_file = "ex10_2d.msh";
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

  real_t G_nd = Constants::G * ND.Density() * ND.Time() * ND.Time();
  real_t poisson_rhs_factor = -4.0 * M_PI * G_nd;
  real_t phi_block_factor = 1.0 / (4.0 * M_PI * G_nd);
  real_t surface_load_scale = ND.Density() * ND.Length();

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
  attr_earth.Append(LIQUID_ATTR);
  attr_earth.Append(SOLID_ATTR);

  Array<int> attr_cond;
  attr_cond.Append(SOLID_ATTR);

  SubMesh mesh_earth(SubMesh::CreateFromDomain(*mesh, attr_earth));
  SubMesh mesh_cond(SubMesh::CreateFromDomain(*mesh, attr_cond));

  cout << "mesh_cond domain attributes: ";
  mesh_cond.attributes.Print(cout);

  cout << "mesh_cond boundary attributes: ";
  mesh_cond.bdr_attributes.Print(cout);

  int order_phi = order_u;
  int order_dphi = order_phi - 1;
  int order_prop = order_phi;

  H1_FECollection fec_u(order_u, dim);
  H1_FECollection fec_phi(order_phi, dim);
  L2_FECollection fec_dphi(order_dphi, dim);
  L2_FECollection fec_prop(order_prop, dim);

  FiniteElementSpace fes_phi(mesh, &fec_phi);
  FiniteElementSpace fes_phi_cond(&mesh_cond, &fec_phi);
  FiniteElementSpace fes_phi_earth(&mesh_earth, &fec_phi);
  FiniteElementSpace fes_dphi_cond(&mesh_cond, &fec_dphi, dim);
  FiniteElementSpace fes_u(&mesh_cond, &fec_u, dim);
  FiniteElementSpace fes_prop(mesh, &fec_prop);

  cout << "Number of u-unknowns: " << fes_u.GetVSize() << endl;
  cout << "Number of phi-unknowns: " << fes_phi.GetVSize() << endl;

  GridFunction u_gf(&fes_u);
  GridFunction phi_gf(&fes_phi);
  GridFunction phi_gf_cond(&fes_phi_cond);
  GridFunction phi_gf_earth(&fes_phi_earth);

  GridFunction phi0_gf(&fes_phi);
  GridFunction phi0_gf_cond(&fes_phi_cond);
  GridFunction dphi0_gf_cond(&fes_dphi_cond);

  u_gf = 0.0;
  phi_gf = 0.0;
  phi_gf_cond = 0.0;
  phi_gf_earth = 0.0;
  phi0_gf = 0.0;
  phi0_gf_cond = 0.0;
  dphi0_gf_cond = 0.0;

  FunctionCoefficient rho_coeff(rho_func);
  FunctionCoefficient mu_coeff(mu_func);
  FunctionCoefficient lamb_coeff(lamb_func);
  FunctionCoefficient loading_coeff(loading_func);

  FunctionCoefficient rho_liquid_coeff(rho_liquid_func);

  Array<int> ess_tdof_list;

  Array<int> bdr_marker(mesh->bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[SURFACE_BDR_ATTR - 1] = 1;

  Array<int> bdr_marker_outer(mesh->bdr_attributes.Max());
  bdr_marker_outer = 0;
  bdr_marker_outer[OUTER_BDR_ATTR - 1] = 1;

  Array<int> bdr_marker_cond(mesh_cond.bdr_attributes.Max());
  bdr_marker_cond = 0;
  bdr_marker_cond[SURFACE_BDR_ATTR - 1] = 1;

  Array<int> bdr_marker_ls(mesh_cond.bdr_attributes.Max());
  bdr_marker_ls = 0;
  bdr_marker_ls[LS_BDR_ATTR - 1] = 1;

  Array<int> liquid_marker(mesh->attributes.Max());
  liquid_marker = 0;
  liquid_marker[LIQUID_ATTR - 1] = 1;

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
  a0s.Finalize();

  OperatorPtr A0;
  Vector B0, Phi0;

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

  DiscreteLinearOperator Grad(&fes_phi_cond, &fes_dphi_cond);
  Grad.AddDomainInterpolator(new GradientInterpolator);
  Grad.Assemble();

  mesh_cond.Transfer(phi0_gf, phi0_gf_cond);
  Grad.Mult(phi0_gf_cond, dphi0_gf_cond);

  GradientGridFunctionCoefficient dphi0_coeff(&phi0_gf);
  VectorGridFunctionCoefficient dphi0_cond_coeff(&dphi0_gf_cond);

  ScalarVectorProductCoefficient dphi0_sig_cond_coeff(loading_coeff,
                                                      dphi0_cond_coeff);

  cout << "Equilibrium state computed." << endl;

  if (visualization) {
    GridFunction phi0_vis(phi0_gf);
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

  NormCoefficient g_coeff(dphi0_cond_coeff);
  ProductCoefficient rho_liquid_g_coeff(rho_liquid_coeff, g_coeff);

  Array<int> block_offsets(3);
  block_offsets[0] = 0;
  block_offsets[1] = fes_u.GetVSize();
  block_offsets[2] = fes_phi.GetVSize();
  block_offsets.PartialSum();

  cout << "***********************************************************\n";
  cout << "dim(u)       = " << block_offsets[1] - block_offsets[0] << "\n";
  cout << "dim(phi)     = " << block_offsets[2] - block_offsets[1] << "\n";
  cout << "dim(u+phi)   = " << block_offsets.Last() << "\n";
  cout << "***********************************************************\n";

  BlockVector X(block_offsets);
  BlockVector Rhs(block_offsets);

  X = 0.0;
  Rhs = 0.0;

  LinearForm b1(&fes_u);
  b1.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(dphi0_sig_cond_coeff),
                           bdr_marker_cond);
  b1.Assemble();

  LinearForm b2(&fes_phi);
  b2.AddBoundaryIntegrator(new BoundaryLFIntegrator(loading_coeff), bdr_marker);
  b2.Assemble();

  cout << "||b1|| = " << b1.Norml2() << endl;
  cout << "||b2|| = " << b2.Norml2() << endl;

  if (dim == 2) {
    GridFunction one_phi(&fes_phi);
    one_phi = 1.0;

    LinearForm outer_l(&fes_phi);
    outer_l.AddBoundaryIntegrator(new BoundaryLFIntegrator(one),
                                  bdr_marker_outer);
    outer_l.Assemble();

    real_t mass = b2(one_phi);
    real_t outer_length = outer_l(one_phi);

    b2.Add(-mass / outer_length, outer_l);
  }

  Rhs.GetBlock(0) = b1;
  Rhs.GetBlock(1) = b2;

  BilinearForm *a11(new BilinearForm(&fes_u));
  BilinearForm *a22(new BilinearForm(&fes_phi));

  auto a12 = new mfemElasticity::MixedBilinearFormSubMesh(&fes_phi, &fes_u,
                                                  &fes_phi_cond, true);

  auto a21 = new mfemElasticity::MixedBilinearFormSubMesh(&fes_u, &fes_phi,
                                                  &fes_phi_cond, false);

  ConstantCoefficient c0(phi_block_factor);

  ProductCoefficient half_rho_coeff(0.5, rho_coeff);
  ProductCoefficient minus_half_rho_coeff(-0.5, rho_coeff);

  auto *a11_integ_0 = new ElasticityIntegrator(lamb_coeff, mu_coeff);

  auto *a11_integ_1 = new mfemElasticity::DomainVectorGradVectorIntegrator(
      dphi0_cond_coeff, half_rho_coeff);

  ScalarVectorProductCoefficient a11_integ_2_coeff(minus_half_rho_coeff,
                                                   dphi0_cond_coeff);

  auto *a11_integ_2 =
      new mfemElasticity::DomainVectorDivVectorIntegrator(a11_integ_2_coeff);

  auto *a11_integ_1_t = new TransposeIntegrator(a11_integ_1, 0);

  auto *a11_integ_2_t = new TransposeIntegrator(a11_integ_2, 0);

  a11->AddDomainIntegrator(a11_integ_0);
  a11->AddDomainIntegrator(a11_integ_1);
  a11->AddDomainIntegrator(a11_integ_2);
  a11->AddDomainIntegrator(a11_integ_1_t);
  a11->AddDomainIntegrator(a11_integ_2_t);
  a11->AddBoundaryIntegrator(new BoundaryFluxIntegrator(rho_liquid_g_coeff),
                             bdr_marker_ls);
  a11->Assemble();
  a11->Finalize();

  a22->AddDomainIntegrator(new DiffusionIntegrator(c0));
  a22->AddDomainIntegrator(new MassIntegrator(invg_dr_rho_coeff),
                           liquid_marker);
  a22->Assemble();
  a22->Finalize();

  ConstantCoefficient eps22(shifting_factor * phi_block_factor);

  BilinearForm *a22s(new BilinearForm(&fes_phi));
  a22s->AddDomainIntegrator(new DiffusionIntegrator(c0));
  a22s->AddDomainIntegrator(new MassIntegrator(invg_dr_rho_coeff),
                            liquid_marker);
  a22s->AddDomainIntegrator(new MassIntegrator(eps22));
  a22s->Assemble();
  a22s->Finalize();

  a12->AddDomainIntegrator(new GradientIntegrator(rho_coeff));
  a12->AddBoundaryIntegrator(
      new BoundaryFluxMixedIntegrator(rho_liquid_coeff, -1.0), bdr_marker_ls);
  a12->Assemble();
  a12->Finalize();

  a21->AddDomainIntegrator(
      new TransposeIntegrator(new GradientIntegrator(rho_coeff)));
  a21->AddBoundaryIntegrator(
      new TransposeIntegrator(
          new BoundaryFluxMixedIntegrator(rho_liquid_coeff, -1.0)),
      bdr_marker_ls);
  a21->Assemble();
  a21->Finalize();

  SparseMatrix &A11(a11->SpMat());
  SparseMatrix &A22_0(a22->SpMat());
  SparseMatrix &A22s(a22s->SpMat());
  SparseMatrix &A12(a12->SpMat());
  SparseMatrix &A21(a21->SpMat());

  auto A22 = SumOperator(&A22_0, 1.0, &DtN, phi_block_factor, false, false);

  cout << "Symmetry tests: A11 = " << A11.IsSymmetric()
       << ", A22_0 = " << A22_0.IsSymmetric() << endl;

  BlockOperator EGOp(block_offsets);

  EGOp.SetBlock(0, 0, &A11);
  EGOp.SetBlock(0, 1, &A12);
  EGOp.SetBlock(1, 0, &A21);
  EGOp.SetBlock(1, 1, &A22);

  GSSmoother prec11(A11);
  GSSmoother prec22(A22s);

  BlockDiagonalPreconditioner EGPrec(block_offsets);
  EGPrec.SetDiagonalBlock(0, &prec11);
  EGPrec.SetDiagonalBlock(1, &prec22);

  MINRESSolver solver;
  solver.SetRelTol(rel_tol);
  solver.SetAbsTol(0.0);
  solver.SetMaxIter(5000);
  solver.SetOperator(EGOp);
  solver.SetPreconditioner(EGPrec);
  solver.SetPrintLevel(1);

  TwoBlockRigidBodySolver rigid_solver(&fes_u, &fes_phi, &block_offsets,
                                       &dphi0_coeff);

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
         << " iterations. Residual norm is " << solver.GetFinalNorm() << "."
         << endl;
  }

  cout << "Block solve takes " << chrono.RealTime() << "s." << endl;

  u_gf.SetFromTrueDofs(X.GetBlock(0));
  phi_gf.SetFromTrueDofs(X.GetBlock(1));

  mesh_earth.Transfer(phi_gf, phi_gf_earth);

  if (visualization) {
    GridFunction u_vis(u_gf);
    GridFunction phi_vis(phi_gf_earth);

    ND.UnscaleDisplacement(u_vis);
    ND.UnscaleGravityPotential(phi_vis);

    char vishost[] = "localhost";
    int visport = 19916;

    socketstream u_sock(vishost, visport);
    u_sock.precision(8);

    u_sock << "solution\n"
           << mesh_cond << u_vis << "window_title 'Dimensional deformation [m]'"
           << endl;

    if (dim == 2) {
      u_sock << "keys Rjlbc\n" << flush;
    } else {
      u_sock << "keys RRRilc\n" << flush;
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
  delete a12;
  delete a21;
  delete a22;
  delete a22s;
  delete mesh;

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
  real_t r = coord.Norml2();
  const real_t rc = 3483.0 / 6371.0;

  if (r > 1.0) {
    return 0.0;
  }

  real_t rho_dim;

  if (r < rc) {
    real_t rho_center = 13.4e3;
    real_t rho_cmb = 9.9e3;

    rho_dim = rho_center + (rho_cmb - rho_center) * (r / rc);
  } else {
    real_t rho_cmb_mantle = 5.6e3;
    real_t rho_surface = 3.3e3;

    real_t s = (r - rc) / (1.0 - rc);
    rho_dim = rho_cmb_mantle + (rho_surface - rho_cmb_mantle) * s;
  }

  return ND.ScaleDensity(rho_dim);
}

real_t drho_dr_func(const Vector &coord) {
  real_t r = coord.Norml2();
  const real_t rc = 3483.0 / 6371.0;

  if (r > 1.0) {
    return 0.0;
  }

  real_t drho_dim_dr_nd;

  if (r < rc) {
    real_t rho_center = 13.4e3;
    real_t rho_cmb = 9.9e3;

    drho_dim_dr_nd = (rho_cmb - rho_center) / rc;
  } else {
    real_t rho_cmb_mantle = 5.6e3;
    real_t rho_surface = 3.3e3;

    drho_dim_dr_nd = (rho_surface - rho_cmb_mantle) / (1.0 - rc);
  }

  return drho_dim_dr_nd / ND.Density();
}

real_t rho_liquid_func(const Vector &coord) {
  real_t r = coord.Norml2();
  const real_t rc = 3483.0 / 6371.0;

  real_t rho_center = 13.4e3;
  real_t rho_cmb = 9.9e3;

  if (r > rc) {
    r = rc;
  }

  real_t rho_dim = rho_center + (rho_cmb - rho_center) * (r / rc);

  return ND.ScaleDensity(rho_dim);
}

real_t mu_func(const Vector &coord) {
  real_t r = coord.Norml2();
  const real_t rc = 3483.0 / 6371.0;

  if (r < rc || r > 1.0) {
    return 0.0;
  }

  real_t s = (r - rc) / (1.0 - rc);

  real_t mu_cmb = 280e9;
  real_t mu_surface = 70e9;

  real_t mu_dim = mu_cmb + (mu_surface - mu_cmb) * s;

  real_t polar_perturb = polar_func(coord);
  real_t azimuthal_perturb = 0.05 * azimuthal_func(coord);

  mu_dim *= (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);

  return ND.ScaleStress(mu_dim);
}

real_t lamb_func(const Vector &coord) {
  real_t r = coord.Norml2();
  const real_t rc = 3483.0 / 6371.0;

  if (r < rc || r > 1.0) {
    return 0.0;
  }

  real_t s = (r - rc) / (1.0 - rc);

  real_t mu_cmb = 280e9;
  real_t mu_surface = 70e9;

  real_t K_cmb = 650e9;
  real_t K_surface = 130e9;

  real_t mu_dim = mu_cmb + (mu_surface - mu_cmb) * s;
  real_t K_dim = K_cmb + (K_surface - K_cmb) * s;

  real_t lamb_dim = K_dim - 2.0 * mu_dim / 3.0;

  real_t polar_perturb = polar_func(coord);
  real_t azimuthal_perturb = 0.05 * azimuthal_func(coord);

  lamb_dim *= (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);

  return ND.ScaleStress(lamb_dim);
}

real_t loading_func(const Vector &coord) {
  real_t factor = 1e-1;

  real_t pressure_high = 10e6;
  real_t pressure_low = 1e6;

  real_t pressure_profile = 0.0;

  real_t r = coord.Norml2();

  if (r == 0.0) {
    pressure_profile = pressure_high;
  } else if (coord.Size() == 2) {
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
