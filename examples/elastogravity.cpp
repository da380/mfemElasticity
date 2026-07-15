// The code for mesh generation is meshing/concentric_circles.cpp for 2d and meshing/concentric_spheres.cpp for 3d; in case it does not pass the mesh quality test of the DtN class for 3d, use meshing/offset_sphere.cpp 
#include <algorithm>
#include <cmath>
#include <mfem.hpp>
#include <mfemElasticity.hpp>

using namespace std;
using namespace mfem;
constexpr real_t pi = 3.141592653589793238462643383279502884;
constexpr real_t G_const = 6.67430e-11;
constexpr real_t L_scale = 6371e3;
constexpr real_t rho_scale = 5000.0;

const real_t T_scale = 1.0 / sqrt(G_const * rho_scale);
const real_t gravity_scale = L_scale / (T_scale * T_scale);
const real_t potential_scale = L_scale * L_scale / (T_scale * T_scale);
const real_t stress_scale = rho_scale * L_scale * L_scale / (T_scale * T_scale);

real_t rho_func(const Vector &coord);
real_t mu_func(const Vector &coord);
real_t lamb_func(const Vector &coord);
real_t loading_func(const Vector &coord);

real_t azimuthal_func(const Vector &coord);
real_t polar_func(const Vector &coord);

int main(int argc, char *argv[]) {
  StopWatch chrono;

  const char *mesh_file = "ex5_2d.msh";
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
                 "Order (degree) of the finite elements.");
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

  const real_t G_nd = G_const * rho_scale * T_scale * T_scale;
  const real_t poisson_rhs_factor = -4.0 * pi * G_nd;
  const real_t phi_block_factor = 1.0 / (4.0 * pi * G_nd);
  const real_t surface_load_scale = rho_scale * L_scale;

  cout << "\nReference scales:\n";
  cout << "  Length scale L        = " << L_scale << " m\n";
  cout << "  Time scale T          = " << T_scale << " s\n";
  cout << "  Density scale rho0    = " << rho_scale << " kg/m^3\n";
  cout << "  Potential scale Phi0  = " << potential_scale << " m^2/s^2\n";
  cout << "  Gravity scale g0      = " << gravity_scale << " m/s^2\n";
  cout << "  Stress scale p0       = " << stress_scale << " Pa\n";
  cout << "  Surface load scale    = " << surface_load_scale << " kg/m^2\n";

  cout << "\nDimensionless constants:\n";
  cout << "  G_nd                  = " << G_nd << endl;
  cout << "  -4*pi*G_nd            = " << poisson_rhs_factor << endl;
  cout << "  1/(4*pi*G_nd)         = " << phi_block_factor << endl;
  cout << endl;

  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  cout << "Mesh dimension: " << dim << endl;
  cout << "Domain attributes: ";
  mesh->attributes.Print(cout);
  cout << "Boundary attributes: ";
  mesh->bdr_attributes.Print(cout);

  Array<int> attr_cond(mesh->attributes.Max());
  attr_cond = 0;
  attr_cond[0] = 1;

  SubMesh mesh_cond(SubMesh::CreateFromDomain(*mesh, attr_cond));

  int order_phi = order_u;
  int order_dphi = order_phi - 1;

  H1_FECollection fec_u(order_u, dim), fec_phi(order_phi, dim);
  L2_FECollection fec_dphi(order_dphi, dim);

  FiniteElementSpace fes_phi(mesh, &fec_phi),
      fes_phi_cond(&mesh_cond, &fec_phi),
      fes_dphi_cond(&mesh_cond, &fec_dphi, dim);

  FiniteElementSpace fes_u(&mesh_cond, &fec_u, dim);

  cout << "Number of u-unknowns: " << fes_u.GetVSize() << endl;
  cout << "Number of phi-unknowns: " << fes_phi.GetVSize() << endl;

  GridFunction u_gf(&fes_u), phi_gf(&fes_phi), phi_gf_cond(&fes_phi_cond);
  GridFunction phi0_gf(&fes_phi), phi0_gf_cond(&fes_phi_cond),
      dphi0_gf_cond(&fes_dphi_cond);

  u_gf = 0.0;
  phi_gf = 0.0;
  phi_gf_cond = 0.0;
  phi0_gf = 0.0;
  phi0_gf_cond = 0.0;
  dphi0_gf_cond = 0.0;

  FunctionCoefficient rho_coeff(rho_func);
  FunctionCoefficient mu_coeff(mu_func);
  FunctionCoefficient lamb_coeff(lamb_func);
  FunctionCoefficient loading_coeff(loading_func);

  Array<int> ess_tdof_list;

  Array<int> bdr_marker(mesh->bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[mesh->bdr_attributes.Max() - 2] = 1;

  Array<int> bdr_marker_outer(mesh->bdr_attributes.Max());
  bdr_marker_outer = 0;
  bdr_marker_outer[mesh->bdr_attributes.Max() - 1] = 1;

  Array<int> bdr_marker_cond(mesh_cond.bdr_attributes.Max());
  bdr_marker_cond = 0;
  bdr_marker_cond[mesh_cond.bdr_attributes.Max() - 1] = 1;

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

  VectorGridFunctionCoefficient dphi0_cond_coeff(&dphi0_gf_cond);
  ScalarVectorProductCoefficient dphi0_sig_cond_coeff(loading_coeff,
                                                      dphi0_cond_coeff);

  cout << "Equilibrium state computed." << endl;

  if (visualization) {
    GridFunction phi0_vis(phi0_gf);
    phi0_vis *= potential_scale;

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
      sol_sock << "keys RRRilmc\n" << flush;
    }
  }

  Vector U, Phi;
  u_gf.GetTrueDofs(U);
  phi_gf.GetTrueDofs(Phi);

  LinearForm *b1(new LinearForm(&fes_u));
  b1->AddBoundaryIntegrator(
      new VectorBoundaryLFIntegrator(dphi0_sig_cond_coeff), bdr_marker_cond);
  b1->Assemble();

  LinearForm *b2(new LinearForm(&fes_phi));
  b2->AddBoundaryIntegrator(new BoundaryLFIntegrator(loading_coeff),
                            bdr_marker);
  b2->Assemble();

  BilinearForm *a11_0(new BilinearForm(&fes_u));
  BilinearForm *a11_1(new BilinearForm(&fes_u));
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

  // A11_0: pure elastic part
  a11_0->AddDomainIntegrator(a11_integ_0);
  a11_0->Assemble();
  a11_0->Finalize();

  // A11_1: extra gravity-related part
  a11_1->AddDomainIntegrator(a11_integ_1);
  a11_1->AddDomainIntegrator(a11_integ_2);
  a11_1->AddDomainIntegrator(a11_integ_1_t);
  a11_1->AddDomainIntegrator(a11_integ_2_t);
  a11_1->Assemble();
  a11_1->Finalize();

  a22->AddDomainIntegrator(new DiffusionIntegrator(c0));
  a22->Assemble();
  a22->Finalize();

  ConstantCoefficient eps22(shifting_factor * phi_block_factor);

  BilinearForm *a22s(new BilinearForm(&fes_phi));
  a22s->AddDomainIntegrator(new DiffusionIntegrator(c0));
  a22s->AddDomainIntegrator(new MassIntegrator(eps22));
  a22s->Assemble();
  a22s->Finalize();

  a12->AddDomainIntegrator(new GradientIntegrator(rho_coeff));
  a12->Assemble();
  a12->Finalize();

  a21->AddDomainIntegrator(
      new TransposeIntegrator(new GradientIntegrator(rho_coeff)));
  a21->Assemble();
  a21->Finalize();

  SparseMatrix &A11_0(a11_0->SpMat());
  SparseMatrix &A11_1(a11_1->SpMat());
  SparseMatrix &A22_0(a22->SpMat());
  SparseMatrix &A22s(a22s->SpMat());
  SparseMatrix &A12(a12->SpMat());
  SparseMatrix &A21(a21->SpMat());

  auto A22 = SumOperator(&A22_0, 1.0, &DtN, phi_block_factor, false, false);

  GSSmoother prec11(A11_0);
  GSSmoother prec22(A22s);

  // MINRESSolver solver1;
  CGSolver solver1;
  solver1.SetRelTol(rel_tol);
  solver1.SetMaxIter(3000);
  solver1.SetOperator(A11_0);
  solver1.SetPreconditioner(prec11);
  solver1.SetPrintLevel(1);

  mfemElasticity::RigidBodySolver rigid_solver(&fes_u);
  rigid_solver.SetSolver(solver1);

  CGSolver solver2;
  solver2.SetRelTol(rel_tol);
  solver2.SetMaxIter(3000);
  solver2.SetOperator(A22);
  solver2.SetPreconditioner(prec22);
  solver2.SetPrintLevel(1);

  OrthoSolver ortho_solver2;
  if (dim == 2) {
    ortho_solver2.SetSolver(solver2);
  }

  int max_iter = 1000;
  int iter = 0;
  real_t rel_tol_coup = 1e-6;

  LinearForm b1_ext(&fes_u);
  LinearForm b2_ext(&fes_phi);

  GridFunction one_phi(&fes_phi);
  LinearForm outer_l(&fes_phi);
  real_t outer_length = 0.0;
  if (dim == 2) {
    one_phi = 1.0;
    outer_l.AddBoundaryIntegrator(new BoundaryLFIntegrator(one),
                                  bdr_marker_outer);
    outer_l.Assemble();
    outer_length = outer_l(one_phi);
  }

  Vector Phi_new(Phi.Size()), Phi_diff(Phi.Size());
  Vector U_new(U.Size()), U_diff(U.Size());

  Phi_new = 0.0;
  Phi_diff = 0.0;
  U_new = 0.0;
  U_diff = 0.0;

  // under-relaxation factor
  real_t omega = 0.5;

  chrono.Clear();
  chrono.Start();
  for (int i = 0; i < max_iter; i++) {
    iter++;

    b1_ext = *b1;
    b2_ext = *b2;

    // --------------------------------------------------
    // Elasticity solve
    //
    // A11_0 U_new = b1 - A12 Phi_old - A11_1 U
    // --------------------------------------------------
    A12.AddMult(Phi, b1_ext, -1.0);
    A11_1.AddMult(U, b1_ext, -1.0);

    cout << "Elasticity solve: " << endl;

    rigid_solver.Mult(b1_ext, U_new);

    // under-relaxation
    U_new *= omega;
    U_new.Add(1.0 - omega, U);

    if (iter == 1) {
      real_t norm = solver1.GetFinalNorm();
      solver1.SetAbsTol(norm);

      cout << "solver1 adaptive abs tol = " << norm << endl;
    }

    U_diff = U_new;
    U_diff -= U;

    U = U_new;

    // --------------------------------------------------
    // Poisson solve
    // --------------------------------------------------

    A21.AddMult(U, b2_ext, -1.0);

    cout << "Poisson solve: " << endl;

    if (dim == 2) {
      real_t mass = b2_ext(one_phi);

      b2_ext.Add(-mass / outer_length, outer_l);
    }

    if (dim == 2) {
      ortho_solver2.Mult(b2_ext, Phi_new);
    } else {
      solver2.Mult(b2_ext, Phi_new);
    }

    if (iter == 1) {
      real_t norm = solver2.GetFinalNorm();
      solver2.SetAbsTol(norm);

      cout << "solver2 adaptive abs tol = " << norm << endl;
    }

    Phi_diff = Phi_new;
    Phi_diff -= Phi;

    real_t phi_den = Phi_new.Norml2();
    real_t u_den = U_new.Norml2();

    real_t phi_res = Phi_diff.Norml2() / max(phi_den, real_t(1e-30));

    real_t u_res = U_diff.Norml2() / max(u_den, real_t(1e-30));

    Phi = Phi_new;

    cout << "Iteration " << iter << ", phi residual = " << phi_res
         << ", u residual = " << u_res << endl;

    if (phi_res < rel_tol_coup && u_res < rel_tol_coup) {
      chrono.Stop();

      cout << "Converged at iteration " << iter << "." << endl;

      cout << "Takes " << chrono.RealTime() << "s." << endl;

      break;
    }
  }

  if (iter == max_iter) {
    chrono.Stop();
    cout << "Not Converged after " << iter << " iterations." << endl;
    cout << "Takes " << chrono.RealTime() << "s." << endl;
  }

  u_gf.SetFromTrueDofs(U);
  phi_gf.SetFromTrueDofs(Phi);
  mesh_cond.Transfer(phi_gf, phi_gf_cond);

  if (visualization) {
    GridFunction u_vis(u_gf);
    GridFunction phi_vis(phi_gf_cond);

    u_vis *= L_scale;
    phi_vis *= potential_scale;

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
      u_sock << "keys RRRilmc\n" << flush;
    }

    socketstream phi_sock(vishost, visport);
    phi_sock.precision(8);
    phi_sock
        << "solution\n"
        << mesh_cond << phi_vis
        << "window_title 'Dimensional gravity potential perturbation [m^2/s^2]'"
        << endl;

    if (dim == 2) {
      phi_sock << "keys Rjlbc\n" << flush;
    } else {
      phi_sock << "keys RRRilmc\n" << flush;
    }
  }

  delete b1;
  delete b2;
  delete a11_0;
  delete a11_1;
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

  if (r > 1.0) {
    return 0.0;
  }

  real_t rho_surface = 2.6e3;
  real_t rho_center = 1.3e4;

  real_t rho_dim = rho_center + (rho_surface - rho_center) * r;

  return rho_dim / rho_scale;
}

real_t mu_func(const Vector &coord) {
  real_t r = coord.Norml2();

  real_t mu_surface = 70e9;
  real_t mu_center = 140e9;

  real_t mu_dim = mu_center + (mu_surface - mu_center) * r;

  real_t polar_perturb = polar_func(coord);
  real_t azimuthal_perturb = 0.05 * azimuthal_func(coord);

  mu_dim *= (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);

  return mu_dim / stress_scale;
}

real_t lamb_func(const Vector &coord) {
  real_t r = coord.Norml2();

  real_t lamb_surface = 100e9;
  real_t lamb_center = 300e9;

  real_t lamb_dim = lamb_center + (lamb_surface - lamb_center) * r;

  real_t polar_perturb = polar_func(coord);
  real_t azimuthal_perturb = 0.05 * azimuthal_func(coord);

  lamb_dim *= (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);

  return lamb_dim / stress_scale;
}

real_t loading_func(const Vector &coord) {
  real_t factor = 1e-1;

  real_t pressure_high = 10e6;
  real_t pressure_low = 1e6;

  real_t pressure_profile = 0.0;

  if (coord.Size() == 2) {
    real_t r = coord.Norml2();

    if (r == 0.0) {
      pressure_profile = pressure_high;
    } else {
      real_t theta = acos(coord[1] / r);

      pressure_profile =
          (pressure_low + pressure_high) / 2.0 +
          (pressure_high - pressure_low) / 2.0 * cos(2.0 * theta);
    }
  } else {
    real_t r = coord.Norml2();

    if (r == 0.0) {
      pressure_profile = pressure_high;
    } else {
      real_t theta = acos(coord[2] / r);

      pressure_profile =
          (pressure_low + pressure_high) / 2.0 +
          (pressure_high - pressure_low) / 2.0 * cos(2.0 * theta);
    }
  }

  real_t azimuthal_perturb = 0.2 * azimuthal_func(coord);

  real_t pressure_dim = -pressure_profile * (1.0 + azimuthal_perturb) * factor;

  real_t sigma_dim = pressure_dim / gravity_scale;

  return sigma_dim / (rho_scale * L_scale);
}
