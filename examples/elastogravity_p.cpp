#include <mfem.hpp>
#include <mfemElasticity.hpp>
#include <cmath>

using namespace std;
using namespace mfem;

real_t rho_func(const Vector &coord);
real_t mu_func(const Vector &coord);
real_t lamb_func(const Vector &coord);
real_t loading_func(const Vector &coord);

const real_t G_const = 6.6743e-11;
const real_t R_const = 6371e3;


int main(int argc, char *argv[])
{
    StopWatch chrono;

    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    const char *mesh_file = "ex5.msh";
    real_t rel_tol = 1e-10;
    int order_u = 1;
    int deg = 16;
    bool visualization = false;

    //Parsing
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
            "Mesh file to use.");
    args.AddOption(&rel_tol, "-rt", "--rel-tol",
            "Relative tolerance for linear solving.");
    args.AddOption(&order_u, "-o", "--order",
            "Order (degree) of the finite elements.");
    args.AddOption(&deg, "-deg", "--degree", "Truncation degree for the DtN map.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
            "--no-visualization",
            "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    //Mesh
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    Array<int> attr_cond = pmesh.attributes;
    attr_cond.DeleteLast();

    ParSubMesh pmesh_cond(ParSubMesh::CreateFromDomain(pmesh, attr_cond));

    //FE Space
    int order_phi = order_u; int order_dphi = order_phi - 1;  
    H1_FECollection fec_u(order_u, dim), fec_phi(order_phi, dim);
    L2_FECollection fec_dphi(order_dphi, dim), fec_u_l2(order_u, dim);
    ParFiniteElementSpace fes_phi(&pmesh, &fec_phi), fes_phi_cond(&pmesh_cond, &fec_phi), 
                          fes_dphi(&pmesh, &fec_dphi, dim), fes_dphi_cond(&pmesh_cond, &fec_dphi, dim);
    ParFiniteElementSpace fes_u(&pmesh_cond, &fec_u, dim), fes_u_ext(&pmesh, &fec_u, dim);
    HYPRE_BigInt u_size = fes_u.GlobalTrueVSize();
    HYPRE_BigInt phi_size = fes_phi.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of u-unknowns: " << u_size << endl;
        cout << "Number of phi-unknowns: " << phi_size << endl;
    }
    ParGridFunction u_gf(&fes_u), u_gf_ext(&fes_u_ext), phi_gf(&fes_phi), phi_gf_cond(&fes_phi_cond); 
    ParGridFunction phi0_gf(&fes_phi), phi0_gf_cond(&fes_phi_cond), dphi0_gf(&fes_dphi), dphi0_gf_cond(&fes_dphi_cond);
    u_gf = 0.0; u_gf_ext = 0.0; phi_gf = 0.0; phi_gf_cond = 0.0; phi0_gf = 0.0; phi0_gf_cond = 0.0; dphi0_gf = 0.0; dphi0_gf_cond = 0.0;
    FunctionCoefficient rho_coeff(rho_func);
    FunctionCoefficient mu_coeff(mu_func);
    FunctionCoefficient lamb_coeff(lamb_func);
    FunctionCoefficient loading_coeff(loading_func);

    Array<int> ess_tdof_list;

    auto Earth_body_marker = Array<int>(pmesh.attributes.Max());
    Earth_body_marker = 1;
    Earth_body_marker[pmesh.attributes.Max() - 1] = 0;

    auto bdr_marker = Array<int>(pmesh.bdr_attributes.Max());
    bdr_marker = 0;
    bdr_marker[pmesh.bdr_attributes.Max() - 2] = 1;

    auto bdr_marker_cond = Array<int>(pmesh_cond.bdr_attributes.Max());
    bdr_marker_cond = 0;
    bdr_marker_cond[pmesh_cond.bdr_attributes.Max() - 1] = 1;

    //Compute the equilibrium state
    auto dtn = mfemElasticity::PoissonDtNOperator(MPI_COMM_WORLD, &fes_phi, deg);
    dtn.Assemble();
    auto DtN = dtn.RAP();

    ProductCoefficient rhs_coeff(-4.0 * M_PI * G_const, rho_coeff);
    ParLinearForm b0(&fes_phi);
    b0.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
    b0.Assemble();

    ParBilinearForm a0(&fes_phi);
    auto one = ConstantCoefficient(1.0);
    a0.AddDomainIntegrator(new DiffusionIntegrator(one));
    a0.Assemble();

    HypreParMatrix A0;
    Vector B0, Phi0;

    a0.FormLinearSystem(ess_tdof_list, phi0_gf, b0, A0, Phi0, B0);
    cout << "Size of linear system: " << A0.Height() << endl;

    auto S = SumOperator(&A0, 1.0, &DtN, 1.0, false, false);

    HypreBoomerAMG M(A0);

    auto solver0 = CGSolver(MPI_COMM_WORLD);
    solver0.SetOperator(S);
    solver0.SetPreconditioner(M);
    solver0.SetRelTol(rel_tol);
    solver0.SetMaxIter(3000);
    solver0.SetPrintLevel(0);
    solver0.Mult(B0, Phi0);

    a0.RecoverFEMSolution(Phi0, b0, phi0_gf);

    DiscreteLinearOperator Grad(&fes_phi_cond, &fes_dphi_cond);
    Grad.AddDomainInterpolator(new GradientInterpolator);
    Grad.Assemble();

    GridFunctionCoefficient phi0_coeff(&phi0_gf);
    pmesh_cond.Transfer(phi0_gf, phi0_gf_cond);
    Grad.Mult(phi0_gf_cond, dphi0_gf_cond);

    GradientGridFunctionCoefficient dphi0_coeff(&phi0_gf);
    VectorGridFunctionCoefficient dphi0_cond_coeff(&dphi0_gf_cond);
    ScalarVectorProductCoefficient dphi0_sig_cond_coeff(loading_coeff, dphi0_cond_coeff);
    
    if (myid == 0)
    {
        cout<<"Equilibrium state computed."<<endl;
    }

    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock << "parallel " << num_procs << " " << myid << "\n";
        sol_sock.precision(8);
        sol_sock << "solution\n" << pmesh << phi0_gf << flush;
    }

    //Coupled problem
    Vector U, Phi, U_ext, Phi_cond;
    u_gf.GetTrueDofs(U);
    phi_gf.GetTrueDofs(Phi);
    u_gf_ext.GetTrueDofs(U_ext);
    phi_gf_cond.GetTrueDofs(Phi_cond);
        
    ParLinearForm *b1(new ParLinearForm(&fes_u)); 
    b1->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(dphi0_sig_cond_coeff), bdr_marker_cond); 
    b1->Assemble();
    
    ParLinearForm *b2(new ParLinearForm(&fes_phi));
    b2->AddBoundaryIntegrator(new BoundaryLFIntegrator(loading_coeff), bdr_marker);
    b2->Assemble();
    
    ParBilinearForm *a11(new ParBilinearForm(&fes_u));
    ParBilinearForm *a22(new ParBilinearForm(&fes_phi));

    auto a12 = new ParMixedBilinearForm(&fes_phi_cond, &fes_u);

    auto a21 = new ParMixedBilinearForm(&fes_u_ext, &fes_phi);

    ConstantCoefficient c0(1.0 / (4.0 * M_PI * G_const));
    ProductCoefficient half_rho_coeff(0.5, rho_coeff), minus_half_rho_coeff(-0.5, rho_coeff), minus_rho_coeff(-1.0, rho_coeff);

    auto *a11_integ_0 = new ElasticityIntegrator(lamb_coeff, mu_coeff);
    auto *a11_integ_1 = new mfemElasticity::DomainVectorGradVectorIntegrator(dphi0_cond_coeff, half_rho_coeff);
    ScalarVectorProductCoefficient temp(minus_half_rho_coeff, dphi0_cond_coeff);
    auto *a11_integ_2 = new mfemElasticity::DomainVectorDivVectorIntegrator(temp);
    auto *a11_integ_1_t = new TransposeIntegrator(a11_integ_1, 0);
    auto *a11_integ_2_t = new TransposeIntegrator(a11_integ_2, 0);
    a11->AddDomainIntegrator(a11_integ_0);
    a11->AddDomainIntegrator(a11_integ_1);
    a11->AddDomainIntegrator(a11_integ_2);
    a11->AddDomainIntegrator(a11_integ_1_t);
    a11->AddDomainIntegrator(a11_integ_2_t);
    a11->Assemble();
    a11->Finalize();

    ParBilinearForm *a11_0(new ParBilinearForm(&fes_u));
    a11_0->AddDomainIntegrator(a11_integ_0);
    a11_0->Assemble();
    a11_0->Finalize();

    a22->AddDomainIntegrator(new DiffusionIntegrator(c0));
    a22->Assemble();
    a22->Finalize();
    
    a12->AddDomainIntegrator(new GradientIntegrator(rho_coeff));
    a12->Assemble();
    a12->Finalize();

    a21->AddDomainIntegrator(new TransposeIntegrator(new GradientIntegrator(rho_coeff)), 
                                                     Earth_body_marker);
    a21->Assemble();
    a21->Finalize();

    std::unique_ptr<HypreParMatrix> A11(a11->ParallelAssemble());
    std::unique_ptr<HypreParMatrix> A11_0(a11_0->ParallelAssemble());
    std::unique_ptr<HypreParMatrix> A22_0(a22->ParallelAssemble());
    auto A22 = SumOperator(A22_0.get(), 1.0, &DtN, 1.0 / (4.0 * M_PI * G_const), false, false);

    std::unique_ptr<HypreParMatrix> A12(a12->ParallelAssemble());
    std::unique_ptr<HypreParMatrix> A21(a21->ParallelAssemble());
    //TransposeOperator A21(A12.get());

    HypreBoomerAMG prec11;
    prec11.SetElasticityOptions(&fes_u);
    //HypreSmoother prec11;
    //prec11.SetType(HypreSmoother::l1GS);
    prec11.SetOperator(*A11_0);
    HypreBoomerAMG prec22(*A22_0);

    MINRESSolver solver1(MPI_COMM_WORLD);
    //CGSolver solver1(MPI_COMM_WORLD);
    solver1.SetRelTol(rel_tol);
    solver1.SetMaxIter(5000);
    solver1.SetOperator(*A11);
    solver1.SetPreconditioner(prec11);
    solver1.SetPrintLevel(0);

    mfemElasticity::RigidBodySolver rigid_solver(MPI_COMM_WORLD, &fes_u);
    rigid_solver.SetSolver(solver1);

    CGSolver solver2(MPI_COMM_WORLD);
    solver2.SetRelTol(rel_tol);
    solver2.SetMaxIter(5000);
    solver2.SetOperator(A22);
    solver2.SetPreconditioner(prec22);
    solver2.SetPrintLevel(0);

    int max_iter = 1000;
    int iter = 0;
    real_t rel_tol_coup = 1e-6;
    std::unique_ptr<HypreParVector> B1(b1->ParallelAssemble());
    std::unique_ptr<HypreParVector> B2(b2->ParallelAssemble());
    Vector b1_ext(B1->Size());
    Vector b2_ext(B2->Size());
    Vector Phi_temp(Phi.Size()), Phi_diff(Phi.Size());
    Phi_temp = 0.0; Phi_diff = 0.0;
    chrono.Clear();
    chrono.Start();
    for (int i = 0; i < max_iter; i++)
    {
        iter++;
        b1_ext = *B1;
        b2_ext = *B2;
        pmesh_cond.Transfer(phi_gf, phi_gf_cond);
        phi_gf_cond.GetTrueDofs(Phi_cond);

        A12->AddMult(Phi_cond, b1_ext, -1.0);
        rigid_solver.Mult(b1_ext, U);
        u_gf.SetFromTrueDofs(U);

        pmesh_cond.Transfer(u_gf, u_gf_ext);
        u_gf_ext.GetTrueDofs(U_ext);
        A21->AddMult(U_ext, b2_ext, -1.0);
        solver2.Mult(b2_ext, Phi_temp);
        Phi_diff = Phi_temp; Phi_diff -= Phi;

        real_t local_num = Phi_diff * Phi_diff;
        real_t global_num = 0.0;
        MPI_Allreduce(&local_num, &global_num, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        real_t local_den = Phi_temp * Phi_temp;
        real_t global_den = 0.0;
        MPI_Allreduce(&local_den, &global_den, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        real_t res = sqrt(global_num) / sqrt(global_den);
        Phi = Phi_temp;
        phi_gf.SetFromTrueDofs(Phi);

        if (myid == 0)
        {
            cout << "Iteration " << iter << ", residual = " << res << endl;
        }

        if (res < rel_tol_coup)
        {
            chrono.Stop();
            if (myid == 0)
            {
                cout << "Converged at iteration " << iter << endl;
                cout << "Time = " << chrono.RealTime() << " s" << endl;
            }
            break;
        }

        if (i == max_iter - 1)
        {
            chrono.Stop();
            if (myid == 0)
            {
                cout << "Did not converge in " << max_iter << " iterations." << endl;
                cout << "Time = " << chrono.RealTime() << " s" << endl;
            }
        }
    }

    pmesh_cond.Transfer(phi_gf, phi_gf_cond);

    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream u_sock(vishost, visport);
        u_sock << "parallel " << num_procs << " " << myid << "\n";
        u_sock.precision(8);
        u_sock << "solution\n" << pmesh_cond << u_gf << "window_title 'Deformation'" << endl;
        socketstream phi_sock(vishost, visport);
        phi_sock << "parallel " << num_procs << " " << myid << "\n";
        phi_sock.precision(8);
        phi_sock << "solution\n" << pmesh_cond << phi_gf_cond << "window_title 'Gravity potential perturbation'" << endl;
    }

    delete b1;
    delete b2;
    delete a11; 
    delete a12;
    delete a21;
    delete a22;

    return 0;
}

real_t rho_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    if (r > R_const){
        return 0.0;
    } else{
        real_t r_norm = r / R_const;
        //real_t theta = acos(coord[2] / r); // polar angle
        //real_t phi = atan2(coord[1], coord[0]); // azimuthal angle
        real_t rho_surface = 2.6e3; 
        real_t rho_center = 1.3e4;   
        real_t base_rho =  rho_center + (rho_surface - rho_center) * r_norm;
        return base_rho;
    }
}

real_t mu_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t r_norm = r / 6371e3;
    real_t theta = acos(coord[2] / r); // polar angle
    real_t phi = atan2(coord[1], coord[0]); // azimuthal angle
    real_t mu_surface = 70e9;  // Pa
    real_t mu_center = 140e9;    // Pa
    real_t base_mu =  mu_center + (mu_surface - mu_center) * r_norm;
    real_t polar_perturb = 0.015 * (1.0 + cos(2.0 * theta));
    real_t azimuthal_perturb = 0.05 * sin(2.0 * phi);
    return base_mu * (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
}

real_t lamb_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t r_norm = r / 6371e3;
    real_t theta = acos(coord[2] / r);
    real_t phi = atan2(coord[1], coord[0]);
    real_t lamb_surface = 100e9;
    real_t lamb_center = 300e9;   
    real_t base_lamb = lamb_center + (lamb_surface - lamb_center) * r_norm;
    real_t polar_perturb = 0.015 * (1.0 + cos(2.0 * theta));
    real_t azimuthal_perturb = 0.05 * sin(2.0 * phi);
    return base_lamb * (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
}

real_t loading_func(const Vector &coord)
{
    real_t factor = 1e-1;
    real_t r = coord.Norml2();
    real_t theta = acos(coord[2] / r);
    real_t phi = atan2(coord[1], coord[0]);
    // Max loading at poles (glaciers): e.g., 10 MPa (~1 km ice)
    const real_t polar_load = 10e6;
    // Equatorial loading (oceans): e.g., 1 MPa (~100 m water depth)
    const real_t equator_load = 1e6;
    real_t base_load = (equator_load + polar_load) / 2.0 + (polar_load - equator_load) / 2.0 * cos(2.0 * theta);
    real_t azimuthal_perturb = 0.2 * sin(2.0 * phi);
    return -base_load * (1.0 + azimuthal_perturb) * factor;
}

