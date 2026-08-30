// ============================================================================
// viscoelasticity.cpp
//
// Quasi-static Maxwell viscoelasticity built on top of the elastic problem
// abstraction in elastic.hpp.
//
// Constitutive model (isotropic Maxwell, internal-variable form):
//
//     T = kappa (div u) I + 2 mu (d(u) - m),        dm/dt = (d(u) - m) / tau,
//
// where d(u) is the deviatoric strain and m a trace-free symmetric tensor
// internal variable. The ODE state consists of m alone: the displacement is
// slaved to it through the quasi-static solve
//
//     a(u, v) = F_ext(v) + int 2 mu m : d(v) dx        for all v,
//
// so each right-hand-side evaluation performs one elastic solve. This is the
// structure of the internal-variable formulation used in Yu et al. (2025) for
// glacial isostatic adjustment.
//
// Discretisation choices embodied here:
//
//  - The internal variable lives on a discontinuous (L2) nodal space of
//    trace-free symmetric tensors, with the component conventions of
//    mfemElasticity's TraceFreeSymmetricMatrixIndex.
//
//  - The coupling operator B, with B[i][j] = int Phi_i : d(phi_j) dx, is
//    assembled once and is purely geometric (unit coefficient). All material
//    weighting -- here the factor 2 mu -- is applied pointwise at the
//    internal-variable nodes by the model class. This keeps material data in
//    exactly one place per layer and makes an anisotropic relaxation law a
//    pointwise change only.
//
//  - The strain map u -> d(u) defaults to nodal interpolation
//    (DeviatoricStrainInterpolator). It sits behind a virtual so that the
//    Galerkin-consistent projection M^{-1} B can be swapped in later; the
//    projection choice is what makes the discrete adjoint exactly the
//    forward code run backwards.
//
// Time stepping:
//
//  - Mult() provides the standard explicit right-hand side, usable with any
//    of MFEM's explicit ODE solvers. Stability then requires dt <~ tau_min:
//    this is the restriction that makes rate-form explicit stepping
//    unattractive for realistic viscosity contrasts.
//
//  - ExponentialStep() advances m by the exact solution of the relaxation
//    equation with the deviatoric strain frozen at the step start
//    (exponential time differencing). One elastic solve per step, no
//    stability restriction from tau; first-order accurate in the coupling.
//    ExponentialEulerSolver wraps it in the ODESolver interface so drivers
//    are indifferent to the choice.
//
//  - ImplicitSolve() documents the backward-Euler path: eliminating m^{n+1}
//    yields an elastic solve with the deviatoric modulus scaled by
//    1/(1 + dt/tau), which requires the elastic problem to implement the
//    ScalableDeviatoricStiffness capability. It is gated but not yet wired.
//
// Sample runs:
//    ./viscoelasticity -m ../data/star.mesh -o 2 -r 2
//    ./viscoelasticity -m ../data/star.mesh -o 2 -r 2 -s 4 -n 200
//    ./viscoelasticity -m ../data/beam-quad.mesh -p 1 -o 2 -r 1 -tau 0.5
// ============================================================================

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

#include "elastic.hpp"

/*----------------------------------------------------------------------------
  Viscoelastic base class
----------------------------------------------------------------------------*/

/**
 * @brief Base class for quasi-static viscoelastic evolution operators.
 *
 * The ODE state is the internal-variable vector; its layout is defined by the
 * concrete model. For single-mechanism models it is the vdof vector of a
 * GridFunction on InternalVariableSpace() (Ordering::byNODES, so component c
 * occupies the contiguous block [c*nd, (c+1)*nd) with nd scalar dofs);
 * multi-mechanism models stack such blocks and reset height/width
 * accordingly.
 *
 * The class owns the plumbing shared by all models: the tensor and scalar
 * companion spaces, the geometric coupling operator B, the strain map, and
 * the elastic-update pipeline. Concrete models supply three pointwise
 * ingredients:
 *
 *   - AddInternalForces(m): push the effective force (e.g. B^T (2 mu m))
 *     into the elastic problem through AddForce();
 *   - Rate(m, d, k): the pointwise rate k = f(m, d);
 *   - LocalUpdate(dt, m, d): the exact (or approximate) relaxation update
 *     with d held fixed, used by the exponential integrator.
 *
 * Observation pattern: after an accepted step the displacement held by the
 * elastic problem corresponds to the last internal solve, not to the accepted
 * state. Drivers wanting consistent (u, m) output call SolveElastic(m, t)
 * followed by SyncFields(m) before saving.
 */
class ViscoelasticOperator : public mfem::TimeDependentOperator {
 protected:
  QuasiStaticLinearElasticProblem& _problem;

  std::unique_ptr<mfem::FiniteElementCollection> _dfec;
  std::unique_ptr<mfem::FiniteElementSpace> _dfes;  ///< Tensor space for m.
  std::unique_ptr<mfem::FiniteElementSpace> _sfes;  ///< Scalar companion.

  std::unique_ptr<mfem::MixedBilinearForm> _B;      ///< Geometric coupling.
  std::unique_ptr<mfem::DiscreteLinearOperator> _D; ///< Nodal strain map.

  mutable mfem::Vector _d;      ///< Scratch: deviatoric strain at m-nodes.
  mutable mfem::Vector _force;  ///< Scratch: dual vector on displacements.

  mfem::GridFunction _m_gf;  ///< Output view of the internal variable.

  /// Number of scalar dofs per tensor component.
  int ScalarDofs() const { return _sfes->GetVSize(); }

  /// Number of stored tensor components: dim (dim + 1) / 2 - 1.
  int Components() const { return _dfes->GetVDim(); }

  /// Nodal values of a scalar coefficient at the internal-variable nodes.
  void ProjectToInternalNodes(mfem::Coefficient& c, mfem::Vector& nodal) {
    mfem::GridFunction s(_sfes.get());
    s.ProjectCoefficient(c);
    nodal = s;
  }

  /// y = w o x, with the nodal field w applied to every tensor component.
  void ApplyNodalWeight(const mfem::Vector& w, const mfem::Vector& x,
                        mfem::Vector& y) const {
    const int nd = ScalarDofs();
    const int nc = Components();
    y.SetSize(x.Size());
    for (int c = 0; c < nc; c++) {
      const int o = c * nd;
      for (int p = 0; p < nd; p++) {
        y[o + p] = w[p] * x[o + p];
      }
    }
  }

  /// Push the dual vector B^T zeta into the elastic problem's forcing. The
  /// argument is a stress-like tensor field (e.g. zeta = 2 mu m) in the
  /// internal-variable layout.
  void AddCoupledForce(const mfem::Vector& zeta) const {
    _B->MultTranspose(zeta, _force);
    _problem.AddForce(_force);
  }

  /// Elastic update at the given time: reset forcing, add internal forces,
  /// solve. Leaves the displacement in the elastic problem.
  bool ElasticUpdate(const mfem::Vector& m, mfem::real_t time) const {
    _problem.AssembleForce(time);
    AddInternalForces(m);
    return _problem.Solve();
  }

  /// Deviatoric strain of u at the internal-variable nodes. The default is
  /// nodal interpolation; virtual so that the Galerkin-consistent projection
  /// M^{-1} B u can be substituted (the choice that makes the discrete
  /// adjoint exactly time-reversed forward code).
  virtual void ComputeDeviatoricStrain(const mfem::GridFunction& u,
                                       mfem::Vector& d) const {
    _D->Mult(u, d);
  }

  // --- Model interface -----------------------------------------------------

  /// Add the effective internal-variable forces to the elastic problem
  /// (zero or more AddForce() calls).
  virtual void AddInternalForces(const mfem::Vector& m) const = 0;

  /// Pointwise rate k = f(m, d) at the internal-variable nodes.
  virtual void Rate(const mfem::Vector& m, const mfem::Vector& d,
                    mfem::Vector& k) const = 0;

  /// Advance m over dt with the deviatoric strain d held fixed, using the
  /// exact relaxation solution where available.
  virtual void LocalUpdate(mfem::real_t dt, mfem::Vector& m,
                           const mfem::Vector& d) const = 0;

 public:
  /**
   * @brief Construct the shared viscoelastic plumbing.
   *
   * @param problem The elastic problem; must outlive this operator.
   * @param order   Polynomial order of the (L2) internal-variable space.
   */
  ViscoelasticOperator(QuasiStaticLinearElasticProblem& problem, int order)
      : mfem::TimeDependentOperator(0, 0.0,
                                    mfem::TimeDependentOperator::EXPLICIT),
        _problem{problem} {
    using namespace mfem;

    FiniteElementSpace& ufes = _problem.DisplacementSpace();
    Mesh* mesh = ufes.GetMesh();
    const int dim = mesh->Dimension();
    const int tfdim = dim * (dim + 1) / 2 - 1;

    _dfec = std::make_unique<L2_FECollection>(order, dim);
    _dfes = std::make_unique<FiniteElementSpace>(mesh, _dfec.get(), tfdim);
    _sfes = std::make_unique<FiniteElementSpace>(mesh, _dfec.get(), 1);

    // Geometric coupling operator: no coefficient, by design. All material
    // weighting is applied pointwise by the model classes.
    _B = std::make_unique<MixedBilinearForm>(&ufes, _dfes.get());
    _B->AddDomainIntegrator(
        new mfemElasticity::
            DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator());
    _B->Assemble();
    _B->Finalize();

    // Nodal strain map u -> d(u).
    _D = std::make_unique<DiscreteLinearOperator>(&ufes, _dfes.get());
    _D->AddDomainInterpolator(
        new mfemElasticity::DeviatoricStrainInterpolator());
    _D->Assemble();
    _D->Finalize();

    _d.SetSize(_dfes->GetVSize());
    _force.SetSize(ufes.GetVSize());
    _m_gf.SetSpace(_dfes.get());
    _m_gf = 0.0;

    height = width = _dfes->GetVSize();
  }

  /// The space on which the internal variable is defined.
  mfem::FiniteElementSpace& InternalVariableSpace() { return *_dfes; }

  /// Output view of the internal variable; refreshed by SyncFields().
  const mfem::GridFunction& InternalVariable() const { return _m_gf; }

  /// Explicit right-hand side: one elastic solve, then the pointwise rate.
  void Mult(const mfem::Vector& m, mfem::Vector& k) const override {
    MFEM_VERIFY(ElasticUpdate(m, GetTime()),
                "ViscoelasticOperator::Mult: elastic solve failed.");
    ComputeDeviatoricStrain(_problem.Displacement(), _d);
    Rate(m, _d, k);
  }

  /**
   * @brief Backward-Euler style implicit stage solve (not yet wired).
   *
   * Eliminating m^{n+1} = (m^n + (dt/tau) d^{n+1}) / (1 + dt/tau) from the
   * equilibrium equation leaves an elastic solve in which the deviatoric
   * modulus is scaled pointwise by 1/(1 + dt/tau), with the right-hand side
   * carrying B^T(2 mu m^n / (1 + dt/tau)). That solve requires the elastic
   * problem to implement ScalableDeviatoricStiffness.
   */
  void ImplicitSolve(const mfem::real_t /*dt*/, const mfem::Vector& /*m*/,
                     mfem::Vector& /*k*/) override {
    auto* scalable = dynamic_cast<ScalableDeviatoricStiffness*>(&_problem);
    if (!scalable) {
      MFEM_ABORT(
          "ViscoelasticOperator::ImplicitSolve: the elastic problem does not "
          "implement ScalableDeviatoricStiffness. Use an explicit solver or "
          "the exponential integrator instead.");
    }
    MFEM_ABORT(
        "ViscoelasticOperator::ImplicitSolve: implicit stepping is not yet "
        "implemented.");
  }

  /**
   * @brief One step of the first-order exponential (ETD) integrator.
   *
   * Solves the elastic problem at the step start, freezes the deviatoric
   * strain, and applies the exact relaxation update through LocalUpdate().
   * Unconditionally stable with respect to the relaxation times, at first
   * order in the elastic coupling.
   */
  void ExponentialStep(mfem::Vector& m, mfem::real_t& time, mfem::real_t dt) {
    SetTime(time);
    MFEM_VERIFY(ElasticUpdate(m, time),
                "ViscoelasticOperator::ExponentialStep: elastic solve failed.");
    ComputeDeviatoricStrain(_problem.Displacement(), _d);
    LocalUpdate(dt, m, _d);
    time += dt;
    SetTime(time);
  }

  /// Re-solve the elastic problem for the given state and time, e.g. to
  /// obtain a displacement consistent with (m, t) for observation.
  bool SolveElastic(const mfem::Vector& m, mfem::real_t time) {
    SetTime(time);
    return ElasticUpdate(m, time);
  }

  /// Register the problem's fields plus the internal variable.
  virtual void RegisterFields(mfem::DataCollection& dc) {
    _problem.RegisterFields(dc);
    dc.RegisterField("internal_variable", &_m_gf);
  }

  /// Copy the state into the registered output GridFunction.
  virtual void SyncFields(const mfem::Vector& m) { _m_gf.SetFromTrueDofs(m); }
};

/*----------------------------------------------------------------------------
  Maxwell model
----------------------------------------------------------------------------*/

/**
 * @brief Isotropic Maxwell viscoelasticity: dm/dt = (d(u) - m) / tau.
 *
 * The model owns its material data: the shear modulus mu entering the
 * effective force B^T(2 mu m), and the relaxation time tau = eta / mu. Both
 * are sampled once at the internal-variable nodes.
 *
 * INVARIANT: the shear modulus passed here must be the same mu used in the
 * elastic problem's stiffness. The elastic and viscoelastic layers own their
 * material data separately by design; this is the documented consistency
 * condition that keeps them compatible.
 */
class MaxwellViscoelasticOperator : public ViscoelasticOperator {
 private:
  mfem::Vector _two_mu;  ///< Nodal values of 2 mu.
  mfem::Vector _itau;    ///< Nodal values of 1 / tau.
  mutable mfem::Vector _zeta;  ///< Scratch: 2 mu m.

 public:
  MaxwellViscoelasticOperator(QuasiStaticLinearElasticProblem& problem,
                              int order, mfem::Coefficient& mu,
                              mfem::Coefficient& tau)
      : ViscoelasticOperator(problem, order) {
    ProjectToInternalNodes(mu, _two_mu);
    _two_mu *= 2.0;
    ProjectToInternalNodes(tau, _itau);
    for (int p = 0; p < _itau.Size(); p++) {
      MFEM_VERIFY(_itau[p] > 0.0,
                  "MaxwellViscoelasticOperator: tau must be positive.");
      _itau[p] = 1.0 / _itau[p];
    }
    _zeta.SetSize(Height());
  }

  /// Smallest relaxation time over the internal-variable nodes; the explicit
  /// stability limit is a small multiple of this.
  mfem::real_t MinRelaxationTime() const { return 1.0 / _itau.Max(); }

 protected:
  void AddInternalForces(const mfem::Vector& m) const override {
    ApplyNodalWeight(_two_mu, m, _zeta);
    AddCoupledForce(_zeta);
  }

  void Rate(const mfem::Vector& m, const mfem::Vector& d,
            mfem::Vector& k) const override {
    const int nd = ScalarDofs();
    const int nc = Components();
    k.SetSize(m.Size());
    for (int c = 0; c < nc; c++) {
      const int o = c * nd;
      for (int p = 0; p < nd; p++) {
        k[o + p] = (d[o + p] - m[o + p]) * _itau[p];
      }
    }
  }

  void LocalUpdate(mfem::real_t dt, mfem::Vector& m,
                   const mfem::Vector& d) const override {
    // Exact solution of dm/dt = (d - m)/tau with d frozen:
    //   m <- exp(-dt/tau) m + (1 - exp(-dt/tau)) d.
    const int nd = ScalarDofs();
    const int nc = Components();
    for (int p = 0; p < nd; p++) {
      const mfem::real_t a = std::exp(-dt * _itau[p]);
      const mfem::real_t b = 1.0 - a;
      for (int c = 0; c < nc; c++) {
        const int i = c * nd + p;
        m[i] = a * m[i] + b * d[i];
      }
    }
  }
};

/*----------------------------------------------------------------------------
  Exponential integrator in the ODESolver interface
----------------------------------------------------------------------------*/

/**
 * @brief ODESolver wrapper around ViscoelasticOperator::ExponentialStep(),
 * so that drivers can switch between Runge-Kutta and exponential stepping
 * without changing the time loop.
 */
class ExponentialEulerSolver : public mfem::ODESolver {
 private:
  ViscoelasticOperator* _op = nullptr;

 public:
  void Init(mfem::TimeDependentOperator& f_) override {
    mfem::ODESolver::Init(f_);
    _op = dynamic_cast<ViscoelasticOperator*>(&f_);
    MFEM_VERIFY(_op,
                "ExponentialEulerSolver requires a ViscoelasticOperator.");
  }

  void Step(mfem::Vector& x, mfem::real_t& t, mfem::real_t& dt) override {
    _op->ExponentialStep(x, t, dt);
  }
};

/*----------------------------------------------------------------------------
  Driver
----------------------------------------------------------------------------*/

using namespace std;
using namespace mfem;

int main(int argc, char* argv[]) {
  // Set the default options.
  const char* mesh_file = "../data/star.mesh";
  int order = 2;
  int m_order = -1;  // internal-variable order; < 0 means "same as order"
  int ref_levels = 1;
  int problem_type = 0;
  int solver_type = 0;  // 0 = exponential; 1 = FE; 2 = RK2; 3 = RK3; 4 = RK4
  real_t t_final = 5.0;
  int n_steps = 50;
  real_t tau0 = 1.0;
  bool paraview = true;
  bool visualization = true;

  // Read in command line options and process.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order for the displacement.");
  args.AddOption(&m_order, "-mo", "--m-order",
                 "Order of the internal-variable space (< 0: same as -o).");
  args.AddOption(&ref_levels, "-r", "--refinement",
                 "Number of uniform mesh refinements.");
  args.AddOption(&problem_type, "-p", "--problem",
                 "Problem type: 0 = pure traction (any mesh), 1 = clamped "
                 "(needs two boundary attributes, e.g. beam-quad.mesh).");
  args.AddOption(&solver_type, "-s", "--solver",
                 "Time integrator: 0 = exponential Euler, 1 = forward Euler, "
                 "2 = RK2, 3 = RK3 SSP, 4 = RK4.");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
  args.AddOption(&n_steps, "-n", "--n-steps", "Number of time steps.");
  args.AddOption(&tau0, "-tau", "--relaxation-time",
                 "Maxwell relaxation time tau = eta / mu.");
  args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                 "Save time slices to a ParaView data collection.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Send the final displacement to a running GLVis server.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);
  if (m_order < 0) {
    m_order = order;
  }

  // Read in the mesh and refine if requested.
  Mesh mesh(mesh_file, 1, 1);
  const int dim = mesh.Dimension();
  for (int l = 0; l < ref_levels; l++) {
    mesh.UniformRefinement();
  }

  // Construct the requested elastic problem behind the common interface.
  unique_ptr<QuasiStaticLinearElasticProblem> problem;
  if (problem_type == 0) {
    problem = make_unique<TractionProblem>(&mesh, order);
  } else if (problem_type == 1) {
    problem = make_unique<ClampedProblem>(&mesh, order);
  } else {
    cerr << "Unknown problem type: " << problem_type << "\n";
    return 1;
  }

  // Material parameters of the viscoelastic model. The shear modulus MUST
  // equal the one inside the elastic problem (both are 1 here); the elastic
  // and viscoelastic layers own their material data separately by design,
  // and this equality is the documented invariant that keeps them
  // consistent.
  ConstantCoefficient mu(1.0);
  ConstantCoefficient tau(tau0);

  MaxwellViscoelasticOperator visco(*problem, m_order, mu, tau);
  cout << "Displacement unknowns:      "
       << problem->DisplacementSpace().GetTrueVSize() << "\n"
       << "Internal-variable unknowns: " << visco.Height() << "\n";

  // Select the time integrator.
  unique_ptr<ODESolver> ode;
  switch (solver_type) {
    case 0:
      ode = make_unique<ExponentialEulerSolver>();
      break;
    case 1:
      ode = make_unique<ForwardEulerSolver>();
      break;
    case 2:
      ode = make_unique<RK2Solver>(0.5);
      break;
    case 3:
      ode = make_unique<RK3SSPSolver>();
      break;
    case 4:
      ode = make_unique<RK4Solver>();
      break;
    default:
      cerr << "Unknown solver type: " << solver_type << "\n";
      return 1;
  }
  ode->Init(visco);

  real_t t = 0.0;
  real_t dt = t_final / n_steps;
  Vector m(visco.Height());
  m = 0.0;

  if (solver_type != 0 && dt > 2.5 * visco.MinRelaxationTime()) {
    cout << "Warning: dt = " << dt
         << " exceeds the explicit stability limit of roughly 2.8 tau_min = "
         << 2.8 * visco.MinRelaxationTime()
         << ". Expect blow-up; use -s 0 or reduce the step.\n";
  }

  // Time slices are written through the fields the operator registers.
  ParaViewDataCollection dc("viscoelastic", &mesh);
  if (paraview) {
    dc.SetPrefixPath("ParaView");
    dc.SetLevelsOfDetail(order);
    dc.SetDataFormat(VTKFormat::BINARY);
    dc.SetHighOrderOutput(true);
    visco.RegisterFields(dc);
  }

  // Initial state: relaxed internal variable, elastic response at t = 0.
  if (!visco.SolveElastic(m, t)) {
    cerr << "Elastic solve failed at t = " << t << "\n";
    return 2;
  }
  visco.SyncFields(m);
  if (paraview) {
    dc.SetCycle(0);
    dc.SetTime(t);
    dc.Save();
  }

  // March through time. After each accepted step the displacement is
  // refreshed so that (u, m) are consistent at time t for output; skip the
  // refresh if only the final state is of interest.
  for (int step = 1; step <= n_steps; step++) {
    ode->Step(m, t, dt);

    if (!visco.SolveElastic(m, t)) {
      cerr << "Elastic solve failed at t = " << t << "\n";
      return 2;
    }
    visco.SyncFields(m);
    cout << "step " << step << ", t = " << t << ", ||m||_2 = " << m.Norml2()
         << "\n";

    if (paraview) {
      dc.SetCycle(step);
      dc.SetTime(t);
      dc.Save();
    }
  }

  // Save the final state in MFEM's native format.
  {
    ofstream mesh_ofs("refined.mesh");
    mesh_ofs.precision(8);
    mesh.Print(mesh_ofs);
    ofstream sol_ofs("sol.gf");
    sol_ofs.precision(8);
    problem->Displacement().Save(sol_ofs);
  }

  // Visualise if glvis is open.
  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n";
    mesh.Print(sol_sock);
    problem->Displacement().Save(sol_sock);
    sol_sock << flush;
    if (dim == 2) {
      sol_sock << "keys Rjlvvvvvmm\n" << flush;
    } else {
      sol_sock << "keys m\n" << flush;
    }
  }

  return 0;
}
