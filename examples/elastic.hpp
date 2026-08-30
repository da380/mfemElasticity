// ============================================================================
// elastic.hpp
//
// Quasi-static linear elastic problems: the abstract interface, a partial
// implementation owning the shared bookkeeping, and two concrete problems
// (pure traction, clamped). Extracted from the single-file example so that
// the viscoelastic layer can be built on top of it; see
// quasi_static_elasticity.cpp for a driver exercising these classes alone,
// and viscoelasticity.cpp for the time-dependent layer.
// ============================================================================

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "mfemElasticity.hpp"

/*----------------------------------------------------------------------------
  Optional capabilities
----------------------------------------------------------------------------*/

/**
 * @brief Optional capability interface for elastic problems whose deviatoric
 * stiffness can be scaled pointwise.
 *
 * Implicit time stepping of Maxwell-type viscoelastic models leads, after
 * eliminating the internal variable, to a modified elastic solve in which the
 * deviatoric modulus is scaled by a pointwise factor s(x) in (0, 1]; backward
 * Euler, for instance, gives s = 1/(1 + dt/tau). Problems able to reassemble
 * their operator in the split form
 *
 *     kappa div-div  +  s 2 mu dev-dev
 *
 * advertise this by additionally deriving from this interface.
 * ViscoelasticOperator::ImplicitSolve() requires it; the explicit and
 * exponential-integrator paths do not.
 */
class ScalableDeviatoricStiffness {
 public:
  virtual ~ScalableDeviatoricStiffness() = default;

  /// Apply the pointwise deviatoric scale s, reassembling the operator and
  /// preconditioner as needed.
  virtual void SetDeviatoricScale(mfem::Coefficient& s) = 0;

  /// Restore the unscaled operator.
  virtual void ClearDeviatoricScale() = 0;
};

/*----------------------------------------------------------------------------
  Interface
----------------------------------------------------------------------------*/

/**
 * @brief Abstract interface for quasi-static linear elastic problems.
 *
 * The interface fixes the forcing protocol used at each evaluation time:
 *
 *     AssembleForce(t);   // reset the forcing to the external loads at t
 *     AddForce(f);        // superpose zero or more dual vectors
 *     Solve();            // displacement <- K^{-1}(external + increments)
 *
 * Contract:
 *
 *  - AssembleForce(t) brings *all* time-dependent data to time t: load
 *    coefficients are advanced via SetTime() and the external linear form is
 *    reassembled; Dirichlet boundary values (where present) are refreshed in
 *    the solution vector; and previously added increments are cleared. The
 *    call is idempotent at fixed t and cheap enough to be made at every
 *    stage of a time integrator.
 *
 *  - AddForce(f) accumulates a dual vector given in the same layout as a
 *    LinearForm on DisplacementSpace() prior to FormLinearSystem, that is,
 *    vdofs of size GetVSize(). In parallel this becomes the local L-dof
 *    layout, with the prolongation transpose applied once inside Solve();
 *    callers never handle true dofs. If the problem carries additional
 *    unknowns (e.g. a gravitational potential), the vector still refers to
 *    the displacement space alone and is scattered internally.
 *
 *  - Solve() uses a persistent operator and preconditioner assembled at
 *    construction; only the right-hand side is rebuilt per call. It returns
 *    false if the linear solver did not converge, leaving state intact so
 *    that a driver can react.
 *
 *  - Displacement() exposes the primal solution; callers must treat it as
 *    read-only.
 *
 * Note that the protocol bakes in superposition of forces, i.e. linearity of
 * the elastic operator: "Linear" in the class name is part of the contract,
 * not merely of the current implementations.
 */
class QuasiStaticLinearElasticProblem {
 public:
  virtual ~QuasiStaticLinearElasticProblem() = default;

  /// The (vector) finite element space on which displacements are defined.
  virtual mfem::FiniteElementSpace& DisplacementSpace() = 0;

  /// Bring all time-dependent data to time t and reset the forcing.
  virtual void AssembleForce(mfem::real_t t) = 0;

  /// Superpose a dual vector (LinearForm layout) on the current forcing.
  virtual void AddForce(const mfem::Vector& f) = 0;

  /// Solve for the displacement; returns false on solver failure.
  virtual bool Solve() = 0;

  /// Read-only access to the current displacement solution.
  virtual const mfem::GridFunction& Displacement() const = 0;

  /// Register output fields (displacement, and whatever else the problem
  /// carries) with a DataCollection owned by the driver.
  virtual void RegisterFields(mfem::DataCollection& dc) = 0;
};

/*----------------------------------------------------------------------------
  Partial implementation
----------------------------------------------------------------------------*/

/**
 * @brief Partial implementation owning the bookkeeping shared by concrete
 * elastic problems.
 *
 * Owns the finite element space, the persistent bilinear form and external
 * linear form, the registry of time-dependent coefficients, the increment
 * accumulator, and the Solve() skeleton.
 *
 * A derived class is expected, within its constructor, to:
 *   1. add its integrators to *_a and *_b;
 *   2. fill _ess_tdof_list if it imposes essential boundary conditions, and
 *      override UpdateBoundaryValues() to (re)project the Dirichlet data;
 *   3. call FinalizeSystem(), and then construct its solver and
 *      preconditioner on the reduced matrix _A;
 * and to implement SolveLinearSystem() with its solver of choice.
 */
class ElasticProblemBase : public QuasiStaticLinearElasticProblem {
 protected:
  /// Non-owning pointer to the mesh (memory managed externally).
  mfem::Mesh* _mesh;

  std::unique_ptr<mfem::FiniteElementCollection> _fec;
  std::unique_ptr<mfem::FiniteElementSpace> _fes;

  std::unique_ptr<mfem::BilinearForm> _a;  ///< Stiffness; assembled once.
  std::unique_ptr<mfem::LinearForm> _b;    ///< External load; reassembled.
  mfem::Vector _increment;  ///< Accumulated AddForce() contributions.
  mfem::Vector _rhs;        ///< Scratch: external load + increments.

  mfem::GridFunction _u;            ///< Displacement solution.
  mfem::Array<int> _ess_tdof_list;  ///< Empty unless the derived class
                                    ///< imposes essential conditions.

  mfem::SparseMatrix _A;  ///< Reference to the reduced system matrix.
  mfem::Vector _X, _B;    ///< Reduced solution and right-hand side.

  mfem::real_t _t = 0.0;  ///< Time of the most recent AssembleForce().

  std::vector<mfem::Coefficient*> _td_coefs;         // non-owning
  std::vector<mfem::VectorCoefficient*> _td_vcoefs;  // non-owning

  /// Register coefficients whose SetTime() must be called by
  /// AssembleForce(). Lifetimes are managed by the derived class.
  void RegisterTimeDependent(mfem::Coefficient& c) { _td_coefs.push_back(&c); }
  void RegisterTimeDependent(mfem::VectorCoefficient& c) {
    _td_vcoefs.push_back(&c);
  }

  /// Hook refreshing Dirichlet values in the solution vector at time t.
  /// Called by AssembleForce() after the coefficients have been advanced.
  /// Default: nothing (no essential conditions).
  virtual void UpdateBoundaryValues(mfem::real_t /*t*/) {}

  /// Assemble the stiffness and eliminate the essential dofs once. To be
  /// called at the end of the derived constructor, *before* building a
  /// preconditioner on _A.
  void FinalizeSystem() {
    _a->Assemble();
    _a->FormSystemMatrix(_ess_tdof_list, _A);
  }

  /// Solve _A X = B with the derived class's solver; return convergence.
  virtual bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X) = 0;

  /// Relative tolerance for the linear solves, measured against the
  /// right-hand side (see SetWarmStartTolerance).
  mfem::real_t _rel_tol = 1e-12;

  /// Make a warm-started solve converge to the same target as a cold one.
  ///
  /// MFEM's iterative solvers apply the relative tolerance to the *initial*
  /// residual. With iterative_mode on, the initial residual of a re-solve at
  /// a slowly changing right-hand side is already tiny, so the relative
  /// target becomes unreachable and the solver runs to max_iter. The remedy
  /// is an absolute tolerance equal to what a cold start (x0 = 0, r0 = B)
  /// would have used, namely rel * sqrt((M B, B)) in the norm induced by the
  /// preconditioner M -- exactly the quantity the solver compares against.
  ///
  /// Returns false when B vanishes, in which case the solution is zero and
  /// no solve is needed (an absolute tolerance of zero could not be met from
  /// a non-zero warm start).
  bool SetWarmStartTolerance(mfem::IterativeSolver& solver, mfem::Solver& prec,
                             const mfem::Vector& B) const {
    mfem::Vector z(B.Size());
    prec.Mult(B, z);
    const mfem::real_t nom = mfem::InnerProduct(B, z);
    if (!(nom > 0.0)) {
      return false;
    }
    solver.SetAbsTol(_rel_tol * std::sqrt(nom));
    return true;
  }

 public:
  ElasticProblemBase(mfem::Mesh* mesh, int order)
      : _mesh{mesh},
        _fec{std::make_unique<mfem::H1_FECollection>(order,
                                                     mesh->Dimension())},
        _fes{std::make_unique<mfem::FiniteElementSpace>(mesh, _fec.get(),
                                                        mesh->Dimension())} {
    _a = std::make_unique<mfem::BilinearForm>(_fes.get());
    _b = std::make_unique<mfem::LinearForm>(_fes.get());
    _increment.SetSize(_fes->GetVSize());
    _increment = 0.0;
    _u.SetSpace(_fes.get());
    _u = 0.0;
  }

  mfem::FiniteElementSpace& DisplacementSpace() override { return *_fes; }

  const mfem::GridFunction& Displacement() const override { return _u; }

  /// Time of the most recent AssembleForce() call.
  mfem::real_t Time() const { return _t; }

  void AssembleForce(mfem::real_t t) override {
    _t = t;
    for (auto* c : _td_coefs) {
      c->SetTime(t);
    }
    for (auto* c : _td_vcoefs) {
      c->SetTime(t);
    }
    // LinearForm::Assemble() zeroes before assembling, so repeated calls at
    // fixed t are idempotent.
    _b->Assemble();
    _increment = 0.0;
    UpdateBoundaryValues(t);
  }

  void AddForce(const mfem::Vector& f) override {
    MFEM_VERIFY(f.Size() == _increment.Size(),
                "AddForce: expected a dual vector in the vdof layout of "
                "DisplacementSpace().");
    _increment += f;
  }

  bool Solve() override {
    MFEM_VERIFY(_A.Height() > 0,
                "Solve: FinalizeSystem() was not called during setup.");
    _rhs = *_b;
    _rhs += _increment;
    // Fold the boundary data into the reduced system. The eliminations act
    // on the scratch copy _rhs, keeping the assembled external load
    // pristine. copy_interior = 1 retains the interior of _u in _X so that
    // solvers in iterative_mode warm start from the previous solution.
    _a->FormLinearSystem(_ess_tdof_list, _u, _rhs, _A, _X, _B, 1);
    const bool ok = SolveLinearSystem(_B, _X);
    _a->RecoverFEMSolution(_X, _rhs, _u);
    return ok;
  }

  void RegisterFields(mfem::DataCollection& dc) override {
    dc.RegisterField("displacement", &_u);
  }
};

/*----------------------------------------------------------------------------
  Concrete problem 0: pure traction
----------------------------------------------------------------------------*/

/**
 * @brief Pure traction (Neumann) problem: a time-scaled uniform traction is
 * applied on all external boundaries.
 *
 * Without essential conditions the stiffness matrix retains the rigid-body
 * null space, so the preconditioned CG solve is wrapped in an
 * mfemElasticity::RigidBodySolver. The wrapper also projects the right-hand
 * side orthogonal to the rigid modes, and hence enforces the Fredholm
 * compatibility condition on the load: any net force or torque is removed.
 */
class TractionProblem : public ElasticProblemBase {
 private:
  mfem::ConstantCoefficient _lambda{1.0};
  mfem::ConstantCoefficient _mu{1.0};
  std::unique_ptr<mfem::VectorFunctionCoefficient> _traction;
  mfem::Array<int> _ext_marker;

  std::unique_ptr<mfem::GSSmoother> _prec;
  std::unique_ptr<mfem::CGSolver> _cg;
  std::unique_ptr<mfemElasticity::RigidBodySolver> _rigid;

 public:
  TractionProblem(mfem::Mesh* mesh, int order)
      : ElasticProblemBase(mesh, order) {
    using namespace mfem;

    const int dim = _mesh->Dimension();

    _a->AddDomainIntegrator(new ElasticityIntegrator(_lambda, _mu));

    // Time-scaled uniform traction, t -> (0, 1 + t, ...).
    _traction = std::make_unique<VectorFunctionCoefficient>(
        dim, [](const Vector& /*x*/, real_t t, Vector& f) {
          f = 0.0;
          f[1] = 1.0 + t;
        });
    RegisterTimeDependent(*_traction);

    _ext_marker.SetSize(_mesh->bdr_attributes.Max());
    _ext_marker = 0;
    _mesh->MarkExternalBoundaries(_ext_marker);
    _b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*_traction),
                              _ext_marker);

    FinalizeSystem();

    _prec = std::make_unique<GSSmoother>(_A);

    _cg = std::make_unique<CGSolver>();
    _cg->SetPreconditioner(*_prec);
    _cg->SetOperator(_A);
    _cg->SetRelTol(_rel_tol);
    _cg->SetMaxIter(10000);
    _cg->SetPrintLevel(IterativeSolver::PrintLevel().Summary());

    _rigid = std::make_unique<mfemElasticity::RigidBodySolver>(_fes.get());
    _rigid->SetSolver(*_cg);
    // RigidBodySolver propagates its own iterative_mode to the wrapped
    // solver on each Mult(), so the warm-start flag is set here rather than
    // on the CG solver. The final projection inside the wrapper removes any
    // rigid component that the warm start might otherwise carry along.
    _rigid->iterative_mode = true;
  }

 protected:
  bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X) override {
    if (!SetWarmStartTolerance(*_cg, *_prec, B)) {
      X = 0.0;
      return true;
    }
    _rigid->Mult(B, X);
    return _cg->GetConverged();
  }
};

/*----------------------------------------------------------------------------
  Concrete problem 1: clamped, in the style of MFEM's ex2
----------------------------------------------------------------------------*/

/**
 * @brief Mixed problem: the boundary with attribute 1 is clamped, while a
 * time-scaled traction pulls downwards on attribute 2; all other boundaries
 * are traction-free.
 *
 * The essential conditions make the reduced operator positive definite, so a
 * plain preconditioned CG is used and no rigid-body projection is required.
 * Intended for meshes with at least two boundary attributes, e.g.
 * data/beam-quad.mesh or data/beam-tet.mesh.
 */
class ClampedProblem : public ElasticProblemBase {
 private:
  mfem::ConstantCoefficient _lambda{1.0};
  mfem::ConstantCoefficient _mu{1.0};
  std::unique_ptr<mfem::VectorFunctionCoefficient> _traction;
  std::unique_ptr<mfem::VectorConstantCoefficient> _clamp_value;
  mfem::Array<int> _ess_bdr;
  mfem::Array<int> _pull_marker;

  std::unique_ptr<mfem::GSSmoother> _prec;
  std::unique_ptr<mfem::CGSolver> _cg;

 public:
  ClampedProblem(mfem::Mesh* mesh, int order)
      : ElasticProblemBase(mesh, order) {
    using namespace mfem;

    const int dim = _mesh->Dimension();

    MFEM_VERIFY(_mesh->bdr_attributes.Max() >= 2,
                "ClampedProblem: the mesh must have boundary attributes 1 "
                "(clamped) and 2 (traction), e.g. data/beam-quad.mesh.");

    _a->AddDomainIntegrator(new ElasticityIntegrator(_lambda, _mu));

    // Essential conditions on attribute 1 (all components).
    _ess_bdr.SetSize(_mesh->bdr_attributes.Max());
    _ess_bdr = 0;
    _ess_bdr[0] = 1;
    _fes->GetEssentialTrueDofs(_ess_bdr, _ess_tdof_list);

    Vector clamp(dim);
    clamp = 0.0;
    _clamp_value = std::make_unique<VectorConstantCoefficient>(clamp);

    // Time-scaled pull on attribute 2, t -> (0, ..., -f0 (1 + t)).
    _traction = std::make_unique<VectorFunctionCoefficient>(
        dim, [](const Vector& /*x*/, real_t t, Vector& f) {
          f = 0.0;
          f[f.Size() - 1] = -0.05 * (1.0 + t);
        });
    RegisterTimeDependent(*_traction);

    _pull_marker.SetSize(_mesh->bdr_attributes.Max());
    _pull_marker = 0;
    _pull_marker[1] = 1;
    _b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*_traction),
                              _pull_marker);

    FinalizeSystem();

    _prec = std::make_unique<GSSmoother>(_A);

    _cg = std::make_unique<CGSolver>();
    _cg->SetPreconditioner(*_prec);
    _cg->SetOperator(_A);
    _cg->SetRelTol(_rel_tol);
    _cg->SetMaxIter(10000);
    _cg->SetPrintLevel(IterativeSolver::PrintLevel().Summary());
    _cg->iterative_mode = true;  // warm start from the previous solution
  }

 protected:
  void UpdateBoundaryValues(mfem::real_t /*t*/) override {
    // Homogeneous clamp, re-projected on every AssembleForce() so that a
    // time-dependent Dirichlet coefficient could be dropped in without any
    // change to the protocol.
    _u.ProjectBdrCoefficient(*_clamp_value, _ess_bdr);
  }

  bool SolveLinearSystem(const mfem::Vector& B, mfem::Vector& X) override {
    if (!SetWarmStartTolerance(*_cg, *_prec, B)) {
      X = 0.0;
      return true;
    }
    _cg->Mult(B, X);
    return _cg->GetConverged();
  }
};

