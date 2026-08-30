# Quasi-static elastic and viscoelastic problems — design and plan

*30 August 2026. Companion to `status_and_roadmap.md` §5 and to the SubMesh and anisotropy plans. MFEM 4.9.1.*

The idea in `examples/elastic.hpp` / `viscoelasticity.cpp` is taken as given: a **quasi-static linear elastic problem** is anything that, at a time `t`, solves a linear elliptic system whose displacement part can receive extra dual-vector forces; a **viscoelastic problem** owns internal variables that live on the displacement mesh, and drives the elastic problem through a small interface (`AssembleForce / AddForce / Solve / Displacement`). Self-gravitation, fluid cores, rotation and sea level all live *inside* the elastic problem, so the viscoelastic layer works unchanged for coupled problems.

This document fixes the interfaces, the numerics that the interfaces have to support (generalised Maxwell, implicit and exponential stepping, multiple solid regions), the class layout, and the pitfalls. Part II reviews the three plans for C++ standard compliance and cross-cutting issues.

---

# Part I — the elastic/viscoelastic layer

## 1. The mathematics the design has to support

### 1.1 Generalised Maxwell (Prony series)

Isotropic, with `d = dev(ε(u))`, `K` branches:

```
σ = κ tr(ε) I + 2μ_∞ d + Σ_k 2μ_k (d − m_k),        ṁ_k = (d − m_k)/τ_k,  m_k(0) = 0
```

Single Maxwell is `μ_∞ = 0, K = 1` (the *ggae388* form with `τ = η/μ`); Burgers is `μ_∞ = 0, K = 2`; any linear rheology with a finite relaxation spectrum is a Prony series. The **unrelaxed** (instantaneous) shear modulus is `μ_U = μ_∞ + Σ μ_k`; the elastic operator is assembled with `μ_U` and the branches only ever appear as forces `Σ_k Bᵀ(2μ_k m_k)`. Hence the split that the interface encodes:

```
K_U u = f_ext(t) + Σ_k Bᵀ (2μ_k m_k)          (elastic solve with fixed operator)
ṁ_k = (D u − m_k)/τ_k                          (pointwise ODE, D = strain map)
```

Anisotropic generalisation (cf. the anisotropy plan): `σ = C_U ε − Σ_k C_k m_k`, `C_U = C_∞ + Σ C_k`, with `C_k` the relaxable tensor of branch `k` (a modelling choice — deviatoric projection, or "L, N only" for TI).

### 1.2 Time stepping — what each scheme needs from the elastic problem

Let `h_k = dt/τ_k`, `d^n = D u^n`.

| Scheme | Update of `m_k` | Elastic solve | Order / stability | Needs |
|---|---|---|---|---|
| Explicit RK (`Mult`) | any MFEM explicit `ODESolver` | one per stage, operator `K_U` | RK order; `dt ≲ 2.8 τ_min` (RK4) | nothing new |
| ETD1 (exponential Euler) | `m ← e^{−h} m + (1−e^{−h}) d^n` | one per step, `K_U` | 1st, unconditionally stable | nothing new |
| **Exponential trapezoid** (implicit in `d^{n+1}`) | `m ← e^{−h} m + α(h) d^n + β(h) d^{n+1}`, `α = (1−e^{−h})/h − e^{−h}`, `β = 1 − (1−e^{−h})/h` | one per step with **effective modulus** `μ_eff = μ_∞ + Σ μ_k (1−β_k)` and force `Bᵀ Σ 2μ_k (e^{−h_k} m_k^n + α_k d^n)` | 2nd, exact for linearly varying `d`, no step restriction | `SetEffectiveShearModulus` |
| Backward Euler (`ImplicitSolve`) | `m ← (m + h d^{n+1})/(1+h)` | `μ_eff = μ_∞ + Σ μ_k/(1+h_k)`, force `Bᵀ Σ 2μ_k m_k^n/(1+h_k)` | 1st, L-stable | same |
| SDIRK via `ImplicitSolve` | as BE with `dt → γ dt` per stage | one per stage, possibly several distinct `dt` | 2nd–3rd | same + operator cache keyed on `dt` |

Recommendation: **exponential trapezoid as the workhorse** for GIA (large `dt/τ` in the asthenosphere, second order, one solve per step, operator constant while `dt` is constant), ETD1 and explicit RK for verification, BE/SDIRK through MFEM's `ImplicitSolve` for completeness. Crank–Nicolson is deliberately absent (not L-stable; oscillates for `h ≫ 1`).

All implicit-type schemes eliminate `m^{n+1}` and therefore need one capability from the elastic problem: **reassemble the deviatoric part of the stiffness with a pointwise scale** — the effective modulus `μ_eff(x)` is a nodal field on the internal-variable mesh (`τ_k(x)` varies), not a constant. In the coupled problem only the `A_uu` block changes.

### 1.3 The bulk/deviatoric split without a new integrator

`κ div u div v + 2μ dev ε(u):dev ε(v) = λ div u div v + 2μ ε(u):ε(v)` with `λ = κ − 2μ/d`. So MFEM's own `ElasticityIntegrator` gives both pieces:

```
K_κ           = ElasticityIntegrator(kappa, /*q_l=*/1.0, /*q_m=*/0.0)      // κ div-div
K_dev(μ)      = ElasticityIntegrator(mu,    /*q_l=*/-2.0/d, /*q_m=*/1.0)   // 2μ dev:dev
K_U           = K_κ + K_dev(μ_U)
K(dt)         = K_κ + K_dev(μ_eff)
```

(The `(Coefficient &m, real_t q_l, real_t q_m)` constructor sets `λ = q_l·m`, `μ = q_m·m`.) `K_κ` is assembled once; `K_dev` is reassembled when `μ_eff` changes. With the anisotropic integrator the same holds with `C_∞ + Σ(1−β_k) C_k` as a `MatrixSumCoefficient`.

**2-D caveat.** `d` here must be the *same* `d` as in `DeviatoricStrainInterpolator`/`TraceFreeSymmetricMatrixIndex`, which use the space dimension. So in 2-D the library models a *2-D continuum* (2-D deviator, `κ_2D = λ + μ`), **not** plane strain of a 3-D Maxwell body (whose deviator has a `1/3`). That is fine for test problems; state it once in the documentation and never mix the two conventions.

### 1.4 Strain map and adjoint consistency

The coupling `B_ij = ∫ Φ_i : d(φ_j)` (tensor test × displacement trial, unit coefficient) is assembled once. Two strain maps `D : u ↦ d` at the internal-variable nodes:

- nodal interpolation (`DeviatoricStrainInterpolator`, current default);
- Galerkin, `D = M⁻¹ B` with `M` the (element-block-diagonal) L2 tensor mass matrix.

If the internal-variable order is `≥ p − 1` and quadrature is exact, the two coincide (the L2 projection of a representable field is the field). They differ only when the internal variable is under-resolved. The Galerkin map makes the discrete adjoint of a step exactly the transposed step (needed for *ggae388*'s adjoint), costs one block-diagonal solve per evaluation, and for nodal L2 bases with matching quadrature `M` is diagonal. Make `D = M⁻¹B` the default with `m_order = p − 1`; keep interpolation as an option. Either way `D` and `B` live in the base class, not in the model.

## 2. Interfaces

### 2.1 Material and rheology (new: `rheology.hpp`)

The material data must have **one owner**, from which both the elastic operator and the internal-variable model draw, replacing the "documented invariant" that `μ` agrees.

```cpp
/// One Prony branch: relaxable modulus and relaxation time (τ = η/μ_k).
struct MaxwellBranch {
  mfem::Coefficient *mu;   // non-owning
  mfem::Coefficient *tau;
};

/// Isotropic generalised Maxwell rheology: bulk modulus, long-term shear
/// modulus and K Prony branches. Provides the unrelaxed shear modulus the
/// elastic problem must be assembled with.
class GeneralisedMaxwellRheology {
 public:
  GeneralisedMaxwellRheology(mfem::Coefficient &kappa, mfem::Coefficient &mu_inf,
                             const std::vector<MaxwellBranch> &branches);
  /// Convenience for the classical Maxwell body: mu_inf = 0, one branch.
  static GeneralisedMaxwellRheology Maxwell(mfem::Coefficient &kappa,
                                            mfem::Coefficient &mu, mfem::Coefficient &tau);

  mfem::Coefficient &BulkModulus() const;
  mfem::Coefficient &UnrelaxedShearModulus() const;   // owned SumCoefficient chain
  int NumBranches() const;
  const MaxwellBranch &Branch(int k) const;
  int SpaceDim() const;
  /// λ_U = κ − 2 μ_U / d, for problems that want (λ, μ) form.
  mfem::Coefficient &UnrelaxedLame() const;
 private:
  std::vector<std::unique_ptr<mfem::Coefficient>> owned_;   // sums/products
};
```

An anisotropic counterpart later holds `C_∞` and `C_k` as `MatrixCoefficient`s and provides `UnrelaxedTensor()`; the interface below only ever asks for "the coefficient(s) to assemble the elastic operator with" and "the coefficient(s) of the relaxable parts", so the two share the same shape.

### 2.2 The elastic-problem interface (`elastic_problem.hpp`)

Kept as sketched, with three additions: multiple displacement fields, the effective-modulus hook, and material access.

```cpp
/// Contract per evaluation time t:
///   AssembleForce(t);          // all time-dependent data to t; external loads; clear increments
///   AddForce(i, f); ...        // superpose dual vectors on displacement field i (L-vector layout)
///   Solve();                   // displacement(s) <- K^{-1} (external + increments)
/// Solve() may be internally iterative or nonlinear (sea level), but it is a
/// black box to callers; linearity in the *forces* is part of the contract.
class QuasiStaticLinearElasticProblem {
 public:
  virtual ~QuasiStaticLinearElasticProblem() {}

  /// Solid regions carrying a displacement unknown (1; 2 for inner core + mantle).
  virtual int NumDisplacementFields() const = 0;
  virtual mfem::FiniteElementSpace &DisplacementSpace(int i = 0) = 0;
  virtual const mfem::GridFunction &Displacement(int i = 0) const = 0;

  virtual void AssembleForce(mfem::real_t t) = 0;
  virtual void AddForce(int i, const mfem::Vector &f) = 0;
  virtual bool Solve() = 0;

  /// The rheology the operator was assembled with (one per field, or shared).
  virtual const GeneralisedMaxwellRheology &Rheology(int i = 0) const = 0;

  /// Implicit / exponential-trapezoid stepping: reassemble the deviatoric
  /// part of field i's stiffness with shear modulus mu_eff (a nodal field on
  /// the internal-variable mesh). Default: unsupported.
  virtual bool SupportsEffectiveShearModulus() const { return false; }
  virtual void SetEffectiveShearModulus(int i, mfem::Coefficient &mu_eff)
  { MFEM_ABORT("effective shear modulus not supported by this problem"); }
  virtual void ClearEffectiveShearModulus() {}

  virtual void RegisterFields(mfem::DataCollection &dc) = 0;
};
```

Notes on the contract:

- `AddForce` takes the vdof (L-vector) layout of `DisplacementSpace(i)`; in parallel the problem applies `Pᵀ` once in `Solve`. Callers never see true dofs. `AddForce` *accumulates*; `AssembleForce` clears.
- `AssembleForce(t)` is called at every RK stage with possibly non-monotone `t`; it must be cheap (reassemble the linear form only) and idempotent.
- `SetEffectiveShearModulus` changes the operator; the implementation must invalidate its preconditioner/solver setup and, in coupled problems, any Schur-complement caches. `ClearEffectiveShearModulus` restores `μ_U`. Implementations may cache by the *identity* of `mu_eff`'s underlying `GridFunction` plus a version counter, but simplest is: reassemble on every call and let the viscoelastic layer call it only when `dt` changes.
- `Rheology(i)` lets the viscoelastic layer verify consistency at construction (`&problem.Rheology(i) == &rheology` or, in debug, nodal sampling of `μ_U`).

### 2.3 `ElasticProblemBase` — the shared implementation

```cpp
class ElasticProblemBase : public QuasiStaticLinearElasticProblem {
 protected:
  mfem::FiniteElementSpace *fes_;           // serial or parallel; owned or not
#ifdef MFEM_USE_MPI
  mfem::ParFiniteElementSpace *pfes_ = nullptr;   // dynamic_cast of fes_, null in serial
#endif
  const GeneralisedMaxwellRheology *rheology_;

  std::unique_ptr<mfem::BilinearForm> a_kappa_;   // κ div-div, assembled once      (Par variants via factory)
  std::unique_ptr<mfem::BilinearForm> a_dev_;     // 2μ dev:dev, reassembled on SetEffectiveShearModulus
  std::unique_ptr<mfem::LinearForm>   b_;         // external loads
  mfem::Vector increment_, rhs_;
  mfem::GridFunction u_;                          // ParGridFunction in parallel
  mfem::Array<int> ess_tdof_list_;
  mfem::OperatorHandle A_;                        // SparseMatrix or HypreParMatrix
  mfem::Vector X_, B_;
  std::vector<mfem::Coefficient*> td_coefs_; std::vector<mfem::VectorCoefficient*> td_vcoefs_;
  mfem::real_t rel_tol_ = 1e-12;
  bool operator_dirty_ = true;

  virtual void UpdateBoundaryValues(mfem::real_t t) {}
  virtual void SetupSolver(const mfem::Operator &A) = 0;       // build/refresh preconditioner + solver
  virtual bool SolveLinearSystem(const mfem::Vector &B, mfem::Vector &X) = 0;
  bool SetWarmStartTolerance(mfem::IterativeSolver &, mfem::Solver &prec, const mfem::Vector &B) const;
  void AssembleOperator();                                     // K = K_κ + K_dev(μ_current); sets operator_dirty_
 public:
  ElasticProblemBase(mfem::FiniteElementSpace *fes, const GeneralisedMaxwellRheology &rh);
  // interface methods implemented once for serial and parallel:
  void AssembleForce(mfem::real_t t) override;
  void AddForce(int i, const mfem::Vector &f) override;
  bool Solve() override;                    // FormLinearSystem / RecoverFEMSolution; parallel via OperatorHandle
  bool SupportsEffectiveShearModulus() const override { return true; }
  void SetEffectiveShearModulus(int, mfem::Coefficient &mu_eff) override;   // reassemble a_dev_, operator_dirty_ = true
  void ClearEffectiveShearModulus() override;
};
```

Serial/parallel: one class. Forms are created through a small factory (`MakeBilinearForm(fes)` returns `ParBilinearForm` when `pfes_` is non-null), `FormLinearSystem` goes through `OperatorHandle`, the solver setup in the derived class picks `GSSmoother`/`HypreBoomerAMG` by the same test. This is the pattern the Poisson operators already use and it halves the code.

`Solve()` skeleton:

```
rhs = b + increment;  (Par: rhs is an L-vector; FormLinearSystem does Pᵀ)
if (operator_dirty_) { A_ = K_κ + K_dev (SparseMatrix::Add or HypreParMatrix::Add); FormSystemMatrix; SetupSolver(A_); operator_dirty_ = false; }
FormLinearSystem(ess, u, rhs, A_, X, B, copy_interior=1);
if (!SetWarmStartTolerance(...)) { X = 0; } else ok = SolveLinearSystem(B, X);
RecoverFEMSolution(X, rhs, u);
```

`TractionProblem` (rigid-body wrapper) and `ClampedProblem` stay as the two reference implementations; the coupled `SelfGravitatingElasticProblem` overrides `SetupSolver`/`SolveLinearSystem` with the block or Schur solver and `NumDisplacementFields`.

### 2.4 `ViscoelasticOperator` (`viscoelastic.hpp`)

One class, generalised Maxwell, any number of displacement fields; the "model" virtuals of the sketch collapse into pointwise loops over branches, since the rate law is fixed. Nonlinear rheologies (power-law, Andrade) would subclass and override `Rate`/`LocalUpdate`; keep those two virtual.

```cpp
class ViscoelasticOperator : public mfem::TimeDependentOperator {
 public:
  enum class StrainMap { Galerkin, Interpolation };

  ViscoelasticOperator(QuasiStaticLinearElasticProblem &problem,
                       int internal_order = -1,           // < 0: displacement order − 1
                       StrainMap map = StrainMap::Galerkin);

  // --- state layout -------------------------------------------------------
  int NumFields() const;                 // = problem.NumDisplacementFields()
  int NumBranches(int i) const;          // = problem.Rheology(i).NumBranches()
  const mfem::Array<int> &Offsets() const;            // block offsets, (field, branch) major
  mfem::Vector Branch(mfem::Vector &m, int i, int k) const;   // aliasing view, no copy
  mfem::FiniteElementSpace &InternalVariableSpace(int i);

  // --- ODE interface -------------------------------------------------------
  void Mult(const mfem::Vector &m, mfem::Vector &k) const override;         // explicit RHS
  void ImplicitSolve(mfem::real_t dt, const mfem::Vector &m, mfem::Vector &k) override;  // BE / SDIRK stage
  // --- exponential steps (called by the two ODESolver adaptors below) ----
  void ExponentialEulerStep(mfem::Vector &m, mfem::real_t &t, mfem::real_t dt);
  void ExponentialTrapezoidStep(mfem::Vector &m, mfem::real_t &t, mfem::real_t dt);

  // --- observation ---------------------------------------------------------
  bool SolveElastic(const mfem::Vector &m, mfem::real_t t);   // consistent u for (m, t)
  void SyncFields(const mfem::Vector &m);                       // to registered GridFunctions
  void RegisterFields(mfem::DataCollection &dc);
  mfem::real_t MinRelaxationTime() const;

 protected:
  struct Field {                                   // one per displacement field
    mfem::FiniteElementSpace *ufes;
    std::unique_ptr<mfem::FiniteElementCollection> fec;    // L2
    std::unique_ptr<mfem::FiniteElementSpace> dfes;         // trace-free tensors, vdim n_s−1
    std::unique_ptr<mfem::FiniteElementSpace> sfes;         // scalar companion (nodal coefficients)
    std::unique_ptr<mfem::MixedBilinearForm> B;             // ∫ Φ : d(φ), unit coefficient (Par in parallel)
    std::unique_ptr<mfem::Operator> D;                      // strain map: M⁻¹B (block-diag solve) or interpolator
    std::vector<mfem::Vector> two_mu;                        // per branch, nodal 2μ_k
    std::vector<mfem::Vector> itau;                          // per branch, nodal 1/τ_k
    mfem::GridFunction mu_eff;                               // nodal effective modulus (on sfes)
    std::unique_ptr<mfem::GridFunctionCoefficient> mu_eff_coef;
    std::vector<mfem::GridFunction> m_out;                   // output views per branch
    mutable mfem::Vector d, force, zeta;                     // scratch
  };
  std::vector<Field> fields_;
  QuasiStaticLinearElasticProblem &problem_;
  mfem::real_t cached_dt_ = -1.0;                            // dt for which μ_eff was set

  // pointwise kernels (isotropic; virtual for nonlinear rheologies)
  virtual void Rate(const Field &, int k, const mfem::Vector &m_k, const mfem::Vector &d, mfem::Vector &k_out) const;
  virtual void LocalUpdate(const Field &, int k, mfem::real_t dt, mfem::Vector &m_k, const mfem::Vector &d) const;
  void AddInternalForces(const mfem::Vector &m) const;        // Σ_k Bᵀ(2μ_k m_k) per field → problem.AddForce
  bool ElasticUpdate(const mfem::Vector &m, mfem::real_t t) const;
  void SetEffectiveModulus(mfem::real_t dt, Scheme);          // fills mu_eff per field, calls problem.SetEffectiveShearModulus once per dt change
};

/// ODESolver adaptors so drivers can switch integrators without changing the time loop.
class ExponentialEulerSolver     : public mfem::ODESolver { /* Step → op->ExponentialEulerStep */ };
class ExponentialTrapezoidSolver : public mfem::ODESolver { /* Step → op->ExponentialTrapezoidStep */ };
```

Step bodies (per field `i`, branch `k`; `h = dt·itau`):

```
Mult(m, k):            AssembleForce(GetTime()); AddInternalForces(m); Solve(); d = D u;  k_k = (d − m_k)·itau
ExponentialEuler:      as Mult up to d;  m_k ← e^{−h} m_k + (1 − e^{−h}) d;  t += dt
ExponentialTrapezoid:  d_n = D u^n (from a solve at m^n, t^n — reuse if the previous step left u consistent);
                       if dt != cached_dt: mu_eff = μ_∞ + Σ μ_k (1−β_k); SetEffectiveShearModulus; cached_dt = dt
                       AssembleForce(t+dt); AddForce(Σ_k Bᵀ 2μ_k (e^{−h_k} m_k + α_k d_n)); Solve(); d_np1 = D u^{n+1}
                       m_k ← e^{−h_k} m_k + α_k d_n + β_k d_np1;  t += dt
ImplicitSolve(dt,m,k): (t already set by the ODESolver)  mu_eff = μ_∞ + Σ μ_k/(1+h_k) (cache on dt)
                       AssembleForce(t); AddForce(Σ_k Bᵀ 2μ_k m_k/(1+h_k)); Solve(); d = D u
                       m_new_k = (m_k + h_k d)/(1+h_k);  k = (m_new − m)/dt
```

`ClearEffectiveShearModulus()` must be called before any subsequent `Mult`/ETD1 (they need `K_U`); track the state in the operator and switch lazily.

## 3. Class layout and files

```
include/mfemElasticity/rheology.hpp          MaxwellBranch, GeneralisedMaxwellRheology
include/mfemElasticity/elastic_problem.hpp   QuasiStaticLinearElasticProblem, ElasticProblemBase,
                                             TractionProblem, ClampedProblem (reference implementations)
include/mfemElasticity/viscoelastic.hpp      ViscoelasticOperator, ExponentialEulerSolver, ExponentialTrapezoidSolver
include/mfemElasticity/self_gravitating.hpp  SelfGravitatingElasticProblem (after the SubMesh work)
src/…                                        matching .cpp files; the stale elasticity.hpp/.cpp removed
examples/quasi_static_elasticity.cpp, viscoelasticity.cpp   drivers only
tests/TestRheology.cpp, TestViscoelastic.cpp
```

Dependencies: `viscoelastic.hpp` depends only on the interface and on `bilininteg.hpp` (strain integrators/interpolators); it never includes `poisson.hpp` or `submesh.hpp`. That dependency direction is the whole point.

## 4. Crucial details and things to avoid

1. **`AssembleForce` is called at RK stage times.** MFEM's explicit solvers call `f->SetTime(t + c_i dt)` then `Mult`; use `GetTime()` inside `Mult`, never a stored `t`. Time-dependent load coefficients must therefore be registered with the problem (`RegisterTimeDependent`), never sampled once.
2. **`ImplicitSolve` semantics.** MFEM asks for `k` with `k = f(m + dt·k)`; return the *rate*, not `m^{n+1}`. SDIRK schemes call it with `γ·dt`; if `dt` changes, `μ_eff` changes and the operator must be reassembled — cache on `dt`, and expect several distinct values per step for multi-stage schemes (a 2-entry cache in the problem is worth having; or restrict implicit MFEM solvers to BE/SDIRK23, which use one γ).
3. **Warm starts and tolerances.** Already fixed in `elastic.hpp` (`SetWarmStartTolerance`); keep it in the base class only. After `SetEffectiveShearModulus` the previous `u` is still the right warm start.
4. **Operator invalidation.** `SetEffectiveShearModulus` → reassemble `K_dev`, re-add to `K_κ`, `FormSystemMatrix`, rebuild preconditioner. In the self-gravitating problem: only `A_uu`; the AMG for `A_uu` must be rebuilt, the Poisson block and DtN must not be touched. Make this explicit in `SetupSolver` (take the block that changed).
5. **Ordering assumptions.** The pointwise loops assume `Ordering::byNODES` on the L2 tensor space (component `c` occupies `[c·nd, (c+1)·nd)`); assert it in the `Field` constructor. The trace-free component convention is `TraceFreeSymmetricMatrixIndex`; the anisotropic plan's tensor ordering is chosen to match.
6. **Nodal coefficient sampling.** `2μ_k`, `1/τ_k` are sampled once at the L2 nodes (`ProjectCoefficient` on `sfes`) — correct for attribute-wise discontinuous data because L2 nodes are element-interior. `μ_eff` is a `GridFunction` on `sfes` wrapped in a `GridFunctionCoefficient`, which the elastic problem's `ElasticityIntegrator` evaluates at quadrature points by interpolation — consistent only if `sfes` order ≥ the variation of `τ`; with `p−1` and piecewise-constant material that is exact.
7. **Which force integrator.** For isotropic relaxation the force is `Bᵀ(2μ_k m_k)` with the trace-free `B`; for an anisotropic `C_k` applied to a trace-free `m_k` the product is generally *not* trace-free, so the force needs the full symmetric `DomainSymmetricMatrixStrainIntegrator` (`n_s` components), not the trace-free one. Keep both `B`s available; choose by rheology type.
8. **Internal-variable order.** Default `p − 1` (exactly resolves `d(u)`); `p` is allowed but wasteful. `ProjectCoefficient` of `τ` at order 0 nodes is what most models actually need.
9. **Explicit stability.** `dt < ≈ 2.8 τ_min` (RK4) is a *sufficient* estimate: the coupled modes relax no faster than `1/τ_min`. Keep the driver warning.
10. **Zero-force steps.** `SetWarmStartTolerance` returns false for `b = 0`; the base sets `X = 0`. With a nonzero warm start that is the right answer only for the *reduced* system; fine here.
11. **Restart.** `m` plus `t` is the full state; the elastic problem holds no history. Provide `SolveElastic(m, t)` for a consistent `u` after restart (already sketched).
12. **Don't** let the viscoelastic layer own or see `φ`, `ω`, `SL`. If something in it needs the potential (it shouldn't), the interface is wrong.
13. **Don't** call `problem.Solve()` inside `Rate`/`LocalUpdate`; one solve per stage, in `ElasticUpdate`, so the count is predictable and adjoint-friendly.
14. **Parallel `AddForce`.** L-vector in, `Pᵀ` inside `Solve`; `Bᵀ` of a `ParMixedBilinearForm` in the local layout gives exactly that L-vector — no `ParallelAssemble` needed on the force path.
15. **MFEM `TimeDependentOperator` sizing.** `height = width = total state size`; `Mult` output must be sized (`k.SetSize`) — MFEM's solvers preallocate, but be explicit.

## 5. Tests

1. **Rheology algebra.** `UnrelaxedShearModulus` equals `μ_∞ + Σ μ_k` at random points; `Maxwell()` factory; `UnrelaxedLame`.
2. **Split identity.** `K_κ + K_dev(μ)` equals `ElasticityIntegrator(λ = κ − 2μ/d, μ)` element-wise, 2-D and 3-D.
3. **Strain maps.** `D_Galerkin u == D_interp u` for `m_order = p − 1` and a polynomial `u` (exact quadrature); `B` and `M D` are transposes.
4. **0-D analytic checks on a uniform block.** (a) Constant pure-shear traction, Maxwell: `d(t)` grows linearly with slope `σ/(2η)` after the instantaneous response; (b) relaxation test: prescribed constant displacement (clamped variant), reaction stress `σ(t) = 2[μ_∞ + Σ μ_k e^{−t/τ_k}] d` — exact Prony series; two branches with `τ₁/τ₂ = 100`. Both with all integrators at `dt ≪ τ_min`; then with `dt ≫ τ_1` for the implicit/exponential ones (must still be exact for (b), since `d` is constant).
5. **Temporal convergence.** Time-varying traction `f(t)`; reference = RK4 at tiny `dt`; observed orders: ETD1 → 1, BE → 1, exponential trapezoid → 2, SDIRK23 → 2.
6. **Long-time limit.** Clamped Maxwell body: `u(t→∞)` equals the elastic solution with `μ = μ_∞` (for `μ_∞ > 0`) — a check of the effective-modulus path.
7. **Interface conformance.** A mock `QuasiStaticLinearElasticProblem` with two displacement fields verifies the state layout and that forces are routed to the right field.
8. **Physics.** Viscoelastic self-gravitating sphere relaxing to hydrostatic equilibrium under a degree-2 load vs the radial codes (once `SelfGravitatingElasticProblem` exists).

## 6. Phases

| Phase | Deliverable | Effort |
|---|---|---|
| 1 | `rheology.hpp`; `elastic_problem.hpp` promoted from `examples/elastic.hpp` with the κ/dev split, effective-modulus hook, multi-field interface, serial+parallel in one class; `Traction`/`Clamped` reference problems; tests 1–2 | 2 d |
| 2 | `viscoelastic.hpp`: generalised Maxwell, block state, Galerkin strain map, `Mult`, ETD1, exponential trapezoid, `ImplicitSolve`; adaptors; tests 3–7 | 2–3 d |
| 3 | `SelfGravitatingElasticProblem` implementing the interface (after SubMesh phases 1–2); test 8 | 3–5 d |
| 4 | Anisotropic branches (`C_k`), nonlinear rheology hooks, adjoint stepping | later |

---

# Part II — review of all plans

## 7. C++ standard

Facts: MFEM 4.9's `CMakeLists.txt:22` sets `CMAKE_CXX_STANDARD 17` as its default; this library's `CMakeLists.txt` sets 20; the committed library code already uses C++17 (`std::optional` in `mesh.hpp/.cpp`, structured bindings in `mesh.cpp:105,273,282` and `poisson.cpp:24,277,642,715,1136,1246`) and C++20 (`<numbers>`/`std::numbers::pi_v` in `legendre.hpp:5,25`), plus `constexpr std::sqrt/std::log` in `legendre.hpp:27–34`, which is a GCC extension in every standard before C++26 and will not compile with Clang or MSVC.

Recommendation: target **C++17 as the floor**, because that is what MFEM itself requires today, and anything upstreamed would be built with it. If you nevertheless want C++14 for the coupled-problem core, the changes are mechanical: replace the eight structured bindings with `std::tie`/`std::get`, drop `std::optional` (it is only `#include`d, unused), replace `std::numbers` with a `static const` computed once (`4*std::atan(1.0)`), and make the `constexpr` constants `static const` initialised in `legendre.cpp`. `std::make_unique` (C++14) is fine either way. Nothing in the three plans needs C++17: the SubMesh mixin template, the coefficient classes, the interfaces above are all C++11/14 constructs. Two habits to adopt in library code from now on: no `auto [a, b]`, no `[[nodiscard]]`/`if constexpr`/`std::string_view`/`std::optional`, no `<numbers>`; keep those for `examples/`.

Also set `CMAKE_CXX_STANDARD` from one place (top-level `CMakeLists.txt`), and add `-Wall -Wextra -Wpedantic` for the library target so extensions like the `constexpr` math fail loudly.

## 8. Cross-cutting review notes on the three plans

- **SubMesh plan.** The `MixedBilinearForm(tr, te, mbf)` borrow constructor and the `mat` member are protected/public API that has been stable since 4.0; the plan depends on nothing else internal. One refinement: `SubMeshMixedBilinearForm::Assemble` should `MFEM_VERIFY` that `GetFBFI()->Size() == 0` (interior-face integrators) *and* that no `SetAssemblyLevel` was called, and should keep the shadow space alive for the object's lifetime (`FormRectangularSystemMatrix` does not need it, but `Mult` after `Assemble` on the helper does not exist — the helper is destroyed at the end of `Assemble`, which is intended). For the self-gravitating problem the `AddForce` path never touches the coupling blocks; `SetEffectiveShearModulus` never touches them either — consistent with §4.4 here.
- **Anisotropy plan.** `FromVelocities` returning by value requires the class to be movable while holding references to owned `ProductCoefficient`s — hold them in a `std::vector<std::unique_ptr<Coefficient>>` and store raw pointers, not references, to keep it movable. The relaxable tensor `C_k` per branch slots into `MaxwellBranch` as a `MatrixCoefficient*` alternative to `mu`; keep `MaxwellBranch` a small struct with either scalar or tensor members set, and let the viscoelastic operator choose the force integrator by which is set (§4.7).
- **Status/roadmap.** The roadmap's step 3 ("library `elasticity.hpp`") is exactly Phase 1 here; step 4 is Phase 3 here and Phase 3 of the SubMesh plan. Suggested global order: SubMesh phase 1 (serial injection + form) → this Phase 1 → this Phase 2 → anisotropy A/B (any time) → SubMesh phase 2 (parallel) → `SelfGravitatingElasticProblem` → physics benchmarks.
- **One naming convention.** The library mixes `_member` (`solvers.hpp`, `poisson.hpp`) and `member_` (`elastic.hpp`, `radial_model.hpp`) and `Google` vs `MFEM` brace styles; `.clang-format` exists — pick `member_` (the newer code) and run it once before the new files land, so the upstream-candidate core is uniform.
