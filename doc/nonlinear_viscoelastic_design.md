# Nonlinear viscoelasticity (state-dependent relaxation times) and adaptive time stepping — design plan

*2 September 2026. Companion to `viscoelastic_design.md` and to `fluid_solid_design.md` Part II. Source: Crawford, Al-Attar, Tromp, Mitrovica, Austermann & Lau (2017) Appendix A, which follows Simo & Hughes (1998).*

For discussion before code. §1 pins down the equations, §2 the numerics, §3 the classes and names, §4 tests, §5 effort, §6 the decisions I would like confirmed.

---

## 1. The model

### 1.1 Constitutive law

The generalised Maxwell body of the library, isotropic or anisotropic, is

  σ = C_U ε − Σ_k C_k m_k,   C_U = C_∞ + Σ_k C_k,   ṁ_k = (ε − m_k)/τ_k ,                     (1)

with the elastic part linear. Crawford et al. (A10)–(A11) keep (1) and make the relaxation times depend on the state,

  τ_k = τ_k0 / (1 + γ_k (‖T‖ / 2μ_0)^{n_k − 1}),   ‖T‖² = T : T,   T = dev σ,                  (2)

so that the slow-deformation limit is the composite Newtonian/power-law fluid (A7) with effective viscosity η = Σ_k μ_k τ_k(T), diffusion creep (n = 1) at low stress and dislocation creep (n = 3) at high stress. The isotropic trace-free representation gives T = Σ_k 2μ_k (d − m_k), (2.42); in general T = dev(C_U ε − Σ_k C_k m_k). Nothing else changes: the weak form (A19) is linear in the rates, the elastic solve is the same linear solve, and the only new ingredient is a pointwise scalar function

  τ_k = τ_k0(x) · f_k(x; ε, m_1, …, m_K)                                                        (3)

evaluated at the internal-variable nodes. I propose the library takes exactly this: per branch, a reference time (the coefficient we have) and an optional **relaxation law** f_k with its own parameter fields, evaluated from the nodal state. The power law (2) is the first instance; others (temperature-dependent, Burgers-type with a stress threshold, laws in terms of m rather than σ) fit the same slot.

### 1.2 What stays exactly as it is

- The stiffness, the rigid-mode handling, self-gravitation, fluid regions, anisotropy: untouched. The problem layer never sees τ.
- The state is still the concatenation of the m_k; `Rate`, the exponential updates and the strain maps act nodally with a scalar τ_k(p) that is now recomputed from the node's state instead of read from a table.
- A linear branch is a law that ignores the state, so one code path serves both.

### 1.3 Consequences worth knowing before the numerics

- τ_k can vary by orders of magnitude across the domain and in time (for n = 3 and stresses a few times the transition stress τ_e, (2) reduces τ by 10–100). Explicit stepping is then out of the question and the accuracy scale of the problem changes during the run: the case for adaptivity.
- The nonlinearity is *diagonal* in space (per node) and scalar (through ‖T‖). No nonlinear global system arises; everything global remains a linear elastic solve. This is what makes the extension "almost trivial" and it shapes the design: nonlinear iteration is cheap nodally, and the only expensive object that changes with the state is the *effective* stiffness of the implicit and trapezoid schemes.

---

## 2. Numerics

### 2.1 Which schemes need what

| Scheme | Linear now | With state-dependent τ |
|---|---|---|
| Explicit (RK4 etc., `Mult`) | rates with tabulated τ | rates with τ from the node's state; nothing else. Stability dt ≲ 2.8 τ_min(state), impractical at high stress |
| ETD1 (`ExponentialEulerStep`) | exact relaxation with d frozen | same with τ frozen at the step start: first order, unconditionally stable, one solve per step. Free |
| Exponential trapezoid | effective modulus β_k(dt/τ_k), one solve | β_k needs τ_k over the step. With τ_k evaluated at a **midpoint state** the scheme keeps second order; the midpoint state needs a predictor (§2.2) |
| Backward Euler / SDIRK (`ImplicitSolve`) | m^{n+1} = (m^n + h d^{n+1})/(1 + h), effective modulus | h = dt/τ(m^{n+1}, d^{n+1}) makes the elimination nonlinear: same predictor–corrector structure (§2.2), with the nodal equation solved exactly per node inside |

### 2.2 Predictor–corrector for the effective-modulus schemes

For the trapezoid step (the workhorse) I propose:

1. **Predictor.** τ*_k := τ_k at the start state (m^n, d^n). Optionally one ETD1 half-step of m (nodal, no solve) to estimate the midpoint m; the strain d^{n+1/2} is not available without a solve, so the predictor's τ* is first-order accurate.
2. **Solve** with weights β_k(dt/τ*_k) → u^{n+1}, d^{n+1}, m^{n+1}.
3. **Corrector.** τ*_k := τ_k at the midpoint state ((m^n + m^{n+1})/2, (d^n + d^{n+1})/2); repeat 2.
4. Iterate 3 to a tolerance on the relative change of τ* (default: one corrector pass, which gives second order for smooth τ(t); `SetCorrectorIterations(max, tol)` to iterate to convergence).

Each corrector pass costs an elastic solve with a *different* effective operator (the weights changed). Backward Euler is the same with the end state instead of the midpoint and the nodal equation m = (m^n + h(τ(m, d)) d)/(1 + h) solved per node by a few scalar fixed-point/Newton iterations on ‖T‖.

### 2.3 The effective operator changes every step — the real cost

In the linear case the effective stiffness K(C_∞ + Σ(1 − β_k)C_k) is assembled once per (scheme, dt). With state-dependent τ, or with a variable dt even for a linear body, β_k changes at every step and every corrector pass. Three options, in increasing sophistication:

(a) **Reassemble and rebuild the preconditioner each time** (what `SetRelaxationWeights` does today). Correct, simplest, but the BoomerAMG setup is the dominant cost: several solves' worth per pass.

(b) **Reassemble, keep the preconditioner lazily.** The matrix is reassembled (one assembly, comparable to a few matvecs at order 1–2), but the preconditioner built for an earlier effective matrix is kept as long as the Krylov iteration count stays within a factor (say 2) of the count at its setup; then it is rebuilt. Exact solution every step; the preconditioner only has to be "close". This is a small change local to `LinearElasticProblemBase::SetupSolver` (and the block preconditioner of the self-gravitating class), policy-selectable, and it also serves David's adjoint use case (jumps needing the unrelaxed operator at many observation times: with a cache of two preconditioners, unrelaxed and effective, nothing is rebuilt). **Recommended first step.**

(c) **Matrix-free effective operator.** K_eff(β) u = K_U u − Σ_k Bᵀ β_k C_k D u with everything already in hand (K_U assembled once, B, D = (G⁻¹⊗M⁻¹)B, nodal C_k): apply it matrix-free inside the Krylov solve, preconditioned by (b)'s cached AMG of some assembled K_eff. No reassembly at all; the identity Bᵀ β C D = K(βC) holds exactly for the Galerkin strain map with an exactly resolving internal order (the same fact that makes the elimination exact today). Needs the problem layer to accept an operator correction (`SetStiffnessCorrection(Operator&)`), which the self-gravitating block operator can take in its (0,0) block. Worth doing when profiling says assembly matters (3-D order 2 with several corrector passes per step); not for the first version.

I propose (b) now with (c) as a documented follow-up.

### 2.4 Variable step size

Exponential integrators have no stability limit here, so adaptivity is purely about accuracy — resolving the times when τ collapses (loading transients) and striding across quiet periods. Proposal:

- **Embedded pair, no extra solves.** After a trapezoid step, the ETD1 prediction m̂^{n+1} (τ and d frozen at the start, nodal work only) is a first-order companion of the second-order m^{n+1}; err = ‖m^{n+1} − m̂^{n+1}‖ / (atol + rtol ‖m^{n+1}‖) in a nodal max or L2 sense (global in parallel). Standard controller: dt_new = dt · clamp(0.9 err^{−1/2}, 0.2, 4); reject and retry when err > 1. The predictor is already computed in §2.2 step 1, so the estimate is free.
- **State-based guard** (optional, cheap): dt ≤ c · min_p τ_k(p) over nodes with significant stress, as a safety net when the error estimate underresolves a fast relaxation — off by default.
- **API.** An `AdaptiveExponentialTrapezoidSolver : mfem::ODESolver` with `SetTolerances(rtol, atol)`, `SetStepBounds(dt_min, dt_max)`; `Step(m, t, dt)` takes the step (possibly reduced), and returns in `dt` the proposed next step (MFEM's convention; `ODESolver::Run` then works). Observation times are hit by clipping dt, as usual.
- The linear body benefits identically (β depends on dt), with (b) keeping the cost down.

### 2.5 Adjoints later

The adjoint of (1)–(2) needs ∂τ_k/∂(state) at the nodes. The law interface should carry an optional derivative (`Gradient` with respect to the stress-norm argument), unused now, so that a law written today serves the adjoint without a rewrite.

---

## 3. Classes and names

### 3.1 The nonlinear operator as the base — in fact as the only operator

Since the nonlinearity is a pointwise scalar and the global structure is unchanged, there is no need for two operators: `ViscoelasticOperator` becomes the general one, and *linearity is a property of the rheology* (`Rheology::IsLinear()`: every branch law state-independent). The operator queries it and takes the short paths (no corrector pass, no τ re-evaluation, effective operator assembled once) when it holds. This is the "nonlinear one as the base" but without an inheritance layer to maintain, and every existing test runs through the same code.

### 3.2 Rheology side

```cpp
/// Pointwise relaxation law: tau_k = tau_k0 * Factor(state). Parameters are
/// coefficients sampled once at the internal nodes.
class RelaxationLaw {
 public:
  virtual bool IsStateDependent() const = 0;
  virtual int NumParameters() const = 0;
  virtual mfem::Coefficient& Parameter(int i) const = 0;
  /// tau/tau0 from the nodal parameters and the local state: strain and
  /// stress (tensor components, library ordering, full symmetric), and the
  /// branch's own internal variable.
  virtual mfem::real_t Factor(const mfem::real_t* params, const LocalState& s) const = 0;
  /// d(Factor)/d(stress components), for adjoints; default: not available.
  virtual bool HasGradient() const { return false; }
};

/// Crawford et al. (2017, A11): 1 / (1 + gamma (|dev sigma| / 2 mu0)^(n-1)).
class PowerLawRelaxation : public RelaxationLaw { ... gamma, n, mu0 ... };

struct MaxwellBranch      { Coefficient* mu;      Coefficient* tau; const RelaxationLaw* law = nullptr; };
struct AnisotropicBranch  { MatrixCoefficient* C; Coefficient* tau; const RelaxationLaw* law = nullptr; };

class Rheology {  // as now, plus
  virtual const RelaxationLaw* Law(int k) const = 0;   // nullptr: linear
  bool IsLinear() const;                               // all laws null or state-independent
  virtual void UnrelaxedModulus(T, ip, DenseMatrix& CU) const;  // C_U pointwise, for the nodal stress (isotropic: 2 mu_U P_dev + kappa P_vol)
};
```

`LocalState` holds pointers to the node's strain, stress and internal variables in the operator's layout; the operator computes the nodal stress σ = C_U ε − Σ_k C_k m_k from data it already has (nodal C_k) plus a sampled C_U (or, in the trace-free isotropic case, T = Σ 2μ_k(d − m_k) directly, no C_U needed).

Naming: `IsotropicMaxwellRheology` (renamed from `GeneralisedMaxwellRheology` on 2 Sep 2026), alongside `AnisotropicMaxwellRheology`; "generalised Maxwell" describes both and the state-dependent times, so it is better as the doc term than as a class prefix. No "Linear" in the rheology names: linearity is `IsLinear()`.

### 3.3 Problem side

- `QuasiStaticLinearElasticProblem` (interface) keeps its name — the elastic part stays linear and this is what the future nonlinear-elastic interface will contrast with (`QuasiStaticElasticProblem`, with residual and tangent, later).
- The linear elastic-problem base is `LinearElasticProblemBase` (renamed from `ElasticProblemBase` on 2 Sep 2026), freeing the plain name for the base of the nonlinear-elastic family when it arrives. `TractionProblem`, `ClampedProblem`, `SelfGravitatingElasticProblem` keep their names (they are linear problems on that base; a nonlinear-elastic self-gravitating problem would be a different class anyway).
- One new policy on the base, for §2.3(b): `SetPreconditionerPolicy(Rebuild::Always | Rebuild::WhenSlow(factor))`, default `WhenSlow(2)`; and the two-slot cache (unrelaxed, effective) that David's adjoint use needs.

### 3.4 Operator side

- `ViscoelasticOperator`: `Field` gains nodal law parameters per branch, the sampled C_U where needed, a nodal stress scratch vector, and `EvaluateRelaxationTimes(field, state)`; `Rate`, `LocalExponentialUpdate`, `SetEffectiveModulus` take τ from that evaluation. New controls `SetCorrectorIterations(max = 1, tol = 1e-3)`. `MinRelaxationTime()` becomes state-dependent (evaluated at the current cache).
- `ExponentialTrapezoidSolver` unchanged in name; `AdaptiveExponentialTrapezoidSolver` new (§2.4). `ExponentialEulerSolver` unchanged.

---

## 4. Tests

1. **Linear through the general path**: a state-independent law gives bit-identical results to the tabulated τ (all schemes, serial and MPI).
2. **Homogeneous power-law bar** (as the TI test): uniaxial stress, homogeneous state, so the FE solution follows the nodal ODE ṁ = (d(m) − m)/τ(‖T(m)‖) — reference by fine RK4 on the 6×6 system. Checks: ETD1 first order, trapezoid with one corrector second order, with converged corrector second order, backward Euler first order; the long-time strain rate equals the composite-viscosity fluid (A7) at the applied stress.
3. **Stress dependence**: doubling the applied stress in the power-law regime (n = 3) multiplies the long-time strain rate by 8; in the Newtonian regime (stress ≪ τ_e) by 2.
4. **Adaptive stepping**: on the same bar with a step load, the adaptive solver meets its tolerance against the fine reference with far fewer steps than the fixed-step run of equal accuracy; steps shrink at the load and grow after; rejected steps are counted.
5. **Preconditioner policy**: `WhenSlow` gives the same solutions as `Always` to solver tolerance, with fewer AMG setups (counted).
6. **Parallel**: 2 and 4 at 1/2/4 ranks (norms against serial).
7. **Self-gravitating**: a Maxwell-with-power-law mantle under the surface load runs through `SelfGravitatingElasticProblem` unchanged (smoke test; physics with the Love-number machinery later).

---

## 5. Effort

| Step | Content | Days |
|---|---|---|
| 1 | `RelaxationLaw`, `PowerLawRelaxation`, nodal stress and τ evaluation; explicit and ETD1 paths; renames (§3.2–3.3) | 1 |
| 2 | Predictor–corrector trapezoid and backward Euler; corrector controls | 1 |
| 3 | Lazy preconditioner policy and the two-slot cache in the problem layer | ½ |
| 4 | Adaptive solver | ½–1 |
| 5 | Tests 1–7, docs | 1 |

Total ≈ 4–5 days, after which the matrix-free effective operator (§2.3c) is an optional optimisation.

---

## 6. Decisions to confirm

1. **One operator, linearity a property of the rheology** (§3.1), rather than a nonlinear class deriving from or extended by the linear one.
2. **The law abstraction** (§3.2): pointwise factor on τ_k0 from (strain, stress, own m_k) with sampled parameters; the power law as the first instance. Are there other laws you want in the first cut (temperature dependence through a coefficient field; a stress threshold)?
3. **Predictor–corrector with a midpoint τ** for the trapezoid (§2.2), default one corrector pass. Alternative: always iterate to tolerance (more solves, cleaner error behaviour).
4. **Cost policy** (§2.3): lazy preconditioner now, matrix-free effective operator later.
5. **Adaptivity** by the embedded ETD1/trapezoid pair (§2.4), with MFEM's `Step`-modifies-`dt` convention.
6. **Names** (§3.2–3.3): `IsotropicMaxwellRheology` / `AnisotropicMaxwellRheology`; `LinearElasticProblemBase`; the interface `QuasiStaticLinearElasticProblem` unchanged; `ViscoelasticOperator` unchanged (general). Alternatives welcome — this is the cheapest moment to rename.
7. **Adjoint hook** (§2.5): reserve `HasGradient`/`Gradient` on the law now, implement with the adjoint work.

---

## 7. Implementation status (2 September 2026)

All of §1–§4 is implemented, serial and MPI, with the decisions of §6 as proposed.

- `relaxation_law.hpp`: `LocalState` (full symmetric strain, stress and branch variable at a node; the trace-free representation is expanded, its stress being the deviatoric stress), `RelaxationLaw` (parameters as coefficients sampled at the internal nodes; `Factor`; optional `Gradient` for adjoints), `PowerLawRelaxation` (A11) with its gradient. Tests: `TestRelaxationLaw` (values, linear limits, the transition stress, gradient against finite differences).
- Rheology: `MaxwellBranch` / `AnisotropicBranch` carry an optional law; `Rheology::Law(k)`, `IsLinear()`, `UnrelaxedModulus()` (for the nodal stress of anisotropic bodies); the `Maxwell` and `DeviatoricMaxwell` factories take a law. Renames done: `IsotropicMaxwellRheology`, `LinearElasticProblemBase`.
- `ViscoelasticOperator` (§3.1: one operator, linearity a property): nodal law parameters, current vs reference times, `EvaluateRelaxationTimes`; explicit and ETD1 use the times of the current state; the trapezoid and the implicit stages run the predictor–corrector of §2.2 (`SetCorrectorIterations`, default one pass, early exit on a converged time field); the effective weights are re-sent to the problem whenever the times changed (a version counter). The ETD1 companion of every trapezoid step is kept for `ErrorEstimate(rtol, atol)`.
- `AdaptiveExponentialTrapezoidSolver` (§2.4) with `Integrate()` to a final time. The estimate is that of the first-order companion, hence conservative for the propagated second-order solution by a factor of order τ/dt: at rtol 1e-4 the relaxation test takes 134 steps where the trapezoid alone needs about 10 for the same accuracy. Documented on the class; a second-order companion (step doubling, three solves per step) would be the way to a sharp estimate if it ever matters.
- Problem layer (§2.3b): `SetPreconditionerReuse(factor)` (default 2) keeps the preconditioner, with the form and matrix it was built on, across reassemblies while the iteration count stays within the factor of its setup count (`NoteIterations` from every solver, including the self-gravitating block solvers); `NumPreconditionerSetups()`. The matrix is always the current one. Found on the way: the cache must hold the form the preconditioner was built on, not the most recent one (a use-after-free otherwise, caught by the tests).
- Tests (`TestViscoelastic`, `TestViscoelasticPar`, `TestElasticProblem`): a γ = 0 law reproduces the linear body (all schemes, 1e-10); stress-controlled power-law creep is the linear creep at τ(‖T‖) (exact for the trapezoid, rate ratio between two stress levels as the composite law predicts); strain-controlled relaxation against the nodal ODE: ETD1 and backward Euler first order, the trapezoid second order with one corrector and with a converged corrector; the adaptive solver meets its tolerance; preconditioner reuse gives the exact solutions with fewer setups and rebuilds after a large drift. Parallel power-law creep at 1/2/4 ranks.
- Examples: `viscoelastic_schemes` (cost/accuracy sweep and, with `-targets`, the cost of reaching a given final error for every integrator, on a clamped beam: single relaxation time, a stiff two-branch body (`-tau-ratio`) or the power law. Findings in the file header: with nothing stiff RK4 is cheapest and the first-order schemes hopeless; in the stiff and power-law cases SDIRK23 is the best fixed-step scheme (17/65/129 solves for 1e-2/1e-3/1e-4 against RK4's 1025), the adaptive trapezoid best at tight tolerances (123), and backward Euler/ETD1 cost 10–100× more; the adaptive runs reassemble every step but keep one preconditioner), `viscoelastic_loading` (GIA-style Cartesian loading and rebound with a low-viscosity channel), `self_gravitating_relaxation` (Phase 5 of the fluid plan: Maxwell mantle, elastic inner core, fluid core, Heaviside surface load; the two-layer PREM-like model subsides from 124 m to 149 m at the pole over five Maxwell times of 181 yr for η = 1e21 Pa s — physics to be checked against the radial codes with the Love-number work), and `viscoelasticity -gamma g -nexp n` (power law on the Maxwell branch, isotropic or TI) and `-rtol r` (adaptive stepping between the output times); with γ = 5 and rtol 1e-3 on the beam: 302 accepted and 55 rejected steps, one preconditioner setup.

Not done: the matrix-free effective operator (§2.3c); laws beyond the power law (the slot takes them); the adjoint gradient is implemented for the power law but unused.
