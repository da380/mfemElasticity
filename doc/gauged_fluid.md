# The gauged treatment of fluid regions

Method notes for the gauged-fluid option of `LinearQuasiStaticProblemBase`
(`SetGaugedFluid()`) and its self-gravitating specialisation
(`quasi_static_problem.hpp`, `mixed_problem.hpp`). The formulation
follows Maitra & Al-Attar (2024, §3.7.3) as developed in the working notes
of 25 September 2026 (`doc/BenchmarkPapers/research_notes_2026-09-25.pdf`,
§4): the fluid keeps its displacement in a fully referential description,
and the relabelling gauge is fixed by a small shear penalty whose bias is
removed by iterated Tikhonov refinement. It is the alternative to Dahlen's
treatment of `doc/self_gravitation.md`, which eliminates the fluid
displacement in favour of the potential. A self-contained mathematical
account of the penalty/refinement machinery and of the sliding-interface
discontinuity (spectral analysis, failure condition, semi-convergence,
augmented-Lagrangian connection) is `doc/gauge_penalty_iteration.tex`
(compiled: `gauge_penalty_iteration.pdf`).

## 1. Formulation

### The fluid as a solid without shear

An inviscid fluid region `M_F` carries a displacement like the solid, with
the isotropic stiffness `kappa (div u)(div u')` and zero shear; the
gravity, coupling and load terms are the same integrands as in the solid
(the elastic-gravitational form is structure-blind). What distinguishes a
fluid is a symmetry: the solution is defined only up to a **linearised
relabelling** — a rearrangement of fluid particles with no Eulerian effect.
For a barotropic fluid these are the fields

```
K = { w on M_F :  div(rho w) = 0,   w.m = 0 on the interfaces },
```

Friedman & Schutz's "trivial displacements". `K` lies in the kernel of the
continuous operator (loads do no work on relabellings), so the fluid
displacement is gauge; the observables are the solid displacement, the
potential, and in the fluid anything built from `div(rho u)` and the
interface normal displacement (e.g. the pressure perturbation).

### Gauge fixing: penalty plus iterated refinement

Discretely the finite element space does not contain `K` exactly: the
operator has a large *near*-kernel of "almost relabellings" with
eigenvalues `O(h^p)` instead of zero, and a Krylov method on the raw system
amplifies their `O(h^q)` load components into mesh-dependent fluid
displacements that pollute the solid through the coupling. The remedy
(option 3 of the research notes, "nearly free"):

1. **Penalty.** Add `eps * 2 mu_g dev e(u) : dev e(u')` on the fluid — a
   fluid with shear modulus `eps mu_g`. The solver operates on
   `A + eps Q`, whose near-kernel eigenvalues are bounded below by
   `eps mu_g` times the shear eigenvalue; the observables acquire an
   `O(eps)` bias.
2. **Iterated Tikhonov refinement.** `U_{k+1} = U_k + (A + eps Q)^{-1}
   (f - A U_k)`. After an exact step the physical residual is
   `f - A U = eps Q delta` with `delta` the last increment (for the coupled
   self-gravitating system, `[eps Q delta_u; 0]`), so **no second operator
   is needed**: each refinement solves the regularised system against
   `eps Q` applied to the previous increment. The error contracts by
   `O(eps mu_g / mu_solid)` per step, and the iterates converge to the
   `Q`-minimal solution of the *physical* discrete system — the observables
   become independent of `eps` (checked in the tests to ~1e-4).

**Semi-convergence.** The refinement must not be over-driven: each step
also adds `O(h-residual / (eps mu_g))` of the near-kernel junk that the
penalty exists to suppress (iterated Tikhonov on a consistent-but-discrete
singular system is semi-convergent). Too small an `eps` or too many
refinements bring the junk back; `eps ~ 1e-2` with **2–3
refinements** balances the `O(eps^k)` bias against the `O(k h^q / eps)`
junk, and both improve with mesh refinement. The operating point is
certified two-sided (1 Oct 2026 sweep, fluid_core h = 0.3, order 2,
3 refinements): `eps = 1e-1` leaves a 2.6e-2 observable bias in the
degree-2 load `h`; below the operating point the cost grows ~4.5× per
decade (3.7k → 6.3k → 28.6k → 138k iterations over 1e-1 … 1e-4) and by
`eps = 1e-4` the solution is semi-convergently polluted (`h₂ = −3.17`
against −1.00, despite the solver reporting convergence). The window is
problem- and resolution-dependent in principle but holds at PREM
structure (prem_4, h = 0.2, 5e-4 against pyslfp); every GaugeRefine
carries a contraction-rate tripwire (warn above 0.9: semi-convergence,
`eps` too small; note above 0.2: residual bias beyond the refinement
budget). The worst case is a load
aligned with a near-null mode: a *uniform* tidal gradient (a rigid-mode
load) leaks ~3% of a comparable physical response into the solid on the
coarse test mesh at `eps = 1e-2`, k = 3, growing like `k / eps` beyond
(`TestMixedProblemGauged`); physical (degree-2) loads do not sit on the
near-kernel.

## 2. Why no displacement discontinuity is needed (linear problem)

The natural reading of "fluid–solid interface" asks for a sliding contact:
normal displacement continuous, tangential jump free. Discretising that
needs genuinely new machinery (§5). The observation that avoids it: **the
tangential trace of the fluid displacement is itself gauge**, so imposing
tangential continuity — i.e. using one continuous H1 space over solid and
fluid together — is an admissible *partial gauge fixing*, not a physical
constraint.

The argument: relabellings satisfy the single scalar constraint
`div(rho w) = 0` with `w.m = 0`; their tangential trace on the interface is
unconstrained. Given any fluid solution with tangential slip `J` against
the solid, extend `J` arbitrarily into the fluid as `w~` with `w~.m = 0`,
and correct it by `z` solving `div(rho z) = -div(rho w~)` with vanishing
traces; the compatibility condition is `∮ rho w~.m dS = 0`, which holds.
So a relabelling with any prescribed tangential trace exists, and every
solution can be re-gauged to match the solid's tangential trace. Note what
is *not* used: no sphericity, no motion along level surfaces — only that
the preserved quantity is a single scalar whose linearised constraint is a
divergence equation. The argument therefore survives aspherical
(relabelled) configurations unchanged.

**Where it fails.** A two-parameter fluid — an independently advected
entropy or compositional label `s` — adds the constraint `w.grad s = 0`:
relabellings are confined to `s`-surfaces, only the component of `J` along
the interface's intersection with them can be re-gauged, and full
continuity over-constrains. The same holds in the non-linear problem,
where the slip map is finite. Those cases need the true sliding interface
(§5). The linear barotropic problem — the one Dahlen's treatment addresses
— does not.

Two consequences of the continuous-space gauge are worth noting. The
tangential traction it transmits is the fluid's deviatoric stress,
`O(eps)`, and dies with the refinement like the rest of the bias. And the
conformal-Killing residual ambiguity of the gauge functional (rotations of
a spherical core) is absorbed by the interface trace: an inner-core
rotation must now drag the fluid at `O(eps)` cost, so no Tisserand-type
constraint is needed — and `AddRegionRotations()` should *not* be used in
this mode (the penalty owns those modes; projecting them would bias the
solution by the `O(eps)` restoring they actually feel).

## 3. What changes against Dahlen's treatment

In `LinearQuasiStaticMixedSelfGravitatingProblem` the gauged mode is selected by
construction: **no FluidRegions** — the fluid attributes join the
displacement SubMesh, the rheology gives them `kappa_f` with zero shear,
the `density` coefficient covers them, and `SetGaugedFluid()` adds the
penalty. Nothing else changes, because every operator of the class
integrates over the whole SubMesh:

| Dahlen (`doc/self_gravitation.md`) | Gauged |
|---|---|
| displacement on the *solid* SubMesh | displacement on the whole body |
| fluid mass term (F1), `A_phiphi = (K+DtN)/4piG + M_F`, `M_F <= 0`, near-indefinite for steep cores | no F1: the potential block is the plain Laplace–DtN operator, SPD with margin |
| interface terms (F2), (F3) with `m.grad Phi0` signs | none: interface conditions are natural |
| coupling `c(phi,v) = int_{M_S} rho grad phi . v - int_Sigma rho_F phi (m.v)` | `c(phi,v) = int_M rho grad phi . v` |
| tidal load on the potential row, `-M_F Psi` | none (the fluid's density perturbation is `-div(rho u)`) |
| 2-D: constant-mode inconsistency through the interface coupling | consistent (the constant is exactly null, as without fluids) |
| needs `rho'_F`, level surfaces, interface conventions | needs `kappa_f` and the penalty |
| fluid block absent; well-conditioned | displacement block has the `1/eps` fluid near-kernel |

The refinement on the coupled system is `GaugeRefine()` overridden: zero
potential load and no tidal term in the refinement solves, the potential
accumulated alongside the displacement, and the accumulated pair left as
the next warm start.

### Equivalence and the Adams–Williamson condition

The two treatments describe the same physics **iff the fluid is materially
barotropic**: one thermodynamic parameter, so the constitutive `d rho/d p`
equals the background's, i.e.

```
N^2 = 0   <=>   kappa_f = rho^2 |grad Phi0| / |d rho / d r|      (Adams–Williamson)
```

Then the static `kappa`-response coincides with Dahlen's secular response,
and `TestMixedProblemGauged` (which builds `kappa_f` from the
Adams–Williamson condition on the discrete background gravity) confirms
agreement of solid displacement and potential to the discretisation level
under a zero-mean load. For a two-parameter fluid (`N^2 != 0`) the static
`kappa`-response is the "frozen" elastic one and *differs* from the
secular Dahlen response by `O(N^2)`; the gauge freedom also shrinks (§2),
though the penalty machinery still regularises the nearly-neutral case.
Benchmark comparisons against Dahlen or pyslfp at `l >= 1` must therefore
use neutrally stratified cores (PREM's outer core is Adams–Williamson by
construction; a uniform-density core is not).

### Degree 0

At degree 0 the treatments differ **by design**: Dahlen's fluid is
described by the potential alone and never sees `kappa_f`, which is the
recorded degree-0 gap of the Love-number campaign (pyslfp uses the fluid's
bulk modulus at `l = 0`). The gauged fluid carries `kappa_f` at every
degree, restoring the compressible degree-0 physics;
`TestMixedProblemGauged.Degree0DiffersFromDahlen2D` pins the gap to the
fluid treatment, and the benchmark comparison at `l = 0` is the decisive
check against pyslfp.

## 4. Implementation

All in `LinearQuasiStaticProblemBase` (`quasi_static_problem.hpp`):

- `SetGaugedFluid(fluid_marker, mu_gauge, eps, refinements)` builds a
  template form holding one deviatoric `ElasticityIntegrator(eps mu_g,
  -2/dim, 1)` on the marked attributes. Assembly then produces, next to
  the physical `A` (which `SystemMatrix()` still returns), the penalty
  `eps Q` on true dofs (no elimination; residuals are zeroed on essential
  dofs) and the regularised `A + eps Q` in one extra assembly — a form
  borrowing the physical integrators with the penalty appended (MFEM's
  borrowing forms own no integrators). The solver and preconditioner are
  built on `A + eps Q`; `RegularizedMatrix()` exposes it.
- `Solve()` runs `SolveLinearSystem` once, then `GaugeRefine()`:
  `r = eps Q delta`, one further `SolveLinearSystem` per refinement,
  cold-started. `GaugeResiduals()` reports `|eps Q delta|` per step — their
  decay is the observed contraction.
- Essential boundary conditions must not touch the gauged attributes
  (the penalty's column elimination is not folded into the loads); the
  geophysical use cases have none.
- `mu_gauge` sets the scale of the penalty; the fluid's own bulk modulus
  is a natural choice. The contraction factor is `~ eps mu_g / mu_solid`.

The mass-weighted rigid gauge (`SetMassWeightedGauge`) matters for
comparisons against exact radial solutions: the Euclidean true-dof gauge
leaves a mesh-asymmetry rigid translation (~1e-5 on the test meshes) that
uniform refinement does not remove, and which dominates e.g. a Lamé
comparison at the 0.5% level (`TestFluidGauge`).

## 5. The sliding interface

When a genuine tangential discontinuity is needed (two-parameter fluids,
the non-linear slip map), the construction is: two displacement fields on
two SubMeshes of the same parent (solid and fluid), with normal continuity
enforced weakly on the paired interface. Nodal (strong) coupling of normal
components is not an option — the discrete normal is ill-defined at
vertices of a faceted interface — but the *identification* of the two
trace spaces is nodal and exact:

**The pairing.** Each space has a signed dof injection `Pi` into the
parent space (`SubMeshDofInjection`), and the interface dofs are exactly
the dofs whose parent images coincide. So

```
J = Pi_s^T Pi_f          (solid dofs x fluid dofs, entries +-1)
```

identifies the fluid trace with the solid trace nodally, with all
orientation bookkeeping inherited from the injections
(`NewSubMeshPairingMatrix`; in parallel `NewSubMeshPairingTrueDofMatrix`,
a hypre product of the true-dof injections, which handles the two sides of
an interface dof living on *different ranks* for free). Because the two
trace spaces share the parent's face geometry and nodal basis, every
interface bilinear form assembles **once**, on the solid side's boundary
elements: with `B` the normal-normal form there, the penalty on the
normal jump is the block matrix

```
theta [ B, -B J; -J^T B, J^T B J ].
```

**Constraint without a stiff penalty.** Exactness at moderate `theta`
comes from augmented-Lagrangian iterations, the same iterate-and-refine
pattern as the gauge, and the two interleave in one loop:

```
(A + eps Q + theta P) U_{k+1} = f - w_k + eps Q u_{f,k}
w_{k+1} = w_k + theta P U_{k+1}
```

whose fixed point satisfies the physical system with `P U = 0` and no
`eps` in the observables. The normal jump contracts to a floor set by the
interleaved gauge source (theta-independent relative to the trace; ~6e-5
of the normal trace on the coarse test disc), while the tangential jump
stays free — the slip the continuous space cannot represent.

**Status.** The pairing and the penalty/AL machinery are implemented and
verified on the gravity-free cavity (`TestSlidingInterface`,
`TestSlidingInterfacePar`); `examples/sliding_fluid_ellipse.cpp` runs
them on an *elliptical* body — the geometry where the discontinuity is
genuinely needed: even the uniform load drives an O(ellipticity)
tangential slip (impossible on the disc, where the response is
conformal), and the frictionless interface's independent-rotation null
modes disappear (an elliptical interface transmits torque through the
normal forces alone). in the barotropic setting the sliding solution
matches the condensed rank-one reference (and hence the continuous-space
gauge of §2) on the solid, the normal jump vanishes and the tangential
jump does not. The null space is larger than the welded one — a
frictionless spherical interface transmits no torque, so shell and core
rotate *independently* — and all such modes are projected. Still open for
the self-gravitating case: the three-block `[u_s; u_f; phi]` solver, and
the interface gravity terms of the energy when the slip is physical (a
stratified fluid transports its boundary values along the interface),
whose derivation from the referential energy with the linearised slip map
needs checking before implementation. The mortar (Lagrange multiplier)
route — a contact-type saddle system, Wohlmuth (2001) — remains the
cross-check and the non-linear path.

## 6. Verification

- **Purely elastic, exact references** (`TestFluidGauge`,
  `TestFluidGaugePar`, example `gauged_fluid_cavity`). A gravity-free
  static fluid supports a uniform pressure, so its exact response
  condenses onto the solid as the rank-one cavity term
  `(kappa_f/V) b(u) b(v)`, `b(v) = ∮ v.m dS`: the gauged solution matches
  the Sherman–Morrison solve of the condensed problem to 2e-3 under a
  degree-2 pressure. Uniform pressure has the exact Lamé solution with a
  fluid core (matched to 3e-5 in 2-D, 5e-4 in 3-D; the fluid's exact
  response `gamma x` is conformal, hence also the `Q`-minimal gauge, so
  the *whole* field is compared). The fluid pressure's uniformity is a
  further gauge-invariant observable (1e-4 relative spread on the coarse
  mesh under uniform load).
- **Contraction and eps-independence.** Residual decay ~0.1 per step at
  `eps = 1e-2, mu_g = kappa_f`; observables agree to ~1e-4..1e-5 between
  `eps = 1e-2` and `1e-3` after refinement.
- **Self-gravitating equivalence** (`TestMixedProblemGauged`,
  `TestMixedProblemGaugedPar`): against the Dahlen path on the
  three-layer model with an Adams–Williamson core, solid displacement and
  potential to 2e-2 (the two discretisations' own error) under a zero-mean
  load; the degree-0 difference asserted present; SchurCG and BlockMINRES
  agree to 1e-6 within the gauged problem; serial and parallel agree to
  1e-6.
- **Known leakage**: the rigid-tidal worst case of §1 (semi-convergence).
