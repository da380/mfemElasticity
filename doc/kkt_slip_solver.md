# KKT enforcement of the slip constraint and its preconditioning

Method note for the monolithic (KKT) enforcement of the fluid–solid
slip constraint in
`LinearQuasiStaticReferentialSelfGravitatingSlipProblem::EnableKKT()`,
alongside the penalty + augmented-Lagrangian (AL) path it complements.
The two paths share their assembly and solve the same physical problem;
the choice is per use-case, and both are kept. Companion notes:
`doc/slip_interface.tex` (the slip formulation and the AL machinery),
`doc/gauge_penalty_iteration.tex` (the fluid-gauge refinement, common
to both paths), `doc/null_space.md` (projected solvers).

## 1. The two enforcement paths

The single-valued slip organisation carries the broken displacement
pair `(u_s, u_f)` and the potential `ζ¹`, with the interface condition

```
ν · [[u]] = ν · (u_s − J u_f) = 0   on Σ,
```

`J` the interface pairing. The *AL path* folds the penalty
`θ [Bₙ, −BₙJ; −JᵀBₙ, JᵀBₙJ]` into the solver blocks — an exact square
of the jump against the boundary mass `Bₙ` — and drives the jump to
zero by multiplier sweeps, each a full block-MINRES solve. The *KKT
path* instead adds the multiplier `λ` as an unknown on the interface
trace and solves, in one MINRES,

```
[ S   Cᵀ ] [ x ]   [ b ]
[ C   0  ] [ λ ] = [ 0 ],
```

with `S` the θ-augmented blocks (§3) and `C` the constraint row. The
outer loop then consists of the fluid-gauge Tikhonov refinements alone:
the gauge indeterminacy of `u_f` is an underdetermination, not an
interface condition, and stays penalty + refinement in **both** paths.

**The constraint row is covariant by construction.** `C` is built from
the one-sided flux kernel `∮ λ (ν·v) dS`
(`BoundaryNormalScalarIntegrator` with the equilibrium mapping):
Nanson's relation makes flux-type boundary forms exact under mappings —
every area factor cancels — so the constraint needs no covariance
certification, unlike the slip interface *energy* forms.

**The multiplier is physical.** By the first-variation identification
of `doc/slip_interface.tex`, `λ` is the interface normal-traction
perturbation paired with `ν` — a first-class observable
(`KKTMultiplier()`), and a free consistency cross-check against the AL
path's accumulated `ConstraintMultiplier()`.

**Enforcement is weak.** `C x = 0` zeroes the jump against the
multiplier space, so the L2 jump of the solution is the projection
remainder — the *discrete constraint floor*, the same level a converged
AL iteration reaches. Do not expect the KKT jump below the AL one; the
difference is the route, not the destination.

## 2. Discretisation of the multiplier

`λ` lives on the interface boundary dofs of a scalar space on the solid
SubMesh (passed to `EnableKKT()`; the benchmark builds it with
`-kkt-order`). Two choices matter:

- **Equal order** (the displacement's): the strictest constraint space;
  used by the serial cross-check test.
- **One order lower** (`-kkt-order 1` at `-o 2`): the mortar-style
  choice. It improves the inf-sup margin of the trace pairing, shrinks
  the multiplier block, and measured ~1.6× cheaper at an endpoint shift
  of 4e-4 — a change of constraint discretisation, well below mesh
  error. This is the recommended benchmark setting.

## 3. The augmented blocks and why their kernel must match

Keeping the θ-penalty inside `S` (Golub–Greif augmented-Lagrangian
preconditioning) serves two purposes: it regularises the otherwise
near-null relative normal motion at the interface — the same role it
plays in the AL path, and the reason *lowering* θ makes MINRES worse
even though the multiplier owns the constraint (measured: θ = 100 →
16.9k iterations, θ = 10 → 27.7k, θ = 1 → 49.5k) — and it shapes the
constraint Schur complement.

The augmentation must use the **same kernel as the constraint row**.
With the consistent choice

```
S = A + θ · N M̂⁻¹ Nᵀ        (expanded over (u_s, u_f) with J),
```

`N` the constraint kernel and `M̂` the lumped interface mass of the
multiplier space, the Schur complement `C S⁻¹ Cᵀ` clusters at
`M̂ / θ`, and the diagonal preconditioner of §4 is near-optimal. The
original `θ·Bₙ` augmentation (the AL penalty's kernel: the full L2
trace form, not the multiplier-space projection) leaves a kernel
mismatch exactly of the size of the equal-order-trace vs
multiplier-space gap; swapping it for the consistent form was worth
1.35× (22.9k → 16.9k iterations). Assembly is the same pattern as the
Bₙ folds: `N D Nᵀ` with `D = diag(1/m̂)` on the interface dofs (the
other columns of `N` are zero, so `D` elsewhere is irrelevant), serial
sparse products or hypre `ParMult`/`RAP`.

θ itself: the AL study found a conditioning cliff below θ ≈ 100 and a
flat plateau above; the KKT measurements reproduce the cliff-from-below
and find θ = 300 worth only −5%. θ = 100 is the default in both paths.

## 4. The multiplier-block preconditioner, with the scaling warning

Since Schur ≈ `M̂ / θ`, the preconditioner must apply its **inverse**:

```
P_λ = θ · M̂⁻¹        (lumped: entries θ / m̂_i).
```

The dimensional check is worth stating because getting it backwards
cost a factor ~3.5 before it was caught: interface mass entries scale
like `h^(d−1)`, so confusing `θ·M̂` for `θ·M̂⁻¹` mis-scales the
multiplier block by `θ²·m̂²` relative — about 1e4 at h = 0.3. A
preconditioner change must leave the endpoint untouched; the fix moved
the solution by 1.8e-7 (solver tolerance) while cutting 79.4k → 22.9k
iterations. Any future change to this block should re-verify both
properties.

The remaining blocks reuse the problem's own preconditioners, rebuilt
on the consistently augmented matrices (AMG with systems options on the
fluid block in parallel, GS serially; the shifted mapped Laplacian on
`ζ`). The null pairs are the three-block rigid pairs extended by a zero
multiplier component — the constraint row annihilates them (no jump for
common translations; `ν` is radial against the independent tangential
rotations of a frictionless axisymmetric interface).

## 5. What the head-to-head established (fluid_core, h = 0.3, order 2)

| solver | iterations | wall |
|---|---|---|
| AL, 3 sweeps, exact | 4.4k | 76 s |
| AL, 8 sweeps, exact | 11.4k | 200 s |
| KKT, P1 multiplier, 3 refinements, exact | 10.7k | 178 s |
| KKT, same, inexact refinement sweeps | 7.4k | 124 s |

- **Correctness**: KKT ≡ AL within discretisation error everywhere; at
  degree 0 the two agree to 2e-7 with *identical* error against the
  pyslfp reference. The endpoint differences between sweep counts and
  enforcement paths (up to 2e-2 at degree 1) all sit below the mesh
  error of this resolution against pyslfp — notably, the 3-sweep runs
  are not worse than 8-sweep ones by the external reference, so the AL
  default of 8 sweeps oversolves at this h (re-examine at finer h
  before changing defaults).
- **θ-independence as a verification**: the KKT endpoint is invariant
  to θ across 1…300 at the 5e-10 level, as the formulation demands; the
  AL endpoint moves with θ through its constraint floor. This is a
  sharp regression check on the constraint blocks.
- **The inexact-sweep schedule** (`SetSweepTolerance`; geometric
  tightening, final sweep at full tolerance) applies to both paths —
  to the AL multiplier sweeps and to the KKT gauge refinements — at an
  endpoint shift ~1e-3 and ~2e-2 of the respective exact endpoints'
  spread, an order below mesh error. Exact mode (`-sweep-tol 0`,
  the library default) is mandatory for finite-difference (shift)
  studies and strict solver comparisons; the benchmark enforces this
  automatically on `-map-shift` runs.

Two design dead-ends are recorded so they are not retried: an inner
tolerance *coupled to the measured jump* deadlocks (the measured jump
cannot fall below the solver-error floor the loose tolerance itself
sets, while the multiplier error is amplified by θ), and a
plateau-based early exit false-triggers on the same measurement noise
and truncates the multiplier updates. The geometric schedule with a
guaranteed full-tolerance final sweep is the design that survives.

## 6. Usage

```cpp
// Library: single-valued organisation only; mutually exclusive with
// EnableBrokenZeta(). fes_scalar_solid: scalar space on the solid
// SubMesh (one order below the displacement recommended).
slip.SetConstraint(100.0, 3);          // theta; outer refinements
slip.EnableKKT(&fes_scalar_solid);
slip.SetSweepTolerance(0.0);           // 0 = reproducible endpoint
```

Benchmark: `-method slip -kkt -kkt-order 1 -al 3` (θ via `-theta`,
inexact refinements via `-sweep-tol`). The broken-ζ organisation keeps
AL only for now; a second multiplier for the scalar-jump constraint is
the natural extension, and the outward-shift (+ε) singularity probe —
the Schur spectrum of the shifted configuration — is the planned first
use of the KKT frame there.

## 7. The gauge-KKT experiment (parked)

The same KKT idea applies in principle to the fluid gauge: minimise
`½(Qu, u)` subject to the full coupled equations `Au = f`, with `Q` the
unit-scale deviatoric gauge form — no `ε` anywhere, the gauge fixed as
the minimum-Q-energy representative, and the physics exact
(`EnableGaugeKKT()` on the mixed class; the doubled saddle
`[diag(Q,0), A; A, 0]` with the clean `A` in the constraint rows).

It is implemented, correct, and parked for cost. Measured (2-D
three-layer gauged lab and fluid_core h = 0.3 in 3-D): the iterates
reach mesh-level physics early and satisfy the stationarity condition
`Qu + Av = 0` to 1e-5, but MINRES never converges algebraically —
30k iterations against the penalty path's ~800 per solve — under
block-diagonal preconditioning at every grade tried (serial GS,
parallel AMG, and the Murphy–Golub–Wathen composite
`P̂(A)·(A+Q)·P̂(A)` on the multiplier slot, which *degrades* the solve
with smoother-grade components and does not converge with AMG-grade
ones). The structural reason: the "constraint" operator is square and
only approximately singular, so the textbook saddle-preconditioning
theory (which assumes a full-rank rectangular constraint Jacobian)
does not apply, and the near-kernel reappears wherever the scheme
needs `A`-solves. Constraint preconditioning (Keller–Gould–Wathen)
remains the one standard heavy scheme untried; MFEM's `FGMRESSolver`
and the in-house machinery make it about a day's work, but its two
inner `A`-solves per outer iteration are solves of exactly the
near-singular system, so even success is expected to land at
penalty-path cost. The genuinely open question — a parameter-free
Krylov formulation for minimum-energy selection over an *approximate*
kernel whose cost does not degenerate with the near-kernel — is
research-grade numerical analysis, not missing software.

The production path for the gauge therefore remains the `ε`-penalty
with iterated Tikhonov refinement, whose operating point `ε ~ 1e-2`
is now certified from both sides: O(ε) observable bias above it, and
below it a measured ~4.5× cost growth per decade with semi-convergent
pollution by `ε = 1e-4` (the gauge component escapes the refinements).
The contrast with §1–6 is the day's lesson in where KKT pays: the slip
constraint is a genuine full-rank interface condition and its
multiplier system reaches cost parity with AL; the gauge is a
selection over a near-kernel, and there the penalty is not a
compromise but the right regularisation.
