# Linearised elasticity with gravity: the general framework

Method notes collecting the linearised theory of a self-gravitating,
pre-stressed elastic body from the papers in `doc/Elasticity/`, as the
foundation for generalising the hydrostatic implementation
(`doc/self_gravitation.md`, `doc/gauged_fluid.md`) to general reference
bodies, non-hydrostatic initial stress and sliding interfaces. Summary
level, with an eye to implementation; the quasi-static linearised problem
only (rotation enters solely through the centrifugal background; the
non-linear theory is deferred).

Sources, and how they are used here:

- **Maitra & Al-Attar (2021)**, `ggaa591.pdf` ("stress dependence of the
  elastic tensor"): the notation adopted here (close to Marsden & Hughes),
  and the authoritative treatment of the elastic-tensor/initial-stress
  relations of §3.
- **Maitra & Al-Attar (2024)**, `ggae092.pdf` ("elastodynamics of rotating
  planets"): the cleanest equations of motion; the referential gravity
  (their eq. 21) and the exact tangential-slip machinery (§3). Its
  *linearised* fluid–solid boundary integral drops a surface-Jacobian
  factor (harmless for statics); the linearised boundary terms are taken
  from AC18 instead.
- **Al-Attar, Crawford, Valentine & Trampert (2018)**, `ggy141.pdf`
  ("AC18"): the correct linearised equations in weak form about a general
  equilibrium (their eqs. 108–137), including the fluid–solid surface
  terms and the particle-relabelling transformation laws.
- **Woodhouse & Deuss (2007)**, `Woodhouse-Deuss.pdf` ("WD"): the
  traditional mixed (Eulerian-potential) formulation, and the *explicit*
  statement of the pressure-modified moduli (their eqs. 59–64) that books
  tend to bury.
- **Al-Attar & Woodhouse (2010)**, `181-1-567.pdf` ("AW10"): the
  parametrisation of equilibrium stress fields in general models, and the
  modern rendering of Dahlen's hydrostatic-region argument (§6 below).

## 1. Notation and kinematics

Following Maitra & Al-Attar (2021): reference body `B` with referential
coordinates `x`; motion `φ(x, t)`; deformation gradient `F = Dφ`
(`F_ij = ∂φ_i/∂x_j`, second index referential), Jacobian `J = det F`,
right Cauchy–Green tensor `C = Fᵀ F`. Referential density
`ρ(x) = J ϱ∘φ`. Stress measures: first Piola–Kirchhoff `P` (traction per
referential area, `T = P·N̂`), second Piola–Kirchhoff `S`, Cauchy `σ`,
related by `P = F S = J (σ∘φ) F⁻ᵀ`. Hyperelasticity: strain energy
`W(x, F) = V(x, C)` (frame indifference), and

```
P = D_F W,     S = 2 D_C V.
```

Two elastic tensors:

```
A = D²_F W     "first":  hyperelastic (major) symmetry only, A_iAjB = A_jBiA
C = 4 D²_C V   "second": classical symmetries, 21 components
```

related by `A = L_F C L_Fᵀ + R_S` — the `R` term carrying the stress
explicitly. A reference configuration is **natural** when the equilibrium
is `φ_e(x) = x` (`F_e = 1`); then `P_e = S_e = σ_e =: T⁰`, and

```
A_iAjB = C_iAjB + δ_ij T⁰_AB .                                   (*)
```

An **equilibrium** satisfies `Div P_e + ρ γ_e = 0` with `γ_e` the (self-)
gravitational acceleration plus centrifugal force; nothing requires `T⁰`
hydrostatic. The reference body itself is arbitrary (it need not be a
configuration the body ever occupies): all physical statements are
invariant under particle relabellings `ξ`, with

```
ρ̃ = J_ξ ρ∘ξ,   W̃(x̃, F̃) = J_ξ W(ξ(x̃), F̃ F_ξ⁻¹),
Ã_ijkl = J_ξ (F_ξ⁻¹)_jm (F_ξ⁻¹)_ln A_imkn ∘ ξ,   ϖ̃ = J_{ξ|Σ} ϖ∘ξ,
```

(AC18 eqs. 83, 134–136) — exactly the transformation laws the mapping
layer (`doc/mappings.md`) implements, with the surface multiplier picking
up the *surface* Jacobian.

## 2. The linearised quasi-static problem

Write `φ = φ_e + u` and expand the action to second order. About a
natural reference the quadratic elastic-plus-initial-stress energy is (WD
eqs. 12–16, AC18 eq. 97)

```
E₂(u) = 1/2 ∫_B  C ε(u) : ε(u)  +  T⁰_AB ∂_A u_k ∂_B u_k  dV,      (E)
```

i.e. the classical-symmetry tensor `C` acting on the symmetric strain,
plus the initial stress acting on the **full** gradient (it enters
through the second-order part of the Green strain). Equivalently
`E₂ = 1/2⟨A·Du, Du⟩` with `A` from (*). Both terms are symmetric in
`u ↔ v`; `A` has no minor symmetries, and this is the same "major
symmetry only" structure as the pulled-back tensors of the mapping layer.

The linearised momentum equation in weak form, about a general
equilibrium, with self-gravity and a steadily rotating frame
(AC18 eq. 137, quasi-static: drop `∂²_t u` and Coriolis):

```
∫_B { ⟨A·Du, Dw⟩ + ρ⟨Ω×(Ω×u) − γ¹(u), w⟩ } dV + (surface terms, §4)
    = loads,                                                        (M)
```

where `γ¹` is the linearised gravitational acceleration (§3) and the
centrifugal term can be absorbed into the background potential
`Φ₀ → Φ₀ + ψ` for constant `Ω`.

### The elastic tensor dictionary — the point that cannot be missed

Seismology does not tabulate `C`. With hydrostatic initial stress
`T⁰ = −p⁰ 1`, WD (eq. 59–61) rearrange the equations so that pressure
never appears explicitly: the `p⁰`-parts of `A` combine with the
equilibrium condition `∇p⁰ = −ρ∇Φ₀` into the familiar gravity terms
`ρ[∇(u·∇Φ₀) − ∇Φ₀ div u]`, leaving a stiffness term `C^eff ε(u):ε(v)`
with the **effective** tensor

```
C^eff_ijkl = C_ijkl + p⁰ ( δ_ij δ_kl − δ_il δ_jk − δ_ik δ_jl ),
```

which again has all classical symmetries. **PREM's moduli are the
isotropic components of `C^eff`, not of `C`** (WD eqs. 62–64 define
κ, μ, A, C, N, L, F from `C^eff`). The conversion is:

```
λ_bare = λ_PREM − p⁰,    μ_bare = μ_PREM + p⁰,    κ_bare = κ_PREM − p⁰/3 .
```

This is *not* a small correction: at the CMB `p⁰ ≈ 136 GPa` against
`μ_PREM ≈ 290 GPa` — a ~45 % shift in the bare shear modulus. In a fluid
(`W = V(x, J)`, `P = −pJF⁻ᵀ`): `μ_PREM = 0` but `μ_bare = p⁰` — the bare
strain-energy tensor of a fluid has *pressure-sized shear components*,
which conspire with the explicit `T⁰` term of (*) to make the incremental
traction purely normal. Consequences for the code:

- The current hydrostatic implementation is exactly WD eq. (59): it
  assembles `C^eff` (PREM moduli) plus the rearranged gravity terms, and
  feeding PREM values into it is **correct as it stands**.
- Any *referential* formulation that assembles (E) — the general
  initial-stress path — must use the bare `C`. Feeding PREM moduli into
  the `C`-slot of (E) is wrong by the conversion above. The clean
  implementation is a coefficient-level conversion
  `C = C^eff − p⁰(δδ − δδ − δδ)` (an `ElasticTensorCoefficient`
  decorator taking `p⁰` as a Coefficient), so that model files keep
  seismological moduli and the conversion cannot be silently forgotten.
- The two assemblies must agree for hydrostatic `T⁰`: a sharp
  whole-operator identity test (same quadrature, same meshes), analogous
  to the mapped 2a identities.

The heuristic §1.1 of Maitra & Al-Attar (2021) also parametrises how the
*moduli themselves* respond to incremental stress (their `Π` tensor,
Dahlen vs Tromp & Trampert as special cases); that is a constitutive
question about Earth models, not a formulation question, and does not
enter the linearised solver.

## 3. Gravity: three formulations

Let `Φ₀` be the background potential (with centrifugal contribution when
rotating) and `g = ∇Φ₀`.

1. **Mixed, Eulerian potential** (WD eqs. 21–22; the current code): keep
   the perturbation `φ¹` of the *spatial* potential as an unknown:

   ```
   (1/4πG) ∫_ℝ³ ∇φ¹·∇χ dV = ∫_B ρ ∇χ·u dV      (Poisson row)
   coupling in (M):  ∫_B ρ ⟨∇φ¹ + (∇∇Φ₀)·u, w⟩ dV
   ```

   Constant-coefficient Laplacian (assembled once, DtN on a fixed outer
   sphere), local coupling `∫ρ∇φ¹·w`. This is what
   `LinearQuasiStaticSelfGravitatingProblem` does, with the `∇∇Φ₀` term
   integrated by parts into the symmetrised form.

2. **Mixed, referential potential** (Maitra & Al-Attar 2024, eq. 21):
   `ζ = φ∘φ` extended over all space via a diffeomorphic extension of
   the motion; the Poisson operator becomes `(1/4πG)⟨a ∇ζ, ∇χ⟩` with

   ```
   a = J F⁻¹ F⁻ᵀ = J C⁻¹,
   ```

   *identical to the pulled-back diffusion operator of the mapping layer*
   (`PullbackDiffusionCoefficient`). Linearised about a natural
   reference: `ζ¹ = φ¹ + u·g` and
   `δa(u) = (div u)1 − Du − Duᵀ`, so the coupling moves into the Poisson
   row as `(1/4πG)∫⟨δa(u)∇Φ₀, ∇χ⟩` — derivatives shift from `φ¹` onto
   `u` and the background. Same content, different sparsity.

3. **Eliminated, non-local** (AC18 eqs. 99/122): substitute the Green's
   function, giving a dense double integral `γ¹(u)`. A theoretical device
   that moves the derivations along — it was never intended as, and is
   never good as, a numerical formulation (dense operator). Not pursued.

**The choice here is (2), the referential form** (decided 29 Sep 2026).
It costs only assembly (the operator is the
`TransformedDiffusionIntegrator` machinery), and it leaves the DtN
untouched: the equilibrium mapping is the identity at and beyond the DtN
sphere — the motion of the exterior is gauge — so `a = 1` there and the
outer closure is the plain one. Note the mapping is *not* the identity
throughout the buffer in general: unless `φ_e` happens to be the identity
on the body's physical surface, its surface values must be tapered
smoothly (and diffeomorphically) to the identity at the buffer's external
boundary. Any smooth extension is admissible — the extension is gauge —
but a rule is needed: a radial blending heuristic, or a small elliptic
(harmonic-extension) solve in the buffer with the surface values and the
identity as Dirichlet data, done once in pre-processing. The
background-state layer owns this rule; the analytic benchmark mappings
already taper by construction. Beyond convenience there is a physical
reason (Maitra & Al-Attar 2024, a small but subtle point): the Eulerian
form's gravity coupling contains the *bare displacement*, not a strain
(the `u·∇∇Φ⁰`-type terms sample the reference potential at displaced
positions). If secular motions are excited — true polar wander, slow
rotations, any drift that grows displacement without growing strain —
that linearisation fails on an aspherical reference, where the sampled
potential actually changes; a spherically symmetric reference is immune
only because its potential is invariant under the drift. The referential
form's gravity enters through `Du` and composition alone, with no bare
displacement anywhere, and is indifferent to secular motion. Since the
general implementation is being built afresh, this implicit approximation
of the traditional theory is retired along with the others.

### 3.1 The linearised referential system

Write the static part of the action (Maitra & Al-Attar 2024, eq. 21) as

```
V[φ, ζ] = ∫_B W(x, F) dV + ∫_B ρ ζ dV + (1/8πG) ∫_{ℝ³} ⟨a(F) ∇ζ, ∇ζ⟩ dV,
a(F) = J F⁻¹F⁻ᵀ,
```

with the motion extended diffeomorphically to ℝ³ (§3's taper). Stationarity
in ζ is the referential Poisson equation; stationarity in φ the momentum
equation. Perturb `φ = φ_e + u`, `ζ = ζ⁰ + ζ¹` and expand to second
order. With `H_u = F_e⁻¹ Du` (all indices referential; for a vector basis
function `u = φ_a e_k`, `H` is rank-one), the derivatives of `a` are

```
a′(u)   = (tr H_u) a_e − H_u a_e − a_e H_uᵀ,
a″(u,v) = [tr H_u tr H_v − tr(H_u H_v)] a_e
          − tr H_u (H_v a_e + a_e H_vᵀ) − tr H_v (H_u a_e + a_e H_uᵀ)
          + (H_u H_v + H_v H_u) a_e + a_e (H_uᵀH_vᵀ + H_vᵀH_uᵀ)
          + H_u a_e H_vᵀ + H_v a_e H_uᵀ,
```

symmetric in `u ↔ v` as it must be. The `∫ρζ` term is linear in `ζ` and
contributes only to the background equations; `ρ` is never differentiated
anywhere. The linearised system in `(u, ζ¹)`, test functions `(v, χ)`,
with `g₀ = ∇ζ⁰` and the **referential gravity flux** `w = a_e g₀`
(the Nanson-transported background gravity):

```
u-row:  ∫_B ⟨A·Du, Dv⟩ dV                         [material + geometric, §2]
        + (1/8πG) ∫ ⟨a″(u,v) g₀, g₀⟩ dV            [gravity–gravity]
        + (1/4πG) ∫ ⟨a′(v) g₀, ∇ζ¹⟩ dV             [coupling, transpose]
        = ℓ_u(v)

ζ-row:  (1/4πG) ∫ ⟨a_e ∇ζ¹, ∇χ⟩ dV                 [mapped Laplace + DtN]
        + (1/4πG) ∫ ⟨a′(u) g₀, ∇χ⟩ dV              [coupling]
        = ℓ_ζ(χ)
```

The coupling expands to
`⟨a′(u) g₀, ∇χ⟩ = (tr H_u)(w·∇χ) − (H_u w)·∇χ − (H_uᵀ g₀)·(a_e ∇χ)`,
and the gravity–gravity term reduces to products of per-dof scalars:
for the basis function `u = φ_a e_k`, `H` is rank-one and every
contraction in `⟨a″ g₀, g₀⟩` is built from `m_a = F_e⁻ᵀ∇φ_a`,
`∇φ_a·w`, `F_e⁻ᵀg₀` and `F_e⁻¹w` — O(d) work per dof pair, no
fourth-order coefficient ever formed (the assembly pattern of the mapped
elastic recipe).

Structural remarks:

- **At `φ_e = id`, hydrostatic**: `H = Du`, `a_e = 1`, `w = g₀ = ∇Φ₀`,
  and the coupling reduces to
  `⟨[(div u)1 − Du − Duᵀ]∇Φ₀, ∇χ⟩/4πG`. The system is related to the
  Eulerian one of the current implementation by the invertible change of
  variables `ζ¹ = φ¹ + u·∇Φ₀`: congruent, not identical — the tier-(i)
  verification maps solutions across, it does not compare operators
  entry-wise.
- **No gravity interface or surface terms.** `ρ` is a fixed referential
  field, so no `[∇φ¹·n̂ + 4πGρ u·n̂]`-type jump conditions arise: the
  natural flux continuity of `a∇ζ` carries all of it, boundary motion
  living inside `a(F)`. Second derivatives of `Φ₀` never appear either.
- **Saddle structure as before**: symmetric, elastic-positive in `u`,
  Laplace-type in `ζ¹`; the block solvers and projectors carry over.
- **Rigid modes have no potential partners.** The referential potential
  `ζ = φ∘φ` is *invariant* under rigid motions of the body (the spatial
  potential co-moves), so the null pairs are `(t, 0)` for translations
  (`Du = 0`: every term vanishes identically) and `(Wφ_e, 0)` for
  rotations (`Du = W F_e`, so `sym(F_eᵀ D u) = sym(F_eᵀ W F_e) = 0`
  identically; the geometric and gravity terms cancel by the rotational
  invariance of the equilibrium energy). Contrast the Eulerian
  formulation's coupled near-null pairs `(u_r, −u_r·∇Φ₀)`: the projector
  machinery simplifies, and rigid near-nullity is not degraded by secular
  drift (§3's bare-displacement point).

**The gravitational stress tensor** (Maitra & Al-Attar 2024 write
gravity as a stress `N` alongside `T`, giving momentum-equation terms
`⟨N, Dw⟩ + ∫_∂B ⟨N n̂, w⟩`): equivalent after integration by parts to the
body-force form, at the price of an extra external boundary term. Elegant
analytically; adopted nowhere here — the body-force form is what the
mixed formulations above use, and it keeps the free surface natural.

## 4. Boundaries and interfaces

- **Free surface / welded interfaces**: natural conditions of the weak
  form; nothing to assemble.
- **Fluid–solid interfaces**: the exact theory constrains the motion by
  `φ₋ ∘ χ = φ₊` with a slip map `χ ∈ Diff(Σ)`, and the tangential-slip
  constraint on test functions is enforced by a scalar multiplier `ϖ`
  paired with the Nanson vector `J F⁻ᵀ n̂` (Maitra & Al-Attar 2024,
  eqs. 164–171); the traction condition across the interface carries the
  *surface* Jacobian, `t₋ = J_χ t₊∘χ`. Linearised about a general
  equilibrium (AC18 eqs. 108–116; their weak form eq. 137):

  ```
  − ∫_Σ ϖ⁰ ⟨Q·Δu, Dw₁⟩ − ∫_Σ ϖ⁰ ⟨Du₁, Q·Δw⟩ + ∫_Σ ϖ⁰ ⟨S·Δu, Δw⟩ dS,
  with the constraint  ⟨F₁⁻¹ Δu, n̂⟩ = 0  on Σ,   Δu = u₂ − u₁,
  ```

  where `ϖ⁰` is the equilibrium multiplier (minus the equilibrium
  pressure on `Σ`), and `Q`, `S` are equilibrium-geometry operators
  (AC18 eqs. 107/109) built from a level-set function of `Σ`. At a
  natural reference these reduce to the Woodhouse & Dahlen surface
  terms — WD eq. 23's `π⁰`-terms with tangential surface derivatives —
  and for a hydrostatic equilibrium further to the (F2)-type terms of
  the current implementation. The perturbation `ϖ¹` of the multiplier
  is the Lagrange multiplier of the linearised slip constraint: exactly
  the mortar variable of the sliding-interface machinery
  (`doc/gauged_fluid.md` §5), whose penalty/AL implementation stands in
  for it.
- **Gravity across interfaces**: `[φ¹] = 0` and `[∇φ¹·n̂ + 4πGρu·n̂] = 0`
  are natural in the mixed weak forms (WD eq. 24); no assembly.

## 5. Fluids, stratification and the gauge

A fluid region has `W = V(x, J)`: bulk response only, `P = −pJF⁻ᵀ`,
`p = −D_J V`. In the linearisation this supplies `C` with
`κ_bare = κ_PREM − p⁰/3` **and** `μ_bare = p⁰` (§2) — the general-form
statement of "a fluid at pressure is not a shear-free solid". The
quasi-static solution in a fluid is defined only up to linearised
relabellings in the appropriate material class; everything in
`doc/gauged_fluid.md` (gauge penalty, Tikhonov refinement, the
continuous-space gauge for barotropic fluids, the sliding interface
otherwise) applies verbatim with the general operators of this note, and
the barotropic/Adams–Williamson discussion there identifies when the
hydrostatic short-cuts remain exact.

## 6. Equilibrium stress in general models

The general form (E) takes `T⁰` as data, but `T⁰` is not free: it must
satisfy `Div T⁰ = ρ∇Φ₀` with continuous tractions, and be hydrostatic in
fluid regions. AW10 organise the solution set completely:

- **Affine structure.** Any two equilibrium stress fields differ by an
  element of `ker(Div₀)` (divergence-free, traction-free); so
  `T⁰ = T_m + Σ aᵢ Sᵢ` with one particular solution and a basis of
  divergence-free fields (constructed via generalised spherical harmonics
  for spherical models, AW10 §4).
- **Particular solutions with desirable properties.** The minimum-norm
  equilibrium stress solves a *linear elastic* boundary-value problem
  (AW10 eqs. 56–57: shear modulus `μ`, `κ = 2μ/3`); the
  minimum-**deviatoric** equilibrium stress solves the steady
  *incompressible Stokes* problem (AW10 eqs. 72–74). The latter is the
  same mathematical object as the Stokes/deviatoric-stress-minimisation
  route to hydrostatic equilibrium figures (research notes Thread A):
  one Stokes solver serves both the figures project and the generation of
  physically plausible `T⁰` for general models.
- Solvability requires the body force and boundary tractions to exert no
  net force or torque (AW10 eq. 58) — the same compatibility our
  projected solvers enforce.
- The deviatoric part of `T⁰` reads as *stress-induced anisotropy* of the
  seismological tensor (AW10 eqs. 76–79, the D&T `Υ = Γ + stress terms`
  split): a further entry in the §2 dictionary.

### Dahlen's hydrostatic-region argument, and where it stops

AW10 §2.2–2.3 give the clean modern form of Dahlen's essential argument,
for perturbations of a *spherical* hydrostatic reference. In a fluid
region the linearised balance `−∇p¹ = ρ⁰∇φ¹ + ρ¹∇Φ₀` is processed in
three moves: (i) cross with `r̂` — using that `∇Φ₀ = g r̂` exactly — to
conclude that `p̂¹ + ρ⁰φ¹` is a function of `r` alone; (ii) normalise
away the degree-0 parts of `ρ¹` and the boundary perturbations by
absorbing them into the spherical reference (their eq. 11), turning
"function of `r` with zero spherical mean" into zero, so the aspherical
pressure is slaved, `p̂¹ = −ρ⁰φ¹`; (iii) curl again to slave the fluid
density pointwise, `ρ¹ = g⁻¹∂_rρ⁰ φ¹` — precisely the `ρ'_F`
coefficient of the Dahlen loading treatment. What remains free is one
*constant* pressure perturbation per connected fluid region (their
conclusion (iii)): the degree-0 hole, quarantined at `l = 0` because
spherical harmonics decouple the Poisson equation degree by degree.

**About a general (aspherical) hydrostatic reference the argument does
not readily extend, and the failure is structural, not technical.**
Step (i) survives in level-surface form: with `ρ⁰ = ρ(Φ₀)` (forced by
hydrostatics), crossing with `n̂ = ∇Φ₀/g` still slaves the *tangential*
structure, `δp + ρ⁰δφ = f(Φ₀)` on each equipotential. But step (ii) has
no analogue: the free data is now a whole function `f(Φ₀)` per fluid
region, and there is neither a symmetry-generated family of reference
models to absorb it into, nor a spectral decomposition in which it
decouples — level-surface averaging does not commute with the Laplacian
on aspherical equipotentials, so the "degree-0-like" free field couples
to *all* harmonics through the Poisson equation. In the spherical case
the free function collapses to a constant via the decoupled `l = 0`
radial balance; aspherically the corresponding reduction is an implicit,
globally coupled problem on the equipotential foliation (hydrostatic
ellipticity theory being its perturbative instance for rotation). The
elimination that defines Dahlen's fluid treatment therefore stops being
an elimination, and the underdetermination that shows up spherically as
the isolated `l = 0` gap is no longer quarantined. This is the
equilibrium-theory counterpart of the loading-problem findings in
`doc/gauged_fluid.md`: the gauged/referential formulation, which keeps
the fluid's displacement and bulk modulus and never performs the
elimination, is indifferent to all of this — there is no privileged
degree anywhere in it.

A corollary worth keeping in view for model building (AW10, below their
eq. 39): in a hydrostatic model, lateral density variations in the fluid
core are *slaved* to the potential perturbation generated elsewhere;
freely specified core heterogeneity is inconsistent with a hydrostatic
core.

## 7. Implementation map

What the general linearised quasi-static problem needs, against what
exists:

| Term | Status |
|---|---|
| `C ε(u):ε(v)`, 21-component bare tensor | `ElasticTensorIntegrator` (Mandel) — exists |
| PREM → bare conversion `C = C^eff − p⁰(δδ−δδ−δδ)` | small `ElasticTensorCoefficient` decorator — **new**, trivial |
| initial-stress term `∫ T⁰_AB ∂_A u_k ∂_B v_k` | matrix-coefficient vector diffusion — **new integrator**, small |
| background state `Φ₀, p⁰, T⁰` | hydrostatic path exists; general `T⁰` supplied as a coefficient (equilibrium consistency is the modeller's burden) |
| gravity, mixed Eulerian | exists (`self_gravitating.*`) |
| gravity, mixed referential | `TransformedDiffusionIntegrator` + linearised-coefficient coupling — mostly exists via mappings |
| fluid–solid slip, linearised | pairing + penalty/AL machinery exists; the `ϖ⁰ Q/S` equilibrium-geometry terms — **new**, from AC18 eqs. 121/137 |
| `∇∇Φ₀`/centrifugal terms | exist (hydrostatic form); general-`T⁰` arrangement follows (M) with no rearrangement |
| relabelling covariance | the mapping layer *is* AC18 eqs. 134–136 |
| consistency test | hydrostatic `T⁰ = −p⁰1`: general assembly ≡ current implementation, as an operator identity |

Deliberately deferred: time dependence and Coriolis (the Tisserand-frame
machinery of Maitra & Al-Attar 2024), the gravitational stress tensor,
finite deformation (where the referential potential and the slip map
earn their keep), and the stress dependence of the moduli themselves
(`Π` of Maitra & Al-Attar 2021).
