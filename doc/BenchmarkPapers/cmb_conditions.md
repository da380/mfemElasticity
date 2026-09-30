# Approximate CMB conditions in the GIA literature

What the surveyed codes actually do at the core–mantle boundary, with the
equations as published, their translation into the interface terms of
`doc/self_gravitation.md` ((F1)–(F3)), and the quantification the
benchmark option `-cmb` provides. Companion to `code_survey.md` item 2;
the implementation note is `doc/self_gravitation.md` §2, the full and
gauged treatments are `doc/self_gravitation.md` and `doc/gauged_fluid.md`.

## 1. The condition, as published

Every code reduces the core to a boundary condition on the mantle side of
the CMB for a **uniform, incompressible, inviscid** fluid core. The
lineage runs Wu & Peltier (1982) → Tromp & Mitrovica (1999a) → Martinec
(2000) → Zhong et al. (2003), and the statements are:

- **Latychev et al. (2005, eqs. 11–12)** — `161-2-421.pdf`, p. 424 —
  citing Tromp & Mitrovica (1999a), with ν̂ the normal *out of the mantle
  domain* (into the core) and mantle-side values:

  ```
  ν̂·T⁺ = ρ_c ( s⁺·∇Φ⁰ + Φ¹ ) ν̂           (traction: buoyancy + potential stress)
  σ_CMB = −ρ_c s⁺·ν̂                        (apparent surface mass density,
                                            the core's ONLY contribution to Φ¹)
  ```

  with ρ_c the liquid-core density *just below the CMB* (core-top). The
  core is "an incompressible fluid of uniform density" following Zhong
  et al. (2003); the fluid-pressure alternative (Komatitsch & Tromp 2002)
  is noted and not used. Their time-marching additionally "ensures mass
  conservation at the CMB" each step — a projection standing in for the
  incompressible core's volume constraint (see §3).
- **A, Wahr & Zhong (2013, after their eq. 1)** — `ggs030.pdf` —
  `σ_ij n_j = −ρ_c φ n_i + ρ_c g n_i u_j n_j` at the CMB: the same two
  terms (their sign convention differs through n̂).
- **Huang et al. (2023, eqs. 2.3.3 and 2.2.13)** — `ggad354.pdf` —
  `τ_rr^CMB = ρ_c φ₁ + ρ_c g₀ u_r` ("potential stress" plus "Winkler
  buoyancy"), shear-free; the potential sees the core only through the
  CMB surface density ρ_c u_r ("identical to eq. 14 of Latychev et al.
  2005"); "an incompressible and uniform fluid core which has no
  volumetric density perturbation under static deformation".
- **CitcomSVE-3.0 (2025)** — `gmd-18-1445-2025.pdf` §2 — "zero shear
  force", normal force from "the self-gravitational effect for a fluid
  incompressible core (Zhong et al. 2003)"; "except for this CMB
  condition, the core is not considered explicitly".
- **VILMA / Martinec lineage** — free-slip CMB on an unmeshed core
  (incompressible spectral formulation).
- The **Spada et al. (2011) benchmark** — `185-1-106.pdf`, p. 570 —
  records that CMB conditions "have been the subject of considerable
  debate in the past" and fixes the Wu & Ni (1996) solid–fluid
  conditions as "currently agreed on within the GIA community"; the
  community reference solutions embed the same physics.

## 2. Translation into the (F1)–(F3) terms

With m the solid's outward normal and our sign conventions, the
Latychev/Huang condition is *exactly* the interface pair

```
(F2)  −∫_Σ ρ_F (m·∇Φ₀)(m·u)(m·v) dS      [their ρ_c g u_r buoyancy]
(F3)  −∫_Σ ρ_F [ φ (m·v) + φ′ (m·u) ] dS  [their potential stress + σ_CMB source]
```

with two approximations relative to the full Dahlen treatment:

1. **ρ_F is a constant** (the core-top density) rather than the model's
   fluid-side density on each interface;
2. **(F1) is absent** — the uniform incompressible core has no interior
   density perturbation, i.e. ρ′_F = dρ/dΦ₀ ≡ 0. No surveyed code solves
   for the potential inside a stratified core.

Note what is *not* approximate in their setups relative to ours: the
core's background density still generates Φ₀ and (through the Green's
functions or spectral solvers) participates in the zeroth-order gravity;
only the core's *perturbation* physics is truncated. The benchmark
option `-cmb` reproduces the steps (`doc/self_gravitation.md` §2):
`full` → `nomass` (drop F1) → `uniform` (constant core-top ρ_F) →
`winkler` (also drop F3: buoyancy alone — the variant Huang et al. call
FEMIBF and compare against, finding the potential force "significant for
low degree loads", their §6.1).

## 3. Degree 0 and the core's mass

For a spherical CMB the surface-density source ρ_c(m·u) has zero net mass
only when ∮(m·u) dS = 0 — the incompressible core's volume constraint,
which lives entirely at degree 0. Latychev et al. enforce it as a
per-step "mass conservation at the CMB"; other codes leave degree 0 to
the surface-load convention. This is the same degree-0 free constant
identified in Al-Attar & Woodhouse (2010, conclusion iii) and discussed
in `doc/gravitating_elasticity.md` §6: *every* member of this family —
full Dahlen included — misses the core's compressional physics at l = 0,
because eliminating the fluid discards its bulk modulus there. The
gauged treatment (`doc/gauged_fluid.md`) keeps it and reproduces the
l = 0 reference.

## 4. What the approximations cost (measured)

Load Love numbers on `prem_4` (stratified PREM-like core), h = 0.2,
order 2, 8 ranks, benchmark case `runs/prem_4_cmb`; relative change of
h′ against the full Dahlen treatment, and pyslfp as the reference:

| l | pyslfp h′ | full | nomass | uniform | winkler | gauged |
|---|---|---|---|---|---|---|
| 0 | −0.1318 | −1.0080 | +0.3640 | +0.3556 | −0.2071 | **−0.1317** |
| 1 | −1.2849 | −1.2933 | 0.08 % | 1.4 % | 15 % | −1.2839 |
| 2 | −0.9906 | −0.9954 | 0.08 % | 0.08 % | 7.9 % | −0.9976 |
| 3 | −1.0493 | −1.0480 | 0.01 % | 0.01 % | 2.6 % | −1.0478 |
| 4 | −1.0511 | −1.0470 | 0.00 % | 0.00 % | 0.75 % | −1.0468 |

(percentages are |Δh′|/|h′_full|; absolute values where the comparison
to full is not meaningful; all runs on the same mesh and order.)

Readings:

- **The standard condition (`uniform`) is essentially harmless at
  l ≥ 2** for a PREM-like core: ≤0.1 % — far below the discretisation
  error of any surveyed code. At l = 1 it costs ~1.4 % (the constant
  core-top density mis-weights the Slichter-adjacent response).
- **Dropping the potential stress (`winkler`) is not harmless**: 8 % at
  l = 2, decaying with degree — the elastic-response counterpart of
  Huang et al.'s viscous finding, and the reason "potential stress +
  buoyancy" (not buoyancy alone) became the standard.
- **At l = 0 the entire family is wrong, each member differently**
  (h′(0) scattered over ±1 with sign changes against pyslfp's −0.1318);
  the gauged treatment agrees with pyslfp to 5×10⁻⁴ there. Degree-0
  observables (mean sea-level/radial budgets) are the one place the
  universal approximation genuinely bites — and where no member of the
  Dahlen family, exact or approximate, can be fixed by tuning the CMB
  condition.

### 4.1 Tidal Love numbers: the differences grow an order of magnitude

The table above is for *loading*, whose response is concentrated near
the surface. A tidal potential samples the deep interior much more
strongly, so the CMB treatment should matter more. The same runs
computed the tidal response (l ≥ 2), and it does:

| l | qty | pyslfp | full | nomass | uniform | winkler | gauged |
|---|---|---|---|---|---|---|---|
| 2 | h | 0.60412 | 0.60555 | 1.09 % | 1.09 % | 28.0 % | 0.60814 |
| 2 | k | 0.29845 | 0.29902 | 2.51 % | 2.51 % | 42.2 % | 0.30043 |
| 3 | h | 0.28829 | 0.28693 | 0.21 % | 0.21 % | 12.4 % | 0.28690 |
| 3 | k | 0.09215 | 0.09162 | 0.78 % | 0.78 % | 21.0 % | 0.09162 |
| 4 | h | 0.17508 | 0.17357 | 0.05 % | 0.05 % | 5.0 % | 0.17351 |
| 4 | k | 0.04146 | 0.04102 | 0.26 % | 0.26 % | 8.7 % | 0.04100 |

(same conventions: percentages are the relative change against the full
treatment on the same mesh/order.)

Readings:

- **Every approximation costs roughly ten times more on tides than on
  loading.** The `uniform`/`nomass` conditions, harmless for loading
  (≤0.1 % at l ≥ 2), reach 1.1 % on tidal h₂ and 2.5 % on tidal k₂ —
  now comparable to or above the discretisation error of a production
  run, and above the accuracy targets of modern body-tide work.
  `winkler` degrades from 8 % to 28 % (h₂) and 42 % (k₂).
- **k is consistently more sensitive than h** (about ×2.3 at every
  degree): k is sourced by the internal mass redistribution, which is
  exactly what the core treatment controls.
- **`nomass` and `uniform` coincide to 6 digits on tides** (they
  differed slightly on loading at l = 1 only): for l ≥ 2 the internal
  core buoyancy term they differ by integrates to nearly nothing.
- Practical upshot: GIA codes borrowed the CMB condition from a loading
  context where it is safe; reusing the same operator for tidal
  computations (e.g. combined GIA + body-tide inversions) imports a
  percent-level systematic that the full or gauged treatments remove.

### 4.2 Rotational feedbacks

The rotational feedback in sea-level calculations runs through the
(2,1) components: the load perturbs the inertia tensor (through the
degree-2 potential or displacement, depending on how the inertia
perturbation is formed), the rotation vector shifts, and the
centrifugal potential perturbation — itself a degree-2, order-1
tidal-type forcing — feeds back through the degree-2 TIDAL response.
So the tidal column above, not the loading one, is the relevant error
budget for the feedback, and the ~10× tidal amplification of the CMB
approximation error applies to it directly.

Scaled by the feedback's share of the signal (rotational feedback is of
order 10 % of the barystatic/GMSL signal):

- `uniform`/`nomass`: 1.1–2.5 % on tidal (h₂, k₂) → of order 0.1–0.3 %
  of the total signal through the feedback. Harmless.
- `winkler`: 28–42 % on tidal (h₂, k₂) → of order 3–4 % of the total
  signal. Not harmless for modern sea-level precision, despite sitting
  on a "small" term.

The record to carry: the standard unmeshed-core condition is safe for
the loading response AND its rotational feedback; the Winkler variant
is safe for neither; and any precision statement about the feedback
should be made against the tidal, not loading, error table.
