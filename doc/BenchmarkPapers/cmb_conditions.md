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
