# 3-D GIA modelling codes: status survey

A survey of the numerical codes for 3-D glacial isostatic adjustment and
self-gravitating (visco)elastic loading, drawn from the papers in this
folder plus a web check of current availability (29 September 2026).
Written as background for the 3-D verification campaign: what each code
does, what it assumes, what it cannot do, and where mfemElasticity's
careful treatments (full gravity, meshed and possibly stratified fluid
cores, exact DtN, aspherical boundaries by relabelling) sit relative to
the field. `ggw032.pdf` in this folder is Al-Attar & Crawford (2016), the
theoretical basis of the relabelling benchmarks — it is not a GIA code
paper and is not surveyed here.

## The published benchmarks (both 1-D)

**Spada et al. 2011** (GJI 185, 106; `185-1-106.pdf`). Eight codes —
normal-mode (TABOO, FastLove-HiDeg, MHPLove, VEENT), Post–Widder Laplace
inversion (ALMA), spectral–finite-element (VILMA), ABAQUS FE (flat and
spherical) — on a 3-layer incompressible Maxwell model with elastic
lithosphere and uniform inviscid fluid core (M3–L70–V01). Agreement to
6+ significant figures among the semi-analytic codes; FE within ~10%
(grid effects). Fixes community conventions worth knowing: Wu & Ni (1996)
CMB conditions, degree-1 Love numbers in the CM frame (k₁ᴸ = −1),
hydrostatic initial rotational state. Deliberately excludes
compressibility (dense normal-mode spectra) and the sea-level equation.
Its Tables 3–14 remain the standard reference numbers for validating a
new code on spherical models — the natural targets when we reproduce
figures, with the caveat that the viscoelastic content is normal-mode
based while our comparisons would be time-domain.

**Martinec et al. 2018** (GJI 215, 389; `ggy280.pdf`). Ten codes on the
sea-level equation: five synthetic cases from no-SLE through fixed
coastlines to fully moving coastlines with floating ice, on the same
incompressible non-rotating 1-D model. Agreement ~1.5% max in U and N;
main discrepancy sources are spatial discretisation and Gibbs behaviour
at load margins. Only five of the ten codes could run moving coastlines.
Also 1-D only; 3-D structure explicitly deferred.

**A 3-D benchmark does not exist in print.** The initiative is EGU
abstract EGU22-1447 (Klemann and 16 co-authors, out of the 2021
PALSEA-SERCE workshop) proposing a catalogue of synthetic experiments;
no paper or preprint has followed as of September 2026 (the June 2025
GIA workshop in Sidney BC held a benchmarking breakout, with no published
outcome). Every 3-D code below is verified only against 1-D
semi-analytic references. This is the gap the relabelled-benchmark
methodology addresses: exact aspherical reference solutions without
needing a second 3-D code.

## The 3-D codes

### Martinec spectral–finite-element lineage (VILMA, VEGA)
- Theory: Martinec 1999 (GJI 137, 469; `137-2-469.pdf`) — tensor
  spherical harmonics in angle, FE in radius, explicit time stepping
  with a viscous "memory" term; lateral viscosity couples degrees/orders
  through Clebsch–Gordan machinery. Implemented as Martinec (2000) and
  productionised at GFZ as VILMA (SLE, moving coastlines, rotational
  feedback, climate-model coupling).
- Assumes: elastic moduli and density **radial only** (lateral variation
  in viscosity alone), all boundaries spherical, incompressible (a
  compressible prototype exists), Maxwell, explicit Euler (Δt limited by
  the minimum Maxwell time), practical lateral resolution ~degree 50–128.
- Fluid core: free-slip CMB conditions on an unmeshed core.
- Status: Martinec's own implementation and GFZ's VILMA are not
  distributed ("open source in preparation" per natESM documentation). A
  "VILMA v2" rewrite appeared on GitHub (github.com/fesmc/vilma) in
  September 2026 — created days ago, no license, "not yet ready for
  scientific production use". Worth watching.

### Latychev finite-volume code (Seakon)
- Latychev et al. 2005 (GJI 161, 421; `161-2-421.pdf`): node-centred
  finite volumes on an unstructured tetrahedral grid, all four unknowns
  (u, φ₁) in one monolithic system, GMRES+ILU, MPI. Full first-order
  gravity; elastically compressible but incompressible in the fluid
  limit; arbitrary 3-D variations in viscosity *and* elastic parameters
  across arbitrary internal surfaces. Explicit Euler in time (first
  order, Δt ≤ min Maxwell time). Grids honour radial PREM
  discontinuities — no boundary topography. Core unmeshed: uniform
  incompressible inviscid fluid via CMB condition. No SLE in the 2005
  paper (added later via the Mitrovica–Milne theory); rotation added
  later; recently extended to transient rheology (Lau et al. 2026,
  JGR — search-snippet level only). Modern versions are understood to
  use Bailey's exponential explicit time-stepper in place of forward
  Euler, easing the Δt ≤ Maxwell-time constraint (D. Al-Attar, private
  communication from J. Mitrovica; not in the 2005 paper).
- Status: after two decades of collaboration-only access, the code was
  deposited publicly in May 2026 at the Brown Digital Repository
  (doi 10.26300/y2ct-jp25, CC BY-NC): a ~100 GB snapshot configured for
  192 CPUs accompanying one paper, via Globus. Effectively an archival
  release, not a maintained distribution; no public documentation (a
  draft user guide at Memorial University is an unfinished skeleton).

### Zhong / A / Yuan finite-element lineage (→ CitcomSVE-3.0)
- A, Wahr & Zhong 2013 (GJI 192, 557; `ggs030.pdf`): compressible 3-D FE
  (CitcomS heritage), trapezoid-rule Maxwell stepping, full first-order
  gravity — but the Poisson equation is *not* solved by FE: φ comes from
  spherical-harmonic Green's-function integrals, which **requires the
  background density to be layered (1-D)**; self-gravity enters through
  an iteration (6–8 per step). 3-D viscosity and Lamé parameters; polar
  wander feedback; SLE with partial ocean-function time dependence.
  Degree-dependent errors ~1% (low degrees) to ~3–5% (short wavelengths,
  point GPS rates).
- CitcomSVE-3.0 (GMD 18, 1445, 2025; `gmd-18-1445-2025.pdf`): the
  packaged, parallel (multigrid, >75% efficiency to 6144 cores), open
  release. Compressible since 3.0; fully 3-D viscosity (linear or
  nonlinear) and elastic moduli; full Kendall et al. (2005) SLE with
  moving coastlines; polar wander; CM frame with degree-1 handled. The
  potential is still computed in spherical harmonics to a cutoff degree
  (20–64) — the dominant cost (¼–½ of runtime) and the resolution limit
  on geoid outputs; density must be radially layered; core unmeshed
  (incompressible-core CMB condition); regular spherical grid (no
  unstructured refinement, no boundary topography); trilinear elements,
  second-order convergence; benchmark errors <0.1% to degree 4, <2% to
  degree 16 at ~50 km resolution, degree (2,1)/polar-wander term the
  least accurate (1.4–17.5%).
- Status: genuinely open — github.com/shjzhong/CitcomSVE (LGPL-3.0) and
  Zenodo (10.5281/zenodo.13932410), maintained (pushes through
  Aug 2025), with a manual. The reference open 3-D GIA code at present.

### ABAQUS lineage (Wu → van der Wal → FEMIBSF)
- Huang et al. 2023 (GJI 235, 2231; `ggad354.pdf`): the current state —
  the commercial FE kernel solves ∇·σ = 0, and *all* gravity terms
  (pre-stress advection, perturbed potential, compressibility buoyancy)
  are applied as iteratively updated body/surface forces, with the
  authors' own OpenMP Poisson solver doing direct Green's-function
  volume integration over all elements (no spherical harmonics —
  deliberate, to avoid degree leakage; expensive). Compressible,
  spherical, CM-frame degree-1 handled; core unmeshed (potential stress
  + Winkler buoyancy at the CMB; homogeneous incompressible core).
  Designed for 3-D structure but **validated only on 1-D models**, vs
  analytic homogeneous solutions, VILMA-C and ICEAGE (<1–4% typical).
  Reproduces the Rayleigh–Taylor instabilities of homogeneous
  compressible models. No rotation (sketched only), no SLE in the paper.
- Assumptions/limits: accuracy strongly mesh-dependent (coarse grids
  systematically underestimate the driving potential; degree-15 errors
  can exceed 100% before refinement); hanging nodes forbidden; the
  earlier Wu-method caveat that ABAQUS compressibility lacks
  buoyancy-consistency is what the iterative scheme repairs. Not
  releasable as a whole (commercial kernel); scripts not published. The
  same family includes van der Wal's composite-rheology Fennoscandia and
  Antarctica models and van Calcar's coupled ice-sheet–GIA model.

### Cambridge adjoint spectral code (Lloyd et al. 2024)
- `ggad455.pdf` (GJI 236, 1139): the group's own line — Al-Attar & Tromp
  (2013) / Crawford et al. (2018) rate formulation, generalized
  spherical harmonics to degree 64 × radial spectral elements,
  pseudo-spectral lateral-viscosity terms, non-iterative SLE with
  shoreline migration, and the distinctive capability: adjoint Fréchet
  kernels for 3-D viscosity and initial sea level (verified against
  finite differences). Compressible, full gravity, fluid outer core in
  the formulation.
- Assumes: **1-D elastic and density structure** (3-D in viscosity
  only), no rotation yet, Maxwell (transient implementable), explicit
  RK2 stepping with Δt ~ half the minimum Maxwell time — hence viscosity
  floored at 2×10¹⁹ Pa s, and ~TB-scale storage of forward fields for
  kernels. The degree-64 truncation of the paper is not intrinsic:
  forward runs go fine at degree ~256 on a decent server; 64 reflects
  the adjoint runs' need to store forward fields for kernel
  construction, where a better job could be done but has not been
  (D. Al-Attar).
- Status: revived in 2026 and public on David's GitHub as **sl3d** (two
  versions, 1-D and 3-D); runs locally, needs a server for production
  work. Crawford has left the field and the code had been dormant.

### SPECFEMX (Gharti; not a GIA code yet, but adjacent)
- Spectral-infinite-element method: spectral elements on unstructured
  hex meshes with infinite elements carrying the gravitational potential
  to infinity — the one code family solving the unbounded Poisson
  problem on genuinely 3-D density without spherical harmonics. Papers:
  Gharti & Tromp 2017 (SIEM Poisson), Gharti et al. 2018 (gravity
  anomalies), 2019a (coseismic + post-earthquake viscoelastic
  deformation), 2019b (earthquake-induced gravity perturbations), 2023
  (self-gravitating rotating wave propagation). These are not in this
  folder.
- Status (github.com/homnath/SPECFEMX, GPL-3.0, verified): actively
  developed — commits through July 2026 despite the quiet look of the
  front page. Post-earthquake viscoelastic and gravity-perturbation
  capabilities are marked "experimental" in the manual. The README
  documents in-progress work on sea-level change via the Crawford et
  al. (2018) rate formulation, with open items (ice term missing from
  the weak form, load/BC questions, memory-integral time marching): a
  quasi-static self-gravitating loading + SLE capability is being built
  and is explicitly incomplete. No published GIA application.

### Others, briefly
- **G-ADOPT viscoelastic** (Scott et al., GMD 19, 2717, Apr 2026):
  Firedrake-based FE with *automatically derived adjoints*, compressible,
  lateral and nonlinear/transient rheology — but **no self-gravity yet**.
  Open source; the closest thing to an adjoint-capable community
  competitor, and its missing gravity is exactly the hard part.
- **ASPECT-based GIA** (Weerdesteijn et al. 2023): viscoelastic loading
  in a very active open FEM framework, but Cartesian boxes, no
  self-gravity, no SLE.
- **FastIsostasy** (GMD 2024, Julia, active): deliberately approximate
  regional 2-D (LV-ELVA) solver for coupling to ice-sheet models; not a
  field-equation competitor.
- **ML surrogates** (GMD 2024): emulators trained on 3-D runs, not
  solvers.

## Cross-cutting reading

Setting mfemElasticity's design choices against the field:

1. **Gravity.** All serious codes keep the three first-order gravity
   terms, but the Poisson strategies split into: spherical-harmonic /
   Green's-function solves that force a **1-D background density**
   (A13, CitcomSVE, the spectral codes), direct volume integration that
   is exact but expensive (FEMIBSF, Seakon's boundary integrals), and
   genuine unbounded-domain discretisations (SPECFEMX's infinite
   elements). An FE Poisson solve on the same mesh with an exact DtN
   closure — our arrangement — handles 3-D density at no structural
   cost; in the survey only SPECFEMX is comparably general, and it has
   no working GIA capability yet.
2. **Fluid cores.** Universally unmeshed: every GIA code reduces the
   core to a CMB boundary condition for a uniform (or neutrally
   stratified) incompressible inviscid fluid. None solves for the
   potential inside a stratified fluid core (the ρ′ term). The submesh
   fluid machinery here has no counterpart in the field, and David's
   reservations about the degree-0 fluid treatment (Dahlen's
   formulation) touch physics none of these codes even represents. On
   the campaign's list: implement the Latychev-style approximate CMB
   condition here as an option and compare against the meshed core, to
   quantify whether the universal approximation matters.
3. **Boundary and interface topography.** No surveyed code handles
   aspherical internal boundaries: grids and spectral expansions honour
   spherical PREM interfaces (Seakon's unstructured grid could in
   principle, but the published model does not; CitcomSVE's grid is a
   regular spherical shell). Relabelling carries topography exactly on
   every interface while the computational domain stays spherical —
   both a modelling capability and, via the pulled-back forms, the
   verification methodology.
4. **Verification.** Every 3-D code above is validated against 1-D
   semi-analytic references only, typically to ~1% at low degrees and
   several per cent at short wavelengths; a published 3-D benchmark does
   not exist. Relabelled exact solutions would be the first aspherical
   reference with machine-precision (variant 2a) and
   convergence-certified (variant 2b) content.
5. **Time stepping.** The explicit-Euler Δt ≤ Maxwell-time constraint
   recurs across the field (Martinec lineage, the published Seakon,
   Lloyd et al.) and is what forces viscosity floors; CitcomSVE and A13
   use slightly better trapezoid stepping, and modern Seakon reportedly
   an exponential (Bailey-type) explicit integrator (see above).
   Relevant when our viscoelastic layer meets 3-D models.
6. **Adjoints.** Only the Cambridge line has published GIA sensitivity
   kernels; G-ADOPT has automatic adjoints without gravity. A
   mapping-aware form with F explicit — analytic shape derivatives —
   plus full gravity would be a combination nobody has.

## Sources

Folder PDFs: Martinec 1999 (`137-2-469`), Latychev et al. 2005
(`161-2-421`), Spada et al. 2011 (`185-1-106`), A, Wahr & Zhong 2013
(`ggs030`), Al-Attar & Crawford 2016 (`ggw032`, theory), Martinec et
al. 2018 (`ggy280`), Huang et al. 2023 (`ggad354`), Lloyd et al. 2024
(`ggad455`), Yuan, Zhong & A 2025 (`gmd-18-1445-2025`). Web status
(verified 29 Sep 2026 unless noted): CitcomSVE GitHub/Zenodo; Seakon
Brown deposit bdr:429nt562; natESM VILMA page; github.com/fesmc/vilma;
github.com/homnath/SPECFEMX; GMD G-ADOPT paper; EGU22-1447. The Lau et
al. (2026) Seakon transient-rheology extension and the 2025 workshop
breakout details rest on search snippets only.
