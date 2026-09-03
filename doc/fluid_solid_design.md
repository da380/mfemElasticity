# Fluid–solid ("mixed") self-gravitating problems, and the anisotropic problem layer — design plan

*2 September 2026. Companion to `status_and_roadmap.md` §8 step 7 and to `submesh_coupling_design.md` §4. Sources: Yu, Al-Attar, Syvret & Lloyd (2025, ggae388) Appendix A, which transcribes the bilinear form of Al-Attar & Tromp (2014); David's unpublished Love-number notes (`~/Documents/MyPapers/Unpublished/love/notes.tex`), which correct a typo in (A2); the earlier `elastogravity_two_layer` / `elastogravity_three_layer` examples.*

Part I (§1–§6) is the fluid–solid extension of `LinearQuasiStaticSelfGravitatingProblem`. Part II (§7) is the anisotropic extension of `LinearQuasiStaticProblemBase` / `ViscoelasticOperator`, which is independent and can be done at any time. §8 lists the decisions I would like confirmed before writing code.

Everything in §1 is written so that it can be checked line by line against (A2)–(A3). Equations that I have *rewritten* (rather than transcribed) are marked "(rewritten)" and come with the derivation.

---

## Part I — fluid regions

## 1. Continuous problem

### 1.1 Geometry and notation

- M = M_S ∪ M_F: the body; M_S the solid regions (one or more), M_F the fluid regions. Regions are nested; the outermost is solid (no ocean; PREM is used with the ocean stripped, not replaced by solid). Internal solid–solid interfaces need nothing special.
- B ⊃ M: the computational ball with spherical outer boundary and the DtN operator, as now.
- Σ_FS: fluid–solid interfaces with the fluid on the *inner* side (CMB); Σ_SF: fluid *outer* side (ICB). n̂ is the "upward" normal on every interface (pointing from the − side to the + side, i.e. radially outward). ρ⁻, ρ⁺ the densities on the lower/upper side.
- Φ₀ the background potential, ∇Φ₀ = g n̂ on level surfaces with g = |∇Φ₀| > 0. Hydrostatic equilibrium makes every fluid–solid interface a level surface of Φ₀, and makes the fluid barotropic: ρ = ρ(Φ₀) in M_F.
- σ the surface mass load on ∂M (positive = mass added), ψ an applied (tidal) potential.

### 1.2 The bilinear form (A3), transcribed

A(u,φ | u′,φ′) =
  ∫_{M_S} κ (div u)(div u′) + ∫_{M_S} 2μ d : d′
  + ½ ∫_{M_S} ρ [ ∇(u·∇Φ₀)·u′ + ∇(u′·∇Φ₀)·u ]
  − ½ ∫_{M_S} ρ [ (u·∇Φ₀) div u′ + (u′·∇Φ₀) div u ]
  + ∫_{M_S} ρ ( ∇φ·u′ + ∇φ′·u )
  + (1/4πG) ∫_{ℝ³} ∇φ·∇φ′
  + ∫_{M_F} g⁻¹ ∂_r ρ  φ φ′                                                  (F1)
  + ∫_{Σ_FS} ρ⁻ g (n̂·u)(n̂·u′) dS − ∫_{Σ_SF} ρ⁺ g (n̂·u)(n̂·u′) dS               (F2)
  + ∫_{Σ_FS} ρ⁻ (φ u′ + φ′ u)·n̂ dS − ∫_{Σ_SF} ρ⁺ (φ u′ + φ′ u)·n̂ dS.            (F3)

The first six lines are what `LinearQuasiStaticSelfGravitatingProblem` already assembles (with M_S = M). (F1)–(F3) are new. A is symmetric, and the rigid-mode identity of the main text (eq. 9) is retained.

### 1.3 The equation (A2), with the typo corrected

A(u,φ | u′,φ′) + ∫_{∂M} (u′·∇Φ₀ + φ′) σ dS
  + ∫_{M_S} ρ u′·∇ψ + ∫_{M_F} g⁻¹∂_rρ φ′ ψ + ∫_{Σ_FS} ρ⁻ ψ n̂·u′ dS − ∫_{Σ_SF} ρ⁺ ψ n̂·u′ dS = 0,

for all (u′, φ′). The last surface integral is over Σ_SF (the notes' correction; (A2) as printed has Σ_FS twice). With ψ = 0 this is the current class's equation plus (F1)–(F3).

### 1.4 The fluid volume term in general form (rewritten)

Barotropy gives ∇ρ = (dρ/dΦ₀) ∇Φ₀ in M_F, so

  g⁻¹ ∂_r ρ = dρ/dΦ₀ = ∇ρ·∇Φ₀ / |∇Φ₀|²  =: ρ′_F .

I propose the library takes ρ′_F as a user coefficient (radial models have dρ/dr and g analytically), with a helper that evaluates ∇ρ_h·∇Φ₀/|∇Φ₀|² from a projected density when no analytic form exists (this is what the examples do, with a radial derivative; noisy at order 1). Note ρ′_F < 0 wherever density increases downward, so (F1) is a *negative* mass term on the fluid — see §3.2 for the consequence.

### 1.5 The interface terms in one convention (rewritten)

Let Σ_F = Σ_FS ∪ Σ_SF and let **m be the outward normal of the solid** on Σ_F (it points into the fluid on both kinds of interface: m = −n̂ on Σ_FS, m = +n̂ on Σ_SF). Let ρ_F be the density on the fluid side (ρ⁻ on Σ_FS, ρ⁺ on Σ_SF). Then, using ∇Φ₀ = g n̂ on the (level) interface, so that m·∇Φ₀ = −g on Σ_FS and +g on Σ_SF:

  (F2) = −∫_{Σ_F} ρ_F (m·∇Φ₀) (m·u)(m·u′) dS,                                  (F2′)
  (F3) = −∫_{Σ_F} ρ_F [ φ (m·u′) + φ′ (m·u) ] dS,                               (F3′)
  tidal surface terms = −∫_{Σ_F} ρ_F ψ (m·u′) dS.                               (T′)

Check on Σ_FS (m = −n̂, m·∇Φ₀ = −g): (F2′) = +ρ⁻ g (n̂·u)(n̂·u′), (F3′) = +ρ⁻ φ (n̂·u′) + … ✓. On Σ_SF (m = n̂): −ρ⁺ g (…), −ρ⁺ φ (n̂·u′) ✓.

Physical reading: the Lagrangian pressure perturbation on the fluid side is p^L = p₁ + u·∇p = −ρ_F (φ + u·∇Φ₀), and u·∇Φ₀ = (m·u)(m·∇Φ₀) on a level surface; the traction on the solid is −p^L m, and moving −∫_Σ t·u′ to the left-hand side gives exactly (F2′) + the u′-half of (F3′). The φ′-half of (F3′) is the surface mass ρ_F (m·u) displaced across the interface as seen by Poisson's equation, and (F1) is the Eulerian density perturbation ρ₁ = ρ′_F φ in the fluid.

The point of the rewriting: **the code never has to classify an interface as FS or SF, and needs no sign parameter.** Every interface integral uses the solid SubMesh's own outward normal and the fluid-side density; the sign of m·∇Φ₀ does the rest. This removes the `sign = ±1` arguments of the examples' `BoundaryFluxMixedIntegrator` and the two hand-written variants of the (n·u)(n·u′) term.

### 1.6 Rows of the system, as the code splits them

With c(φ, u′) := ∫_{M_S} ρ ∇φ·u′ − ∫_{Σ_F} ρ_F φ (m·u′) dS (the coupling form, symmetric in the sense c(φ,u′) = c(φ′,u) with roles swapped), a(u,u′) the elastic-plus-gravity displacement form already assembled, a_Σ(u,u′) := (F2′), m_F(φ,φ′) := (F1), and the current sign conventions ℓ_u(u′) = −∫_{∂M} σ ∇Φ₀·u′, ℓ_φ(φ′) = −∫_{∂M} σ φ′:

  displacement row:  a(u,u′) + a_Σ(u,u′) + c(φ,u′)                   = ℓ_u(u′) − c(ψ,u′),
  potential row:     (1/4πG)[∫_B ∇φ·∇φ′ + DtN(φ,φ′)] + m_F(φ,φ′) + c(φ′,u) = ℓ_φ(φ′) − m_F(ψ,φ′).

In matrix form with the code's blocks: [A_uu + A_Σ, C; Cᵀ, A_φφ + M_F] [u; φ] = [B_u − C Ψ; B_φ − M_F Ψ], where Ψ is ψ interpolated on the potential space, A_φφ = (K + DtN)/(4πG) as now and M_F unscaled. **The tidal load is the same operators applied to Ψ**: no new assembly, one matrix–vector product per block. (Check: for the pure-solid case the tidal displacement load is −∫ρ ∇ψ·u′ and the potential load −m_F(ψ,·) = 0, as expected from (A2).)

### 1.7 Null space

- **Global rigid modes** u_r = a + b×x on M_S with φ_r = −u_r·∇Φ₀ *on all of ℝ³* (the rigid field extended into the fluid and beyond) remain exact null vectors of A: the paper states eq. (9) is retained. In the fluid, φ_r gives ρ₁ = ρ′_F φ_r = −u_r·∇ρ, the Eulerian density change of a rigidly moved fluid, and on the interfaces (m·u_r) φ′ balances the Lagrangian/Eulerian bookkeeping. Discretely they are near-null, as now, and the existing construction φ_r = −A_φφ⁻¹ Cᵀ u_r carries over unchanged provided C and A_φφ include the new terms.
- **Enclosed solid regions** (a solid inner core inside a fluid): a rotation u = b×x restricted to the inner core (zero elsewhere) is an exact null vector whenever ρ and Φ₀ are invariant under rotation about b within and on the region — for a spherically symmetric inner core: e(u) = 0, div u = 0, u·∇Φ₀ = 0, m·u = 0 on the ICB, and ∫ρ (b×x)·∇φ′ = −b·∮ ρ φ′ (n̂×x) dS = 0. So for spherical models there are 6 + 3 (2-D: 3 + 1) near-null modes; for aspherical inner cores the extra three are physically near-null with a restoring torque of the order of the asphericity, and projecting them out is the sensible quasi-static regularisation. A translation of the enclosed region is **not** null (Slichter-type gravitational restoring force), and must not be projected.
- The earlier three-block projector projects only the global modes (plus the 2-D constant). The inner-core rotations are therefore unregularised in `elastogravity_three_layer`; MINRES survives because the load is nearly orthogonal to them, but the converged u_ic can carry an arbitrary rotation.
- **2-D**: the constant potential is a null vector of (K + DtN) but *not* of (K + DtN)/(4πG) + M_F. Physically the constant is a gauge (φ → 0 at infinity cannot be imposed in 2-D) and ρ₁ = ρ′_F φ is not gauge invariant, so 2-D + fluid is inconsistent at the level of a constant. The discrete operator is then non-singular but has a near-null direction with a tiny (negative) eigenvalue ≈ ∫_{M_F} ρ′_F / |B|, and any net mass in the load would be absorbed by a huge constant potential. I recommend keeping the 2-D machinery exactly as it is (make the load compatible, project the constant out of every potential solve and of the coupled system), now as an explicit regularisation rather than a null-space fact. This is also what the examples do. 3-D is unaffected.

---

## 2. Discretisation

### 2.1 Unknowns and meshes: one displacement field on a possibly disconnected SubMesh

The three-layer model needs displacement on the inner core and on the mantle. Two options:

(a) two displacement fields on two SubMeshes (the earlier examples; `NumDisplacementFields() = 2` in the interface, 3×3 block operator, two shadows, two injections, two rigid-mode sets);

(b) **one** `SubMesh::CreateFromDomain(parent, {inner core, mantle})` with one H1 vector space, one displacement GridFunction, one stiffness form (materials piecewise by attribute via `PWCoefficient`; the SubMesh inherits the parent's attributes), one shadow of φ, one injection, and the interfaces as boundary attributes of that single SubMesh.

I recommend (b). Everything in the current class works unchanged; the viscoelastic layer sees one field (internal variables also live on the elastic inner core, with μ_k = 0 or τ_k = ∞ there, which is harmless); the only new null-space item is the region-restricted rotations of §1.7, which are trivial to add because the components share no dofs. Option (a) buys nothing for nested Earth models (`submesh_coupling_design.md` §4 already concluded sibling submeshes are not needed) and costs a lot of bookkeeping. **Is a disconnected SubMesh allowed?** MFEM's documentation of `CreateFromDomain` says the attributes "have to mark exactly one connected subset of the parent Mesh", but nothing in the implementation checks or uses connectivity: `SubMeshUtils::AddElementsToMesh` copies elements attribute by attribute, the vdof map is element-based, and `ParSubMesh`'s shared-entity groups are per vertex/edge/face. Verified empirically on 2 Sep 2026 on `data/elastogravity_three_layer_2d.msh` (order-2 gmsh, attributes {1, 3}), serial and at 1–4 ranks: the SubMesh has two connected components (644 elements, 116 boundary elements), attributes {1, 3} and boundary attributes {1, 2, 3} inherited; `Transfer` parent→sub is exact; `SubMeshDofInjection` satisfies PᵀP = I; a `(Par)SubMeshMixedBilinearForm` with a domain mass integrator and a boundary mass integrator on the ICB and CMB gives 1ᵀA1 = |M_S| + |ICB| + |CMB| to 4×10⁻⁵ (the geometric error of the order-2 boundary), identically at every rank count. So option (b) stands; the Phase-1 test suite keeps this check (and adds 3-D). The documented restriction is worth a note to the MFEM developers rather than a workaround here.

Interface boundary attributes: the gmsh meshes define ICB/CMB/surface as physical surfaces, so the SubMesh inherits attributes 1/2/3 (three-layer example) and no "cut" attribute appears; where a parent has no boundary elements on an interface, the cut gets max+1, as documented in `submesh.hpp`.

### 2.2 Each new term → assembly

| Term | Where it lives | How |
|---|---|---|
| (F1) m_F(φ,φ′) = ∫_{M_F} ρ′_F φφ′ | parent mesh, fluid attributes | `MassIntegrator(rho_F_prime)` with a domain-attribute marker on `k_phi_form_`'s sibling; A_φφ ← (K + DtN)/(4πG) + M_F |
| (F2′) a_Σ(u,u′) = −∫_{Σ_F} ρ_F (m·∇Φ₀)(m·u)(m·u′) | solid SubMesh boundary, interface marker | new `BoundaryNormalNormalIntegrator(q)` in `bilininteg.hpp`, q = −ρ_F (m·∇Φ₀) via a new `BoundaryNormalDotCoefficient(∇Φ₀)` (§2.3); added to `StiffnessIntegrators()` as a boundary integrator — it is reassembled with the shear modulus, which is harmless |
| (F3′) coupling −∫_{Σ_F} ρ_F φ (m·u′) | `SubMeshMixedBilinearForm(fes_phi, fes_u)` boundary integrator on the SubMesh | new mixed `BoundaryNormalScalarIntegrator(q)` (trial scalar, test vector), q = −ρ_F; Cᵀ by transposition as now |
| tidal load | both rows | Ψ = interpolate ψ on `fes_phi`; B_u −= C Ψ, B_φ −= M_F Ψ (true dofs) |
| background potential Φ₀ | parent | source −(ρ_S injected from the SubMesh) − (ρ_F assembled on the parent's fluid attributes) |

The interface integrators evaluate the normal by `CalcOrtho` of the boundary transformation, as the examples do, which is outward for correctly oriented boundary elements. Checked in the same 2 Sep experiment with MFEM's `BoundaryNormalLFIntegrator` (which uses exactly that normal): ∫ x·m dS over the ICB from the inner-core side plus the CMB from the mantle side equals 2π(R_ICB² − R_CMB²), i.e. m is outward from the solid on both inherited interior interfaces, in serial and at 1–4 ranks; over the whole SubMesh boundary it equals 2|M_S|. Phase 1 keeps this as a test and adds 3-D and a *cut* interface (parent without boundary elements there). If a case turns up where orientation is not guaranteed, the integrators can orient m through the adjacent element (`Tr.mesh->GetBdrElementAdjacentElement`), a five-line guard.

### 2.3 New small classes

- `BoundaryNormalNormalIntegrator(Coefficient& q)`: ∫ q (m·u)(m·v) on boundary elements, vector H1 (promotion of the examples' `BoundaryFluxIntegrator`, without the sign argument).
- `BoundaryNormalScalarIntegrator(Coefficient& q)`: ∫ q ψ (m·v), scalar trial, vector test (promotion of `BoundaryFluxMixedIntegrator`).
- `BoundaryNormalDotCoefficient(VectorCoefficient& V)`: V·m at boundary quadrature points (needs the boundary `ElementTransformation`; asserts `ElementType == BDR_ELEMENT`).
- `BarotropicDensityGradientCoefficient(const GridFunction& rho, const GridFunction& phi0)`: ∇ρ·∇Φ₀/|∇Φ₀|², optional convenience.

All CPU, serial and parallel (element-local). gtests: analytic values on a circle/sphere (u = x gives ∫ q r² dS; ψ = 1, v = x gives ∫ q r dS), transposition consistency between the mixed integrator and its `TransposeIntegrator`, and the orientation check above on a SubMesh cut and on an inherited interface.

### 2.4 Coefficient requirements (to document on the API)

- ρ_S (solid): evaluated on the SubMesh, as now.
- ρ_F: evaluated on the *parent's* fluid elements (for Φ₀ and, through ρ′_F, for M_F) **and** on the *solid SubMesh's boundary elements* on Σ_F (for (F2′),(F3′)). A `FunctionCoefficient` of position serves both; a `PWCoefficient` keyed by domain attribute does not (boundary transformations carry the boundary attribute), so the API accepts a separate `interface_density`, defaulting to `density`.
- ρ′_F: parent's fluid elements only.

---

## 3. Solvers

### 3.1 Block MINRES (default) and Schur CG

Both carry over. The block operator gets A_Σ inside the (0,0) block (it is part of the stiffness form), the interface term inside C, and M_F inside A_φφ. The projector gets the region rotations of §1.7 (with their discrete φ_r = −A_φφ⁻¹Cᵀu_r, which is ≈ 0). `RigidModeResiduals()` reports the extra modes too. Gauge: unchanged (u ⊥ all projected modes, φ solved from u).

### 3.2 Definiteness of the potential block — a check David can do in his head

A_φφ + M_F = (K + DtN)/(4πG) + M_F with M_F ≤ 0. The inner potential solves (CG now) and the Schur complement assume this is SPD. Crude bound for a uniform fluid ball of radius R with the exterior harmonic extension: the smallest eigenvalue of the Laplace–DtN part relative to the L² norm on the ball is attained by φ = sin(kr)/r matched to A/r outside, which first becomes singular at kR = π/2 with k² = 4πG|ρ′_F|. PREM outer core: ∂_rρ ≈ −1.0×10⁻³ kg m⁻⁴, g ≈ 8 m s⁻², so |ρ′_F| ≈ 1.3×10⁻⁴ kg m⁻⁵ s² and k² ≈ 1.1×10⁻¹³ m⁻², k·R_CMB ≈ 1.1 against the singular value 1.57 (the shell geometry, the inner core and the mantle's Dirichlet energy all push the true margin wider). So the block is expected positive for Earth-like models but not by a wide margin, and it is *not* guaranteed for larger or steeper fluid bodies. The term is small in the sense David means (lower order, no derivatives), but it eats a good fraction of the coercivity constant. Consequences I propose:

- Preconditioner P_φ stays the SPD shifted Laplacian without M_F (as in the examples).
- With fluid regions present, the inner potential solver becomes MINRES (preconditioned by P_φ) instead of CG, so that a slightly indefinite block still solves; MINRES on an SPD matrix costs the same as CG.
- A one-off diagnostic `PotentialBlockMinEigenvalue()` (a few Lanczos steps) so a user can see the margin.

### 3.3 Two-dimensional problems

As §1.7: keep `MakeCompatible`, the `OrthoSolver` on the potential solves and the constant in the block projector; document that with fluid regions this is a regularisation. The preconditioner is unaffected.

---

## 4. API

Constructor argument rather than a later `Add...()` call, because Φ₀, the coupling, the gravity integrators and the null space are all built in the constructor and all depend on the fluid data:

```cpp
struct FluidRegion {
  mfem::Array<int> attributes;          // parent-mesh attributes of the fluid
  mfem::Coefficient* density;           // rho_F on those elements (for Phi_0)
  mfem::Coefficient* density_gradient;  // rho'_F = d rho/d Phi_0 = g^-1 d_r rho
  mfem::Array<int> interface_marker;    // SubMesh bdr attributes of its fluid–solid interfaces
  mfem::Coefficient* interface_density = nullptr;  // rho_F on those boundary elements; default density
};

LinearQuasiStaticSelfGravitatingProblem(fes_u, fes_phi, rheology, density_S, G, dtn_degree,
                              background_potential = nullptr,
                              const std::vector<FluidRegion>& fluids = {});

/// Near-null rotations of a solid region enclosed by fluid (SubMesh attributes).
void AddRegionRotations(const mfem::Array<int>& solid_attributes);

/// Applied potential psi (registered time-dependent); evaluated on the parent.
void SetTidalPotential(mfem::Coefficient& psi);

/// Diagnostics
std::vector<mfem::real_t> RigidModeResiduals();   // now includes region rotations
mfem::real_t PotentialBlockMinEigenvalue(int lanczos_steps = 30);
```

`SetSurfaceLoad`, `ExternalLoad`, `ExternalPotentialLoad`, `AddForce`, `SetEffectiveShearModulus`, the two solver types and all outputs are unchanged. The pure-solid case is `fluids = {}` and behaves bit-for-bit as now (the new integrators are not added, the inner solver stays CG).

The examples' `TwoBlock`/`ThreeBlockRigidBodySolver(Parallel)`, `BoundaryFluxIntegrator`, `BoundaryFluxMixedIntegrator`, `RadialDerivativeCoefficient`, `NormCoefficient` in `examples/common.hpp` then have no remaining users once `elastogravity_two_layer`/`_three_layer` become drivers of the class (§5.3), and go, as agreed on 2 Sep.

---

## 5. Tests and verification

### 5.1 Unit level (gtest, serial + MPI)

1. The three integrators/coefficients of §2.3 against analytic surface integrals; orientation of m on SubMesh interfaces (cut and inherited), 2-D and 3-D, 1/2/4 ranks.
2. Symmetry of the assembled block operator with fluid terms: |⟨Ax,y⟩ − ⟨x,Ay⟩| ≤ 1e-12 ‖A‖‖x‖‖y‖ for random x, y (catches a sign slip in either half of (F3′)).
3. Rigid-mode residuals: global modes and inner-core rotations decrease with order on the canned two-/three-layer meshes; inner-core *translation* residual does **not** vanish (guards against projecting it by mistake).
4. Tidal load consistency: for a degree-1 ψ = a·x (a uniform field) the load −[C Ψ; M_F Ψ] is, in the continuum, exactly the rigid-mode component that the projector removes, so the solved displacement must be O(rigid-mode residual); and the tidal load of the pure-solid case must equal −∫ρ ∇ψ·u′ assembled directly.
5. Solver equivalence: SchurCG and BlockMINRES agree to the rigid-mode residual level with fluid regions, as they do without.

### 5.2 Against the earlier examples

At order 2 on `elastogravity_two_layer_2d.msh` and `elastogravity_three_layer_2d.msh`, ‖u‖ and ‖φ‖ from the class versus the examples, expecting agreement to ~5 figures as in the pure-solid case (`status_and_roadmap.md` §8 step 4). Caveat from `submesh_coupling_design.md` §7: those examples were never run with the interface terms before the injection rewrite, so any number recorded before 2 Sep is *without* (F2)–(F3); the comparison is new-code vs new-code. A further sanity check: make the outer core solid on the same mesh and compare with the pure-solid class; the two answers differ by the fluid physics, so this only checks that the interface and volume terms are of the expected size, not their correctness.

### 5.3 Physics: Love numbers against the radial codes

`data/prem.nocrust` is in the tree. Following the notes: for each degree l solve three forced problems (unit surface displacement-load, unit gravitational-load, unit tidal potential ψ = (r/a)^l Y_lm) and read h_l, k_l, h^t_l, k^t_l from the surface spherical-harmonic coefficients; compare with pyslfp/gia3D (David's codes) for a PREM-like model with a fluid outer core and a solid inner core, l = 2…8 say, at order 2–3 in 3-D (`concentric_spheres` generates the mesh). Degree 1: fix the potential's degree-1 part at the surface, as the notes say, i.e. work in the geocentre frame — in the FEM this is the rigid-translation projection already in place. This is the first real physics verification of the whole stack (roadmap step 4's open item) and the fluid work should not be called done before it.

### 5.4 Examples

`elastogravity_two_layer(_p)` and `elastogravity_three_layer(_p)` rewritten as drivers of the class (same command lines, same meshes), and the `self_gravitating_elasticity(_p)` driver gains `-fluid` options; a `love_numbers(_p)` driver for §5.3.

---

## 6. Phases

1. **Integrators and coefficients** (½–1 d): §2.3 with tests, including the orientation question.
2. **Fluid regions in the class** (1–2 d): `FluidRegion`, Φ₀ source, M_F, A_Σ, interface coupling, tidal load, MINRES inner solver when fluids are present, region rotations, diagnostics; tests 5.1.2–5.1.5; serial and parallel together as before.
3. **Example migration** (½ d): §5.4 and the `common.hpp` deletion; §5.2 comparison recorded in this document.
4. **Love-number benchmark** (1–2 d, needs David's radial numbers): §5.3. Also closes roadmap step 4's open item for the pure-solid case (run the same driver with the core made solid).
5. **Viscoelastic relaxation of a self-gravitating body with a fluid core** (½ d once 4 is done): a GIA-style run with a Maxwell mantle and elastic inner core through `ViscoelasticOperator`, checked against the radial codes' time-domain Love numbers.

---

## Part II — anisotropic elastic and viscoelastic problems

## 7. Extending the problem layer to anisotropy

### 7.1 What exists

`elastic_tensor.hpp` (2 Sep): Mandel-basis reduced tensors in the library's `SymmetricMatrixIndex` order, `IsotropicElasticTensorCoefficient`, `TransverselyIsotropicElasticTensorCoefficient` (Love's A, C, F, L, N and a radial or given axis), `VoigtElasticTensorCoefficient`, `RotatedElasticTensorCoefficient`, `DeviatoricProjectionElasticTensorCoefficient` (P_dev C P_dev and its complement), `ElasticTensorIntegrator(MatrixCoefficient&)`. All tested. Nothing in the problem layer uses them.

The problem layer is isotropic by construction in three places: `IsotropicMaxwellRheology` (κ, μ_∞, branches (μ_k, τ_k)); `LinearQuasiStaticProblemBase` assembles two `ElasticityIntegrator`s and swaps a *scalar* effective shear modulus; `ViscoelasticOperator` keeps *trace-free* internal variables m_k with force Bᵀ(2μ_k m_k) through the trace-free coupling form B and forms μ_eff = μ_∞ + Σ_k β_k μ_k pointwise.

### 7.2 Equations

Anisotropic generalised Maxwell body (viscoelastic plan line 30):

  σ = C_U ε − Σ_k C_k m_k,   C_U = C_∞ + Σ_k C_k,   ṁ_k = (ε − m_k)/τ_k,

with C_k the relaxable tensor of branch k — a modelling choice made by the coefficient (P_dev C P_dev; or "L and N only" for TI; the isotropic case is C_k = 2μ_k P_dev, for which only dev ε matters and m_k can stay trace-free). The effective-modulus elimination used by backward Euler and the exponential trapezoid becomes

  C_eff = C_∞ + Σ_k β_k(x) C_k,   β_k = β(dt/τ_k(x)) the same scalar weights as now,

and the branch force is Bᵀ_sym (C_k m_k) with the *full* symmetric coupling form (`DomainSymmetricMatrixStrainIntegrator`, n_s components), since C_k m_k is not trace-free in general (viscoelastic plan item 7). The stepping formulas (ETD1, exponential trapezoid, BE/SDIRK) are unchanged: they act componentwise on m_k with scalar τ_k.

Self-gravitation is untouched: the gravity terms, the coupling and the fluid terms do not involve C.

### 7.3 Design: let the rheology own the "coefficient to assemble with"

Smallest change that keeps one code path for isotropic and anisotropic:

```cpp
class Rheology {                       // abstract; IsotropicMaxwellRheology derives
 public:
  virtual int SpaceDim() const = 0;
  virtual int NumBranches() const = 0;
  virtual mfem::Coefficient& RelaxationTime(int k) const = 0;
  virtual bool TraceFreeInternalVariables() const = 0;   // isotropic: true
  /// Add the unrelaxed stiffness integrators to `form` (called once).
  virtual void AddStiffnessIntegrators(mfem::BilinearForm& form) = 0;
  /// Point the stiffness at C_inf + sum_k beta_k C_k (beta_k nodal scalar coefficients);
  /// nullptr restores the unrelaxed modulus. Implementations swap a redirectable coefficient.
  virtual void SetRelaxationWeights(const std::vector<mfem::Coefficient*>* beta) = 0;
  /// Apply C_k at a point (Mandel), for the branch force; isotropic: 2 mu_k I on the trace-free part.
  virtual void BranchModulus(int k, mfem::ElementTransformation&, const mfem::IntegrationPoint&,
                             mfem::DenseMatrix& Ck) const = 0;
};
class AnisotropicMaxwellRheology : public Rheology {   // C_inf, (C_k, tau_k) as MatrixCoefficient/Coefficient
  // AddStiffnessIntegrators: one ElasticTensorIntegrator(C_current) with C_current a redirectable
  // MatrixCoefficient; SetRelaxationWeights builds MatrixSumCoefficient(C_inf, sum ScalarMatrixProduct(beta_k, C_k)).
};
```

- `LinearQuasiStaticProblemBase` stops knowing about κ and μ: it calls `rheology.AddStiffnessIntegrators(*integrators_)` in the constructor and replaces `SetEffectiveShearModulus(i, mu_eff)` by `SetRelaxationWeights(i, betas)` / `ClearRelaxationWeights()`; `SupportsEffectiveShearModulus()` becomes `SupportsRelaxationWeights()`. For the isotropic rheology `SetRelaxationWeights` computes μ_eff = μ_∞ + Σ β_k μ_k as a `SumCoefficient` chain — or, as now, nodally in the viscoelastic operator; the current nodal μ_eff GridFunction is exact at nodes and the chain is exact everywhere, so this is a (tiny) improvement rather than a regression. `LinearQuasiStaticSelfGravitatingProblem` needs no change beyond the renamed calls.
- `ViscoelasticOperator`: `Field` gets `nc = n_s` and the symmetric `B` when `!TraceFreeInternalVariables()`, a nodal cache of C_k (n_s×n_s per node per branch; 21 doubles per node per branch in 3-D — for TI one could store the 5 constants and the axis instead, later), the strain map D → full ε through the symmetric B and the symmetric-basis Gram matrix G (Mandel scaling makes G diagonal: 1 on the diagonal components, 2 on the off-diagonal ones, since `SymmetricMatrixIndex` basis tensors E_c have ‖E_c‖² = 1 or 2), and `AddCoupledForce` applies C_k nodally before Bᵀ. Rates, exponential updates and β_k are unchanged. `SetEffectiveModulus` becomes "compute nodal β_k GridFunctions and call `problem.SetRelaxationWeights`".
- Tests: an anisotropic problem with C = isotropic must reproduce the isotropic results to round-off for every scheme (the strongest test, and it exercises the full-symmetric path); TI relaxation of a homogeneous bar with the closed-form 1-D solution; symmetry of the assembled operator; parallel as usual.

Effort: 2–3 d. It touches the viscoelastic tests broadly, so best done after Part I has landed and not interleaved with it.

### 7.4 Relation to the fully referential formulation

Nothing here conflicts with the later Al-Attar & Crawford / Maitra & Al-Attar referential formulation seeded by `TransformedDiffusionIntegrator`: an anisotropic C transforms like any fourth-order tensor under the relabelling, and the `ElasticTensorIntegrator` is the natural place to pull it back.

---

## 8. Decisions to confirm

1. **Single disconnected SubMesh** for all solid regions (§2.1 option b) rather than one field per region. My recommendation: (b); verified to work in serial and parallel despite MFEM's documentation saying otherwise (§2.1). Response: Okay, good you checked. This is better. You might also want to see about things like Boomer working on such meshes, and maybe do that early (say for a toy Poisson problem). If it can't then that is a big issue and I would then go back to the regions done separately. 

2. **Uniform interface convention** (F2′)/(F3′) with the solid's outward normal and the fluid-side density, i.e. no FS/SF classification in the code. Please check the rewriting in §1.5. Response: Yes. this seems correct. I uploaded a copy of Al-Attar & Tromp 2014 (see Documents/MyPapers/GJI) which I think has the terms correct. 
3. **ρ′_F as user input** (dρ/dΦ₀ = g⁻¹∂_rρ), with the ∇ρ_h·∇Φ₀/|∇Φ₀|² helper only as a fallback. Response: If the more general case is doable, then I'd have the option avaiable through overloads. 
4. **Enclosed-region rotations** projected out on request (`AddRegionRotations`), not automatically; translations never. The examples currently project neither. Response: Yes, I think this needs looking into in any case in terms of stability of the solutions etc. 
5. **2-D + fluid**: keep the constant-mode regularisation (§1.7, §3.3). Response: Yes. The 2D problem isn't physical or important, but useful for testing on smaller scale problems. Not worth the time if its proving an issue. 
6. **Potential block possibly indefinite** (§3.2): MINRES inner solves when fluids are present, SPD preconditioner without M_F, and the min-eigenvalue diagnostic. If David is confident the block is always SPD for the models of interest, CG can stay and the diagnostic becomes an assertion. Response: Well, let's see. It's an easy change if needed to switch between solvers. but I've dealt with these equations in separated form (via spherical harmonics) and this has never proved an issue (we use say Cholesky which would see this away from deg 1).
7. **Tidal load** as `SetTidalPotential(ψ)` built from the same operators (§1.6). This gives the tidal Love numbers of the notes for free; the "displacement-load / gravitational-load" split of the generalised Love numbers is already available through `ExternalLoad`/`ExternalPotentialLoad`. Response: Yes, this is highly desirable as an option. 
8. **Fluids passed to the constructor**, not added afterwards (§4). Response: Yes, agreed. A good idea. 
9.  **Anisotropy design** of §7.3 (rheology owns the integrators and the relaxation-weight swap; `SetEffectiveShearModulus` replaced by `SetRelaxationWeights`), to be done after Part I. Response: Yes, sure. As a related question. At present we have the ability in the viscoelastic solvers to reset the stiffness using different moduli are required by some of the integrators. Is there a case for precomputing and storing both versions ever?

---

## 9. Implementation status (2 September 2026, after David's review)

Phases 1–3 of §6 are implemented, serial and parallel, with tests; Phase 4 (Love numbers) and Phase 5 are open and need David's radial numbers.

### 9.1 What was built

- **Integrators and coefficients** (§2.3): `BoundaryNormalNormalIntegrator`, `BoundaryNormalScalarIntegrator` (`bilininteg.hpp`), `BoundaryNormalDotCoefficient`, `BarotropicDensityGradientCoefficient` (new `coefficient.hpp`). Tests `TestBoundaryNormalIntegrators` (exact values on Cartesian meshes, sphere values to the geometric error, orientation of m on inherited and cut SubMesh interfaces in 2-D and 3-D) and the MPI program `TestBoundaryNormalIntegratorsPar` (1/2/4 ranks: values against serial; BoomerAMG on the disconnected ParSubMesh, see 9.3).
- **`FluidRegion` in `LinearQuasiStaticSelfGravitatingProblem`** (§4): constructor argument; Φ₀ from solid (injected) + fluid (parent) densities with the plain Laplace–DtN operator; then `M_F` assembled and added to the potential block; interface terms in the stiffness (`A_Σ`) and in the coupling `C`; `SetTidalPotential` (loads `−CΨ`, `−M_FΨ`); `AddRegionRotations`; `PotentialBlockMinEigenvalue` (Lanczos + Sturm bisection, no LAPACK needed); `ModeResidual(u)` for arbitrary modes; accessors `Coupling()`, `PotentialOperator()`, `PotentialLoad()`. `density_gradient == nullptr` selects the fallback ρ′_F from an element-wise L2 projection of the density (David's Q3: both routes available; the fallback is the default, the analytic one an override).
- **Tests** `TestSelfGravitatingFluid` (gtest) and `TestSelfGravitatingFluidPar` (MPI) on the three-layer meshes (`data/elastogravity_three_layer_{2d,3d}.msh`, the 3-D one new: 13.8k order-2 tets, `concentric_spheres -r 0.1931-0.5467-1.0-1.2 -s 0.14-0.3 -o 2`) and the two-layer disc; test model in `SelfGravitatingTestCommon.hpp`.
- **Examples**: `elastogravity_two_layer(_p)` and `elastogravity_three_layer(_p)` replaced by one driver `elastogravity_layered(_p)` (the layering is read off the mesh; models in `layered_model.hpp`, the earlier profiles verbatim; options `-s`, `-diag`, `-tidal`, `-solid-core`, `-no-fluid-mass`). `common.hpp` reduced to `Nondimensionalisation`, `Constants` and the `TwoBlockRigidBodySolver`s still used by `elastogravity(_p)`. The generic `self_gravitating_elasticity` driver was left as the pure-solid driver rather than given `-fluid` options.

### 9.2 Deviations from the plan, and why

1. **Block projector.** The plan kept the coupled near-null vectors (u_r, −A_φφ⁻¹Cᵀu_r) in the MINRES projector with a gauge fix afterwards. With fluids this produced a 1% disagreement between the two solvers in 3-D, all of it an inner-core translation: the gauge fix injects a residual of size (rigid-mode residual) × (rigid content of the iterate), which the soft Slichter mode amplifies. The projector now removes (u_r, 0) (and (0, 1) in 2-D), i.e. MINRES solves exactly the restricted system whose Schur complement the Schur solver uses; the two agree to 1e-10 in every test, no gauge fix, no inner solves in the projector setup. The pure-solid class inherits this (its solver-agreement tolerances were tightened to 1e-7).
2. **2-D inner solves** run CG on P A_φφ P (own `ProjectedOperator`/`ProjectedSolver`) instead of MFEM's `OrthoSolver` (CG on A_φφ with a projected right-hand side): with M_F ≠ 0 the two are different regularisations and the solvers disagreed by 0.4%. Consequence noticed on the way: **2-D + fluid is inconsistent by more than a gauge** — the interface coupling −∫ρ_F φ (m·v) is not invariant under a constant shift of φ, so the constant-mode regularisation changes the 2-D answer at the percent level (a 2.8% potential-row residual along the constant). Kept as agreed (Q5) for cheap testing only; 3-D is unaffected.
3. **Projected preconditioners and an inner-tolerance floor.** CG on the projected 2-D potential block diverged in parallel once the residual reached round-off: the shifted-Laplacian preconditioner amplifies the round-off constant component of the residual by 1/ε per iteration (Gauss–Seidel in serial does not, which is why it never showed). Every projected solve now uses the projected preconditioner P M P as well (2-D potential solves, the Schur CG, the block MINRES; `ProjectedSolver` around the preconditioner), and the inner relative tolerance is floored at 1e-13 because CG's criterion is on the squared residual. With this the parallel 2-D order-2 solves take 25 BoomerAMG-CG iterations and the parallel tests run in seconds.
4. **CG kept for the inner solves** (Q6), with the eigenvalue diagnostic. The Ritz values in the tests: 3-D three-layer test model 6.3e-3 (positive); the steep-gradient variant −1.3e-2 (indefinite, as intended for the test). For the PREM-like `elastogravity_layered` 3-D model the smallest Ritz value is 2.3e-4 against a largest of 0.18.

### 9.3 Findings

- **Parallel results match serial** to all printed digits for `elastogravity_layered_p` at 2 and 4 ranks (2-D order 2 and 3-D order 1); the MPI test programs pass at 1/2/4 ranks. A driver-level lesson: `ParFiniteElementSpace::GlobalTrueVSize()` is collective and must not sit inside a root-only block (it hung the first parallel driver).
- **BoomerAMG on the disconnected ParSubMesh** (Q1): fine. CG+AMG iteration counts, disconnected {inner core, mantle} vs connected {mantle}, at 1/2/4 ranks: Laplacian 14–18 vs 14–18 (identical); elasticity with the elasticity options 146–426 vs 125–261 with a 1e-3 mass shift (twice the near-rigid modes) and 146–195 vs 125–162 with a unit shift. No fallback to per-region fields needed.
- **Rigid-mode residuals with fluids** decrease with order as before (2-D: 4.6e-4 → 1.8e-6; 3-D: 3.7e-4 → 4.4e-6 for the global modes; inner-core rotations 1.3e-3 → 4.8e-6 and 2.3e-3 → 1.7e-5). A sign error in (F2′) or in either half of (F3′) would have left them O(1).
- **Slichter mode** (Q4): the inner-core translation residual converges to a finite value (2-D 5.6e-4 → 1.8e-4, 3-D 2.1e-4 → 8.3e-5) while the rotations' vanishes; every residual scales with G, so this convergence is the test that it is a soft mode and not a null one. It is soft: in the non-dimensional test model its Rayleigh quotient is ~2e-4 of ‖A‖, so discretisation error shows up first as a spurious inner-core translation; refinement, not projection, is the remedy.
- **Fluid mass term is not small in effect.** For the PREM-like three-layer model in 3-D (order 1), dropping ρ′_F changes ‖u‖ from 5.2e-5 to 1.6e-5 and ‖φ‖ from 1.27e-3 to 5.0e-4 — a factor of about three. This is consistent with §3.2: k R_CMB ≈ 1.06 against π/2 for that core, so the potential block is at roughly half its positivity margin and the response is amplified. Worth checking against the radial codes early in Phase 4.
- **Against the earlier examples** (§5.2; new code vs new code): Φ₀ agrees to all printed digits. In 3-D at order 1 with the old displacement projected to the same rigid gauge, ‖u‖ 4.48e-5 (old) vs 5.18e-5 (new) with the fluid mass term, 1.74e-5 vs 1.61e-5 without; ‖φ‖ 1.07e-3 vs 1.27e-3 and 5.2e-4 vs 5.0e-4. Two differences in the earlier code account for discrepancies of this size: its rigid projection uses the *interpolated* −u_r·∇Φ₀ rather than the discrete null vector, and its ρ′_F comes from the radial derivative of an H1 projection of the discontinuous density, which is badly polluted in the element layers at the ICB and CMB. In 2-D the old and new answers differ by a factor of two, which is the constant-mode inconsistency of item 2 above. The comparison is therefore inconclusive and the Love-number benchmark is the real test.

### 9.4 Open

- Phase 4 (§5.3): needs David's pyslfp/gia3D numbers for a PREM-like fluid-core model; a `love_numbers` driver with surface spherical-harmonic extraction is to be written.
- Phase 5 (§6): the driver `self_gravitating_relaxation` exists (2 Sep 2026); the comparison with the radial codes' time-domain Love numbers is pending with Phase 4.
- Q9 (caching both stiffness versions): see the discussion in the session summary; not done.

---

## 10. Part II implementation status (2 September 2026)

§7.3 is implemented as designed, with one structural refinement.

- **`Rheology`** (abstract, `rheology.hpp`): `SpaceDim`, `NumBranches`, `RelaxationTime(k)`, `TraceFreeInternalVariables()`, `BranchShearModulus(k)` (isotropic only), `BranchModulus(k, T, ip, Ck)` (Mandel @f$C_k@f$ pointwise; @f$2\mu_k P_{dev}@f$ for the isotropic body), and `MakeStiffness()`.
- **`ElasticStiffness`** (abstract): the refinement. Rather than the rheology itself owning "the coefficient to assemble with" (which would make two problems sharing one rheology interfere), the rheology is a factory of per-problem stiffness objects holding the redirectable coefficient: `AddIntegrators(form)`, `SetRelaxationWeights(beta)`, `ClearRelaxationWeights()`, `IsRelaxed()`. The isotropic one adds the two split `ElasticityIntegrator`s and swaps a `RedirectableCoefficient` between @f$\mu_U@f$ and the chain @f$\mu_\infty + \sum_k \beta_k\mu_k@f$; the anisotropic one adds one `ElasticTensorIntegrator` on a `RedirectableMatrixCoefficient` swapped between @f$C_U@f$ and @f$C_\infty + \sum_k \beta_k C_k@f$ (MFEM's `MatrixSumCoefficient` / `ScalarMatrixProductCoefficient`).
- **`IsotropicMaxwellRheology : Rheology`** keeps its whole API. **`AnisotropicMaxwellRheology : Rheology`** takes @f$C_\infty@f$ and `AnisotropicBranch{C_k, tau_k}` (Mandel `MatrixCoefficient`s); a `DeviatoricMaxwell(C, tau)` (relax @f$P_{dev} C P_{dev}@f$, keep the complement) factory; `UnrelaxedTensor()`. Since 3 Sep 2026 purely elastic solids are `IsotropicElasticRheology` / `AnisotropicElasticRheology` rather than branchless Maxwell bodies, and the Maxwell classes expose their elastic limits as `UnrelaxedElastic()` / `LongTermElastic()` (see `viscoelastic_design.md` §6).
- **Problem layer**: `LinearQuasiStaticProblem::Rheology(i)` returns `const Rheology&`; `SupportsEffectiveShearModulus / SetEffectiveShearModulus / ClearEffectiveShearModulus` replaced by `SupportsRelaxationWeights / SetRelaxationWeights(i, beta) / ClearRelaxationWeights`; `LinearQuasiStaticProblemBase` takes any `Rheology` and no longer knows about @f$\kappa, \mu@f$ (`CurrentShearModulus()` gone; `Stiffness()` exposes the relaxation state). `LinearQuasiStaticSelfGravitatingProblem` takes a `Rheology` too, so a self-gravitating anisotropic body needs no further change.
- **`ViscoelasticOperator`**: per field `nc = n_s` and the full-strain `DomainSymmetricMatrixStrainIntegrator` / `StrainInterpolator` when the rheology is not trace-free (Gram matrix `diag(1, 2)` in the symmetric basis); nodal branch data are either @f$2\mu_k@f$ or the sampled @f$n_s\times n_s@f$ matrix @f$W = \hat C\,\mathrm{diag}(w_t/a_t)/a_s@f$ acting on unscaled components (`ApplyBranchModulus`); the effective modulus is now "nodal @f$\beta_k@f$ `GridFunction`s + `SetRelaxationWeights`" for both rheologies (the isotropic @f$\mu_{eff}@f$ is evaluated as the coefficient chain at quadrature points rather than interpolated nodally: equal at the nodes, exact in between). Rates, exponential updates and the stepping formulas are unchanged.
- **Tests** (serial + MPI 1/2/4): `TestRheology` (anisotropic tensors; the two stiffness objects assemble the same matrix for an isotropic tensor, unrelaxed, weighted and cleared); `TestQuasiStaticProblem(+Par)` (relaxation weights; anisotropic = isotropic); `TestViscoelastic(+Par)`: the anisotropic path with an isotropic tensor reproduces the isotropic operator to 1e-10 for every scheme (displacement, and the deviatoric part of the full internal variable equals the trace-free one); a transversely isotropic Maxwell bar under uniaxial stress (tilted axis, homogeneous state) matches the 6×6 (3×3 in 2-D) Mandel ODE reference to 1e-6 (RK4) / 1e-3 (trapezoid at dt = 0.02); self-gravitating tests migrated to relaxation weights on a Maxwell rheology.
- **Example**: `viscoelasticity -ti f` (transversely isotropic Maxwell body with anisotropy factor f; `-ti 1` reproduces the isotropic run through the anisotropic path).

Not done: a compact TI storage of the nodal tensors (36 doubles per node per branch in 3-D as it stands); the relaxable-part choice "L and N only" needs no code (a TI coefficient with A = C = F = 0 as the branch tensor).

On David's Q9 (cache both stiffness versions, for adjoint runs with many observation-time jumps): the design keeps this local to `LinearQuasiStaticProblemBase` (a small `(relaxed?, dt)`-keyed cache of `A_` plus preconditioner); not implemented yet.
