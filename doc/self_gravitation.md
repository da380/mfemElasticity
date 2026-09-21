# The self-gravitating quasi-static problem

Method notes for `LinearQuasiStaticSelfGravitatingProblem`
(`self_gravitating.hpp`): the equations, how each term is assembled, the
solvers and the null space, and what the verification runs have shown. The
weak form follows Al-Attar & Tromp (2014, eq. 2.52) and Yu, Al-Attar, Syvret
& Lloyd (2025, eq. 3 and Appendix A).

## 1. Continuous problem

### Geometry and notation

- `M = M_S ∪ M_F` is the body, with `M_S` the solid regions (one or more)
  and `M_F` the fluid regions. `B ⊃ M` is the computational ball with a
  spherical outer boundary carrying the DtN condition.
- Φ₀ is the background potential. Hydrostatic equilibrium makes every
  fluid–solid interface a level surface of Φ₀ and makes the fluid barotropic,
  ρ = ρ(Φ₀) in `M_F`.
- σ is a surface mass load on ∂M (positive when mass is added); ψ is an
  applied (tidal) potential.

### Bilinear form

```
A(u,φ | u′,φ′) =
    ∫_{M_S} κ (div u)(div u′) + ∫_{M_S} 2μ d : d′
  + ½ ∫_{M_S} ρ [ ∇(u·∇Φ₀)·u′ + ∇(u′·∇Φ₀)·u ]
  − ½ ∫_{M_S} ρ [ (u·∇Φ₀) div u′ + (u′·∇Φ₀) div u ]
  + ∫_{M_S} ρ ( ∇φ·u′ + ∇φ′·u )
  + (1/4πG) ∫_{ℝ³} ∇φ·∇φ′
  + ∫_{M_F} ρ′_F φ φ′                                   (F1)
  − ∫_{Σ_F} ρ_F (m·∇Φ₀)(m·u)(m·u′) dS                   (F2)
  − ∫_{Σ_F} ρ_F [ φ (m·u′) + φ′ (m·u) ] dS              (F3)
```

The first six lines are the solid body; (F1)–(F3) appear with fluid regions.
`A` is symmetric. The elastic terms generalise to an anisotropic tensor
without touching anything else: the gravity, coupling and fluid terms do not
involve the elastic moduli.

**Fluid volume term.** Barotropy gives ∇ρ = (dρ/dΦ₀)∇Φ₀ in the fluid, so

```
ρ′_F := dρ/dΦ₀ = ∇ρ·∇Φ₀ / |∇Φ₀|²  ( = g⁻¹ ∂_r ρ for a radial model ).
```

`FluidRegion::density_gradient` supplies it analytically; when null it is
evaluated from an element-wise L2 projection of the density and the discrete
∇Φ₀ (`BarotropicDensityGradientCoefficient`). ρ′_F is negative wherever
density increases downward, so (F1) is a *negative* mass term on the
potential (§3).

**Interface convention.** Σ_F is the union of all fluid–solid interfaces,
**m the outward normal of the solid** on it (pointing into the fluid whether
the fluid lies below, as at the core–mantle boundary, or above, as at the
inner-core boundary), and ρ_F the density on the fluid side. With n̂ the
upward normal and g = |∇Φ₀|, one has m·∇Φ₀ = −g where the fluid is below and
+g where it is above, and (F2)–(F3) reproduce the published forms, which
treat the two kinds of interface separately with opposite signs. Written this
way the code never classifies an interface and needs no sign parameter: every
interface integral uses the solid SubMesh's own outward normal, and the sign
of m·∇Φ₀ does the rest.

Physically, the Lagrangian pressure perturbation on the fluid side is
p = −ρ_F (φ + u·∇Φ₀), with u·∇Φ₀ = (m·u)(m·∇Φ₀) on a level surface; the
traction on the solid is −p m, which gives (F2) and the u′-half of (F3). The
φ′-half of (F3) is the surface mass ρ_F (m·u) displaced across the
interface, and (F1) is the Eulerian density perturbation ρ′_F φ in the fluid.

### Rows of the system

With the coupling form
`c(φ, u′) = ∫_{M_S} ρ ∇φ·u′ − ∫_{Σ_F} ρ_F φ (m·u′) dS`, `a` the elastic
plus gravity displacement form, `a_Σ` = (F2), `m_F` = (F1), and the loads
`ℓ_u(u′) = −∫_{∂M} σ ∇Φ₀·u′`, `ℓ_φ(φ′) = −∫_{∂M} σ φ′`:

```
displacement:  a(u,u′) + a_Σ(u,u′) + c(φ,u′)                        = ℓ_u(u′) − c(ψ,u′)
potential:     (1/4πG)[∫_B ∇φ·∇φ′ + DtN(φ,φ′)] + m_F(φ,φ′) + c(φ′,u) = ℓ_φ(φ′) − m_F(ψ,φ′)
```

In blocks: `[A_uu + A_Σ, C; Cᵀ, A_φφ] [u; φ] = [B_u − CΨ; B_φ − M_F Ψ]` with
`A_φφ = (K + DtN)/(4πG) + M_F` and Ψ the interpolant of ψ on the potential
space. **The tidal load is the same operators applied to Ψ**: no extra
assembly.

## 2. Discretisation

**One displacement field on a possibly disconnected SubMesh.** All solid
regions (inner core and mantle, say) form one
`SubMesh::CreateFromDomain(parent, {solid attributes})` with one H1 vector
space; materials vary by attribute through the rheology. MFEM's
documentation asks for a connected subset, but nothing in the implementation
depends on connectivity (see `mfem_notes.md`), and BoomerAMG converges on
the disconnected space as on a connected one. The fluid carries no
displacement, so the solid regions never couple to each other directly.

| Term | Lives on | Assembly |
|---|---|---|
| (F1) `m_F` | parent, fluid attributes | `MassIntegrator(ρ′_F)` with a domain marker, added to the potential block |
| (F2) `a_Σ` | solid SubMesh boundary, interface marker | `BoundaryNormalNormalIntegrator(q)`, q = −ρ_F (m·∇Φ₀) through `BoundaryNormalDotCoefficient`; part of the stiffness integrators, so it is reassembled with them (harmlessly) |
| (F3) coupling | `SubMeshMixedBilinearForm(fes_φ, fes_u)` | `BoundaryNormalScalarIntegrator(−ρ_F)` as a boundary integrator on the SubMesh, beside the domain term ρ∇φ·v; `Cᵀ` by transposition |
| tidal load | both rows | Ψ interpolated on `fes_φ`; `B_u −= CΨ`, `B_φ −= M_F Ψ` |
| Φ₀ | parent | Poisson–DtN solve with the solid density (injected from the SubMesh) and the fluid densities (on the parent), unless a coefficient is supplied |

The interface integrators take the normal from `CalcOrtho` of the boundary
transformation, which is outward from the SubMesh for MFEM's boundary
elements, on interfaces inherited from the parent and on cut ones alike (the
tests check ∫ x·m dS on both).

**Where coefficients are evaluated.** The solid density is evaluated on the
SubMesh. A fluid density is evaluated on the *parent's* fluid elements (for
Φ₀ and ρ′_F) **and** on the *SubMesh's boundary elements* on Σ_F (for
(F2)–(F3)). A coefficient of position serves both; one keyed by domain
attribute does not, since a boundary transformation carries the boundary
attribute. Hence the separate `FluidRegion::interface_density`.

## 3. Solvers

**Block MINRES (default).** MINRES on the `[u; φ]` system, which is
symmetric and indefinite (a saddle point: the energy is minimised in u and
maximised in φ), with a block-diagonal SPD preconditioner: Gauss–Seidel or
BoomerAMG with elasticity options on `A_uu`, and on the shifted Laplacian
`(K + εM)/4πG`, without `M_F`.

**Schur CG.** CG on `S = A_uu − C A_φφ⁻¹ Cᵀ`, symmetric and, for a
gravitationally stable body, positive on the complement of the rigid modes.
Each application costs an inner potential solve, which makes it roughly an
order of magnitude slower than MINRES (2-D order 2: 1.1 s against 0.07 s).
It is the reference against which MINRES is checked.

The coupling is not weak, ρgR/μ ≈ 2.4 for the Earth, which is why a
segregated (block Gauss–Seidel) iteration between u and φ is not offered: it
needs under-relaxation and its contraction factor depends on that ratio.

**Definiteness of the potential block.** `A_φφ = (K + DtN)/4πG + M_F` with
`M_F ≤ 0`. For a uniform fluid ball of radius R the Laplace–DtN part first
becomes singular against the mass term at kR = π/2, with k² = 4πG|ρ′_F|. For
PREM's outer core k·R_CMB ≈ 1.1, so the block is positive for Earth-like
models, but not by a wide margin, and not for much larger or steeper fluid
bodies. The inner solves use CG; `PotentialBlockMinEigenvalue()` (Lanczos)
reports the margin, and a negative value means the inner solves cannot be
trusted. The proximity to the margin also makes the fluid mass term far from
negligible: for a PREM-like three-layer model dropping ρ′_F changes ‖u‖ and
‖φ‖ by a factor of about three.

## 4. Null space and gauge

**Global rigid modes.** `u_r = a + b×x` on `M_S` paired with
`φ_r = −u_r·∇Φ₀` on all of space is an exact null vector of `A`. Discretely
it is only *near*-null: `RigidModeResiduals()` gives about 1e-3 at order 1
and 1e-6 at order 2, decreasing with refinement. The system is therefore
regularised by projection rather than relying on an exact null space
(`null_space.md`).

**What is projected.** Both solvers solve the system restricted to
displacements orthogonal to the rigid modes in the Euclidean true-dof inner
product. The MINRES projector removes `(u_r, 0)` (and `(0, 1)` in 2-D), not
the coupled pair `(u_r, φ_r)`; the restricted block system then has exactly
the Schur complement the Schur solver uses, so the two agree to solver
tolerance with no gauge correction afterwards. Projecting the coupled pair
and fixing the gauge later injects a residual of size (rigid-mode residual) ×
(rigid content of the iterate), which the soft Slichter mode below amplifies
into a percent-level disagreement.

**Gauge.** By default the displacement has no rigid component in the
true-dof inner product, and the potential is the discrete solution of its
equation for that displacement. `SetMassWeightedGauge()` fixes instead zero
net momentum and angular momentum, ∫ρ u·(a + b×x) = 0; the potential then
follows the re-gauged displacement, since a rigid shift moves the body's
potential with it.

**Enclosed solid regions.** For a spherically symmetric model a rotation of
a solid inner core alone is an exact null vector: no strain, u·∇Φ₀ = 0,
m·u = 0 on the interface, and ∫ρ (b×x)·∇φ′ = 0. For an aspherical inner core
it is physically near-null, with a restoring torque of the order of the
asphericity. `AddRegionRotations()` projects these modes on request. A
*translation* of the enclosed region is **not** null: gravity restores it
(the Slichter mode), and it must never be projected. It is, however, soft
(a Rayleigh quotient of order 1e-4 of ‖A‖ in the test model), so
discretisation error shows first as a spurious inner-core translation;
refinement, not projection, is the remedy. The tests use the distinction:
the rotation residuals vanish with refinement while the translation residual
converges to a finite value.

**Projected preconditioners.** Every projected solve applies the projector
to the preconditioner as well (`P M P`). An unprojected AMG or
shifted-Laplacian preconditioner amplifies the round-off null component of
the residual by 1/ε per iteration, and CG then diverges once the residual
reaches round-off; this shows in parallel, while serial Gauss–Seidel happens
not to trigger it. The inner relative tolerance is floored at 1e-13, because
CG's stopping criterion is on the squared residual.

## 5. Two dimensions

The constant potential is a null vector of the Laplace–DtN block (a
potential cannot be pinned at infinity in 2-D). Potential loads are made
compatible by subtracting a uniform flux through the outer boundary, and
every potential solve runs on `P A_φφ P` with the constant projected out;
the block solver restricts the potential likewise.

With fluid regions this is a regularisation, not a null-space fact: the
constant is not null once `M_F ≠ 0`, and the interface coupling
−∫ρ_F φ (m·v) is not invariant under a constant shift of φ. The 2-D fluid
problem is thus inconsistent at the level of a constant, the choice of
regularisation changes the answer at the percent level, and 2-D fluid runs
are for cheap testing only. Three dimensions are unaffected.

## 6. Verification

- **Symmetry and null space.** The assembled block operator is symmetric to
  round-off, and the rigid-mode and region-rotation residuals decrease with
  order. A sign error in (F2) or in either half of (F3) would leave them
  O(1), so this is the sharpest check of the interface terms.
- **Tidal load.** A uniform field ψ = a·x loads, in the continuum, exactly a
  rigid mode, so the solved displacement must be of the order of the
  rigid-mode residual.
- **Solver equivalence.** Schur CG and block MINRES agree to 1e-7 or better
  with and without fluid regions, serial and parallel.
- **Love numbers** (`examples/love_numbers.cpp`). Load and tidal h, k of a
  homogeneous sphere against the incompressible formulas (Wu & Peltier 1982)
  with κ/μ = 200 agree to 0.1–0.7 % at degrees 2–4 on a fine order-2 mesh,
  the remainder being the finite bulk modulus; on the coarse test mesh the
  error is a few per cent and is the mesh's own. The load's own potential is
  taken from the rigid-body solve on the same mesh (`SolveLoadPotential()`):
  subtracting the exact value instead leaves the direct potential's
  discretisation error in k′, a 30 % effect on a coarse 3-D mesh.
- **Element order on curved meshes.** Order-1 displacements on an order-2
  geometry do not contain the rigid rotations, and results shift at the
  10 % level. Use a displacement order at least that of the geometry.
