# Mapped (relabelled) forms

Design and method notes for the mapping-aware integrator layer: the
pull-back of the elasto-gravity weak forms through a diffeomorphism of the
reference (spherical) domain, the single assembly recipe that implements
every pulled-back term, and the `Diffeomorphism` interface. The layer
serves the relabelled 3-D benchmarks (`benchmarks/`), aspherical models
with interface and surface topography, and — because the mapping appears
explicitly in the forms — analytic shape derivatives later.

The construction follows the particle-relabelling picture of Al-Attar &
Crawford (2016; `doc/BenchmarkPapers/ggw032.pdf`, with a dictionary of
conventions at the end of Section 3); every formula below is
self-contained (chain rule, change of variables, Nanson's relation) so it
can be checked line by line.

## 1. Setup and conventions

- `M` is the reference body inside the reference computational ball `B`,
  all boundaries and internal interfaces genuine spheres. `ξ : B → B̃` is a
  diffeomorphism onto the physical domain; indices `A, B` label reference
  coordinates `x`, indices `i, j` physical coordinates `ξ`.
- Deformation gradient, Jacobian, right Cauchy–Green tensor of the mapping:

  ```
  F_iA = ∂ξ_i/∂x_A,     J = det F > 0,     C = Fᵀ F.
  ```

- **ξ is the identity on and outside the DtN sphere** (in practice: from
  somewhere inside the buffer outward). The outer boundary is then a
  genuine sphere and `SurfaceHarmonics`, the DtN map and the boundary
  harmonic analysis are used unchanged. Interface and surface topography
  live entirely in ξ; the reference fluid–solid interfaces remain spheres,
  so the submesh coupling is also untouched.
- Fields pull back **componentwise** (no rotation of components; vectors
  keep their ambient Cartesian components):

  ```
  u_i(x) = ũ_i(ξ(x)),     φ(x) = φ̃(ξ(x)).
  ```

- The three transformation rules everything follows from:

  ```
  gradients:   ∂ũ_i/∂ξ_j = (∂u_i/∂x_A) F⁻¹_Aj        (∇̃ = F⁻ᵀ ∇)
  volumes:     dξ = J dx
  surfaces:    m̃ dS̃ = ν dS,  ν = J F⁻ᵀ n             (Nanson)
               i.e.  dS̃ = |ν| dS,   m̃ = ν/|ν|.
  ```

- **Coefficient convention.** Integrators take the *physical* material
  fields expressed in reference coordinates — `ρ` below always means
  `ρ̃ ∘ ξ` — and the geometric factors `J`, `F` appear explicitly in the
  form. Nothing is absorbed into "relabelled" densities. Two consequences:
  with ξ = id the forms reduce term-by-term to the existing ones, and in
  the relabelled benchmark (physical model *defined* as the push-forward of
  a spherical model) the referential coefficients are exactly the radial
  coefficients the spherical code already uses.
- Coefficients that are physically gradients (`∇̃Φ̃₀`) are supplied as the
  referential gradient and mapped by `F⁻ᵀ`; a small pull-back coefficient
  in the layer does this.

## 2. The pulled-back forms

Each term of the bilinear form `A` (see `self_gravitation.md`) pulls back
by applying the three rules. Writing `∇` and `div` for reference-domain
operators:

```
Poisson       ∫ ∇̃φ̃·∇̃φ̃′ dξ            = ∫ J C⁻¹ ∇φ·∇φ′ dx
elastic       ∫ c̃_ijkl ∂_jũ_i ∂_lũ′_k dξ = ∫ c′_iAkB ∂_Au_i ∂_Bu′_k dx,
                                          c′_iAkB = J c_ijkl F⁻¹_Aj F⁻¹_Bl
mass / L2     ∫ ρ̃ φ̃ φ̃′ dξ              = ∫ J ρ φ φ′ dx
advective     ∫ ρ̃ ∇̃φ̃·ũ′ dξ            = ∫ J ρ (F⁻ᵀ∇φ)·u′ dx
divergence    d̃iv ũ                     = tr(∇u F⁻¹) = ∂_Au_j F⁻¹_Aj
surface       ∫_Σ̃ f̃ (m̃·ũ)(m̃·ũ′) dS̃   = ∫_Σ f (m̃·u)(m̃·u′) |ν| dS,
                                          m̃ = ν/|ν|, ν = J F⁻ᵀ n
DtN           unchanged (ξ = id on the DtN sphere)
```

The pulled-back elasticity tensor `c′` retains only the **major** symmetry
`c′_iAkB = c′_kBiA` (hyperelasticity survives the pull-back); the minor
symmetries are lost because `iA` mixes a physical with a referential index.
In 3-D that is 45 independent components against the 21 of `c` — the reason
`c′` is never stored as a field, and in fact (Section 3) never formed at
all.

## 3. One assembly recipe

Every domain term above is its standard counterpart with two
substitutions at each quadrature point:

```
dshape  →  dshape · F⁻¹          (shape derivatives w.r.t. ξ, by chain rule)
w       →  J · w                 (volume element)
```

after which the *standard* assembly runs. For the elastic term this means:
compute `dshape · F⁻¹`, then build the usual Mandel strain–displacement
matrix `B` from it and contract with the **referential** tensor `ĉ`
(21 components, `ElasticTensorCoefficient` unchanged). Symmetrising in the
ξ-derivatives is exact because `c̃` has its minor symmetries in physical
indices — so the Mandel machinery is reused as-is and the 45-component
`c′` never appears. Per quadrature point the extra cost is one `d × d`
inversion and one small matrix product; the extra storage over the
spherical problem is the mapping itself.

Boundary terms use Nanson's relation instead: the normal-and-area factor
`m dS` is replaced by `ν dS = J F⁻ᵀ n dS`, with unit normal `ν/|ν|` and
area factor `|ν|` wherever the integrand needs them separately.

Precedent: `TransformedDiffusionIntegrator` already implements exactly
this recipe for the Poisson term (`a = J C⁻¹ = J F⁻¹F⁻ᵀ`).

### Material symmetry survives the pull-back

If the physical body is isotropic, the pulled-back tensor is not
isotropic in the naive sense, but it takes the closed form

```
c′_iAkB = J [ λ G_iA G_kB + μ ( δ_ik (C⁻¹)_AB + G_kA G_iB ) ],   G = F⁻ᵀ,
```

with λ, μ the referential Lamé fields: it is parametrised entirely by the
two scalars and the mapping. Likewise a physically TI body pulls back to
a form parametrised by its five Love moduli and the referential axis
field ñ∘ξ (components untouched — relabelling does not rotate ambient
components). The recipe of this section computes exactly these
contractions without forming them: material symmetry enters through the
unchanged referential `ElasticTensorCoefficient`, geometry through F, and
storage never exceeds the referential model plus the mapping. Al-Attar &
Crawford (2016) state both versions of this fact: eq. (143) is the
linearised "apparent anisotropy of a specific form" of an isotropic body
relative to a non-natural reference configuration (with the corollary
that relabelling-induced anisotropy cannot produce shear-wave
splitting), and eq. (75) the finite-strain analogue for a modified
Saint Venant–Kirchhoff material.

### Dictionary to Al-Attar & Crawford (2016)

The paper (`doc/BenchmarkPapers/ggw032.pdf`; GJI 205, 575–593) writes the
relabelling ξ : M̃ → M from the *new* reference body to the old, and its
eq. (71) absorbs the Jacobian into the relabelled material parameters:
ρ̃ = J_ξ ρ∘ξ, W̃(x̃, F̃) = J_ξ W[ξ, F̃ F_ξ⁻¹]. The forms here keep J
explicit and take coefficients as the physical fields composed with the
mapping (Section 1), so the paper's relabelled density is `J ρ` in this
note's symbols. The linearised first Piola–Kirchhoff elastic tensor of
its Section 4 carries the same major-only symmetry as c′ above.

## 4. The `Diffeomorphism` interface (`mappings.hpp`)

A mapping is asked for exactly two things at a quadrature point — ξ and F
(J follows from F). The base class derives from `mfem::VectorCoefficient`
with `Eval` returning ξ, so a mapping is directly usable everywhere a
coefficient of position already goes (`TransformedFunctionCoefficient`,
`Mesh::Transform`, projection onto a nodal space); it adds `EvalGradient`
returning F and a `Jacobian` helper for det F.

Concrete classes:

- `CallableDiffeomorphism` — ξ and F from callables of position: analytic
  mappings with exact derivatives.
- `RadialDiffeomorphism` — ξ = f(x)·x with exact `F = f I + x (∇f)ᵀ`,
  from coefficients for f and ∇f, or from a radial profile f(r), f′(r)
  (needing f′(0) = 0 for smoothness at the origin). Both constructors
  require the derivative: the analytic classes are exact by design, and a
  caller with f alone goes through `Interpolate`, which states honestly
  that the gradient is then a discrete one. (Replaces the retired
  `RadialDiffeomorphismCoefficient`, which supplied only ξ.)
- `GridFunctionDiffeomorphism` — ξ = x + h for a displacement grid
  function, `F = I + ∇h` through the element transformation. The discrete
  representative: interpolated mappings, mappings supplied in discretised
  form (planetmodel can export these — the same entry point), and later
  the shape-inversion variable.

Free functions (serial, with parallel overloads under `MFEM_USE_MPI`):

- `Interpolate(xi, mesh)` — the nodal interpolant ξ_h of ξ on the mesh's
  geometric space (its nodal collection, order and ordering), returned as
  an owning `GridFunctionDiffeomorphism`.
- `MappedMesh(mesh, xi)` — a copy of the mesh with nodes moved to ξ_h:
  the element-by-element image mesh. Set the reference mesh's curvature
  before calling.
- `MaxIdentityDeviation(xi, mesh, bdr_marker)` — the max of |ξ(x) − x|
  over quadrature points of the marked boundary; drivers wrap it in an
  `MFEM_VERIFY` to assert the identity-outside-the-body convention
  (Section 1), which mappings satisfy by construction and this detects
  the taper that does not quite reach zero.

Helper coefficients: `JacobianCoefficient` (J),
`PullbackDiffusionCoefficient` (J C⁻¹, feeding the Poisson pieces),
`PullbackGradientCoefficient` (F⁻ᵀ v for referential gradients such as
∇Φ₀). The pull-backs `TransformedFunctionCoefficient` and
`TransformedVectorFunctionCoefficient` also live here (mapping-layer
coefficients, per the file-layout rule).

**Boundary (Nanson) machinery.** `Diffeomorphism::MapNormal` returns
ν = cof(F)·n for a reference unit normal n (adjugate-based, no inversion
of F); on a boundary transformation of a `GridFunctionDiffeomorphism`
the gradient comes from the adjacent volume element
(`GridFunction::GetVectorGradient` does this), which is what makes the
discrete identity hold on boundary terms — the mapped mesh's boundary
geometry is the trace of its volume geometry. Built on it:
`NansonAreaCoefficient` (|ν| = dS̃/dS, composing physical surface loads
onto the stock boundary linear-form integrators, as `JacobianCoefficient`
does for volume densities) and `MappedBoundaryNormalDotCoefficient`
(m̃·V = (ν·V)/|ν|, the mapped counterpart of
`BoundaryNormalDotCoefficient` for the m̃·∇̃Φ̃₀ factor of the interface
terms). The boundary integrators own exactly the normals of their own
formulas plus the measure: `BoundaryNormalScalarIntegrator` maps
(m·v) dS → (ν·v) dS (Nanson exactly — the |ν| factors cancel), and
`BoundaryNormalNormalIntegrator` maps (m·u)(m·u′) dS →
(ν·u)(ν·u′)/|ν| dS (two unit normals, one measure); each has a mapped
constructor and referential coefficients, and the problem layer remains
the only composer of coefficient and integrator.

**Consumers.** The existing generalised integrators
(`DomainVectorScalarIntegrator`, `DomainVectorGradScalarIntegrator`,
`DomainDivVectorScalarIntegrator`, ..., `ElasticTensorIntegrator`, and the
boundary integrators) gain an optional trailing `Diffeomorphism*`
argument, default `nullptr` meaning the identity with zero overhead —
rather than a parallel `Transformed*` family: the recipe touches two
lines of each `AssembleElementMatrix`, the coefficient semantics are
unchanged (referential fields, Section 1), and every present call site
compiles as-is. `TransformedDiffusionIntegrator` stays as the Poisson
entry point (MFEM's `DiffusionIntegrator` cannot take the argument) and
gains a `Diffeomorphism` constructor beside its current ones.

## 5. The discrete change-of-variables identity (benchmark variant 2a)

The layer supports two evaluation modes, used for different purposes:

- **Exact mode** (`CallableDiffeomorphism`, `RadialDiffeomorphism`): F is
  analytic at each quadrature point — no projection error in the operator.
  This is the production mode and the one for convergence benchmarks
  (variant 2b): the pulled-back solution converges to the mapped exact
  solution under refinement.

- **Interpolated mode** (`Interpolate` + `GridFunctionDiffeomorphism`):
  F is the gradient of the nodal interpolant ξ_h of ξ on the mesh's
  geometric space. Let the mapped mesh be the reference mesh with nodes
  moved to ξ_h (`MappedMesh`). Then for any element and quadrature point,
  the mapped element map is ξ_h composed with the reference element map,
  so by the chain rule its Jacobian is `F_h · J_ref` and its weight
  `J_h · |J_ref|` — precisely the recipe of Section 3 with `F = F_h`.
  Assembling the standard integrators on the mapped mesh and the mapped
  integrators on the reference mesh with ξ_h therefore produces the
  **same element matrices to machine precision**, provided the same
  integration rule is passed to both (the default rules must not be
  relied on, since they need not coincide). This is the
  change-of-variables identity as a unit test: no physics, no reference
  solution, every mapped integrator checked against its standard
  counterpart on an arbitrarily deformed mesh.

The distinction matters: with *exact* F the two discrete systems are not
identical — they differ by the geometric interpolation error of ξ_h — so
exact-mode agreement with the mapped mesh is a convergence statement, not
an identity. Machine precision belongs to the interpolated mode; superior
accuracy belongs to the exact mode.

These identity tests sit alongside the integrators' existing unit tests
(known-solution actions, symmetry, patch and covariance tests) and are
part of the verification chain in the same sense: each mapped operator is
certified at the element-matrix level, so the benchmark hierarchy runs
continuously from element matrices to Love numbers — get the bits right
and the whole follows.

## 6. Order of implementation

1. `mappings.hpp/.cpp` with tests (`TestMappings`): the classes and free
   functions of Section 4; F exact for the analytic mappings, the
   Jacobian factorisation of `MappedMesh` through `Interpolate` at
   geometric orders 1–3 in 2-D and 3-D, `MaxIdentityDeviation`.
2. Poisson: mapped stiffness (`J C⁻¹` into the existing machinery) plus
   `J ρ` source; the 2a identity test against `DiffusionIntegrator` on
   the mapped mesh; a mapped Poisson solve converging to a mapped exact
   solution (2b).
3. `ElasticTensorIntegrator` with the optional mapping; identity test.
4. The remaining domain integrators of `A` (`TestMappedDomainIntegrators`),
   then the Nanson boundary terms (`TestMappedBoundaryIntegrators`: the
   F3 and composed-F2 identities and both load compositions); identity
   tests for each.
5. The relabelled elasto-gravity benchmark, the `benchmarks/relabelling`
   family: variant 2a (machine precision), then 2b against pyslfp
   through the mapping. The `meshes/aspherical_body.py` family provides
   independently-generated aspherical meshes where 2b wants them.

   *Done (30 Sep 2026), with one finding.* Both variants live in the
   relabelling family (`benchmarks/relabelling/README.md`):
   `love_benchmark -map` is 2b (interior relabelling of
   `relabelling.hpp`, pointwise identity with identity gradient on every
   interface, exact-F; referential and broken-ζ methods), and
   `relabelled_identity` is 2a at the level of the full coupled solve.
   The solver-level identity holds to ~1e-6 — bounded only by the DtN
   centring itself on the (shifted) mesh centroid, superlinear in the
   amplitude — wherever every assembled term is covariant, which
   certifies the whole mapped chain at once: stiffness, equilibrium
   stress, gravity couplings, Poisson block, loads, extensions and
   projectors. The finding, now RESOLVED for the welded family: the
   fluid *gauge penalty* was the one non-covariant assembly piece (a
   stock integrator on each side's own fluid geometry), and because the
   gauge subspace has stiffness `O(ε)`, an `O(εA)` penalty difference
   produced `O(A)` differences in the gauge representative (u 4e-3
   welded-gauged at the time of the finding). The penalty is now
   assembled covariantly — the pulled-back deviatoric tensor through
   `ElasticTensorIntegrator`'s mapped form — and the welded gauged
   identity is STRICT (u 1.6e-7, ζ 4e-8 at A = 0.02, h = 0.3). Two
   lessons of the fix: the pull-back penalty must use
   `ElasticTensorIntegrator(C, map)` (the certified 2a form), NOT
   `MaterialStiffnessIntegrator`, whose mapped form belongs to the
   referential physics and expects a RELABELLED tensor; and when both
   sides of an identity assemble the same term, both should use the
   same integrator class (the identity map included), so the quadrature
   defaults coincide. Through the slipping interface the identity
   remains informational (u 1.2e-2: the interface constraint forms are
   not yet certified covariant — the remaining follow-up). Two prerequisites of the
   identity that the driver documents: the two sides must share one
   vacuum-extension matrix (gauge *data*; the builder samples the body
   side of the surface, which differs at `O(A)` between the meshes),
   and a nodal-interpolant mapping must be transplanted onto every
   SubMesh the classes assemble on (`MultiMeshDiffeomorphism` in the
   benchmark; a cross-mesh `GridFunction` evaluation is garbage).
