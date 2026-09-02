# Transversely isotropic and general anisotropic elasticity — design and plan

*30 August 2026. Companion to `status_and_roadmap.md` §6. MFEM 4.9.1 (`ec39b3509c`).*

MFEM's `ElasticityIntegrator` (`fem/bilininteg.cpp:3208`) assembles ∫ λ div u div v + 2μ ε(u):ε(v) with two scalar coefficients, in the byNODES layout `elmat(dof·c + i, dof·c' + i')`, with default quadrature order `2·OrderGrad(el)`. Nothing in MFEM handles a fourth-order stiffness tensor. The plan below adds one integrator that takes the tensor as a matrix coefficient, and a small family of coefficient classes that *produce* that matrix in a fixed convention — isotropic, transversely isotropic with an arbitrary axis field, general anisotropic from Voigt data, and rotated local frames. The convention question is the only real design decision, so it comes first.

---

## 1. Convention: the reduced tensor is a Mandel matrix in the library's own ordering

A symmetric second-order tensor in `d` dimensions has `n_s = d(d+1)/2` components. The library already fixes an ordering for these through `SymmetricMatrixIndex::ComponentOffset(j,k)` (`bilininteg.hpp`): lower triangle, column-major,

```
d = 3:  s(0,0)=0  s(1,0)=1  s(2,0)=2  s(1,1)=3  s(2,1)=4  s(2,2)=5      (11, 12, 13, 22, 23, 33)
d = 2:  s(0,0)=0  s(1,0)=1  s(1,1)=2                                     (11, 12, 22)
```

That ordering is used by the strain interpolators and by the internal-variable spaces of the viscoelastic layer, so the elasticity tensor must use it too — otherwise applying C to the internal variable m would need a permutation at every node. It is *not* Voigt order (11, 22, 33, 23, 13, 12); a fixed permutation converts.

Scaling: use the **Mandel** (orthonormal) reduced basis, not Voigt's engineering-strain basis. With `a_s = 1` for diagonal components and `√2` for shear,

```
ε̂_s = a_s ε_jk,   σ̂_s = a_s σ_jk,   so that   ε:σ = Σ_s ε̂_s σ̂_s
Ĉ_st = a_s a_t C_(jk)(lm)          i.e.  Ĉ = D C_Voigt D,  D = diag(a)
```

Consequences that make this the right choice:

- Ĉ is a genuine symmetric matrix representing the linear map ε ↦ σ with respect to an orthonormal basis, so its eigenvalues are the tensor's eigen-stiffnesses, positive-definiteness is a plain matrix property, and rotations act on it by orthogonal 6×6 matrices.
- The strain energy is `½ ε̂ᵀ Ĉ ε̂` with no 2's or 4's anywhere; the element matrix is `Bᵀ Ĉ B` with a B that produces ε̂ directly.
- Isotropy is `Ĉ = λ 1̂1̂ᵀ + 2μ I` where `1̂` has 1 on the three (two) diagonal slots — and the bulk/deviatoric split is the pair of orthogonal projectors `P_vol = 1̂1̂ᵀ/d`, `P_dev = I − P_vol`. This is what the viscoelastic relaxation needs (§4).

Voigt is what geophysicists *write* (PREM's A, C, F, L, N; tomographic C_ij), so Voigt is the **input** convention of the general-anisotropy coefficient and conversions are provided; Mandel-in-library-order is the **internal** convention that every coefficient class emits and the integrator consumes. Two small static helpers make the conversions explicit and testable:

```cpp
struct SymmetricTensorBasis {           // in bilininteg.hpp next to SymmetricMatrixIndex
  static int  Size(int d);                          // n_s
  static int  Index(int d, int j, int k);           // = SymmetricMatrixIndex::ComponentOffset
  static real_t Scale(int j, int k);                // 1 or sqrt(2)
  static int  VoigtIndex(int d, int j, int k);      // 11,22,33,23,13,12
  static void FromVoigt(int d, const DenseMatrix &Cv, DenseMatrix &Cm);   // permute + D·Cv·D
  static void ToVoigt  (int d, const DenseMatrix &Cm, DenseMatrix &Cv);
  static void Pack  (int d, const real_t *Cijkl, DenseMatrix &Cm);        // full tensor -> Mandel
  static void Unpack(int d, const DenseMatrix &Cm, real_t *Cijkl);
  static void Apply (const DenseMatrix &Cm, const Vector &eps, Vector &sig); // tensor-component vectors in/out (scales internally)
};
```

---

## 2. Coefficient classes

All derive from `mfem::MatrixCoefficient` (height = width = n_s) so they compose with MFEM's algebra — `MatrixSumCoefficient`, `ScalarMatrixProductCoefficient`, `MatrixRestrictedCoefficient` (attribute-restricted), `PWMatrixCoefficient` (piecewise by attribute) — and inherit `SetTime`. A thin common base pins the convention:

```cpp
/// A MatrixCoefficient whose Eval() returns the elasticity tensor as an
/// n_s × n_s Mandel matrix in SymmetricTensorBasis ordering.
class ElasticTensorCoefficient : public mfem::MatrixCoefficient {
 public:
  explicit ElasticTensorCoefficient(int dim)
      : mfem::MatrixCoefficient(SymmetricTensorBasis::Size(dim)), _dim{dim} {}
  int SpaceDim() const { return _dim; }
 protected:
  int _dim;
};
```

The integrator accepts any `MatrixCoefficient` of size n_s (so sums/products of the classes below work), and documents that the convention is the caller's responsibility; the classes below guarantee it.

### 2.1 `IsotropicElasticTensorCoefficient(dim, Coefficient &lambda, Coefficient &mu)`

`Ĉ = λ 1̂1̂ᵀ + 2μ I`. Exists to (i) give the isotropic limit for tests against `ElasticityIntegrator`, (ii) let the anisotropic integrator be used everywhere so there is one code path. A second constructor from (κ, μ).

### 2.2 `TransverselyIsotropicElasticTensorCoefficient`

Constructors:

```cpp
// Love's constants (Dziewonski & Anderson 1981 notation) and a unit axis field.
TransverselyIsotropicElasticTensorCoefficient(int dim,
    Coefficient &A, Coefficient &C, Coefficient &F, Coefficient &L, Coefficient &N,
    VectorCoefficient &axis);
// PREM-style velocities: A = ρ v_PH², C = ρ v_PV², N = ρ v_SH², L = ρ v_SV², F = η (A − 2L).
static TransverselyIsotropicElasticTensorCoefficient FromVelocities(int dim,
    Coefficient &rho, Coefficient &vpv, Coefficient &vph, Coefficient &vsv, Coefficient &vsh,
    Coefficient &eta, VectorCoefficient &axis);   // owns the five derived ProductCoefficients
```

`Eval` builds the full tensor from the axis form and packs it:

```
C_ijkl = (A − 2N) δ_ij δ_kl
       + N (δ_ik δ_jl + δ_il δ_jk)
       + (F − A + 2N)(δ_ij n_k n_l + n_i n_j δ_kl)
       + (L − N)(δ_ik n_j n_l + δ_il n_j n_k + δ_jk n_i n_l + δ_jl n_i n_k)
       + (A + C − 2F − 4L) n_i n_j n_k n_l
```

(Checked against the canonical Voigt matrix for n = e₃: C₁₁ = A, C₃₃ = C, C₁₃ = F, C₄₄ = L, C₆₆ = N, C₁₂ = A − 2N.) The axis is normalised inside `Eval`, so the caller may pass any non-zero direction field. Because the formula is written with δ and n only, it is dimension-generic: in `d = 2` with an in-plane axis it yields exactly the **plane-strain** restriction of the 3-D TI tensor (the in-plane components of C_ijkl), which is the physically right 2-D model for a meridional slice with radial anisotropy. Plane stress is not supported (state it).

Axis fields: `RadialUnitVectorCoefficient(dim, x0 = 0)` for radial anisotropy (the PREM case), or any `VectorCoefficient`/`VectorGridFunctionCoefficient` for a general symmetry-axis field (e.g. from LPO models).

Moduli fields: the five `Coefficient`s are whatever the model supplies — `RadialModelCoefficient` (already in the library, `f(r, attribute)`) for PREM-like models, `PWConstCoefficient` per attribute, `GridFunctionCoefficient` for tomographic perturbations.

Debug check: `MFEM_ASSERT` the TI stability conditions once per `Eval` in debug builds (`L > 0, N > 0, C > 0, A > N, (A − N) C > F²`), which catches sign/units mistakes in inputs early.

### 2.3 `VoigtElasticTensorCoefficient(int dim, MatrixCoefficient &C_voigt)`

General anisotropy from *any* MFEM matrix coefficient in Voigt convention and Voigt ordering (6×6 or 3×3): `MatrixConstantCoefficient` for a literal single-crystal table, `MatrixFunctionCoefficient`, `PWMatrixCoefficient` per attribute, `MatrixArrayCoefficient` with 21 scalar coefficients. `Eval` = `FromVoigt`. This is the entry point for tomographic 21-component models.

### 2.4 `RotatedElasticTensorCoefficient(ElasticTensorCoefficient &C_local, MatrixCoefficient &R)`

Local-frame anisotropy: C given in a material frame, `R(x)` a rotation field. `Eval`: `Unpack → C'_ijkl = R_ia R_jb R_kc R_ld C_abcd → Pack` (81·81 flops per point in 3-D; negligible next to quadrature). The TI class could be implemented this way (canonical tensor + rotation to the axis), but the axis form is cheaper and avoids constructing a frame around n; keep both routes and test them against each other.

### 2.5 Relaxation splits (for the viscoelastic layer)

Backward-Euler and ETD stepping of an anisotropic Maxwell body need the stiffness written as `C = C_u + s(x) C_r` with `C_r` the part that relaxes. Two classes make the modelling choice explicit and keep the integrator ignorant of it:

- `DeviatoricProjectionElasticTensorCoefficient(C)` → `P_dev C P_dev`, and its complement `C − P_dev C P_dev`. For isotropic C this reproduces exactly `2μ dev-dev` and `κ div-div`; for TI it relaxes the shear-type response including the deviatoric part of A, C, F.
- `TransverselyIsotropicElasticTensorCoefficient` with only (L, N) — i.e. A = C = F = 0 in the axis formula: the "relax only the shear moduli" choice.

Either is then combined with MFEM's own `ScalarMatrixProductCoefficient(s, C_r)` and `MatrixSumCoefficient(C_u, s·C_r)` — no new integrator variant needed, and `ScalableDeviatoricStiffness` in `elastic.hpp` becomes "swap the coefficient and reassemble".

### 2.6 Later: `GridFunctionElasticTensorCoefficient`

21 (or 5 + axis) components stored in an L2 vector `GridFunction`, for models read from files. Straightforward once 2.1–2.4 exist; not in the first pass.

---

## 3. The integrator

```cpp
/// (u, v) ↦ ∫_Ω ε(v) : C : ε(u) dx for an anisotropic elasticity tensor C
/// supplied as an n_s × n_s MatrixCoefficient in Mandel form and
/// SymmetricTensorBasis ordering (see ElasticTensorCoefficient). Vector
/// H1 space with byNODES ordering, as for mfem::ElasticityIntegrator.
class ElasticTensorIntegrator : public mfem::BilinearFormIntegrator {
  mfem::MatrixCoefficient &C;
#ifndef MFEM_THREAD_SAFE
  mfem::DenseMatrix dshape, gshape, B, Cq, CB;
#endif
 public:
  ElasticTensorIntegrator(mfem::MatrixCoefficient &C, const mfem::IntegrationRule *ir = nullptr);
  void AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Tr, DenseMatrix &elmat) override;
  // later: ComputeElementFlux / ComputeFluxEnergy for ZZ estimation; AssembleElementMatrix2 not needed
};
```

Per quadrature point:

```
gshape = dshape · J⁻¹                         (dof × d, physical gradients)
B (n_s × d·dof), column (c, i) = dof·c + i:
   diagonal row s(j,j):   B[s, (j, i)] = gshape(i, j)
   shear row  s(j,k), j>k: B[s, (j, i)] = gshape(i, k)/√2,   B[s, (k, i)] = gshape(i, j)/√2
Cq = C.Eval(Tr, ip)   (verify size n_s once)
elmat += w · Bᵀ Cq B        via Mult(Cq, B, CB); AddMult_a_AtB(w, B, CB, elmat)
```

`ε̂ = B u` by construction (the 1/√2 rows give √2·ε_jk = (∂_k u_j + ∂_j u_k)/√2), so `Bᵀ Ĉ B` is exactly the discrete form of `∫ ε:C:ε` with no engineering-strain factors. Cost per point O(n_s (d·dof)²), comparable to MFEM's isotropic loop; a blocked variant that avoids forming B (assembling the d×d blocks from `gshape(:,j) gshape(:,k)ᵀ` weighted by Ĉ entries) can come later if profiling asks for it.

Default quadrature: MFEM's `2·OrderGrad(el)`; a spatially varying C may want more — expose through the usual `IntRule`.

Assertions: `C.GetHeight() == C.GetWidth() == n_s(dim)`; `dim == Tr.GetSpaceDim()`.

Relation to the existing integrators: the internal-variable coupling `∫ (C_r m) : ε(v)` stays with the existing unit-coefficient `DomainSymmetricMatrixStrainIntegrator` / `…TraceFree…` plus a **pointwise** application of `C_r` at the internal-variable nodes via `SymmetricTensorBasis::Apply` — exactly the "material weighting at the nodes" structure of `viscoelasticity.cpp`, now with a tensor instead of `2μ`. That is why the ordering must match (§1).

---

## 4. Tests (`tests/TestElasticTensor.cpp`, `tests/TestElasticTensorIntegrator.cpp`)

Pointwise (coefficient) tests, random points, d = 2 and 3:

1. `FromVoigt ∘ ToVoigt = id`; `Pack ∘ Unpack = id`; `Apply` agrees with the unpacked contraction `σ_jk = C_jklm ε_lm`.
2. Isotropic class equals `FromVoigt` of the textbook Voigt matrix; its eigenvalues are `{3κ, 2μ (×5)}` (`{2κ_2D, 2μ (×2)}` in 2-D).
3. TI with axis e₃ reproduces the canonical Voigt matrix from (A, C, F, L, N); TI with A = C = λ + 2μ, F = λ, L = N = μ and a *random* axis equals the isotropic class; `FromVelocities` with PREM values at a few radii is positive definite.
4. Rotation covariance: TI with axis `R e₃` equals `RotatedElasticTensorCoefficient(TI with axis e₃, R)` for random rotations R. Same for a random Voigt matrix rotated two ways (Unpack-rotate-Pack vs Mandel 6×6 orthogonal transform).

Element-matrix tests (existing `TestCommon.hpp` machinery, all element types, orders 1–3):

5. Isotropic tensor: `ElasticTensorIntegrator` element matrices equal `mfem::ElasticityIntegrator` to round-off on every element of a random-jiggled mesh.
6. Symmetry of `elmat`; positive semi-definiteness; the rigid modes (translations, rotations from `solvers.hpp`) are in the null space of the assembled matrix for a TI material with a random axis field.
7. Energy patch test: for a linear displacement field u = G x on a single element or a small mesh, `½ uᵀ A u = |Ω| ½ ε̂ᵀ Ĉ ε̂` exactly (up to quadrature round-off) for a *constant* TI tensor — checks B, scaling and assembly together.
8. Rotation covariance at element level: rotate the mesh nodes by R and the axis by R; `elmat' = (I_dof ⊗ R) elmat (I_dof ⊗ R)ᵀ` (in byNODES layout this is a block permutation, easy to apply).
9. 2-D/3-D consistency: a plane-strain 2-D TI problem with in-plane axis versus a 3-D slab with one element through thickness and `u_z = 0` — same in-plane element energies.

Physics-level (later, with the self-gravitating problem): Love numbers / surface loading of a radially anisotropic PREM sphere against your radial codes; this is the test that validates the *use* of the tensor, not its algebra.

---

## 5. Phases

| Phase | Deliverable | Effort |
|---|---|---|
| A | `SymmetricTensorBasis`; `ElasticTensorCoefficient` base; Isotropic, TI (both constructors), Voigt, Rotated, `RadialUnitVectorCoefficient`; tests 1–4 | 1 d |
| B | `ElasticTensorIntegrator`; tests 5–9; an `ex2`-style example (static TI sphere under a surface load, isotropic limit compared with `ElasticityIntegrator`) | 1 d |
| C | Relaxation-split coefficients (§2.5) and wiring into the viscoelastic layer's `ScalableDeviatoricStiffness`; `GridFunctionElasticTensorCoefficient` | 1 d, when the implicit stepper is built |
| D (optional) | Blocked assembly without B; `ComputeElementFlux` for ZZ; partial-assembly kernel (GPU) | — |

Phase A/B are independent of the SubMesh work and of the solver work; they can be done first or in parallel.

---

**Status (2 Sep 2026): Phases A and B done**, in `include/mfemElasticity/elastic_tensor.hpp` / `src/elastic_tensor.cpp`, with the §6 recommendations taken as decided: library ordering with Mandel scaling internally, Voigt only at input; the integrator takes a plain `MatrixCoefficient&` with a size check; name `ElasticTensorIntegrator`; plane strain in 2-D. Notes:

- `SymmetricTensorBasis` also provides `Component` (inverse of `Index`), the volumetric/deviatoric projectors and `RotationMatrix` (the Mandel representation `Q` of `ε ↦ R ε Rᵀ`, orthogonal). `RotatedElasticTensorCoefficient` uses `Q C Qᵀ` rather than the 81·81 loop; the tests check the two against each other.
- `DeviatoricProjectionElasticTensorCoefficient` (§2.5, first variant) is included: `P_dev C P_dev` or its complement; for isotropic C the tests confirm `2μ` dev-dev and `d κ` on the volumetric projector. The "(L, N) only" TI split is just the TI class with A = C = F = 0 and needs no code. Wiring into the viscoelastic layer (anisotropic branches `C_k`, tensor force integrator) is the viscoelastic plan's Phase 4 and is not done.
- `GridFunctionElasticTensorCoefficient` (§2.6) is not done.
- MFEM trap met: the dense products `Mult`, `MultABt`, `MultAAt`, `AddMult_a_AtB` do **not** resize their output (only `MFEM_ASSERT`, i.e. nothing in release builds); an unsized output silently produces an empty result. Every product in the library and tests is sized explicitly, and the test comparison helper treats a size mismatch as failure.
- Tests: `tests/TestElasticTensor.cpp` (tests 1–4 plus the coefficient-level 2-D/3-D consistency of test 9; 2-D and 3-D) and `tests/TestElasticTensorIntegrator.cpp` (tests 5–9: isotropic agreement with `ElasticityIntegrator` to 1e-13 on curved (order-2) meshes for tri/quad/tet/hex and orders 1–3; symmetry, non-negativity and the rigid modes in the null space for a radially anisotropic TI material — on an *isoparametric* geometry, since a rigid rotation on a curved element is only representable when the geometry order does not exceed the displacement order; the energy patch test; element-level rotation covariance; a plane-strain quad sheet against an extruded hex slab). `examples/anisotropic_elasticity.cpp` solves a radially anisotropic clamped body and, with `-iso`, reproduces `ElasticityIntegrator` to 2e-16.

## 6. Decisions to confirm

- **Ordering**: library `SymmetricMatrixIndex` order with Mandel scaling internally; Voigt only at input/output. (Alternative — Voigt throughout — would force per-node permutations and √2 bookkeeping in the viscoelastic coupling; not recommended.)
- **Integrator argument type**: plain `MatrixCoefficient&` (composable with MFEM algebra) rather than the narrower `ElasticTensorCoefficient&` (safer). Recommendation: `MatrixCoefficient&`, with the size assertion; the convention is documented on the integrator.
- **Name**: `ElasticTensorIntegrator` (says what it takes) vs `AnisotropicElasticityIntegrator` (says what it is for). Either; pick one and use it for the header `elastic_tensor.hpp` too.
- **2-D semantics**: plane strain only.
