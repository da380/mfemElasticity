# Elastic tensors: convention, coefficients and integrator

Notes for `elastic_tensor.hpp` and the `ElasticTensorIntegrator` of
`bilininteg.hpp`. MFEM's `ElasticityIntegrator` is isotropic (two scalar
coefficients) and nothing in MFEM handles a fourth-order stiffness tensor.
The library adds one integrator that takes the tensor as a matrix
coefficient, and a family of coefficients that produce that matrix in one
fixed convention.

## Convention: a Mandel matrix in the library's component order

A symmetric second-order tensor in `d` dimensions has `n_s = d(d+1)/2`
components. The library orders them by `SymmetricComponentOrder::Offset`
(`index.hpp`), lower triangle, column-major:

```
d = 3:  (11, 21, 31, 22, 32, 33)  ->  0 … 5
d = 2:  (11, 21, 22)              ->  0 … 2
```

The strain interpolators and the viscoelastic internal variables use this
order, so the elastic tensor uses it too; otherwise applying `C` to an
internal variable would need a permutation at every node. It is *not* Voigt
order (11, 22, 33, 23, 13, 12).

The scaling is **Mandel** (an orthonormal reduced basis), not Voigt's
engineering strain. With `a_s = 1` for diagonal components and `√2` for
shear ones,

```
ε̂_s = a_s ε_jk,   σ̂_s = a_s σ_jk,   ε : σ = Σ_s ε̂_s σ̂_s,   Ĉ_st = a_s a_t C_(jk)(lm).
```

What this buys:

- Ĉ is a genuine symmetric matrix representing ε ↦ σ in an orthonormal
  basis. Its eigenvalues are the eigen-stiffnesses, positive-definiteness is
  a plain matrix property, and a rotation `R` acts as `Q Ĉ Qᵀ` with `Q`
  orthogonal (`SymmetricTensorBasis::RotationMatrix`).
- The strain energy is `½ ε̂ᵀ Ĉ ε̂` with no factors of 2 or 4, and the element
  matrix is `Bᵀ Ĉ B` with a `B` that produces ε̂ directly.
- Isotropy is `Ĉ = λ 1̂1̂ᵀ + 2μ I`, and the bulk/deviatoric split is the pair
  of orthogonal projectors `P_vol = 1̂1̂ᵀ/d`, `P_dev = I − P_vol`, which is
  what viscoelastic relaxation needs.

Voigt is what geophysicists write (Love's A, C, F, L, N; tomographic C_ij),
so Voigt is an **input** convention with explicit conversions
(`SymmetricTensorBasis::FromVoigt` / `ToVoigt`, `Pack` / `Unpack` for the
full C_ijkl). Mandel in library order is the **internal** convention that
every coefficient emits and the integrator consumes.

## Coefficients

All derive from `mfem::MatrixCoefficient` (n_s × n_s), so they compose with
MFEM's algebra: `MatrixSumCoefficient`, `ScalarMatrixProductCoefficient`,
`PWMatrixCoefficient`.

- **`IsotropicElasticTensorCoefficient`**: from (λ, μ) or (κ, μ). It provides
  the isotropic limit for tests and lets the anisotropic path serve
  everywhere.
- **`TransverselyIsotropicElasticTensorCoefficient`**: Love's A, C, F, L, N
  and a symmetry-axis field `n`, or PREM-style velocities and η,

  ```
  C_ijkl = (A − 2N) δ_ij δ_kl + N (δ_ik δ_jl + δ_il δ_jk)
         + (F − A + 2N)(δ_ij n_k n_l + n_i n_j δ_kl)
         + (L − N)(δ_ik n_j n_l + δ_il n_j n_k + δ_jk n_i n_l + δ_jl n_i n_k)
         + (A + C − 2F − 4L) n_i n_j n_k n_l .
  ```

  For n = e₃ this is the canonical Voigt matrix (C₁₁ = A, C₃₃ = C, C₁₃ = F,
  C₄₄ = L, C₆₆ = N, C₁₂ = A − 2N). The axis is normalised inside `Eval`.
  `RadialUnitVectorCoefficient` gives radial anisotropy. The formula is
  written with δ and n only, so it is dimension-generic: in 2-D with an
  in-plane axis it yields the **plane-strain** restriction of the 3-D
  tensor. Plane stress is not supported.
- **`VoigtElasticTensorCoefficient`**: general anisotropy from any MFEM
  matrix coefficient in Voigt form and order.
- **`RotatedElasticTensorCoefficient`**: a tensor given in a material frame
  together with a rotation field, applied as `Q Ĉ Qᵀ`.
- **`DeviatoricProjectionElasticTensorCoefficient`**: `P_dev C P_dev` or its
  complement. For an isotropic `C` these are exactly 2μ dev–dev and
  κ div–div. Together with a TI coefficient holding only L and N, these are
  the two ready-made choices of "the part that relaxes" for an anisotropic
  Maxwell body (`viscoelasticity.md`).

## The integrator

`ElasticTensorIntegrator(MatrixCoefficient& C)` assembles ∫ ε(v) : C : ε(u)
on a vector H1 space in byNODES layout. It accepts any `MatrixCoefficient`
of size n_s, so sums and products of the classes above work; the convention
is then the caller's responsibility, and the classes above guarantee it. Per
quadrature point, with `g` the physical shape gradients (dof × d) and
columns indexed by (component c, dof i):

```
diagonal row s(j,j):     B[s, (j,i)] = g(i,j)
shear row s(j,k), j > k: B[s, (j,i)] = g(i,k)/√2,   B[s, (k,i)] = g(i,j)/√2
elmat += w · Bᵀ Ĉ B
```

so that `ε̂ = B u` by construction. The default quadrature is MFEM's
`2·OrderGrad(el)`; a rapidly varying `C` may want more through the usual
`IntRule`.

The internal-variable coupling ∫ (C_k m) : ε(v) does not go through this
integrator. It uses the unit-coefficient strain integrators with `C_k`
applied pointwise at the internal-variable nodes, which is why the
component orders must match.

## How it is tested

The isotropic tensor reproduces `mfem::ElasticityIntegrator` to round-off on
curved meshes. A TI tensor whose moduli are isotropic equals the isotropic
class for a random axis. Rotating the axis agrees with rotating the tensor,
and rotating mesh and axis together rotates the element matrix. A linear
displacement field gives the exact energy `|Ω| ½ ε̂ᵀ Ĉ ε̂`. Rigid modes lie in
the null space for a radially anisotropic material, on isoparametric
geometry (`null_space.md` explains why). A plane-strain sheet matches an
extruded slab with u_z = 0.
