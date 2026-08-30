# mfemElasticity — status review and roadmap

*Review date: 30 August 2026. Branch `develop` at `7ce9054` plus uncommitted work in `examples/`. MFEM 4.9.1 (development clone at `/home/david/dev/mfem`, `ec39b3509c`, 8 April 2026).*

This document records where the library stands, what was checked, an assessment of the cross-mesh (parent/SubMesh) coupling machinery, and a proposed order of work. It is written for the two people developing the code, so it assumes familiarity with both GJI papers (Yu, Al-Attar, Syvret & Lloyd 2025, *ggae388*; Yu, Myhill & Al-Attar 2026, *ggag231*).

---

## 1. What was looked at

- Every file in `include/`, `src/`, `tests/`, and the examples that touch coupling or time stepping (`elastogravity*.cpp`, `ex5`–`ex8`, `grid_transfer*.cpp`, `elastic.hpp`, `quasi_static_elasticity.cpp`, `viscoelasticity.cpp`, `ex1`, `ex2`).
- Branch relationships: `main` vs `develop` vs `developZY4`, and the uncommitted diff in the working tree.
- The two papers, read for the weak forms actually being discretised.
- MFEM's native `SubMesh`/`ParSubMesh`, `TransferMap`/`ParTransferMap`, `fem/transfer.hpp`, `MixedBilinearForm`, `BlockOperator`, `ElasticityIntegrator`, and a search for any DtN / anisotropic elasticity support.
- A fresh serial build (`build_serial`, clean, no warnings) and smoke runs of `ex2`, `ex5`, `ex6`, `ex7`, `ex8`, `grid_transfer`, `quasi_static_elasticity`, `viscoelasticity`.

---

## 2. Where things stand

### 2.1 Library (`include/mfemElasticity`, `src/`)

| Component | What it is | State | Tests |
|---|---|---|---|
| `bilininteg.hpp/cpp` — `DomainVector*`, `DomainDiv*`, `DomainMatrix*`, `DomainSymmetricMatrix*`, `DomainTraceFree*` integrators; `Index` helpers; `DeformationGradient/Strain/DeviatoricStrainInterpolator` | Mixed integrators between vector/scalar/tensor nodal spaces: all the non-standard terms in the self-gravitating form (ρ v·∇(u·∇Φ), ρ∇Φ·v div u, the internal-variable coupling m:d(v)), plus nodal strain interpolators | Mature, documented, consistent style | gtest coverage for every integrator and interpolator |
| `lininteg.hpp/cpp` — `DomainLFDeformationGradientIntegrator` | ∫ M:∇v for a matrix coefficient M | Done | gtest |
| `poisson.hpp/cpp` — `PoissonDtNOperator`, `PoissonMultipoleOperator`, `PoissonLinearisedMultipoleOperator`, `TransformedDiffusionIntegrator`, helpers | The *ggag231* machinery: low-rank matrix-free DtN (B = CᵀC), multipole RHS operators (static and linearised), split boundary communicator in parallel; 2D and 3D | Done, serial + parallel; verified by `ex2` against the offset uniform sphere | No unit tests (only the example) |
| `mesh.hpp/cpp` — markers, `SphericalBoundaryRadius`, `MeshCentroid`, `SphericalMeshHelper`, `SplitBoundaryCommunicator` | Geometry helpers for the spherical outer boundary | Done | None |
| `legendre.hpp/cpp` | Normalised associated Legendre recursions for the harmonic expansions | Done | Implicitly via `ex2` |
| `solvers.hpp/cpp` — `RigidTranslation`, `RigidRotation`, `RigidBodySolver` | Wraps a solver with projections orthogonal to the six (three in 2D) rigid modes | Done, serial + parallel | None |
| `submesh.hpp/cpp` — `SubMeshProlongationMatrix`, `MixedBilinearFormSubMesh`, `ParMixedBilinearFormSubMesh` | The custom cross-mesh coupling (§3) | Works in the examples; design open | None |
| `elasticity.hpp` / `elasticity.cpp` | A stale sketch of `QuasiStaticElasticityProblem`; the `.cpp` is empty; the header is not in the umbrella `mfemElasticity.hpp` | Superseded by `examples/elastic.hpp` | — |
| `radial_model.hpp/cpp` | `RadialModelCoefficient` (namespace `RadialModel`, not `mfemElasticity`; duplicate `#pragma once`; not in the umbrella header) | Orphan | — |

### 2.2 Examples

| Example | Purpose | Ran? |
|---|---|---|
| `ex1`/`ex1p` | Static elasticity with tractions + `RigidBodySolver`; working tree adds a half-written `DisplacementCentroidShift` (empty `Mult`) and a post-hoc centroid-removal block | builds |
| `ex2`/`ex2p` | Poisson on the whole space: Neumann / DtN / multipole, static and linearised, vs the uniform-sphere exact solution | yes (2D and 3D DtN, multipole; converges; `-res` only writes a GLVis field, no printed error norm) |
| `ex3`/`ex3p` | Pull-back (transformed-domain) Laplace via `TransformedDiffusionIntegrator` | builds |
| `ex4`/`ex4p` | DtN operator demo | builds |
| `ex5` | `SubMeshProlongationMatrix` sanity check | yes — sub/parent linear-form values agree to 1e-15 |
| `ex6` | Monolithic block system across mesh/submesh built from P and `BlockMatrix::CreateMonolithic` | yes |
| `ex7`/`ex7p`, `ex8`/`ex8p` | Coupled Poisson pair (submesh Ω₁ ⊂ Ω₂) with an exact solution: block Gauss–Seidel (`ex7`) vs monolithic block MINRES via `MixedBilinearFormSubMesh` (`ex8`) | yes — both give identical L2 errors (ψ₁ 0.061, ψ₂ 0.030 on a coarse order-1 mesh), which is a good consistency check of `MixedBilinearFormSubMesh` |
| `grid_transfer`/`_p` | Block Gauss–Seidel using MFEM's `SubMesh::Transfer` only | yes, converges in 10 iterations |
| `elastogravity`/`_p` | **The target problem**: static self-gravitating elastic sphere under a surface load, block Gauss–Seidel with under-relaxation ω = 0.5 | **not built on `develop`** — the targets exist only in `developZY4`'s `examples/CMakeLists.txt`; needs `ex5_2d.msh`, which is not in `data/` |
| `elastogravity_block`/`_p` | Same problem, monolithic MINRES + block-diagonal preconditioner + a coupled rigid-body null-space projector | same |
| `quasi_static_elasticity` | Driver for the new `elastic.hpp` abstraction | yes (traction and clamped) |
| `viscoelasticity` | Maxwell internal-variable model on top of `elastic.hpp`, ETD1 / RK stepping | **fails** — see §2.4 |
| `seismic`, `wave`, `ode` | Time-domain wave examples (develop only) | builds |

### 2.3 Branches and hygiene

- `main` ⊂ `develop`: `develop` is 3 small commits ahead (seismic updates, `ex1`).
- `developZY4` vs `develop`: ZY4 has the `elastogravity*` CMake targets and a reworked `meshing/offset_disk.cpp`; `develop` has the newer `ode`/`seismic`/`wave`/`quasi_static`/`viscoelasticity` examples that ZY4 lacks. Nothing conflicts in library code — the two lines only diverge in `examples/CMakeLists.txt`, `ex1*.cpp`, `offset_disk.cpp`. A merge of ZY4 into develop is cheap and should be done before either of you goes further.
- Uncommitted: `examples/elastic.hpp`, `viscoelasticity.cpp` (new), rewritten `quasi_static_elasticity.cpp`, `ex1.cpp` edits. Worth committing as a WIP so the postdoc can see the abstraction.
- `apps/CMakeFiles/**` (compiler-probe outputs, `a.out`) is tracked in git; `.gitignore` has `**/*build*` but not `CMakeFiles`. `texput.log` and the two PDFs are also at the top level.
- Build dirs are configured with `BUILD_TESTS=OFF`, so the gtest suite is not being run routinely.

### 2.4 Problems found when running things

1. **`viscoelasticity` aborts on the first time step** (`star.mesh`, `-p 0`): "elastic solve failed". Cause: `TractionProblem` sets `iterative_mode = true` on the rigid-body wrapper, so the second solve warm-starts from the converged displacement. MFEM's `CGSolver` measures the *relative* tolerance against the **initial** residual, which is already ~1e-12·‖b‖, so the target becomes ~1e-24 and CG stalls at 10 000 iterations. This is the same pitfall the postdoc worked around with "adaptive abs tol" after iteration 1 in `elastogravity.cpp`. The fix is a proper absolute tolerance tied to ‖b‖ (or a residual test relative to ‖b‖), not `iterative_mode = false`. Worth solving once, in the library, since warm starts are exactly what time stepping wants.
2. **`viscoelasticity` segfaults on `beam-quad.mesh`** (any `-p`, any order) in `DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator::AssembleElementMatrix2`, `src/bilininteg.cpp:519` (`w *= Q->Eval(Trans, ip)`). A Debug build under gdb shows `Q` is an **uninitialised member**: `include/mfemElasticity/bilininteg.hpp` declares `mfem::Coefficient* Q;` in `DomainSymmetricMatrixStrainIntegrator` and `DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator` without a default, and the coefficient-free constructor never sets it (the other integrators use `Q = nullptr`). `viscoelasticity.cpp` is the first caller to construct it without a coefficient; on `star.mesh` the stack garbage happened to be zero. Two-character fix; the gtest suite passes only because every test passes a coefficient.
3. **Likely index bug** in `TransformedDiffusionIntegrator` for the scalar-`Coefficient` (radial) path, `src/poisson.cpp:1327-1332`: `F(j,k) = x(j)*df(j)` assigns the same value to every column of row j; for ξ = f(x)x the Jacobian is `F(j,k) = f δ_jk + x_j ∂_k f`, i.e. `x(j)*df(k)`. `ex3` uses the vector-coefficient path (`RadialDiffeomorphismCoefficient`), so this branch is untested.
4. `elastogravity_block.cpp`'s `BlockRigidBodySolverLocal` builds the coupled null vectors (u = a + b×x, φ = −u·∇Φ) correctly but then orthonormalises and projects with an ad-hoc inner product that ignores the φ block whenever the u block is non-zero (`BlockDot`, `alpha_phi = 0`). The resulting projector is oblique, not orthogonal; it happens to work because the u-parts are already orthogonal after Gram–Schmidt, but it is not the range projection MINRES needs (see §4.3).
5. gmsh meshes load with "Elements with wrong orientation: 10624/16136 (fixed)"; harmless, but the meshing tools could emit consistently oriented elements.

---

## 3. Coupling across a mesh and its SubMesh

### 3.1 The problem

The self-gravitating quasi-static system (eq. 3 of *ggae388*) couples

- **u** on the body M (a `SubMesh` of the full mesh), through κ div-div + 2μ dev-dev + the two ρ∇Φ terms, and
- **φ** on the full ball B ⊃ M (body + buffer + DtN on ∂B), through (4πG)⁻¹∫_B ∇φ·∇φ' + the DtN term,

with the off-diagonal coupling ∫_M ρ ∇φ·u' (and its transpose). The off-diagonal integrals are over M, but one of the two fields lives on the parent mesh. That is the only place where two meshes meet, and it is what MFEM does not provide.

### 3.2 What MFEM offers natively (checked in the 4.9.1 source)

- `SubMesh`/`ParSubMesh` keep the parent element/vertex/face maps and the parent's partition (no repartitioning). Curved (high-order nodes) and nonconforming parents are supported; NURBS is not.
- `TransferMap`/`ParTransferMap` (and `SubMesh::Transfer`) are a **pure vdof permutation with sign flips**, built by `SubMeshUtils::BuildVdofToVdofMap` (`mesh/submesh/submesh_utils.cpp:93-215`). The sub and parent spaces must have the same FE collection, order and vdim. In parallel `ParTransferMap` reconciles shared dofs with a `GroupCommunicator::Sum` and divides by multiplicity. The map itself (`sub_to_parent_map_`) is private, which is why the repo re-derives it via the public `BuildVdofToVdofMap`.
- `MixedBilinearForm::Assemble` (`fem/bilinearform.cpp:1550-1600`) indexes trial and test by the same element id and uses the test space's transformation: **trial and test must be on the same mesh**. There is no native mixed form spanning a mesh and its submesh; the `multidomain` miniapp couples subdomains only through `Transfer` inside a segregated iteration.
- `fem/transfer.hpp` operators all assume a refinement hierarchy or the same mesh; arbitrary non-matching meshes need GSLIB or Moonolith.
- `BlockOperator`, `BlockDiagonalPreconditioner`, `BlockLowerTriangularPreconditioner` accept any `Operator*` blocks, including rectangular and matrix-free ones.
- No DtN, no infinite elements, no anisotropic elasticity integrator anywhere in the library or miniapps.

So what you built is genuinely necessary; the question is only the form it should take.

### 3.3 The three mechanisms currently in the repo

**(a) `SubMesh::Transfer` inside a segregated loop** (`grid_transfer.cpp`, `ex7`): assemble each block on its own mesh; move the *other* field across with `Transfer` and evaluate the coupling as a same-mesh form on the submesh. Uses only native MFEM. Only supports fixed-point/Gauss–Seidel type couplings — you never hold the off-diagonal block as an operator, so no Krylov method can see the whole system.

**(b) `SubMeshProlongationMatrix`** (`submesh.cpp:11-76`; used in `ex5`, `ex6`): an explicit sparse injection P (parent vdofs × sub vdofs, entries ±1, one per column). Off-diagonal blocks are then products: `A_01 = M_sub Pᵀ`, `A_10 = P M_sub` where `M_sub` is any form assembled entirely on the submesh between the sub space and a "shadow" copy of the parent space. Serial only at present. Clean, general (any integrator type, boundary or interior-face included), and the off-diagonal blocks never need custom assembly. `ex6` shows it composing into a monolithic `BlockMatrix`.

**(c) `MixedBilinearFormSubMesh` / `ParMixedBilinearFormSubMesh`** (`submesh.cpp:80-360`; used in `ex8`, `elastogravity*`): a re-implementation of `MixedBilinearForm::Assemble` that loops over submesh elements and remaps the vdofs of one side through the vdof map before `AddSubMatrix`. Serial and parallel; the parallel version works because `ParMixedBilinearForm::ParallelAssemble` does Pᵀ_test · mat · P_trial with each space's own prolongation, and `mat` is already in the local-vdof layout of both spaces. Verified against (a) by `ex7`/`ex8` agreeing to all printed digits.

Assessment of (c):

- It copies ~100 lines of MFEM internals, twice (the `Par` class is a verbatim duplicate of the serial one), and depends on protected members (`elemmat`, `trial_vdofs`, `domain_integs_marker`) whose semantics shift between MFEM releases. It silently ignores boundary, trace-face and interior-face integrators and partial assembly. `static_cast` rather than `MFEM_VERIFY` on the SubMesh.
- The `extended_trial` flag plus the requirement to construct and pass a "shadow" space by hand (`fes_phi_cond`) makes the call sites hard to read (`MixedBilinearFormSubMesh(&fes_phi, &fes_u, &fes_phi_cond, true)`).
- What it produces is *exactly* `M_sub Pᵀ` (or `P M_sub`). The custom assembly buys nothing over (b) except saving one sparse product.

### 3.4 Recommendation: one injection operator, everything else native

Keep the `SubMesh` design (it is the right one: no wasted u-dofs in the buffer shell, no pinned dofs, clean null space), but make the *only* custom object a single, well-tested injection operator, and express all coupling through it:

```
Π : sub true-dofs → parent true-dofs,   Π ∈ {0, ±1},  one non-zero per column
Πᵀ Π = I                                 (restriction ∘ prolongation is exact)
Π Πᵀ = diag(1 on parent dofs that lie in M, 0 elsewhere)
```

- Serial: what `SubMeshProlongationMatrix` already is (rename, keep).
- Parallel: `Π = R_parent · P_loc · P_sub` as a `HypreParMatrix`, where `P_loc` is the local ±1 map, `P_sub` and `R_parent` are the sub space's prolongation and the parent space's restriction. Because each sub true-dof is owned by exactly one rank and its value is consistent across ranks, this equals `P_parentᵀ · P_loc · R_subᵀ`, so Πᵀ is simultaneously the correct *primal* restriction (parent → sub, exact) and the correct *dual* prolongation. Both facts follow from Π being a boolean injection. This is easiest to build directly from the vdof map and the two true-dof offset arrays (a CSR with one entry per row) rather than via `RAP`.
- Coupling blocks are then compositions, never re-assembled:  
  `A_uφ = M_sub,uφ̃ · Πᵀ`, `A_φu = Π · M_sub,φ̃u` with `M_sub` a plain `(Par)MixedBilinearForm` on the submesh between the u space and the shadow φ̃ space (same FEC as φ, on the submesh).  
  In serial, `Mult(SparseMatrix, SparseMatrix)`; in parallel `ParMult`. Or leave them as `ProductOperator`s inside a `BlockOperator` — only the diagonal blocks need explicit matrices for preconditioning.
- Essential BCs: eliminate on the diagonal blocks as usual and zero the corresponding rows/columns of the products (`FormRectangularSystemMatrix` semantics), or — simpler for this application — there are no essential dofs on u or φ at all, only the DtN, so nothing to eliminate.
- The same Π built from `SubMesh::CreateFromBoundary` gives you *surface* couplings for free: the fluid–solid interface terms of *ggae388* Appendix A (∫_Σ ρ⁻ g (n·u)(n·u') and ∫_Σ ρ⁻ (φ u' + φ' u)·n) are boundary forms on a codim-1 submesh with two Π's (one to the solid submesh, one to the parent). This is the strongest argument for the Π design: (c) would need a third and fourth re-implementation for boundary integrators.
- Retire `MixedBilinearFormSubMesh`, or keep it as a ten-line convenience that calls the above. Tests: for random grid functions check Πᵀ Π = I and that `Transfer` and Π agree; check the product `M_sub Πᵀ` equals the current `MixedBilinearFormSubMesh` matrix entrywise on `ex7.msh`; run the same in parallel on 1, 2, 4 ranks including a rank with no submesh elements (MFEM has no test for that case).

One caveat on granularity: `ParSubMesh` inherits the parent partition, so the u-problem is load-balanced only as well as the body is spread over ranks. For an Earth model with a thin buffer that is fine; for a small body in a large ball it is not, and there is no native rebalancing of a `ParSubMesh`. Keep the buffer thin (the *ggag231* result that b/a ≈ 1.2–1.4 with ℓ_max ≈ 8–16 already reaches the discretisation floor is what makes this design viable).

---

## 4. Solving the coupled elastogravity system

### 4.1 Structure

With the DtN folded into the φ block the discrete system is

```
[ A_uu    A_uφ Πᵀ ] [u]   [f_u]
[ Π A_φu  A_φφ+B  ] [φ] = [f_φ]
```

- Symmetric (the bilinear form is symmetric; `A_uφ = A_φuᵀ`), by eq. 6 of *ggae388*.
- **Indefinite**: `A_uu` is positive semi-definite (for a gravitationally stable model), `A_φφ + B` is positive definite in 3D (semi-definite in 2D, constant mode), but the coupling makes the whole thing a saddle-point matrix (u minimises, φ maximises). MINRES is therefore the right Krylov method for the monolithic form; CG is not.
- Null space (3D): the six coupled rigid vectors `(a + b×x, −(a + b×x)·∇Φ)`. In 2D add the constant in φ (the log term), which is what the `OrthoSolver`/"mass/length" bookkeeping in the 2D branches is handling.
- Coupling strength is not small: ρgR/μ ≈ 5500·9.8·6.4e6/1.4e11 ≈ 2.4 for the Earth. That is why the block Gauss–Seidel in `elastogravity.cpp` needs under-relaxation and should not be expected to converge robustly for all models; it is a fixed-point iteration whose contraction factor depends on this ratio.

### 4.2 Options, in order of recommendation

1. **CG on the Schur complement in u.** `S = A_uu − A_uφ Πᵀ (A_φφ + B)⁻¹ Π A_φu` is symmetric positive semi-definite with the rigid null space only. Each `S` application costs one φ solve (AMG-preconditioned CG on `A_φφ + εM`, DtN matrix-free, both set up once). This is what the time-stepping loop wants: the elastic solve is re-entered at every step (or RK stage) with a new RHS only, warm starts are natural, and the `RigidBodySolver` wrapper applies unchanged. Inner tolerances can be relaxed relative to the outer one (inexact Schur complement → use flexible CG or a fixed inner iteration count).
2. **MINRES on the block system with a block-diagonal SPD preconditioner** — what `elastogravity_block` does. Right method; fix the null-space projector (below), and in parallel use `HypreBoomerAMG` with `SetElasticityOptions` on `A_uu` (already there) and AMG on `A_φφ + εM`. Expect iteration counts to grow with the coupling strength; a block-triangular preconditioner inside FGMRES is the usual next step if MINRES is slow.
3. **Block Gauss–Seidel / under-relaxed fixed point** — current `elastogravity.cpp`. Fine as a reference and for weakly coupled bodies; not a default.

### 4.3 Null space: two different projections

Two operations are being conflated in the current code (`RigidBodySolver` and `BlockRigidBodySolverLocal`):

- *Range compatibility*: the RHS must satisfy b ⊥ null(A) in the **Euclidean** dof inner product (because A is symmetric, range(A) = null(A)^⊥). This is the projection that makes CG/MINRES well-posed, and for physically consistent loads it is a no-op up to round-off (zero net force and torque, and for the coupled system the φ-parts are automatically consistent).
- *Solution gauge*: which representative u + (a + b×x) to return. The physically meaningful choice is zero net momentum and angular momentum, i.e. orthogonality in the **ρ-weighted L² inner product** (mass matrix), not in dof space. The post-processing "centroid shift" in the working-tree `ex1` is exactly this for translations; it belongs inside `RigidBodySolver` as an option (`M`-orthogonal projection using `VectorMassIntegrator(ρ)`), with the rotations handled the same way.

For the coupled system the null vectors have a φ-part, so the projector must act on the full block vector with a consistent (block-Euclidean, or block-mass) inner product; `BlockDot`'s `alpha_phi = 0` should go. Generalise `RigidBodySolver` to take an arbitrary list of null vectors (and an optional mass operator) so that the elastic-only and coupled cases share code.

### 4.4 Tolerances and warm starts

As found in §2.4, MFEM's relative tolerance is relative to the initial residual. Any warm-started solve must set `abs_tol` from ‖b‖ (or the first step's residual). Put this logic in one place (the problem class's `Solve`), not in each example.

---

## 5. The elastic / viscoelastic layer (`examples/elastic.hpp`, `viscoelasticity.cpp`)

This is the newest and, in design terms, the best-thought-out part of the tree. The `AssembleForce(t) → AddForce(f) → Solve()` protocol with a persistent operator and a hidden φ is the right seam: a `SelfGravitatingElasticProblem` implementing the same interface (carrying φ internally and using the Schur-complement solver of §4.2) drops straight under `ViscoelasticOperator` without touching the time integrator. Suggestions:

- Promote it into the library (`include/mfemElasticity/elasticity.hpp`, replacing the stale sketch), with the serial/parallel split handled once (a `FiniteElementSpace*` plus `dynamic_cast<ParFiniteElementSpace*>` at the few points that differ — the pattern the Poisson operators already use).
- `MaxwellViscoelasticOperator` assumes the same μ as the elastic problem "by documented invariant". Better to have the elastic problem *export* its material coefficients so the viscoelastic layer reads them; a mismatched μ is a silent error.
- Time stepping: ETD1 is first order in the coupling. ETD2RK (one extra elastic solve per step) or the backward-Euler path already documented (`ScalableDeviatoricStiffness`, reassemble κ div-div + s·2μ dev-dev) are the two obvious upgrades; for GIA with dt ≫ τ_min in the asthenosphere both are unconditionally stable. Note the reassembly in the implicit path also invalidates the AMG setup each time dt changes — acceptable if dt is piecewise constant.
- Internal variable on L² nodes with nodal interpolation of d(u): exact per element when `m_order ≥ order − 1`; the alternative `M⁻¹B` projection you flag for adjoint consistency is the one to adopt once the adjoint is in scope.
- The two failures in §2.4 are in this layer and are the first things to fix.

---

## 6. Anisotropic `ElasticityIntegrator`

MFEM's `ElasticityIntegrator` (`fem/bilininteg.cpp:3208-3287`) is isotropic (λ, μ scalar coefficients), full assembly only for what you need (the PA/EA paths and `ElasticityComponentIntegrator` are GPU-oriented and can be ignored). There is no anisotropic integrator anywhere in MFEM, so this is new code, but it is short:

- **General anisotropy**: `AnisotropicElasticityIntegrator(MatrixCoefficient& C)` with C the Voigt matrix (3×3 in 2D, 6×6 in 3D, engineering-strain convention). At each quadrature point build the strain–displacement matrix `B` (`(dim(dim+1)/2) × dim·dof`) from `CalcPhysDShape` and accumulate `w BᵀCB`. Ordering must match `SymmetricMatrixIndex` (lower-triangle, column-major) so the existing strain interpolators and tests can be reused. ~150 lines, mirrors the isotropic loop structure.
- **Transverse isotropy**: a `TransverselyIsotropicCoefficient : MatrixCoefficient` taking (A, C, L, N, F) as `Coefficient`s (Love's notation, or equivalently the five Cᵢⱼ) and a symmetry axis as a `VectorCoefficient` (for radial anisotropy `n = x/|x|`). Build Cᵢⱼₖₗ from the standard axis form  
  `C = (A−2N) δδ + N(δδ+δδ) + (F−A+2N)(δ nn + nn δ) + (L−N)(δ nn + … four terms) + (A+C−2F−4L) nnnn`  
  and pack to Voigt. Isotropic limit A = C = λ+2μ, L = N = μ, F = λ recovers `ElasticityIntegrator` exactly, which gives the first test; the second is invariance under rotating the axis with the mesh; the third is a 1D layered TI benchmark against the radial codes.
- Viscoelastic split: the implicit path scales "the deviatoric modulus". For anisotropic C the bulk/deviatoric split is a modelling choice (e.g. relax only the shear-type moduli L, N, and the deviatoric part of A, C, F), not a mathematical identity. Decide this before wiring `ScalableDeviatoricStiffness` for the anisotropic integrator; the general integrator should just take C and a second "relaxable part" C_v if needed.
- `ComputeElementFlux` (ZZ error estimation) can be added later; not needed now.

---

## 7. Housekeeping worth doing early

- Merge `developZY4` → `develop`; add the `elastogravity*` targets and the missing `ex5_2d.msh` generation recipe (or a small canned mesh) so the target problem builds from `develop`.
- Commit the WIP (`elastic.hpp`, `viscoelasticity.cpp`, `quasi_static_elasticity.cpp`); remove `apps/CMakeFiles` from git and add `CMakeFiles/`, `*.log`, `*.pdf` to `.gitignore`.
- Turn `BUILD_TESTS=ON` in the configure scripts / build dirs and run ctest routinely; add tests for the Poisson operators (exact uniform-sphere solutions exist in `uniform_sphere.hpp`), `RigidBodySolver`, and the injection operator.
- Delete `elasticity.hpp`/`.cpp` (stale) and fold `radial_model` into the namespace and umbrella header or drop it.
- Serial/parallel duplication (`mesh.cpp`, `submesh.cpp`, the three Poisson operators' constructors) can be collapsed with the `dynamic_cast` pattern; not urgent but it halves the surface for bugs.
- `MFEM_THREAD_SAFE` branches exist but several classes still declare the scratch members unconditionally; either commit to the macro or drop it.

---

## 8. Proposed order of work

Ordered so that each step is testable on its own and unblocks the next; rough effort in developer-days.

1. **Stabilise the base** (1–2 d): merge ZY4; commit WIP; tests on; fix the CG tolerance bug and the `beam-quad` segfault; verify `ex2` prints an L2 error.
2. **Injection operator Π, serial + parallel, with tests** (2–3 d): replace `SubMeshProlongationMatrix`/`MixedBilinearFormSubMesh` call sites in `ex6`, `ex8`, `elastogravity*`; check ex7/ex8 agreement is preserved; parallel test with empty ranks.
3. **Library `elasticity.hpp`** (2 d): promote the `QuasiStaticLinearElasticProblem` design, serial + parallel; generalised null-space projector (list of null vectors, optional mass weighting); tolerance handling.
4. **`SelfGravitatingElasticProblem`** (3–5 d): Π-based coupling blocks, Schur-complement CG as default with MINRES/block-diagonal as an alternative, DtN assembled once, 2D constant-mode handling; reproduce `elastogravity` results; then a Love-number / surface-load benchmark of a PREM-like sphere against your radial codes (this is the first real physics verification of the whole stack and should come before anything below).
5. **Anisotropic integrator + TI coefficient + tests** (2–3 d): independent of 2–4, can run in parallel with them.
6. **Viscoelastic upgrades** (3 d): ETD2 or implicit stepping; material coefficients exported from the elastic problem; gravity-coupled viscoelastic run (GIA-style relaxation of a uniform sphere, checkable against the radial codes).
7. **Fluid core / interfaces** (later): boundary submesh + Π for the Appendix-A terms; this is where the Π design pays off.
8. **Adjoint** (later): the symmetric structure means the adjoint is the forward solver with different loads; the `M⁻¹B` strain projection and time-reversed stepping are the only new pieces.

---

### Addendum: gtest run

A Debug configure of the repo with `BUILD_TESTS=ON` (googletest fetched at configure time) builds cleanly and **all 75 tests pass** (integrators and interpolators, dims 1–3, orders, tri/quad/tet/hex), ~32 s. The suite does not exercise the coefficient-free constructors, which is how §2.4 item 2 slipped through; a one-line test per integrator constructing it without a coefficient would have caught it.
