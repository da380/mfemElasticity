# Mixed bilinear forms across a mesh and its SubMesh — design and plan

*30 August 2026. Companion to `status_and_roadmap.md` §3. Written against MFEM 4.9.1 (`ec39b3509c`).*

## 0. What we are replacing, and why the idea is right

`MixedBilinearFormSubMesh` (`src/submesh.cpp`) does the right thing conceptually: it assembles the coupling **on the submesh**, using a *shadow* space (a copy of the parent-mesh space's FE collection placed on the submesh) for the parent-side field, and then re-labels the shadow's vdofs as parent vdofs when scattering element matrices into the global matrix. The integrand only ever sees one mesh — the submesh — so any integrator works unchanged. That is the correct decomposition and it should be kept.

What is wrong is the *execution*: the class re-implements `MixedBilinearForm::Assemble` by hand, and that has consequences:

1. **Only domain integrators are assembled.** `Assemble` loops `domain_integs` and nothing else. `ex11`/`ex12` (fluid core models) add `BoundaryFluxMixedIntegrator`s to `a13`, `a31`, `a23`, `a32` for the fluid–solid interface terms of *ggae388* Appendix A; with the library as committed on `develop` those calls are accepted and then **silently ignored**. Either Ziheng has uncommitted library changes, or those runs are missing the interface coupling. Check with him before anything else.
2. The serial and parallel classes are verbatim copies (~130 lines each) of MFEM internals, relying on protected members (`elemmat`, `trial_vdofs`, `domain_integs_marker`) and on the exact `AssembleElementMatrix2`/`TransformDual` calling sequence, which has already changed twice across MFEM 4.5→4.9.
3. `Assemble` in `MixedBilinearForm` is not virtual; the derived `Assemble` hides it, so a call through a base pointer assembles nothing. (MFEM's own `DiscreteLinearOperator` does the same, so this is tolerated in MFEM style, but it must be documented.)
4. The caller must build the shadow space by hand and pass it with a boolean (`MixedBilinearFormSubMesh(&fes_phi, &fes_u, &fes_phi_cond, true)`), so every call site carries a fourth object and an easily-inverted flag. `static_cast<SubMesh*>` instead of a check.
5. No tests.

The design below keeps the idea and removes the hand-written assembly entirely: the geometric part becomes a *standard* `MixedBilinearForm` on the submesh, and the cross-mesh part becomes a small, testable, purely algebraic object.

---

## 1. The principle

> Assemble on the submesh with MFEM's own `MixedBilinearForm`, between the submesh space and a shadow of the parent space. Then apply a dof injection to re-index one side of the resulting sparse matrix from shadow vdofs to parent vdofs. Nothing else is custom.

Two layers, each independently testable:

| Layer | Object | Responsibility |
|---|---|---|
| A | `SubMeshDofInjection` | The vdof map shadow ↔ parent (signed); vector transfer; CSR row/column re-indexing; a true-dof `HypreParMatrix` Π in parallel |
| B | `SubMeshMixedBilinearForm` / `ParSubMeshMixedBilinearForm` | A drop-in `MixedBilinearForm` whose two spaces live on a mesh and its SubMesh; `Assemble()` = helper form on the submesh + Layer A re-indexing; everything else inherited |

Two facts of the MFEM API make Layer B almost free:

- `MixedBilinearForm(FiniteElementSpace *tr, FiniteElementSpace *te, MixedBilinearForm *mbf)` (`fem/bilinearform.cpp:1347`) creates a form on *different* spaces that **borrows** `mbf`'s integrators and markers (`extern_bfs = 1`, so it does not delete them). Domain, boundary, trace-face and boundary-trace-face lists are all copied. (Interior-face integrators are not copied by this constructor in 4.9 — an MFEM omission; see §5.)
- `ParMixedBilinearForm::ParallelAssemble(SparseMatrix *m)` (`fem/pbilinearform.cpp`) forms `P_testᵀ · m · P_trial` from *any* local matrix `m` in the two spaces' local-vdof numbering. A re-indexed local matrix is exactly such an `m`, so the parallel class needs no parallel-specific assembly code at all.

---

## 2. Layer A — `SubMeshDofInjection`

### 2.1 Interface

```cpp
namespace mfemElasticity {

/// Signed injection of the vdofs of a space on a SubMesh into the vdofs of
/// the corresponding space on the parent mesh. Built from
/// SubMeshUtils::BuildVdofToVdofMap; valid for From::Domain and
/// From::Boundary submeshes. The sub space must be the "shadow" of the
/// parent space: same FiniteElementCollection object, vdim and ordering.
class SubMeshDofInjection : public mfem::Operator {
 public:
  /// Build the injection for an existing shadow space.
  SubMeshDofInjection(const mfem::FiniteElementSpace &sub_fes,
                      const mfem::FiniteElementSpace &parent_fes);

  /// Convenience: construct (and own) the shadow of parent_fes on submesh.
  static std::unique_ptr<mfem::FiniteElementSpace>
  MakeShadowSpace(const mfem::FiniteElementSpace &parent_fes,
                  mfem::SubMesh &submesh);              // Par overload below

  int SubVSize() const;      // = Width()
  int ParentVSize() const;   // = Height()

  /// y_parent = P x_sub  (zero outside the submesh)
  void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
  /// y_sub = Pᵀ x_parent  (exact restriction)
  void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;

  /// P as an explicit sparse matrix (parent_vsize × sub_vsize, entries ±1).
  std::unique_ptr<mfem::SparseMatrix> NewSparseMatrix() const;

  /// Given M (sub-rows × sub-cols) return a matrix (parent-rows × sub-cols)
  /// with row i of M placed at row map(i) and scaled by sign(i).  == P M
  std::unique_ptr<mfem::SparseMatrix> RemapRows(const mfem::SparseMatrix &M) const;
  /// Given M (any-rows × sub-cols) return (any-rows × parent-cols).  == M Pᵀ
  std::unique_ptr<mfem::SparseMatrix> RemapColumns(const mfem::SparseMatrix &M) const;

#ifdef MFEM_USE_MPI
  /// Π : sub true-dofs → parent true-dofs, as a HypreParMatrix (§2.4).
  std::unique_ptr<mfem::HypreParMatrix> NewTrueDofMatrix() const;
#endif

 private:
  mfem::Array<int>  parent_vdof_;   // |map|, size sub_vsize
  mfem::Array<real_t> sign_;        // ±1
  const mfem::FiniteElementSpace *sub_fes_, *parent_fes_;
};

}
```

`SubMeshProlongationMatrix` becomes `NewSparseMatrix()` (same code, kept).

### 2.2 Construction

```
submesh = dynamic_cast<const SubMesh*>(sub_fes.GetMesh());   MFEM_VERIFY(submesh)
MFEM_VERIFY(submesh->GetParent() == parent_fes.GetMesh())
MFEM_VERIFY(sub_fes.FEColl() == parent_fes.FEColl() && same vdim && same ordering)
SubMeshUtils::BuildVdofToVdofMap(sub_fes, parent_fes, submesh->GetFrom(),
                                 submesh->GetParentElementIDMap(), raw_map);
for i: parent_vdof_[i] = FiniteElementSpace::DecodeDof(raw_map[i], sign_[i]);
```

`BuildVdofToVdofMap` asserts identical element dof counts on the two sides; the shadow space guarantees it. In parallel the same call works on `ParFiniteElementSpace` (it is element-local; `ParSubMesh` reuses the parent partition, so every submesh element's parent element is local). `MakeShadowSpace` is `new FiniteElementSpace(&submesh, parent_fes.FEColl(), parent_fes.GetVDim(), parent_fes.GetOrdering())` (the FEC object is *shared*, which also makes `SubMesh::Transfer` usable on the shadow). For `From::Boundary` submeshes the shadow is the trace space: the parent's `H1_FECollection(p, 3)` on the 2-manifold mesh works as-is.

### 2.3 CSR re-indexing (exact, O(nnz))

Column remap (parent on the trial side): copy `I`; for each nonzero `k`: `J'[k] = parent_vdof_[J[k]]`, `A'[k] = sign_[J[k]] · A[k]`; width = parent vsize. Optionally `SortColumnIndices()`.

Row remap (parent on the test side): count nonzeros per sub row; `I'[parent_vdof_[i]+1] = I[i+1]-I[i]`; prefix-sum; copy row `i` into row `parent_vdof_[i]` with `A' = sign_[i]·A`. Rows of the parent matrix outside the submesh are empty. Height = parent vsize.

Both are equal to `P·M` and `M·Pᵀ` (test: compare with `mfem::Mult` on the explicit `P`), but need no sparse product and preserve the nonzero pattern exactly.

### 2.4 True-dof injection Π (parallel)

Needed for block operators, the Schur-complement solver, and transferring true-dof vectors.

> **Correction (1 Sep 2026, found during implementation).** The construction this section originally led with —
> `Π = R_parent · P_loc · P_sub` via two `LeftDiagMult` calls — is **wrong** whenever a shared parent dof on the
> submesh boundary is owned by a rank whose local elements there all lie outside the submesh: `R_parent` selects only
> *owned* parent ldofs, and on the owning rank the corresponding `P_loc` row is empty, so the Π entry is silently
> lost. This is not exotic: a 4×4 quad mesh, submesh = the half x > 0.5, slab-partitioned on 2 ranks, order-2 H1
> already fails (‖ΠᵀΠx − x‖∞ ≈ 1.47; the dofs on the interface x = 0.5 are owned by rank 0, which has no submesh
> elements). Very plausibly the reason earlier hand-rolled parallel maps "never worked".
>
> The implemented construction is what was previously listed as the fallback: build **Πᵀ row by row over *owned sub*
> true dofs**. The owner of a sub true dof always has the submesh element (ParSubMesh inherits the parent partition),
> and `ParFiniteElementSpace::GetGlobalTDofNumber(parent_ldof)` returns the correct global parent true dof even for
> *unowned* parent ldofs (conforming spaces), so no ownership case is ever missed:
>
> ```
> for each sub ldof l:  lt = sub_pfes->GetLocalTDofNumber(l);  if (lt < 0) continue;
>     J[lt] = parent_pfes->GetGlobalTDofNumber(parent_vdof_[l]);  data[lt] = sign_[l];
> Πᵀ = HypreParMatrix(comm, sub_tdofs, glob_sub, glob_parent, I=identity-offsets, J, data,
>                     sub tdof offsets, parent tdof offsets);      Π = Πᵀ.Transpose();
> ```

Π is a boolean injection (one ±1 per column, at most one per row), so **Πᵀ is at once the exact primal restriction parent→sub and the correct dual prolongation** — the two uses never need separate operators. Πᵀ Π = I; Π Πᵀ = diag(indicator of parent dofs in the submesh).

`ParSubMesh::Transfer`/`ParTransferMap` reconcile shared dofs at the L-vector level with a `GroupCommunicator::Sum` divided by multiplicity; Π does the same at the T-vector level with no custom communication. Test them against each other.

---

## 3. Layer B — `SubMeshMixedBilinearForm`

### 3.1 Interface (serial)

```cpp
/// A MixedBilinearForm whose trial and test spaces live on a mesh and on a
/// SubMesh of that mesh (either way round). Integrals are taken over the
/// SubMesh (its elements, boundary elements or faces), so every integrator
/// type accepted by MixedBilinearForm is supported. Markers refer to the
/// SubMesh's attributes (domain attributes are inherited from the parent;
/// boundary attributes are inherited where the parent had a boundary element,
/// and equal max(parent bdr attr)+1 on the cut).
class SubMeshMixedBilinearForm : public mfem::MixedBilinearForm {
 public:
  SubMeshMixedBilinearForm(mfem::FiniteElementSpace *trial_fes,
                           mfem::FiniteElementSpace *test_fes);

  /// Hides MixedBilinearForm::Assemble (non-virtual), as
  /// DiscreteLinearOperator does. Call through this type.
  void Assemble(int skip_zeros = 1);

  const SubMeshDofInjection &Injection() const;
  bool ParentIsTrial() const;

 private:
  std::unique_ptr<mfem::FiniteElementSpace> shadow_;   // owned
  std::unique_ptr<SubMeshDofInjection> injection_;
  bool parent_is_trial_;
};
```

Constructor logic:

```
sub  = whichever of {trial, test} has SubMesh::IsSubMesh(mesh) && GetParent() == other mesh
MFEM_VERIFY(exactly one such side, "... use MixedBilinearForm when both spaces share a mesh")
shadow_   = MakeShadowSpace(parent_side_fes, *submesh)
injection_ = SubMeshDofInjection(*shadow_, parent_side_fes)
```

`Assemble`:

```
MFEM_VERIFY(assembly == AssemblyLevel::LEGACY && ext == nullptr, "partial assembly not supported")
MixedBilinearForm helper(parent_is_trial ? shadow_ : trial_fes,
                         parent_is_trial ? test_fes : shadow_,
                         this);                        // borrows integrators + markers
helper.Assemble(skip_zeros);
helper.Finalize(skip_zeros);
delete mat;
mat = (parent_is_trial ? injection_->RemapColumns(helper.SpMat())
                       : injection_->RemapRows(helper.SpMat())).release();
```

That is the whole class. `SpMat()`, `Mult`, `AddMult`, `EliminateTrialDofs`, `EliminateTestDofs`, `FormRectangularSystemMatrix` (which wraps `mat` in a `RectangularConstrainedOperator` using the *real* spaces' prolongations and essential lists), `FormRectangularLinearSystem`, `RecoverFEMSolution` — all inherited and correct, because they only read `mat`, `trial_fes`, `test_fes`.

### 3.2 Parallel

```cpp
class ParSubMeshMixedBilinearForm : public mfem::ParMixedBilinearForm {
 public:
  ParSubMeshMixedBilinearForm(mfem::ParFiniteElementSpace *trial, mfem::ParFiniteElementSpace *test);
  void Assemble(int skip_zeros = 1);   // identical body; helper is a serial MixedBilinearForm
};
```

The helper is deliberately a **serial** `MixedBilinearForm` on the `ParFiniteElementSpace`s (they are `FiniteElementSpace`s; `MixedBilinearForm::Assemble` is element-local and this is precisely what `ParMixedBilinearForm` inherits anyway). After re-indexing, the inherited `ParallelAssemble()` does `P_testᵀ mat P_trial` via `MakeRectangularBlockDiag` + `RAP`. Ranks with no submesh elements produce an empty local matrix of the right size and contribute nothing — correct.

To avoid the duplicated `Assemble` body, put it in a free function `void AssembleOnSubMesh(MixedBilinearForm &self, FiniteElementSpace *shadow, const SubMeshDofInjection &, bool parent_is_trial, int skip_zeros)` that both classes call; it needs `mat` — either make it a `protected static` of a small CRTP mixin, or give the base a friend. Simplest: a mixin template `SubMeshMixedFormImpl<Base>` holding the shadow/injection and implementing `Assemble` with `Base::mat`. Two one-line classes then instantiate it for `MixedBilinearForm` and `ParMixedBilinearForm`.

### 3.3 Compatibility shim

Keep `MixedBilinearFormSubMesh(tr, te, sub, extended_trial)` for one release as a deprecated thin wrapper that ignores `sub` (after verifying it *is* the shadow) and forwards; then delete. All 26 call sites in the examples become two-argument constructions.

---

## 4. What this covers

| Coupling | Meshes | How |
|---|---|---|
| ∫_M ρ ∇φ·u′ (elastogravity, ex9–ex12) | u on domain-submesh M, φ on parent | domain integrator on M, shadow of φ |
| Fluid–solid interface ∫_Σ ρ⁻ φ (u′·n), ∫_Σ ρ⁻g (n·u)(n·u′) | u on solid submesh, φ on parent; Σ = boundary of the solid submesh | **boundary integrator** on the submesh with the shadow of φ — the case currently dropped |
| Surface load / sea level ∫_∂M σ (φ′ + u′·∇Φ) | σ on Σ = `CreateFromBoundary(parent)`, φ on parent (or u on M) | domain integrator on Σ; shadow of φ on Σ is the trace space; `From::Boundary` map |
| DtN, multipole | φ on parent only | unchanged |
| Two displacement submeshes sharing an interface | sibling submeshes | **not needed** for the Earth model (the fluid carries no u); if ever needed, compose two injections through the root parent as `TransferMap` does for SubMesh↔SubMesh (§7) |

---

## 5. Constraints and edge cases (decide up front, assert in code)

- **Shadow = same FEC object.** Not just "same order": `BuildVdofToVdofMap` needs identical per-element dof layout, and sharing the collection pointer is the only way to guarantee it and keep `SubMesh::Transfer` usable on the shadow. Verify in the constructor.
- **Ordering.** The shadow copies the parent's `Ordering` (byNODES/byVDIM); the map is built on vdofs so both work. Test both.
- **H1 and L2 only, initially.** The signed map handles H(div)/H(curl) orientation flips, but `DofTransformation` for ND/RT of order ≥ 2 on tets is applied inside the helper's `Assemble` on *submesh* orientations; whether those coincide with the parent's is exactly what `TransferMap::CorrectFaceOrientations` exists to fix. `MFEM_VERIFY(!fes->IsVariableOrder() && continuity in {H1, L2})` until there is a test.
- **Nonconforming parents.** `NCSubMesh` exists; the vdof map is element-based and unaffected; hanging-node constraints are applied by `FormRectangularSystemMatrix` through each real space's conforming prolongation, which is the right place. Untested; assert `!Nonconforming()` for now or add a test.
- **Interior-face integrators.** Not copied by the borrow constructor; and `ParMixedBilinearForm::ParallelAssemble` takes a different path when `interior_face_integs.Size() > 0` (face-neighbour columns) that a re-indexed matrix would break. Assert none for now. If needed later: push them into the helper via `GetFBFI()` and use the serial path only.
- **Partial assembly / device.** Out of scope; assert `LEGACY`.
- **`Update()` after mesh refinement.** Out of scope; rebuild the object.
- **Non-virtual `Assemble`.** Documented; keep the call sites typed as the derived class (they are).
- **`skip_zeros`.** Pass through to the helper; the re-indexing preserves the pattern.
- **NURBS.** Unsupported by SubMesh itself.

---

## 6. Tests (gtest, `tests/TestSubMeshCoupling.cpp`; parallel ones under `MFEM_USE_MPI` with 1, 2, 4 ranks)

Meshes: `data/circular_offset.msh` (2D, two attributes, curved order-2 boundary), `data/spherical_offset.msh` / `ex7.msh` (3D tets), a Cartesian quad/hex mesh with a marked half (fast, exact geometry).

1. **Injection algebra.** `Pᵀ P = I`; `P Pᵀ` idempotent; `Mult`/`MultTranspose` agree with `NewSparseMatrix()`; `MultTranspose` of a projected analytic parent function equals `SubMesh::Transfer` of it to the shadow (both orderings; H1 and L2; orders 1–3).
2. **Re-indexing = product.** `RemapRows(M) == Mult(P, M)` and `RemapColumns(M) == Mult(M, Pᵀ)` entrywise for a random-coefficient mass matrix `M` on the submesh.
3. **Form = old class.** On `ex7.msh`, `SubMeshMixedBilinearForm` with `GradientIntegrator(ρ)` equals the current `MixedBilinearFormSubMesh` matrix entrywise, both orientations (trial/test on parent). Keep the old class in the test until removal, then freeze its matrix as a reference `.mtx`.
4. **Boundary integrators.** Submesh = attribute-1 region of a mesh with an internal boundary; `SubMeshMixedBilinearForm` with `AddBoundaryIntegrator(MassIntegrator, marker)` where the trial space is the parent's, versus a plain `MixedBilinearForm` on the *parent* with the same boundary integrator and marker, restricted to the submesh rows via `Pᵀ`: identical. This is the test that would have caught the ex11/ex12 problem.
5. **Boundary submesh.** Σ = `CreateFromBoundary(parent, outer)`; form on Σ between an L2 space on Σ and the trace shadow of an H1 parent space with a mass integrator, versus `BoundaryLFIntegrator`-style parent assembly: identical.
6. **Consistency.** `ex7` (Gauss–Seidel via `Transfer`) and `ex8` (block MINRES via the new class) still give identical L2 errors — already true today; must stay true.
7. **Parallel.** The parallel form's `ParallelAssemble()` on 1, 2, 4 ranks applied to a projected analytic trial function, then dotted with a projected test function, equals the serial number to round-off; Π from `NewTrueDofMatrix()` satisfies ΠᵀΠ = I and `Πᵀ x` equals `ParSubMesh::Transfer` on true dofs; a partition in which at least one rank owns no submesh element (force it with a tiny submesh and 4 ranks).
8. **Empty-rank + boundary integrators in parallel.** Same as 4 with 4 ranks.

---

## 7. Phases

| Phase | Deliverable | Effort |
|---|---|---|
| 1 | `SubMeshDofInjection` (serial), `SubMeshMixedBilinearForm`, tests 1–4, 6; old class kept | 1–2 d |
| 2 | `ParSubMeshMixedBilinearForm` via the mixin; `NewTrueDofMatrix()`; tests 7–8 | 1–2 d |

**Status (1 Sep 2026):** Layer A is implemented and tested — `SubMeshDofInjection` in `include/mfemElasticity/submesh.hpp` / `src/submesh.cpp` (serial + `NewTrueDofMatrix()`, §2.4 as corrected above), with `tests/TestSubMeshDofInjection.cpp` (gtest: 192 configurations over dim × element type × order 1–3 × H1/L2 × vdim × ordering, domain and boundary submeshes; tests 1–2 of §6) and `tests/TestSubMeshDofInjectionPar.cpp` (standalone MPI program, registered with ctest at 1/2/4 ranks: 1920 checks per rank count, including empty-rank partitions and submesh boundaries aligned with rank boundaries on both ownership sides; the Π parts of test 7). Layer B and the remaining tests are still to do.

Worked examples for review: `examples/submesh_injection.cpp` and `examples/submesh_injection_p.cpp` (on `data/circular_offset.msh`) demonstrate field transfer in both directions and solve a self-checking toy block system — Poisson on the parent coupled to an auxiliary field on the submesh through a cross-mesh mass matrix, built via `RemapRows`/`RemapColumns` in serial and as `ParMult(Π, M̂)` in parallel (the §3 Π-composition pattern, by hand). Both validate against a monolithic single-mesh solve (agreement ~1e-14) and against `u = Πᵀφ`. One MFEM trap encountered and worth remembering for Layer B call sites: in serial legacy assembly, `FormLinearSystem`'s returned `X` *aliases* the grid function's memory and `RecoverFEMSolution` relies on that aliasing — after solving in a `BlockVector`, copy the block back explicitly instead of calling `RecoverFEMSolution`.
| 3 | Boundary-submesh shadows (`From::Boundary`), test 5; migrate the 26 call sites; delete `MixedBilinearFormSubMesh`; `SubMeshProlongationMatrix` → `NewSparseMatrix()`; docs | 1 d |
| 4 (optional) | Root-parent composition for sibling submeshes (only if a use appears); matrix-free `SubMeshMixedOperator` for partial assembly; a `SubMeshDiscreteLinearOperator` for things like `GradientInterpolator` parent→submesh-L2 (today done as `Transfer` + `Grad`, which is fine) | — |

Phase 1 is independent of everything else in the roadmap and is the natural first coding task; Phases 2–3 unblock the `SelfGravitatingElasticProblem` (roadmap step 4).

---

## 8. Immediate action

Before Phase 1: confirm with Ziheng whether his working copy has a `MixedBilinearFormSubMesh::Assemble` that handles boundary integrators. If not, `ex11`/`ex12` results on `develop` do not include the interface terms, and any comparison made with them should be re-run once Phase 1 lands.
