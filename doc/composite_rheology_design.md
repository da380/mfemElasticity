# Composite rheologies: different materials in different regions of one displacement space

*Draft for review, 3 Sep 2026; §5 answered and both phases implemented the same day (§4 end, §6).*

## 1. What is wanted

A single `LinearQuasiStaticProblem` carries one displacement space on a (possibly disconnected) SubMesh of the solid regions. Since 3 Sep 2026 there is no second displacement field, so the only way to give the mantle, the lithosphere and the inner core different materials is through the one `Rheology` the problem is built with. Today that works for **parameters** (every modulus and relaxation time is an `mfem::Coefficient`, so a `PWCoefficient` by attribute already gives each region its own κ, μ, μ_k, τ_k), but not for **structure**:

- different numbers of Maxwell branches per region (say two in the mantle, one in the lithosphere, none in the inner core);
- an elastic region inside a viscoelastic body without paying for internal variables there;
- an anisotropic region inside an isotropic body (or the reverse);
- a state-dependent relaxation law in one region only.

The workaround available now is a global Maxwell rheology with branch moduli that vanish where the branch does not apply (an elastic region is a region where every μ_k is zero), and the anisotropic path with an isotropic tensor where the body is isotropic. It is exact, but stores and evolves internal variables everywhere, samples tensors where scalars would do, and makes the model description opaque. The design below makes the region structure explicit, in two phases: the first changes only the rheology layer and reproduces the workaround's results with a clean description; the second removes the waste in the operator.

## 2. Regions

A region is a set of element attributes of the displacement mesh. On a SubMesh the attributes are inherited from the parent mesh, so the regions are exactly the gmsh physical volumes the model was built from (`data/elastogravity_three_layer_*.msh` already distinguishes mantle and inner core this way). Regions are given as `mfem::Array<int>` markers sized to `mesh.attributes.Max()`, the convention used everywhere else in the library; they must be disjoint and cover the mesh (verified at construction: every attribute present on the mesh is marked exactly once).

Everything below relies on one fact already used by `ViscoelasticOperator`: the internal-variable nodes are L2 nodes, hence element-interior, so any attribute-wise datum is sampled without ambiguity. The `ElementTransformation` passed to every coefficient and to `BranchModulus`/`UnrelaxedModulus` carries `Attribute`, so dispatch by region costs one array lookup.

## 3. Phase 1: `CompositeRheology` (rheology layer only)

```cpp
struct RheologyRegion {
  mfem::Array<int> marker;   // element attributes of the region
  const Rheology* rheology;  // not owned; must outlive the composite
};

class CompositeRheology : public Rheology {
 public:
  CompositeRheology(int dim, std::vector<RheologyRegion> regions);
  int NumRegions() const;
  const Rheology& Region(int r) const;
  const mfem::Array<int>& RegionMarker(int r) const;
  int BranchRegion(int k) const;      // region owning global branch k
  int LocalBranch(int k) const;       // its index within that region
  ...
};
```

**Branches.** The composite's branch list is the concatenation of the regions' lists, in region order. Branch `k` belongs to region `r(k)` and is local branch `j(k)` there. All per-branch virtuals dispatch: `RelaxationTime(k)` returns a `RegionCoefficient` wrapping the region's τ (value inside the region, a positive dummy of 1 outside, never used because the modulus vanishes there); `Law(k)` is the region's law; `BranchShearModulus(k)` wraps the region's μ_k with zero outside; `BranchModulus(k, T, ip, C)` returns the region's tensor for `T.Attribute` in the region and zero otherwise. `UnrelaxedModulus(T, ip, C)` dispatches to the region containing `T.Attribute`. `IsLinear()` is the conjunction over regions (it is already computed from `Law(k)` in the base class, so nothing to do).

**Trace-free or full.** `TraceFreeInternalVariables()` is true only when every region is trace-free. Otherwise the operator uses full symmetric tensors everywhere, and the isotropic regions present their branches through `BranchModulus` as @f$2\mu_k P_{dev}@f$, which `IsotropicMaxwellRheology` already implements (the anisotropic path with an isotropic tensor is tested to reproduce the isotropic operator to 1e-10). No new code in the operator.

**Stiffness.** `ElasticStiffness::AddIntegrators(form)` gains an optional attribute marker, passed on to `BilinearForm::AddDomainIntegrator(integ, marker)`, which MFEM supports for domain integrators. `CompositeStiffness` holds one `ElasticStiffness` per region (from `Region(r).MakeStiffness()`) and adds each with its marker. `SetRelaxationWeights(beta)` receives one coefficient per *global* branch and slices the vector per region; `ClearRelaxationWeights()` and `IsRelaxed()` forward. Since the weights only enter through the region's own integrators, whatever the nodal β_k are outside the region is irrelevant.

**Elastic regions** are regions whose rheology has no branches (`IsotropicElasticRheology`, `AnisotropicElasticRheology`); they contribute integrators and nothing else. A body that is elastic everywhere is a composite of elastic regions and passes through the operator with an empty state, as now.

**Cost.** Identical to the workaround: state size Σ_k n_c n_d over all global branches, i.e. each branch stored on the whole mesh. For a mantle with two branches and an elastic inner core, roughly the inner core's share of the internal variables is wasted; for a lithosphere with its own branch, that branch is stored under the whole mantle as well. Tolerable for the Love-number benchmark, not for production 3-D runs with several regional branch sets, which is what Phase 2 is for.

**Tests** (serial and MPI 1/2/4).
1. *Split of a homogeneous body*: a Maxwell bar divided into two attribute regions with the same rheology in each equals the unsplit bar to round-off for every scheme (displacement and internal variables).
2. *Masking equivalence*: elastic + Maxwell regions, and one-branch + two-branch Maxwell regions, equal the global-rheology workaround (μ_k as `PWCoefficient`s) to round-off.
3. *Stiffness*: the composite stiffness of two isotropic elastic regions equals a single isotropic stiffness with `PWCoefficient` moduli, unrelaxed and weighted.
4. *Mixed isotropic/anisotropic regions*: an isotropic region next to an anisotropic one with the same (isotropic) tensor equals the all-isotropic composite.
5. *Self-gravitating*: a PREM-like sphere with an elastic inner core and a Maxwell mantle runs through `LinearQuasiStaticSelfGravitatingProblem` unchanged (smoke test; the physics is the Love-number benchmark).

Estimated effort: one day, including the tests. No change to any problem class or to `ViscoelasticOperator`.

## 4. Phase 2: region-restricted internal variables (operator)

The operator keeps one global internal-variable discretisation (`dfes_`, `sfes_`, `B_`, the strain map) but gives each branch an **active node set**: the L2 dofs of the elements in `RegionMarker(BranchRegion(k))`, as an `mfem::Array<int>` of scalar dof indices, and a compact state block of size `n_c × n_d,k`. Concretely:

- `Rheology` gains `virtual const mfem::Array<int>* BranchMarker(int k) const` (null = everywhere; the composite returns the region marker). Nothing else in the rheology interface changes.
- `BranchSize(k)` becomes per branch; `Offsets()` and `Branch(m, k)` follow. `SyncFields` scatters into the full-mesh output `GridFunction` (zero outside the region), so visualisation is unchanged.
- The strain map is computed once on the whole mesh (it is element-local and cheap), and each branch gathers its active nodes. Rates, exponential updates, branch moduli, relaxation times, laws and weights are all pointwise and simply loop over the active set; the nodal data (`branch_modulus_`, `itau0_`, `law_params_`) are sampled on the active set only.
- The coupled force `Bᵀ ζ_k`: rather than scattering ζ_k into a full-layout vector and applying the whole-mesh `B_ᵀ` (a full-mesh matvec per branch, whatever the region's size), assemble one coupling form per region with the region marker (`MixedBilinearForm::AddDomainIntegrator(integ, marker)`; MFEM assembles only the marked elements, so the rows outside the region are empty) and keep its rows restricted to the active set. Then both the state and the per-step work of a branch scale with its region. The strain map `d = D u` is likewise restricted per region. *Added 3 Sep 2026 after David asked whether the internal variables should live on region SubMeshes with the injection machinery: not needed, because L2 dofs are element-interior, so the submesh L2 space is a plain row selection of the parent one and the injection degenerates to that selection; the marked forms give the same locality without mesh objects. A SubMesh would only pay off for a different internal-variable order or element type per region, which is not wanted.*
- The effective-modulus weights stay full-mesh nodal `GridFunction`s (one per global branch, values outside the region left at 1), since the stiffness only reads them inside the region anyway.

Everything the time steppers do is unchanged; the change is the gather/scatter around the pointwise loops and the state layout. Results must equal Phase 1 to round-off (the test), at a state size that is the sum over regions of their own branches.

Estimated effort: one day. Deferred until a model needs it; Phase 1 is sufficient for the Love-number benchmark.

**Done, 3 Sep 2026** (same day, after the Love-number driver). As designed, with two simplifications found on the way: (i) the per-branch coupling rows are taken from the one global `B_` by row extraction rather than from a marker-assembled form per region — an L2 node's row involves its own element only, so the two are identical, and a whole-mesh branch simply aliases `B_`; (ii) the strain map stays global: it is computed once per elastic solve for all branches (one `B` multiply and the block-diagonal mass inverse), not per branch, so restricting it would save nothing. `Rheology::BranchMarker(k)` (null = everywhere; the composite returns its region marker); `ViscoelasticOperator` gains `BranchNodes(k)`, `NumBranchNodes(k)`, `BranchSize(k)`, `BranchToFull()`; the nodal material data, relaxation times and law parameters are stored per branch at its active nodes, the effective-modulus weights stay full-mesh nodal fields, and the state-dependent stress at a node sums the branches living there. A whole-mesh rheology is unchanged to the bit (its block layout coincides with the full one). Tests: the composite tests of §3 now also assert the state sizes (the split homogeneous bar has exactly the unsplit bar's state; an elastic region carries none), serial and MPI 1/2/4.

## 5. Decisions to confirm

1. **Regions by element attribute** of the displacement mesh, disjoint and covering, as above. Alternative: regions as a coefficient (an indicator function), which would allow regions not aligned with attributes; I see no need. Response: Agreed, by attribute only. 
2. **Elastic regions are spelled as regions with an elastic rheology**, not as an option of the composite. A body that is elastic everywhere then has two spellings (an elastic rheology, or a composite of elastic regions); both work. Response: yes, this is fine and the redundancy seems harmless. 
3. **Isotropic regions in a body with any anisotropic region use full internal variables** (Phase 1) rather than a per-region trace-free choice. A per-region choice would need two internal-variable layouts in one operator; the cost of full tensors is n_s/(n_s − 1), i.e. 4/3 in 2-D and 6/5 in 3-D, and only in that mixed case. Response: take the simpler option for now. we could later consider refinement it its every an issue. 
4. **Global branch numbering is by region order.** Everything a user sees (`Branch(m, k)`, `InternalVariable(k)`, output field names) uses the global index; `CompositeRheology::BranchRegion(k)` and `LocalBranch(k)` map back. Output names could be `internal_variable_<region>_<branch>`, with regions named by the user (an optional `name` in `RheologyRegion`); I would add that. Response: Okay, good idea. 
5. **Order.** Phase 1 before the Love-number benchmark (the PREM-like model has an elastic inner core and a fluid outer core, so a composite mantle + inner core is the natural first model), Phase 2 after it. Response agreeed. 

## 6. Implementation status (3 Sep 2026)

Phase 1 is implemented as designed, with the §5 decisions as answered (regions by attribute; elastic regions are regions with an elastic rheology; full internal variables whenever any region is anisotropic; global branch numbering by region order with optional region names; Phase 2 deferred until after the Love-number benchmark).

- `rheology.hpp`: `RheologyRegion {marker, rheology, name}` and `CompositeRheology(dim, regions)` with `NumRegions()`, `Region(r)`, `RegionMarker(r)`, `RegionName(r)`, `RegionOf(attribute)`, `BranchRegion(k)`, `LocalBranch(k)`, `RegionBranchOffset(r)`, `VerifyCoverage(mesh)`. Disjointness and equal marker sizes are checked at construction; coverage of the attributes present on the mesh when the stiffness is attached to a form (i.e. in the problem constructor). Region names default to `region<r>`.
- Dispatch: `RelaxationTime(k)` is the region's τ inside and `kOutsideRelaxationTime = 1e300` outside (rather than the 1 of §3, so that an internal variable outside its region neither moves nor limits an explicit step); `BranchShearModulus(k)` is the region's μ_k inside and 0 outside; `BranchModulus(k, ...)` the region's tensor inside and zero outside; `UnrelaxedModulus` and `Law(k)` dispatch by attribute.
- `ElasticStiffness::AddIntegrators(form, marker = nullptr)`: every stiffness takes an optional element marker (MFEM's marked domain integrators; the marker pointers are copied by the borrowing form constructor the problem uses for reassembly, so the composite stiffness owns copies of the region markers). A composite nested in a composite intersects its markers with the outer one.
- Output names: `Rheology::BranchLabel(k)` (default `branch<k>`; composite `<region>_<local label>`), used by `ViscoelasticOperator::RegisterFields` as `internal_variable_<label>`; a single branch with the default label keeps the plain `internal_variable`.
- No change to any problem class. `ViscoelasticOperator` changed only in the field names in Phase 1; Phase 2 (§4) then gave it region-restricted state blocks.

Tests, all serial (gtest, `TestCompositeRheology.cpp`, dims 2–3, both element types, orders 1–2) and MPI 1/2/4 (`TestCompositeRheologyPar.cpp`, the first three): (1) split of a homogeneous Maxwell bar, displacement and per-region internal variables to 1e-12 for the exponential trapezoid and backward Euler; (2) elastic + Maxwell regions against the piecewise-modulus workaround, and one-branch + two-branch regions against the piecewise two-branch body; (3) stiffness of two elastic regions against piecewise moduli, and of two Maxwell regions with relaxation weights against a piecewise weight; (4) isotropic + anisotropic regions (full internal variables) against the all-isotropic composite to 1e-10; (5) the three-layer self-gravitating body (elastic inner core, fluid outer core, Maxwell mantle) stepping through the operator with a growing displacement; plus the bookkeeping and the construction checks (overlap, coverage) as death tests.
