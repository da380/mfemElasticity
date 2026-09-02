# Meshing tools — assessment and plan for a parameterised model workflow

*1 September 2026. Companion to `status_and_roadmap.md`. Covers everything in `meshing/` (built under `BUILD_GMSH`, off by default). Written for the two people developing the code. Status: assessment complete; §5 records David's decisions; §6 (same day) revises the plan around them and around `sphmod`, and supersedes §4 where they differ.*

## 0. Aim

The meshing directory exists to generate meshes for our geophysical problems. The scope is, and for now remains, models that are **geometrically spherical**: diffeomorphic to a union of concentric spherical annuli (discs/annuli in 2D). What we want to work towards is a coherent, small set of tools plus proper IO for *parameterised models* — so that a model description, not a recompiled C++ file, determines the mesh, and so that the solver side can recover what the mesh means (which attribute is which layer, what the length scale is) without hard-coded conventions.

## 1. Inventory of what is there

Roughly 3150 lines across 19 files. Everything links against the gmsh C++ API (`gmsh::shared` via CMake `find_package(gmsh)`).

### 1.1 `common.hpp/cpp` (~1150 lines) — the substance, in a grab-bag

The genuinely valuable pieces, all of which the plan keeps in some form:

- **`PREMModel`** — reads a PREM-style text file, extracts layer boundary radii, non-dimensionalises by `Rref`, supports dropping the outermost `ignored_layers` and adding a buffer layer. (Property columns are read into vectors but currently unused.)
- **`Topography` / `LonLatField`** — loads lon–lat XYZ grids (CRUST-1.0 crustal thickness, depth-to-Moho, etc.), builds a structured grid from scattered samples, bilinear interpolation with pole handling, pointwise `+` for composing fields.
- **The diffeomorphism machinery** — `RadialSurface` (spherical / ellipsoidal / gridded-field implementations), `RadialMapping` with `CubicBandLinearDecay` (cubic interpolation of two interface topographies within the band between them, linear decay outside within a prescribed thickness), and `PerturbAllNodes`, which radially displaces every mesh node after meshing. This is the "geometrically spherical" concept realised in code: mesh the concentric reference model, then push it to the target geometry by a radial map.
- **`TagLayersByRadius`** — sorts fragment-produced volumes and surfaces by mean radius and assigns physical groups 1..N inner→outer, with physical names `volume_i` / `surface_i`.

Clutter mixed in: a `Timer`, a `-`-separated CLI double parser, `Deg2Rad`/`Rad2Deg`, and an **empty stub `Circles` class**. Note also that `common.hpp` declares `createCircle(Circle&&, double)` while `common.cpp` defines `createCircle(Circle, double)` — the header declaration is never defined.

### 1.2 `CircularMesh.{hpp,cpp}` / `SphereMesh.{hpp,cpp}` — a second, competing strategy

These build perturbed *geometry before meshing*: a `Circle`/`Sphere` carries an optional radial perturbation `f(θ)` / `f(θ,φ)`, samples it, and constructs OCC B-spline curves (2D) or eight `addSurfaceFilling` patches per sphere (3D). So the codebase contains two incompatible approaches to non-spherical boundaries:

1. **Perturb geometry, then mesh** (`CircularMesh`/`SphereMesh`): the mesh conforms exactly to the perturbed boundary; but layer-by-layer construction, fragmenting, and tagging get harder, and each interface needs its own patched surface.
2. **Mesh spherical, then perturb nodes** (`PerturbAllNodes` + `RadialMapping`): topology of the layered model is preserved by construction, arbitrary numbers of interfaces come for free from the concentric build, high-order nodes are moved like any other node. Used by the most mature driver (`geomesh_prem_crust`). Caveat: node perturbation of an already-generated order-2/3 mesh needs an accuracy/validity check (element tangling for large topography or exaggeration; interior high-order nodes no longer optimally placed).

Also: `common.hpp` and `CircularMesh.hpp` both define classes named `Circle` and `Circles` (and `common.hpp`/`SphereMesh.hpp` both define `Sphere`) with different layouts. The two headers cannot be included together; which `Circle` a TU sees depends on include choices. This is a latent ODR landmine and by itself justifies the consolidation.

### 1.3 The ten drivers — copy-paste variations on one idea

| Driver | Dim | Geometry | Sizing | Tagging | Notes |
|---|---|---|---|---|---|
| `disk` | 2D | 1 circle, geo kernel | size callback | groups 1 | hard-coded params |
| `disk2` | 2D | 2 concentric circles | size callback | 1,2 | hard-coded params |
| `diskn` | 2D | n concentric discs, OCC fragment | size callback | 1..n | radii are a hard-coded literal list |
| `offset_disk` | 2D | 2 non-concentric circles | size callback | 1,2 | hard-coded params |
| `concentric_circles` | 2D | n concentric, OCC cut | Distance/Threshold fields | 1..n named | CLI: `-r a-b-c` |
| `ball` | 3D | 1 sphere (`SphereMesh`-style patches via `createSphere`) | size callback | 1 | |
| `offset_ball` | 3D | small ball inside larger, offset centres | size callback | 1,2 | |
| `offset_sphere` | 3D | two offset spheres, parameterised offset/angle | size callback | 1,2 | own 3rd CLI-parser style; forwards unknown args to gmsh |
| `concentric_spheres` | 3D | n concentric, OCC cut | Distance/Threshold fields | 1..n named | takes only the *first* boundary surface of each cut volume |
| `geomesh_prem_crust` | 3D | PREM radii + mean-Moho + mean-topo spheres, OCC fragment | Distance/Threshold fields | `TagLayersByRadius` | + CRUST-1.0 node perturbation via `CubicBandLinearDecay`. The prototype of what the directory wants to be. |

Every driver re-derives the same boundary-refinement law — size `small` at an interface growing linearly to `big` over a distance `fac·r`, scaled by radius ratio between interfaces — some as a `setSizeCallback`, some as gmsh `Distance`+`Threshold` background fields. Every driver hand-rolls CLI parsing (three distinct styles). All write MSH v2.2 (what our MFEM reader wants). Several pop up the FLTK GUI unless `-nopopup`.

### 1.4 Quality flags (for the eventual cleanup, not urgent)

- `concentric_spheres` tags only `surfaceEntities[0]` per cut volume — correct for the inner surface only because of gmsh tag ordering luck; `TagLayersByRadius` supersedes it.
- `diskn`'s fragment-map bookkeeping (`outDimTagsMap` index juggling) is fragile and the kind of thing the OCC `fragment` + radius-sorted tagging does robustly.
- `offset_sphere.cpp` line 1–5: the sample-run comment is un-commented continuation lines that only compile by accident of backslash-newline splicing.
- `common_test.cpp` is entirely commented out; there are no real tests.
- `status_and_roadmap.md` §2.5: meshes load into MFEM with ~10k "elements with wrong orientation (fixed)" — consistent orientation should become an explicit output requirement.

## 2. The de facto contract with the solver side

The attribute convention **is an interface**, and today it is enforced nowhere and documented nowhere:

- Volumes are physical groups 1..N inner→outer; boundary surfaces likewise (only interior interfaces + outer boundary; no names in most drivers).
- Examples then hard-code positional logic — `ex10.cpp:140` uses `bdr_attributes.Max() - 2` to find an interface, `ex12` builds markers from `Max()` — and every example must independently know `Rref` and which attribute is fluid/solid.

Nothing travels with the `.msh` file: not the attribute↔layer mapping, not the radii, not the length scale, not which topography (if any) was applied. Reproducing a mesh means recovering the exact command line. This gap — not the code duplication — is the real cost, and closing it is the heart of "IO for parameterised models".

## 3. Design principles (proposed)

1. **A model description file is the single source of truth for a mesh.** One driver reads it; no geometric or sizing parameter lives in C++ code. All ten current drivers become parameter files.
2. **The mesh ships with its meaning.** Every generated `.msh` is accompanied by a sidecar file recording: model name and description-file hash, dimension, `Rref` and unit conventions, ordered layers with `{attribute, name, [inner radius, outer radius], solid|fluid}`, ordered boundary attributes with `{attribute, name, mean radius, interface between layers i,j}`, applied topographies and exaggeration, gmsh version/options. Solver codes read the sidecar instead of `bdr_attributes.Max() - k`.
3. **One geometry backbone: concentric OCC spheres → `fragment` → `TagLayersByRadius` → mesh → `PerturbAllNodes`.** The diffeomorphism route preserves layer topology by construction and is the only one that scales in the number of interfaces. (Pending §5 Q4 on whether the perturb-geometry-first route has a surviving use.)
4. **One sizing policy, defined per interface** (`size at interface`, `far size`, `decay width`), implemented once — standardising on the `Distance`/`Threshold` background-field mechanism of `geomesh_prem_crust` unless a concrete case for the callbacks emerges.
5. **2D is the same model, one flag away.** The disc/annulus tools are the cheap testing analogue of the ball/shell tools and must not fork the code path: same description schema, same tagging convention, same sidecar.
6. **Generated meshes are validated before they are trusted**: load into MFEM, assert attribute counts and physical names against the sidecar, check surface mean radii, element orientation (zero "fixed" elements on load), and — for perturbed meshes — positivity of Jacobians at quadrature points.

## 4. Draft plan of work

Ordered so each step is independently useful; nothing here is committed until the §5 decisions are made, since Q1 (language) changes the shape of every step.

1. **Write the model-description schema and the sidecar schema** (a short spec in this document, plus one worked example reproducing `geomesh_prem_crust`'s sample run and one reproducing `concentric_spheres`). Get both of us to sign off before code.
2. **Consolidate the model layer**: `PREMModel`, `Topography`, `LonLatField`, `RadialSurface`, `RadialMapping` become a small model library with tests (interpolation values, PREM radii against known file, mapping displacement at band edges). Delete the stub `Circles`, the duplicate `Circle`/`Sphere` pair, the dead declaration.
3. **One mesher**: description file → concentric build → fragment → tag → size → mesh → perturb → orient → write `.msh` + sidecar. Subsumes `concentric_circles/spheres`, `diskn`, `disk*`, `ball`, `geomesh_prem_crust`.
4. **Benchmark geometries** (`offset_*`): either fold in as layers-with-centres in the same schema or keep as one small separate driver reading the same style of description file (§5 Q3).
5. **Validation harness** (§3.6) wired into the test suite; runs on the two worked-example descriptions.
6. **Retire the old drivers**; keep the sample command lines as description files under `meshing/models/` (or similar), so nothing reproducible is lost.
7. *(Contingent on §5 Q2)* **Solver-side reader**: a small class in the main library that reads the sidecar (and possibly the model file's material section) to hand examples their attribute markers and coefficients by name instead of by position.

## 5. Open questions (blocking; for David)

1. **Language.** Stay with the gmsh C++ API, or move the meshing tools to Python (gmsh's primary interface — far less boilerplate, trivial parameter-file handling; `PREMModel`/`Topography` re-implementations are ~100 lines of NumPy)? C++'s case: the model layer could be shared with the mfemElasticity library itself and there is no new toolchain. This decision shapes every step of §4.

Response: I think python is the right choice here. I suggest we start within this repo a little python sub-project that aims to implement all the necessary code while leaving the current stuff alone until we are done and that can all be cut. I tend to use poetry for python environments. Here we are just really writing scripts so that might be overkill, but it's harmless. poetry is installed locally, and so please set up a new project using that to work in. 

2. **Reach of the model IO.** Mesher-side only (mesh + sidecar), or should the solver side of the library eventually read the *same* model description to build coefficients (density, moduli per attribute)? The latter makes the description file the single source of truth for an entire run and argues for C++ or a language-neutral format (TOML/JSON) with readers on both sides.

Response: Ultimately, we will want to read in the planetary model which describes its geometry and physical parameters and build the mesh accordingly. In particular, those parameters including information on which regions are, say, solid, or fluid, or elastic or viscoelastic, and so do get mapped directly into the meshing. We also need facility to add suitable buffer layers for DtN or multipole things. If you look in the data dir for the project, you will see "prem.nocrust" this is a simple version of the PREM model. The details are less important than the basic format which is somewhat typical of 1D global earth models -- note that repeated radii map onto discontinuities in physical parameters. For 3D models we need further information, but in our context such models would be build on a spherically symmetrtic base through geometric mappings (potentially) and then 3D fields defined on the reference body. I have another work i progress repo in /home/da380/dev/sphmod that is working towards some of these ideas. There might be a good case for putting this all in one place, the meshing too. To plan and discuss further, I think. There are standards in the field, but they tend not to be good, so there is a chance to do better here. 

3. **Offset/eccentric geometries.** Are the two-sphere benchmark configurations (`offset_disk`, `offset_ball`, `offset_sphere`) part of the parameterised family (layers with individual centres in the schema), or one-off benchmark generators to keep separate and simple?

Response: Largely for benchmarks. But good if this could fold into the general constructs too. 

4. **Is the perturb-geometry-before-meshing route needed?** i.e., anything in `CircularMesh`/`SphereMesh` to keep — e.g., exact boundary representation for large topography where node perturbation would tangle elements — or can that pair be retired once the backbone (§3.3) is in place?

Response: I am not opposed to taking this approach, but it seemed considerably harder than deforming an existing mesh (for us and our gmsh skills -- a great library with horrible docs and API!). For most use cases the asphericity is so small on a global scale I think the post-processing method will be fine, but I'm open to thoughts. 

Secondary (non-blocking, decide during step 1): description-file format (TOML vs JSON vs YAML); sidecar as a separate file next to the `.msh` (proposed — embedding in a `.msh` `$Comments` block is too fragile) ; naming of the physical groups (`volume_i`/`surface_i` vs meaningful names from the description file — proposal: emit both, with names primary).

Response: Yes, I agree, this needs thought. But should in part fall out of the more general points above. 

---

## 6. Second round — revised plan after the §5 responses

*Same day, after reading `~/dev/sphmod` and testing it against `data/prem.nocrust`.*

### 6.1 Decisions now made

- **D1 — Python, in-repo, via poetry.** A new sub-project lives at `sphmesh/`; the C++ `meshing/` directory is left untouched until parity, then cut. Scaffold is in place (see §6.5).
- **D2 — mesh-then-perturb is the backbone** (Q4): for global-scale asphericity the post-meshing radial node map is adequate and far simpler in gmsh; `CircularMesh`/`SphereMesh` will be retired with the rest of the C++. The element-validity check (§3.6) stays as the guard on this choice.
- **D3 — offset geometries are benchmarks** (Q3): a small `benchmarks` module sharing the tagging/sizing/writing utilities, not a distortion of the layered-model schema.

### 6.2 What sphmod changes

`sphmod` (github.com/da380/sphmod, sibling checkout at `~/dev/sphmod`) already **is** the model layer that §4 step 2 proposed to consolidate out of `common.cpp` — and it is better than what would have been consolidated:

| `meshing/common.*` (C++) | `sphmod` equivalent | Notes |
|---|---|---|
| `PREMModel` (radii only; property columns read but unused) | `Skeleton` + `Field`s via `MineosDeck` / `load_deck`; exact polynomial `PREM` class | Repeated radii → discontinuity list is native (`Skeleton`); properties are first-class piecewise polynomials, not dead vectors |
| ad-hoc `ignored_layers` / `buffer_ratio` | to be expressed as explicit Skeleton edits (see 6.3) | |
| solid/fluid knowledge (absent) | derivable per layer: `vs == 0` | verified on `prem.nocrust`: 10 layers, fluid layer = outer core |
| `Topography`/`LonLatField` | nothing yet (`SphericalGRF`/`LayeredGRF` cover random 3D fields; `pyslfp` export exists as a demo) | the one real port job; long-term home probably sphmod (Q below) |
| `RadialSurface`/`RadialMapping`/`CubicBandLinearDecay`/`PerturbAllNodes` | nothing | second real port job; stays in sphmesh (it is meshing-specific) |

Checked from the new venv: `load_deck('data/prem.nocrust', columns=('rho','vp','vs','qkappa','qmu'), header_lines=3)` works as-is (the file is an isotropic 6-column deck, so it goes through the generic reader, not `MineosDeck`); gmsh 4.15.2 installs from PyPI and imports cleanly alongside it.

**Consequence:** sphmesh implements *no* model representation and *no* deck IO. Its job is exactly: `sphmod.Model` + mesh recipe → gmsh → `.msh` + sidecar.

### 6.3 Architecture: three layers, and where the boundaries sit

1. **sphmod** — the planetary model: geometric `Skeleton`, named radial `Field`s; in time, the richer 3D story (reference-body fields, geometric mappings, region metadata such as elastic/viscoelastic) and any new model *file format*. This is where the "the field's standards are poor, we can do better" ambition belongs — it is a model question, not a meshing question.
2. **sphmesh** (new) — consumes a `Model` plus a **mesh recipe**: dimension (2/3); which skeleton boundaries are honoured as mesh interfaces (a coarsening choice — e.g. merging the upper-mantle discontinuities — replacing the old `ignored_layers`); buffer layers appended outside the surface for DtN/multipole coupling (tagged as buffer, distinct in the sidecar, cf. the b/a ≈ 1.2–1.4 result in `status_and_roadmap.md` §3); per-interface sizing (size-at-interface, far size, decay width); element order; algorithms; optional radial mapping (topography per interface + decay scheme). Output: `.msh` v2.2 with radius-ordered *and named* physical groups + JSON sidecar.
3. **mfemElasticity (C++)** — consumes `.msh` + sidecar. Near-term the examples can keep positional conventions; a small sidecar reader (JSON) replaces `bdr_attributes.Max() - k` when convenient (§4 step 7 unchanged).

**Sequencing insight that revises §4 step 1:** do *not* design the grand model-description format first. With Python, the API is the interface — a ten-line script constructs the `Model` (from a deck or `PREM()`) and the recipe, and calls the mesher. The only text artifacts to stabilise now are small and boring: the **recipe** (TOML, optional — a convenience wrapper over the API for reproducibility) and the **sidecar** (JSON, machine-written). The rich planetary-model format is deferred to a joint sphmod planning session. This unblocks coding without prejudging the format question David flagged in §5 Q2.

**Region metadata, near-term:** solid/fluid inferred from `vs == 0` (overridable in the recipe); elastic/viscoelastic is not in deck files, so near-term it is a per-layer annotation in the recipe, recorded in the sidecar; long-term it moves into the sphmod model format.

**Where sphmesh finally lives** (the "one place" question): recommendation — keep sphmod free of gmsh (it is a scientific library with clean dependencies; gmsh is a 100 MB binary wheel), and keep sphmesh a separate *package* that depends on it. Whether sphmesh's repo home stays here or moves next to sphmod can be decided once it stabilises; nothing in it will import mfemElasticity, so the move is cheap either way. Starting here (per D1) keeps it next to its only current consumer and its test meshes.

### 6.4 Revised plan of work

1. ~~Scaffold~~ **done** (§6.5).
2. **Concentric core**: `geometry` + `sizing` + `sidecar` modules — `Skeleton` (+ recipe) → concentric OCC build → fragment → radius-sorted tagging with names → Distance/Threshold sizing → mesh → oriented `.msh` v2.2 + sidecar; 2D and 3D from the same code path. Pytest validation via the gmsh API (physical group counts/names, mean surface radii vs skeleton, positive Jacobians, element counts sane). Reproduces `concentric_circles`/`concentric_spheres`/`diskn`/`disk*`/`ball`.
3. **Mapping module**: port `RadialSurface`/`RadialMapping`/`CubicBandLinearDecay`/`PerturbAllNodes`; `surface` module for lon–lat grids (CRUST-1.0 reader). Reproduce the `geomesh_prem_crust` sample run (order 2, exaggeration 20) and compare layer volumes/attribute structure against the C++ output as a port check.
4. **Buffer layers + region annotations** in recipe and sidecar; check a submesh-coupling example can consume the result (ties into `submesh_coupling_design.md`).
5. **Benchmarks module**: offset two-disc/two-ball/two-sphere generators.
6. **Validation harness in CI** (pytest; gmsh headless). Optionally a C++ test that loads a generated mesh into MFEM and asserts zero orientation fixes.
7. **Retire `meshing/`** (drivers *and* library), keeping each old sample command line as a recipe file + script under `sphmesh/`.
8. **Joint sphmod session**: 3D model format, region metadata, topography's final home, and whether sphmesh migrates.

### 6.5 Scaffold as set up (1 Sep 2026)

`sphmesh/` with poetry (`package-mode` standard, `src/` layout): `pyproject.toml` (Python ≥3.12; deps numpy, gmsh ≥4.13, sphmod as a path dependency on the sibling checkout `../../sphmod` with a documented git fallback; pytest in the dev group), `README.md`, `src/sphmesh/__init__.py` (module map only, no code), `tests/`. `poetry install` run: lockfile written, venv created, smoke test above passed. Nothing committed.

### 6.6 New questions (non-blocking for step 2, but wanted before step 3)

1. **Name and location OK?** `sphmesh/` at the repo root, package `sphmesh`, path-dependency on a sibling `~/dev/sphmod` checkout (the postdoc will need that clone, or we switch to a git dependency).
2. **Formats OK?** Sidecar = JSON (machine-written, read from C++ eventually); recipe = TOML (human-written, optional next to the API).
3. **Topography's home**: lon–lat surface grids start in `sphmesh.surface`; agreed that their long-term home is sphmod (as fields on the reference body), to be settled in the joint session?
4. **Deck reading for `prem.nocrust`-style files**: worth adding a named convenience in sphmod (e.g. an isotropic-deck class alongside `MineosDeck`), or is `load_deck(..., columns=(...), header_lines=3)` fine as the documented idiom?

---

## 7. Third round — the work moves to sphmod

*Same day. David's decision on §6.3's "where sphmesh finally lives": the meshing layer goes into **sphmod itself**, with gmsh as an optional dependency (extra) so model-only users stay light. sphmod is due a break-up anyway (Love numbers, random fields exported elsewhere); the meshing subpackage joins that reorganisation. mfemElasticity then simply generates its meshes by calling sphmod.*

Consequences here:

- The consumer-side requirements are written up in **`doc/sphmod_meshing_requirements.md`** — a self-contained document to be copied into the sphmod repository and worked on there. It captures everything sphmod needs to know: the `.msh` v2.2 + JSON sidecar contract, attribute conventions, buffer layers, region metadata, the mapping machinery to port from `meshing/common.*`, the CRUST-1.0 data, the `geomesh_prem_crust` acceptance benchmark, and packaging constraints.
- The `sphmesh/` scaffold of §6.5 is **superseded** (never committed); delete it when convenient. Its dependency/layout choices carry over to the sphmod extra.
- §6.4 steps 2–5 and 8 execute in sphmod against the requirements document; steps 6–7 (C++ load test, retiring `meshing/`) remain mfemElasticity work, gated on the sphmod side reproducing the acceptance benchmark.
