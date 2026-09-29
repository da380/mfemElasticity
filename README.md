# mfemElasticity

Extensions to the [MFEM library](https://mfem.org) for quasi-static elastic and
viscoelastic problems in geophysics, including self-gravitation. The main
pieces are

- the component ordering and node/component indexing of vector, matrix and
  symmetric tensor fields (`index.hpp`);
- mixed bilinear/linear form integrators between vector, scalar and tensor
  nodal spaces, a general (anisotropic) linear elasticity integrator and the
  transformed diffusion integrator for problems posed on a reference domain
  (`bilininteg.hpp`, `lininteg.hpp`), with general-purpose coefficients
  (`coefficient.hpp`);
- isotropic, transversely isotropic (radially anisotropic), Voigt-matrix and
  rotated elasticity tensor coefficients for that integrator
  (`elastic_tensor.hpp`);
- the exterior Poisson machinery: a matrix-free Dirichlet-to-Neumann operator
  on a spherical outer boundary and multipole right-hand-side operators
  built on the harmonics below (`poisson.hpp`, `mesh.hpp`);
- coupling of forms between a mesh and one of its `SubMesh`es through a
  boolean dof injection (`submesh.hpp`);
- a linear quasi-static problem interface with traction and clamped
  reference implementations, a generalised Maxwell rheology and a
  viscoelastic time-dependent operator (`quasi_static_problem.hpp`,
  `rheology.hpp`, `viscoelastic.hpp`);
- the self-gravitating problem: displacement on a SubMesh of the
  body coupled to the potential perturbation on the enclosing ball with the
  DtN outer condition, implementing the same interface so the viscoelastic
  layer runs on it unchanged (`self_gravitating.hpp`);
- rigid-body and general null-space projectors for singular systems
  (`null_space.hpp`);
- real orthonormal harmonics on a circle or sphere, synthesis of surface
  fields and interior harmonic potentials from coefficients, and the
  analysis of a finite-element field (scalar, or the radial or tangential
  part of a vector) on any spherical boundary into coefficients, serial and
  parallel (`spherical_harmonics.hpp`; `examples/love_numbers.cpp` reads
  load and tidal Love numbers off one solve per degree);
- the manifest that planetmodel writes beside a mesh, read into the
  attribute lists and markers the problems take, with the mesh and the
  fields of the model opened as it says (`mesh_manifest.hpp`).

Serial and parallel (MPI) paths are provided throughout.

## Documentation

The API is documented in the headers (Doxygen; build with `BUILD_DOCS`).
Method notes, which record the equations, why things are done the way they
are, and what has been learned, are in `doc/`:

| Note | Subject |
|---|---|
| `submesh_coupling.md` | forms between a mesh and its SubMesh: the dof injection, its parallel construction, constraints |
| `self_gravitation.md` | the self-gravitating problem with fluid regions: weak form, interface convention, solvers, null space, 2-D caveats, verification |
| `viscoelasticity.md` | the quasi-static problem interface, generalised Maxwell rheologies, time stepping, strain maps, state-dependent relaxation, composite rheologies |
| `elastic_tensors.md` | the Mandel convention, the elastic tensor coefficients and the anisotropic integrator |
| `null_space.md` | projected solvers for singular systems, the two gauges, element order on curved meshes |
| `mfem_notes.md` | MFEM facts and pitfalls met along the way |

`meshes/README.md` covers mesh generation, and `benchmarks/README.md` the
comparison of the self-gravitating solver with radial reference solutions:
Love numbers, radial functions and fields, for a ladder of spherically
layered models.

## Installation

MFEM must be built first (a parallel MFEM, with hypre and METIS, for the MPI
build). The project uses CMake; in-source builds are refused.

**Serial:**
```bash
cmake -S . -B build_serial \
      -DCMAKE_PREFIX_PATH=/path/to/mfem_serial_build \
      -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
cmake --build build_serial -j
```

**Parallel:**
```bash
cmake -S . -B build_parallel \
      -DUSE_MPI=ON \
      -DCMAKE_PREFIX_PATH=/path/to/mfem_parallel_build \
      -DMPI_C_COMPILER=/path/to/mpicc \
      -DMPI_CXX_COMPILER=/path/to/mpic++ \
      -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
cmake --build build_parallel -j
```

`MFEM_DIR` can be given instead of `CMAKE_PREFIX_PATH`. When `USE_MPI` is on,
the `mpiexec` next to the MPI compiler wrapper is used for the MPI tests unless
`MPIEXEC_EXECUTABLE` is set.

### Build options

All default to `OFF`.

- `USE_MPI`: build against a parallel MFEM and enable the parallel classes,
  examples and tests.
- `BUILD_EXAMPLES`: build the programs in `examples/`.
- `BUILD_TESTS`: build the googletest suite in `tests/` (googletest is fetched
  at configure time); run with `ctest` in the build directory.
- `BUILD_BENCHMARKS`: build the benchmark drivers and the launchers of
  their scripts (with `USE_MPI`; see `benchmarks/README.md`).
- `BUILD_DOCS`: generate the Doxygen API documentation.
- `GENERATE_MESHES`: generate the gmsh meshes in the build's `data/` with
  the scripts in `meshes/` (default: on when examples or tests are on; see
  Meshes below). `MESHES_PYTHON` names the Python to use.

## Examples

Examples are run from the build's `examples/` directory; they find their
meshes in `../data`, which the build fills from the source tree's `data/`
and from the scripts in `meshes/`. Each has `-h` for its options. A program
listed with `_p` has a parallel counterpart of that name; `transformed_diffusion`
and `elastogravity_layered` are one program each, parallel when the library
is built with MPI and serial otherwise.

| Program | What it does |
|---|---|
| `poisson_dtn` / `_p` | Poisson equation on the whole space: Neumann, DtN and multipole outer conditions, static and linearised, against the exact uniform-sphere solution |
| `transformed_diffusion` | Laplace equation on a mapped domain solved on the reference domain with `TransformedDiffusionIntegrator` |
| `submesh_injection` / `_p` | Tour of `SubMeshDofInjection`: moving fields and assembling coupling blocks between a mesh and a submesh |
| `coupled_poisson` / `_p` | Two Poisson equations, one on a submesh, coupled and solved monolithically |
| `elastogravity_layered` | `LinearQuasiStaticSelfGravitatingProblem` under a surface mass load or a tidal potential, Schur CG and block MINRES solvers compared, rigid-mode diagnostics (uniform solid of any shape; two-layer: fluid core + mantle; three-layer: solid inner core + fluid outer core + mantle, one disconnected solid SubMesh) |
| `self_gravitating_relaxation` | Viscoelastic relaxation of the layered self-gravitating model with a fluid core under a Heaviside surface load (Maxwell mantle, elastic inner core) |
| `quasi_static_elasticity` | Driver for the `LinearQuasiStaticProblem` interface |
| `love_numbers` | Load and tidal Love numbers of a homogeneous self-gravitating sphere (disc) by degree, one solve each, against the incompressible-sphere formulas |
| `viscoelasticity` | Generalised Maxwell viscoelasticity with `ViscoelasticOperator` |
| `viscoelastic_schemes` | Cost and accuracy table of every time integrator (ETD1, exponential trapezoid, BE, SDIRK23, RK4, adaptive) on a clamped beam, linear or power-law |
| `viscoelastic_loading` | GIA-style loading and rebound of a layered Cartesian box with a low-viscosity channel |
| `anisotropic_elasticity` | Radially anisotropic (transversely isotropic) elasticity with `ElasticTensorIntegrator` |

### Meshes

The gmsh meshes the examples and tests read are not in the repository. They
are generated into the build's `data/` directory, at build time, by the
Python scripts in `meshes/`, which use the
[planetmodel](https://pypi.org/project/planetmodel/) package to drive gmsh.
The scripts are short and meant to be copied and changed; `meshes/README.md`
lists them, the files they write and the attribute conventions. Each `.msh`
file comes with a JSON manifest beside it saying which attribute is which
layer or interface and at what radius. `meshes/aspherical_body.py` shows how
a body gets a non-spherical shape from a formula, with and without a buffer
shell, and any example accepts the result through `-m`.

Generation is on whenever examples or tests are built (`GENERATE_MESHES`,
default follows those two options). It needs a Python 3.12 or later that can
import `planetmodel.mesh3d`. CMake uses the one named by `MESHES_PYTHON` if
given, otherwise the Python it finds, and if that lacks planetmodel it creates
a virtual environment under the build directory and installs
`planetmodel[meshing,mfem]` into it (about 100 MB, once per build directory). A
fresh build spends a few minutes generating meshes, most of it on the two
large 3D ones; later builds regenerate a mesh only when its script changes.
