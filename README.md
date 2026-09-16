# mfemElasticity

Extensions to the [MFEM library](https://mfem.org) for quasi-static elastic and
viscoelastic problems in geophysics, including self-gravitation. The main
pieces are

- mixed bilinear/linear form integrators between vector, scalar and tensor
  nodal spaces, and a general (anisotropic) linear elasticity integrator
  (`bilininteg.hpp`, `lininteg.hpp`);
- isotropic, transversely isotropic (radially anisotropic), Voigt-matrix and
  rotated elasticity tensor coefficients for that integrator
  (`elastic_tensor.hpp`);
- the exterior Poisson machinery: a matrix-free Dirichlet-to-Neumann operator
  on a spherical outer boundary and multipole right-hand-side operators
  (`poisson.hpp`, `mesh.hpp`, `legendre.hpp`);
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
  (`solvers.hpp`);
- real orthonormal harmonics on a circle or sphere, synthesis of surface
  fields and interior harmonic potentials from coefficients, and the
  analysis of a finite-element field (scalar, or the radial component of a
  vector) on any spherical boundary into coefficients, serial and parallel
  (`spherical_harmonics.hpp`; `examples/love_numbers.cpp` reads load and
  tidal Love numbers off one solve per degree).

Serial and parallel (MPI) paths are provided throughout. Design notes and the
roadmap are in `doc/`.

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
- `BUILD_DOCS`: generate the Doxygen API documentation.
- `GENERATE_MESHES`: generate the gmsh meshes in the build's `data/` with
  the scripts in `meshes/` (default: on when examples or tests are on; see
  Meshes below). `MESHES_PYTHON` names the Python to use.

## Examples

Examples are run from the build's `examples/` directory; they find their
meshes in `../data`, which is copied from the source tree at build time. Each
has `-h` for its options.

| Program | What it does |
|---|---|
| `poisson_dtn` / `_p` | Poisson equation on the whole space: Neumann, DtN and multipole outer conditions, static and linearised, against the exact uniform-sphere solution |
| `transformed_diffusion` / `_p` | Laplace equation on a mapped domain solved on the reference domain with `TransformedDiffusionIntegrator` |
| `submesh_injection` / `_p` | Tour of `SubMeshDofInjection`: moving fields and assembling coupling blocks between a mesh and a submesh |
| `coupled_poisson` / `_p` | Two Poisson equations, one on a submesh, coupled and solved monolithically |
| `elastogravity_layered` / `_p` | `LinearQuasiStaticSelfGravitatingProblem` with a fluid outer core (two-layer: fluid core + mantle; three-layer: solid inner core + fluid outer core + mantle, one disconnected solid SubMesh) |
| `self_gravitating_relaxation` | Viscoelastic relaxation of the layered self-gravitating model with a fluid core under a Heaviside surface load (Maxwell mantle, elastic inner core) |
| `quasi_static_elasticity` | Driver for the `LinearQuasiStaticProblem` interface |
| `self_gravitating_elasticity` / `_p` | `LinearQuasiStaticSelfGravitatingProblem`: self-gravitating body under a surface mass load, Schur CG and block MINRES solvers compared, rigid-mode diagnostics |
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
