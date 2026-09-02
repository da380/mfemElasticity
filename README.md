# mfemElasticity

Extensions to the [MFEM library](https://mfem.org) for quasi-static elastic and
viscoelastic problems in geophysics, including self-gravitation. The main
pieces are

- mixed bilinear/linear form integrators between vector, scalar and tensor
  nodal spaces (`bilininteg.hpp`, `lininteg.hpp`);
- an anisotropic elasticity integrator with isotropic, transversely isotropic
  (radially anisotropic), Voigt-matrix and rotated tensor coefficients
  (`elastic_tensor.hpp`);
- the exterior Poisson machinery: a matrix-free Dirichlet-to-Neumann operator
  on a spherical outer boundary and multipole right-hand-side operators
  (`poisson.hpp`, `mesh.hpp`, `legendre.hpp`);
- coupling of forms between a mesh and one of its `SubMesh`es through a
  boolean dof injection (`submesh.hpp`);
- a quasi-static linear elastic problem interface with traction and clamped
  reference implementations, a generalised Maxwell rheology and a
  viscoelastic time-dependent operator (`elastic_problem.hpp`,
  `rheology.hpp`, `viscoelastic.hpp`);
- the self-gravitating elastic problem: displacement on a SubMesh of the
  body coupled to the potential perturbation on the enclosing ball with the
  DtN outer condition, implementing the same interface so the viscoelastic
  layer runs on it unchanged (`self_gravitating.hpp`);
- rigid-body and general null-space projectors for singular systems
  (`solvers.hpp`).

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
- `BUILD_GMSH`: build the gmsh-based mesh generators in `meshing/` (requires
  the gmsh C++ SDK). These are being replaced by the Python package in
  `sphmesh/`; see `doc/meshing_design.md`.
- `BUILD_DOCS`: generate the Doxygen API documentation.

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
| `elastogravity` / `_p` | Self-gravitating elastic body under a surface load: block MINRES with the DtN outer condition |
| `elastogravity_two_layer` / `_p` | The same with a liquid core and solid mantle |
| `elastogravity_three_layer` / `_p` | The same with solid inner core, liquid outer core and mantle |
| `quasi_static_elasticity` | Driver for the `QuasiStaticLinearElasticProblem` interface |
| `self_gravitating_elasticity` / `_p` | `SelfGravitatingElasticProblem`: self-gravitating body under a surface mass load, Schur CG and block MINRES solvers compared, rigid-mode diagnostics |
| `viscoelasticity` | Generalised Maxwell viscoelasticity with `ViscoelasticOperator` |
| `anisotropic_elasticity` | Radially anisotropic (transversely isotropic) elasticity with `ElasticTensorIntegrator` |

The gmsh meshes in `data/` were produced with the tools in `meshing/`; the
generating command is recorded at the top of each example that uses one.
