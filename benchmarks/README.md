# Benchmarks

Comparisons of the library's solvers with independent reference solutions.
An example shows how a class is used; a benchmark says how accurate the
answer is, and at what cost. Each benchmark is a directory holding the
Python scripts that define its cases, make its meshes and references, launch
its runs and plot the results, and the C++ driver the runs execute.

| directory | problem | reference |
|---|---|---|
| `love_numbers/` | load and tidal Love numbers of spherically layered, self-gravitating elastic bodies | the radial solver of [pyslfp](https://github.com/da380/pyslfp) |

## Setting up

The scripts run in the poetry environment of this directory, which is made
once:

```
cd benchmarks
poetry install        # planetmodel (gmsh, PyMFEM), pyslfp, matplotlib
```

The drivers are parallel programs, built with the library when it is
configured with `-DUSE_MPI=ON -DBUILD_BENCHMARKS=ON`. A benchmark is run
from the build tree, like the examples: beside the drivers, in
`<build>/benchmarks/<benchmark>/`, the build puts launchers `run`, `plot`
and `make_case` of the scripts here, which start them with the Python
environment found at configuration (`BENCHMARKS_PYTHON` names another), the
drivers and the MPI launcher of that build. Nothing is written to the
source tree.

What a benchmark writes goes under `runs/` where it is started: one
directory per model, holding its cases, the results of every run on them,
and the tables and figures made from those. A build script that clears the
build directory clears `runs/` with it; `--out` puts the results
elsewhere.

## Love numbers

```
cd <build>/benchmarks/love_numbers
./run homogeneous --h 0.3 0.2 0.14 --order 2 3 --np 8
./run all --h 0.2 --order 2 3 --np 8 --field
./plot runs
```

`run.py` makes a case for each model and element size (the mesh, the model
on it and the reference), runs the drivers on it for each order, and skips
what is already there, so a sweep can be extended later or on another
machine. `plot.py` prints the relative errors and writes the figures, for
one model or for all of them, with a summary in `runs/summary.md`. On a
larger machine only the numbers change:

```
./run earth_like prem_4 --h 0.1 0.07 0.05 --order 2 3 --np 128 --lmax 10 \
    --field --partition --out /scratch/love
./plot /scratch/love
```

With `--partition` each case is cut into the parts of the ranks beforehand
(`partition_case`, a serial program), and each rank reads its own; without
it every rank reads the whole mesh, which is simpler and costs each the
memory of the whole.

`--launcher-args` passes the MPI launcher further arguments (binding, host
files), `--mpiexec` names another launcher than the build's, and `--dry-run`
prints the commands of a sweep without running them. Each script has
`--help`.

### The pieces

| file | what it does |
|---|---|
| `models.py` | the models, as planetmodel models in SI, and the units they are solved in |
| `make_case.py` | one case: mesh, fields and manifest through planetmodel, reference through pyslfp |
| `love_benchmark.cpp` | Love numbers: solves each degree, writes the numbers and the radial functions of the solutions |
| `field_benchmark.cpp` | fields: solves for a cap load of many degrees and orders, compares u and phi with the reference over the mesh |
| `partition_case.cpp` | cuts a case into the parts of a number of ranks |
| `benchmark_case.hpp` | what the drivers share: a case set up as a problem, the analysis of its solution |
| `reference_field.hpp` | the reference solution as fields on the mesh |
| `run.py` | sweeps over models, element sizes and orders |
| `plot.py` | the tables of errors and the figures |

### One model, two solvers

A model is defined once, in `models.py`, and both solvers are given that
object. planetmodel meshes its skeleton, with a buffer shell outside the
surface, and writes the density and the bulk and shear moduli as L2
GridFunctions on the mesh, so that a discontinuity at an interface stays
one; the manifest beside the mesh names the layers and interfaces, records
the units and G, says which layers are fluid, and holds the model's exact
one-sided field values on each interface. The driver reads all of it
through `MeshManifest`
(`mesh_manifest.hpp`) and sets nothing about the model itself. pyslfp
solves the same model on its own radial mesh.

### The models

| name | layers |
|---|---|
| `homogeneous` | one uniform solid |
| `two_solid` | two uniform solids, every parameter jumping between them |
| `fluid_core` | a uniform fluid core under a uniform solid mantle |
| `inner_core` | a solid inner core, a fluid outer core and a mantle, each uniform |
| `linear_solid` | one solid, density and velocities linear in radius |
| `stratified_core` | a fluid core whose density falls with radius, under a mantle, every parameter linear in radius |
| `earth_like` | inner core, fluid outer core and mantle, every parameter linear in radius within each |
| `prem_4` | PREM without its ocean, isotropic, on four layers: inner core, outer core, lower mantle, upper mantle with the crust |
| `prem_6` | the same on six: the transition zone and the crust are layers of their own |

The radius is the Earth's and the values are of the Earth's order, which
makes the ratio of gravitational to elastic forces, rho g a / mu, of order
one. A new model is a function in `models.py` and an entry in `MODELS`;
every one is planetmodel's `LayeredIsotropicElastic`, whose layers take a
constant or polynomial coefficients in the radius. A fluid whose density
varies with radius is the test of the term rho'_F phi of the weak form,
which vanishes in a uniform one.

The PREM models keep the named boundaries and merge what lies between
them (`Model.coarsened`), each parameter within a merged layer being the
cubic closest to PREM's, so that the model is smooth within the layers of
the mesh. Their
load Love numbers are within a fifth of a per cent of PREM's. The crust of
`prem_6` is 21 km thick and its mesh has millions of elements: a model for
the larger machine.

### Units

A case is solved in the units of `models.scales`: the outer radius as the
unit of length, 5000 kg m^-3 as the unit of density, and a unit of time
which is free. By default it is 1 / sqrt(G rho), which makes G one;
`make_case.py --time-scale` sets another, and G then takes the value the
manifest records, which is the one the driver uses. The Love numbers
compared are dimensionless.

### What is compared

With u_l, v_l and phi_l the coefficients on the surface of the displacement
u = U Y r^ + V grad_1 Y and of the potential perturbation, for the load
sigma = Y_l0 or the tidal potential psi = (r/a)^l Y_l0:

```
load:   h'_l = -g u_l / phi_sigma      l'_l = -g v_l / phi_sigma
        k'_l = phi_l / phi_sigma - 1
tidal:  h_l  = -g u_l      l_l = -g v_l      k_l  = phi_l
```

where phi_sigma is the load's own potential on the surface, solved on the
same mesh for the load alone: without the displacement and without the
fluid's term rho'_F phi, which are the body's response.

- **Degree one.** The solution is known up to a rigid translation, which
  the solver and the reference fix differently. The solver's is taken to
  the reference's frame, that of the centre of mass of the body and its
  load, in which the potential outside has no part of degree one and
  k'_1 = -1: the translation d = phi_1(a) / g(a) is added to U and V
  everywhere and g(r) d taken from phi.
- **Degree zero with a fluid layer** is not compared. The finite-element
  problem describes a fluid by the potential alone, which says nothing of
  its compressibility; the radial solver solves degree zero in the fluid
  with its bulk modulus.
- **The mesh's asphericity.** A load of one harmonic excites others on an
  unstructured mesh; the largest of them relative to the one wanted is in
  the results as `spurious`.

### Radial functions

The reference is a set of radial functions, U, V and phi by degree. Those of
a finite-element solution for the harmonic Y are, in each layer, the
polynomials in the radius that times Y (times grad_1 Y for V) are closest
to the solution there in the least-squares sense, which for a solution of
the reference's form are its radial functions. `profiles.png` draws the two
over each other, with the harmonic coefficients on the interfaces as
points. The displacement has none in a fluid layer.

### Fields

`field_benchmark` applies a smooth cap of load centred off the axes, which
excites every order, summed over the degrees from `lmin` to `lmax`. The
cap's harmonic coefficients are computed on the surface of the mesh, the
load applied is their sum, and the reference is the same coefficients times
its radial functions: both are the response to the same load exactly. The
errors are relative L2 norms, of u over the solid and of phi over the body
and its buffer shell, where the reference continues harmonically. The
coefficients on the surface give the maps `field_<run>.png` and the error
by degree `field_spectrum.png`; `--paraview` writes the fields, the
reference and their difference.

What the fields show that the numbers on the surface do not: with a solid
inner core and a displacement of order two, the translation of the inner
core under a load of degree one is wrong by more than its own size on a
mesh whose Love numbers are good to a per cent. The restoring force of that
translation is weak (the Slichter mode), so a small error in the forces is
a large one in the displacement. Order three has it right.

### Resolution

The element size `--h` is that on every interface, in units of the outer
radius, capped at `--angular` times the interface's radius, so that a small
inner core is still a sphere, and at `--thin` times the thickness of the
layers it bounds; the size grows to twice `--h` away from the interfaces.
gmsh's Netgen optimiser runs on the tetrahedra before they are curved, which
removes the slivers that the mesher leaves (the worst element of the meshes
with a core goes from a quality of 0.02 to above 0.25). The geometry is of order two. A displacement of order three on
it converges markedly faster than one of order two: at h = 0.2 every model
here agrees with the reference to a few parts in ten thousand at degrees one
to five with order three, and to about a per cent with order two. A model
with an inner core takes two to three times the iterations of one without.

### h-refinement

Each `--h` of a sweep is a case of its own, meshed afresh at that size:
refinement is by re-meshing, not by MFEM's uniform refinement, so a ladder
can take steps finer than a factor of two (say `--h 0.3 0.24 0.19 0.15
0.12`). `plot.py` draws the relative error of each Love number against the
element size at fixed order in `convergence.png`, and of the cap-load
fields in `field_convergence.png`, with the observed rate p of error ~ h^p
fitted over the ladder and a table of the rates by quantity and degree
printed alongside.
