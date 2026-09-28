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

The drivers are parallel programs, built with the library when it is
configured with `-DUSE_MPI=ON -DBUILD_BENCHMARKS=ON`; they land in
`<build>/benchmarks`. The scripts run in the poetry environment of this
directory:

```
cd benchmarks
poetry install        # planetmodel (gmsh, PyMFEM), pyslfp, matplotlib
```

Nothing a benchmark writes belongs in the repository: give the scripts an
output directory outside it, or use the default `runs/`, which is ignored.

## Love numbers

```
cd benchmarks/love_numbers
poetry run python run.py homogeneous --h 0.3 0.2 0.14 --order 2 3 --np 8
poetry run python plot.py runs/homogeneous
```

`run.py` makes a case for each element size (the mesh, the model on it and
the reference), runs the driver on it for each order, and skips what is
already there, so a sweep can be extended later or on another machine.
`plot.py` prints the relative errors and writes the figures. On a larger
machine only the numbers change:

```
poetry run python run.py inner_core --h 0.1 0.07 0.05 --order 2 3 --np 128 \
    --lmax 10 --mpiexec /path/to/mpiexec --out /scratch/love
```

`--mpiexec` must be the launcher of the MPI the driver was built with (the
environment variable `MPIEXEC` is read when the option is absent),
`--launcher-args` passes it further arguments, and `--dry-run` prints the
commands of a sweep without running them. Each script has `--help`.

### The pieces

| file | what it does |
|---|---|
| `models.py` | the models, as planetmodel models in SI, and the units they are solved in |
| `make_case.py` | one case: mesh, fields and manifest through planetmodel, reference through pyslfp |
| `love_benchmark.cpp` | the driver: reads a case, solves each degree, writes the results as JSON |
| `run.py` | sweeps over element sizes and orders |
| `plot.py` | the table of errors and the figures |

### One model, two solvers

A model is defined once, in `models.py`, and both solvers are given that
object. planetmodel meshes its skeleton, with a buffer shell outside the
surface, and writes the density and the bulk and shear moduli as L2
GridFunctions on the mesh, so that a discontinuity at an interface stays
one; the manifest beside the mesh names the layers and interfaces, records
the units and G, and lists under `meta.fluid_layers` the attributes of the
fluid layers. The driver reads all of it through `MeshManifest`
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

The radius is the Earth's and the values are of the Earth's order, which
makes the ratio of gravitational to elastic forces, rho g a / mu, of order
one. A new model is a function in `models.py` and an entry in `MODELS`;
uniform layers are planetmodel's `LayeredIsotropicElastic`, and
`LayeredIsotropicPolynomial` in `models.py` takes polynomials in the radius.
A fluid whose density varies with radius is the test of the term
rho'_F phi of the weak form, which vanishes in a uniform one.

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

- **Degree one.** The load numbers depend on the frame, h', l' and k' by
  the same constant, so h' - k' and l' - k' are compared.
- **Degree zero with a fluid layer** is not compared. The finite-element
  problem describes a fluid by the potential alone, which says nothing of
  its compressibility; the radial solver solves degree zero in the fluid
  with its bulk modulus.
- **The mesh's asphericity.** A load of one harmonic excites others on an
  unstructured mesh; the largest of them relative to the one wanted is in
  the results as `spurious`.

The results also hold the coefficients of U, V and phi on every interface
bounding a solid layer, per unit forcing and in the model's units, and the
reference the radial solutions U, V and phi by radius.

### Resolution

The element size `--h` is that on every interface, in units of the outer
radius, capped at `--angular` times the interface's radius so that a small
inner core is still a sphere; the size grows to twice that away from the
interfaces. The geometry is of order two. A displacement of order three on
it converges markedly faster than one of order two: at h = 0.2 every model
here agrees with the reference to a few parts in ten thousand at degrees one
to five with order three, and to about a per cent with order two. A model
with an inner core takes two to three times the iterations of one without.
