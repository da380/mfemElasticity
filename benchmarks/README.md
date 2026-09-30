# Benchmarks

Comparisons of the library's solvers with independent reference solutions.
An example shows how a class is used; a benchmark says how accurate the
answer is, and at what cost.

The benchmarks are organised by family, one directory each with its own
scripts, drivers and outputs, and a README of its own; `common/` holds
what they share (the models, the case builder, the case and mapping
layers of the drivers), and `campaign.py` is the master script that runs
every family with one command. New families (viscoelastic, adjoint, ...)
are a directory and a campaign stage each.

| family | problem | reference |
|---|---|---|
| `love_numbers/` | load and tidal Love numbers of spherically layered, self-gravitating elastic bodies, across formulations and solvers, with the response as fields | the radial solver of [pyslfp](https://github.com/da380/pyslfp) |
| `relabelling/` | the mapped (aspherical) machinery: the same spherical physics from relabelled coordinates, the discrete change-of-variables identity, an independently meshed aspherical reference body | the same, exact under the relabelling |
| `perturbation/` | derivatives of the response with respect to the model: the degree-0 interface-shift check | pyslfp on the perturbed models |

## Setting up

The scripts run in the poetry environment of this directory, which is made
once:

```
cd benchmarks
poetry install        # planetmodel (gmsh, PyMFEM), pyslfp, matplotlib
```

The drivers are parallel programs, built with the library when it is
configured with `-DUSE_MPI=ON -DBUILD_BENCHMARKS=ON`, into
`<build>/benchmarks/bin/`. A benchmark is run from the build tree, like
the examples: in `<build>/benchmarks/<family>/` the build puts launchers
of that family's scripts (`run`, `plot`, `make_case`,
`perturbation_check`), which start them with the Python environment
found at configuration (`BENCHMARKS_PYTHON` names another), the drivers
and the MPI launcher of that build. Nothing is written to the source
tree.

What a family writes goes where it is started — `runs/` beside its
launchers for the sweeps, one directory per model holding its cases,
results and figures; `--out` puts the results elsewhere. A build script
that clears the build directory clears the outputs with it.

## The master script

`<build>/benchmarks/campaign` runs the whole campaign — every family,
every stage — behind two profiles whose every default is a flag:

```
cd <build>/benchmarks
./campaign --profile local                    # rehearsal on this machine
./campaign --profile server --out /scratch/love
./campaign --profile server --stages methods field --lmax 20
./campaign --profile server --stages scaling --dry-run
```

- `--profile local`: small meshes, few degrees, 8 ranks; runs end to end
  in around an hour and exercises every stage — the shape to validate
  before a server run.
- `--profile server`: the full model set, an h-ladder at orders 2 and 3,
  degrees to 16, 100 ranks (sized for a 128-core shared-memory machine),
  the field benchmark, and a weak-scaling stage whose rungs hold the
  unknowns per rank roughly constant (h ~ np^(-1/3)); consider
  `--partition` so no rank reads a whole fine mesh.

The stages are `methods`, `cmb`, `field`, `mapped`, `identity`,
`aspherical`, `perturbation`, `scaling` and `plot`; `--stages` picks a
subset, and everything delegates to the family scripts, which skip work
whose results exist, so an interrupted campaign resumes where it
stopped. The tree mirrors the families
(`<out>/love_numbers`, `<out>/relabelling`, `<out>/perturbation`,
`<out>/scaling`), each family's figures land in its subtree (every
family has a `plot` of its own, which the campaign's plot stage runs),
and every command and stage outcome goes to `<out>/campaign_log.md`,
with the weak-scaling timings collated in `<out>/scaling_summary.md`
and the CMB cost-accuracy tables in
`<out>/love_numbers/cmb_summary.md`.

## The shared pieces (`common/`)

| file | what it does |
|---|---|
| `models.py` | the models, as planetmodel models in SI, and the units they are solved in |
| `make_case.py` | one case: mesh, fields and manifest through planetmodel, reference through pyslfp |
| `benchmark_case.hpp` | what the drivers share: a case set up as a problem, the analysis of its solution |
| `relabelling.hpp` | the mapped layer of the drivers: relabellings, interface shifts, exact radial profiles |
| `reference_field.hpp` | the reference solution as fields on the mesh |
| `drivers.py` | where a build's drivers are, and how the scripts run and log commands |

A model is defined once, in `models.py`, and both solvers are given that
object. planetmodel meshes its skeleton, with a buffer shell outside the
surface, and writes the density and the bulk and shear moduli as L2
GridFunctions on the mesh, so that a discontinuity at an interface stays
one; the manifest beside the mesh names the layers and interfaces, records
the units and G, says which layers are fluid, and holds the model's exact
one-sided field values on each interface. The driver reads all of it
through `MeshManifest` (`mesh_manifest.hpp`) and sets nothing about the
model itself. pyslfp solves the same model on its own radial mesh.
