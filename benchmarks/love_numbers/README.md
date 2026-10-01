# Love numbers

Load and tidal Love numbers of the spherically layered models, against
the radial solver of pyslfp, across formulations, solvers and CMB
treatments, with the response compared as fields as well.

```
cd <build>/benchmarks/love_numbers
./run homogeneous --h 0.3 0.2 0.14 --order 2 3 --np 8
./run all --h 0.2 --order 2 3 --np 8 --field
./plot runs
```

`run.py` makes a case for each model and element size (the mesh, the model
on it and the reference, through `common/make_case.py`), runs the drivers
on it for each order, and skips what is already there, so a sweep can be
extended later or on another machine. `plot.py` prints the relative errors
and writes the figures, for one model or for all of them, with a summary
in `runs/summary.md`. On a larger machine only the numbers change:

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

## The methods

`--method` sweeps the FORMULATIONS over the same case, one results file
(and one series in every figure) each, with the wall times and iteration
counts compared in `timing.png` and in the summary:

```
./run fluid_core --h 0.3 --order 2 --np 8 \
    --method dahlen gauged referential slip slip_broken --solver 1 0
./run fluid_core --h 0.3 --order 2 --np 8 --cmb uniform
./plot runs
```

| method | formulation |
|---|---|
| `dahlen` | Eulerian, fluid eliminated (`doc/self_gravitation.md`); `--cmb full/nomass/uniform/winkler` picks its fluid-interface treatment, `uniform` the standard unmeshed-core condition of the GIA codes |
| `gauged` | Eulerian, the fluid in the displacement space with a gauge shear penalty (`doc/gauged_fluid.md`) |
| `referential` | the welded gauged REFERENTIAL formulation (`doc/gravitating_elasticity.md`): bare moduli from the exported hydrostatic pressure `p0`, equilibrium stress `-p0 I` |
| `slip` | the slipping fluid–solid interface, single-valued potential (`doc/slip_interface.tex`): broken displacement pair, normal-jump constraint by penalty + augmented Lagrangian |
| `slip_broken` | the same with the potential broken per region as well (no fluid extension at all) |

`--solver 1 0` runs the Eulerian methods under both of their linear
solvers (block MINRES, and the Schur-complement CG with the `_schur`
suffix); the referential family has its own projected-MINRES solver and
ignores the choice. The referential methods need a case made since the
`p0` field was added (re-make older cases), solve the LOAD problems only
for now, and support one fluid core (no inner core yet: the slipping
machinery's nested-shell extensions are future work). At degree zero
with a fluid, every method EXCEPT `dahlen` is comparable with the
reference — the Dahlen treatment differs there by design, the
compressible descriptions do not.

`--map A` runs the referential methods from relabelled coordinates — the
relabelling family's benchmark, described in `../relabelling/README.md`.

`cmb_report.py` collates the CMB treatments' cost against their
accuracy — what an approximation gains in setup and solve time,
iterations and potential unknowns, against what it loses to the
reference and to the full treatment on the same mesh (the latter is the
approximation error alone, mesh bias cancelled) — into
`runs/cmb_summary.md`:

```
./run fluid_core --h 0.3 --order 2 --np 8 --cmb full nomass uniform winkler
./cmb_report runs
```

Mind the model: `uniform` assumes a uniform core, so on `fluid_core` it
matches full to rounding (4e-11 measured — a free check of the
implementation) and the treatments only separate on the stratified
cores (`stratified_core`, `earth_like`, the PREM models). And mind the
cost axis: here the approximations change the interface condition, not
the mesh — the core stays meshed and the potential unknowns stay put —
so the measured gain is solver-side (winkler runs ~15-20% fewer
iterations). A code that leaves the core unmeshed saves its elements
as well; that part of the gain is bounded by the core's share of the
element count, which the results record.

Mind the physics when refining a cross-method ladder: on a core that is
not neutrally stratified the compressible treatments (`gauged` and the
referential family) differ from the Dahlen/pyslfp secular response by
O(N²) at every degree — a genuine difference of fluid physics, not an
error (`doc/gauged_fluid.md`). `fluid_core`'s uniform core has N² < 0,
so its converged ladders split at l >= 1; the PREM cores are near
Adams–Williamson but carry inner cores the slipping methods do not
support yet. At coarse resolution (h ~ 0.3) the split sits below the
mesh error and the cross-method agreement is a real check.

## The models

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
one. A new model is a function in `common/models.py` and an entry in
`MODELS`; every one is planetmodel's `LayeredIsotropicElastic`, whose
layers take a constant or polynomial coefficients in the radius. A fluid
whose density varies with radius is the test of the term rho'_F phi of the
weak form, which vanishes in a uniform one.

The PREM models keep the named boundaries and merge what lies between
them (`Model.coarsened`), each parameter within a merged layer being the
cubic closest to PREM's, so that the model is smooth within the layers of
the mesh. Their load Love numbers are within a fifth of a per cent of
PREM's. The crust of `prem_6` is 21 km thick and its mesh has millions of
elements: a model for the larger machine.

## Units

A case is solved in the units of `models.scales`: the outer radius as the
unit of length, 5000 kg m^-3 as the unit of density, and a unit of time
which is free. By default it is 1 / sqrt(G rho), which makes G one;
`make_case --time-scale` sets another, and G then takes the value the
manifest records, which is the one the driver uses. The Love numbers
compared are dimensionless.

## What is compared

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

## Radial functions

The reference is a set of radial functions, U, V and phi by degree. Those of
a finite-element solution for the harmonic Y are, in each layer, the
polynomials in the radius that times Y (times grad_1 Y for V) are closest
to the solution there in the least-squares sense, which for a solution of
the reference's form are its radial functions. `profiles.png` draws the two
over each other, with the harmonic coefficients on the interfaces as
points. The displacement has none in a fluid layer.

## Fields

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

## Combined solves

The problem is linear, so `love_benchmark -combined` (`run.py --combined`,
results `results_o<p>[_<method>]_combined.json`) solves twice in all: once
for the sum of the unit loads Y_l0 of every degree from `lmin` to `lmax`,
and once for the sum of the tidal potentials of degree two and above,
starting cold. Every degree's numbers are analysed from the one solution,
the harmonic analysis giving all the coefficients in one pass; the
centre-of-mass correction at degree one acts on the degree-one
coefficients alone and is unchanged. The results keep their per-degree
form with `"combined": true`; the outer iterations and wall time of the two
solves are under `"combined_solves"`, and the per-degree counts and times
are null.

The caveats. The discrete operator is not exactly rotationally invariant,
so a load of degree l leaks into the responses of the other degrees: the
solves by degree discard it, the combined solve adds it to each degree's
number. On `fluid_core` at h = 0.3, order 2, the combined numbers differ
from those by degree by 2e-5 to 2e-3 relative (Dahlen) and 3e-5 to 4.5e-3
(gauged), the largest in l' of degree two whose value is small; in every
entry but one the difference is below the error against the reference
(by a factor of 40 to 90 in the median, 3 at the least: gauged load k' of
degree two), and the exception (gauged tidal h at degree
three, 2.9e-4 against an error of 2.0e-4) is where that error happens to
be small. The degrees share one residual tolerance, relative to the
combined load, so those with the smallest response are solved least
accurately in relative terms (at `-rt 1e-10` the effect is below 1e-8).
The `spurious` entry is then relative to the largest coefficient of a
harmonic that was not forced. Dahlen's fluid leaves degree zero out of
the combined load, as its degree-zero response is not comparable
anyway; the other methods keep it. The cost there: Dahlen 130 + 137
iterations (0.75 + 0.61 s) against 484 + 375 (2.1 + 1.65 s) by degree over
the same degrees, gauged 783 + 840 (4.5 + 4.3 s) against 3916 + 2413
(21.3 + 12.9 s).

## Resolution

The element size `--h` is that on every interface, in units of the outer
radius, capped at `--angular` times the interface's radius, so that a small
inner core is still a sphere, and at `--thin` times the thickness of the
layers it bounds; the size grows to twice `--h` away from the interfaces.
gmsh's Netgen optimiser runs on the tetrahedra before they are curved, which
removes the slivers that the mesher leaves (the worst element of the meshes
with a core goes from a quality of 0.02 to above 0.25). The geometry is of
order two. A displacement of order three on it converges markedly faster
than one of order two: at h = 0.2 every model here agrees with the
reference to a few parts in ten thousand at degrees one to five with order
three, and to about a per cent with order two. A model with an inner core
takes two to three times the iterations of one without.

## h-refinement

Each `--h` of a sweep is a case of its own, meshed afresh at that size:
refinement is by re-meshing, not by MFEM's uniform refinement, so a ladder
can take steps finer than a factor of two (say `--h 0.3 0.24 0.19 0.15
0.12`). `plot.py` draws the relative error of each Love number against the
element size at fixed order in `convergence.png`, and of the cap-load
fields in `field_convergence.png`, with the observed rate p of error ~ h^p
fitted over the ladder and a table of the rates by quantity and degree
printed alongside.
