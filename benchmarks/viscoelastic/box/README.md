# Viscoelastic: the box benchmarks

Non-gravitating, solid-only viscoelastic problems on rectangular meshes
(2-D and 3-D, `Mesh::MakeCartesian*`, periodic sideways for slabs), each
against an EXACT reference. They test the time integrators and the
viscoelastic discretisation without gravity, fluids or spherical
geometry, which the quasi-static elastic benchmarks test separately.
The full write-up, with results, is the box section of
`doc/benchmarks.tex`.

| file | what it is |
|---|---|
| `viscoelastic_box.cpp` | the driver: a case, a mesh, a scheme -> observables in time with their cost |
| `cases.py` | the models and the case writer (schema in its docstring) |
| `histories.py` | load histories S(t) as pieces, and their exact convolutions |
| `reference.py` | the exact references |
| `compare.py` | history / elastic / relax errors of one run |
| `study.py` | the sweeps and figures: `steppers`, `slab`, `stehfest`, `contrast` |

## The problems

- **Homogeneous box** (`uniaxial_stress`: pure traction; `uniaxial_strain`:
  the whole boundary prescribed). The state is homogeneous, so the finite
  elements are exact in space and every error is the integrator's.
  Rheologies `maxwell`, `sls`, `burgers` (two branches), `prony_wide`
  (four branches, tau from 1e-3 to 1e3).
- **Column** (a slab under a uniform pressure, mode 0): uniaxial strain,
  each depth a 0-D problem; layers of contrasting viscosity, or a
  viscosity varying geometrically with depth (`gradient`).
- **Fourier slab** (pressure cos(kx x) cos(ky y) on top, clamped base,
  periodic sides): the Cartesian analogue of load Love numbers, W (vertical)
  and U (horizontal) at the surface. Models with elastic lids, a
  low-viscosity channel, standard linear solids, Burgers and Prony mantles,
  and a lidless Maxwell slab that flows (no gravity, so no isostasy).

Load histories (`histories.py`): heaviside, ramp, smooth_step,
load_unload (a jump), sawtooth (glacial cycles), periodic (switched on:
an initial transient into the periodic state), multi_sine, ramped_periodic.
Breakpoints sit at irrational times; `-align` (default) puts them on the
step grid, `-no-align` lets them fall inside steps.

## The references

Every response is a finite sum of real relaxation modes,
`y(t) = R_inf S(t) + sum_i r_i (e^{p_i .} * S)(t)`, with the convolutions in
closed form. Homogeneous box and column: the modes of the small linear
system of the branch variables (a symmetric eigenproblem, 40-digit
mpmath). Fourier slab: the correspondence-principle transfer function by
propagator matrices, which is rational in s; AAA gives its poles and
residues (fit error ~1e-14), checked against fixed-Talbot inversion of
the Heaviside response (~1e-11). `reference.py --stehfest 12 16` stores
Gaver-Stehfest inversions too, for the inversion study.

Over relaxation rates spanning many decades (viscosity contrasts of 1e4
and more) AAA cannot resolve the slab's transfer function (fit errors,
spurious complex or positive poles). The reference checks itself: when
the modal form fails (fit > 1e-9, a positive pole, or Talbot disagreeing
by > 1e-7) and the load is a Heaviside, the history comes from the
Talbot inversion instead (`"method": "talbot"`); other histories are
flagged `"reliable": false`.

## Running

From `<build>/benchmarks/viscoelastic/box`:

```
./cases lid --dim 2 --mode 1 --history periodic --out lid.json
./reference lid.json --out lid_ref.json
mpiexec -np 4 ../../bin/viscoelastic_box -c lid.json -o 2 -nx 20 -nz 10 \
    -scheme exptrap -dt 0.05 -out lid_res.json
./compare lid_res.json --reference lid_ref.json --plot lid.png

./study steppers            # ~6000 runs on one element, ~15 min
./study slab                # profile "local" (default): laptop-sized
./study stehfest
./study contrast
./study all --quick
./study all --profile server   # + the long 3-D runs (p = 2 at nz = 10:
                               #   ~1.5 h each on 4 ranks), more ranks
```

The campaign runs them as its opt-in stage `viscoelastic_box`
(`./campaign --profile server --stages viscoelastic_box`).

Each study writes `<out>/<study>/summary.json` (one record per run:
errors, solves, assemblies, preconditioner setups, iterations) and its
figures; `--figures-only` redraws. Runs are cached by their options, so
delete `runs/` after changing the driver.

## Things to know

- **The mesh comes first**: every layer interface is a multiple of 0.2
  (the coarsest grid, nz = 5), so every uniform mesh with nz a multiple
  of 5 has the material discontinuities on element faces, and the driver
  refuses a mesh that cuts a layer. `-no-conform` allows it, for the
  interface study only: an interface inside elements is staircased by the
  composite rheology (one region per layer, by element centre) and its
  error stalls; `-rheology pointwise` resolves it at the points but
  converges only at first order. The error does not show at t = 0+ when
  the layers are elastically alike (an elastic lid on an unrelaxed
  Maxwell mantle).
- **The output spacing caps the step**: fixed-step schemes divide each
  interval between output times (and aligned breakpoints) into equal
  steps no longer than `-dt`.
- **At a jump** the step ending there sees the load's left limit and the
  next starts from the right limit (`ViscoelasticOperator::
  InvalidateDisplacement`); the evaluation snaps the stepped time to the
  jump time, since rounding can put it an ulp past.
