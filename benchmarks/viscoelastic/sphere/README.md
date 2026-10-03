# Viscoelastic: the sphere benchmarks

Non-gravitating, solid-only viscoelastic problems on spherically layered
balls (3-D) and discs (2-D) under a surface pressure `S(t) Y_l`, against
EXACT references for radial models, and against a much smaller step on
the same mesh for laterally varying viscosity. The companion of the box
sub-family (`../box`), whose histories, comparison and AAA/Talbot
machinery it reuses; the write-up with results is the sphere section of
`doc/benchmarks.tex`.

| file | what it is |
|---|---|
| `viscoelastic_sphere.cpp` | the driver: a case, a mesh, a scheme -> surface coefficients of every degree in time, with their cost |
| `meshes.py` | layered balls and discs by planetmodel, interfaces as mesh surfaces |
| `cases.py` | the models and the case writer (schema in its docstring) |
| `reference.py` | the exact references of radial models |
| `derive_euler.py` | derives (sympy) and checks the Euler matrix the reference uses |
| `study.py` | the sweeps and figures: `radial`, `lateral` |

## The problem and the reference

The body is free (rigid modes projected out) and the load has degree
`l >= 2`, so there is no net force or torque; without gravity there is
no isostasy, and a body with nothing elastic in it flows. For
`u = U(r) Y rhat + V(r) grad_1 Y` and constant moduli the static
equations are of Euler type, `r dY/dr = M Y` for `Y = (U, V, rR, rS)`,
with exponents `l - 1, l + 1, -l, -l - 2` (3-D) or `l +- 1, -l +- 1`
(2-D) whatever the moduli: regular solutions at the centre, propagators
`(r2/r1)^M` across layers, and transfer functions rational in the
Laplace variable through the correspondence principle. `derive_euler.py`
derives `M` from Navier's equations and checks `reference.py` against it.

## The mesh comes first

Every interface radius is a multiple of 0.2 and planetmodel makes it a
surface of the mesh (interface error ~1e-16). Meshes are uniform (one
size `h` everywhere), curved of order 2, built once and cached in
`<out>/meshes`. The lateral model's viscosity varies smoothly (a tanh
across the polar axis), so it has no interface for the mesh to follow.

## Running

From `<build>/benchmarks/viscoelastic/sphere` (outputs refuse to go into
the source tree):

```
./cases core_lid --dim 2 --degree 2 --h 0.1 --out cl.json
./reference cl.json --out cl_ref.json
mpiexec -np 2 ../../bin/viscoelastic_sphere -c cl.json -o 2 -dt 0.05 \
    -out cl_res.json
../box/compare cl_res.json --reference cl_ref.json

./study radial                 # profile "local": discs, coarse balls
./study lateral                # 2-D, contrast 1, 1e2, 1e4
./study all --profile server   # finer balls, the 3-D lateral study
```

Costs (local): a disc at `h = 0.1`, order 2, 2 ranks, ~0.12 s per solve;
a ball at `h = 0.25`, 4 ranks, ~0.55 s per solve.
