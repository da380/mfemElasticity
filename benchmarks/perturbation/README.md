# Perturbation

Derivatives of the response with respect to the model, checked against
the radial solver on perturbed models. The family grows with the
adjoint work (the l >= 1 topography kernels of the boundary
perturbation theory); its first member needs no new theory:

## The degree-0 interface shift

One fixed spherical mesh; a family of PHYSICAL models with one interior
interface at r_k + eps, described from it by the piecewise-linear
radial map of `common/relabelling.hpp`
(`love_benchmark -map-shift`). Every member of the family is still
spherical, so pyslfp solves it exactly at every finite eps; and because
the mesh is fixed, the discretisation bias largely cancels in the
finite difference, so

    [R_3D(+eps) - R_3D(-eps)] / 2 eps   vs   the same of pyslfp

compares the DERIVATIVE of the response with respect to the interface
radius — the first perturbation-theory verification of the mapped
machinery.

```
cd <build>/benchmarks/perturbation
./perturbation_check ../love_numbers/runs/fluid_core/h0.3 \
    --eps 0.01 0.02 --method referential slip_broken --np 8
```

For each eps of the ladder the script builds the perturbed model's
exact radial profiles and pyslfp reference (under `--out`, default
beside the case), runs `love_benchmark -map-shift eps -profiles ...`
for each method, and prints, per degree and Love number: the absolute
agreement at each eps and the derivative comparison, with the
eps-ladder showing the O(eps^2) remainder. The models with a shiftable
interface are named in `models.SHIFTABLE` (`fluid_core` moves its CMB —
through the slipping machinery too — and `two_solid` its
mid-mantle discontinuity).

`plot.py` draws what the tables say, one figure per method
(`perturbation_<method>_o<p>.png` beside the results): the absolute
agreement with pyslfp by degree, one line per eps of the ladder, and
the derivative check as an identity plot — the 3-D central difference
against the 1-D one, one point per degree and Love number, agreement
meaning the point sits on the line:

```
./plot ../love_numbers/runs/fluid_core/h0.3 --out <root> \
    --method referential slip_broken
```

**Known issue (parked, 30 Sep 2026): the slipping methods and outward
shifts.** With the interface moved OUTWARD (`eps > 0`) the slip_broken
solve leaves the augmented-Lagrangian constraint system near singular:
at theta = 100 the solver caps out polluted, and stiffening to
theta = 1e3 with 16 AL sweeps amplifies the failure by orders of
magnitude instead of curing it — so it is a formulation question at
the moving slipping boundary, not a solver-tuning one. The inward leg
(`eps < 0`), the unmapped runs and the fixed-interface mapped runs
(`--map`) are all healthy, and the welded referential method handles
both shift signs cleanly, so the campaign's shift legs run referential
only. The question matters ahead of slipping-interface TOPOGRAPHY
(where the boundary moves outward over half the sphere) and is to be
worked through on paper before touching the slip machinery.

The perturbed model's mass differs from the base model's at O(eps)
(dM/dr_k = 4 pi r_k^2 [rho]), so the driver takes the mapped model's
mass, surface gravity and interface gravities from the exact radial
profiles (`RadialProfiles::EnclosedMass`), not from mesh integrals of
the base fields; the normalisations of the Love numbers depend on it.
