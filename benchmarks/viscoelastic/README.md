# Viscoelastic

Benchmarks of the viscoelastic time stepping, growing toward the full
sequence: stepper survey, analytic (correspondence-principle)
references, convergence studies, gravitating spherical references
through pyslfp in the Laplace domain, and the sl3d comparison.

| file | what it is |
|---|---|
| `survey.py` | the stepper survey (beam problem, cost to a target) |
| `laplace_reference.py` | Maxwell Love-number histories by the correspondence principle |
| `viscoelastic_love.cpp` | the 3-D driver: the same histories by finite elements |
| `compare.py` | FE histories against the reference: error tables, figures |

## The stepping survey

`survey.py` answers the practical question — which integrator is worth
using, and where — by driving `examples/viscoelastic_schemes` (a beam,
every integrator of `ViscoelasticOperator`, a cost-to-target search)
over the two axes that decide it:

- **stiffness**: the relaxation-time contrast tau_max/tau_min, standing
  in for laterally varying viscosity (eta ~ 1e18 Pa s beside 1e21 is a
  contrast of 1e3 and a tau_min near a year against 1e5-year spans).
  The explicit stability limit dt < ~2.8 tau_min binds; the implicit
  and exponential schemes never notice.
- **regime**: the load period over the relaxation time (a Deborah
  number). Fast loads control the step for every scheme; slow loads
  leave relaxation in control and reward A-stability.

Cost is counted in ELASTIC SOLVES — the unit that transfers to the
3-D self-gravitating problems, where one solve is one quasi-static
system. Figures `ve_stiffness.png` and `ve_regimes.png` land beside
the per-run JSON:

```
cd <build>/benchmarks/viscoelastic
./survey
./survey --targets 1e-2 1e-3 --contrasts 1 10 100
./survey --figures-only
```

The nonlinear (power-law) axis and elastic regions run through the
example's own `-gamma` and `-mu-inf` options; a survey axis for them
follows with the correspondence-principle references.

## Love-number histories: the Laplace reference

For a layered Maxwell body under a Heaviside load the Laplace transform
of the response is the elastic response with the shear modulus
`mu_j(s) = mu_j s tau_j / (1 + s tau_j)` in every Maxwell layer, divided
by `s`; the bulk modulus, the density and the fluid layers are
untouched. Gaver–Stehfest inversion (order 12) needs the transform at
real positive `s` only, so every evaluation is a pyslfp elastic solve of
a modified model. Supported: the uniform-layer models (`homogeneous`,
`homogeneous_lithosphere`, `two_solid`, `fluid_core`, `inner_core`).

Every solid layer has its own Maxwell time, centre outward; `inf`
keeps a layer elastic:

```
./laplace_reference homogeneous --tau 1                    # all Maxwell
./laplace_reference homogeneous_lithosphere --tau 1 inf    # elastic lid
./laplace_reference fluid_core --tau 1                     # Maxwell mantle
```

`homogeneous_lithosphere` (models.py) is the homogeneous body cut at
5700 km: elastically identical, but its ~670 km outer shell is a layer
of its own that can stay elastic.

Default output times: 13 geometric times `0.011 * 2^(5k/6) tau_min`,
k = 0..12 (0.011 to 11.3 tau). The ratio is irrational in the step
grid of any fixed-step run, which is what keeps the comparison honest
(below).

### Schema

```
{"model", "layers": [{"name", "fluid", "tau"}],   # tau: number, "inf", null (fluid)
 "lmax", "stehfest", "degree": [l...], "times": [t...],
 "elastic": {"h_load": [per degree], "l_load", "k_load", "h_tide", "l_tide", "k_tide"},
 "relaxed": {same, or nulls}, "relaxed_note",
 "poles": [{"degree", "s", "efold_time"}], "horizon",
 "histories": [{"time", "valid", "h_load": [per degree], ...}],
 "sanity": {"t0", "t0_error", "tolerance", "monotone": [...]}}
```

`viscoelastic_love` writes the same `degree`, `times`, `elastic` (its
t = 0+ solve) and `histories` entries (load numbers; the tidal ones with
`-tide`), plus the case header, the layers and taus, the scheme, and
per output time the cumulative stepping solves and seconds, and totals
under `cost`.

### Caveats: the degenerate limits

- **The all-Maxwell relaxed limit** (no elastic layer): the body
  relaxes to a fluid ball with nothing to carry the load, and pyslfp
  refuses it. The reference records the relaxed limit as unavailable.
  An elastic layer (`--tau ... inf`) gives a relaxed solve.
- **The instability horizon.** Every uniform-density compressible
  layer is convectively unstable, `N^2 = -rho g^2 / kappa < 0`; while
  its shear holds it is stable, but once the shear relaxes the
  buoyancy is unopposed. The transform then has poles at real POSITIVE
  s — growth rates of Rayleigh–Taylor-like modes — accumulating
  towards s = 0. The reference scans the real axis (coarse, then dense
  where the horizon is decided: close pairs of poles hide from a
  coarse grid) and marks every time beyond `ln2 / s_pole` invalid,
  since past it the Stehfest samples are no longer Laplace integrals of
  the history. Measured (tau = 1): homogeneous, largest pole s = 0.0199
  (degree 1; horizon 35 tau); homogeneous_lithosphere, s = 0.0051
  (degree 2; horizon 136 tau); fluid_core, s = 0.0085 (degree 4;
  horizon 82 tau). The default times stop at 11.3 tau, inside all three.
  The finite-element run carries the same growing modes; nothing in
  the comparison hides them.
- **The relaxed solve is not the t -> infinity limit.** The direct
  relaxed solve gives the Maxwell layers mu = 0, and pyslfp treats them
  as its static (neutral, Dahlen-type) fluid. A compressible Maxwell
  layer that is not neutrally stratified relaxes to its kappa-response
  instead (doc/gauged_fluid.md, the Adams–Williamson caveat) — and here
  to no equilibrium at all. With the elastic lithosphere the degree-2
  and degree-3 h' histories OVERSHOOT the static-fluid value (h'_2 at
  11.3 tau is 1.18 of the way there), and degree 1 drifts away from it.
  The reference reports the fraction covered as information, not as a
  check.
- **Degree zero with a fluid core.** Dahlen's fluid differs from the
  reference at degree zero by design (love_numbers/README.md); the
  driver leaves degree 0 out of the combined load there, as
  `love_benchmark -combined` does. Use `-lmin 2` (the default).

### Sanity checks (stored and printed)

- **t -> 0**: the inversion at `1e-6 tau_min` against the elastic
  numbers, ~1e-6..1e-5 for every model — the inversion's accuracy.
- **monotone**: h', k' (and h, k) monotone over the valid times, with
  wiggles below 10x that accuracy forgiven.
- **direction**: where a static-fluid value exists, the fraction of the
  way to it covered by the last valid time (informational, above).

## Love-number histories: the finite-element driver

`viscoelastic_love` loads a case exactly as `love_benchmark` does
(`-c case.json`, `-o`, `-deg`, `-lmax`, `-method dahlen|gauged`), gives
every solid layer the Maxwell time of `-tau` (comma separated, centre
outward, `inf` elastic; a CompositeRheology of per-layer
IsotropicMaxwellRheology / elastic regions on the case's own moduli),
switches on the unit loads of degrees `-lmin..-lmax` together at t = 0,
and evolves with `ViscoelasticOperator`:

| `-scheme` | |
|---|---|
| `sdirk23` | default: the survey's best fixed-step scheme, L-stable, 2 solves/step |
| `exptrap` | exponential trapezoid, 1 solve/step |
| `be`, `etd1` | first order (for convergence studies) |
| `adaptive` | adaptive exponential trapezoid, `-rtol`, `-atol` |

Fixed-step schemes take, within each interval between output times, the
same number of equal steps: at least `-steps` (4), more to keep
`dt <= -dt-max` (0.25 tau_min). Output times are hit exactly.

```
mpiexec -np 8 <build>/benchmarks/bin/viscoelastic_love \
    -c case.json -o 2 -lmax 4 -tau 1 -out results.json
./compare results.json --reference laplace_fluid_core_tauf_1.json
```

### The comparison, and the stroboscopic trap

`compare.py` carries the survey's HISTORY metric over: per degree and
number, the worst error over every valid output time and t = 0+,
relative to the largest value of the reference history. A metric at the
final time alone could be met by a coarse step that is wrong in
between; the survey's beam showed the related trap of output times in
phase with the step grid (or a periodic load), where an aliased
trajectory scores well stroboscopically. Here the load is a Heaviside
step, so nothing is periodic, and the defaults keep it that way: the
output times are geometric with an irrational ratio, chosen
independently of `-dt-max`, and every interval is subdivided afresh, so
the times sample the relaxation at unrelated phases of the stepping.
Two companions separate the error sources: `elastic` (t = 0+, the
spatial error alone) and `relax`, the error of `f(t) - f(0)` relative to
its range — the relaxation itself, which the time stepping answers for.
A step-convergence check is two runs at different `-steps`/`-dt-max`
compared against each other through the same reference.

### Measured (1 Oct 2026; order 2, 8 ranks, tau = 1, degrees 2..4)

`references/` holds the three reference histories (and the first
smoke run, reproduced exactly by the per-layer script). History errors
against them, SDIRK23 at the defaults (79 steps, 158 solves):

| case | h' | k' | l' | cost |
|---|---|---|---|---|
| fluid_core h0.3 | 0.5–1.1e-2 | 0.6–1.3e-2 | 1.5e-2 / 6.3e-2 / 0.46 | 87 s |
| homogeneous h0.3 | 0.6–2.3e-2 | 0.7–2.9e-2 | 0.19 / 0.40 / 1.04 | 38 s |
| homogeneous h0.2 | 0.2–1.0e-2 | 0.3–1.2e-2 | 0.08 / 0.13 / 0.30 | 85 s |
| homogeneous_lithosphere h0.3 (tau 1, inf) | 0.3–1.0e-2 | 0.4–1.6e-2 | 1.8e-2 / 5.1e-2 / 0.10 | 78 s |

The time stepping is not what limits these: on fluid_core, halving
the step moves the histories by ~2e-5 (h', k'), the adaptive
trapezoid at rtol 1e-3 agrees to 1e-5 but costs 512 solves AND 512
assemblies (310 s: every accepted step is a new effective modulus),
the exponential trapezoid at the same step to ~1.4e-4 for 79 solves.
The errors are spatial and converge with h (homogeneous h0.3 -> h0.2:
h'_2 2.2e-2 -> 1.0e-2), the relaxation part more slowly than the
elastic part; l' — small, sign-changing, tangential — is the worst
resolved and its error grows late, where the unstable buoyancy modes
of the uniform compressible layers start to matter. A Stehfest order
of 16 moves the reference by < 5e-4 there, so the reference is not
the cause. Note the h'_2 error on fluid_core passing through zero near
t = 2: a check at that one time would have scored 2e-4, the history
metric scores 1e-2.

The opt-in campaign stage runs the local version (fluid_core, h 0.3,
order 2, degrees 2..4, SDIRK23) and plots it:

```
./campaign --profile local --stages viscoelastic plot
```
