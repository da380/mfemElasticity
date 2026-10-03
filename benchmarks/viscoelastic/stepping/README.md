# Viscoelastic: the stepping survey

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
cd <build>/benchmarks/viscoelastic/stepping
./survey
./survey --targets 1e-2 1e-3 --contrasts 1 10 100
./survey --figures-only
```

The nonlinear (power-law) axis and elastic regions run through the
example's own `-gamma` and `-mu-inf` options; a survey axis for them
follows with the correspondence-principle references.
