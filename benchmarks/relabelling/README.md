# Relabelling

The aspherical verification (doc/mappings.md): the SAME spherical
physical problem described from laterally relabelled coordinates, so the
pyslfp reference stays exact while every mapped code path acts. The
interior relabelling (`common/relabelling.hpp`) is the pointwise
identity — with identity gradient — on every interface and on and
outside the surface, and genuinely three-dimensional in between: every
pulled-back coefficient carries full lateral variation, F is non-radial
everywhere between the interfaces, and the pulled-back elastic tensor
has no minor symmetry. Three legs:

```
cd <build>/benchmarks/love_numbers
./run fluid_core --h 0.3 --order 2 --np 8 \
    --method referential slip_broken --map 0.02
mpiexec -np 8 ../bin/relabelled_identity \
    -c runs/two_solid/h0.3/case.json -o 2 -map 0.02
mpiexec -np 8 ../bin/aspherical_reference \
    -m ../../data/aspherical_buffer_3d.mesh \
    -c runs/homogeneous/h0.3/case.json -o 2
```

- `--map A` (exact analytic F, through the Love-number family's run.py):
  the mapped solve must reproduce the same pyslfp numbers as the
  unmapped one, at the discretisation level plus the geometric floor.
  Referential methods only (`referential`, `slip_broken`; the
  single-valued slip refuses maps by design, the Eulerian classes are
  unmapped). Needs a case with `radial_profiles.txt` (re-make older
  cases); the mapped runs' `spurious` column now measures lateral
  leakage, a free check that the lateral machinery cancels exactly.
- `aspherical_reference` (exact F, independent mesh): the
  independently-meshed leg — the same spherical physics described from
  an ASPHERICAL reference body whose mesh comes from its own generator
  (`meshes/aspherical_body.py`, parametrised by `--eps/--beta/--scale/
  --name` for sweeps), with phi_e the exact inverse of the generator's
  radial stretch: the mapping is NOT the identity on the surface, the
  load carries its Nanson area factor, and the solution is compared as
  fields against the pyslfp radial solutions composed through the map
  (relative L2 of u over the body and of the Eulerian
  phi1 = zeta1 - b.u; degree one skipped). Single solid layer for now
  (the strict regime). Measured: the errors match the spherical
  campaign at comparable resolution, are INDEPENDENT of the shape
  amplitude (eps 0.05 vs 0.02 identical) and fall with refinement — the
  aspherical machinery adds nothing measurable.
  `plot.py` draws the runs' relative L2 field errors against degree,
  one series per results file, into `aspherical.png` beside them
  (non-converged degrees hollow); flat in the shape amplitude and
  falling with refinement is the verified picture:

      ./plot <dir-with-aspherical*.json>

- `relabelled_identity` (interpolated F): the discrete
  change-of-variables identity at the level of the full coupled SOLVE —
  the mapped assembly on the reference mesh against the standard
  assembly on the nodal-image mesh, entrywise. STRICT on every welded
  case, the gauged fluid included: agreement to ~1e-6 on the solids
  (u 4e-7 and 1e-6 at A = 0.02, h = 0.3) and to 1.6e-7 on the welded
  gauged fluid core, bounded only by the DtN centring on the shifted
  mesh centroid, with any unmapped assembly term showing up linearly in
  the amplitude. The gauge penalty is assembled covariantly (the
  pulled-back deviatoric tensor through `ElasticTensorIntegrator`'s
  mapped form — note it must be that integrator: the material-stiffness
  form expects a RELABELLED tensor and is not the plain pull-back).
  Through the slipping interface the run stays informational (u 1.2e-2
  at A = 0.02: the interface constraint forms are not yet certified
  covariant). The driver also prints per-attribute differences and
  operator-action probes (the same field through both sides' assembled
  operators, and the penalty alone), which localise any future
  non-covariant term to its block.
