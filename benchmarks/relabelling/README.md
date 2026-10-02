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
  `referential` meets this; **mapped `slip_broken` currently does
  not** — on fluid_core h 0.3 / o2 its spurious column is 0.63 at
  l = 1 and 6–7e-2 at l >= 2, against 3–8e-3 unmapped and for mapped
  `referential` — consistent with the identity leg below staying
  informational through the slipping interface. The non-covariant
  term has not been localised yet; treat mapped `slip_broken` numbers
  as unverified until it is (open item).
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
  Through the slipping interface the run stays
  informational (u 1.2e-2, zeta 8e-3 at A = 0.02, h = 0.3 on
  fluid_core; 4.5e-2/5.6e-2 at h = 0.2 — it does NOT decay with
  refinement, so this is not interpolation noise). The driver's probes
  now cover the broken organisation too: the sixteen assembled solver
  blocks (`ApplyBrokenBlock`) and the stored constraint kernels
  (`ApplyBrokenKernel`). Measured (A = 0.02, h = 0.3): every
  zeta-zeta block agrees to machine precision, every u-involving
  block differs at 1-2.5e-4; the kernels are exonerated (Bn 2e-16 —
  the Nanson nu = adj(F)^T n is invariant under shears of F that fix
  the face — and Pb/Kvz at 1e-7), and the base solid row and C
  couplings are welded-certified, so by elimination the non-covariant
  content sits in the slip interface forms B_Sigma / G_Sigma
  themselves: their `P_T F^{-1}` slot and the gravity A-vector are NOT
  invariant under face-fixing shears of the interpolated F, which MFEM
  evaluates from the adjacent volume element. The exact-F benchmark
  leakage is a separate, still-open mechanism: G_Sigma sits inside a
  razor-sharp degree-1 cancellation (`-gs-scale 0.5` blows up even the
  unmapped spurious to 0.46), so the mapped leakage is plausibly a
  small relative perturbation of that cancellation, hugely amplified;
  the slip rigid null-pair residuals (`love_benchmark -diag`) survive
  the map essentially unchanged, so the null-pair structure itself is
  not broken. Next instruments: a B_Sigma scale hook, a
  difference-field map of mapped-vs-unmapped solutions, or the theory
  pass on the mapped B_Sigma/G_Sigma forms.
