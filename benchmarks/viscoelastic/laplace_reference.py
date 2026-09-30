"""Gravitating viscoelastic Love-number histories through the
correspondence principle, with pyslfp as the elastic engine.

For a layered MAXWELL body under a Heaviside load, the Laplace transform
of the response is the elastic response with the s-dependent shear
modulus, divided by s:

    mu_j(s) = mu_j s tau_j / (1 + s tau_j),   tau_j = eta_j / mu_j,

with the bulk modulus and the density untouched (elastic in bulk) and
fluid layers unchanged. Gaver-Stehfest inversion needs the transform at
REAL positive s only, where mu_j(s) is real — so every evaluation is an
ordinary pyslfp elastic solve on a modified model, and the machinery
that benchmarks the elastic solvers benchmarks the viscoelastic ones
too: spherical, layered, self-gravitating, fluid cores included.

    ./laplace_reference fluid_core --tau 1.0 \
        --times 0.03 0.1 0.3 1 3 10 --lmax 4

`--tau` is the mantle's (every solid layer's) Maxwell time; times are in
the same unit; the output JSON holds, per time and degree, the load and
tidal Love numbers on their way from the elastic (t -> 0) to the
relaxed (t -> infinity) limits, with the two limits computed directly
as elastic solves for a cross-check. Restricted to models whose layers
have CONSTANT moduli (the velocities are rebuilt from the modified
moduli, which needs no refitting there).

The Stehfest order trades accuracy against cancellation; n = 12 in
double precision gives ~4-5 significant figures on smooth transforms,
plenty below the 3-D discretisation error it will referee.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))

import models  # noqa: E402
from planetmodel import LayeredIsotropicElastic  # noqa: E402

#: The uniform-layer models this reference supports, with their
#: constructor data pulled from models.py by rebuilding.
SUPPORTED = ("homogeneous", "two_solid", "fluid_core", "inner_core")


def stehfest_weights(n: int) -> list[float]:
    """The Gaver-Stehfest weights V_k, k = 1..n (n even)."""
    V = []
    for k in range(1, n + 1):
        s = 0.0
        for j in range((k + 1) // 2, min(k, n // 2) + 1):
            s += (j ** (n // 2) * math.factorial(2 * j) /
                  (math.factorial(n // 2 - j) * math.factorial(j) *
                   math.factorial(j - 1) * math.factorial(k - j) *
                   math.factorial(2 * j - k)))
        V.append((-1) ** (k + n // 2) * s)
    return V


def maxwell_model(name: str, tau: float, s: float | None):
    """The model with mu -> mu s tau / (1 + s tau) in every solid
    layer (s = None: the elastic model; s = 0 relaxed), in SI, then
    scaled with pressure attached; tau in the benchmark's time unit."""
    base = models.model(name)
    sk = base.skeleton
    rho, vp, vs = [], [], []
    for i, layer in enumerate(base.layers):
        r0 = float(sk.boundaries[i])
        f = layer.fields
        d = float(f["rho"](r0 + 1.0))
        p_wave = float(f["vp"](r0 + 1.0))
        s_wave = float(f["vs"](r0 + 1.0))
        mu = d * s_wave**2
        kappa = d * (p_wave**2 - 4.0 * s_wave**2 / 3.0)
        if s is not None and mu > 0.0:
            # tau is given in benchmark time units; s arrives in the
            # same units, so s*tau is dimensionless as required.
            mu = mu * (s * tau) / (1.0 + s * tau) if s > 0.0 else 0.0
        vs.append(math.sqrt(mu / d))
        vp.append(math.sqrt((kappa + 4.0 * mu / 3.0) / d))
        rho.append(d)
    rebuilt = LayeredIsotropicElastic(
        [float(b) for b in sk.boundaries], rho=rho, vp=vp, vs=vs,
        layer_names=[la.name for la in base.layers],
        interface_names=list(base.skeleton.interface_names)
        if hasattr(base.skeleton, "interface_names") else None,
        name=base.name)
    return models.with_pressure(models.scaled(rebuilt))


def love(model, lmax: int) -> dict:
    import make_case
    r = make_case.reference(model, lmax, per_layer=3)
    return {q: r[q] for q in ("degree", "h_load", "l_load", "k_load",
                              "h_tide", "l_tide", "k_tide")}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=SUPPORTED)
    p.add_argument("--tau", type=float, default=1.0,
                   help="Maxwell time of the solid layers, in the "
                        "benchmark time unit")
    p.add_argument("--times", type=float, nargs="+",
                   default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                   help="times after the Heaviside load, same unit")
    p.add_argument("--lmax", type=int, default=4)
    p.add_argument("--stehfest", type=int, default=12,
                   help="Stehfest order (even)")
    p.add_argument("--out", type=Path, default=None,
                   help="results file (default: "
                        "laplace_<model>_tau<tau>.json)")
    args = p.parse_args()
    MFEMV = args.stehfest
    if MFEMV % 2:
        raise SystemExit("--stehfest must be even")
    V = stehfest_weights(MFEMV)
    ln2 = math.log(2.0)

    print(f"{args.model}: elastic and relaxed limits...", flush=True)
    elastic = love(maxwell_model(args.model, args.tau, None), args.lmax)
    try:
        relaxed = love(maxwell_model(args.model, args.tau, 0.0),
                       args.lmax)
    except Exception as e:  # the fully relaxed body is inviscid
        print(f"  relaxed limit unavailable ({type(e).__name__}): the "
              "whole body relaxes to a fluid in these models")
        relaxed = {q: [None] * len(elastic[q]) for q in elastic}
        relaxed["degree"] = elastic["degree"]

    quantities = ("h_load", "l_load", "k_load", "h_tide", "l_tide",
                  "k_tide")
    histories = []
    cache: dict[float, dict] = {}
    for t in args.times:
        acc = {q: [0.0] * len(elastic["degree"]) for q in quantities}
        for k in range(1, MFEMV + 1):
            s = k * ln2 / t
            if s not in cache:
                print(f"  s = {s:.4g}", flush=True)
                cache[s] = love(maxwell_model(args.model, args.tau, s),
                                args.lmax)
            r = cache[s]
            # f(t) ~ ln2/t sum V_k F(k ln2/t), F(s) = R(s)/s for the
            # Heaviside load; the two factors of s cancel to R(s)/k*...
            for q in quantities:
                for i in range(len(acc[q])):
                    v = r[q][i]
                    if v is None:
                        continue
                    acc[q][i] += V[k - 1] * v / k
        histories.append({"time": t, **{q: acc[q] for q in quantities}})
        print(f"t = {t:g} done", flush=True)

    out = args.out or Path(f"laplace_{args.model}_tau{args.tau:g}.json")
    out.write_text(json.dumps({
        "model": args.model, "tau": args.tau, "lmax": args.lmax,
        "stehfest": MFEMV, "degree": elastic["degree"],
        "elastic": {q: elastic[q] for q in quantities},
        "relaxed": {q: relaxed[q] for q in quantities},
        "histories": histories}, indent=1))
    print(f"wrote {out}")

    # The cross-check: the history must run from the elastic limit at
    # t << tau to the relaxed one at t >> tau.
    l2 = elastic["degree"].index(2) if 2 in elastic["degree"] else None
    if l2 is not None:
        e, r0 = elastic["h_load"][l2], relaxed["h_load"][l2]
        first = histories[0]["h_load"][l2]
        last = histories[-1]["h_load"][l2]
        print(f"h'_2: elastic {e:.4f} .. relaxed "
              f"{'n/a' if r0 is None else f'{r0:.4f}'}; "
              f"t = {args.times[0]:g}: {first:.4f}, "
              f"t = {args.times[-1]:g}: {last:.4f}")


if __name__ == "__main__":
    main()
