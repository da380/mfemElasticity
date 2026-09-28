"""Build one case of the Love-number benchmark: the mesh, the model on it
and the reference solution.

A case is a directory holding

  case.mesh          the MFEM mesh of the body and its buffer shell
  case.json          the manifest: layers, interfaces, fields, scales, G,
                     and under `meta` the attributes of the fluid layers
  case.<name>.gf     rho, kappa and mu as L2 GridFunctions on the mesh
  reference.json     the Love numbers and radial solutions of pyslfp

all from one planetmodel model in the benchmark's units (see models.py),
so that the finite-element solver and the radial solver are given the
same body. The mesh has the element size `--h` on every interface, or
`--angular` times the interface's radius where that is smaller, growing to
`--h-max` over the distance `--decay`, in units of the outer radius.

    python make_case.py homogeneous --h 0.15 --out cases/homogeneous_h0.15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from planetmodel import Model, is_fluid
from planetmodel.mesh3d import (InterfaceSizing, MeshSpec, Shell,
                                build_layered_mesh, export_mfem)
from pyslfp.love_numbers import LoveNumbers, love_numbers, solve_degree

import models

#: The fields the finite-element solver reads.
FIELDS = ("rho", "kappa", "mu")

#: The basename of the files of a case.
BASENAME = "case"

#: The forcings whose radial solutions are kept, for plotting.
PROFILE_FORCINGS = ("load", "tide")


def sizing(h: float, h_max: float, decay: float, angular: float):
    """The sizing rule of a case: the size `h` on every interface, but no
    more than `angular` times the interface's radius, so that a small
    sphere is still resolved in angle; `h_max` far from the interfaces."""
    def rule(interfaces, outer_radius):
        return {face.index: InterfaceSizing(min(h, angular * face.radius),
                                            h_max, decay)
                for face in interfaces}
    return rule


def fluid_attributes(model: Model) -> list[int]:
    """The element attributes of the model's fluid layers."""
    return [i + 1 for i, layer in enumerate(model.layers) if is_fluid(layer)]


def build_mesh(model: Model, out: Path, *, h: float, h_max: float, decay: float,
               angular: float, buffer: float, order: int,
               verbose: bool) -> dict:
    """The mesh, the fields and the manifest; returns a summary."""
    spec = MeshSpec(model.geometry, sizing(h, h_max, decay, angular),
                    dimension=3, order=order,
                    shells=[Shell(ratio=buffer, name="buffer")],
                    meta={"model": model.name,
                          "fluid_layers": fluid_attributes(model)})
    scratch = out / "gmsh"
    scratch.mkdir(parents=True, exist_ok=True)
    built = build_layered_mesh(spec, scratch / BASENAME, verbose=verbose)
    export = export_mfem(built, out / BASENAME, model=model, fields=FIELDS)
    return {"h": h, "h_max": h_max, "decay": decay, "angular": angular,
            "buffer": buffer,
            "order": order, "counts": dict(export.counts),
            "validation": str(built.validation)}


def _clean(a: np.ndarray) -> list:
    """An array as a list for JSON, NaN as null."""
    return [None if not np.isfinite(x) else float(x)
            for x in np.asarray(a, dtype=float)]


def profiles(model: Model, lmax: int, *, per_layer: int) -> dict:
    """U, V and phi by radius for each degree and forcing, sampled within
    each layer; a radius on a boundary appears once for each side."""
    b = np.asarray(model.skeleton.boundaries, dtype=float)
    nudge = 1e-9 * b[-1]
    radii, layer = [], []
    for i, (lo, hi) in enumerate(zip(b[:-1], b[1:])):
        radii.append(np.linspace(lo + nudge, hi - nudge, per_layer))
        layer.append(np.full(per_layer, i + 1))
    radii, layer = np.concatenate(radii), np.concatenate(layer)
    out = {"radius": _clean(radii), "layer": [int(i) for i in layer],
           "solutions": []}
    for l in range(lmax + 1):
        for forcing in PROFILE_FORCINGS:
            if forcing == "tide" and l < 2:
                continue
            U, V, phi = solve_degree(model, l, forcing=forcing).evaluate(radii)
            out["solutions"].append({"degree": l, "forcing": forcing,
                                     "U": _clean(U), "V": _clean(V),
                                     "phi": _clean(phi)})
    return out


def reference(model: Model, lmax: int, *, per_layer: int) -> dict:
    """What pyslfp gives for the model, in the model's units."""
    love: LoveNumbers = love_numbers(model, lmax)
    load, tide = love.conventional(), love.tidal()
    s = model.scales
    return {
        "model": model.name,
        "scales": {"length": s.length, "mass": s.mass, "time": s.time},
        "G": float(love.G),
        "radius": float(love.radius),
        "surface_gravity": float(love.surface_gravity),
        "degree": [int(l) for l in love.degree],
        # dimensionless: load numbers h', l', k' and tidal numbers h, l, k
        "h_load": _clean(load["h"]), "l_load": _clean(load["l"]),
        "k_load": _clean(load["k"]),
        "h_tide": _clean(tide["h"]), "l_tide": _clean(tide["l"]),
        "k_tide": _clean(tide["k"]),
        # per unit surface density or potential, in the model's units
        "generalised": {name: _clean(getattr(love, name))
                        for name in ("h_u", "l_u", "k_u", "h_phi", "l_phi",
                                     "k_phi", "h_t", "l_t", "k_t")},
        "reciprocity_residual": _clean(love.reciprocity_residual()),
        "profiles": profiles(model, lmax, per_layer=per_layer),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=sorted(models.MODELS))
    p.add_argument("--out", type=Path, required=True, metavar="DIR",
                   help="the case directory")
    p.add_argument("--h", type=float, default=0.15,
                   help="element size on every interface")
    p.add_argument("--h-max", type=float, default=None,
                   help="element size far from the interfaces (default 2 h)")
    p.add_argument("--decay", type=float, default=None,
                   help="distance over which the size grows (default 10 h)")
    p.add_argument("--angular", type=float, default=0.4,
                   help="largest element size on an interface over its radius")
    p.add_argument("--buffer", type=float, default=0.2,
                   help="thickness of the buffer shell over the radius")
    p.add_argument("--order", type=int, default=2, help="geometry order")
    p.add_argument("--lmax", type=int, default=10,
                   help="highest degree of the reference")
    p.add_argument("--time-scale", type=float, default=None,
                   help="time scale in seconds (default: G equal to one)")
    p.add_argument("--profile-points", type=int, default=41,
                   help="radii per layer in the reference's radial solutions")
    p.add_argument("--reference-only", action="store_true",
                   help="write reference.json alone, leaving the mesh")
    p.add_argument("--verbose", action="store_true",
                   help="let gmsh print as it works")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = models.scaled(models.model(args.model), time_scale=args.time_scale)

    ref = reference(model, args.lmax, per_layer=args.profile_points)
    (args.out / "reference.json").write_text(json.dumps(ref, indent=1))
    print(f"reference.json: degrees 0..{args.lmax}, G = {ref['G']:.6g}, "
          f"g = {ref['surface_gravity']:.6g}")
    if args.reference_only:
        return

    h_max = 2.0 * args.h if args.h_max is None else args.h_max
    decay = 10.0 * args.h if args.decay is None else args.decay
    summary = build_mesh(model, args.out, h=args.h, h_max=h_max, decay=decay,
                         angular=args.angular, buffer=args.buffer, order=args.order,
                         verbose=args.verbose)
    (args.out / "mesh_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{BASENAME}.mesh: {summary['counts']}; {summary['validation']}")


if __name__ == "__main__":
    main()
