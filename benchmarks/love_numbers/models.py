"""The spherically symmetric models of the Love-number benchmark.

A ladder of models of increasing complexity on an Earth-sized body, each
a planetmodel `Model` in SI:

  homogeneous      one solid layer of constant density and moduli
  two_solid        two solid layers with a jump in every parameter
  fluid_core       a uniform fluid core under a uniform solid mantle
  inner_core       a solid inner core, a fluid outer core and a mantle,
                   each uniform

The values are round numbers of the Earth's order, so that the ratio
rho g a / mu of gravitational to elastic forces is of order one, as in
the Earth. `model(name)` returns the model in SI and `scaled(model)`
the same model in the benchmark's units, in which the mesh is built and
the finite-element problem solved: lengths in units of the outer radius,
densities in units of `DENSITY_SCALE`, and times in units of
`time_scale`, by default the one that makes G equal to one.
"""
from __future__ import annotations

import math
from collections.abc import Callable

from planetmodel import LayeredIsotropicElastic, Model, kappa_mu
from planetmodel.units import G_SI, Scales

#: The outer radius of every model, in metres.
RADIUS = 6371.0e3

#: The density scale of the benchmark's units, in kg m^-3.
DENSITY_SCALE = 5000.0

#: Radii of the internal boundaries, in metres.
ICB = 1200.0e3
CMB = 3500.0e3
MID_MANTLE = 5000.0e3


def homogeneous() -> Model:
    return LayeredIsotropicElastic(
        [0.0, RADIUS], rho=[5500.0], vp=[10000.0], vs=[5500.0],
        layer_names=["body"], interface_names=["surface"], name="homogeneous")


def two_solid() -> Model:
    return LayeredIsotropicElastic(
        [0.0, MID_MANTLE, RADIUS], rho=[6500.0, 3500.0],
        vp=[12000.0, 8500.0], vs=[6500.0, 4500.0],
        layer_names=["lower", "upper"],
        interface_names=["discontinuity", "surface"], name="two_solid")


def fluid_core() -> Model:
    return LayeredIsotropicElastic(
        [0.0, CMB, RADIUS], rho=[11000.0, 4500.0], vp=[9000.0, 11000.0],
        vs=[0.0, 6000.0], layer_names=["core", "mantle"],
        interface_names=["cmb", "surface"], name="fluid_core")


def inner_core() -> Model:
    return LayeredIsotropicElastic(
        [0.0, ICB, CMB, RADIUS], rho=[13000.0, 11000.0, 4500.0],
        vp=[11000.0, 9000.0, 11000.0], vs=[3500.0, 0.0, 6000.0],
        layer_names=["inner_core", "outer_core", "mantle"],
        interface_names=["icb", "cmb", "surface"], name="inner_core")


MODELS: dict[str, Callable[[], Model]] = {
    "homogeneous": homogeneous,
    "two_solid": two_solid,
    "fluid_core": fluid_core,
    "inner_core": inner_core,
}


def model(name: str) -> Model:
    """The named model, in SI."""
    if name not in MODELS:
        raise KeyError(f"no model named {name!r}; models are {sorted(MODELS)}")
    return MODELS[name]()


def scales(*, time_scale: float | None = None) -> Scales:
    """The benchmark's units: the outer radius, `DENSITY_SCALE` and a time
    scale in seconds, by default 1 / sqrt(G DENSITY_SCALE), for which G is
    one."""
    if time_scale is None:
        time_scale = 1.0 / math.sqrt(G_SI * DENSITY_SCALE)
    return Scales(length=RADIUS, mass=DENSITY_SCALE * RADIUS ** 3,
                  time=float(time_scale))


def scaled(si: Model, *, time_scale: float | None = None) -> Model:
    """`si` in the benchmark's units, with the bulk and shear moduli added
    to every layer as the fields `kappa` and `mu`."""
    out = si.converted(scales(time_scale=time_scale))
    for i, layer in enumerate(out.layers):
        kappa, mu = kappa_mu(layer)
        out = out.with_field(i, "kappa", kappa).with_field(i, "mu", mu)
    return out
