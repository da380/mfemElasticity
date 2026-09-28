"""The spherically symmetric models of the Love-number benchmark.

A ladder of models of increasing complexity on an Earth-sized body, each
a planetmodel `Model` in SI:

  homogeneous      one solid layer of constant density and moduli
  two_solid        two solid layers with a jump in every parameter
  fluid_core       a uniform fluid core under a uniform solid mantle
  inner_core       a solid inner core, a fluid outer core and a mantle,
                   each uniform
  linear_solid     one solid layer, density and velocities linear in radius
  stratified_core  a fluid core whose density falls with radius, under a
                   mantle, every parameter linear in radius
  earth_like       inner core, fluid outer core and mantle, every parameter
                   linear in radius within each
  prem_4           PREM without its ocean, isotropic, on four layers: inner
                   core, outer core, lower mantle, and the upper mantle
                   with the crust
  prem_6           the same on six: the transition zone and the crust are
                   layers of their own

The PREM models keep the boundaries named and merge what lies between
them: within a merged layer each parameter is the cubic closest to PREM's,
so that the model is smooth within its layers, as the mesh takes it to be.
Their radii are PREM's stretched by 6371 / 6368, which puts the surface at
the radius of the other models.

A model of uniform layers is planetmodel's `LayeredIsotropicElastic`; one
whose parameters vary within a layer is `LayeredIsotropicPolynomial` here,
which takes the coefficients of a polynomial in r / RADIUS for each
parameter of each layer. The values are round numbers of the Earth's order, so that the ratio
rho g a / mu of gravitational to elastic forces is of order one, as in
the Earth. `model(name)` returns the model in SI and `scaled(model)`
the same model in the benchmark's units, in which the mesh is built and
the finite-element problem solved: lengths in units of the outer radius,
densities in units of `DENSITY_SCALE`, and times in units of
`time_scale`, by default the one that makes G equal to one.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from planetmodel import (DENSITY, PREM, SCALAR, Elastic, Geometry,
                         LayeredIsotropicElastic, Model, RadialField,
                         SelfGravitating, Skeleton, kappa_mu, polynomial_layer)
from planetmodel.units import G_SI, Scales

#: The outer radius of every model, in metres.
RADIUS = 6371.0e3

#: The density scale of the benchmark's units, in kg m^-3.
DENSITY_SCALE = 5000.0

#: Radii of the internal boundaries, in metres.
ICB = 1200.0e3
CMB = 3500.0e3
MID_MANTLE = 5000.0e3


class LayeredIsotropicPolynomial(Elastic, SelfGravitating, Model):
    """An isotropic model whose density and velocities are polynomials in
    the radius within each layer.

    `boundaries` are the skeleton's, centre outward, in metres; `rho`,
    `vp` and `vs` give for each layer the coefficients c_k of
    sum_k c_k (r / RADIUS)^k, in SI, and a layer whose vs is zero is fluid.
    """

    def __init__(self, boundaries: Sequence[float], *,
                 rho: Sequence[Sequence[float]], vp: Sequence[Sequence[float]],
                 vs: Sequence[Sequence[float]],
                 layer_names: Sequence[str] | None = None,
                 interface_names: Sequence[str] | None = None,
                 name: str | None = None) -> None:
        sk = Skeleton(boundaries)
        geometry = Geometry(sk, layer_names=layer_names,
                            interface_names=interface_names)
        layers = []
        for i in range(sk.nlayers):
            iv = sk.interval(i)
            layers.append({
                key: RadialField(iv, polynomial_layer(iv, c[i], scale=RADIUS),
                                 character=DENSITY if key == "rho" else SCALAR,
                                 name=key)
                for key, c in (("rho", rho), ("vp", vp), ("vs", vs))})
        super().__init__(geometry, layers, scales=Scales.SI, name=name)


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


def linear_solid() -> Model:
    return LayeredIsotropicPolynomial(
        [0.0, RADIUS], rho=[(8000.0, -4500.0)], vp=[(12500.0, -5000.0)],
        vs=[(7000.0, -3000.0)], layer_names=["body"],
        interface_names=["surface"], name="linear_solid")


def stratified_core() -> Model:
    return LayeredIsotropicPolynomial(
        [0.0, CMB, RADIUS],
        rho=[(12500.0, -4500.0), (7000.0, -3000.0)],
        vp=[(10500.0, -4500.0), (15500.0, -6500.0)],
        vs=[(0.0,), (8000.0, -3000.0)],
        layer_names=["core", "mantle"], interface_names=["cmb", "surface"],
        name="stratified_core")


def earth_like() -> Model:
    return LayeredIsotropicPolynomial(
        [0.0, ICB, CMB, RADIUS],
        rho=[(13100.0, -1500.0), (12800.0, -5000.0), (7000.0, -3000.0)],
        vp=[(11300.0, -1000.0), (10800.0, -5000.0), (15500.0, -6500.0)],
        vs=[(3700.0, -500.0), (0.0,), (8000.0, -3000.0)],
        layer_names=["inner_core", "outer_core", "mantle"],
        interface_names=["icb", "cmb", "surface"], name="earth_like")


def prem_like(boundaries_km: Sequence[float], names: Sequence[str],
              interfaces: Sequence[str], *, name: str,
              degree: int = 3) -> Model:
    """PREM without its ocean, isotropic, merged onto the layers between
    `boundaries_km` (radii of PREM, of which the last is its solid surface),
    each parameter the polynomial of `degree` closest to PREM's within a
    layer; a layer of PREM's outer core is fluid."""
    prem = PREM(ocean=False).isotropic()
    fine = np.asarray(prem.skeleton.boundaries, dtype=float)
    stretch = RADIUS / fine[-1]

    def value(key: str, r: np.ndarray) -> np.ndarray:
        index = np.clip(np.searchsorted(fine, r, side="right") - 1, 0,
                        prem.nlayers - 1)
        out = np.empty(r.shape)
        for i in np.unique(index):
            m = index == i
            zero = np.zeros(int(m.sum()))
            out[m] = prem.layer(int(i))[key].evaluate(r[m], zero, zero)
        return out

    coefficients = {"rho": [], "vp": [], "vs": []}
    for lo, hi in zip(boundaries_km[:-1], boundaries_km[1:]):
        lo, hi = 1e3 * lo, 1e3 * hi
        t = np.cos(np.pi * (np.arange(64) + 0.5) / 64)
        r = 0.5 * (lo + hi) + 0.5 * (hi - lo) * t * (1.0 - 1e-9)
        x = r * stretch / RADIUS
        for key, c in coefficients.items():
            y = value(key, r)
            if key == "vs" and np.all(y == 0.0):
                c.append((0.0,))
            else:
                c.append(tuple(np.polynomial.Polynomial.fit(
                    x, y, degree).convert().coef))
    return LayeredIsotropicPolynomial(
        [1e3 * b * stretch for b in boundaries_km], **coefficients,
        layer_names=list(names), interface_names=list(interfaces), name=name)


def prem_4() -> Model:
    return prem_like(
        [0.0, 1221.5, 3480.0, 5701.0, 6368.0],
        ["inner_core", "outer_core", "lower_mantle", "upper_mantle"],
        ["icb", "cmb", "d660", "surface"], name="prem_4")


def prem_6() -> Model:
    return prem_like(
        [0.0, 1221.5, 3480.0, 5701.0, 5971.0, 6346.6, 6368.0],
        ["inner_core", "outer_core", "lower_mantle", "transition_zone",
         "upper_mantle", "crust"],
        ["icb", "cmb", "d660", "d400", "moho", "surface"], name="prem_6")


MODELS: dict[str, Callable[[], Model]] = {
    "homogeneous": homogeneous,
    "two_solid": two_solid,
    "fluid_core": fluid_core,
    "inner_core": inner_core,
    "linear_solid": linear_solid,
    "stratified_core": stratified_core,
    "earth_like": earth_like,
    "prem_4": prem_4,
    "prem_6": prem_6,
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
