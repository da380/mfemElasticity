"""Compare the runs of the Love-number benchmark with their references.

Reads the tree run.py writes for one model, `<runs>/<model>/h*/`, or for
all of them, prints a
table of the relative errors of every run and writes figures beside the
cases:

  love_numbers.png   h', l', k' (load) and h, l, k (tide) by degree: the reference
                     and the finest run of each order
  errors.png         the relative error of each by degree, for every run
  convergence.png    the h-refinement study: the relative error against the
                     element size at fixed order, every size its own mesh,
                     with the observed rate fitted over the ladder
  field_convergence.png  the L2 errors of the cap-load fields against the
                     element size, likewise
  profiles.png       U, V and phi of the load problem by radius: the
                     reference, and the finest run of each order on the
                     interfaces of the solid and, dashed, within the layers
  field_<run>.png    for each run of field_benchmark, maps on the surface of
                     the radial displacement and the potential: reference,
                     run and difference
  field_spectrum.png the error of those runs by degree

and the relative L2 errors of the fields, over the solid (u) and over the
body and its buffer (phi).

At degree one the numbers are those of the centre-of-mass frame, in which
k' is minus one in both solutions and is not plotted; with a fluid layer
degree zero is left out (see README.md).

    python plot.py runs/homogeneous
    python plot.py runs          every model, and the summary runs/summary.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

#: Categorical colours in their fixed order, with a marker for each so that
#: no series is told by colour alone.
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

#: The quantities compared: key, label, forcing in the results, the
#: reference's array.
QUANTITIES = (
    ("h_load", "load $h'$", "load", "h"),
    ("l_load", "load $l'$", "load", "l"),
    ("k_load", "load $k'$", "load", "k"),
    ("h_tide", "tidal $h$", "tide", "h"),
    ("l_tide", "tidal $l$", "tide", "l"),
    ("k_tide", "tidal $k$", "tide", "k"),
)



@dataclass
class Run:
    h: float
    order: int
    unknowns: int
    ranks: int
    seconds: float
    #: quantity -> {degree: value}
    values: dict[str, dict[int, float]]
    label: str
    method: str = "dahlen"
    #: a filename-style tag of the variant ("", "_gauged", "_slip_broken",
    #: "_nomass", "_schur", combinations), telling runs of one (h, order)
    #: apart
    tag: str = ""
    setup_seconds: float = 0.0
    #: wall seconds and outer iterations of the LOAD solve, by degree
    solve_seconds: dict[int, float] = None
    iterations: dict[int, int] = None


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 7, "font.size": 10,
        "axes.titlesize": 11, "legend.frameon": False,
    })


def read_run(path: Path, *, fluid: bool) -> Run:
    r = json.loads(path.read_text())
    method = r.get("method",
                   "gauged" if r.get("fluid_treatment") == "gauged"
                   else "dahlen")
    cmb = r.get("cmb", "full")
    schur = r.get("solver") == "schur_cg"
    map_amplitude = r.get("map_amplitude", 0.0)
    values: dict[str, dict[int, float]] = {key: {} for key, *_ in QUANTITIES}
    seconds = r["setup_seconds"]
    solve_seconds: dict[int, float] = {}
    iterations: dict[int, int] = {}
    # A combined run (love_benchmark -combined) solves each forcing once
    # for all degrees: its cost is written once, under "combined_solves",
    # and the per-degree times and iteration counts are null.
    combined = bool(r.get("combined", False))
    for solve in r.get("combined_solves", {}).values():
        seconds += solve["seconds"]
    for d in r["degrees"]:
        l = d["degree"]
        if "load" in d and d["load"].get("seconds") is not None:
            solve_seconds[l] = d["load"]["seconds"]
            iterations[l] = d["load"]["outer_iterations"]
        for key, _, forcing, name in QUANTITIES:
            if forcing not in d or d[forcing].get(name) is None:
                continue
            if name == "h" and d[forcing].get("seconds") is not None:
                seconds += d[forcing]["seconds"]
            if forcing == "load" and l == 0 and name != "h":
                # k' and l' vanish at degree zero: nothing to be relative to
                continue
            if (forcing == "load" and l == 0 and fluid
                    and method == "dahlen"):
                # Dahlen's fluid differs from the reference at degree zero
                # by design (doc/gauged_fluid.md); the welded and slipping
                # treatments describe the fluid compressibly and are
                # comparable there.
                continue
            if forcing == "load" and l == 1 and name == "k":
                # minus one by the choice of frame
                continue
            values[key][l] = d[forcing][name]
    h = float(path.parent.name[1:])
    tag = "" if method == "dahlen" else f"_{method}"
    label = f"h = {h:g}, order {r['order']}"
    if method != "dahlen":
        label += f" ({method})"
    if cmb != "full":
        tag += f"_{cmb}"
        label += f" (cmb {cmb})"
    if schur:
        tag += "_schur"
        label += " (schur)"
    if map_amplitude:
        tag += f"_map{map_amplitude:g}"
        label += f" (map {map_amplitude:g})"
    if combined:
        tag += "_combined"
        label += " (combined)"
    return Run(h=h, order=r["order"], ranks=r["ranks"], seconds=seconds,
               unknowns=r["displacement_unknowns"] + r["potential_unknowns"],
               values=values, label=label, method=method, tag=tag,
               setup_seconds=r["setup_seconds"],
               solve_seconds=solve_seconds, iterations=iterations)


def reference_values(ref: dict) -> dict[str, dict[int, float]]:
    out = {}
    for key, *_ in QUANTITIES:
        out[key] = {l: v for l, v in zip(ref["degree"], ref[key])
                    if v is not None}
    out["k_load"].pop(1, None)
    out["l_load"].pop(0, None)
    return out


def relative_error(run: Run, ref: dict, key: str) -> dict[int, float]:
    return {l: abs(v - ref[key][l]) / abs(ref[key][l])
            for l, v in run.values[key].items()
            if l in ref[key] and ref[key][l] != 0.0}


def print_table(runs: list[Run], ref: dict) -> None:
    for run in runs:
        print(f"\n{run.label}: {run.unknowns} unknowns, {run.ranks} ranks, "
              f"{run.seconds:.1f} s")
        print("   l " + "".join(f"{label:>25}" for _, label, *_ in QUANTITIES))
        degrees = sorted({l for key, *_ in QUANTITIES
                          for l in run.values[key]})
        for l in degrees:
            row = f"{l:4d} "
            for key, *_ in QUANTITIES:
                if l in run.values[key]:
                    e = relative_error(run, ref, key).get(l, float("nan"))
                    row += f"{run.values[key][l]:15.6f} ({e:7.1e})"
                else:
                    row += " " * 25
            print(row)


def finest_per_order(runs: list[Run]) -> list[Run]:
    # One entry per order and fluid/CMB treatment (each variant is its own
    # series, tagged in the label's parenthetical).
    best: dict[tuple[int, str], Run] = {}
    for run in runs:
        i = run.label.find("(")
        key = (run.order, run.label[i:] if i >= 0 else "")
        if key not in best or run.h < best[key].h:
            best[key] = run
    return [best[k] for k in sorted(best)]


def plot_love_numbers(runs: list[Run], ref: dict, title: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    shown = finest_per_order(runs)
    for ax, (key, label, *_) in zip(axes.flat, QUANTITIES):
        ls = sorted(l for l in ref[key] if l != 1 and any(
            l in run.values[key] for run in shown))
        ax.plot(ls, [ref[key][l] for l in ls], color=MUTED, linewidth=1.5,
                label="reference (pyslfp)", zorder=1)
        for i, run in enumerate(shown):
            rl = [l for l in ls if l in run.values[key]]
            ax.plot(rl, [run.values[key][l] for l in rl], linestyle="none",
                    marker=MARKERS[(i + i // len(COLOURS)) % len(MARKERS)], color=COLOURS[i % len(COLOURS)], markeredgecolor=SURFACE,
                    markeredgewidth=1.0, label=run.label, zorder=2)
        ax.set_title(label, loc="left")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for ax in axes[-1]:
        ax.set_xlabel("degree")
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"{title}: Love numbers", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_errors(runs: list[Run], ref: dict, title: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    shown = runs
    for ax, (key, label, *_) in zip(axes.flat, QUANTITIES):
        for i, run in enumerate(shown):
            e = relative_error(run, ref, key)
            ls = sorted(e)
            ax.semilogy(ls, [e[l] for l in ls], marker=MARKERS[(i + i // len(COLOURS)) % len(MARKERS)],
                        color=COLOURS[i % len(COLOURS)], markeredgecolor=SURFACE,
                        markeredgewidth=1.0, label=run.label)
        ax.set_title(label, loc="left")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for ax in axes[-1]:
        ax.set_xlabel("degree")
    for ax in axes[:, 0]:
        ax.set_ylabel("relative error")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{title}: relative error against the reference", x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_timing(runs: list[Run], title: str, out: Path) -> None:
    """The cost comparison of the methods (and solver and interface
    variants): wall seconds and outer iterations of the load solve by
    degree, one series per run, setup times in the legend."""
    shown = runs
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for i, run in enumerate(shown):
        ls = sorted(run.solve_seconds or {})
        if not ls:
            continue
        axes[0].semilogy(ls, [run.solve_seconds[l] for l in ls],
                         marker=MARKERS[(i + i // len(COLOURS)) % len(MARKERS)], color=COLOURS[i % len(COLOURS)],
                         markeredgecolor=SURFACE, markeredgewidth=1.0,
                         label=f"{run.label}; setup "
                               f"{run.setup_seconds:.1f} s")
        axes[1].plot(ls, [run.iterations[l] for l in ls],
                     marker=MARKERS[(i + i // len(COLOURS)) % len(MARKERS)], color=COLOURS[i % len(COLOURS)],
                     markeredgecolor=SURFACE, markeredgewidth=1.0)
    axes[0].set_title("load solve, wall seconds", loc="left")
    axes[1].set_title("outer iterations", loc="left")
    for ax in axes:
        ax.set_xlabel("degree")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle(f"{title}: cost by method and solver", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fitted_rate(h: list[float], e: list[float]) -> float | None:
    """The observed rate p of e ~ h^p: the least-squares slope of log e
    against log h, or None with fewer than two usable sizes."""
    pts = [(np.log(a), np.log(b)) for a, b in zip(h, e) if b > 0.0]
    if len(pts) < 2:
        return None
    x, y = zip(*pts)
    return float(np.polyfit(x, y, 1)[0])


def ladder(runs: list[Run], order: int) -> list[Run]:
    """The runs of one order, coarsest first: the h-refinement ladder,
    every size a mesh of its own (run.py re-meshes for each h)."""
    return sorted((r for r in runs if r.order == order), key=lambda r: -r.h)


def plot_convergence(runs: list[Run], ref: dict, title: str, out: Path) -> None:
    """The h-refinement study: the relative error against the element
    size, the orders together in each panel, at two representative
    degrees, the observed rate of each line fitted over the ladder."""
    orders = sorted({run.order for run in runs})
    degrees = sorted({l for run in runs for key, *_ in QUANTITIES
                      for l in run.values[key]})
    shown = list(dict.fromkeys(
        l for l in (2, max(degrees, default=2)) if l in degrees))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), sharex=True,
                             squeeze=False)
    for ax, (key, label, *_) in zip(axes.flat, QUANTITIES):
        for i, order in enumerate(orders):
            mine = ladder(runs, order)
            errors = [relative_error(run, ref, key) for run in mine]
            for l, linestyle in zip(shown, ("-", "--")):
                pts = [(run.h, e[l]) for run, e in zip(mine, errors)
                       if l in e and e[l] > 0.0]
                if len(pts) < 2:
                    continue
                rate = fitted_rate(*zip(*pts))
                ax.loglog(*zip(*pts), marker=MARKERS[(i + i // len(COLOURS)) % len(MARKERS)], color=COLOURS[i % len(COLOURS)],
                          linestyle=linestyle,
                          markeredgecolor=SURFACE, markeredgewidth=1.0,
                          label=f"order {order}, degree {l} "
                                f"($p$ = {rate:.1f})")
        ax.set_title(label, loc="left")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(loc="best", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("element size $h$")
    for row in axes:
        row[0].set_ylabel("relative error")
    fig.suptitle(f"{title}: error against element size, each size its own "
                 "mesh; $p$ the observed rate", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_field_convergence(runs_by: dict[str, Run],
                           fields: list[tuple[str, dict]], title: str,
                           out: Path) -> bool:
    """The relative L2 errors of the cap-load fields against the element
    size at fixed order; True when there was a ladder to draw."""
    by_order: dict[int, list[tuple[float, dict]]] = {}
    for key, field in fields:
        run = runs_by.get(key)
        h = run.h if run is not None else float(key.split("_o")[0][1:])
        by_order.setdefault(field["order"], []).append((h, field))
    if not any(len(v) > 1 for v in by_order.values()):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for ax, (key, label) in zip(axes, (("u_error", "$u$ over the solid"),
                                       ("phi_error",
                                        "$\\phi$ over body and buffer"))):
        for i, order in enumerate(sorted(by_order)):
            pts = sorted((h, f[key]) for h, f in by_order[order])
            if len(pts) < 2:
                continue
            rate = fitted_rate(*zip(*pts))
            ax.loglog(*zip(*pts), marker=MARKERS[(i + i // len(COLOURS)) % len(MARKERS)], color=COLOURS[i % len(COLOURS)],
                      markeredgecolor=SURFACE, markeredgewidth=1.0,
                      label=f"order {order} ($p$ = {rate:.1f})")
        ax.set_title(f"relative L2 error of {label}", loc="left")
        ax.set_xlabel("element size $h$")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("relative error")
    fig.suptitle(f"{title}: cap-load fields against element size",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def print_rates(runs: list[Run], ref: dict) -> None:
    """The observed rates over the h ladder of each order, by quantity and
    degree: the slope of log error against log h."""
    for order in sorted({run.order for run in runs}):
        mine = ladder(runs, order)
        sizes = sorted({run.h for run in mine}, reverse=True)
        if len(sizes) < 2:
            continue
        degrees = sorted({l for run in mine for key, *_ in QUANTITIES
                          for l in run.values[key]})
        print(f"\norder {order}, observed rate p of error ~ h^p over h = "
              + ", ".join(f"{h:g}" for h in sizes) + ":")
        print("            " + "".join(f"  l={l:<4d}" for l in degrees))
        for key, label, *_ in QUANTITIES:
            errors = [relative_error(run, ref, key) for run in mine]
            row = ""
            for l in degrees:
                pts = [(run.h, e[l]) for run, e in zip(mine, errors)
                       if l in e and e[l] > 0.0]
                rate = fitted_rate(*zip(*pts)) if len(pts) > 1 else None
                row += f"  {rate:6.2f}" if rate is not None else "       -"
            print(f"  {label.replace('$', ''):>10}" + row)


def plot_profiles(runs: list[Run], reference: dict, results: dict[str, dict],
                  title: str, out: Path) -> None:
    """U, V and phi of the load problem against radius, radius upward."""
    shown = finest_per_order(runs)
    profiles = reference["profiles"]
    radius = np.array(profiles["radius"], dtype=float)
    layer = np.array(profiles["layer"])
    outer = max((p["radius"][-1] for run in shown
                 for d in results[run.label]["degrees"]
                 for p in d["load"].get("profiles", [])), default=0.0)
    solved = sorted({d["degree"] for run in shown
                     for d in results[run.label]["degrees"]})
    degrees = [l for l in (1, 2, 3, 5, 8) if l in solved][:4]
    if not degrees:
        return
    fig, axes = plt.subplots(3, len(degrees), sharey=True, squeeze=False,
                             figsize=(3.4 * len(degrees) + 0.6, 10))
    names = (("U", "u", "$U$"), ("V", "v", "$V$"), ("phi", "phi", "$\\phi$"))
    for col, l in enumerate(degrees):
        ref = next(s for s in profiles["solutions"]
                   if s["degree"] == l and s["forcing"] == "load")
        for row, (key, fe_key, label) in enumerate(names):
            ax = axes[row, col]
            values = np.array([np.nan if v is None else v for v in ref[key]])
            for k in np.unique(layer):
                m = layer == k
                ax.plot(values[m], radius[m], color=MUTED, linewidth=3.0,
                        alpha=0.45,
                        label="reference (pyslfp)" if k == layer[0] else None,
                        zorder=1)
            if key == "phi" and outer > radius[-1]:
                # outside the body the potential is harmonic
                a = radius[-1]
                r = np.linspace(a, outer, 20)
                ax.plot(values[-1] * (a / r) ** (l + 1), r, color=MUTED,
                        linewidth=3.0, alpha=0.45, zorder=1)
            for i, run in enumerate(shown):
                r = results[run.label]
                d = next((d for d in r["degrees"] if d["degree"] == l), None)
                if d is None or fe_key not in d["load"]:
                    continue
                # the radial functions within the layers, dashed over the
                # reference, and the values on the interfaces
                for p in d["load"].get("profiles", []):
                    if p[fe_key]:
                        ax.plot(p[fe_key], p["radius"], color=COLOURS[i % len(COLOURS)],
                                linewidth=1.5, linestyle=(0, (4, 3)),
                                zorder=2)
                ax.plot(d["load"][fe_key],
                        [f["radius"] for f in r["interfaces"]],
                        linestyle="none", marker=MARKERS[(i + i // len(COLOURS)) % len(MARKERS)], color=COLOURS[i % len(COLOURS)],
                        markeredgecolor=SURFACE, markeredgewidth=1.0,
                        label=run.label, zorder=3)
            ax.set_title(f"{label}, degree {l}", loc="left")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    for ax in axes[:, 0]:
        ax.set_ylabel("radius")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{title}: radial solutions of the load problem, per unit "
                 "load", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def harmonics(lmax: int, colatitude: np.ndarray,
              longitude: np.ndarray) -> np.ndarray:
    """The real orthonormal harmonics of the drivers at the given angles,
    with index l^2 + l + m: Y_l0 = X_l0, Y_lm = sqrt 2 X_lm cos m phi and
    Y_l,-m = sqrt 2 X_lm sin m phi for m > 0, X_lm the normalised
    associated Legendre functions with the Condon-Shortley phase."""
    from scipy.special import gammaln, lpmv
    x = np.cos(colatitude)
    out = np.empty(((lmax + 1) ** 2,) + np.shape(x))
    for l in range(lmax + 1):
        for m in range(l + 1):
            norm = np.sqrt((2 * l + 1) / (4 * np.pi) * np.exp(
                gammaln(l - m + 1) - gammaln(l + m + 1)))
            X = norm * lpmv(m, l, x)
            if m == 0:
                out[l * l + l] = X
            else:
                out[l * l + l + m] = np.sqrt(2.0) * X * np.cos(m * longitude)
                out[l * l + l - m] = np.sqrt(2.0) * X * np.sin(m * longitude)
    return out


def check_harmonics(field: dict) -> None:
    """The load of a field run at the directions it was sampled at, against
    its coefficients summed with the harmonics here."""
    c = np.array(field["load"])
    for sample in field["samples"]:
        x = np.array(sample["x"])
        Y = harmonics(field["lmax"], np.arccos(x[2] / np.linalg.norm(x)),
                      np.arctan2(x[1], x[0]))
        if abs(c @ Y - sample["load"]) > 1e-9 * np.abs(c).max():
            raise SystemExit("the harmonics of plot.py are not those of the "
                             "driver: the maps would be wrong")


def diverging() -> matplotlib.colors.Colormap:
    """Two hues about a neutral middle, for signed fields."""
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "diverging", ["#1c4f8f", "#2a78d6", "#e9e8e4", "#e34948", "#96282a"])


def plot_field_maps(field: dict, title: str, out: Path) -> None:
    lat = np.linspace(-90.0, 90.0, 181)
    lon = np.linspace(-180.0, 180.0, 361)
    LON, LAT = np.meshgrid(np.radians(lon), np.radians(lat))
    Y = harmonics(field["lmax"], 0.5 * np.pi - LAT, LON)
    rows = (("u", "radial displacement $U$"), ("phi", "potential $\\phi$"))
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 6.4),
                             subplot_kw={"projection": "mollweide"})
    cmap = diverging()
    for axrow, (key, label) in zip(axes, rows):
        fe = np.tensordot(np.array(field[key]), Y, axes=1)
        ref = np.tensordot(np.array(field[f"{key}_reference"]), Y, axes=1)
        scale = np.abs(ref).max()
        panels = ((ref, "reference", scale), (fe, "finite elements", scale),
                  (fe - ref, "difference", np.abs(fe - ref).max()))
        for ax, (values, name, vmax) in zip(axrow, panels):
            mesh = ax.pcolormesh(LON, LAT, values, cmap=cmap, vmin=-vmax,
                                 vmax=vmax, shading="auto", rasterized=True)
            ax.set_title(f"{label}: {name}", loc="left")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, color=GRID, linewidth=0.5)
            bar = fig.colorbar(mesh, ax=ax, orientation="horizontal",
                               fraction=0.05, pad=0.06)
            bar.formatter.set_powerlimits((-2, 2))
            bar.outline.set_visible(False)
    cap = field["cap"]
    fig.suptitle(
        f"{title}: response on the surface to a cap of radius "
        f"{cap['radius']:g} degrees at latitude {cap['latitude']:g}, "
        f"longitude {cap['longitude']:g}, degrees {field['lmin']} to "
        f"{field['lmax']}; order {field['order']}", x=0.01, ha="left")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_field_spectrum(fields: list[tuple[str, dict]], title: str,
                        out: Path) -> None:
    """By degree, the root mean square over the orders of the error of the
    coefficients on the surface, relative to that of the reference."""
    names = (("u", "$U$"), ("v", "$V$"), ("phi", "$\\phi$"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (key, label) in zip(axes, names):
        for i, (run, field) in enumerate(fields):
            degree = np.array(field["degree"]).astype(int)
            fe = np.array(field[key])
            ref = np.array(field[f"{key}_reference"])
            ls, errors = [], []
            for l in range(field["lmin"], field["lmax"] + 1):
                m = degree == l
                size = np.sqrt(np.mean(ref[m] ** 2))
                if size > 0.0:
                    ls.append(l)
                    errors.append(np.sqrt(np.mean((fe[m] - ref[m]) ** 2))
                                  / size)
            ax.semilogy(ls, errors, marker=MARKERS[(i + i // len(COLOURS)) % len(MARKERS)], color=COLOURS[i % len(COLOURS)],
                        markeredgecolor=SURFACE, markeredgewidth=1.0,
                        label=run)
        ax.set_title(f"{label} on the surface", loc="left")
        ax.set_xlabel("degree")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].set_ylabel("relative error")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{title}: error of the response to the cap load, by degree",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_model(directory: Path) -> list[str]:
    """The table and the figures of one model; returns the lines of its
    summary, one per run."""
    runs, ref, fluid = [], None, False
    results, fields = {}, []
    for case in sorted(directory.glob("h*")):
        if not (case / "reference.json").exists():
            continue
        manifest = json.loads((case / "case.json").read_text())
        # layers[].fluid since manifest schema 5; meta.fluid_layers before.
        fluid = (any(layer.get("fluid") for layer in manifest["layers"])
                 or bool(manifest.get("meta", {}).get("fluid_layers")))
        reference = json.loads((case / "reference.json").read_text())
        ref = reference_values(reference)
        title = reference["model"]
        for path in sorted(case.glob("results_o*.json")):
            if "_shift" in path.stem:
                continue  # a perturbed MODEL, the perturbation
                # family's business (perturbation/plot.py)
            runs.append(read_run(path, fluid=fluid))
            results[runs[-1].label] = json.loads(path.read_text())
        for path in sorted(case.glob("field_o*.json")):
            field = json.loads(path.read_text())
            fields.append((f"{case.name}_o{field['order']}", field))
    if not runs and not fields:
        return []
    runs.sort(key=lambda r: (r.order, -r.h))

    summary = []
    print(f"\n=== {title} ===")
    if runs:
        print_table(runs, ref)
        print_rates(runs, ref)
        plot_love_numbers(runs, ref, title, directory / "love_numbers.png")
        plot_errors(runs, ref, title, directory / "errors.png")
        plot_convergence(runs, ref, title, directory / "convergence.png")
        plot_timing(runs, title, directory / "timing.png")
        plot_profiles(runs, reference, results, title,
                      directory / "profiles.png")
    by_run = {f"h{run.h:g}_o{run.order}{run.tag}": run for run in runs}
    if fields:
        print("\nfields of the cap load, relative L2 error:")
        for run, field in fields:
            check_harmonics(field)
            print(f"  {run:>12}: u {field['u_error']:.2e}, phi "
                  f"{field['phi_error']:.2e}  ("
                  f"{field['displacement_unknowns'] + field['potential_unknowns']}"
                  f" unknowns, {field['seconds']:.1f} s)")
            plot_field_maps(field, title, directory / f"field_{run}.png")
        plot_field_spectrum(fields, title, directory / "field_spectrum.png")
        plot_field_convergence(by_run, fields, title,
                               directory / "field_convergence.png")
    by_field = dict(fields)
    for key in sorted(set(by_run) | set(by_field)):
        run, field = by_run.get(key), by_field.get(key)
        worst = {}
        if run is not None:
            for l in (2, 5):
                errors = [relative_error(run, ref, q).get(l)
                          for q, *_ in QUANTITIES]
                errors = [e for e in errors if e is not None]
                worst[l] = f"{max(errors):.1e}" if errors else "-"
        unknowns = run.unknowns if run is not None else (
            field["displacement_unknowns"] + field["potential_unknowns"])
        method = run.method if run is not None else "dahlen"
        setup = f"{run.setup_seconds:.0f}" if run is not None else "-"
        summary.append(
            f"| {title} | {key} | {method} | {unknowns} | "
            f"{worst.get(2, '-')} | {worst.get(5, '-')} | "
            + (f"{field['u_error']:.1e} | {field['phi_error']:.1e} | "
               if field is not None else "- | - | ")
            + (f"{setup} | {run.seconds:.0f} |"
               if run is not None else "- | - |"))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=Path,
                   help="the tree of one model, <runs>/<model>, or the tree "
                        "of all of them, <runs>")
    args = p.parse_args()

    style()
    if any(args.directory.glob("h*/reference.json")):
        directories = [args.directory]
    else:
        directories = sorted(d for d in args.directory.iterdir() if d.is_dir())
    summary = []
    for directory in directories:
        summary += plot_model(directory)
    if not summary:
        raise SystemExit(f"no results under {args.directory}")
    lines = ["| model | run | method | unknowns | worst error, degree 2 | "
             "worst error, degree 5 | field error, u | field error, phi | "
             "setup s | total s |", "|---|---|---|---|---|---|---|---|---|"
             "---|", *summary]
    out = (args.directory if len(directories) > 1
           else args.directory.parent) / "summary.md"
    if len(directories) > 1:
        out.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nfigures in {args.directory}")


if __name__ == "__main__":
    main()
