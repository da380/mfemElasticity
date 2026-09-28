"""Compare the runs of the Love-number benchmark with their references.

Reads the tree run.py writes for one model, `<runs>/<model>/h*/`, prints a
table of the relative errors of every run and writes three figures beside
the cases:

  love_numbers.png   h', k' (load) and h, k (tide) by degree: the reference
                     and the finest run of each order
  errors.png         the relative error of each by degree, for every run
  convergence.png    the relative error against the number of unknowns,
                     one line per degree, for each order

At degree one the load numbers depend on the frame and h' - k' is compared
in place of each; with a fluid layer degree zero is left out (see
README.md).

    python plot.py runs/homogeneous
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
    ("k_load", "load $k'$", "load", "k"),
    ("h_tide", "tidal $h$", "tide", "h"),
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
    values: dict[str, dict[int, float]] = {key: {} for key, *_ in QUANTITIES}
    seconds = r["setup_seconds"]
    for d in r["degrees"]:
        l = d["degree"]
        for key, _, forcing, name in QUANTITIES:
            if forcing not in d or d[forcing][name] is None:
                continue
            seconds += d[forcing]["seconds"]
            if forcing == "load" and l == 0 and (fluid or name == "k"):
                # k' vanishes at degree zero: nothing to be relative to
                continue
            if forcing == "load" and l == 1:
                # frame-dependent: h' - k' is kept under h, nothing under k
                if name == "h":
                    values[key][l] = d["load"]["h"] - d["load"]["k"]
                continue
            values[key][l] = d[forcing][name]
    h = float(path.parent.name[1:])
    return Run(h=h, order=r["order"], ranks=r["ranks"], seconds=seconds,
               unknowns=r["displacement_unknowns"] + r["potential_unknowns"],
               values=values, label=f"h = {h:g}, order {r['order']}")


def reference_values(ref: dict) -> dict[str, dict[int, float]]:
    out = {}
    for key, *_ in QUANTITIES:
        out[key] = {l: v for l, v in zip(ref["degree"], ref[key])
                    if v is not None}
    # degree one as h' - k', to match the runs
    out["h_load"][1] = ref["h_load"][1] - ref["k_load"][1]
    out["k_load"].pop(1, None)
    return out


def relative_error(run: Run, ref: dict, key: str) -> dict[int, float]:
    return {l: abs(v - ref[key][l]) / abs(ref[key][l])
            for l, v in run.values[key].items()
            if l in ref[key] and ref[key][l] != 0.0}


def print_table(runs: list[Run], ref: dict) -> None:
    for run in runs:
        print(f"\n{run.label}: {run.unknowns} unknowns, {run.ranks} ranks, "
              f"{run.seconds:.1f} s")
        print("   l " + "".join(f"{label:>24}" for _, label, *_ in QUANTITIES))
        degrees = sorted({l for key, *_ in QUANTITIES
                          for l in run.values[key]})
        for l in degrees:
            row = f"{l:4d} "
            for key, *_ in QUANTITIES:
                if l in run.values[key]:
                    e = relative_error(run, ref, key).get(l, float("nan"))
                    row += f"{run.values[key][l]:14.6f} ({e:7.1e})"
                else:
                    row += " " * 24
            print(row + ("   (h' - k')" if l == 1 else ""))


def finest_per_order(runs: list[Run]) -> list[Run]:
    best: dict[int, Run] = {}
    for run in runs:
        if run.order not in best or run.h < best[run.order].h:
            best[run.order] = run
    return [best[o] for o in sorted(best)]


def plot_love_numbers(runs: list[Run], ref: dict, title: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    shown = finest_per_order(runs)
    for ax, (key, label, *_) in zip(axes.flat, QUANTITIES):
        ls = sorted(l for l in ref[key] if l != 1 and any(
            l in run.values[key] for run in shown))
        ax.plot(ls, [ref[key][l] for l in ls], color=MUTED, linewidth=1.5,
                label="reference (pyslfp)", zorder=1)
        for i, run in enumerate(shown):
            rl = [l for l in ls if l in run.values[key]]
            ax.plot(rl, [run.values[key][l] for l in rl], linestyle="none",
                    marker=MARKERS[i], color=COLOURS[i], markeredgecolor=SURFACE,
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    shown = runs[:len(COLOURS)]
    for ax, (key, label, *_) in zip(axes.flat, QUANTITIES):
        for i, run in enumerate(shown):
            e = relative_error(run, ref, key)
            ls = sorted(e)
            ax.semilogy(ls, [e[l] for l in ls], marker=MARKERS[i],
                        color=COLOURS[i], markeredgecolor=SURFACE,
                        markeredgewidth=1.0, label=run.label)
        ax.set_title(label + (" ($h' - k'$ at degree 1)"
                              if key == "h_load" else ""), loc="left")
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


def plot_convergence(runs: list[Run], ref: dict, title: str, out: Path) -> None:
    orders = sorted({run.order for run in runs})
    fig, axes = plt.subplots(len(orders), 4, figsize=(14, 3.6 * len(orders)),
                             sharey=True, squeeze=False)
    for row, order in zip(axes, orders):
        mine = sorted((r for r in runs if r.order == order),
                      key=lambda r: r.unknowns)
        for ax, (key, label, *_) in zip(row, QUANTITIES):
            errors = [relative_error(run, ref, key) for run in mine]
            # the colour and marker are the degree's, the same in every panel
            degrees = [l for l in sorted(set().union(*errors))
                       if l < len(COLOURS)]
            for l in degrees:
                i = l
                pts = [(run.unknowns, e[l]) for run, e in zip(mine, errors)
                       if l in e]
                ax.loglog(*zip(*pts), marker=MARKERS[i], color=COLOURS[i],
                          markeredgecolor=SURFACE, markeredgewidth=1.0,
                          label=f"degree {l}")
            ax.set_title(f"{label}, order {order}", loc="left")
            ax.set_xlabel("unknowns")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        row[0].set_ylabel("relative error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="best", fontsize=8, ncol=2)
    fig.suptitle(f"{title}: convergence", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=Path,
                   help="the tree of one model, <runs>/<model>")
    args = p.parse_args()

    runs, ref, fluid = [], None, False
    for case in sorted(args.directory.glob("h*")):
        if not (case / "reference.json").exists():
            continue
        manifest = json.loads((case / "case.json").read_text())
        fluid = bool(manifest["meta"].get("fluid_layers"))
        reference = json.loads((case / "reference.json").read_text())
        ref = reference_values(reference)
        title = reference["model"]
        for path in sorted(case.glob("results_o*.json")):
            runs.append(read_run(path, fluid=fluid))
    if not runs:
        raise SystemExit(f"no results under {args.directory}")
    runs.sort(key=lambda r: (r.order, -r.h))

    print_table(runs, ref)
    style()
    plot_love_numbers(runs, ref, title, args.directory / "love_numbers.png")
    plot_errors(runs, ref, title, args.directory / "errors.png")
    plot_convergence(runs, ref, title, args.directory / "convergence.png")
    print(f"\nfigures in {args.directory}")


if __name__ == "__main__":
    main()
