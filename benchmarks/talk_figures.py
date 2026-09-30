"""Slide-grade figures for talks, from the benchmark trees already run.

Single-message figures with slide typography (large type, one point per
panel), regenerated at will from the raw results:

  methods.png     five formulations, one reference: agreement by degree
                  and the cost of each (fluid_core, h = 0.3, order 2)
  cmb.png         what the CMB approximations cost, loading vs tidal
                  (prem_4, h = 0.2, order 2), the rotational-feedback
                  column being the tidal one
  identity.png    the relabelled change-of-variables identity: covariant
                  terms certified to solver precision, the gauge penalty
                  as the one known non-covariant piece
  derivative.png  d(Love)/d(interface radius), 3-D against 1-D — the
                  adjoint teaser (fluid_core, referential)
  aspherical.png  field errors on the independently meshed aspherical
                  body

    python talk_figures.py --out talk
    python talk_figures.py --out talk --methods-case <dir> ...

Every input has a flag; the defaults point at the build-tree runs this
repository's campaigns produce (run from <build>/benchmarks).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOURS = {"dahlen": "#2a78d6", "gauged": "#eb6834",
           "referential": "#eda100", "slip": "#008300",
           "slip_broken": "#4a3aa7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GOOD, WARN = "#1baf7a", "#eda100"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2.4, "lines.markersize": 9,
    "figure.facecolor": "white", "savefig.dpi": 200,
})


def love(path: Path) -> dict:
    r = json.loads(path.read_text())
    return {d["degree"]: d for d in r["degrees"]}


def reference(case: Path) -> dict:
    r = json.loads((case / "reference.json").read_text())
    return {l: {q: r[f"{q}_load"][i] for q in "hlk"}
            for i, l in enumerate(r["degree"])}


def save(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def methods_figure(case: Path, out: Path) -> None:
    """Agreement by degree and cost per solve, the five formulations."""
    ref = reference(case)
    series = [("dahlen", ""), ("gauged", "_gauged"),
              ("referential", "_referential"), ("slip", "_slip"),
              ("slip_broken", "_slip_broken")]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.6, 4.8),
                                 gridspec_kw={"width_ratios": [3, 2]})
    names, costs, its = [], [], []
    for name, tag in series:
        r = love(case / f"results_o2{tag}.json")
        ls, errs = [], []
        for l in sorted(set(r) & set(ref)):
            if l == 0 and name == "dahlen":
                continue  # differs by design with a fluid
            worst = 0.0
            for q in ("h", "k"):
                if q == "k" and l < 2:
                    continue
                if abs(ref[l][q]) > 1e-10:
                    worst = max(worst, abs(r[l]["load"][q] - ref[l][q])
                                / abs(ref[l][q]))
            ls.append(l)
            errs.append(worst)
        a0.semilogy(ls, errs, marker="o", color=COLOURS[name], label=name)
        loads = [r[l]["load"] for l in sorted(r)]
        names.append(name)
        costs.append(sum(d["seconds"] for d in loads) / len(loads))
        its.append(sum(d["outer_iterations"] for d in loads) / len(loads))
    a0.set_xlabel("degree $l$")
    a0.set_ylabel("relative error vs pyslfp")
    a0.set_title("five formulations, one radial reference", loc="left")
    a0.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    a0.legend(framealpha=0.95)
    y = range(len(names))[::-1]
    a1.barh(y, costs, color=[COLOURS[n] for n in names], height=0.62)
    for yi, c, it in zip(y, costs, its):
        a1.text(c * 1.15, yi, f"{c:.1f} s   ({it:.0f} its)",
                va="center", fontsize=12, color=MUTED)
    a1.set_yticks(y, names)
    a1.set_xscale("log")
    a1.set_xlim(right=max(costs) * 8)
    a1.set_xlabel("seconds per solve")
    a1.set_title("what each one costs", loc="left")
    a1.grid(axis="y", visible=False)
    save(fig, out / "methods.png")


def cmb_figure(case: Path, out: Path) -> None:
    """Loading vs tidal cost of the CMB approximations, vs full."""
    full = love(case / "results_o2.json")
    treatments = ["nomass", "uniform", "winkler"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    width, l = 0.34, 2
    for j, (forcing, colour, label) in enumerate(
            [("load", "#2a78d6", "loading $k'_2$"),
             ("tide", "#e34948", "tidal $k_2$")]):
        vals = []
        for t in treatments:
            r = love(case / f"results_o2_{t}.json")
            vals.append(abs(r[l][forcing]["k"] - full[l][forcing]["k"])
                        / abs(full[l][forcing]["k"]))
        pos = [i + (j - 0.5) * width for i in range(len(treatments))]
        ax.bar(pos, vals, width, color=colour, label=label)
        for p, v in zip(pos, vals):
            ax.text(p, v * 1.25, f"{v * 100:.2g}%", ha="center",
                    fontsize=12, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 3.0)
    ax.set_xticks(range(len(treatments)), treatments)
    ax.set_ylabel("deviation from the full treatment")
    ax.set_title("the CMB approximations: cheap for loading, "
                 "not for tides", loc="left")
    ax.legend(framealpha=0.95)
    ax.grid(axis="x", visible=False)
    ax.text(0.01, -0.16,
            "rotational feedbacks are driven by the degree-2 tidal "
            "response — the red column is their error budget",
            transform=ax.transAxes, fontsize=12, color=MUTED)
    save(fig, out / "cmb.png")


def identity_figure(campaign: Path, out: Path) -> None:
    """The relabelled change-of-variables identity: every welded case
    strict, the gauge penalty included; the slipping interface is the
    one informational case left."""
    def measured(log: Path) -> float:
        for line in log.read_text().splitlines():
            if "u_A - u_B" in line:
                return float(line.split()[-1])
        raise SystemExit(f"no identity number in {log}")

    bars = []
    for log, label in [
            ("identity_homogeneous_h0.3.txt", "homogeneous (welded solid)"),
            ("identity_two_solid_h0.3.txt", "two solids (welded)"),
            ("identity_fluid_core_h0.3.txt",
             "fluid core (welded, gauged)")]:
        path = campaign / log
        if path.exists():
            bars.append((label, measured(path), GOOD))
    # The slipping interface, measured 30 Sep 2026 (fluid_core,
    # A = 0.02, h = 0.3): the interface constraint forms are not yet
    # certified covariant.
    bars.append(("fluid core (slip interface)", 1.2e-2, WARN))
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    y = range(len(bars))[::-1]
    ax.barh(y, [b[1] for b in bars], color=[b[2] for b in bars],
            height=0.6)
    for yi, (_, v, _c) in zip(y, bars):
        ax.text(v * 1.4, yi, f"{v:.0e}", va="center", fontsize=13,
                color=MUTED)
    ax.set_yticks(y, [b[0] for b in bars])
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 3)
    ax.set_ylim(-0.9, len(bars) - 0.4)
    ax.set_xlabel("relative difference of the two discrete solutions")
    ax.set_title("mapped assembly vs remeshed assembly: "
                 "the same linear system", loc="left")
    ax.axvline(2e-6, color=MUTED, linewidth=1.2, linestyle="--")
    ax.text(2.6e-6, -0.62, "solver / DtN floor", fontsize=12,
            color=MUTED)
    ax.grid(axis="y", visible=False)
    fig.text(0.13, -0.04,
             "green: every mapped term certified at once — the gauge "
             "penalty included\n"
             "amber: the slipping-interface forms (certification in "
             "progress; the physics is benchmarked separately)",
             fontsize=12, color=MUTED, va="top")
    save(fig, out / "identity.png")


def derivative_figure(case: Path, out: Path, eps: float = 0.02) -> None:
    """The 3-D vs 1-D derivative identity plot, referential method."""
    runs, refs = {}, {}
    for s in (eps, -eps):
        runs[s] = love(case / f"results_o2_referential_shift{s:+g}.json")
        r = json.loads((case / f"shift_{s:+g}" /
                        "reference.json").read_text())
        refs[s] = {l: {q: r[f"{q}_load"][i] for q in "hlk"}
                   for i, l in enumerate(r["degree"])}
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    lo = hi = None
    for l in sorted(set(runs[eps]) & set(runs[-eps])):
        if l == 1:
            continue
        for q in "hlk":
            if q != "h" and l < 2:
                continue
            d3 = (runs[eps][l]["load"][q] - runs[-eps][l]["load"][q]) \
                / (2 * eps)
            d1 = (refs[eps][l][q] - refs[-eps][l][q]) / (2 * eps)
            if abs(d1) < 1e-10:
                continue
            ax.plot(d1, d3, "o", color="#2a78d6", markersize=10)
            ax.annotate(f"${q}'_{l}$", (d1, d3),
                        textcoords="offset points", xytext=(9, 5),
                        fontsize=13, color=MUTED)
            lo = d1 if lo is None else min(lo, d1, d3)
            hi = d1 if hi is None else max(hi, d1, d3)
    pad = 0.1 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GRID,
            linewidth=1.6, zorder=0)
    ax.set_xlabel("perturbation theory (1-D reference)")
    ax.set_ylabel("3-D solver, fixed mesh")
    ax.set_title("d(Love) / d(interface radius)", loc="left")
    save(fig, out / "derivative.png")


def aspherical_figure(results: Path, out: Path) -> None:
    """Field errors on the independently meshed aspherical body: the
    two punchlines are that doubling the shape amplitude changes
    nothing (the mapped machinery is exact) and refining the mesh
    drops the error (it is all discretisation)."""
    # The 0.05 series lands exactly on the 0.02 one (that IS the
    # result), so it is drawn as open markers riding on the line.
    series = [("aspherical_eps0.02_o2.json",
               "shape amplitude 0.02", "#2a78d6", "o", False),
              ("aspherical_eps0.05_o2.json",
               "amplitude 0.05 — identical", "#eb6834", "s", True),
              ("aspherical_half-h_o2.json",
               "amplitude 0.02, elements halved", "#1baf7a", "^",
               False)]
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    drawn = False
    for name, label, colour, marker, open_marks in series:
        path = results / name
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        ls = [d["degree"] for d in r["degrees"]]
        style = dict(marker=marker, color=colour)
        if open_marks:
            style.update(linestyle="none", markersize=14,
                         markerfacecolor="none", markeredgewidth=2.6,
                         markeredgecolor=colour)
        ax.semilogy(ls, [d["u_error"] for d in r["degrees"]],
                    label=label, **style)
        phi = dict(style)
        if not open_marks:
            phi.update(linestyle="--", linewidth=1.6, markersize=6,
                       alpha=0.75)
        ax.semilogy(ls, [d["phi_error"] for d in r["degrees"]], **phi)
        drawn = True
    if not drawn:  # fall back on whatever single runs are there
        for path in sorted(results.glob("aspherical_*o*.json")):
            r = json.loads(path.read_text())
            ls = [d["degree"] for d in r["degrees"]]
            ax.semilogy(ls, [d["u_error"] for d in r["degrees"]],
                        marker="o", label=path.stem)
    ax.set_xlabel("degree $l$")
    ax.set_ylabel("relative $L^2$ error through the map")
    ax.set_title("aspherical body, independent mesh, exact reference",
                 loc="left")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(framealpha=0.95, title="solid: displacement,  "
              "dashed: potential", title_fontsize=12)
    save(fig, out / "aspherical.png")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path("love_numbers")
    p.add_argument("--out", type=Path, default=Path("talk"))
    p.add_argument("--methods-case", type=Path,
                   default=base / "runs_methods/fluid_core/h0.3")
    p.add_argument("--cmb-case", type=Path,
                   default=base / "runs/prem_4_cmb/h0.2")
    p.add_argument("--identity-logs", type=Path,
                   default=Path("runs_campaign/relabelling"))
    p.add_argument("--aspherical", type=Path, default=Path("talk_data"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    methods_figure(args.methods_case, args.out)
    cmb_figure(args.cmb_case, args.out)
    identity_figure(args.identity_logs, args.out)
    derivative_figure(args.methods_case, args.out)
    aspherical_figure(args.aspherical, args.out)


if __name__ == "__main__":
    main()
