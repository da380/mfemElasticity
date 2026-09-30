"""Figures for the perturbation family: the interface-shift check.

Reads what perturbation_check leaves behind — the shifted references
`shift_<+-eps>/reference.json` and the runs
`results_o<p>_<method>_shift<+-eps>.json`, with the base run beside the
case — and draws one figure per method into the results directory,
`perturbation_<method>_o<p>.png`:

  left    the absolute agreement with pyslfp by degree, one line per
          eps: flat across the ladder means the mapped solve tracks the
          perturbed models as well as the unmapped solve tracks the
          base one;
  right   the derivative check as an identity plot — the fixed-mesh
          central difference of the 3-D runs against the same
          difference of the 1-D references, one point per degree and
          Love number, agreement meaning the point sits on the line.

    python plot.py <case> --out <root> --method referential slip_broken
    python plot.py runs/fluid_core/h0.3     # results beside the case

The comparison rules are perturbation_check's: h' everywhere, l' and k'
from degree two, degree one skipped, near-zero references and
derivatives left out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))

from perturbation_check import love_from, love_reference  # noqa: E402

COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def compared(l: int) -> list[str]:
    return [q for q in "hlk" if q == "h" or l >= 2]


def gather(case: Path, root: Path, method: str, order: int):
    """eps -> (run values, reference values), from the files present."""
    out = {}
    for f in sorted(root.glob(f"results_o{order}_{method}_shift*.json")):
        eps = float(f.stem.split("shift")[1])
        out[eps] = (love_from(f),
                    love_reference(root / f"shift_{eps:+g}" /
                                   "reference.json"))
    base = case / f"results_o{order}_{method}.json"
    if not base.exists():
        base = root / f"results_o{order}_{method}.json"
    if base.exists() and out:
        out[0.0] = (love_from(base), love_reference(case /
                                                    "reference.json"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case", type=Path,
                   help="the case directory, runs/<model>/h<h>")
    p.add_argument("--out", type=Path, default=None,
                   help="where perturbation_check wrote (default: the "
                        "case directory)")
    p.add_argument("--method", nargs="+",
                   default=["referential", "slip_broken"])
    p.add_argument("--order", type=int, default=2)
    args = p.parse_args()
    root = args.out if args.out is not None else args.case

    drawn = 0
    for method in args.method:
        runs = gather(args.case, root, method, args.order)
        if len(runs) < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2),
                                 facecolor=SURFACE)

        # Absolute agreement by degree, one line per eps.
        for i, eps in enumerate(sorted(runs)):
            values, refs = runs[eps]
            ls, errs = [], []
            for l in sorted(set(values) & set(refs)):
                if l == 1:
                    continue
                worst = 0.0
                for q in compared(l):
                    if abs(refs[l][q]) > 1e-10:
                        worst = max(worst, abs(values[l][q] - refs[l][q])
                                    / abs(refs[l][q]))
                ls.append(l)
                errs.append(worst)
            axes[0].semilogy(ls, errs, color=COLOURS[i % 8],
                             marker=MARKERS[i % 8], markersize=4.5,
                             linewidth=1.2, label=f"eps = {eps:+g}")

        # The derivative identity: 3-D central difference against 1-D.
        pairs = sorted({abs(e) for e in runs if e != 0.0
                        and -abs(e) in runs and abs(e) in runs})
        lo = hi = None
        for i, a in enumerate(pairs):
            plus, minus = runs[a], runs[-a]
            labelled = False
            for l in sorted(set(plus[0]) & set(minus[0])):
                if l == 1:
                    continue
                for q in compared(l):
                    d3 = (plus[0][l][q] - minus[0][l][q]) / (2 * a)
                    d1 = (plus[1][l][q] - minus[1][l][q]) / (2 * a)
                    if abs(d1) < 1e-10:
                        continue
                    axes[1].plot(d1, d3, linestyle="none",
                                 marker=MARKERS[i % 8], markersize=6,
                                 color=COLOURS[i % 8],
                                 label=None if labelled
                                 else f"eps = {a:g}")
                    labelled = True
                    axes[1].annotate(f"$ {q}'_{l}$", (d1, d3),
                                     textcoords="offset points",
                                     xytext=(5, 3), fontsize=7,
                                     color=MUTED)
                    lo = d1 if lo is None else min(lo, d1, d3)
                    hi = d1 if hi is None else max(hi, d1, d3)
        if lo is not None:
            pad = 0.08 * (hi - lo or 1.0)
            axes[1].plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                         color=GRID, linewidth=1.0, zorder=0)

        axes[0].set_title("agreement with pyslfp by degree", loc="left",
                          fontsize=10, color=INK)
        axes[0].set_xlabel("degree $l$", color=INK)
        axes[1].set_title("d(Love)/d($r_{interface}$): 3-D against 1-D",
                          loc="left", fontsize=10, color=INK)
        axes[1].set_xlabel("1-D central difference", color=INK)
        axes[1].set_ylabel("3-D central difference", color=INK)
        for axis in axes:
            axis.grid(True, color=GRID, linewidth=0.7)
            axis.set_facecolor(SURFACE)
            axis.tick_params(colors=MUTED)
            axis.legend(fontsize=8, framealpha=0.9)
        fig.suptitle(f"interface shift, {method} (order {args.order}, "
                     f"{args.case.parent.name})", x=0.01, ha="left",
                     fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = root / f"perturbation_{method}_o{args.order}.png"
        fig.savefig(out, dpi=180)
        print(f"wrote {out}")
        drawn += 1
    if not drawn:
        raise SystemExit(f"no shifted results for {args.method} under "
                         f"{root} (run perturbation_check first)")


if __name__ == "__main__":
    main()
