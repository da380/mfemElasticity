"""The whole benchmark campaign as one script, with settable defaults.

Collates every family of benchmarks/ — the Love-number cross-method
sweep with its CMB conditions and field benchmark, the relabelling
family (mapped runs, the solver-level identity, the independently
meshed aspherical body), the perturbation family (the degree-0
interface-shift check) and a weak-scaling study — behind two profiles:

  --profile local    a rehearsal on this machine: small meshes, few
                     degrees, 8 ranks; runs end to end in around an
                     hour and exercises every stage.
  --profile server   the production shape: the full model set, an
                     h-ladder at orders 2 and 3, higher degrees, 100
                     ranks on a 128-core shared-memory machine, plus
                     the weak-scaling stage.

Every profile default is a flag; --stages picks a subset; everything
delegates to the family scripts and drivers, all of which skip work
whose results exist, so an interrupted campaign resumes where it
stopped. The tree mirrors the families:

  <out>/love_numbers/<model>/h<h>/   cases and Love/field results
                     <model>/*.png   the family's figures (plot.py)
  <out>/relabelling/                 identity logs, aspherical results
                                     and aspherical.png
  <out>/perturbation/<model>/        shifted profiles, references, runs
                                     and perturbation_*.png
  <out>/scaling/                     the weak-scaling rungs
  <out>/campaign_log.md              every command and stage outcome

plot.py renders the Love tree at the end and the scaling rungs collate
into scaling_summary.md.

    ./campaign --profile local
    ./campaign --profile server --out /scratch/love
    ./campaign --profile server --stages methods field --lmax 20
    ./campaign --profile server --stages scaling --dry-run

Method-model compatibility is encoded here: the slipping methods take a
single fluid core, the mapped stages the referential family, and the
solver-level identity the gauge-free solid models, where it is strict.
Mind doc/gauged_fluid.md when reading converged cross-method ladders on
models whose core is not neutrally stratified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

from drivers import find_programs, run  # noqa: E402

#: Which methods and stages run on which models, beyond the Eulerian
#: pair, which run on everything.
REFERENTIAL_MODELS = ("homogeneous", "two_solid", "linear_solid",
                      "fluid_core", "inner_core", "stratified_core",
                      "earth_like", "prem_4")
SLIP_MODELS = ("fluid_core",)
SOLID_MODELS = ("homogeneous", "two_solid", "linear_solid")
MAPPED_MODELS = ("homogeneous", "two_solid", "linear_solid", "fluid_core")
# Strict for every welded case since the covariant gauge penalty:
# solids and the gauged fluid core alike.
IDENTITY_MODELS = ("homogeneous", "two_solid", "linear_solid",
                   "fluid_core")
ASPHERICAL_MODELS = ("homogeneous", "linear_solid")
# The shift legs run the referential method only: an OUTWARD interface
# shift through the slipping machinery leaves the AL constraint system
# near-singular (stiffening theta amplifies the failure; the inward leg
# and the fixed-interface mapped runs are healthy) — a parked
# formulation question, see perturbation/README.md.
PERTURBATION = {"fluid_core": ("referential",),
                "two_solid": ("referential",)}

PROFILES = {
    "local": dict(models=["homogeneous", "two_solid", "fluid_core"],
                  h=[0.3], orders=[2], lmax=4, dtn_degree=16, np=8,
                  solvers=[1], cmb=["full", "nomass", "uniform", "winkler"],
                  map_amplitude=0.02,
                  shift_eps=[0.02], pert_lmax=3, field=True, scaling=[],
                  aspherical_scale=1.0, aspherical_lmax=4),
    "server": dict(models=["homogeneous", "two_solid", "linear_solid",
                           "fluid_core", "inner_core", "stratified_core",
                           "earth_like", "prem_4"],
                   h=[0.2, 0.15, 0.1], orders=[2, 3], lmax=16,
                   dtn_degree=24, np=100, solvers=[1, 0],
                   cmb=["full", "nomass", "uniform", "winkler"],
                   map_amplitude=0.02, shift_eps=[0.01, 0.02],
                   pert_lmax=4, field=True,
                   # weak scaling: h ~ np^(-1/3), roughly constant
                   # unknowns per rank up the ladder
                   scaling=[(12, 0.2), (25, 0.157), (50, 0.125),
                            (100, 0.099)],
                   aspherical_scale=0.5, aspherical_lmax=8),
}

STAGES = ("methods", "cmb", "field", "mapped", "identity", "aspherical",
          "perturbation", "scaling", "plot")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=sorted(PROFILES), default="local")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=None,
                   help="stages to run (default: all of the profile's)")
    p.add_argument("--out", type=Path, default=Path("runs_campaign"),
                   metavar="DIR", help="root of the results tree")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--h", type=float, nargs="+", default=None)
    p.add_argument("--orders", type=int, nargs="+", default=None)
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--dtn-degree", type=int, default=None)
    p.add_argument("--np", type=int, default=None)
    p.add_argument("--solvers", type=int, nargs="+", default=None)
    p.add_argument("--cmb", nargs="+", default=None)
    p.add_argument("--map-amplitude", type=float, default=None)
    p.add_argument("--shift-eps", type=float, nargs="+", default=None)
    p.add_argument("--partition", action="store_true",
                   help="partition each case for the ranks, so that no "
                        "rank reads the whole mesh (large runs)")
    p.add_argument("--programs", type=Path, default=None)
    p.add_argument("--mpiexec", default="mpiexec")
    p.add_argument("--dry-run", action="store_true",
                   help="print every command without running anything")
    args = p.parse_args()

    prof = dict(PROFILES[args.profile])
    overrides = {"models": args.models, "h": args.h, "orders": args.orders,
                 "lmax": args.lmax, "dtn_degree": args.dtn_degree,
                 "np": args.np, "solvers": args.solvers, "cmb": args.cmb,
                 "map_amplitude": args.map_amplitude,
                 "shift_eps": args.shift_eps}
    prof.update({k: v for k, v in overrides.items() if v is not None})
    prof["dtn_degree"] = max(prof["dtn_degree"], prof["lmax"])
    stages = args.stages or list(STAGES)

    programs = Path(".") if args.dry_run and args.programs is None \
        else find_programs(args.programs)
    py, mpiexec, out = sys.executable, args.mpiexec, args.out
    loves = out / "love_numbers"
    if not args.dry_run:
        for d in (loves, out / "relabelling", out / "perturbation"):
            d.mkdir(parents=True, exist_ok=True)
    log: list[str] = [f"# campaign --profile {args.profile}"
                      f" --stages {' '.join(stages)}"
                      f"  ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    failures: list[str] = []
    dry = ["--dry-run"] if args.dry_run else []

    def stage(name: str, ok: bool) -> None:
        log.append(f"[{name}] {'ok' if ok else 'FAILED'}")
        if not ok:
            failures.append(name)

    def sh(command: list, *, log_file: Path | None = None) -> bool:
        return run([str(c) for c in command], log=log_file,
                   dry_run=args.dry_run)

    def run_py(family: str, script: str, *extra) -> bool:
        # The family scripts have --dry-run of their own, so under
        # --dry-run they still run and print their commands.
        return run([py, str(HERE / family / f"{script}.py"),
                    *(str(e) for e in extra), "--programs", str(programs),
                    "--mpiexec", mpiexec, *dry], dry_run=False)

    common = ["--h", *prof["h"], "--order", *prof["orders"],
              "--np", prof["np"], "--lmax", prof["lmax"],
              "--dtn-degree", prof["dtn_degree"], "--out", loves,
              *(["--partition"] if args.partition else [])]

    # 1. The cross-method sweep, with the Eulerian solver axis.
    if "methods" in stages:
        for model in prof["models"]:
            methods = ["dahlen", "gauged"]
            if model in REFERENTIAL_MODELS:
                methods.append("referential")
            if model in SLIP_MODELS:
                methods += ["slip", "slip_broken"]
            stage(f"methods:{model}", run_py(
                "love_numbers", "run", model, *common,
                "--method", *methods, "--solver", *prof["solvers"]))

    # 2. The approximate CMB conditions (Dahlen path, fluid models):
    # cost against accuracy, collated by cmb_report.py in the plot stage.
    if "cmb" in stages:
        for model in prof["models"]:
            if model in SOLID_MODELS:
                continue  # no fluid layer, nothing to approximate
            stage(f"cmb:{model}", run_py(
                "love_numbers", "run", model, *common,
                "--cmb", *prof["cmb"]))

    # 3. The field benchmark (Eulerian pair); off in the local profile
    # unless asked for by name.
    if "field" in stages and (prof["field"] or args.stages):
        for model in prof["models"]:
            stage(f"field:{model}", run_py(
                "love_numbers", "run", model, *common, "--field-only",
                "--method", "dahlen", "gauged"))

    # 4. The relabelled mapped runs: the same spherical physics from
    # laterally mapped coordinates, against the same references.
    if "mapped" in stages:
        for model in prof["models"]:
            if model not in MAPPED_MODELS:
                continue
            methods = ["referential"]
            if model in SLIP_MODELS:
                methods.append("slip_broken")
            stage(f"mapped:{model}", run_py(
                "love_numbers", "run", model, *common,
                "--method", *methods, "--map", prof["map_amplitude"]))

    # 5. The solver-level change-of-variables identity, strict on the
    # gauge-free solid models.
    if "identity" in stages:
        for model in prof["models"]:
            if model not in IDENTITY_MODELS:
                continue
            for h in prof["h"]:
                case = loves / model / f"h{h:g}" / "case.json"
                if not case.exists() and not args.dry_run:
                    stage(f"identity:{model}:h{h:g}", False)
                    continue
                stage(f"identity:{model}:h{h:g}", sh(
                    [mpiexec, "-np", prof["np"],
                     programs / "relabelled_identity", "-c", case,
                     "-o", max(prof["orders"]),
                     "-map", prof["map_amplitude"]],
                    log_file=out / "relabelling" /
                    f"identity_{model}_h{h:g}.txt"))

    # 6. The independently meshed aspherical body (single solid layer).
    # The stock mesh serves at scale one; a finer profile regenerates it.
    if "aspherical" in stages:
        scale = prof["aspherical_scale"]
        if scale == 1.0:
            mesh = programs.parent.parent / "data" / \
                "aspherical_buffer_3d.mesh"
        else:
            mesh = out / "relabelling" / \
                f"aspherical_buffer_3d_s{scale:g}.mesh"
            if not mesh.exists():
                stage("aspherical:mesh", sh(
                    [py, HERE.parent / "meshes" / "aspherical_body.py",
                     "--dim", 3, "--buffer", "--scale", scale,
                     "--name", f"_s{scale:g}",
                     "--out", out / "relabelling"]))
        for model in prof["models"]:
            if model not in ASPHERICAL_MODELS:
                continue
            h = min(prof["h"])
            case = loves / model / f"h{h:g}" / "case.json"
            if not case.exists() and not args.dry_run:
                stage(f"aspherical:{model}", False)
                continue
            stage(f"aspherical:{model}", sh(
                [mpiexec, "-np", prof["np"],
                 programs / "aspherical_reference", "-m", mesh,
                 "-c", case, "-o", max(prof["orders"]),
                 "-lmax", prof["aspherical_lmax"],
                 "-out", out / "relabelling" /
                 f"aspherical_{model}_o{max(prof['orders'])}.json"],
                log_file=out / "relabelling" /
                f"aspherical_{model}.txt"))

    # 7. The degree-0 interface-shift perturbation check. The base runs
    # first through run.py (a no-op when the methods stage covered it).
    if "perturbation" in stages:
        for model, methods in PERTURBATION.items():
            if model not in prof["models"]:
                continue
            h = max(prof["h"])
            ok = run_py("love_numbers", "run", model, "--h", h,
                        "--order", max(prof["orders"]),
                        "--np", prof["np"], "--lmax", prof["pert_lmax"],
                        "--dtn-degree", prof["dtn_degree"],
                        "--out", loves, "--method", *methods)
            stage(f"perturbation:{model}", ok and run_py(
                "perturbation", "perturbation_check",
                loves / model / f"h{h:g}",
                "--out", out / "perturbation" / model,
                "--eps", *prof["shift_eps"], "--method", *methods,
                "--order", max(prof["orders"]),
                "--lmax", prof["pert_lmax"], "--np", prof["np"]))

    # 8. Weak scaling: a couple of degrees per rung, the timings land in
    # the results files and the summary below.
    if "scaling" in stages and prof["scaling"]:
        for np_s, h_s in prof["scaling"]:
            stage(f"scaling:np{np_s}", run_py(
                "love_numbers", "run", "fluid_core", "--h", h_s,
                "--order", max(prof["orders"]), "--np", np_s,
                "--lmax", 2, "--dtn-degree", prof["dtn_degree"],
                "--out", out / "scaling",
                *(["--partition"] if args.partition else [])))

    # 9. Plots and the summaries, each skipped quietly when its stage
    # left nothing to draw.
    if "plot" in stages and not args.dry_run:
        stage("plot", sh([py, HERE / "love_numbers" / "plot.py", loves]))
        if "cmb" in stages:
            stage("cmb_report", sh(
                [py, HERE / "love_numbers" / "cmb_report.py", loves]))
        if any((out / "relabelling").glob("aspherical*.json")):
            stage("plot:relabelling", sh(
                [py, HERE / "relabelling" / "plot.py",
                 out / "relabelling"]))
        for model in PERTURBATION:
            root = out / "perturbation" / model
            if model in prof["models"] and any(root.glob("results_*")):
                stage(f"plot:perturbation:{model}", sh(
                    [py, HERE / "perturbation" / "plot.py",
                     loves / model / f"h{max(prof['h']):g}",
                     "--out", root, "--order", str(max(prof["orders"])),
                     "--method", *PERTURBATION[model]]))
        rows = ["| ranks | h | elements | unknowns | setup s | "
                "s / solve | iterations |",
                "|---|---|---|---|---|---|---|"]
        for np_s, h_s in prof.get("scaling", []):
            path = (out / "scaling" / "fluid_core" / f"h{h_s:g}" /
                    f"results_o{max(prof['orders'])}.json")
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            loads = [d["load"] for d in r["degrees"] if "load" in d]
            rows.append(
                f"| {r['ranks']} | {h_s:g} | {r['elements']} | "
                f"{r['displacement_unknowns'] + r['potential_unknowns']} "
                f"| {r['setup_seconds']:.1f} | "
                f"{sum(d['seconds'] for d in loads) / len(loads):.2f} | "
                f"{sum(d['outer_iterations'] for d in loads) / len(loads):.0f} |")
        if len(rows) > 2:
            (out / "scaling_summary.md").write_text("\n".join(rows) + "\n")
            log.append("[scaling] summary in scaling_summary.md")

    log += ["", ("FAILED: " + ", ".join(failures)) if failures
            else "all stages ok", ""]
    if not args.dry_run:
        # Appended, not overwritten: an interrupted campaign resumes,
        # and the log keeps every invocation's record.
        with (out / "campaign_log.md").open("a") as f:
            f.write("\n".join(log) + "\n")
        print(f"\n{log[-2]}\nlog: {out / 'campaign_log.md'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
