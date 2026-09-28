"""Run the Love-number benchmark over element sizes and orders.

For one model, each element size `--h` is a case (made with make_case.py
if its directory does not hold one) and each order `--order` a run of
love_benchmark on it under MPI:

  <out>/<model>/h<h>/                 the case: mesh, fields, reference
  <out>/<model>/h<h>/results_o<p>.json
  <out>/<model>/h<h>/log_o<p>.txt

A run whose results file exists is skipped unless `--force` is given, so a
sweep can be extended or resumed. plot.py reads the tree.

    python run.py homogeneous --h 0.3 0.2 0.15 --order 2 --np 8
    python run.py inner_core --h 0.1 0.07 --order 2 3 --np 128 --lmax 8

The launcher is `--mpiexec`, else the environment's MPIEXEC, else
`mpiexec`; it must belong to the MPI the program was built with.
`--launcher-args` are passed to it before the program (binding, host
files). `--dry-run` prints the commands without running anything.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import models

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent.parent


def find_program(given: Path | None) -> Path:
    """love_benchmark: the one given, or the one of a build tree of the
    repository."""
    if given is not None:
        return given.resolve()
    for build in sorted(REPOSITORY.glob("build*")):
        candidate = build / "benchmarks" / "love_benchmark"
        if candidate.exists():
            return candidate
    raise SystemExit("love_benchmark not found in a build tree of the "
                     "repository (configure with -DUSE_MPI=ON "
                     "-DBUILD_BENCHMARKS=ON), and no --program given")


def run(command: list[str], *, log: Path | None, dry_run: bool) -> bool:
    """Run a command, its output to `log` and the terminal; True on
    success."""
    print("  $ " + shlex.join(command), flush=True)
    if dry_run:
        return True
    start = time.time()
    with subprocess.Popen(command, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as process:
        lines = []
        for line in process.stdout:
            lines.append(line)
            sys.stdout.write("    " + line)
        process.wait()
    if log is not None:
        log.write_text("$ " + shlex.join(command) + "\n" + "".join(lines))
    print(f"  ({time.time() - start:.1f} s, exit {process.returncode})",
          flush=True)
    return process.returncode == 0


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=sorted(models.MODELS))
    p.add_argument("--h", type=float, nargs="+", default=[0.2],
                   help="element sizes on the interfaces, one case each")
    p.add_argument("--order", type=int, nargs="+", default=[2],
                   help="finite element orders, one run each per case")
    p.add_argument("--np", type=int, default=os.cpu_count() // 2 or 1,
                   help="number of MPI ranks")
    p.add_argument("--lmax", type=int, default=6, help="highest degree")
    p.add_argument("--lmin", type=int, default=0, help="lowest degree")
    p.add_argument("--dtn-degree", type=int, default=16)
    p.add_argument("--rel-tol", type=float, default=1e-10)
    p.add_argument("--solver", type=int, default=1, choices=(0, 1),
                   help="0: Schur-complement CG, 1: block MINRES")
    p.add_argument("--buffer", type=float, default=0.2,
                   help="thickness of the buffer shell over the radius")
    p.add_argument("--angular", type=float, default=0.4,
                   help="largest element size on an interface over its radius")
    p.add_argument("--out", type=Path, default=Path("runs"), metavar="DIR",
                   help="root of the results tree")
    p.add_argument("--program", type=Path, default=None,
                   help="love_benchmark (default: found in a build tree)")
    p.add_argument("--mpiexec", default=os.environ.get("MPIEXEC", "mpiexec"),
                   help="the MPI launcher")
    p.add_argument("--launcher-args", default="",
                   help="further arguments of the launcher, as one string")
    p.add_argument("--program-args", default="",
                   help="further arguments of love_benchmark, as one string")
    p.add_argument("--force", action="store_true",
                   help="run again where results exist, and rebuild cases")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running them")
    args = p.parse_args()

    program = Path("love_benchmark") if args.dry_run and args.program is None \
        else find_program(args.program)
    failures = []
    for h in args.h:
        case = args.out / args.model / f"h{h:g}"
        print(f"{args.model}, h = {h:g}: {case}", flush=True)
        if args.force or not (case / "case.json").exists():
            ok = run([sys.executable, str(HERE / "make_case.py"), args.model,
                      "--h", f"{h:g}", "--buffer", f"{args.buffer:g}",
                      "--angular", f"{args.angular:g}",
                      "--lmax", str(max(args.lmax, 10)), "--out", str(case)],
                     log=None, dry_run=args.dry_run)
            if not ok:
                failures.append(f"{case}: make_case.py")
                continue
        for order in args.order:
            results = case / f"results_o{order}.json"
            if results.exists() and not args.force:
                print(f"  order {order}: {results} exists, skipped")
                continue
            command = [args.mpiexec, "-np", str(args.np),
                       *shlex.split(args.launcher_args), str(program),
                       "-c", str(case / "case.json"), "-o", str(order),
                       "-lmin", str(args.lmin), "-lmax", str(args.lmax),
                       "-deg", str(max(args.dtn_degree, args.lmax)),
                       "-rt", f"{args.rel_tol:g}", "-s", str(args.solver),
                       "-out", str(results), *shlex.split(args.program_args)]
            ok = run(command, log=case / f"log_o{order}.txt",
                     dry_run=args.dry_run)
            if not ok:
                results.unlink(missing_ok=True)
                failures.append(f"{case}: order {order}")
    if failures:
        raise SystemExit("failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
