"""Command line kept separate so worker thread settings precede NumPy imports."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def parser():
    p = argparse.ArgumentParser(description="Lorentzian EPRL finite-complex calculations")
    sub = p.add_subparsers(dest="command", required=True)
    estimate = sub.add_parser("estimate", help="Output-tensor sizes, without loading native libraries")
    estimate.add_argument("config")
    prepare = sub.add_parser("prepare", help="Freeze inputs and create restartable cutoff tasks")
    prepare.add_argument("config")
    prepare.add_argument("run")
    prepare.add_argument("--library", default=os.environ.get("SL2CFOAM_LIBRARY"))
    prepare.add_argument("--tables", default=os.environ.get("SL2CFOAM_TABLES"))
    worker = sub.add_parser("worker", help="Run one cutoff task (also used by Slurm)")
    worker.add_argument("run")
    task = worker.add_mutually_exclusive_group(required=True)
    task.add_argument("--dl", type=int)
    task.add_argument("--task-index", type=int)
    run = sub.add_parser("run", help="Run prepared tasks locally; logs go to RUN/logs")
    run.add_argument("run")
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--threads", type=int, default=1)
    submit = sub.add_parser("submit", help="Submit a prepared run as one Slurm array")
    submit.add_argument("run")
    submit.add_argument("--jobs", type=int, default=1)
    submit.add_argument("--threads", type=int, default=4)
    submit.add_argument("--mem", default="4G")
    submit.add_argument("--time", default="00:30:00")
    submit.add_argument("--partition")
    submit.add_argument("--account")
    submit.add_argument("--dry-run", action="store_true")
    analyze = sub.add_parser("analyze", help="CSV, plots and optional published-reference check")
    analyze.add_argument("run")
    analyze.add_argument("--allow-partial", action="store_true")
    analyze.add_argument("--reference", action="store_true")
    analyze.add_argument("--rtol", type=float, default=1e-5)
    topology = sub.add_parser("refinement-plan", help="Generate same-boundary 1->5 combinatorics")
    topology.add_argument("--steps", type=int, default=2)
    return p


def main():
    args = parser().parse_args()
    # One BLAS thread, with independent processes and OpenMP used explicitly.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from . import jobs
    try:
        if args.command == "estimate":
            print(json.dumps(jobs.estimate(jobs.load_config(args.config)), indent=2))
        elif args.command == "prepare":
            if not args.library or not args.tables:
                raise ValueError("Set SL2CFOAM_LIBRARY and SL2CFOAM_TABLES, or use --library/--tables")
            manifest = jobs.prepare(args.config, args.run, args.library, args.tables)
            print(f"Prepared {len(manifest['spec']['shells'])} cutoff tasks in {Path(args.run).resolve()}")
        elif args.command == "worker":
            dl = args.dl
            if args.task_index is not None:
                shells = jobs.read_manifest(args.run)["spec"]["shells"]
                if not 0 <= args.task_index < len(shells):
                    raise ValueError("Task index outside the prepared array")
                dl = shells[args.task_index]
            jobs.worker(args.run, dl)
        elif args.command == "run":
            run_local(args)
        elif args.command == "submit":
            submit(args)
        elif args.command == "analyze":
            from .analysis import collect
            if not 0 < args.rtol < 1:
                raise ValueError("rtol must be between 0 and 1")
            summary = collect(args.run, args.allow_partial, args.reference, args.rtol)
            print(f"Report: {Path(args.run).resolve() / 'analysis'}")
            print(summary["diagnostic"])
            if args.reference:
                print(f"Published reference: {'PASS' if summary['reference_pass'] else 'FAIL'}")
                if not summary["reference_pass"]:
                    sys.exit(1)
        else:
            from .topology import refinement_plan
            if not 0 <= args.steps <= 100:
                raise ValueError("steps must be between 0 and 100")
            print(json.dumps(refinement_plan(args.steps), indent=2))
    except (ValueError, OSError, RuntimeError, FloatingPointError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def run_local(args):
    from concurrent.futures import ThreadPoolExecutor
    from .jobs import read_manifest
    if args.jobs < 1 or args.threads < 1:
        raise ValueError("jobs and threads must be positive")
    run = Path(args.run).resolve()
    manifest = read_manifest(run)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))

    def launch(dl):
        log = run / "logs" / f"dl_{dl:03d}.log"
        with log.open("a") as stream:
            result = subprocess.run([sys.executable, "-m", "qlg", "worker", str(run), "--dl", str(dl)],
                                    stdout=stream, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            raise RuntimeError(f"DL={dl} failed; see {log}")
        print(f"DL={dl}: complete", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(launch, manifest["spec"]["shells"]))


def submit(args):
    from .jobs import read_manifest
    if args.jobs < 1 or args.threads < 1:
        raise ValueError("jobs and threads must be positive")
    run = Path(args.run).resolve()
    manifest = read_manifest(run)
    # Use this interpreter, preserving a virtual environment across batch nodes.
    command = shlex.join([sys.executable, "-m", "qlg", "worker", str(run)])
    script = run / "job.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                      f"export OMP_NUM_THREADS={args.threads}\n"
                      "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
                      f'exec {command} --task-index "$SLURM_ARRAY_TASK_ID"\n')
    count = len(manifest["spec"]["shells"])
    cmd = ["sbatch", "--parsable", f"--array=0-{count - 1}%{args.jobs}",
           "--nodes=1", "--ntasks=1", f"--cpus-per-task={args.threads}",
           f"--mem={args.mem}", f"--time={args.time}",
           f"--chdir={Path.cwd()}", f"--output={run / 'logs/slurm-%A_%a.out'}"]
    for key in ("partition", "account"):
        if getattr(args, key):
            cmd.append(f"--{key}={getattr(args, key)}")
    cmd.append(str(script))
    print(shlex.join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
