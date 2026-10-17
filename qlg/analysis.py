"""Cutoff diagnostics. Successive differences are diagnostics, not error bounds."""

import csv
from importlib.resources import files
import json
import math
from pathlib import Path

import numpy as np

from .jobs import atomic_json, read_manifest


def relative_difference(value, previous):
    if value == 0:
        return 0.0 if previous == 0 else None
    return abs(value - previous) / abs(value)


def reference_value(spec, dl, name):
    if spec["model"] != "delta4" or spec["gamma"] != 1.2 or spec["two_j"] != 2:
        raise ValueError("Published table requires Delta4, gamma=1.2 and j=1")
    state = next(b for b in spec["boundary"] if b["name"] == name)
    labels = state["two_i"]
    if len(set(labels)) != 1 or labels[0] not in (0, 2, 4) or not 0 <= dl <= 25:
        raise ValueError("Published reference supports uniform i=0,1,2 and DL=0,...,25")
    table = np.loadtxt(files("qlg").joinpath("reference/delta4_gamma1.2_j1.tsv"))
    return float(table[dl, labels[0] // 2])


def collect(run_path, allow_partial=False, reference=False, rtol=1e-5):
    run = Path(run_path).resolve()
    manifest = read_manifest(run)
    spec = manifest["spec"]
    results, missing = [], []
    for dl in spec["shells"]:
        path = run / f"dl_{dl:03d}" / "result.json"
        if not path.exists():
            missing.append(dl)
            continue
        result = json.loads(path.read_text())
        if result["run_id"] != manifest["run_id"] or result["dl"] != dl:
            raise ValueError(f"Incompatible result: {path}")
        if not result["amplitudes"] or not all(math.isfinite(v) for v in result["amplitudes"].values()):
            raise ValueError(f"Invalid amplitudes: {path}")
        results.append(result)
    if missing and not allow_partial:
        raise ValueError(f"Missing cutoff tasks {missing}; use --allow-partial for an interim report")
    if not results:
        raise ValueError("No completed cutoff tasks")
    if any(set(r["amplitudes"]) != set(results[0]["amplitudes"]) for r in results):
        raise ValueError("Boundary-state sets differ between results")
    rows, last = [], {}
    for result in results:
        dl = result["dl"]
        for name, value in result["amplitudes"].items():
            previous = last.get(name)
            row = {"dl": dl, "boundary": name, "amplitude": value,
                   "previous_dl": previous[0] if previous else None,
                   "absolute_change": abs(value - previous[1]) if previous else None,
                   "relative_change": relative_difference(value, previous[1]) if previous else None,
                   "wall_seconds": result["wall_seconds"], "max_rss_kb": result["max_rss_kb"],
                   "bulk_sector_cancellation": result["bulk_sector_cancellation"][name]}
            if reference:
                expected = reference_value(spec, dl, name)
                error = relative_difference(expected, value)
                row.update(reference=expected, reference_relative_error=error,
                           reference_pass=error is not None and error <= rtol)
            rows.append(row)
            last[name] = (dl, value)
    summary = {"run_id": manifest["run_id"], "complete": not missing,
               "missing_shells": missing, "computed_shells": [r["dl"] for r in results],
               "diagnostic": "Auxiliary-shell cutoff convergence on a fixed complex; not refinement convergence.",
               "reference_pass": all(r["reference_pass"] for r in rows) if reference else None,
               "reference_rtol": rtol if reference else None, "rows": rows}
    report = run / "analysis"
    report.mkdir(exist_ok=True)
    atomic_json(report / "summary.json", summary)
    with (report / "amplitudes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, report)
    return summary


def plot(rows, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for name in dict.fromkeys(r["boundary"] for r in rows):
        group = [r for r in rows if r["boundary"] == name]
        axes[0].plot([r["dl"] for r in group], [r["amplitude"] for r in group], "o-", label=name)
        positive = [r for r in group if r["relative_change"] is not None and r["relative_change"] > 0]
        if positive:
            axes[1].semilogy([r["dl"] for r in positive], [r["relative_change"] for r in positive], "o-")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set_ylabel("Signed amplitude")
    axes[0].legend()
    axes[1].set_ylabel("Relative change from previous computed cutoff")
    for ax in axes:
        ax.set_xlabel(r"Auxiliary-shell cutoff $\Delta l$")
        ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(report / f"cutoff_convergence.{suffix}", dpi=160)
    plt.close(fig)
