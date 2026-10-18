"""Plot measured full K=3 amplitude ratios, without fitting a shell tail."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-dl", type=int, default=4)
    args = parser.parse_args()
    points = {}
    for filename in args.reports:
        for row in json.loads(Path(filename).read_text())["results"]:
            if row.get("status") != "completed" or "states" not in row or row["K"] != 3:
                continue
            ratios = [s["sign"] / row["states"][0]["sign"] * math.exp(
                s["log_absolute_amplitude"] - row["states"][0]["log_absolute_amplitude"])
                      for s in row["states"][1:]]
            if row["dl"] in points:
                assert all(math.isclose(a, b, rel_tol=1e-10) for a, b in zip(ratios, points[row["dl"]]))
            points[row["dl"]] = ratios
    shells = sorted(dl for dl in points if dl >= args.minimum_dl)
    if not shells:
        parser.error("No completed K=3 three-state cases in the selected range")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
    for k, (axis, color, label) in enumerate(zip(axes, ("#245d9c", "#bd5a20"),
            (r"$|A_1/A_0|\,\times 10^6$", r"$A_2/A_0\,\times 10^6$"))):
        axis.plot(shells, [abs(points[dl][k]) * 1e6 for dl in shells], "o-", color=color)
        axis.set(xlabel=r"Auxiliary shell cutoff $\Delta l$", ylabel=label, xticks=shells)
        axis.grid(alpha=.2)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "qlg-side-project", "Date": None} if output.suffix == ".svg" else None
    fig.savefig(output, metadata=metadata)
    if output.suffix == ".svg":
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
