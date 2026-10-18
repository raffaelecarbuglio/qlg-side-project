"""Audit the equal-j boundary probe across edgewise refinements.

This tests an additive area on each inherited coarse triangle. It is not a
boundary-state embedding: shape, correlations and effective dynamics remain
unspecified. Areas use units in which a_j=sqrt(j*(j+1)).
"""

import argparse
from itertools import combinations
import math
from pathlib import Path

import numpy as np

from qlg.jobs import atomic_json
from qlg.backend import digest
from qlg.topology import boundary, edgewise_simplex


def audit(divisions):
    vertices, cells = edgewise_simplex(divisions)
    outside = boundary(cells)
    triangles = sorted({f for t in outside for f in combinations(t, 3)})
    coarse = []
    for axes in combinations(range(5), 3):
        pieces = [f for f in triangles if all(vertices[v][a] == 0
                  for v in f for a in range(5) if a not in axes)]
        areas = []
        for f in pieces:
            p = np.array([vertices[v] for v in f], dtype=float) / divisions
            edges = p[1:] - p[0]
            areas.append(math.sqrt(np.linalg.det(edges @ edges.T)) / 2)
        assert len(pieces) == divisions ** 2
        np.testing.assert_allclose(areas, math.sqrt(3) / (2 * divisions ** 2), rtol=1e-13)
        np.testing.assert_allclose(sum(areas), math.sqrt(3) / 2, rtol=1e-13)
        coarse.append(dict(coarse_triangle=list(axes), fine_triangles=len(pieces),
                           auxiliary_area_sum=sum(areas)))
    return dict(divisions=divisions, four_simplices=len(cells), boundary_tetrahedra=len(outside),
                boundary_triangles=len(triangles), inherited_triangles=coarse,
                equal_j1_additive_area_per_coarse_triangle=divisions ** 2 * math.sqrt(2),
                ratio_to_coarse_j1_area=divisions ** 2,
                target_equal_piece_area_for_coarse_j1=math.sqrt(2) / divisions ** 2,
                minimum_nonzero_piece_area=math.sqrt(3) / 2,
                all_pieces_nonzero_can_match_coarse_j1=divisions ** 2 * math.sqrt(3) / 2 <= math.sqrt(2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = dict(source_sha256={p: digest(Path(__file__).parents[1] / p) for p in
                                ("benchmarks/check_boundary.py", "qlg/topology.py")},
                  spectrum_reference="Rovelli, arXiv:1102.3660v5, equations 37-38",
                  area_units="common 8*pi*gamma*l_P^2 factor omitted",
                  interpretation="Additive geometric-area audit, not a complete boundary embedding",
                  results=[audit(q) for q in (1, 2, 4, 8)])
    atomic_json(Path(args.output), report)
    for row in report["results"]:
        print(f"q={row['divisions']}: {row['four_simplices']} cells; fixed-j1 coarse-area "
              f"ratio={row['ratio_to_coarse_j1_area']}")


if __name__ == "__main__":
    main()
