"""Exact finite tensor contractions; no importance-sampling assumption."""

import math

import numpy as np


def ring_amplitude(matrices):
    """Sum M0[a,b] M1[b,c] M2[c,d] M3[d,a], keeping signed amplitudes."""
    if len(matrices) != 4:
        raise ValueError("The Delta4 ring has four vertices")
    product = np.asarray(matrices[0], dtype=np.longdouble)
    for matrix in matrices[1:]:
        product = product @ np.asarray(matrix, dtype=np.longdouble)
    return np.trace(product)


def delta4_sector(tensor, boundary_indices):
    """Twelve boundary indices: three per vertex, in raw axes (0,3,4).

    All four internal intertwiners are summed, not fixed or restricted.
    Homogeneous boundary spins permit reuse of one native vertex tensor.
    """
    if len(boundary_indices) != 12:
        raise ValueError("Delta4 needs twelve boundary intertwiner indices")
    matrices = []
    for k in range(0, 12, 3):
        a, b, c = boundary_indices[k:k + 3]
        matrices.append(tensor[a, :, :, b, c])
    return ring_amplitude(matrices)


def signed_sum(values):
    values = list(values)
    total = math.fsum(values)
    absolute = math.fsum(abs(x) for x in values)
    # This measures cancellations between bulk-spin sectors only.
    cancellation = abs(total) / absolute if absolute else None
    return total, cancellation
