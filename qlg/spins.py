"""All quantum labels are integers representing twice the physical spin."""

from math import prod

# Library order: j12,j13,j14,j15,j23,j24,j25,j34,j35,j45.
# Ordered legs of intertwiners i1,...,i5, matching upstream conventions.
LEGS = ((0, 1, 2, 3), (4, 5, 6, 0), (7, 8, 1, 4),
        (9, 2, 5, 7), (3, 6, 8, 9))
AXIS_ORDER = (5, 4, 3, 2, 1)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def intertwiner_range(a, b, c, d):
    """Intersection of the (a,b) and (c,d) angular-momentum channels."""
    for x in (a, b, c, d):
        integer(x, "twice-spin")
    if (a + b) % 2 != (c + d) % 2:
        return ()
    low, high = max(abs(a - b), abs(c - d)), min(a + b, c + d)
    return tuple(range(low, high + 1, 2))


def vertex_ranges(spins):
    if len(spins) != 10:
        raise ValueError("A vertex needs ten twice-spins")
    return tuple(intertwiner_range(*(spins[k] for k in legs)) for legs in LEGS)


def vertex_shape(spins):
    return tuple(len(r) for r in reversed(vertex_ranges(spins)))


def vertex_bytes(spins):
    """Output tensor only; native workspace and tables need additional memory."""
    return 8 * prod(vertex_shape(spins))


def delta4_bulk_spins(two_j):
    integer(two_j, "two_j", 1)
    # Three j's must couple to x: x has the parity of j and x <= 3j.
    return tuple(range(two_j % 2, 3 * two_j + 1, 2))


def delta4_spins(two_j, two_x):
    return (two_j,) * 7 + (two_x,) + (two_j,) * 2
