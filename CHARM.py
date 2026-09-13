"""CHARM: convex-hull area rank-bridge change-point detection.

This module contains the exact CHARM engine described by Xuesong Fu and Yao
Hu. It uses fixed full-sample average ranks, completed rank-bridge graphs,
exact convex-hull areas, a deterministic geometric multiscale family, and a
locally double-centered dependent Gaussian multiplier bootstrap.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

__version__ = "1.0.0"
CODE_VERSION = "CHARM-core-v1.0.0"

WINDOW_MIN_FRACTION = 0.10
WINDOW_MAX_FRACTION = 0.30
WINDOW_RATIO = 2.0 ** (-0.5)


def make_geometric_windows(n: int) -> List[int]:
    """Return the finalized deterministic geometric half-window family.

    The endpoints are G_min=ceil(0.10 n) and G_max=floor(0.30 n).  Starting
    from G_max, successive scales are floor(G_max * 2^{-j/2}) until the next
    value falls below G_min.  Returned windows are unique and sorted
    increasingly.
    """
    n = int(n)
    if n < 10:
        raise ValueError("The geometric scale family requires n >= 10.")

    G_min = max(2, int(math.ceil(WINDOW_MIN_FRACTION * float(n))))
    G_max = int(math.floor(WINDOW_MAX_FRACTION * float(n)))
    if G_max < G_min or 2 * G_max >= n:
        raise ValueError(
            "Invalid geometric scale endpoints: require 2 <= G_min <= G_max "
            "and 2*G_max < n."
        )

    windows_descending: List[int] = []
    j = 0
    while True:
        G = int(math.floor(G_max * (WINDOW_RATIO**j)))
        if G < G_min:
            break
        if G >= 2 and 2 * G < n:
            windows_descending.append(G)
        j += 1

    windows = sorted(set(windows_descending))
    if not windows:
        raise ValueError("The finalized geometric scale family is empty.")
    return windows


def multiplier_bandwidth(n: int) -> int:
    """Finalized dependence bandwidth ell_n = ceil(n^(1/4))."""
    if int(n) <= 0:
        raise ValueError("n must be positive.")
    return int(math.ceil(float(n) ** 0.25))


def global_average_rank_groups(
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct fixed coordinatewise full-sample average-rank groups.

    Continuous simulation data use a vectorized exact no-ties path.  Any
    coordinate containing ties automatically falls back to the general
    stable average-rank grouping algorithm.  Both branches return identical
    rank groups to the reference implementation.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    n, p = X.shape
    order = np.argsort(X, axis=0, kind="mergesort")
    sorted_x = np.take_along_axis(X, order, axis=0)

    group_id = np.empty((n, p), dtype=np.int32)
    group_x = np.zeros((p, n), dtype=np.float64)
    n_groups = np.empty(p, dtype=np.int32)
    base_group = np.arange(n, dtype=np.int32)
    base_x = (np.arange(1, n + 1, dtype=np.float64) / (n + 1.0))

    if n <= 1:
        tied_columns = np.zeros(p, dtype=bool)
    else:
        tied_columns = np.any(sorted_x[1:, :] == sorted_x[:-1, :], axis=0)

    for a in range(p):
        column_order = order[:, a]
        if not bool(tied_columns[a]):
            group_id[column_order, a] = base_group
            group_x[a, :] = base_x
            n_groups[a] = n
            continue

        column_sorted = sorted_x[:, a]
        gid_sorted = np.empty(n, dtype=np.int32)
        group = 0
        i = 0
        while i < n:
            j = i + 1
            while j < n and column_sorted[j] == column_sorted[i]:
                j += 1
            gid_sorted[i:j] = group
            average_rank = (i + 1 + j) / 2.0
            group_x[a, group] = average_rank / (n + 1.0)
            group += 1
            i = j
        group_id[column_order, a] = gid_sorted
        n_groups[a] = group

    return group_id, group_x, n_groups



def global_rank_group_probabilities(
    group_id: np.ndarray,
    n_groups: np.ndarray,
) -> np.ndarray:
    """Padded empirical probabilities of full-sample rank groups."""
    group_id = np.asarray(group_id, dtype=np.int32)
    n, p = group_id.shape
    probabilities = np.zeros((p, n), dtype=np.float64)
    for a in range(p):
        m = int(n_groups[a])
        counts = np.bincount(group_id[:, a], minlength=m).astype(np.float64)
        probabilities[a, :m] = counts / float(n)
    return probabilities


# =============================================================================
# Exact completed-graph convex-hull area engine
# =============================================================================


@njit(cache=True)
def _completed_graph_hull_area(
    labels: np.ndarray,
    x_group: np.ndarray,
    number_of_groups: int,
    G: int,
    lower_x: np.ndarray,
    lower_y: np.ndarray,
    upper_x: np.ndarray,
    upper_y: np.ndarray,
) -> float:
    """Exact hull area of one rank bridge from grouped jump labels.

    At every active rank group x_j, both the pre-jump and post-jump values are
    part of the completed graph.  Hence the lower hull sees min(S_{j-1},S_j)
    and the upper hull sees max(S_{j-1},S_j).  The bridge is first computed in
    unscaled cumulative-label units and divided by sqrt(2G) at the end.
    """
    n_lower = 1
    n_upper = 1
    lower_x[0] = 0.0
    lower_y[0] = 0.0
    upper_x[0] = 0.0
    upper_y[0] = 0.0

    cumulative = 0.0

    for group in range(number_of_groups):
        jump = labels[group]
        if jump == 0:
            continue

        before = cumulative
        after = cumulative + jump
        x_new = x_group[group]
        y_low = before if before < after else after
        y_high = after if after > before else before

        # Greatest convex minorant / lower convex chain.
        while n_lower >= 2:
            cross = (
                (lower_x[n_lower - 1] - lower_x[n_lower - 2])
                * (y_low - lower_y[n_lower - 2])
                - (lower_y[n_lower - 1] - lower_y[n_lower - 2])
                * (x_new - lower_x[n_lower - 2])
            )
            if cross <= 0.0:
                n_lower -= 1
            else:
                break
        lower_x[n_lower] = x_new
        lower_y[n_lower] = y_low
        n_lower += 1

        # Least concave majorant / upper convex chain.
        while n_upper >= 2:
            cross = (
                (upper_x[n_upper - 1] - upper_x[n_upper - 2])
                * (y_high - upper_y[n_upper - 2])
                - (upper_y[n_upper - 1] - upper_y[n_upper - 2])
                * (x_new - upper_x[n_upper - 2])
            )
            if cross >= 0.0:
                n_upper -= 1
            else:
                break
        upper_x[n_upper] = x_new
        upper_y[n_upper] = y_high
        n_upper += 1

        cumulative = after

    # The equal-size left and right windows force the terminal bridge value to 0.
    x_new = 1.0
    y_new = 0.0

    while n_lower >= 2:
        cross = (
            (lower_x[n_lower - 1] - lower_x[n_lower - 2])
            * (y_new - lower_y[n_lower - 2])
            - (lower_y[n_lower - 1] - lower_y[n_lower - 2])
            * (x_new - lower_x[n_lower - 2])
        )
        if cross <= 0.0:
            n_lower -= 1
        else:
            break
    lower_x[n_lower] = x_new
    lower_y[n_lower] = y_new
    n_lower += 1

    while n_upper >= 2:
        cross = (
            (upper_x[n_upper - 1] - upper_x[n_upper - 2])
            * (y_new - upper_y[n_upper - 2])
            - (upper_y[n_upper - 1] - upper_y[n_upper - 2])
            * (x_new - upper_x[n_upper - 2])
        )
        if cross >= 0.0:
            n_upper -= 1
        else:
            break
    upper_x[n_upper] = x_new
    upper_y[n_upper] = y_new
    n_upper += 1

    lower_integral = 0.0
    for i in range(n_lower - 1):
        lower_integral += 0.5 * (
            lower_y[i] + lower_y[i + 1]
        ) * (lower_x[i + 1] - lower_x[i])

    upper_integral = 0.0
    for i in range(n_upper - 1):
        upper_integral += 0.5 * (
            upper_y[i] + upper_y[i + 1]
        ) * (upper_x[i + 1] - upper_x[i])

    area = (upper_integral - lower_integral) / math.sqrt(2.0 * G)
    return area if area > 0.0 else 0.0


@njit(parallel=True, cache=True)
def _exact_area_curves_one_window(
    group_id: np.ndarray,
    group_x: np.ndarray,
    n_groups: np.ndarray,
    G: int,
) -> np.ndarray:
    """Compute exact coordinate hull-area curves for one half-window G."""
    n, p = group_id.shape
    number_of_candidates = n - 2 * G + 1
    output = np.empty((number_of_candidates, p), dtype=np.float64)

    for a in prange(p):
        m = int(n_groups[a])
        labels = np.zeros(m, dtype=np.int32)

        # Initial split b=G: left times 0,...,G-1; right G,...,2G-1.
        for t in range(G):
            labels[group_id[t, a]] -= 1
        for t in range(G, 2 * G):
            labels[group_id[t, a]] += 1

        # Reusable hull work arrays for this coordinate.
        lower_x = np.empty(m + 2, dtype=np.float64)
        lower_y = np.empty(m + 2, dtype=np.float64)
        upper_x = np.empty(m + 2, dtype=np.float64)
        upper_y = np.empty(m + 2, dtype=np.float64)

        for candidate_index in range(number_of_candidates):
            output[candidate_index, a] = _completed_graph_hull_area(
                labels,
                group_x[a],
                m,
                G,
                lower_x,
                lower_y,
                upper_x,
                upper_y,
            )

            if candidate_index + 1 < number_of_candidates:
                b = G + candidate_index
                # Move [b-G,b) vs [b,b+G) to [b-G+1,b+1) vs [b+1,b+G+1).
                labels[group_id[b - G, a]] += 1   # old left observation leaves
                labels[group_id[b, a]] -= 2       # old first right moves to left
                labels[group_id[b + G, a]] += 1   # new right observation enters

    return output


def compute_exact_charm_curves(
    X: np.ndarray,
    windows: Sequence[int],
) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute exact CHARM curves at all candidates and requested scales."""
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    group_id, group_x, n_groups = global_average_rank_groups(X)

    curves: Dict[int, Dict[str, np.ndarray]] = {}
    for G in windows:
        G = int(G)
        if not (2 <= G and 2 * G < n):
            raise ValueError(f"Invalid half-window G={G} for n={n}.")
        area = _exact_area_curves_one_window(group_id, group_x, n_groups, G)
        b_values = np.arange(G, n - G + 1, dtype=np.int32)
        if area.shape != (b_values.size, p):
            raise RuntimeError("Internal CHARM area-curve shape mismatch.")
        coordinate = np.argmax(area, axis=1).astype(np.int32)
        score = area[np.arange(area.shape[0]), coordinate]
        curves[G] = {
            "b": b_values,
            "area": area,
            "score": score,
            "coordinate": coordinate,
        }
    return curves



@njit(cache=True)
def _locally_centered_completed_graph_hull_area(
    left_weight: np.ndarray,
    right_weight: np.ndarray,
    left_count: np.ndarray,
    right_count: np.ndarray,
    left_total_weight: float,
    right_total_weight: float,
    x_group: np.ndarray,
    number_of_groups: int,
    G: int,
    cutoff: float,
    lower_x: np.ndarray,
    lower_y: np.ndarray,
    upper_x: np.ndarray,
    upper_y: np.ndarray,
) -> float:
    """Exact locally centered multiplier CHARM area for one candidate.

    For a rank group g, the jump of the bootstrap bridge is

        (R_wg - L_wg)
        - (R_ng/G) * R_w
        + (L_ng/G) * L_w,

    where R_wg/L_wg are groupwise multiplier sums, R_ng/L_ng are
    groupwise observation counts, and R_w/L_w are total multiplier sums in
    the right/left windows.  This is algebraically identical to separately
    centering the rank indicator process within the two local windows.

    A Kuiper-range upper bound is evaluated first.  The exact convex-hull
    construction is skipped only when that upper bound is already strictly
    below ``cutoff``.  The pruning is exact and does not alter the bootstrap
    maximum.
    """
    inv_G = 1.0 / float(G)
    scale = math.sqrt(2.0 * float(G))

    # First pass: exact vertical range, an upper bound for hull area because
    # the graph lies in a unit-width rectangle.
    cumulative = 0.0
    minimum = 0.0
    maximum = 0.0
    for group in range(number_of_groups):
        if left_count[group] == 0 and right_count[group] == 0:
            continue
        jump = (
            right_weight[group]
            - left_weight[group]
            - float(right_count[group]) * inv_G * right_total_weight
            + float(left_count[group]) * inv_G * left_total_weight
        )
        before = cumulative
        after = cumulative + jump
        if before < minimum:
            minimum = before
        if before > maximum:
            maximum = before
        if after < minimum:
            minimum = after
        if after > maximum:
            maximum = after
        cumulative = after

    range_upper_bound = (maximum - minimum) / scale
    # A small safety margin prevents floating-point roundoff from pruning a
    # candidate whose true area could be essentially tied with the cutoff.
    if range_upper_bound + 1e-12 < cutoff:
        return -1.0

    n_lower = 1
    n_upper = 1
    lower_x[0] = 0.0
    lower_y[0] = 0.0
    upper_x[0] = 0.0
    upper_y[0] = 0.0
    cumulative = 0.0

    for group in range(number_of_groups):
        if left_count[group] == 0 and right_count[group] == 0:
            continue
        jump = (
            right_weight[group]
            - left_weight[group]
            - float(right_count[group]) * inv_G * right_total_weight
            + float(left_count[group]) * inv_G * left_total_weight
        )
        if abs(jump) <= 1e-15:
            continue

        before = cumulative
        after = cumulative + jump
        x_new = x_group[group]
        y_low = before if before < after else after
        y_high = after if after > before else before

        while n_lower >= 2:
            cross = (
                (lower_x[n_lower - 1] - lower_x[n_lower - 2])
                * (y_low - lower_y[n_lower - 2])
                - (lower_y[n_lower - 1] - lower_y[n_lower - 2])
                * (x_new - lower_x[n_lower - 2])
            )
            if cross <= 0.0:
                n_lower -= 1
            else:
                break
        lower_x[n_lower] = x_new
        lower_y[n_lower] = y_low
        n_lower += 1

        while n_upper >= 2:
            cross = (
                (upper_x[n_upper - 1] - upper_x[n_upper - 2])
                * (y_high - upper_y[n_upper - 2])
                - (upper_y[n_upper - 1] - upper_y[n_upper - 2])
                * (x_new - upper_x[n_upper - 2])
            )
            if cross >= 0.0:
                n_upper -= 1
            else:
                break
        upper_x[n_upper] = x_new
        upper_y[n_upper] = y_high
        n_upper += 1

        cumulative = after

    # Separate local centering makes the terminal bridge value exactly zero
    # algebraically; set it to zero explicitly to eliminate numerical drift.
    x_new = 1.0
    y_new = 0.0

    while n_lower >= 2:
        cross = (
            (lower_x[n_lower - 1] - lower_x[n_lower - 2])
            * (y_new - lower_y[n_lower - 2])
            - (lower_y[n_lower - 1] - lower_y[n_lower - 2])
            * (x_new - lower_x[n_lower - 2])
        )
        if cross <= 0.0:
            n_lower -= 1
        else:
            break
    lower_x[n_lower] = x_new
    lower_y[n_lower] = y_new
    n_lower += 1

    while n_upper >= 2:
        cross = (
            (upper_x[n_upper - 1] - upper_x[n_upper - 2])
            * (y_new - upper_y[n_upper - 2])
            - (upper_y[n_upper - 1] - upper_y[n_upper - 2])
            * (x_new - upper_x[n_upper - 2])
        )
        if cross >= 0.0:
            n_upper -= 1
        else:
            break
    upper_x[n_upper] = x_new
    upper_y[n_upper] = y_new
    n_upper += 1

    lower_integral = 0.0
    for i in range(n_lower - 1):
        lower_integral += (
            0.5
            * (lower_y[i] + lower_y[i + 1])
            * (lower_x[i + 1] - lower_x[i])
        )

    upper_integral = 0.0
    for i in range(n_upper - 1):
        upper_integral += (
            0.5
            * (upper_y[i] + upper_y[i + 1])
            * (upper_x[i + 1] - upper_x[i])
        )

    area = (upper_integral - lower_integral) / scale
    return area if area > 0.0 else 0.0


@njit(cache=True)
def _fenwick_prefix_sum(tree: np.ndarray, end: int) -> int:
    """Number of active groups in [0,end)."""
    total = 0
    index = end
    while index > 0:
        total += int(tree[index])
        index -= index & -index
    return total


@njit(cache=True)
def _fenwick_add(tree: np.ndarray, size: int, group: int, delta: int) -> None:
    index = group + 1
    while index <= size:
        tree[index] += delta
        index += index & -index


@njit(cache=True)
def _fenwick_find_by_order(tree: np.ndarray, size: int, order: int) -> int:
    """0-based group index of the order-th active group, order starting at 1."""
    index = 0
    bit = 1
    while (bit << 1) <= size:
        bit <<= 1
    remaining = order
    while bit > 0:
        candidate = index + bit
        if candidate <= size and int(tree[candidate]) < remaining:
            index = candidate
            remaining -= int(tree[candidate])
        bit >>= 1
    return index


@njit(cache=True)
def _build_active_rank_structure(
    left_count: np.ndarray,
    right_count: np.ndarray,
    size: int,
    active: np.ndarray,
    previous_active: np.ndarray,
    next_active: np.ndarray,
    fenwick: np.ndarray,
) -> int:
    for index in range(size + 1):
        fenwick[index] = 0
    head = -1
    previous = -1
    for group in range(size):
        is_active = 1 if left_count[group] + right_count[group] > 0 else 0
        active[group] = is_active
        previous_active[group] = previous if is_active == 1 else -1
        next_active[group] = -1
        if is_active == 1:
            if previous >= 0:
                next_active[previous] = group
            else:
                head = group
            previous = group
            fenwick[group + 1] = 1
    # Linear Fenwick construction from the indicator array.
    for index in range(1, size + 1):
        parent = index + (index & -index)
        if parent <= size:
            fenwick[parent] += fenwick[index]
    return head


@njit(cache=True)
def _activate_rank_group(
    group: int,
    head: int,
    size: int,
    active: np.ndarray,
    previous_active: np.ndarray,
    next_active: np.ndarray,
    fenwick: np.ndarray,
) -> int:
    if active[group] == 1:
        return head
    number_before = _fenwick_prefix_sum(fenwick, group)
    total_active = _fenwick_prefix_sum(fenwick, size)
    predecessor = (
        _fenwick_find_by_order(fenwick, size, number_before)
        if number_before > 0
        else -1
    )
    successor = (
        _fenwick_find_by_order(fenwick, size, number_before + 1)
        if number_before < total_active
        else -1
    )
    previous_active[group] = predecessor
    next_active[group] = successor
    if predecessor >= 0:
        next_active[predecessor] = group
    else:
        head = group
    if successor >= 0:
        previous_active[successor] = group
    active[group] = 1
    _fenwick_add(fenwick, size, group, 1)
    return head


@njit(cache=True)
def _deactivate_rank_group(
    group: int,
    head: int,
    size: int,
    active: np.ndarray,
    previous_active: np.ndarray,
    next_active: np.ndarray,
    fenwick: np.ndarray,
) -> int:
    if active[group] == 0:
        return head
    predecessor = int(previous_active[group])
    successor = int(next_active[group])
    if predecessor >= 0:
        next_active[predecessor] = successor
    else:
        head = successor
    if successor >= 0:
        previous_active[successor] = predecessor
    active[group] = 0
    previous_active[group] = -1
    next_active[group] = -1
    _fenwick_add(fenwick, size, group, -1)
    return head


@njit(cache=True)
def _locally_centered_hull_area_active_ranks(
    left_weight: np.ndarray,
    right_weight: np.ndarray,
    left_count: np.ndarray,
    right_count: np.ndarray,
    left_total_weight: float,
    right_total_weight: float,
    x_group: np.ndarray,
    first_active: int,
    next_active: np.ndarray,
    G: int,
    cutoff: float,
    lower_x: np.ndarray,
    lower_y: np.ndarray,
    upper_x: np.ndarray,
    upper_y: np.ndarray,
) -> float:
    """Exact hull area while traversing only locally active rank groups."""
    inv_G = 1.0 / float(G)
    scale = math.sqrt(2.0 * float(G))

    cumulative = 0.0
    minimum = 0.0
    maximum = 0.0
    group = first_active
    while group >= 0:
        jump = (
            right_weight[group]
            - left_weight[group]
            - float(right_count[group]) * inv_G * right_total_weight
            + float(left_count[group]) * inv_G * left_total_weight
        )
        after = cumulative + jump
        if cumulative < minimum:
            minimum = cumulative
        if cumulative > maximum:
            maximum = cumulative
        if after < minimum:
            minimum = after
        if after > maximum:
            maximum = after
        cumulative = after
        group = int(next_active[group])

    if (maximum - minimum) / scale + 1e-12 < cutoff:
        return -1.0

    n_lower = 1
    n_upper = 1
    lower_x[0] = 0.0
    lower_y[0] = 0.0
    upper_x[0] = 0.0
    upper_y[0] = 0.0
    cumulative = 0.0
    group = first_active
    while group >= 0:
        jump = (
            right_weight[group]
            - left_weight[group]
            - float(right_count[group]) * inv_G * right_total_weight
            + float(left_count[group]) * inv_G * left_total_weight
        )
        if abs(jump) > 1e-15:
            before = cumulative
            after = cumulative + jump
            x_new = x_group[group]
            y_low = before if before < after else after
            y_high = after if after > before else before

            while n_lower >= 2:
                cross = (
                    (lower_x[n_lower - 1] - lower_x[n_lower - 2])
                    * (y_low - lower_y[n_lower - 2])
                    - (lower_y[n_lower - 1] - lower_y[n_lower - 2])
                    * (x_new - lower_x[n_lower - 2])
                )
                if cross <= 0.0:
                    n_lower -= 1
                else:
                    break
            lower_x[n_lower] = x_new
            lower_y[n_lower] = y_low
            n_lower += 1

            while n_upper >= 2:
                cross = (
                    (upper_x[n_upper - 1] - upper_x[n_upper - 2])
                    * (y_high - upper_y[n_upper - 2])
                    - (upper_y[n_upper - 1] - upper_y[n_upper - 2])
                    * (x_new - upper_x[n_upper - 2])
                )
                if cross >= 0.0:
                    n_upper -= 1
                else:
                    break
            upper_x[n_upper] = x_new
            upper_y[n_upper] = y_high
            n_upper += 1
            cumulative = after
        group = int(next_active[group])

    x_new = 1.0
    y_new = 0.0
    while n_lower >= 2:
        cross = (
            (lower_x[n_lower - 1] - lower_x[n_lower - 2])
            * (y_new - lower_y[n_lower - 2])
            - (lower_y[n_lower - 1] - lower_y[n_lower - 2])
            * (x_new - lower_x[n_lower - 2])
        )
        if cross <= 0.0:
            n_lower -= 1
        else:
            break
    lower_x[n_lower] = x_new
    lower_y[n_lower] = y_new
    n_lower += 1

    while n_upper >= 2:
        cross = (
            (upper_x[n_upper - 1] - upper_x[n_upper - 2])
            * (y_new - upper_y[n_upper - 2])
            - (upper_y[n_upper - 1] - upper_y[n_upper - 2])
            * (x_new - upper_x[n_upper - 2])
        )
        if cross >= 0.0:
            n_upper -= 1
        else:
            break
    upper_x[n_upper] = x_new
    upper_y[n_upper] = y_new
    n_upper += 1

    lower_integral = 0.0
    for index in range(n_lower - 1):
        lower_integral += 0.5 * (
            lower_y[index] + lower_y[index + 1]
        ) * (lower_x[index + 1] - lower_x[index])
    upper_integral = 0.0
    for index in range(n_upper - 1):
        upper_integral += 0.5 * (
            upper_y[index] + upper_y[index + 1]
        ) * (upper_x[index + 1] - upper_x[index])
    area = (upper_integral - lower_integral) / scale
    return area if area > 0.0 else 0.0


@njit(parallel=True, cache=True)
def _bootstrap_global_maxima_local_centered_reference(
    group_id: np.ndarray,
    group_x: np.ndarray,
    n_groups: np.ndarray,
    windows: np.ndarray,
    multipliers: np.ndarray,
) -> np.ndarray:
    """Exact global CHARM maxima under local double-centered multipliers."""
    number_of_bootstraps, n = multipliers.shape
    p = group_id.shape[1]
    output = np.zeros(number_of_bootstraps, dtype=np.float64)

    for replicate in prange(number_of_bootstraps):
        weights = multipliers[replicate]
        global_maximum = 0.0

        for window_index in range(windows.size):
            G = int(windows[window_index])
            number_of_candidates = n - 2 * G + 1

            for a in range(p):
                m = int(n_groups[a])
                left_weight = np.zeros(m, dtype=np.float64)
                right_weight = np.zeros(m, dtype=np.float64)
                left_count = np.zeros(m, dtype=np.int32)
                right_count = np.zeros(m, dtype=np.int32)

                left_total_weight = 0.0
                right_total_weight = 0.0

                # Initial candidate b=G:
                # left=[0,G), right=[G,2G).
                for t in range(G):
                    group = group_id[t, a]
                    weight = weights[t]
                    left_weight[group] += weight
                    left_count[group] += 1
                    left_total_weight += weight

                for t in range(G, 2 * G):
                    group = group_id[t, a]
                    weight = weights[t]
                    right_weight[group] += weight
                    right_count[group] += 1
                    right_total_weight += weight

                lower_x = np.empty(m + 2, dtype=np.float64)
                lower_y = np.empty(m + 2, dtype=np.float64)
                upper_x = np.empty(m + 2, dtype=np.float64)
                upper_y = np.empty(m + 2, dtype=np.float64)

                for candidate_index in range(number_of_candidates):
                    area = _locally_centered_completed_graph_hull_area(
                        left_weight,
                        right_weight,
                        left_count,
                        right_count,
                        left_total_weight,
                        right_total_weight,
                        group_x[a],
                        m,
                        G,
                        global_maximum,
                        lower_x,
                        lower_y,
                        upper_x,
                        upper_y,
                    )
                    if area > global_maximum:
                        global_maximum = area

                    if candidate_index + 1 < number_of_candidates:
                        b = G + candidate_index
                        old_left = b - G
                        old_right_first = b
                        new_right = b + G

                        group = group_id[old_left, a]
                        weight = weights[old_left]
                        left_weight[group] -= weight
                        left_count[group] -= 1
                        left_total_weight -= weight

                        group = group_id[old_right_first, a]
                        weight = weights[old_right_first]
                        right_weight[group] -= weight
                        right_count[group] -= 1
                        right_total_weight -= weight
                        left_weight[group] += weight
                        left_count[group] += 1
                        left_total_weight += weight

                        group = group_id[new_right, a]
                        weight = weights[new_right]
                        right_weight[group] += weight
                        right_count[group] += 1
                        right_total_weight += weight

        output[replicate] = global_maximum

    return output


@njit(cache=True)
def _reset_prefix_float(values: np.ndarray, length: int) -> None:
    for index in range(length):
        values[index] = 0.0


@njit(cache=True)
def _reset_prefix_int(values: np.ndarray, length: int) -> None:
    for index in range(length):
        values[index] = 0


@njit(cache=True)
def _initialize_weighted_window(
    group_id: np.ndarray,
    coordinate: int,
    weights: np.ndarray,
    b: int,
    G: int,
    left_weight: np.ndarray,
    right_weight: np.ndarray,
    left_count: np.ndarray,
    right_count: np.ndarray,
) -> Tuple[float, float]:
    left_total = 0.0
    right_total = 0.0
    for t in range(b - G, b):
        group = group_id[t, coordinate]
        weight = weights[t]
        left_weight[group] += weight
        left_count[group] += 1
        left_total += weight
    for t in range(b, b + G):
        group = group_id[t, coordinate]
        weight = weights[t]
        right_weight[group] += weight
        right_count[group] += 1
        right_total += weight
    return left_total, right_total


@njit(parallel=True, cache=True)
def _bootstrap_global_maxima_local_centered(
    group_id: np.ndarray,
    group_x: np.ndarray,
    n_groups: np.ndarray,
    windows: np.ndarray,
    multipliers: np.ndarray,
    warmup_anchors: int,
) -> np.ndarray:
    """Exact optimized global CHARM maxima.

    The returned values are mathematically identical to the reference engine.
    Optimizations are computational only: per-replicate workspace reuse,
    exact active-rank skipping, and a deterministic warm-start that evaluates
    a small subset of candidates before the exhaustive scan.  The warm-start
    supplies only a valid lower bound for exact Kuiper pruning; every candidate
    is still visited in the exhaustive pass.
    """
    number_of_bootstraps, n = multipliers.shape
    p = group_id.shape[1]
    output = np.zeros(number_of_bootstraps, dtype=np.float64)
    number_of_anchors = warmup_anchors if warmup_anchors > 0 else 0

    for replicate in prange(number_of_bootstraps):
        weights = multipliers[replicate]
        global_maximum = 0.0

        # Allocate once per multiplier replicate and reuse for every scale and
        # coordinate.  n is an upper bound for every coordinate's group count.
        left_weight = np.empty(n, dtype=np.float64)
        right_weight = np.empty(n, dtype=np.float64)
        left_count = np.empty(n, dtype=np.int32)
        right_count = np.empty(n, dtype=np.int32)
        lower_x = np.empty(n + 2, dtype=np.float64)
        lower_y = np.empty(n + 2, dtype=np.float64)
        upper_x = np.empty(n + 2, dtype=np.float64)
        upper_y = np.empty(n + 2, dtype=np.float64)
        active = np.empty(n, dtype=np.uint8)
        previous_active = np.empty(n, dtype=np.int32)
        next_active = np.empty(n, dtype=np.int32)
        fenwick = np.empty(n + 1, dtype=np.int32)

        # Exact warm-start.  It does not remove candidates; it merely raises
        # the current lower bound before the exhaustive traversal, allowing
        # more exact upper-bound pruning from the beginning.
        if number_of_anchors > 0:
            for window_index in range(windows.size):
                G = int(windows[window_index])
                span = n - 2 * G
                for a in range(p):
                    m = int(n_groups[a])
                    for anchor_index in range(number_of_anchors):
                        b = G + ((anchor_index + 1) * span) // (number_of_anchors + 1)
                        _reset_prefix_float(left_weight, m)
                        _reset_prefix_float(right_weight, m)
                        _reset_prefix_int(left_count, m)
                        _reset_prefix_int(right_count, m)
                        left_total, right_total = _initialize_weighted_window(
                            group_id,
                            a,
                            weights,
                            b,
                            G,
                            left_weight,
                            right_weight,
                            left_count,
                            right_count,
                        )
                        area = _locally_centered_completed_graph_hull_area(
                            left_weight,
                            right_weight,
                            left_count,
                            right_count,
                            left_total,
                            right_total,
                            group_x[a],
                            m,
                            G,
                            global_maximum,
                            lower_x,
                            lower_y,
                            upper_x,
                            upper_y,
                        )
                        if area > global_maximum:
                            global_maximum = area

        # Exhaustive exact scan over every requested scale, coordinate and
        # legal candidate split.
        for window_index in range(windows.size):
            G = int(windows[window_index])
            number_of_candidates = n - 2 * G + 1

            for a in range(p):
                m = int(n_groups[a])
                _reset_prefix_float(left_weight, m)
                _reset_prefix_float(right_weight, m)
                _reset_prefix_int(left_count, m)
                _reset_prefix_int(right_count, m)
                left_total_weight, right_total_weight = _initialize_weighted_window(
                    group_id,
                    a,
                    weights,
                    G,
                    G,
                    left_weight,
                    right_weight,
                    left_count,
                    right_count,
                )

                first_active = _build_active_rank_structure(
                    left_count,
                    right_count,
                    m,
                    active,
                    previous_active,
                    next_active,
                    fenwick,
                )

                for candidate_index in range(number_of_candidates):
                    area = _locally_centered_hull_area_active_ranks(
                        left_weight,
                        right_weight,
                        left_count,
                        right_count,
                        left_total_weight,
                        right_total_weight,
                        group_x[a],
                        first_active,
                        next_active,
                        G,
                        global_maximum,
                        lower_x,
                        lower_y,
                        upper_x,
                        upper_y,
                    )
                    if area > global_maximum:
                        global_maximum = area

                    if candidate_index + 1 < number_of_candidates:
                        b = G + candidate_index
                        old_left = b - G
                        old_right_first = b
                        new_right = b + G

                        group = group_id[old_left, a]
                        old_total = left_count[group] + right_count[group]
                        weight = weights[old_left]
                        left_weight[group] -= weight
                        left_count[group] -= 1
                        left_total_weight -= weight
                        if old_total > 0 and left_count[group] + right_count[group] == 0:
                            first_active = _deactivate_rank_group(
                                group,
                                first_active,
                                m,
                                active,
                                previous_active,
                                next_active,
                                fenwick,
                            )

                        group = group_id[old_right_first, a]
                        weight = weights[old_right_first]
                        right_weight[group] -= weight
                        right_count[group] -= 1
                        right_total_weight -= weight
                        left_weight[group] += weight
                        left_count[group] += 1
                        left_total_weight += weight

                        group = group_id[new_right, a]
                        old_total = left_count[group] + right_count[group]
                        weight = weights[new_right]
                        right_weight[group] += weight
                        right_count[group] += 1
                        right_total_weight += weight
                        if old_total == 0:
                            first_active = _activate_rank_group(
                                group,
                                first_active,
                                m,
                                active,
                                previous_active,
                                next_active,
                                fenwick,
                            )

        output[replicate] = global_maximum

    return output


def generate_bartlett_dependent_multipliers(
    rng: np.random.Generator,
    n_boot: int,
    n: int,
    block_length: int,
) -> np.ndarray:
    """Generate variance-one Gaussian multipliers with a Bartlett kernel."""
    ell = int(block_length)
    if ell < 1 or ell > n:
        raise ValueError("block_length must lie in [1,n].")
    innovations = rng.normal(size=(int(n_boot), n + ell - 1))
    cumulative = np.concatenate(
        [
            np.zeros((int(n_boot), 1), dtype=np.float64),
            np.cumsum(innovations, axis=1),
        ],
        axis=1,
    )
    multipliers = (
        cumulative[:, ell : ell + n] - cumulative[:, :n]
    ) / math.sqrt(float(ell))
    return np.asarray(multipliers, dtype=np.float64)


def select_block_length(n: int) -> Tuple[int, Dict[str, object]]:
    """Return the finalized multiplier bandwidth ell_n = ceil(n^(1/4))."""
    ell = multiplier_bandwidth(int(n))
    return ell, {
        "selector": "ceil_n_quarter",
        "selected_block_length": int(ell),
    }

def bootstrap_charm_threshold_from_groups(
    group_id: np.ndarray,
    group_x: np.ndarray,
    n_groups: np.ndarray,
    windows: Sequence[int],
    multipliers: np.ndarray,
    alpha: float,
    warmup_anchors: int,
) -> Tuple[float, np.ndarray]:
    """Conditional global threshold from exact locally centered CHARM draws."""
    maxima = _bootstrap_global_maxima_local_centered(
        np.asarray(group_id, dtype=np.int32),
        np.asarray(group_x, dtype=np.float64),
        np.asarray(n_groups, dtype=np.int32),
        np.asarray(windows, dtype=np.int32),
        np.asarray(multipliers, dtype=np.float64),
        int(warmup_anchors),
    )
    maxima = np.asarray(maxima, dtype=np.float64)
    B = int(maxima.size)
    if B <= 0:
        raise ValueError("At least one bootstrap maximum is required.")
    if float(alpha) * (B + 1) < 1.0:
        raise ValueError(
            "Require alpha*(B+1) >= 1 for the finite-sample order-statistic threshold."
        )
    k_alpha = int(math.ceil((B + 1) * (1.0 - float(alpha))))  # 1-based
    gamma = float(np.partition(maxima, k_alpha - 1)[k_alpha - 1])
    return gamma, maxima


def compute_exact_charm_curves_from_groups(
    group_id: np.ndarray,
    group_x: np.ndarray,
    n_groups: np.ndarray,
    windows: Sequence[int],
) -> Dict[int, Dict[str, np.ndarray]]:
    n, p = group_id.shape
    curves: Dict[int, Dict[str, np.ndarray]] = {}
    for G in windows:
        G = int(G)
        area = _exact_area_curves_one_window(group_id, group_x, n_groups, G)
        b_values = np.arange(G, n - G + 1, dtype=np.int32)
        coordinate = np.argmax(area, axis=1).astype(np.int32)
        score = area[np.arange(area.shape[0]), coordinate]
        curves[G] = {"b": b_values, "area": area, "score": score, "coordinate": coordinate}
    return curves


def calibrate_and_run_charm(
    X: np.ndarray,
    windows: Sequence[int],
    args: argparse.Namespace,
    bootstrap_seed: int,
) -> Tuple[List[int], List[dict], Dict[int, Dict[str, np.ndarray]], Dict[str, object]]:
    """Data-only CHARM calibration and detection for one dataset."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite two-dimensional array.")
    group_id, group_x, n_groups = global_average_rank_groups(X)

    observed_start = time.time()
    curves = compute_exact_charm_curves_from_groups(group_id, group_x, n_groups, windows)
    observed_seconds = time.time() - observed_start
    observed_global_max = max(float(np.max(obj["score"])) for obj in curves.values())

    block_length, block_diagnostics = select_block_length(X.shape[0])
    rng = np.random.default_rng(int(bootstrap_seed))
    multipliers = generate_bartlett_dependent_multipliers(
        rng,
        n_boot=int(args.n_boot),
        n=X.shape[0],
        block_length=block_length,
    )
    bootstrap_start = time.time()
    gamma, bootstrap_maxima = bootstrap_charm_threshold_from_groups(
        group_id,
        group_x,
        n_groups,
        windows,
        multipliers,
        float(args.alpha),
        int(args.bootstrap_warmup_anchors),
    )
    bootstrap_seconds = time.time() - bootstrap_start
    bootstrap_pvalue = (1.0 + float(np.sum(bootstrap_maxima >= observed_global_max))) / (
        float(args.n_boot) + 1.0
    )

    candidates = find_multiscale_candidates(
        curves,
        gamma=gamma,
        top_coordinates_to_store=int(args.top_coordinates_to_store),
    )
    merged = merge_multiscale_candidates(candidates)
    cps = [int(item["b"]) for item in merged]
    diagnostics: Dict[str, object] = {
        "gamma": float(gamma),
        "bootstrap_pvalue": float(bootstrap_pvalue),
        "observed_global_max": float(observed_global_max),
        "block_length": int(block_length),
        "observed_seconds": float(observed_seconds),
        "bootstrap_seconds": float(bootstrap_seconds),
        "bootstrap_mean": float(np.mean(bootstrap_maxima)),
        "bootstrap_sd": float(np.std(bootstrap_maxima, ddof=1)) if bootstrap_maxima.size > 1 else 0.0,
        **block_diagnostics,
    }
    return cps, merged, curves, diagnostics

# =============================================================================
# Exactness reference check, including ties
# =============================================================================


def _python_hull_chain_integral(
    points: Sequence[Tuple[float, float]],
    lower: bool,
) -> float:
    chain: List[Tuple[float, float]] = []
    for point in points:
        while len(chain) >= 2:
            x0, y0 = chain[-2]
            x1, y1 = chain[-1]
            x2, y2 = point
            cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            remove = cross <= 0.0 if lower else cross >= 0.0
            if remove:
                chain.pop()
            else:
                break
        chain.append(point)
    integral = 0.0
    for (x0, y0), (x1, y1) in zip(chain[:-1], chain[1:]):
        integral += 0.5 * (y0 + y1) * (x1 - x0)
    return integral


def brute_force_charm_area(
    x: np.ndarray,
    b: int,
    G: int,
) -> float:
    """Slow exact reference using the global average-rank completed graph."""
    x = np.asarray(x)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]

    group_of_time = np.empty(n, dtype=np.int32)
    group_rank: List[float] = []
    group = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        group_of_time[order[i:j]] = group
        group_rank.append(((i + 1 + j) / 2.0) / (n + 1.0))
        group += 1
        i = j

    labels = np.zeros(group, dtype=np.int32)
    for t in range(b - G, b):
        labels[group_of_time[t]] -= 1
    for t in range(b, b + G):
        labels[group_of_time[t]] += 1

    lower_points: List[Tuple[float, float]] = [(0.0, 0.0)]
    upper_points: List[Tuple[float, float]] = [(0.0, 0.0)]
    cumulative = 0.0
    for g in range(group):
        jump = int(labels[g])
        if jump == 0:
            continue
        before = cumulative
        after = cumulative + jump
        lower_points.append((group_rank[g], min(before, after)))
        upper_points.append((group_rank[g], max(before, after)))
        cumulative = after
    lower_points.append((1.0, 0.0))
    upper_points.append((1.0, 0.0))

    lower_integral = _python_hull_chain_integral(lower_points, lower=True)
    upper_integral = _python_hull_chain_integral(upper_points, lower=False)
    return max(
        0.0,
        (upper_integral - lower_integral) / math.sqrt(2.0 * G),
    )




def brute_force_locally_centered_charm_area(
    x: np.ndarray,
    b: int,
    G: int,
    weights: np.ndarray,
) -> float:
    """Slow reference for one locally double-centered multiplier bridge."""
    x = np.asarray(x)
    weights = np.asarray(weights, dtype=np.float64)
    n = x.size
    if weights.shape != (n,):
        raise ValueError("weights must have shape (n,).")

    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    group_of_time = np.empty(n, dtype=np.int32)
    group_rank: List[float] = []
    group = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        group_of_time[order[i:j]] = group
        group_rank.append(((i + 1 + j) / 2.0) / (n + 1.0))
        group += 1
        i = j

    left_weight = np.zeros(group, dtype=np.float64)
    right_weight = np.zeros(group, dtype=np.float64)
    left_count = np.zeros(group, dtype=np.int32)
    right_count = np.zeros(group, dtype=np.int32)
    left_total = 0.0
    right_total = 0.0

    for t in range(b - G, b):
        g = group_of_time[t]
        left_weight[g] += weights[t]
        left_count[g] += 1
        left_total += weights[t]

    for t in range(b, b + G):
        g = group_of_time[t]
        right_weight[g] += weights[t]
        right_count[g] += 1
        right_total += weights[t]

    jumps = (
        right_weight
        - left_weight
        - right_count.astype(np.float64) * right_total / float(G)
        + left_count.astype(np.float64) * left_total / float(G)
    )

    lower_points: List[Tuple[float, float]] = [(0.0, 0.0)]
    upper_points: List[Tuple[float, float]] = [(0.0, 0.0)]
    cumulative = 0.0
    for g, jump in enumerate(jumps.tolist()):
        if abs(float(jump)) <= 1e-15:
            continue
        before = cumulative
        after = cumulative + float(jump)
        lower_points.append((group_rank[g], min(before, after)))
        upper_points.append((group_rank[g], max(before, after)))
        cumulative = after

    lower_points.append((1.0, 0.0))
    upper_points.append((1.0, 0.0))
    lower_integral = _python_hull_chain_integral(lower_points, lower=True)
    upper_integral = _python_hull_chain_integral(upper_points, lower=False)
    return max(
        0.0,
        (upper_integral - lower_integral) / math.sqrt(2.0 * G),
    )


def validate_bootstrap_charm_engine(seed: int = 20260718) -> float:
    """Check the locally centered bootstrap engine against brute force."""
    rng = np.random.default_rng(seed)
    X = np.round(rng.normal(size=(33, 5)), 1)
    windows = np.asarray([4, 7, 10], dtype=np.int32)
    group_id, group_x, n_groups = global_average_rank_groups(X)
    multipliers = rng.normal(size=(3, X.shape[0]))

    fast = _bootstrap_global_maxima_local_centered(
        group_id,
        group_x,
        n_groups,
        windows,
        multipliers,
        3,
    )
    old_engine = _bootstrap_global_maxima_local_centered_reference(
        group_id,
        group_x,
        n_groups,
        windows,
        multipliers,
    )
    reference = np.zeros(multipliers.shape[0], dtype=np.float64)
    for r, weights in enumerate(multipliers):
        maximum = 0.0
        for G in windows.tolist():
            for b in range(int(G), X.shape[0] - int(G) + 1):
                for a in range(X.shape[1]):
                    maximum = max(
                        maximum,
                        brute_force_locally_centered_charm_area(
                            X[:, a],
                            b,
                            int(G),
                            weights,
                        ),
                    )
        reference[r] = maximum

    return float(
        max(
            np.max(np.abs(fast - reference)),
            np.max(np.abs(fast - old_engine)),
        )
    )


def validate_exact_charm_engine(seed: int = 20260717) -> float:
    """Maximum exact-engine error against brute force on deliberately tied data."""
    rng = np.random.default_rng(seed)
    X = np.round(rng.normal(size=(41, 6)), 1)
    windows = [4, 7, 11]
    curves = compute_exact_charm_curves(X, windows)
    maximum_error = 0.0
    for G in windows:
        b_values = curves[G]["b"]
        area = curves[G]["area"]
        for index, b in enumerate(b_values):
            for a in range(X.shape[1]):
                reference = brute_force_charm_area(X[:, a], int(b), int(G))
                maximum_error = max(
                    maximum_error,
                    abs(float(area[index, a]) - reference),
                )
    return maximum_error


# =============================================================================
# Direct multiple-change-point extraction
# =============================================================================


def supra_threshold_regions(
    values: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int]]:
    mask = np.asarray(values > threshold, dtype=bool)
    regions: List[Tuple[int, int]] = []
    index = 0
    while index < mask.size:
        if not mask[index]:
            index += 1
            continue
        stop = index + 1
        while stop < mask.size and mask[stop]:
            stop += 1
        regions.append((index, stop - 1))
        index = stop
    return regions


def find_multiscale_candidates(
    curves: Dict[int, Dict[str, np.ndarray]],
    gamma: float,
    top_coordinates_to_store: int,
) -> List[dict]:
    candidates: List[dict] = []
    for G, obj in curves.items():
        b_values = obj["b"]
        score = obj["score"]
        area = obj["area"]

        for left_index, right_index in supra_threshold_regions(score, gamma):
            local_index = left_index + int(
                np.argmax(score[left_index : right_index + 1])
            )
            area_row = np.asarray(area[local_index], dtype=np.float64)
            ranking = np.lexsort(
                (np.arange(area_row.size, dtype=np.int32), -area_row)
            ).astype(np.int32)
            number_to_store = min(int(top_coordinates_to_store), ranking.size)
            top_ranking = ranking[:number_to_store]
            candidates.append(
                {
                    "b": int(b_values[local_index]),
                    "G": int(G),
                    "value": float(score[local_index]),
                    "coordinate": int(ranking[0]),
                    "ranking": ranking,
                    "top_coordinates": top_ranking,
                    "top_areas": area_row[top_ranking],
                    "region_left": int(b_values[left_index]),
                    "region_right": int(b_values[right_index]),
                }
            )

    candidates.sort(
        key=lambda item: (
            -float(item["value"]),
            int(item["G"]),
            int(item["b"]),
        )
    )
    return candidates


def merge_multiscale_candidates(candidates: List[dict]) -> List[dict]:
    """Apply the finalized rule |b-b'| > min(G,G') to accepted candidates."""
    accepted: List[dict] = []
    for candidate in candidates:
        separated = all(
            abs(int(candidate["b"]) - int(existing["b"]))
            > min(int(candidate["G"]), int(existing["G"]))
            for existing in accepted
        )
        if separated:
            accepted.append(candidate)
    accepted.sort(key=lambda item: int(item["b"]))
    return accepted


def run_charm(
    X: np.ndarray,
    windows: Sequence[int],
    gamma: float,
    top_coordinates_to_store: int,
) -> Tuple[List[int], List[dict], Dict[int, Dict[str, np.ndarray]]]:
    curves = compute_exact_charm_curves(X, windows)
    candidates = find_multiscale_candidates(
        curves,
        gamma=float(gamma),
        top_coordinates_to_store=int(top_coordinates_to_store),
    )
    merged = merge_multiscale_candidates(candidates)
    cps = [int(item["b"]) for item in merged]
    return cps, merged, curves


# =============================================================================
# Bootstrap utilities shared by simulation workers
# =============================================================================


def namespace_from_dict(values: dict) -> argparse.Namespace:
    return argparse.Namespace(**values)


def configure_numba_threads(number_of_threads: int) -> None:
    requested = max(1, int(number_of_threads))
    try:
        set_num_threads(requested)
    except ValueError as exc:
        raise ValueError(
            f"Unable to set --numba-threads={requested}; "
            f"Numba reports a maximum of {get_num_threads()}."
        ) from exc


def fit_charm(
    X: np.ndarray,
    *,
    alpha: float = 0.05,
    n_boot: int = 199,
    seed: int = 20260212,
    windows: Optional[Sequence[int]] = None,
    numba_threads: int = 1,
    top_coordinates_to_store: Optional[int] = None,
    validate_engine: bool = False,
) -> Dict[str, object]:
    """Detect marginal-distribution change points in an n by p array.

    Returned change points are split indices b: adjacent samples are X[b-1]
    and X[b], and the two slices are X[:b] and X[b:].
    """
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 10 or data.shape[1] < 1:
        raise ValueError("X must be a two-dimensional n x p array with n >= 10.")
    if not np.all(np.isfinite(data)):
        raise ValueError("X must contain only finite values.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie in (0, 1).")
    if int(n_boot) <= 0 or float(alpha) * (int(n_boot) + 1) < 1.0:
        raise ValueError("Require n_boot > 0 and alpha*(n_boot+1) >= 1.")

    configure_numba_threads(int(numba_threads))
    if validate_engine:
        observed_error = validate_exact_charm_engine()
        bootstrap_error = validate_bootstrap_charm_engine()
        if observed_error > 1e-11 or bootstrap_error > 1e-11:
            raise RuntimeError(
                "A CHARM exactness check failed: "
                f"observed={observed_error:.3e}, bootstrap={bootstrap_error:.3e}."
            )

    selected_windows = (
        make_geometric_windows(data.shape[0])
        if windows is None
        else sorted(set(int(value) for value in windows))
    )
    if not selected_windows:
        raise ValueError("At least one window width is required.")
    if any(value < 2 or 2 * value >= data.shape[0] for value in selected_windows):
        raise ValueError("Every window G must satisfy 2 <= G and 2*G < n.")

    stored = (
        data.shape[1]
        if top_coordinates_to_store is None
        else int(top_coordinates_to_store)
    )
    if stored <= 0:
        raise ValueError("top_coordinates_to_store must be positive.")

    options = argparse.Namespace(
        n_boot=int(n_boot),
        alpha=float(alpha),
        bootstrap_warmup_anchors=0,
        top_coordinates_to_store=min(stored, data.shape[1]),
    )
    change_points, detections, curves, diagnostics = calibrate_and_run_charm(
        data,
        windows=selected_windows,
        args=options,
        bootstrap_seed=int(seed),
    )
    return {
        "change_points": change_points,
        "detections": detections,
        "curves": curves,
        "windows": selected_windows,
        "critical_value": diagnostics["gamma"],
        "p_value": diagnostics["bootstrap_pvalue"],
        "observed_global_maximum": diagnostics["observed_global_max"],
        "multiplier_bandwidth": diagnostics["block_length"],
        "diagnostics": diagnostics,
    }


__all__ = ["fit_charm", "__version__"]

