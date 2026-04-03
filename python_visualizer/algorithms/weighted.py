"""
algorithms/weighted.py – Dijkstra's algorithm and A* search.
"""
import math
from grid import get_all_nodes


def _manhattan(node_a, node_b) -> int:
    return abs(node_a.row - node_b.row) + abs(node_a.col - node_b.col)


def _closest_node(unvisited: list, use_astar: bool):
    """Pop and return the node with the smallest (total_)distance."""
    best_idx = 0
    for i in range(1, len(unvisited)):
        n = unvisited[i]
        b = unvisited[best_idx]
        if use_astar:
            if n.total_distance < b.total_distance:
                best_idx = i
            elif n.total_distance == b.total_distance:
                if (n.heuristic_dist or math.inf) < (b.heuristic_dist or math.inf):
                    best_idx = i
        else:
            if n.distance < b.distance:
                best_idx = i
    return unvisited.pop(best_idx)


def weighted_search(grid, start, target, nodes_to_animate: list, name: str):
    """
    Run Dijkstra (*name* == "Dijkstra") or A* (*name* == "aStar").

    Appends visited Node objects to *nodes_to_animate* in order.
    Returns True on success, False if no path exists.
    """
    use_astar = (name == "aStar")
    start.distance = 0
    if use_astar:
        start.total_distance = 0

    unvisited = get_all_nodes(grid)

    while unvisited:
        current = _closest_node(unvisited, use_astar)

        # skip walls
        while current.status == "wall" and unvisited:
            current = _closest_node(unvisited, use_astar)

        if current.distance == math.inf:
            return False

        nodes_to_animate.append(current)
        current.status = "visited"

        if current.id == target.id:
            return True

        _update_neighbors(grid, current, target, use_astar)

    return False


def _update_neighbors(grid, current, target, use_astar: bool):
    neighbors = _get_neighbors(current, grid)
    for nb in neighbors:
        dist_to_compare = current.distance + nb.weight
        if use_astar:
            if nb.heuristic_dist is None:
                nb.heuristic_dist = _manhattan(nb, target)
            new_total = dist_to_compare + nb.heuristic_dist
            if new_total < nb.total_distance:
                nb.distance       = dist_to_compare
                nb.total_distance = new_total
                nb.previous_node  = current
        else:
            if dist_to_compare < nb.distance:
                nb.distance      = dist_to_compare
                nb.previous_node = current


def _get_neighbors(node, grid):
    neighbors = []
    r, c = node.row, node.col
    if r > 0:               neighbors.append(grid[r - 1][c])
    if r < len(grid) - 1:   neighbors.append(grid[r + 1][c])
    if c > 0:               neighbors.append(grid[r][c - 1])
    if c < len(grid[0]) - 1: neighbors.append(grid[r][c + 1])
    return [n for n in neighbors if n.status != "visited"]
