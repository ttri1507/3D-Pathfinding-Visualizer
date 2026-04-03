"""
algorithms/unweighted.py – BFS and DFS.
"""
from collections import deque


def unweighted_search(grid, start, target, nodes_to_animate: list, name: str):
    """
    BFS (*name* == "BFS") or DFS (*name* == "DFS").

    Appends visited nodes to *nodes_to_animate*.
    Returns True on success, False otherwise.
    """
    structure = deque([start])
    explored  = {start.id}

    while structure:
        current = structure.popleft() if name == "BFS" else structure.pop()
        nodes_to_animate.append(current)
        current.status = "visited"

        if current.id == target.id:
            return True

        for nb in _get_neighbors(current, grid, name):
            if nb.id not in explored:
                explored.add(nb.id)
                if nb.id != start.id:
                    nb.previous_node = current
                structure.append(nb)

    return False


def _get_neighbors(node, grid, name: str):
    """Return neighbors in BFS or DFS order (DFS reverses for stack behaviour)."""
    neighbors = []
    r, c = node.row, node.col
    candidates = []
    if r > 0:                candidates.append(grid[r - 1][c])
    if r < len(grid) - 1:    candidates.append(grid[r + 1][c])
    if c > 0:                candidates.append(grid[r][c - 1])
    if c < len(grid[0]) - 1: candidates.append(grid[r][c + 1])

    for nb in candidates:
        if nb.status not in ("visited", "wall"):
            neighbors.append(nb)

    if name == "DFS":
        neighbors.reverse()
    return neighbors
