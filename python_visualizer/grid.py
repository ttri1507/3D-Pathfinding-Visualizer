"""
grid.py – Node and Grid data structures for the pathfinding visualiser.
"""
import math

ROWS = 30
COLS = 30

# Node statuses
STATUS_DEFAULT = "default"
STATUS_START   = "start"
STATUS_FINISH  = "finish"
STATUS_WALL    = "wall"
STATUS_VISITED = "visited"
STATUS_PATH    = "path"


class Node:
    """One cell of the grid."""

    def __init__(self, row: int, col: int, rows: int = ROWS, cols: int = COLS):
        self.row = row
        self.col = col
        self.id  = row * cols + col

        self.status   = STATUS_DEFAULT
        self.weight   = 1          # default movement cost
        self.reward   = 0          # for Q-Learning
        self.visits   = 0          # visit counter (Q-Learning)

        # pathfinding helpers
        self.distance        = math.inf
        self.total_distance  = math.inf   # used by A*
        self.heuristic_dist  = None
        self.direction       = None
        self.previous_node   = None

        # Q-Learning
        self.q_value = 0.0

    def reset(self):
        """Return node to its default state (keeps start/finish markers)."""
        if self.status not in (STATUS_START, STATUS_FINISH, STATUS_WALL):
            self.status = STATUS_DEFAULT
        self.distance       = math.inf
        self.total_distance = math.inf
        self.heuristic_dist = None
        self.direction      = None
        self.previous_node  = None

    def hard_reset(self):
        """Completely reset the node (used when clearing walls)."""
        if self.status not in (STATUS_START, STATUS_FINISH):
            self.status = STATUS_DEFAULT
            self.weight  = 1
            self.reward  = 0
            self.visits  = 0
        self.distance       = math.inf
        self.total_distance = math.inf
        self.heuristic_dist = None
        self.direction      = None
        self.previous_node  = None

    def __repr__(self):
        return f"Node({self.row},{self.col},{self.status})"


def build_grid(rows: int = ROWS, cols: int = COLS,
               start=(5, 5), finish=(24, 24)) -> list:
    """Return a 2-D list of Node objects."""
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            node = Node(r, c, rows, cols)
            if (r, c) == start:
                node.status = STATUS_START
            elif (r, c) == finish:
                node.status  = STATUS_FINISH
                node.reward  = 100
            row.append(node)
        grid.append(row)
    return grid


def get_all_nodes(grid: list) -> list:
    """Flatten the 2-D grid into a 1-D list of Nodes."""
    return [node for row in grid for node in row]


def get_shortest_path(finish_node: Node) -> list:
    """
    Backtrack from *finish_node* through previous_node pointers.
    Returns the list of nodes on the shortest path
    (excluding start and finish).
    """
    path = []
    current = finish_node
    if current.previous_node is None:
        return path
    current = current.previous_node
    while current is not None:
        if current.previous_node is None:   # reached start node
            break
        path.insert(0, current)
        current = current.previous_node
    return path
