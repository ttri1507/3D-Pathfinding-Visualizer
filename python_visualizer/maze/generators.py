"""
maze/generators.py – Random maze and Recursive Division maze generators.
"""
import random


def random_maze(grid, nodes_to_animate: list, wall_probability: float = 0.25):
    """
    Place walls randomly.  Nodes with status "start" or "finish" are skipped.
    """
    reserved = {"start", "finish"}
    for row in grid:
        for node in row:
            if node.status in reserved:
                continue
            if random.random() < wall_probability:
                node.weight = 0
                nodes_to_animate.append(node)


def recursive_division(grid, row_start: int, row_end: int,
                       col_start: int, col_end: int,
                       orientation: str, surrounding_walls: bool,
                       nodes_to_animate: list):
    """
    Recursive-division maze.  Appends wall nodes to *nodes_to_animate*.
    """
    if row_end < row_start or col_end < col_start:
        return

    reserved = {"start", "finish"}

    def make_wall(node):
        if node.status in reserved:
            return
        node.weight = 0
        nodes_to_animate.append(node)

    rows = len(grid)
    cols = len(grid[0])

    if not surrounding_walls:
        # Border walls
        for j in range(cols):
            make_wall(grid[0][j])
            make_wall(grid[rows - 1][j])
        for i in range(1, rows - 1):
            make_wall(grid[i][0])
            make_wall(grid[i][cols - 1])
        surrounding_walls = True

    if orientation == "horizontal":
        possible_rows = list(range(row_start, row_end + 1, 2))
        possible_cols = list(range(col_start - 1, col_end + 2, 2))
        if not possible_rows or not possible_cols:
            return
        current_row = random.choice(possible_rows)
        col_gap     = random.choice(possible_cols)

        for j in range(col_start - 1, col_end + 2):
            if j != col_gap and 0 <= j < cols:
                make_wall(grid[current_row][j])

        if current_row - 2 - row_start > col_end - col_start:
            recursive_division(grid, row_start, current_row - 2,
                               col_start, col_end, orientation,
                               surrounding_walls, nodes_to_animate)
        else:
            recursive_division(grid, row_start, current_row - 2,
                               col_start, col_end, "vertical",
                               surrounding_walls, nodes_to_animate)

        if row_end - (current_row + 2) > col_end - col_start:
            recursive_division(grid, current_row + 2, row_end,
                               col_start, col_end, orientation,
                               surrounding_walls, nodes_to_animate)
        else:
            recursive_division(grid, current_row + 2, row_end,
                               col_start, col_end, "vertical",
                               surrounding_walls, nodes_to_animate)
    else:
        possible_cols = list(range(col_start, col_end + 1, 2))
        possible_rows = list(range(row_start - 1, row_end + 2, 2))
        if not possible_cols or not possible_rows:
            return
        current_col = random.choice(possible_cols)
        row_gap     = random.choice(possible_rows)

        for i in range(row_start - 1, row_end + 2):
            if i != row_gap and 0 <= i < rows:
                make_wall(grid[i][current_col])

        if row_end - row_start > current_col - 2 - col_start:
            recursive_division(grid, row_start, row_end,
                               col_start, current_col - 2, "horizontal",
                               surrounding_walls, nodes_to_animate)
        else:
            recursive_division(grid, row_start, row_end,
                               col_start, current_col - 2, orientation,
                               surrounding_walls, nodes_to_animate)

        if row_end - row_start > col_end - (current_col + 2):
            recursive_division(grid, row_start, row_end,
                               current_col + 2, col_end, "horizontal",
                               surrounding_walls, nodes_to_animate)
        else:
            recursive_division(grid, row_start, row_end,
                               current_col + 2, col_end, orientation,
                               surrounding_walls, nodes_to_animate)
