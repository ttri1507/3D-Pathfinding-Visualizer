"""
algorithms/qlearning.py – Tabular Q-Learning for pathfinding.

The agent learns a Q-table where each cell stores the cumulative
reward estimate. After training, the optimal policy is extracted by
following the greedy (max Q-value) path from start to finish.
"""
import random
import math


# Maximum number of Q-table snapshots stored for visualisation playback.
# Kept low to avoid unbounded memory growth during long training runs.
MAX_ANIMATION_SNAPSHOTS = 1300

ACTIONS = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}


class QLearningAgent:
    def __init__(self, grid, rows: int, cols: int,
                 start: tuple, finish: tuple,
                 learning_rate: float = 0.2,
                 discount: float = 0.8,
                 curiosity: float = 0.8,
                 epochs: int = 350_000):

        self.grid          = grid
        self.rows          = rows
        self.cols          = cols
        self.start         = start    # (row, col)
        self.finish        = finish
        self.lr            = learning_rate
        self.discount      = discount
        self.curiosity     = curiosity
        self.epochs        = epochs

        # Q-table: one value per cell (simplified – state → value)
        self.q_table = [[0.0] * cols for _ in range(rows)]
        # snapshots for playback
        self.records: list = []

    # ------------------------------------------------------------------
    def train(self):
        """Run Q-Learning for *epochs* steps and populate self.records."""
        self.records = []
        all_states = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        i = 0
        while i < self.epochs:
            if len(self.records) > MAX_ANIMATION_SNAPSHOTS:
                break
            # choose starting state: random for first 60 %, start for last 40 %
            if i > 0.6 * self.epochs:
                state = list(self.start)
            else:
                s = random.choice(all_states)
                state = list(s)

            while not (state[0] == self.finish[0] and state[1] == self.finish[1]):
                node = self.grid[state[0]][state[1]]
                if node.status == "wall":
                    break

                eps = self.curiosity if i <= 0.9 * self.epochs else 0.4
                action = self._choose_action(state, eps)
                dy, dx = ACTIONS[action]
                next_state = [state[0] + dy, state[1] + dx]

                current_q = self.q_table[state[0]][state[1]]
                max_q     = self.q_table[next_state[0]][next_state[1]]
                reward    = self.grid[next_state[0]][next_state[1]].reward
                td        = reward + self.discount * (max_q - current_q)
                self.q_table[state[0]][state[1]] = round(current_q + self.lr * td, 2)

                self.grid[state[0]][state[1]].visits += 1
                state = next_state
                i += 1

            self.records.append(self._snapshot())

    # ------------------------------------------------------------------
    def optimal_policy(self) -> list:
        """Return the greedy path from start to finish as list of (row,col)."""
        path   = [list(self.start)]
        state  = list(self.start)
        visited_set = set()
        visited_set.add((state[0], state[1]))

        while not (state[0] == self.finish[0] and state[1] == self.finish[1]):
            candidates = {}
            for name, (dy, dx) in ACTIONS.items():
                ns = [state[0] + dy, state[1] + dx]
                if self._is_valid(ns) and (ns[0], ns[1]) not in visited_set:
                    candidates[name] = self.q_table[ns[0]][ns[1]]
            if not candidates:
                break
            best_action = max(candidates, key=candidates.get)
            dy, dx = ACTIONS[best_action]
            state = [state[0] + dy, state[1] + dx]
            visited_set.add((state[0], state[1]))
            path.append(list(state))

        return path

    # ------------------------------------------------------------------
    def _choose_action(self, state: list, epsilon: float) -> str:
        if random.random() < epsilon:
            # explore
            options = list(ACTIONS.keys())
            random.shuffle(options)
            for action in options:
                dy, dx = ACTIONS[action]
                ns = [state[0] + dy, state[1] + dx]
                if self._is_valid(ns):
                    return action
        # greedy
        candidates = {}
        for name, (dy, dx) in ACTIONS.items():
            ns = [state[0] + dy, state[1] + dx]
            if self._is_valid(ns):
                candidates[name] = self.q_table[ns[0]][ns[1]]
        if not candidates:
            # fall back to any valid action
            for name, (dy, dx) in ACTIONS.items():
                ns = [state[0] + dy, state[1] + dx]
                if self._is_valid(ns):
                    return name
        return max(candidates, key=candidates.get)

    def _is_valid(self, state: list) -> bool:
        r, c = state
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return False
        return self.grid[r][c].status != "wall"

    def _snapshot(self) -> list:
        """Return a deep copy of the Q-table for animation."""
        return [row[:] for row in self.q_table]
