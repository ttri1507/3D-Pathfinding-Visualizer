"""
main.py – 3D Pathfinding Visualiser (Python / pygame edition)
=============================================================
Controls
--------
Left-click a cell   : toggle wall
Right-click a cell  : (reserved for future dragging of start/finish)
Toolbar buttons     : select algorithm, maze, speed; visualise; clear

Run:  python main.py
"""
import sys
import math
import threading
import time
import pygame

# ── local modules ────────────────────────────────────────────────────────────
from grid import (build_grid, get_shortest_path,
                  STATUS_DEFAULT, STATUS_START, STATUS_FINISH,
                  STATUS_WALL, STATUS_VISITED, STATUS_PATH)
from algorithms.weighted   import weighted_search
from algorithms.unweighted import unweighted_search
from algorithms.qlearning  import QLearningAgent
from maze.generators       import random_maze, recursive_division

# ── constants ────────────────────────────────────────────────────────────────
ROWS = 30
COLS = 30

TOOLBAR_H   = 60          # pixels reserved for the top toolbar
SIDEBAR_W   = 220         # right-side panel (Q-Learning settings)
WINDOW_W    = 900
WINDOW_H    = TOOLBAR_H + 700
CELL_SIZE   = (WINDOW_W - SIDEBAR_W) // COLS   # ≈ 22 px

GRID_OFFSET_X = 0
GRID_OFFSET_Y = TOOLBAR_H

# Colours (R, G, B)
C_BG           = (30,  30,  40)
C_TOOLBAR      = (20,  20,  32)
C_SIDEBAR      = (25,  25,  38)
C_GRID_LINE    = (60,  60,  80)
C_DEFAULT      = (255, 255, 255)
C_WALL         = (28,  28, 115)
C_START        = (0,   220,  60)
C_FINISH       = (220,  40,  40)
C_VISITED      = (84,   69, 247)
C_VISITED_WAVE = (255,  82, 200)
C_PATH         = (255, 255,   0)
C_BTN_NORMAL   = (30, 136, 229)
C_BTN_HOVER    = (21,  96, 162)
C_BTN_DISABLED = (80,  80, 100)
C_TEXT         = (255, 255, 255)
C_LABEL        = (180, 180, 200)
C_SUCCESS      = (50,  205,  50)
C_ERROR        = (220,  50,  50)

START_DEFAULT  = (5,  5)
FINISH_DEFAULT = (24, 24)

SPEED_MAP = {"Fast": 15, "Medium": 30, "Slow": 80}   # ms per step


# ── tiny Button widget ───────────────────────────────────────────────────────
class Button:
    def __init__(self, rect, label, *, disabled=False, toggle=False):
        self.rect     = pygame.Rect(rect)
        self.label    = label
        self.disabled = disabled
        self.toggle   = toggle
        self.active   = False   # for toggle buttons

    def draw(self, surface, font):
        hovered = self.rect.collidepoint(pygame.mouse.get_pos())
        if self.disabled:
            colour = C_BTN_DISABLED
        elif (self.toggle and self.active) or (not self.toggle and hovered):
            colour = C_BTN_HOVER
        else:
            colour = C_BTN_NORMAL
        pygame.draw.rect(surface, colour, self.rect, border_radius=6)
        txt = font.render(self.label, True, C_TEXT)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def is_clicked(self, pos):
        return not self.disabled and self.rect.collidepoint(pos)


# ── Dropdown widget ──────────────────────────────────────────────────────────
class Dropdown:
    def __init__(self, rect, options, placeholder="Select…"):
        self.rect        = pygame.Rect(rect)
        self.options     = options
        self.placeholder = placeholder
        self.selected    = None
        self.open        = False
        self._item_h     = 26

    def draw(self, surface, font):
        pygame.draw.rect(surface, C_BTN_NORMAL, self.rect, border_radius=5)
        label = self.selected if self.selected else self.placeholder
        txt   = font.render(label, True, C_TEXT)
        surface.blit(txt, txt.get_rect(center=self.rect.center))
        if self.open:
            for i, opt in enumerate(self.options):
                r = pygame.Rect(self.rect.x,
                                self.rect.bottom + i * self._item_h,
                                self.rect.width,
                                self._item_h)
                bg = C_BTN_HOVER if r.collidepoint(pygame.mouse.get_pos()) else C_TOOLBAR
                pygame.draw.rect(surface, bg, r)
                pygame.draw.rect(surface, C_GRID_LINE, r, 1)
                t = font.render(opt, True, C_TEXT)
                surface.blit(t, t.get_rect(center=r.center))

    def handle_click(self, pos):
        if self.rect.collidepoint(pos):
            self.open = not self.open
            return None
        if self.open:
            for i, opt in enumerate(self.options):
                r = pygame.Rect(self.rect.x,
                                self.rect.bottom + i * self._item_h,
                                self.rect.width,
                                self._item_h)
                if r.collidepoint(pos):
                    self.selected = opt
                    self.open = False
                    return opt
            self.open = False
        return None


# ── InputBox widget ──────────────────────────────────────────────────────────
class InputBox:
    def __init__(self, rect, default=""):
        self.rect   = pygame.Rect(rect)
        self.value  = str(default)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.value = self.value[:-1]
            elif event.unicode.isprintable():
                self.value += event.unicode

    def draw(self, surface, font):
        border = C_BTN_NORMAL if self.active else C_GRID_LINE
        pygame.draw.rect(surface, C_SIDEBAR, self.rect)
        pygame.draw.rect(surface, border, self.rect, 2)
        txt = font.render(self.value, True, C_TEXT)
        surface.blit(txt, (self.rect.x + 4, self.rect.y + 4))

    def int_value(self, default=0):
        try:
            return int(self.value)
        except ValueError:
            return default

    def float_value(self, default=0.0):
        try:
            return float(self.value)
        except ValueError:
            return default


# ── Visualiser application ───────────────────────────────────────────────────
class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("3D Pathfinding Visualiser – Python")
        self.clock  = pygame.font.Font(None, 20)
        self.font   = pygame.font.SysFont("segoeui", 15, bold=True)
        self.sfont  = pygame.font.SysFont("segoeui", 13)
        self.clock_ = pygame.time.Clock()

        self.grid  = build_grid(ROWS, COLS, START_DEFAULT, FINISH_DEFAULT)
        self.start  = START_DEFAULT
        self.finish = FINISH_DEFAULT

        self.running      = False   # algorithm is running
        self.status_msg   = "Select an algorithm and press Visualize."
        self.status_color = C_LABEL

        # animation queues
        self._visited_queue: list = []   # nodes to colour visited (with delay)
        self._path_queue:    list = []
        self._maze_queue:    list = []
        self._ql_records:    list = []   # Q-Learning playback snapshots
        self._ql_policy:     list = []   # optimal policy path

        self._step_delay   = SPEED_MAP["Fast"]
        self._last_step_ms = 0
        self._phase        = "idle"   # idle | visiting | pathing | maze | ql_play | policy

        # Q-Learning agent (lazy)
        self._agent: QLearningAgent | None = None

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        bw, bh = 110, 32
        pad     = 6
        y       = (TOOLBAR_H - bh) // 2

        self.dd_algo  = Dropdown((pad, y, 130, bh),
                                 ["Dijkstra", "A* Search", "BFS", "DFS", "Q-Learning"],
                                 "Algorithm")
        self.dd_maze  = Dropdown((pad + 136, y, 130, bh),
                                 ["Random Maze", "Recursive Division"],
                                 "Maze")
        self.dd_speed = Dropdown((pad + 272, y, 100, bh),
                                 ["Fast", "Medium", "Slow"],
                                 "Speed")

        x = pad + 380
        self.btn_viz    = Button((x,         y, bw, bh), "Visualize")
        self.btn_clrp   = Button((x + bw + pad, y, bw, bh), "Clear Path")
        self.btn_clrw   = Button((x + 2*(bw+pad), y, bw, bh), "Clear Walls")
        self.btn_reset  = Button((x + 3*(bw+pad), y, bw, bh), "Reset Grid")

        # ── Sidebar (Q-Learning settings) ────────────────────────────────────
        sx  = WINDOW_W - SIDEBAR_W + 8
        iy  = TOOLBAR_H + 10
        iw  = SIDEBAR_W - 16
        ih  = 22

        def _ib(row, default):
            return InputBox((sx, iy + row * 34, iw, ih), default)

        self.ib_epochs    = _ib(1,  350000)
        self.ib_lr        = _ib(3,  0.2)
        self.ib_discount  = _ib(5,  0.8)
        self.ib_curiosity = _ib(7,  0.8)
        self.ib_sr        = _ib(9,  5)
        self.ib_sc        = _ib(10, 5)
        self.ib_fr        = _ib(12, 24)
        self.ib_fc        = _ib(13, 24)

        by = iy + 15 * 34
        self.btn_train   = Button((sx, by,        iw, bh), "Train Agent")
        self.btn_policy  = Button((sx, by + bh+6, iw, bh), "Show Policy", disabled=True)
        self.btn_rst_agt = Button((sx, by + 2*(bh+6), iw, bh), "Reset Agent")

        self._input_boxes = [self.ib_epochs, self.ib_lr, self.ib_discount,
                             self.ib_curiosity, self.ib_sr, self.ib_sc,
                             self.ib_fr, self.ib_fc]

    # ── helpers ──────────────────────────────────────────────────────────────
    def _cell_at(self, pos):
        """Return (row, col) for a screen position, or None if outside grid."""
        x, y = pos
        gx   = GRID_OFFSET_X
        gy   = GRID_OFFSET_Y
        gw   = COLS * CELL_SIZE
        gh   = ROWS * CELL_SIZE
        if gx <= x < gx + gw and gy <= y < gy + gh:
            col = (x - gx) // CELL_SIZE
            row = (y - gy) // CELL_SIZE
            if 0 <= row < ROWS and 0 <= col < COLS:
                return row, col
        return None

    def _cell_colour(self, node):
        s = node.status
        if s == STATUS_START:   return C_START
        if s == STATUS_FINISH:  return C_FINISH
        if s == STATUS_WALL:    return C_WALL
        if s == STATUS_PATH:    return C_PATH
        if s == STATUS_VISITED: return C_VISITED
        return C_DEFAULT

    def _set_status(self, msg, colour=C_LABEL):
        self.status_msg   = msg
        self.status_color = colour

    def _reset_path_state(self):
        """Clear path/visited colouring but keep walls and start/finish."""
        for row in self.grid:
            for node in row:
                if node.status in (STATUS_VISITED, STATUS_PATH):
                    node.status = STATUS_DEFAULT
                node.distance       = math.inf
                node.total_distance = math.inf
                node.heuristic_dist = None
                node.previous_node  = None

    def _clear_walls(self):
        for row in self.grid:
            for node in row:
                node.hard_reset()
        self._set_status("Walls cleared.")

    def _reset_grid(self):
        self.grid   = build_grid(ROWS, COLS, self.start, self.finish)
        self._agent = None
        self.btn_policy.disabled = True
        self._visited_queue = []
        self._path_queue    = []
        self._maze_queue    = []
        self._ql_records    = []
        self._ql_policy     = []
        self._phase         = "idle"
        self.running        = False
        self._set_status("Grid reset.")

    # ── algorithm runners ────────────────────────────────────────────────────
    def _run_algorithm(self):
        if self.running:
            return
        selected = self.dd_algo.selected
        if not selected:
            self._set_status("Please select an algorithm first.", C_ERROR)
            return

        self._reset_path_state()
        self.running = True
        nodes_visited = []

        start_node  = self.grid[self.start[0]][self.start[1]]
        finish_node = self.grid[self.finish[0]][self.finish[1]]

        if selected == "Q-Learning":
            if self._agent is None or not self._agent.records:
                self._set_status("Train the agent first (sidebar).", C_ERROR)
                self.running = False
                return
            self._ql_records = list(self._agent.records)
            self._phase      = "ql_play"
            self._set_status("Playing Q-Learning heat-map…", C_VISITED)
            return

        name_map = {
            "Dijkstra":  ("Dijkstra",  "weighted"),
            "A* Search": ("aStar",     "weighted"),
            "BFS":       ("BFS",       "unweighted"),
            "DFS":       ("DFS",       "unweighted"),
        }
        algo_name, algo_type = name_map[selected]

        if algo_type == "weighted":
            found = weighted_search(self.grid, start_node, finish_node,
                                    nodes_visited, algo_name)
        else:
            found = unweighted_search(self.grid, start_node, finish_node,
                                      nodes_visited, algo_name)

        path = get_shortest_path(finish_node) if found else []

        self._visited_queue = list(nodes_visited)
        self._path_queue    = list(path)
        self._phase         = "visiting"
        self._last_step_ms  = pygame.time.get_ticks()
        self._set_status("Visualizing…", C_VISITED)
        if not found:
            self._set_status("No path found!", C_ERROR)

    def _apply_maze(self, maze_name: str):
        if self.running:
            return
        self._reset_path_state()
        nodes = []
        if maze_name == "Random Maze":
            random_maze(self.grid, nodes)
        else:
            recursive_division(
                self.grid, 2, ROWS - 3, 2, COLS - 3,
                "horizontal", False, nodes)
        self._maze_queue = list(nodes)
        self._phase      = "maze"
        self._last_step_ms = pygame.time.get_ticks()
        self._set_status("Generating maze…", C_LABEL)

    def _train_agent(self):
        if self.running:
            return
        self._set_status("Training agent… (please wait)", C_VISITED)
        pygame.display.flip()

        sr = self.ib_sr.int_value(5)
        sc = self.ib_sc.int_value(5)
        fr = self.ib_fr.int_value(24)
        fc = self.ib_fc.int_value(24)

        # apply start/finish positions from sidebar
        old_start_node  = self.grid[self.start[0]][self.start[1]]
        old_finish_node = self.grid[self.finish[0]][self.finish[1]]
        old_start_node.status  = STATUS_DEFAULT
        old_start_node.reward  = 0
        old_finish_node.status = STATUS_DEFAULT
        old_finish_node.reward = 0

        self.start  = (sr, sc)
        self.finish = (fr, fc)
        self.grid[sr][sc].status  = STATUS_START
        self.grid[fr][fc].status  = STATUS_FINISH
        self.grid[fr][fc].reward  = 100

        self._agent = QLearningAgent(
            grid          = self.grid,
            rows          = ROWS,
            cols          = COLS,
            start         = self.start,
            finish        = self.finish,
            learning_rate = self.ib_lr.float_value(0.2),
            discount      = self.ib_discount.float_value(0.8),
            curiosity     = self.ib_curiosity.float_value(0.8),
            epochs        = self.ib_epochs.int_value(350_000),
        )

        def _train_thread():
            self._agent.train()
            self.btn_policy.disabled = False
            self._set_status("Agent trained! Press 'Show Policy'.", C_SUCCESS)

        t = threading.Thread(target=_train_thread, daemon=True)
        t.start()

    def _show_policy(self):
        if self._agent is None:
            return
        self._reset_path_state()
        self._ql_policy     = self._agent.optimal_policy()
        self._phase         = "policy"
        self._last_step_ms  = pygame.time.get_ticks()
        self._set_status("Showing optimal policy…", C_PATH)

    # ── step-based animation ─────────────────────────────────────────────────
    def _tick_animation(self):
        now = pygame.time.get_ticks()
        if now - self._last_step_ms < self._step_delay:
            return
        self._last_step_ms = now

        if self._phase == "visiting":
            if self._visited_queue:
                node = self._visited_queue.pop(0)
                if node.status not in (STATUS_START, STATUS_FINISH):
                    node.status = STATUS_VISITED
            else:
                self._phase = "pathing"

        elif self._phase == "pathing":
            if self._path_queue:
                node = self._path_queue.pop(0)
                if node.status not in (STATUS_START, STATUS_FINISH):
                    node.status = STATUS_PATH
            else:
                self._phase  = "idle"
                self.running = False
                self._set_status("Done!", C_SUCCESS)

        elif self._phase == "maze":
            batch = 5   # apply several wall cells per tick for speed
            for _ in range(batch):
                if not self._maze_queue:
                    break
                node = self._maze_queue.pop(0)
                if node.status not in (STATUS_START, STATUS_FINISH):
                    node.status = STATUS_WALL
            if not self._maze_queue:
                self._phase = "idle"
                self._set_status("Maze generated.", C_SUCCESS)

        elif self._phase == "ql_play":
            if self._ql_records:
                snapshot = self._ql_records.pop(0)
                mn, mx = -10.0, 100.0
                for r in range(ROWS):
                    for c in range(COLS):
                        node = self.grid[r][c]
                        if node.status in (STATUS_START, STATUS_FINISH,
                                           STATUS_WALL):
                            continue
                        val = snapshot[r][c]
                        if val == 0.0:
                            continue
                        # heat-map: blue→green→red
                        ratio = 2 * (val - mn) / (mx - mn)
                        blue  = max(0, 1.0 - ratio)
                        red   = max(0, ratio - 1.0)
                        green = 1.0 - blue - red
                        node._ql_color = (int(red*255), int(green*255), int(blue*255))
            else:
                self._phase  = "idle"
                self.running = False
                self._set_status("Q-Learning visualisation done!", C_SUCCESS)

        elif self._phase == "policy":
            if len(self._ql_policy) > 1:
                step = self._ql_policy.pop(0)
                r, c = step
                node = self.grid[r][c]
                if node.status not in (STATUS_START, STATUS_FINISH):
                    node.status = STATUS_PATH
            else:
                self._phase  = "idle"
                self.running = False
                self._set_status("Optimal policy shown.", C_SUCCESS)

    # ── drawing ──────────────────────────────────────────────────────────────
    def _draw_grid(self):
        for r in range(ROWS):
            for c in range(COLS):
                node = self.grid[r][c]
                x = GRID_OFFSET_X + c * CELL_SIZE
                y = GRID_OFFSET_Y + r * CELL_SIZE
                rect = (x + 1, y + 1, CELL_SIZE - 1, CELL_SIZE - 1)

                # special Q-Learning heat colour
                if self._phase == "ql_play" and hasattr(node, "_ql_color"):
                    colour = node._ql_color
                else:
                    colour = self._cell_colour(node)

                pygame.draw.rect(self.screen, colour, rect)

    def _draw_grid_lines(self):
        gw = COLS * CELL_SIZE
        gh = ROWS * CELL_SIZE
        for c in range(COLS + 1):
            x = GRID_OFFSET_X + c * CELL_SIZE
            pygame.draw.line(self.screen, C_GRID_LINE,
                             (x, GRID_OFFSET_Y), (x, GRID_OFFSET_Y + gh))
        for r in range(ROWS + 1):
            y = GRID_OFFSET_Y + r * CELL_SIZE
            pygame.draw.line(self.screen, C_GRID_LINE,
                             (GRID_OFFSET_X, y), (GRID_OFFSET_X + gw, y))

    def _draw_toolbar(self):
        pygame.draw.rect(self.screen, C_TOOLBAR, (0, 0, WINDOW_W, TOOLBAR_H))
        self.dd_algo.draw(self.screen, self.sfont)
        self.dd_maze.draw(self.screen, self.sfont)
        self.dd_speed.draw(self.screen, self.sfont)
        self.btn_viz.disabled   = self.running
        self.btn_clrp.disabled  = self.running
        self.btn_clrw.disabled  = self.running
        self.btn_reset.disabled = self.running
        for btn in (self.btn_viz, self.btn_clrp, self.btn_clrw, self.btn_reset):
            btn.draw(self.screen, self.sfont)

        # status bar
        msg = self.font.render(self.status_msg, True, self.status_color)
        self.screen.blit(msg, (8, WINDOW_H - 22))

    def _draw_sidebar(self):
        sx = WINDOW_W - SIDEBAR_W
        pygame.draw.rect(self.screen, C_SIDEBAR,
                         (sx, TOOLBAR_H, SIDEBAR_W, WINDOW_H - TOOLBAR_H))
        iy  = TOOLBAR_H + 10
        row = 0

        def label(text, dr=0):
            nonlocal row
            row += dr
            t = self.sfont.render(text, True, C_LABEL)
            self.screen.blit(t, (sx + 8, iy + row * 34))
            row += 1

        label("── Q-Learning Settings ──")
        label("Epochs:")
        self.ib_epochs.draw(self.screen, self.sfont)
        label("Learning Rate:")
        self.ib_lr.draw(self.screen, self.sfont)
        label("Discount Factor:")
        self.ib_discount.draw(self.screen, self.sfont)
        label("Curiosity (ε):")
        self.ib_curiosity.draw(self.screen, self.sfont)
        label("Start Row / Col:")
        self.ib_sr.draw(self.screen, self.sfont)
        self.ib_sc.draw(self.screen,  self.sfont)
        label("Finish Row / Col:")
        self.ib_fr.draw(self.screen, self.sfont)
        self.ib_fc.draw(self.screen, self.sfont)

        by = iy + 15 * 34
        self.btn_train.disabled  = self.running
        self.btn_rst_agt.disabled = self.running
        for btn in (self.btn_train, self.btn_policy, self.btn_rst_agt):
            btn.draw(self.screen, self.sfont)

    # ── event handling ───────────────────────────────────────────────────────
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # input boxes
            for ib in self._input_boxes:
                ib.handle_event(event)

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos

                # dropdowns (highest priority – they render on top)
                chosen = self.dd_algo.handle_click(pos)
                chosen_m = self.dd_maze.handle_click(pos)
                chosen_s = self.dd_speed.handle_click(pos)

                if chosen_s:
                    self._step_delay = SPEED_MAP.get(chosen_s, 15)

                if chosen_m and not self.running:
                    self._apply_maze(chosen_m)
                    self.dd_maze.selected = None   # reset label

                # toolbar buttons
                if self.btn_viz.is_clicked(pos):
                    self._run_algorithm()
                elif self.btn_clrp.is_clicked(pos):
                    self._reset_path_state()
                    self._visited_queue = []
                    self._path_queue    = []
                    self._phase = "idle"
                    self.running = False
                    self._set_status("Path cleared.")
                elif self.btn_clrw.is_clicked(pos):
                    self._clear_walls()
                elif self.btn_reset.is_clicked(pos):
                    self._reset_grid()

                # sidebar buttons
                elif self.btn_train.is_clicked(pos):
                    self._train_agent()
                elif self.btn_policy.is_clicked(pos):
                    self._show_policy()
                elif self.btn_rst_agt.is_clicked(pos):
                    self._agent = None
                    self.btn_policy.disabled = True
                    self._set_status("Agent memory cleared.")

                # grid cell toggle (left click)
                elif event.button == 1 and not self.running:
                    cell = self._cell_at(pos)
                    if cell:
                        r, c = cell
                        node = self.grid[r][c]
                        if node.status == STATUS_DEFAULT:
                            node.status = STATUS_WALL
                            node.reward = -100
                        elif node.status == STATUS_WALL:
                            node.status = STATUS_DEFAULT
                            node.reward = 0

            # drag to draw/erase walls
            if pygame.mouse.get_pressed()[0] and not self.running:
                pos  = pygame.mouse.get_pos()
                cell = self._cell_at(pos)
                if cell:
                    r, c = cell
                    node = self.grid[r][c]
                    if node.status == STATUS_DEFAULT:
                        node.status = STATUS_WALL
                        node.reward = -100

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self):
        while True:
            self._handle_events()
            self._tick_animation()

            self.screen.fill(C_BG)
            self._draw_grid()
            self._draw_grid_lines()
            self._draw_toolbar()
            self._draw_sidebar()

            pygame.display.flip()
            self.clock_.tick(60)


# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().run()
