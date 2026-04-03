# 3D Pathfinding Visualiser – Python Edition

A full Python rewrite of the React/Three.js 3D Pathfinding Visualiser,
built with **pygame**.

## Features

| Feature | Details |
|---|---|
| **Algorithms** | BFS, DFS, Dijkstra, A\*, Q-Learning |
| **Maze generators** | Random Maze, Recursive Division |
| **Interactive grid** | Left-click / drag to toggle walls |
| **Speed control** | Fast / Medium / Slow |
| **Q-Learning sidebar** | Configurable epochs, learning rate, discount, curiosity, start/finish positions |

## Requirements

- Python ≥ 3.10
- pygame ≥ 2.1

## Install & Run

```bash
cd python_visualizer
pip install -r requirements.txt
python main.py
```

## Controls

| Action | How |
|---|---|
| Toggle wall | Left-click a cell |
| Draw walls | Hold left mouse button and drag |
| Pick algorithm | "Algorithm" dropdown (top-left) |
| Generate maze | "Maze" dropdown |
| Set speed | "Speed" dropdown |
| Run visualisation | **Visualize** button |
| Clear path | **Clear Path** button |
| Clear walls | **Clear Walls** button |
| Full reset | **Reset Grid** button |
| Train Q-agent | Fill sidebar settings → **Train Agent** |
| Show Q-policy | **Show Policy** (enabled after training) |
| Reset Q-memory | **Reset Agent** |

## Colour Legend

| Colour | Meaning |
|---|---|
| 🟩 Green | Start node |
| 🟥 Red | Finish node |
| ⬜ White | Unvisited |
| 🟦 Dark blue | Wall |
| 🟣 Purple | Visited |
| 🟨 Yellow | Shortest path |

## Project Structure

```
python_visualizer/
├── main.py                   # pygame app, UI, animation loop
├── grid.py                   # Node + Grid data structures
├── algorithms/
│   ├── weighted.py           # Dijkstra & A*
│   ├── unweighted.py         # BFS & DFS
│   └── qlearning.py          # Tabular Q-Learning agent
├── maze/
│   └── generators.py         # Random Maze & Recursive Division
└── requirements.txt
```
