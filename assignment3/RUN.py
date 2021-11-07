from algorithm import RepeatedForwardAStar, AStar
from maze import Cell, Maze

import math

def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


def main():
    dim=10
    maze=Maze(dim, dim)
    maze.initialize_maze(0)
    start_cell = Cell((0, 0))
    test=RepeatedForwardAStar(maze=maze, heuristic=euclidean_heuristic)
    path=test.search(start_cell)
    print(path)

if __name__ == '__main__':
    main()