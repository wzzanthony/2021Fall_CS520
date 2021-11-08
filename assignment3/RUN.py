from algorithm import RepeatedForwardAStar, AStar
from maze import Cell, Maze

import math
import time

def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


def main():
    dim=101
    maze=Maze(dim, dim)
    maze.initialize_maze(0)
    path=[]

    while len(path)==0:
        maze.update_target_position()
        astar_search = AStar(maze, euclidean_heuristic)
        start_cell = Cell((0, 0))
        goal_cell=Cell(maze.target_position)
        path=astar_search.search(start_cell=start_cell, goal_cell=goal_cell)

    test=RepeatedForwardAStar(maze=maze, heuristic=euclidean_heuristic)
    print("start")
    st=time.time()
    path=test.search(start_cell)
    print(path)
    print(maze.target_position)
    print("time:", time.time()-st)

if __name__ == '__main__':
    main()