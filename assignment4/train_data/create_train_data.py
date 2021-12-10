import json
import os
import math

from config import create_data_num, dim
from maze import Maze, Cell
from A_star import AStar

def switch_p(i):
    if i<=25000:
        return 0.075
    elif 2500<i<=50000:
        return 0.15
    elif 50000<i<=75000:
        return 0.225
    else:
        return 0.3


def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))

def main():
    current_path=os.path.dirname(os.path.abspath(__file__))
    current_path=os.path.join(current_path,"Data")
    if not os.path.exists(current_path):
        os.mkdir(current_path)
    random_seed=0
    i=0
    file_name=os.path.join(current_path, 'data.txt')
    while i<=create_data_num:
        # p=switch_p(i)
        maze=Maze(width=dim, height=dim)
        # maze.initialize_maze(p, random_seed=random_seed)
        maze.initialize_maze(0.3, random_seed=random_seed)
        goal_cell = Cell((dim - 1, dim - 1))
        start_cell = Cell((0, 0))
        AStar_search = AStar(maze, euclidean_heuristic)
        AStar_search_path=AStar_search.search(start_cell, goal_cell)
        random_seed+=1
        if len(AStar_search_path) == 0:
            continue
        i+=1
        print("print data line "+str(i))
        maze_list=maze.data
        maze_string=json.dumps(maze_list)
        with open(file_name, 'a') as data_writer:
            data_writer.write(maze_string+'\n')





if __name__ == '__main__':
    main()