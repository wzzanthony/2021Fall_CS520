import json
import os
import math

from config import create_data_num, dim
from maze import Maze, Cell
from A_star import AStar
from algorithm import RepeatedForwardAStar




def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))

def main():
    current_path=os.path.dirname(os.path.abspath(__file__))
    current_path=os.path.join(current_path,"Data")
    if not os.path.exists(current_path):
        os.mkdir(current_path)
    origin_maze_path=os.path.join(current_path,'origin_maze')
    if not os.path.exists(origin_maze_path):
        os.mkdir(origin_maze_path)
    train_data_path=os.path.join(current_path,'train_data')
    if not os.path.exists(train_data_path):
        os.mkdir(train_data_path)
    random_seed=0
    i=0
    while i<=create_data_num:
        maze=Maze(width=dim, height=dim)
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



        origin_maze=maze.data
        maze_string=json.dumps(origin_maze)
        origin_maze_file=os.path.join(origin_maze_path,str(i)+'.json')
        with open(origin_maze_file, 'w') as data_writer:
            data_writer.write(maze_string)



        repeat_Astar=RepeatedForwardAStar(maze, euclidean_heuristic)
        train_data=repeat_Astar.search(start_cell,goal_cell)
        train_data_str=json.dumps(train_data)
        train_data_file=os.path.join(train_data_path,str(i)+'.json')
        with open(train_data_file,'w') as data_writer:
            data_writer.write(train_data_str)




if __name__ == '__main__':
    main()