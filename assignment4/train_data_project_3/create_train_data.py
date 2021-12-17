import json
import os
import math
import time

from config import create_data_num, dim
from maze_agent_7 import Maze, Cell
from algorithm_agent_7_for_data import AStar, RepeatedForwardAStar




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
    compare_data={}
    file_path = os.path.join(current_path, 'data.json')
    while i<=create_data_num:
        maze=Maze(width=dim, height=dim)
        maze.initialize_maze(random_seed=random_seed)
        goal_cell = Cell((dim - 1, dim - 1))
        start_cell = Cell((0, 0))
        AStar_search = AStar(maze, euclidean_heuristic)
        AStar_search_path = AStar_search.search(start_cell, goal_cell)
        random_seed+=1
        if len(AStar_search_path) == 0:
            continue
        path=[]
        while len(path) == 0:
            maze.update_target_position()
            astar_search = AStar(maze, euclidean_heuristic)
            start_cell = Cell((0, 0))
            goal_cell = Cell(maze.target_position)
            path = astar_search.search(start_cell=start_cell, goal_cell=goal_cell)
        i+=1
        print("print data line "+str(i))


        origin_data={}
        origin_data['maze']=maze.data
        origin_data['target_position']=maze.target_position
        origin_data['terrain_maze']=maze.maze_terrain
        maze_string=json.dumps(origin_data)
        origin_maze_file=os.path.join(origin_maze_path,str(i)+'.json')
        with open(origin_maze_file, 'w') as data_writer:
            data_writer.write(maze_string)


        # current_path=os.path.dirname(os.path.abspath("__file__"))
        # file_path=os.path.join(current_path, 'data.json')
        # repeat_Astar=RepeatedForwardAStar(maze, euclidean_heuristic)
        # train_data=repeat_Astar.search(start_cell,goal_cell)
        # train_data_str=json.dumps(train_data)
        # train_data_file=os.path.join(train_data_path,str(i)+'.json')
        # with open(train_data_file,'w') as data_writer:
        #     data_writer.write(train_data_str)



        repeat_Astar=RepeatedForwardAStar(maze, euclidean_heuristic)
        st=time.time()
        _, path,_=repeat_Astar.search(start_cell,goal_cell)
        ed=time.time()-st
        compare_data[str(i)]={}
        compare_data[str(i)]['origin']={}
        compare_data[str(i)]['mimic']={}
        compare_data[str(i)]['origin']['length']=path
        compare_data[str(i)]['origin']['time'] = ed
        compare_data[str(i)]['path']=origin_maze_file

    compare_data_str=json.dumps(compare_data)
    with open(file_path, 'w') as a:
        a.write(compare_data_str)



if __name__ == '__main__':
    main()