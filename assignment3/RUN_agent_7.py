from algorithm_agent_7 import RepeatedForwardAStar, AStar
from maze_agent_7 import Cell, Maze
from config import repeat_creating_maze, dim_of_maze

import math
import time
import json
import os

def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


def main():
    maze=Maze(dim_of_maze, dim_of_maze)
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


def test(save_path, dim):
    data_dict={}
    count=0
    i=0
    while True:
        count+=1
        if count==repeat_creating_maze:
            break
        while True:
            maze=Maze(dim, dim)
            maze.initialize_maze(random_seed=i)
            astar_search = AStar(maze, euclidean_heuristic)
            start_cell = Cell((0, 0))
            goal_cell = Cell((dim-1, dim-1))
            path=astar_search.search(start_cell=start_cell, goal_cell=goal_cell)
            i+=1
            if len(path)!=0:
                break
        print("create maze")
        path=[]
        while len(path) == 0:
            maze.update_target_position()
            astar_search = AStar(maze, euclidean_heuristic)
            start_cell = Cell((0, 0))
            goal_cell = Cell(maze.target_position)
            path = astar_search.search(start_cell=start_cell, goal_cell=goal_cell)
        print("create target")
        test = RepeatedForwardAStar(maze=maze, heuristic=euclidean_heuristic)
        st=time.time()
        repeat_times, total_path_length, total_exam_times=test.search(start_cell=start_cell)
        et=time.time()-st
        data_dict[str(count)] = {}
        data_dict[str(count)]["A*_times"] = repeat_times
        data_dict[str(count)]["time"] = et
        data_dict[str(count)]["total_path_length"] = total_path_length
        data_dict[str(count)]["total_exam_times"] = total_exam_times

        with open(save_path, "w") as file_writer:
            file_writer.write(json.dumps(data_dict))



if __name__ == '__main__':
    current_path=os.path.dirname(os.path.abspath(__file__))
    save_file_path=os.path.join(current_path, "agent_7")
    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)
    file_dir=os.path.join(save_file_path, "agent_7.json")
    test(file_dir, dim=dim_of_maze)