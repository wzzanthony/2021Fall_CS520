import sys
import time
import math
from tqdm import tqdm
import json
import os


from project2_config import PS, NUM_EXPERIMENT
from algorithm import RepeatedForwardAStar

sys.path.append('..\\..\\XLQ_test\\Final_version')
from algorithm import AStar
from maze_1 import Cell, Maze


def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


def format_path_data(data_path):
    formated_path=[]
    for data in data_path:
        formated_path.append(data.get_position())
    return formated_path

def test(dim: int, num_experiment=NUM_EXPERIMENT):
    # data=[0]*len(Q9_QS)
    # print(Q9_QS)
    # print(data)
    data_time_bump = [0.0] * len(PS)
    data_path_bump = [0.0] * len(PS)
    data_time_repeat = [0.0] * len(PS)
    data_path_repeat = [0.0] * len(PS)

    record_path_in_bump=[]
    record_path_in_repeat=[]

    for index_p, p in enumerate(PS):
        count = 0
        sub_record_path_in_bump=[]
        sub_record_path_in_repeat = []
        for random_seed in tqdm(range(num_experiment)):
            maze = Maze(dim, dim)
            maze.initialize_maze(p, random_seed=random_seed)
            goal_cell = Cell((dim - 1, dim - 1))
            start_cell = Cell((0, 0))
            astar_search=AStar(maze, euclidean_heuristic)
            astar_path=astar_search.search(start_cell, goal_cell)
            if len(astar_path)==0:
                continue
            count+=1
            repeat_forward_astar=RepeatedForwardAStar(maze, euclidean_heuristic)

            start_time=time.time()
            path_repeat=repeat_forward_astar.search(start_cell, goal_cell)
            time_repeat=time.time()-start_time

            start_time=time.time()
            path_bump=repeat_forward_astar.search(start_cell, goal_cell, only_bump=True)
            time_bump=time.time()-start_time

            data_time_repeat[index_p]+=time_repeat
            data_time_bump[index_p]+=time_bump

            data_path_repeat[index_p]+=len(path_repeat)
            data_path_bump[index_p]+=len(path_bump)

            sub_record_path_in_repeat.append(format_path_data(path_repeat))
            sub_record_path_in_bump.append(format_path_data(path_bump))

        record_path_in_repeat.append(sub_record_path_in_repeat)
        record_path_in_bump.append(sub_record_path_in_bump)

        if count!=0:
            # average_data_time=data_time/count
            data_path_repeat[index_p]=data_path_repeat[index_p]/count
            data_path_bump[index_p]=data_path_bump[index_p]/count
            data_time_repeat[index_p]=data_time_repeat[index_p]/count
            data_time_bump[index_p]=data_time_bump[index_p]/count
    return data_path_repeat, data_time_repeat, data_path_bump, data_time_bump,record_path_in_repeat, record_path_in_bump




if __name__ == '__main__':
    dim = 101
    data_path_repeat, data_time_repeat, data_path_bump, data_time_bump,record_path_in_repeat, record_path_in_bump = test(dim, NUM_EXPERIMENT)
    path="D:\\520\\data"

    filename="repeatforwardAStar_path_average.json"
    current_path=os.path.join(path,filename)
    with open(current_path,"w") as file_writer:
        data=json.dumps(data_path_repeat)
        file_writer.write(data)

    filename="bumprepeatforwardAStar_path_average.json"
    current_path = os.path.join(path, filename)
    with open(current_path, "w") as file_writer:
        data = json.dumps(data_path_bump)
        file_writer.write(data)

    filename = "repeatforwardAStar_time_average.json"
    current_path = os.path.join(path, filename)
    with open(current_path, "w") as file_writer:
        data = json.dumps(data_time_repeat)
        file_writer.write(data)

    filename = "bumprepeatforwardAStar_time_average.json"
    current_path = os.path.join(path, filename)
    with open(current_path, "w") as file_writer:
        data = json.dumps(data_time_bump)
        file_writer.write(data)

    filename = "repeatforwardAStar_path.json"
    current_path = os.path.join(path, filename)
    with open(current_path, "w") as file_writer:
        data = json.dumps(record_path_in_repeat)
        file_writer.write(data)

    filename = "bumprepeatforwardAStar_path.json"
    current_path = os.path.join(path, filename)
    with open(current_path, "w") as file_writer:
        data = json.dumps(record_path_in_bump)
        file_writer.write(data)


