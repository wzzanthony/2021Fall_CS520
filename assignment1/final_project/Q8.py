import sys
from tqdm import tqdm
import math
import time
import matplotlib.pyplot as plt


from maze_1 import Cell, Maze
from algorithm import RepeatedForwardAStar, AStar
from config import Q8_PS, Q8_NUM_EXPERIMENT



def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


def test(dim:int, num_experiment=Q8_NUM_EXPERIMENT):
    data_time_hall=[0.0]*len(Q8_PS)
    data_path_hall=[0.0]*len(Q8_PS)
    data_time_nohall = [0.0] * len(Q8_PS)
    data_path_nohall = [0.0] * len(Q8_PS)
    for index_p, p in enumerate(Q8_PS):
        count=0
        for random_seed in tqdm(range(num_experiment)):
            maze = Maze(dim, dim)
            maze.initialize_maze(p, random_seed=random_seed)
            goal_cell = Cell((dim - 1, dim - 1))
            start_cell = Cell((0, 0))
            AStar_search=AStar(maze, euclidean_heuristic)
            AStar_search_path=AStar_search.search(start_cell, goal_cell)
            if len(AStar_search_path)==0:
                continue
            count+=1
            repeat_as=RepeatedForwardAStar(maze, euclidean_heuristic)
            start_time=time.time()
            repeat_as_nohall_path=repeat_as.search(start_cell, goal_cell, smart_restart=True)
            repeat_as_nohall_time=time.time()-start_time
            data_time_nohall[index_p]+=repeat_as_nohall_time
            data_path_nohall[index_p]+=len(repeat_as_nohall_path)
            start_time=time.time()
            repeat_hall_path=repeat_as.search(start_cell, goal_cell, smart_restart=True, find_hallway=True)
            repeat_hall_time = time.time() - start_time
            data_time_hall[index_p] += repeat_hall_time
            data_path_hall[index_p] += len(repeat_hall_path)

        if count!=0:
            data_time_hall[index_p]= data_time_hall[index_p]/count
            data_path_hall[index_p]=data_path_hall[index_p]/count
            data_time_nohall[index_p]=data_time_nohall[index_p]/count
            data_path_nohall[index_p]=data_path_nohall[index_p]/count

    return data_time_hall, data_path_hall, data_time_nohall, data_path_nohall


def main():
    dim =101
    data_time_hall, data_path_hall, data_time_nohall, data_path_nohall=test(dim)



    plt.figure(figsize=(8, 5))
    line1, = plt.plot(Q8_PS, data_time_hall, color='red')
    line2, = plt.plot(Q8_PS, data_time_nohall, color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average time')
    plt.title('Density VS. Average time')
    plt.legend(handles=[line1, line2],
               labels=['Improved Repeat Forward A* using hall', 'Improved Repeat Forward A* not using hall'],
               loc='best')
    plt.savefig('result\\Q8_3.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(Q8_PS, data_path_hall, color='red')
    line2, = plt.plot(Q8_PS, data_path_nohall, color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average trajectory path')
    plt.title('Density VS. Average trajectory path')
    plt.legend(handles=[line1, line2],
               labels=['Improved Repeat Forward A* using hall', 'Improved Repeat Forward A* not using hall'],
               loc='best')
    plt.savefig('result\\Q8_4.png')
    plt.show()

if __name__ == '__main__':
    main()







