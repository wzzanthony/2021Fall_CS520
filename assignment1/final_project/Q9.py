import math
from tqdm import tqdm
import sys
import time
import matplotlib.pyplot as plt


from A_star_Q9 import Cell, Maze
from A_star_Q9 import AStar as Heuristics_AStar
from config import Q9_PS, Q9_NUM_EXPERIMENT

sys.path.append('..\\..\\XLQ_test\\Final_version')

from algorithm import AStar
from maze import Cell as Cell_AS





def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))

def test(dim: int, num_experiment=Q9_NUM_EXPERIMENT):
    # data=[0]*len(Q9_QS)
    # print(Q9_QS)
    # print(data)
    data_time_as = [0.0] * len(Q9_PS)
    data_path_as = [0.0] * len(Q9_PS)
    data_time_hs = [0.0] * len(Q9_PS)
    data_path_hs = [0.0] * len(Q9_PS)

    for index_p, p in enumerate(Q9_PS):
        count = 0
        for random_seed in tqdm(range(num_experiment)):
            maze = Maze(dim, dim)
            maze.initialize_maze(p, random_seed=random_seed)
            goal_cell = Cell((dim - 1, dim - 1))
            start_cell = Cell((0, 0))
            goal_cell_as=Cell_AS((dim - 1, dim - 1))
            start_cell_as=Cell_AS((0, 0))
            astar_search=AStar(maze, euclidean_heuristic)
            start_time = time.time()
            astar_path=astar_search.search(start_cell_as, goal_cell_as)
            search_time_as=time.time()-start_time
            if len(astar_path)==0:
                continue
            count+=1
            data_time_as[index_p]+=search_time_as
            data_path_as[index_p]+=len(astar_path)
            heuristics_search=Heuristics_AStar(maze,euclidean_heuristic)
            start_time=time.time()
            trajectory_path=heuristics_search.search(start_cell, goal_cell, p)
            search_time_hs=time.time()-start_time
            data_time_hs[index_p]+=search_time_hs
            data_path_hs[index_p]+=len(trajectory_path)
        if count!=0:
            # average_data_time=data_time/count
            data_path_as[index_p]=data_path_as[index_p]/count
            data_path_hs[index_p]=data_path_hs[index_p]/count
            data_time_as[index_p]=data_time_as[index_p]/count
            data_time_hs[index_p]=data_time_hs[index_p]/count
    return data_path_as, data_time_as, data_path_hs, data_time_hs

def main():
    dim = 101
    data_path_as, data_time_as, data_path_hs, data_time_hs=test(dim, Q9_NUM_EXPERIMENT)

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(Q9_PS, data_time_as, color='red')
    line2, = plt.plot(Q9_PS, data_time_hs, color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average time')
    plt.title('Density VS. Average time')
    plt.legend(handles=[line1, line2],
               labels=['Repeat Forward A*', 'Heuristics A*'],
               loc='best')
    plt.savefig('result\\Q9_1.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(Q9_PS, data_path_as, color='red')
    line2, = plt.plot(Q9_PS, data_path_hs, color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average trajectory path')
    plt.title('Density VS. Average trajectory path')
    plt.legend(handles=[line1, line2],
               labels=['Repeat Forward A*', 'Heuristics A*'],
               loc='best')
    plt.savefig('result\\Q9_2.png')
    plt.show()










if __name__ == '__main__':
    main()


