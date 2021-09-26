import math
from tqdm import tqdm
import sys
import time
import matplotlib.pyplot as plt


from A_star_Q9 import Cell, Maze
from A_star_Q9 import AStar as Heuristics_AStar
from config import Q9_PS, Q9_NUM_EXPERIMENT, Q9_QS

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
    time_PS=[0.0]*len(Q9_PS)
    for index_p, p in enumerate(Q9_PS):
        data_time = [0.0] * len(Q9_QS)
        data_path=[0.0]*len(Q9_QS)
        count=0
        for random_seed in tqdm(range(num_experiment)):
            maze = Maze(dim, dim)
            maze.initialize_maze(p, random_seed=random_seed)
            goal_cell = Cell((dim - 1, dim - 1))
            start_cell = Cell((0, 0))
            goal_cell_as=Cell_AS((dim - 1, dim - 1))
            start_cell_as=Cell_AS((0, 0))
            astar_search=AStar(maze, euclidean_heuristic)
            astar_path=astar_search.search(start_cell_as, goal_cell_as)
            if len(astar_path)==0:
                continue
            count+=1
            heuristics_search=Heuristics_AStar(maze,euclidean_heuristic)
            for index, q in enumerate(Q9_QS):
                start_time=time.time()
                trajectory_path=heuristics_search.search(start_cell, goal_cell, q)
                search_time=time.time()-start_time
                data_time[index]+=search_time
                data_path[index]+=len(trajectory_path)
        if count!=0:
            # average_data_time=data_time/count
            average_data_time=[0]*len(data_time)
            for index_data_time in range(len(data_time)):
                average_data_time[index_data_time]=data_time[index_data_time]/count
            # average_data_path=data_path/count
        min_time=min(average_data_time)
        time_PS[index_p]=Q9_QS[average_data_time.index(min_time)]
        print("P: %f, Q: %F, count: %d"%(p,time_PS[index_p],count))
    return time_PS

def main():
    dim = 101
    result=test(dim, Q9_NUM_EXPERIMENT)

    plt.figure(figsize=(8, 5))
    plt.plot(Q9_PS, result, color='red')
    plt.xlabel('Density')
    plt.ylabel('Weight')
    plt.title('Density VS. Weight')
    plt.savefig('result\\Q9_1.png')
    plt.show()










if __name__ == '__main__':
    main()


