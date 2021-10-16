import math
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

from algorithm import SensingRepeatedForwardAStar, AStar
from maze import Maze, Cell
from config import PS, NUM_EXPERIMENT


def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


def test(dim: int, num_experiment=NUM_EXPERIMENT):
    data_time_know4 = [0.0] * len(PS)
    data_time_bumpin = [0.0] * len(PS)
    data_time_infer = [0.0] * len(PS)
    data_path_know4 = [0.0] * len(PS)
    data_path_bumpin = [0.0] * len(PS)
    data_path_infer = [0.0] * len(PS)
    final_path_know4 = [0.0] * len(PS)
    final_path_bumpin = [0.0] * len(PS)
    final_path_infer = [0.0] * len(PS)
    for index_p, p in enumerate(PS):
        count = 0
        for random_seed in tqdm(range(num_experiment)):
            maze = Maze(dim, dim)
            maze.initialize_maze(p, random_seed=random_seed)
            goal_cell = Cell((dim - 1, dim - 1))
            start_cell = Cell((0, 0))
            AStar_search = AStar(maze, euclidean_heuristic)
            AStar_search_path = AStar_search.search(start_cell, goal_cell)
            if len(AStar_search_path) == 0:
                continue
            count += 1
            sensing = SensingRepeatedForwardAStar(maze, euclidean_heuristic)
            st = time.time()
            path_know4 = sensing.search(start_cell, goal_cell, know_four_neighbours=True)
            et = time.time() - st
            data_time_know4[index_p] += et
            data_path_know4[index_p] += len(path_know4)
            AStar_search = AStar(sensing.discovered_maze, euclidean_heuristic)
            AStar_search_path = AStar_search.search(start_cell, goal_cell)
            final_path_know4[index_p] += len(AStar_search_path)

            sensing.initialize_discovered_maze()
            st = time.time()
            path_bumpin = sensing.search(start_cell, goal_cell, know_four_neighbours=False, use_infer_method=False)
            et = time.time() - st
            data_time_bumpin[index_p] += et
            data_path_bumpin[index_p] += len(path_bumpin)
            AStar_search = AStar(sensing.discovered_maze, euclidean_heuristic)
            AStar_search_path = AStar_search.search(start_cell, goal_cell)
            final_path_bumpin[index_p] += len(AStar_search_path)

            sensing.initialize_discovered_maze()
            st = time.time()
            path_infer = sensing.search(start_cell, goal_cell, know_four_neighbours=False, use_infer_method=True)
            et = time.time() - st
            data_time_infer[index_p] += et
            data_path_infer[index_p] += len(path_infer)
            AStar_search = AStar(sensing.discovered_maze, euclidean_heuristic)
            AStar_search_path = AStar_search.search(start_cell, goal_cell)
            final_path_infer[index_p] += len(AStar_search_path)

        if count != 0:
            data_time_know4[index_p] = data_time_know4[index_p] / count
            data_time_bumpin[index_p] = data_time_bumpin[index_p] / count
            data_time_infer[index_p] = data_time_infer[index_p] / count

            data_path_know4[index_p] = data_path_know4[index_p] / count
            data_path_bumpin[index_p] = data_path_bumpin[index_p] / count
            data_path_infer[index_p] = data_path_infer[index_p] / count

            final_path_know4[index_p] = final_path_know4[index_p] / count
            final_path_bumpin[index_p] = final_path_bumpin[index_p] / count
            final_path_infer[index_p] = final_path_infer[index_p] / count

    return data_time_know4, data_path_know4, final_path_know4, \
           data_time_bumpin, data_path_bumpin, final_path_bumpin,\
           data_time_infer,data_path_infer,final_path_infer


if __name__ == '__main__':
    current_file_path=os.path.dirname(os.path.abspath(__file__))
    new_dir=os.path.join(current_file_path, "data")
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    dim=101

    data_time_know4, data_path_know4, final_path_know4, data_time_bumpin, data_path_bumpin, final_path_bumpin, data_time_infer, data_path_infer, final_path_infer=test(dim, NUM_EXPERIMENT)

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(PS, data_time_know4, color='red')
    line2, = plt.plot(PS, data_time_infer, color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average time')
    plt.title('Density VS. Average time')
    plt.legend(handles=[line1, line2],
               labels=['agent1', 'agent3'],
               loc='best')
    plt.savefig(os.path.join(new_dir, "1.png"))

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(PS, data_time_bumpin, color='red')
    line2, = plt.plot(PS, data_time_infer, color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average time')
    plt.title('Density VS. Average time')
    plt.legend(handles=[line1, line2],
               labels=['agent2', 'agent3'],
               loc='best')
    plt.savefig(os.path.join(new_dir, "2.png"))

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(PS, data_path_know4, color='red')
    line2, = plt.plot(PS, data_path_infer, color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average time')
    plt.title('Density VS. total trajectory length')
    plt.legend(handles=[line1, line2],
               labels=['agent1', 'agent3'],
               loc='best')
    plt.savefig(os.path.join(new_dir, "3.png"))

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(PS, data_path_bumpin, color='red')
    line2, = plt.plot(PS, data_path_infer, color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average time')
    plt.title('Density VS. total trajectory length')
    plt.legend(handles=[line1, line2],
               labels=['agent2', 'agent3'],
               loc='best')
    plt.savefig(os.path.join(new_dir, "4.png"))

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(PS, final_path_know4, color='red')
    line2, = plt.plot(PS, final_path_infer, color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average time')
    plt.title('Density VS. final path length through discovered gridworld')
    plt.legend(handles=[line1, line2],
               labels=['agent1', 'agent3'],
               loc='best')
    plt.savefig(os.path.join(new_dir, "5.png"))

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(PS, final_path_bumpin, color='red')
    line2, = plt.plot(PS, final_path_infer, color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average time')
    plt.title('Density VS. final path length through discovered gridworld')
    plt.legend(handles=[line1, line2],
               labels=['agent2', 'agent3'],
               loc='best')
    plt.savefig(os.path.join(new_dir, "6.png"))