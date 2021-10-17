import math
from tqdm import tqdm
import time
import os
import json
import matplotlib.pyplot as plt

from algorithm import SensingRepeatedForwardAStar, AStar
from maze import Maze, Cell
from config import PS, NUM_EXPERIMENT


def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


def test(dim: int, num_experiment=NUM_EXPERIMENT):
    data_time_agent2 = [0.0] * len(PS)
    data_time_agent1 = [0.0] * len(PS)
    data_time_agent3 = [0.0] * len(PS)
    data_time_agent4 = [0.0] * len(PS)

    data_path_agent2 = [0.0] * len(PS)
    data_path_agent1 = [0.0] * len(PS)
    data_path_agent3 = [0.0] * len(PS)
    data_path_agent4 = [0.0] * len(PS)

    final_path_agent2 = [0.0] * len(PS)
    final_path_agent1 = [0.0] * len(PS)
    final_path_agent3 = [0.0] * len(PS)
    final_path_agent4 = [0.0] * len(PS)

    replan_time_agent2 = [0] * len(PS)
    replan_time_agent1 = [0] * len(PS)
    replan_time_agent3 = [0] * len(PS)
    replan_time_agent4 = [0] * len(PS)

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
            # sensing.initialize_discovered_maze()
            st = time.time()
            path_more_infer = sensing.search(start_cell, goal_cell, know_four_neighbours=False, use_infer_method=True,
                                             infer_more=True)
            et = time.time() - st
            data_time_agent4[index_p] += et
            data_path_agent4[index_p] += len(path_more_infer)
            AStar_search = AStar(sensing.discovered_maze, euclidean_heuristic)
            AStar_search_path = AStar_search.search(start_cell, goal_cell)
            final_path_agent4[index_p] += len(AStar_search_path)
            replan_time_agent4[index_p] += sensing.replan_time

        if count != 0:
            data_time_agent2[index_p] = data_time_agent2[index_p] / count
            data_time_agent1[index_p] = data_time_agent1[index_p] / count
            data_time_agent3[index_p] = data_time_agent3[index_p] / count
            data_time_agent4[index_p] = data_time_agent4[index_p] / count

            data_path_agent2[index_p] = data_path_agent2[index_p] / count
            data_path_agent1[index_p] = data_path_agent1[index_p] / count
            data_path_agent3[index_p] = data_path_agent3[index_p] / count
            data_path_agent4[index_p] = data_path_agent4[index_p] / count

            final_path_agent2[index_p] = final_path_agent2[index_p] / count
            final_path_agent1[index_p] = final_path_agent1[index_p] / count
            final_path_agent3[index_p] = final_path_agent3[index_p] / count
            final_path_agent4[index_p] = final_path_agent4[index_p] / count

            replan_time_agent2[index_p] = replan_time_agent2[index_p] / count
            replan_time_agent1[index_p] = replan_time_agent1[index_p] / count
            replan_time_agent3[index_p] = replan_time_agent3[index_p] / count
            replan_time_agent4[index_p] = replan_time_agent4[index_p] / count

    return data_time_agent2, data_path_agent2, final_path_agent2, \
           data_time_agent1, data_path_agent1, final_path_agent1, \
           data_time_agent3, data_path_agent3, final_path_agent3, \
           replan_time_agent2, replan_time_agent1, replan_time_agent3, \
           data_time_agent4, data_path_agent4, final_path_agent4, replan_time_agent4


if __name__ == '__main__':
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    new_dir = os.path.join(current_file_path, "data_agent4")
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    dim = 101

    data_time_know4, data_path_know4, final_path_know4, data_time_bumpin, data_path_bumpin, final_path_bumpin, data_time_infer, data_path_infer, final_path_infer, replan_time_know4, replan_time_bumpin, replan_time_infer, data_time_agent4, data_path_agent4, final_path_agent4, replan_time_agent4 = test(
        dim, NUM_EXPERIMENT)

    # data_time_know4, data_path_know4, final_path_know4, data_time_bumpin, data_path_bumpin, final_path_bumpin, data_time_infer, data_path_infer, final_path_infer=test(dim, NUM_EXPERIMENT)

    save = {"data_path_agent4": data_path_agent4,
            "data_time_agent4": data_time_agent4,
            "final_path_agent4": final_path_agent4,
            "replan_time_agent4": replan_time_agent4}

    save_json = json.dumps(save)

    with open(os.path.join(new_dir, "save.json"), "w") as writer:
        writer.write(save_json)
