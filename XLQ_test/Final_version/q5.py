# Question 5: Heuristics
# Use A*, and the entire gridworld is unknown


import math
import time
from maze import Cell, Maze
from algorithm import AStar
import matplotlib.pyplot as plt
from tqdm import tqdm


def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def manhattan_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return abs(x1 - x2) + abs(y1 - y2)


def chebyshev_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return max(abs(x1 - x2), abs(y1 - y2))


def test_heuristic(dim: int, heuristic, num_experiment=1000, p=0.33):
    '''
    Given the dim, generate num_experiment mazes, calculate the average time of three heuristic function
    '''
    total_time = 0
    for random_seed in tqdm(range(num_experiment)):
        # initialize
        maze = Maze(dim, dim)
        maze.initialize_maze(p, random_seed=random_seed)
        goal_cell = Cell((dim - 1, dim - 1))
        start_cell = Cell((0, 0))
        astar = AStar(maze, heuristic)

        # time for search
        experiment_start_time = time.time()
        path = astar.search(start_cell, goal_cell)
        experiment_time_cost = time.time() - experiment_start_time
        total_time += experiment_time_cost
    # Average time
    return total_time / num_experiment


def main():
    dim = 101
    p = 0.33
    names = ['Euclidean Distance', 'Manhattan Distance', 'Chebyshev Distance']
    heuristics = [euclidean_heuristic, manhattan_heuristic, chebyshev_heuristic]
    avg_times = []
    for heuristic in heuristics:
        avg_time = test_heuristic(dim, heuristic)
        avg_times.append(avg_time)
    print(avg_times)
    plt.figure(figsize=(8, 5))
    plt.bar(range(3), avg_times)
    plt.xticks(range(3), names)
    plt.xlabel('Heuristic')
    plt.ylabel('Average time')
    plt.title('Heuristic VS. Average time (p={})'.format(p))
    plt.savefig('q5.png')
    plt.show()


main()
