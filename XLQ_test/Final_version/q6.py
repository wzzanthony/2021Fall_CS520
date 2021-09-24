# Question 6: Performance
# Use Repeated Forward A*, and the entire gridworld is unknown

import math
import time
from maze import Cell, Maze
from algorithm import AStar, RepeatedForwardAStar
import matplotlib.pyplot as plt
from tqdm import tqdm
from experiment_config import NUM_EXPERIMENT, PS


def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def test(dim: int, p: float, num_experiment=NUM_EXPERIMENT):
    '''
    Given the dim and p, generate num_experiment mazes, calculate the average values
    '''
    # sum of Trajectory Length
    a = 0
    # total number of solve experiment
    na = 0
    # sum of (Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld)
    b = 0.
    # total number of solve experiment
    nb = 0
    # sum of (Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld)
    c = 0.
    # total number of solve experiment
    nc = 0
    # sum of Number of Cells Processed by Repeated A*
    d = 0

    total_search_time = 0

    for random_seed in tqdm(range(num_experiment)):
        # initialize
        maze = Maze(dim, dim)
        maze.initialize_maze(p, random_seed=random_seed)
        goal_cell = Cell((dim - 1, dim - 1))
        start_cell = Cell((0, 0))

        repeated_forward_astar = RepeatedForwardAStar(maze, euclidean_heuristic)
        search_start = time.time()   # Start time
        trajectory_path = repeated_forward_astar.search(start_cell, goal_cell)
        search_time = time.time() - search_start    # Total time per experiment
        total_search_time += search_time

        if trajectory_path:
            a += len(trajectory_path)
            na += 1

        # search in the Final Discovered Gridworld
        astar = AStar(repeated_forward_astar.discovered_maze, euclidean_heuristic)
        path = astar.search(start_cell, goal_cell)

        if len(path) != 0:
            nb += 1
            b += len(trajectory_path) / len(path)

        # search in the Full GridWorld
        astar = AStar(maze, euclidean_heuristic)
        path = astar.search(start_cell, goal_cell)

        if len(path) != 0:
            nc += 1
            c += len(trajectory_path) / len(path)

        d += len(repeated_forward_astar.cell_processed)
    print(na, nb, nc)
    return a / na, b / nb, c / nc, d / num_experiment, total_search_time / num_experiment


def main():
    dim = 101
    result = []
    for p in PS:
        a, b, c, d, t = test(dim, p)
        result.append((a, b, c, d, t))
        print('p = {:.2f}, '
              'sum of Trajectory Length = {:.2f}, '
              'sum of (Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld) = {:.2f}, '
              'sum of (Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld) = {:.2f}, '
              'sum of Number of Cells Processed by Repeated A* = {:.2f}, '.format(p, a, b, c, d))
    print(result)
    '''
    When num_experiment = 10
    result = [
        ('207.50', '1.03', '1.03', '526.10'),
        ('216.90', '1.08', '1.08', '541.50'),
        ('222.70', '1.11', '1.11', '546.10'),
        ('236.90', '1.18', '1.18', '548.70'),
        ('248.22', '1.23', '1.23', '500.80'),
        ('256.88', '1.28', '1.28', '508.50'),
        ('278.38', '1.38', '1.38', '528.00'),
        ('312.00', '1.55', '1.55', '424.70'),
        ('355.50', '1.77', '1.76', '389.00'),
        ('425.50', '2.12', '2.09', '371.80'),
        ('460.50', '2.25', '2.19', '321.40'),
    ]
    '''
    n = len(PS)

    plt.figure(figsize=(8, 5))
    plt.plot(PS, [e[0] for e in result])
    plt.xlabel('Density')
    plt.ylabel('Average Trajectory Length')
    plt.title('Density VS. Average Trajectory Length')
    plt.savefig('q6_1.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(PS, [e[1] for e in result])
    plt.xlabel('Density')
    plt.ylabel('Average (Trajectory Length / Final Discovered Gridworld Shortest Path Length)')
    plt.title('Density VS. Average (Trajectory Length / Final Discovered Gridworld Shortest Path Length)')
    plt.savefig('q6_2.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(PS, [e[2] for e in result])
    plt.xlabel('Density')
    plt.ylabel('Average (Shortest Path in Final Discovered Gridworld Length / Shortest Path in Full Gridworld Length)')
    plt.title(
        'Density VS. Average (Shortest Path in Final Discovered Gridworld Length / Shortest Path in Full Gridworld Length)')
    plt.savefig('q6_3.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(PS, [e[3] for e in result])
    plt.xlabel('Density')
    plt.ylabel('Number of Cells Processed by Repeated A*')
    plt.title('Density VS. Number of Cells Processed by Repeated A*)')
    plt.savefig('q6_4.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(PS, [e[4] for e in result])
    plt.xlabel('Density')
    plt.ylabel('Search Time*')
    plt.title('Density VS. Average search time)')
    plt.savefig('q6_5.png')
    plt.show()


main()
