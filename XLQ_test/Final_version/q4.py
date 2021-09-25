# Question 4: Solvability
# Use A*, and the entire gridworld is known


import math
import time
from maze import Cell, Maze
from algorithm import AStar
import matplotlib.pyplot as plt
from tqdm import tqdm


def heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def test_solvability(dim: int, p: float, num_experiment=100):
    '''
    Given the dim and the probability, generate num_experiment mazes, and calculate the Solvability
    '''
    num_solved = 0
    for random_seed in tqdm(range(num_experiment)):
        maze = Maze(dim, dim)
        # Set a random_seed
        maze.initialize_maze(p, random_seed=random_seed)
        goal_cell = Cell((dim - 1, dim - 1))
        start_cell = Cell((0, 0))
        astar = AStar(maze, heuristic)
        path = astar.search(start_cell, goal_cell)
        if path:
            num_solved += 1
    return num_solved / num_experiment


def main():
    dim = 101
    # times we plan to do the experiment
    num_experiment = 100
    #  ps = [0.0, 0.1, 0.2, ..., 0.9], 10 ps value in total
    ps = [num / 100 for num in range(0, 100, 5)]
    solvabilities = []
    for p in ps:
        s = test_solvability(dim, p, num_experiment=num_experiment)
        solvabilities.append(s)
        print('p = {:.2f}, solvability = {:.4f}'.format(p, s))
    plt.figure(figsize=(12, 5))
    plt.plot(ps, solvabilities, marker='.', ms=5)
    plt.xlabel('Density')
    plt.ylabel('Solvability')
    plt.title('Density VS. Solvability')
    plt.savefig('q4.png')
    plt.show()


main()
