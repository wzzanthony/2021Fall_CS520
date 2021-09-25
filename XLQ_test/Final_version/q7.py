# Question 7: Performance Part 2
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
    # Question 6
    a_ra = 0
    # total number of solve experiment
    na_ra = 0
    # sum of (Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld)
    b_ra = 0.
    # total number of solve experiment
    nb_ra = 0
    # sum of (Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld)
    c_ra = 0.
    # total number of solve experiment
    nc_ra = 0
    # sum of Number of Cells Processed by Repeated A*
    d_ra = 0

    # Question 7
    # sum of Trajectory Length
    a_rasmart = 0
    # total number of solve experiment
    na_rasmart = 0
    # sum of (Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld)
    b_rasmart = 0.
    # total number of solve experiment
    nb_rasmart = 0
    # sum of (Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld)
    c_rasmart = 0.
    # total number of solve experiment
    nc_rasmart = 0
    # sum of Number of Cells Processed by Repeated A*
    d_rasmart = 0

    total_search_time_smart = 0
    total_search_time = 0

    for random_seed in tqdm(range(num_experiment)):
        # initialize
        maze = Maze(dim, dim)
        maze.initialize_maze(p, random_seed=random_seed)
        goal_cell = Cell((dim - 1, dim - 1))
        start_cell = Cell((0, 0))

        # Question 7
        repeated_forward_astar_smart = RepeatedForwardAStar(maze, euclidean_heuristic)
        search_start_smart = time.time()
        trajectory_path_smart = repeated_forward_astar_smart.search(start_cell, goal_cell, only_bump=True)
        search_time_smart = time.time() - search_start_smart
        total_search_time_smart += search_time_smart

        # Question 6
        repeated_forward_astar = RepeatedForwardAStar(maze, euclidean_heuristic)
        search_start = time.time()  # Start time
        trajectory_path = repeated_forward_astar.search(start_cell, goal_cell)
        search_time = time.time() - search_start  # Total time per experiment
        total_search_time += search_time

        if trajectory_path_smart:
            a_rasmart += len(trajectory_path_smart)
            na_rasmart += 1
            a_ra += len(trajectory_path)
            na_ra += 1

        # search in the Final Discovered Gridworld
        astar = AStar(repeated_forward_astar_smart.discovered_maze, euclidean_heuristic)
        path = astar.search(start_cell, goal_cell)

        if len(path) != 0:
            nb_rasmart += 1
            b_rasmart += len(trajectory_path_smart) / len(path)
            nb_ra += 1
            b_ra += len(trajectory_path) / len(path)

        # search in the Full GridWorld
        astar = AStar(maze, euclidean_heuristic)
        path = astar.search(start_cell, goal_cell)

        if len(path) != 0:
            nc_rasmart += 1
            c_rasmart += len(trajectory_path_smart) / len(path)
            nc_ra += 1
            c_ra += len(trajectory_path) / len(path)

        d_rasmart += len(repeated_forward_astar_smart.cell_processed)
        d_ra += len(repeated_forward_astar.cell_processed)
    print(na_ra, nb_ra, nc_ra, na_rasmart, nb_rasmart, nc_rasmart)
    return a_ra / na_ra, b_ra / nb_ra, c_ra / nc_ra, d_ra / num_experiment, total_search_time / num_experiment,\
           a_rasmart / na_rasmart, b_rasmart / nb_rasmart, c_rasmart / nc_rasmart, d_rasmart / num_experiment, total_search_time_smart / num_experiment


def main():
    dim = 101
    result = []
    for p in PS:
        a, b, c, d, t, a_smart, b_smart, c_smart, d_smart, t_smart = test(dim, p)
        result.append((a, b, c, d, t, a_smart, b_smart, c_smart, d_smart, t_smart))
        print('p = {:.2f}, a = {:.2f}, b = {:.2f}, c = {:.2f}, d = {:.2f}, t = {:.2f}, a_smart = {:.2f}, b_smart = {:.2f}, c_smart = {:.2f}, d_smart = {:.2f}, t_smart = {:.2f}'.format(p, a, b, c, d, t, a_smart, b_smart, c_smart, d_smart, t_smart))
    print(result)

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[0] for e in result], color='blue')
    repeat_smart_A_star_line, = plt.plot(PS, [e[5] for e in result], color='red')
    plt.xlabel('Density')
    plt.ylabel('Average Trajectory Length')
    plt.title('Density VS. Average Trajectory Length')
    plt.legend(handles=[repeat_A_star_line, repeat_smart_A_star_line],
               labels=['Repeat Forward A*', 'Improved Repeat Forward A*'],
               loc='best')
    plt.savefig('q7_1.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[1] for e in result], color='blue')
    repeat_smart_A_star_line, = plt.plot(PS, [e[6] for e in result], color='red')
    plt.xlabel('Density')
    plt.ylabel('Average (Trajectory Length / Final Discovered Gridworld Shortest Path Length)')
    plt.title('Density VS. Average (Trajectory Length / Final Discovered Gridworld Shortest Path Length)')
    plt.legend(handles=[repeat_A_star_line, repeat_smart_A_star_line],
               labels=['Repeat Forward A*', 'Improved Repeat Forward A*'],
               loc='best')
    plt.savefig('q7_2.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[2] for e in result], color='blue')
    repeat_smart_A_star_line, = plt.plot(PS, [e[7] for e in result], color='red')
    plt.xlabel('Density')
    plt.ylabel('Average (Shortest Path in Final Discovered Gridworld Length / Shortest Path in Full Gridworld Length)')
    plt.title(
        'Density VS. Average (Shortest Path in Final Discovered Gridworld Length / Shortest Path in Full Gridworld Length)')
    plt.legend(handles=[repeat_A_star_line, repeat_smart_A_star_line],
               labels=['Repeat Forward A*', 'Improved Repeat Forward A*'],
               loc='best')
    plt.savefig('q7_3.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[3] for e in result], color='blue')
    repeat_smart_A_star_line, = plt.plot(PS, [e[8] for e in result], color='red')
    plt.plot(PS, [e[3] for e in result])
    plt.xlabel('Density')
    plt.ylabel('Number of Cells Processed by Repeated A*')
    plt.title('Density VS. Number of Cells Processed by Repeated A*)')
    plt.legend(handles=[repeat_A_star_line, repeat_smart_A_star_line],
               labels=['Repeat Forward A*', 'Improved Repeat Forward A*'],
               loc='best')
    plt.savefig('q7_4.png')
    plt.show()

    # plt.figure(figsize=(8, 5))
    # repeat_A_star_line, = plt.plot(PS, [e[4] for e in result], color='blue')
    # repeat_smart_A_star_line, = plt.plot(PS, [e[9] for e in result], color='red')
    # plt.xlabel('Density')
    # plt.ylabel('Search Time*')
    # plt.title('Density VS. Average search time)')
    # plt.legend(handles=[repeat_A_star_line, repeat_smart_A_star_line],
    #            labels=['Repeat Forward A*', 'Improved Repeat Forward A*'],
    #            loc='best')
    # plt.savefig('q7_5.png')
    # plt.show()


main()
