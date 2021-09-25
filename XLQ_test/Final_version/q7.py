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
    a_bump = 0
    # total number of solve experiment
    na_bump = 0
    # sum of (Length of Trajectory / Length of Shortest Path in Final Discovered Gridworld)
    b_bump = 0.
    # total number of solve experiment
    nb_bump = 0
    # sum of (Length of Shortest Path in Final Discovered Gridworld / Length of Shortest Path in Full Gridworld)
    c_bump = 0.
    # total number of solve experiment
    nc_bump = 0
    # sum of Number of Cells Processed by Repeated A*
    d_bump = 0

    total_search_time_bump = 0
    total_search_time = 0

    for random_seed in tqdm(range(num_experiment)):
        # initialize
        maze = Maze(dim, dim)
        maze.initialize_maze(p, random_seed=random_seed)
        goal_cell = Cell((dim - 1, dim - 1))
        start_cell = Cell((0, 0))

        # Question 7
        repeated_forward_astar_bump = RepeatedForwardAStar(maze, euclidean_heuristic)
        search_start_bump = time.time()
        trajectory_path_bump = repeated_forward_astar_bump.search(start_cell, goal_cell, only_bump=True)
        search_time_bump = time.time() - search_start_bump
        total_search_time_bump += search_time_bump

        # Question 6
        repeated_forward_astar = RepeatedForwardAStar(maze, euclidean_heuristic)
        search_start = time.time()  # Start time
        trajectory_path = repeated_forward_astar.search(start_cell, goal_cell)
        search_time = time.time() - search_start  # Total time per experiment
        total_search_time += search_time

        if trajectory_path_bump:
            a_bump += len(trajectory_path_bump)
            na_bump += 1
            a_ra += len(trajectory_path)
            na_ra += 1

        # search in the Final Discovered Gridworld
        astar = AStar(repeated_forward_astar_bump.discovered_maze, euclidean_heuristic)
        path = astar.search(start_cell, goal_cell)

        if len(path) != 0:
            nb_bump += 1
            b_bump += len(trajectory_path_bump) / len(path)
            nb_ra += 1
            b_ra += len(trajectory_path) / len(path)

        # search in the Full GridWorld
        astar = AStar(maze, euclidean_heuristic)
        path = astar.search(start_cell, goal_cell)

        if len(path) != 0:
            nc_bump += 1
            c_bump += len(trajectory_path_bump) / len(path)
            nc_ra += 1
            c_ra += len(trajectory_path) / len(path)

        d_bump += len(repeated_forward_astar_bump.cell_processed)
        d_ra += len(repeated_forward_astar.cell_processed)
    print(na_ra, nb_ra, nc_ra, na_bump, nb_bump, nc_bump)
    return a_ra / na_ra, b_ra / nb_ra, c_ra / nc_ra, d_ra / num_experiment, total_search_time / num_experiment, \
           a_bump / na_bump, b_bump / nb_bump, c_bump / nc_bump, d_bump / num_experiment, total_search_time_bump / num_experiment


def main():
    dim = 101
    result = []
    for p in PS:
        a, b, c, d, t, a_bump, b_bump, c_bump, d_bump, t_bump = test(dim, p)
        result.append((a, b, c, d, t, a_bump, b_bump, c_bump, d_bump, t_bump))
        print('p = {:.2f}, a = {:.2f}, b = {:.2f}, c = {:.2f}, d = {:.2f}, t = {:.2f}, a_bump = {:.2f}, b_bump = {:.2f}, c_bump = {:.2f}, d_bump = {:.2f}, t_bump = {:.2f}'.format(p, a, b, c, d, t, a_bump, b_bump, c_bump, d_bump, t_bump))
    print(result)

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[0] for e in result], color='blue')
    repeat_A_star_line_bump, = plt.plot(PS, [e[5] for e in result], color='red')
    plt.xlabel('Density')
    plt.ylabel('Average Trajectory Length')
    plt.title('Density VS. Average Trajectory Length')
    plt.legend(handles=[repeat_A_star_line, repeat_A_star_line_bump],
               labels=['Repeat Forward A*', 'Repeat Forward A* Only Bump'],
               loc='best')
    plt.savefig('q7_1.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[1] for e in result], color='blue')
    repeat_A_star_line_bump, = plt.plot(PS, [e[6] for e in result], color='red')
    plt.xlabel('Density')
    plt.ylabel('Average (Trajectory Length / Final Discovered Gridworld Shortest Path Length)')
    plt.title('Density VS. Average (Trajectory Length / Final Discovered Gridworld Shortest Path Length)')
    plt.legend(handles=[repeat_A_star_line, repeat_A_star_line_bump],
               labels=['Repeat Forward A*', 'Repeat Forward A* Only Bump'],
               loc='best')
    plt.savefig('q7_2.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[2] for e in result], color='blue')
    repeat_A_star_line_bump, = plt.plot(PS, [e[7] for e in result], color='red')
    plt.xlabel('Density')
    plt.ylabel('Average (Shortest Path in Final Discovered Gridworld Length / Shortest Path in Full Gridworld Length)')
    plt.title(
        'Density VS. Average (Shortest Path in Final Discovered Gridworld Length / Shortest Path in Full Gridworld Length)')
    plt.legend(handles=[repeat_A_star_line, repeat_A_star_line_bump],
               labels=['Repeat Forward A*', 'Repeat Forward A* Only Bump'],
               loc='best')
    plt.savefig('q7_3.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[3] for e in result], color='blue')
    repeat_A_star_line_bump, = plt.plot(PS, [e[8] for e in result], color='red')
    plt.plot(PS, [e[3] for e in result])
    plt.xlabel('Density')
    plt.ylabel('Number of Cells Processed by Repeated A*')
    plt.title('Density VS. Number of Cells Processed by Repeated A*)')
    plt.legend(handles=[repeat_A_star_line, repeat_A_star_line_bump],
               labels=['Repeat Forward A*', 'Repeat Forward A* Only Bump'],
               loc='best')
    plt.savefig('q7_4.png')
    plt.show()

    # plt.figure(figsize=(8, 5))
    # repeat_A_star_line, = plt.plot(PS, [e[4] for e in result], color='blue')
    # repeat_A_star_line_bump, = plt.plot(PS, [e[9] for e in result], color='red')
    # plt.xlabel('Density')
    # plt.ylabel('Search Time*')
    # plt.title('Density VS. Average search time)')
    # plt.legend(handles=[repeat_A_star_line, repeat_smart_A_star_line],
    #labels = ['Repeat Forward A*', 'Repeat Forward A* Only Bump'],
    #            loc='best')
    # plt.savefig('q7_5.png')
    # plt.show()


main()
