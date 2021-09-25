import sys
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('..\\..\\XLQ_test\\Final_version')

from maze import Cell, Maze
from algorithm import AStar, RepeatedForwardAStar, RepeatedForwardBFS
from experiment_config import NUM_EXPERIMENT, PS


def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


# def generate_experiment_info(dim:int, p:float, experiment=NUM_EXPERIMENT):


def test(dim: int, p: float, num_experiment=NUM_EXPERIMENT):
    '''
    Given the dim and p, generate num_experiment mazes, calculate the average values
    '''
    # sum of Trajectory Length
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
    total_search_time_ra = 0

    a_BFS = 0
    na_BFS = 0

    b_BFS = 0
    nb_BFS = 0

    c_BFS = 0
    nc_BFS = 0

    d_BFS = 0

    total_search_time_BFS = 0

    for random_seed in tqdm(range(num_experiment)):
        # initialize
        maze = Maze(dim, dim)
        maze.initialize_maze(p, random_seed=random_seed)
        goal_cell = Cell((dim - 1, dim - 1))
        start_cell = Cell((0, 0))

        repeated_forward_astar = RepeatedForwardAStar(maze, euclidean_heuristic)
        search_start = time.time()  # Start time
        trajectory_path = repeated_forward_astar.search(start_cell, goal_cell)
        search_time = time.time() - search_start  # Total time per experiment
        total_search_time_ra += search_time

        repeated_BFS = RepeatedForwardBFS(maze)
        search_start = time.time()
        BFS_path = repeated_BFS.search(start_cell, goal_cell)
        search_time = time.time() - search_start
        total_search_time_BFS += search_time

        if trajectory_path:
            a_ra += len(trajectory_path)
            na_ra += 1

        if BFS_path:
            a_BFS += len(BFS_path)
            na_BFS += 1

        # search in the Final Discovered Gridworld
        astar = AStar(repeated_forward_astar.discovered_maze, euclidean_heuristic)
        path = astar.search(start_cell, goal_cell)

        if len(path) != 0:
            nb_ra += 1
            b_ra += len(trajectory_path) / len(path)

            nb_BFS += 1
            b_BFS += len(BFS_path) / len(path)

        # search in the Full GridWorld
        astar = AStar(maze, euclidean_heuristic)
        path = astar.search(start_cell, goal_cell)

        if len(path) != 0:
            nc_ra += 1
            c_ra += len(trajectory_path) / len(path)

            nc_BFS += 1
            c_BFS += len(BFS_path) / len(path)

        d_ra += len(repeated_forward_astar.cell_processed)
        d_BFS += len(repeated_BFS.cell_processed)
    print(na_ra, nb_ra, nc_ra)
    print(na_BFS, nb_BFS, nc_BFS)
    return a_ra / na_ra, b_ra / nb_ra, c_ra / nc_ra, d_ra / num_experiment, total_search_time_ra / num_experiment, \
           a_BFS / na_BFS, b_BFS / nb_BFS, c_BFS / nc_BFS, d_BFS / num_experiment, total_search_time_BFS / num_experiment


def test_bump_in(dim: int, p: float, num_experiment=NUM_EXPERIMENT):
    '''
    Given the dim and p, generate num_experiment mazes, calculate the average values
    '''
    # sum of Trajectory Length
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
    total_search_time_ra = 0

    a_BFS = 0
    na_BFS = 0

    b_BFS = 0
    nb_BFS = 0

    c_BFS = 0
    nc_BFS = 0

    d_BFS = 0

    total_search_time_BFS = 0

    for random_seed in tqdm(range(num_experiment)):
        # initialize
        maze = Maze(dim, dim)
        maze.initialize_maze(p, random_seed=random_seed)
        goal_cell = Cell((dim - 1, dim - 1))
        start_cell = Cell((0, 0))

        repeated_forward_astar = RepeatedForwardAStar(maze, euclidean_heuristic)
        search_start = time.time()  # Start time
        trajectory_path = repeated_forward_astar.search(start_cell, goal_cell, only_bump=True)
        search_time = time.time() - search_start  # Total time per experiment
        total_search_time_ra += search_time

        repeated_BFS = RepeatedForwardBFS(maze)
        search_start = time.time()
        BFS_path = repeated_BFS.search(start_cell, goal_cell, only_bump=True)
        search_time = time.time() - search_start
        total_search_time_BFS += search_time

        if trajectory_path:
            a_ra += len(trajectory_path)
            na_ra += 1

        if BFS_path:
            a_BFS += len(BFS_path)
            na_BFS += 1

        # search in the Final Discovered Gridworld
        astar = AStar(repeated_forward_astar.discovered_maze, euclidean_heuristic)
        path = astar.search(start_cell, goal_cell)

        if len(path) != 0:
            nb_ra += 1
            b_ra += len(trajectory_path) / len(path)

            nb_BFS += 1
            b_BFS += len(BFS_path) / len(path)

        # search in the Full GridWorld
        astar = AStar(maze, euclidean_heuristic)
        path = astar.search(start_cell, goal_cell)

        if len(path) != 0:
            nc_ra += 1
            c_ra += len(trajectory_path) / len(path)

            nc_BFS += 1
            c_BFS += len(BFS_path) / len(path)

        d_ra += len(repeated_forward_astar.cell_processed)
        d_BFS += len(repeated_BFS.cell_processed)
    print(na_ra, nb_ra, nc_ra)
    print(na_BFS, nb_BFS, nc_BFS)
    return a_ra / na_ra, b_ra / nb_ra, c_ra / nc_ra, d_ra / num_experiment, total_search_time_ra / num_experiment, \
           a_BFS / na_BFS, b_BFS / nb_BFS, c_BFS / nc_BFS, d_BFS / num_experiment, total_search_time_BFS / num_experiment




def main():
    dim = 101
    result = []
    for p in PS:
        a_ra, b_ra, c_ra, d_ra, t_ra, a_BFS, b_BFS, c_BFS, d_BFS, t_BFS = test(dim, p)
        result.append((a_ra, b_ra, c_ra, d_ra, t_ra, a_BFS, b_BFS, c_BFS, d_BFS, t_BFS))

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[0] for e in result], color='red')
    repeat_BFS_line, = plt.plot(PS, [e[5] for e in result], color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average Trajectory Length')
    plt.title('Density VS. Average Trajectory Length')
    plt.legend(handles=[repeat_A_star_line, repeat_BFS_line],
               labels=['Repeat Forward A*', 'Repeat BFS'],
               loc='best')
    plt.savefig('result\\extra_credit_1.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[1] for e in result], color='red')
    repeat_BFS_line, = plt.plot(PS, [e[6] for e in result], color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average (Trajectory Length / Final Discovered Gridworld Shortest Path Length)')
    plt.title('Density VS. Average (Trajectory Length / Final Discovered Gridworld Shortest Path Length)')
    plt.legend(handles=[repeat_A_star_line, repeat_BFS_line],
               labels=['Repeat Forward A*', 'Repeat BFS'],
               loc='best')
    plt.savefig('result\\extra_credit_2.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[2] for e in result], color='red')
    repeat_BFS_line, = plt.plot(PS, [e[7] for e in result], color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average (Shortest Path in Final Discovered Gridworld Length / Shortest Path in Full Gridworld Length)')
    plt.title(
        'Density VS. Average (Shortest Path in Final Discovered Gridworld Length / Shortest Path in Full Gridworld Length)')
    plt.legend(handles=[repeat_A_star_line, repeat_BFS_line],
               labels=['Repeat Forward A*', 'Repeat BFS'],
               loc='best')
    plt.savefig('result\\extra_credit_3.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[3] for e in result], color='red')
    repeat_BFS_line, = plt.plot(PS, [e[8] for e in result], color='blue')
    plt.xlabel('Density')
    plt.ylabel('Number of Cells Processed by Repeated A*')
    plt.title('Density VS. Number of Cells Processed by Repeated A*)')
    plt.legend(handles=[repeat_A_star_line, repeat_BFS_line],
               labels=['Repeat Forward A*', 'Repeat BFS'],
               loc='best')
    plt.savefig('result\\extra_credit_4.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[4] for e in result], color='red')
    repeat_BFS_line, = plt.plot(PS, [e[9] for e in result], color='blue')
    plt.xlabel('Density')
    plt.ylabel('Search Time*')
    plt.title('Density VS. Average search time)')
    plt.legend(handles=[repeat_A_star_line, repeat_BFS_line],
               labels=['Repeat Forward A*', 'Repeat BFS'],
               loc='best')
    plt.savefig('result\\extra_credit_5.png')
    plt.show()


def main_bump_in():
    dim = 101
    result = []
    for p in PS:
        a_ra, b_ra, c_ra, d_ra, t_ra, a_BFS, b_BFS, c_BFS, d_BFS, t_BFS = test_bump_in(dim, p)
        result.append((a_ra, b_ra, c_ra, d_ra, t_ra, a_BFS, b_BFS, c_BFS, d_BFS, t_BFS))

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[0] for e in result], color='red')
    repeat_BFS_line, = plt.plot(PS, [e[5] for e in result], color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average Trajectory Length')
    plt.title('Density VS. Average Trajectory Length')
    plt.legend(handles=[repeat_A_star_line, repeat_BFS_line],
               labels=['Repeat Forward A*', 'Repeat BFS'],
               loc='best')
    plt.savefig('result\\extra_credit_6.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[1] for e in result], color='red')
    repeat_BFS_line, = plt.plot(PS, [e[6] for e in result], color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average (Trajectory Length / Final Discovered Gridworld Shortest Path Length)')
    plt.title('Density VS. Average (Trajectory Length / Final Discovered Gridworld Shortest Path Length)')
    plt.legend(handles=[repeat_A_star_line, repeat_BFS_line],
               labels=['Repeat Forward A*', 'Repeat BFS'],
               loc='best')
    plt.savefig('result\\extra_credit_7.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[2] for e in result], color='red')
    repeat_BFS_line, = plt.plot(PS, [e[7] for e in result], color='blue')
    plt.xlabel('Density')
    plt.ylabel('Average (Shortest Path in Final Discovered Gridworld Length / Shortest Path in Full Gridworld Length)')
    plt.title(
        'Density VS. Average (Shortest Path in Final Discovered Gridworld Length / Shortest Path in Full Gridworld Length)')
    plt.legend(handles=[repeat_A_star_line, repeat_BFS_line],
               labels=['Repeat Forward A*', 'Repeat BFS'],
               loc='best')
    plt.savefig('result\\extra_credit_8.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[3] for e in result], color='red')
    repeat_BFS_line, = plt.plot(PS, [e[8] for e in result], color='blue')
    plt.xlabel('Density')
    plt.ylabel('Number of Cells Processed by Repeated A*')
    plt.title('Density VS. Number of Cells Processed by Repeated A*)')
    plt.legend(handles=[repeat_A_star_line, repeat_BFS_line],
               labels=['Repeat Forward A*', 'Repeat BFS'],
               loc='best')
    plt.savefig('result\\extra_credit_9.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    repeat_A_star_line, = plt.plot(PS, [e[4] for e in result], color='red')
    repeat_BFS_line, = plt.plot(PS, [e[9] for e in result], color='blue')
    plt.xlabel('Density')
    plt.ylabel('Search Time*')
    plt.title('Density VS. Average search time)')
    plt.legend(handles=[repeat_A_star_line, repeat_BFS_line],
               labels=['Repeat Forward A*', 'Repeat BFS'],
               loc='best')
    plt.savefig('result\\extra_credit_10.png')
    plt.show()

def test_main():
    x = [1, 2]
    y1 = [2, 4]
    y2 = [3, 5]
    plt.figure(figsize=(8, 5))
    line1, = plt.plot(x, y1, color="red")
    line2, = plt.plot(x, y2, color="blue")
    plt.xlabel('Density')
    plt.ylabel('Average Trajectory Length')
    plt.title('Density VS. Average Trajectory Length')
    plt.legend(handles=[line1, line2], labels=['test1', 'test2'], loc='best')
    plt.savefig('result\\extra_credit_10.png')
    plt.show()


if __name__ == '__main__':
    main()
    # test_main()
    main_bump_in()