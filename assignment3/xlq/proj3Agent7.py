import math
import queue
import random

import numpy as np

from maze import Cell, Maze

class AStar:
    def __init__(self, maze: Maze, heuristic):
        '''
        maze: from maze_none.py
        '''
        self.maze = maze
        self.heuristic = heuristic

    def search(self, start_cell: Cell, goal_cell: Cell):
        # user a priority queue, always select the cell with smallest fn to explore
        open_list = queue.PriorityQueue()
        start_cell.set_father_node(None)
        # calculate fn and hn
        start_cell.update_fn(self.heuristic, goal_cell)

        open_list.put((start_cell.fn, start_cell))  # (fn, Cell)
        closed_dict = dict()  # cells already visited, {Cell : gn}
        while not open_list.empty():
            fn, current_cell = open_list.get()
            # reach the goal cell
            if current_cell.get_position() == goal_cell.get_position():
                return self.get_path(current_cell)
            # the current cell has been visited
            if current_cell in closed_dict:
                continue
            closed_dict[current_cell] = current_cell.gn
            # Generate the children of n (neighbors believed or known to be unoccupied)
            for child_cell in self.maze.generate_children(current_cell, goal_cell):
                # update hn and fn of child cell
                child_cell.update_fn(self.heuristic, goal_cell)
                '''
                The successors of n are the children n0 that are newly discovered, or g(n0) > g(n) + 1.
                For each successor n0, re-set g(n0) = g(n) + 1, representing the newly discovered shortest path from the start node to n0 newly discovered, insert n0 into the fringe at priority f(n0) = g(n0) + h(n0)
                '''
                if child_cell not in closed_dict:
                    open_list.put((child_cell.fn, child_cell))
                # g(n0) > g(n) + 1, insert n0 into the fringe at priority f(n0) = g(n0) + h(n0),
                elif child_cell.gn < closed_dict[child_cell]:
                    closed_dict.pop(child_cell)
                    open_list.put((child_cell.fn, child_cell))
        # no path found
        return []

    def get_path(self, current_cell: Cell):
        '''
        Get the path, search from current cell to start cell
        '''
        path = []
        # starts from the goal and go back to the start cell
        while current_cell is not None:
            path.append(current_cell)
            current_cell = current_cell.father_node
        return path[::-1]


def initialize_empty_maze(maze: Maze):
    '''
    Create an empty maze, all of the cell are not obstacle
    '''
    empty_maze = Maze(maze.width, maze.height)
    empty_maze.prob_contain_mat = np.ones((maze.height, maze.width)) / (maze.height * maze.width)
    empty_maze.terrain_maze = maze.terrain_maze
    empty_maze.prob_find_mat = np.ones((maze.height, maze.width))
    return empty_maze

# def getDistance(cell1: Cell, cell2: Cell):
#     x1, y1 = cell1.get_position()
#     x2, y2 = cell2.get_position()
#     dist = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
#     return dist


class RepeatedForwardAStar:
    def __init__(self, maze: Maze, heuristic):
        self.maze = maze
        self.heuristic = heuristic
        self.discovered_maze = None

    # def findGoal(self, start_cell:Cell):
    #     goal_cell_lst = []
    #     contain_p_mat = self.maze.initialize_contain_p()
    #     x_lst, y_lst = np.nonzero(contain_p_mat == contain_p_mat.max())
    #     p_max_lst = []
    #     # max_p more than 1 before fixed by distance
    #     if len(x_lst) != 1 and len(y_lst) != 1:
    #         for i in range(len(x_lst)):
    #             p_max_lst.append((x_lst[i], y_lst[i]))  # goal cell position list, [(x1,y1), (x2,y2),...]
    #             goal_cell_lst.append(Cell(position=(x_lst[i], y_lst[i])))   # class list, [goal_cell1, goal_cell2,...]
    #             for p_position in p_max_lst:
    #                 x,y = p_position
    #                 contain_p_mat[x][y] = contain_p_mat[x][y] / getDistance(start_cell, Cell(position=(x_lst[i], y_lst[i])))
    #         x_fix_lst, y_fix_lst = np.nonzero(contain_p_mat == contain_p_mat.max())   # =========!!!!============
    #         if len(x_fix_lst) != 1 and len(y_fix_lst) != 1:
    #             a = random.randint(0, len(x_fix_lst))
    #             return Cell(position=(x_fix_lst[a], y_fix_lst[a]))
    #         elif len(x_fix_lst) == 1 and len(y_fix_lst) == 1:
    #             return Cell(position=(x_fix_lst[0], y_fix_lst[0]))
    #     # max_p more = 1 before fixed by distance
    #     elif len(x_lst) == 1 and len(y_lst) == 1:
    #         return Cell(position=(x_lst[0], y_lst[0]))

    def search(self, start_cell: Cell):
        cnt = 0
        self.discovered_maze = initialize_empty_maze(self.maze)
        current_cell = start_cell
        while True:
            goal_cell = self.discovered_maze.find_next_goal(current_cell)
            astar = AStar(self.discovered_maze, self.heuristic)
            path = astar.search(current_cell, goal_cell)
            cnt += 1
            print(f'move:{cnt}')
            if not path:
                self.discovered_maze.update_prob(goal_cell, True)
                continue

            # agent starts to move, until reach an obstacle
            index_of_first_obstacle = None
            for i, cell in enumerate(path):
                x, y = cell.get_position()
                if self.maze.is_obstacle(x, y):
                    index_of_first_obstacle = i
                    break
            # find the actual valid path
            # actually there is no obstacle in the path, so the whole path is valid
            if index_of_first_obstacle is None:
                valid_path = path
                self.discovered_maze.update_prob(goal_cell)
            # the valid path
            else:
                valid_path = path[: index_of_first_obstacle]
                cell = path[index_of_first_obstacle]
                x, y = cell.get_position()
                self.discovered_maze.update_prob(cell)
                self.discovered_maze.set_obstacle(x, y)
            # if valid_path[-1] == goal_cell:
            #     return moved_path
            if index_of_first_obstacle is None:
                if self.maze.exam_target(valid_path[-1]):
                    find_target = True
                    return cnt
            current_cell = valid_path[-1]
            print(valid_path)
            print(current_cell.get_position())
        return 0


























