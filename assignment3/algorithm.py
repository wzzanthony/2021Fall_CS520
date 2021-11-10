import math
import queue
from maze import Cell, Maze
import numpy as np


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


def initialize_empty_maze(maze):
    '''
    Create an empty maze, all of the cell are not obstacle
    '''
    empty_maze = Maze(maze.width, maze.height)
    empty_maze.possibility=np.ones((maze.width, maze.height))/(maze.height*maze.width)
    empty_maze.maze_terrain=maze.maze_terrain

    return empty_maze


# Repeated Forward A*
class RepeatedForwardAStar:
    def __init__(self, maze: Maze, heuristic):
        '''
        Initialize, accept a maze and the heuristic function
        '''
        self.maze = maze
        self.heuristic = heuristic
        # the Discovered Gridworld
        self.discovered_maze = None
        # the Discovered cells
        self.cell_processed = set()

    # For question 8, find a better re-start cell
    def smart_find_restart_cell(self, moved_path: list) -> Cell:
        '''
        Find the first cell that are not in the hallway
        '''
        # stats = []
        for index in range(-1, -len(moved_path) - 1, -1):  # From the last one to the first one
            cell = moved_path[index]
            x, y = cell.get_position()
            num_neighbours = 0
            num_obstacles = 0
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                # neighbour is valid
                if self.discovered_maze.position_is_valid(nx,
                                                          ny):  # After using search(), discovered_maze has already became a Maze
                    num_neighbours += 1
                    #  neighbour is obstacle
                    if self.discovered_maze.is_obstacle(nx, ny):
                        num_obstacles += 1
            # the first cell that are not in the hallway should has at most 1 obstacle in the neighbours
            if num_obstacles <= 1:
                return cell
        # no such cell found, return the last cell
        return moved_path[-1]

    def search(self, start_cell: Cell, only_bump=True, smart_restart=False):
        '''
        Search the path from start cell to goal cell
        :param start_cell:
        :param goal_cell:
        :param only_bump:
        :param smart_restart: use the strategy to find a better re-start cell
        :return:
        '''

        # current maze that the agent has percepted, initially, there is no obstacle
        self.discovered_maze = initialize_empty_maze(self.maze)

        current_cell = start_cell
        # currently moves of the agent
        moved_path = []
        # repeat until reach the goal
        find_target=False
        count=0
        # while current_cell != goal_cell:
        while not find_target:
            goal_cell_position=self.discovered_maze.find_next_goal(current_cell.get_position())
            count+=1
            goal_cell=Cell(goal_cell_position)
            # search the path in the current percepted maze
            astar = AStar(self.discovered_maze, self.heuristic)
            path = astar.search(current_cell, goal_cell)
            # print('Current at {}, path: {}'.format(current_cell, path))
            # could not find a path
            if not path:
                self.discovered_maze.update_poss(goal_cell_position, True)
                continue
                # return []

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
                self.discovered_maze.update_poss(goal_cell_position)
            # the valid path
            else:
                valid_path = path[: index_of_first_obstacle]
                # agent can only discover obstacle by bumping into them
                if only_bump:
                    cell = path[index_of_first_obstacle]
                    x, y = cell.get_position()
                    self.discovered_maze.update_poss((x,y))
                    self.discovered_maze.set_obstacle(x, y)
                    for each_cell in valid_path:
                        x, y = each_cell.get_position()
                        self.cell_processed.add((x, y))

            # agent moves in the valid path
            moved_path.extend(valid_path)
            # if valid_path[-1] == goal_cell:
            #     return moved_path
            if index_of_first_obstacle is None:
                if self.maze.get_target(valid_path[-1].get_position()):
                    find_target=True
                    print(count)
                    return count

            # Decide the cell to start with
            # smart restart
            if smart_restart:
                current_cell = self.smart_find_restart_cell(moved_path)
            else:
                # just re-start from the last valid cell
                current_cell = valid_path[-1]
                if count%5000==0:
                    print(count)

            # if not only_bump:
            #     # now the agent knows the status of the cells around the path, update the self.discovered_maze
            #     for cell in valid_path:
            #         x, y = cell.get_position()
            #         # the neighbour of the valid cell
            #         for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)):
            #             nx, ny = x + dx, y + dy
            #             # neighbour is valid and it's obstacle
            #             if self.maze.position_is_valid(nx, ny):
            #                 self.cell_processed.add((nx, ny))
            #                 if self.maze.is_obstacle(nx, ny):
            #                     self.discovered_maze.set_obstacle(nx, ny)
        return []

