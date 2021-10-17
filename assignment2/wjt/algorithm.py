import queue

from maze import Maze, Cell


def updateMaze(maze, position, status):
    if status == 1:
        # current cell is block
        maze.set_obstacle(position[0], position[1])
        current_cell = maze.maze[position[0]][position[1]]
        children_list = maze.generate_spacial_sensed_neighbours(current_cell)
        for i, j in children_list:
            current_cell = maze.maze[i][j]
            if current_cell.is_visited() is True and current_cell.num_unconfirmed_neighbours != 0:

                current_cell.num_unconfirmed_neighbours -= 1
                current_cell.num_confirm_block += 1

                if current_cell.num_sensed_block == current_cell.num_confirm_block:
                    # others are empty

                    current_cell.num_confirm_empty += current_cell.num_unconfirmed_neighbours
                    current_cell.num_unconfirmed_neighbours = 0
                    maze.maze[i][j] = current_cell

                    new_children_list = maze.generate_spacial_sensed_neighbours(current_cell)
                    for new_position in new_children_list:
                        info = maze.data[new_position[0]][new_position[1]]
                        if info == 0 or info == "G":
                            maze.set_empty(new_position[0], new_position[1])
                            maze = updateMaze(maze, new_position, 2)

                if (current_cell.num_unconfirmed_neighbours ==
                        (current_cell.num_sensed_block - current_cell.num_confirm_block)):
                    # others are blocks
                    current_cell.num_confirm_block += current_cell.num_unconfirmed_neighbours
                    current_cell.num_unconfirmed_neighbours = 0
                    maze.maze[i][j] = current_cell

                    new_children_list = maze.generate_spacial_sensed_neighbours(current_cell)
                    for new_position in new_children_list:
                        info = maze.data[new_position[0]][new_position[1]]
                        if info == 0 or info == "G":
                            maze.set_obstacle(new_position[0], new_position[1])
                            maze = updateMaze(maze, new_position, 1)
    elif status == 2:
        # current cell in empty
        maze.set_empty(position[0], position[1])
        current_cell = maze.maze[position[0]][position[1]]
        children_list = maze.generate_spacial_sensed_neighbours(current_cell)

        for i, j in children_list:
            current_cell = maze.maze[i][j]

            if current_cell.is_visited() is True and current_cell.num_unconfirmed_neighbours != 0:

                current_cell.num_unconfirmed_neighbours -= 1
                current_cell.num_confirm_empty += 1

                if current_cell.num_sensed_block == current_cell.num_confirm_block:
                    # others are empty
                    current_cell.num_confirm_empty += current_cell.num_unconfirmed_neighbours
                    current_cell.num_unconfirmed_neighbours = 0
                    maze.maze[i][j] = current_cell

                    new_children_list = maze.generate_spacial_sensed_neighbours(current_cell)
                    for new_position in new_children_list:
                        info = maze.data[new_position[0]][new_position[1]]
                        if info == 0 or info == "G":
                            maze.set_empty(new_position[0], new_position[1])
                            maze = updateMaze(maze, new_position, 2)
                if (current_cell.num_unconfirmed_neighbours ==
                        (current_cell.num_sensed_block - current_cell.num_confirm_block)):
                    # others are blocks
                    current_cell.num_confirm_block += current_cell.num_unconfirmed_neighbours
                    current_cell.num_unconfirmed_neighbours = 0
                    maze.maze[i][j] = current_cell

                    new_children_list = maze.generate_spacial_sensed_neighbours(current_cell)
                    for new_position in new_children_list:
                        info = maze.data[new_position[0]][new_position[1]]
                        if info == 0 or info == "G":
                            maze.set_obstacle(new_position[0], new_position[1])
                            maze = updateMaze(maze, new_position, 1)
    return maze

    # if status==1:
    #     # current cell is block
    #     maze.set_obstacle(position[1], position[2])
    #     current_cell=maze.maze[position[0]][position[1]]
    #     children_list=maze.generate_children(current_cell)
    #     for children in children_list:
    #         position=children.get_position()
    #         if maze.maze[position[0]][position[1]].is_visited is True:
    #             current_cell=maze.maze[position[0]][position[1]]
    #             current_cell.num_unconfirmed_neighbours -=1
    #             current_cell.num_confirm_block+=1
    #             if current_cell.num_sensed_block==current_cell.num_confirm_block:
    #                 new_children_list=current_cell.generate_children()
    #                 for new_child in new_children_list:
    #                     new_position=new_child.get_position()
    #                     maze=updateMaze(maze, new_position, 2)
    #             if (current_cell.num_unconfirmed_neighbours==
    #                     (current_cell.num_sensed_block-current_cell.num_confirm_block)):
    #                 new_children_list=current_cell.generate_children()
    #                 for new_child in new_children_list:
    #                     new_position=new_child.get_position()
    #                     maze=updateMaze(maze, new_position,1)
    # elif status==2:
    #     # current cell in empty
    #     maze.set_empty(position[0], position[1])
    #     current_cell = maze.maze[position[0]][position[1]]
    #     children_list = maze.generate_children(current_cell)


class AStar:
    def __init__(self, maze: Maze, heuristic):
        '''
        maze: from maze.py
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
            for child_cell in self.maze.generate_children(current_cell):
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
    """
    Create an empty maze, all of the cell are not obstacle
    """
    empty_maze = Maze(maze.width, maze.height)
    empty_maze = initialize_sense_block(maze, empty_maze)
    return empty_maze


def initialize_sense_block(maze, empty_maze):
    for i in range(maze.width):
        for j in range(maze.width):
            dij = [(1, 1), (1, -1), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (-1, 0)]
            for di, dj in dij.copy():
                ni, nj = i + di, j + dj
                if maze.position_is_valid(ni, nj):
                    empty_maze.maze[i][j].num_sensed_neighbours += 1
                    empty_maze.maze[i][j].num_unconfirmed_neighbours += 1
                    if maze.is_obstacle(ni, nj):
                        empty_maze.maze[i][j].num_sensed_block += 1

    return empty_maze


# Sensing Repeated Forward A*
class SensingRepeatedForwardAStar:
    def __init__(self, maze: Maze, heuristic):
        """
        Initialize, accept a maze and the heuristic function
        """
        self.maze = maze
        self.heuristic = heuristic
        # the Discovered Gridworld
        # self.discovered_maze = None
        self.discovered_maze = initialize_empty_maze(self.maze)
        # the Discovered cells
        self.cell_processed = set()
        self.replan_time=0

    # For question 8, find a better re-start cell
    def smart_find_restart_cell(self, moved_path: list) -> Cell:
        """
        Find the first cell that are not in the hallway
        """
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

    def initialize_visit_cell(self, position, status="empty"):
        current_cell = self.discovered_maze.maze[position[0]][position[1]]
        if status == "empty":
            if not current_cell.is_visited():
                if self.discovered_maze.data[position[0]][position[1]] == "1":
                    current_cell.set_visit()
                    current_cell.set_empty()
                    surround_cells_info = self.discovered_maze.sense_arround(current_cell)
                    current_cell.set_confirm_block(surround_cells_info[0])
                    current_cell.set_confirm_empty(surround_cells_info[1])
                    current_cell.change_unconfirmed_neighbours(sum(surround_cells_info))
                    self.discovered_maze.maze[position[0]][position[1]] = current_cell
                    # self.discovered_maze=updateMaze(self.discovered_maze, position, 2)
                else:
                    current_cell.set_visit()
                    current_cell.set_empty()
                    surround_cells_info = self.discovered_maze.sense_arround(current_cell)
                    current_cell.set_confirm_block(surround_cells_info[0])
                    current_cell.set_confirm_empty(surround_cells_info[1])
                    current_cell.change_unconfirmed_neighbours(sum(surround_cells_info))
                    self.discovered_maze.maze[position[0]][position[1]] = current_cell
                    self.discovered_maze = updateMaze(self.discovered_maze, position, 2)
        else:
            if self.discovered_maze.data[position[0]][position[1]] == "#":
                current_cell.set_block()
                self.discovered_maze.maze[position[0]][position[1]] = current_cell
                # self.discovered_maze=updateMaze(self.discovered_maze, position, 1)
            else:
                current_cell.set_block()
                self.discovered_maze.maze[position[0]][position[1]] = current_cell
                self.discovered_maze = updateMaze(self.discovered_maze, position, 1)
        if status == "empty":
            if current_cell.num_sensed_block == current_cell.num_confirm_block:
                # others are empty
                current_cell.num_confirm_empty += current_cell.num_unconfirmed_neighbours
                current_cell.num_unconfirmed_neighbours = 0
                self.discovered_maze.maze[position[0]][position[1]] = current_cell
                surround_cells_info = self.discovered_maze.generate_spacial_sensed_neighbours(current_cell)
                for i, j in surround_cells_info:
                    info = self.discovered_maze.data[i][j]
                    if info == 0 or info == "G":
                        self.discovered_maze.set_empty(i, j)
                        self.discovered_maze = updateMaze(self.discovered_maze, [i, j], 2)

            elif (current_cell.num_unconfirmed_neighbours ==
                  (current_cell.num_sensed_block - current_cell.num_confirm_block)):
                # others are blocks
                current_cell.num_confirm_block += current_cell.num_unconfirmed_neighbours
                current_cell.num_unconfirmed_neighbours = 0
                self.discovered_maze.maze[position[0]][position[1]] = current_cell
                surround_cells_info = self.discovered_maze.generate_spacial_sensed_neighbours(current_cell)
                for i, j in surround_cells_info:
                    info = self.discovered_maze.data[i][j]
                    if info == 0 or info == "G":
                        self.discovered_maze.set_obstacle(i, j)
                        self.discovered_maze = updateMaze(self.discovered_maze, [i, j], 1)

    def initialize_discovered_maze(self):
        self.discovered_maze = initialize_empty_maze(self.maze)
        self.replan_time=0

    def search(self, start_cell: Cell, goal_cell: Cell, know_four_neighbours=False, use_infer_method=False, smart_restart=False, infer_more=False):
        '''
        Search the path from start cell to goal cell
        :param start_cell:
        :param goal_cell:
        :param agent:
        :param smart_restart: use the strategy to find a better re-start cell
        :return:
        '''

        # current maze that the agent has percepted, initially, there is no obstacle
        # self.discovered_maze = initialize_empty_maze(self.maze)

        current_cell = start_cell
        # currently moves of the agent
        moved_path = []
        # repeat until reach the goal
        while current_cell != goal_cell:
            # search the path in the current percepted maze
            self.replan_time+=1
            astar = AStar(self.discovered_maze, self.heuristic)
            path = astar.search(current_cell, goal_cell)
            # print('Current at {}, path: {}'.format(current_cell, path))
            # could not find a path
            if not path:
                return []
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
            # the valid path
            else:
                valid_path = path[: index_of_first_obstacle]

                # agent can only discover obstacle by bumping into them

                # for each_cell in valid_path:
                #     x, y = each_cell.get_position()
                #     self.cell_processed.add((x, y))
            # agent moves in the valid path
            if len(moved_path) != 0:
                if moved_path[-1] == valid_path[0]:
                    moved_path.pop()
            if len(moved_path)!=0:
                last_cell=moved_path[-1]
            else:
                last_cell=None
            moved_path.extend(valid_path)
            if valid_path[-1] == goal_cell:
                for each_cell in valid_path:
                    x, y = each_cell.get_position()
                    self.cell_processed.add((x, y))
                return moved_path

            if not know_four_neighbours:
                for each_cell in valid_path:
                    x, y = each_cell.get_position()
                    self.discovered_maze.set_empty(x, y)
                    if use_infer_method:
                        self.initialize_visit_cell([x, y], "empty")
                    self.cell_processed.add((x, y))

                if index_of_first_obstacle is not None:
                    cell = path[index_of_first_obstacle]
                    x, y = cell.get_position()
                    self.discovered_maze.set_obstacle(x, y)
                    if use_infer_method:
                        self.initialize_visit_cell([x, y], "block")
                if infer_more:

                    infer_valid_path=valid_path.copy()
                    if isinstance(last_cell, Cell):
                        infer_valid_path.insert(0,last_cell)

                    if len(infer_valid_path)>1:
                        for index_cell_in_vaild_path in range(len(infer_valid_path)-1):
                            position_list=self.discovered_maze.inferMoreEmpty(infer_valid_path[index_cell_in_vaild_path],
                                                                              infer_valid_path[index_cell_in_vaild_path+1])
                            if len(position_list)!=0:
                                for i,j in position_list:
                                    if self.discovered_maze.data[i][j] == 0 or self.discovered_maze.data[i][j] == "G":
                                        self.discovered_maze = updateMaze(self.discovered_maze, [i, j], 2)

            # agent moves in the valid path
            # if len(moved_path)!=0:
            #     if moved_path[-1]==valid_path[0]:
            #         moved_path.pop()
            # moved_path.extend(valid_path)
            # if valid_path[-1] == goal_cell:
            #     return moved_path

            # Decide the cell to start with
            # smart restart
            if smart_restart:
                current_cell = self.smart_find_restart_cell(moved_path)
            else:
                # just re-start from the last valid cell
                current_cell = valid_path[-1]

            if know_four_neighbours:
                # now the agent knows the status of the cells around the path, update the self.discovered_maze
                if index_of_first_obstacle is not None:
                    cell = path[index_of_first_obstacle]

                for cell in valid_path:
                    x, y = cell.get_position()
                    # the neighbour of the valid cell
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)):
                        nx, ny = x + dx, y + dy
                        # neighbour is valid and it's obstacle
                        if self.maze.position_is_valid(nx, ny):
                            self.cell_processed.add((nx, ny))
                            if self.maze.is_obstacle(nx, ny):
                                self.discovered_maze.set_obstacle(nx, ny)
        return []
