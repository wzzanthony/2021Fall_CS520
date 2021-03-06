import math
import queue


class Node:

    def __init__(self, position: list, father, end_point: list, model="E"):
        self.position = position
        self.father_node = father
        if model == "E":
            # Use Euclidean Distance
            self.hn = math.sqrt(math.pow(position[0] - end_point[0], 2) + math.pow(position[1] - end_point[1], 2))
        elif model == "M":
            # Use Manhattan Distance
            self.hn = abs(position[0] - end_point[0]) + abs(position[1] - end_point[1])
        elif model == "C":
            # Use Chebyshev Distance
            self.hn = max(abs(position[0] - end_point[0]), abs(position[1] - end_point[1]))
        else:
            self.hn = math.sqrt(math.pow(position[0] - end_point[0], 2) + math.pow(position[1] - end_point[1], 2))
        if father is None:
            self.gn = 0
        else:
            self.gn = father.get_gn() + 1
        self.fn = self.hn + self.gn

    def set_father_node(self, node_name):
        self.father_node = node_name

    def get_items(self):
        return self.position

    def get_father_node(self):
        return self.father_node

    def get_hn(self):
        return self.hn

    def get_gn(self):
        return self.gn

    def get_fn(self):
        return self.fn

    # add for priority queue
    # def __gt__(self, other):
    #     return self.fn > other.get_fn()
    #
    # def __eq__(self, other):
    #     return self.fn == other.get_fn()

    def __lt__(self, other):
        return self.fn < other.get_fn()


def get_block(maze_info: list, rows: int, columns: int):
    """
    find all the blocks in the maze
    :param maze_info: information of the maze
    :param rows: row of the maze
    :param columns: column of the maze
    :return: block list of the maze
    """
    block_list = []
    for row in range(rows):
        for col in range(columns):
            if maze_info[row][col] == 1:
                block_list.append([row, col])
    return block_list


def check_exit(point: list, maze: list, close_list: dict, rows: int, columns: int):
    """
    check if the points are existing
    :param point: current point
    :param block_list: the point that cannot pass
    :param close_list: the point that already used
    :param rows: total rows of the maze
    :param columns: total columns of the maze
    """
    return_point_list = []
    # the around point of current point
    # we only find point from four directions: up, down, left, right
    point_around = [[point[0] - 1, point[1]], [point[0] + 1, point[1]], [point[0], point[1] - 1],
                    [point[0], point[1] + 1]]
    for each_point in point_around:
        # test if point in the maze
        if 0 <= each_point[0] < rows:
            if 0 <= each_point[1] < columns:
                # test if the point can pass
                if maze[each_point[0]][each_point[1]]!=1:
                    # test if the point has not be used
                    if str(each_point[0])+","+str(each_point[1]) not in close_list:
                        return_point_list.append(each_point)
    return return_point_list


def check_exit_for_repeat(maze:list, point: list, block_list: list, close_list: list, rows: int, columns: int):
    return_point_list = []
    # the around point of current point
    # we only find point from four directions: up, down, left, right
    point_around = [[point[0] - 1, point[1]], [point[0] + 1, point[1]], [point[0], point[1] - 1],
                    [point[0], point[1] + 1]]
    for each_point in point_around:
        # test if point in the maze
        if 0 <= each_point[0] < rows:
            if 0 <= each_point[1] < columns:
                # test if the point can pass
                if maze[each_point[0]][each_point[1]]==1:
                    if each_point not in block_list:
                        block_list.append(each_point)
                elif each_point not in close_list:
                    return_point_list.append(each_point)
    return return_point_list, block_list

def search(open_stack, block_list: list, close_list: list, rows: int, columns: int, model):
    """
    search path in the maze
    :param open_stack: point that should be test in the next iteration
    :param block_list: point that cannot pass
    :param close_list: point that already used
    :param rows: total rows of the maze
    :param columns: total columns of the maze
    :param model: use distance model
    """
    # create new stack to store all the new point


    if open_stack.empty():
        return True, None


    # find the new point
    pre_node = open_stack.get()
    # find the available point
    point_around_list = check_exit(point=pre_node.get_items(),
                                   block_list=block_list,
                                   close_list=close_list,
                                   rows=rows,
                                   columns=columns)
    for each_point_around in point_around_list:
        # if find the end point, return all the tree
        if [rows - 1, columns - 1] == each_point_around:
            end_node = Node(position=[rows - 1, columns - 1],
                            father=pre_node,
                            end_point=[rows - 1, columns - 1],
                            model=model)
            return True, end_node
        # if not find the end point, add the current point to the tree, then return next stack
        else:
            new_node = Node(each_point_around,
                            father=pre_node,
                            end_point=[rows - 1, columns - 1],
                            model=model)
            open_stack.put(new_node)
            close_list.append(each_point_around)
    return False, open_stack



def search_repeat(open_stack, block_list: list, close_list: list, rows: int, columns: int, model):
    """
    search path in the maze
    :param open_stack: point that should be test in the next iteration
    :param block_list: point that cannot pass
    :param close_list: point that already used
    :param rows: total rows of the maze
    :param columns: total columns of the maze
    :param model: use distance model
    """
    # create new stack to store all the new point


    if open_stack.empty():
        return True, None


    # find the new point
    pre_node = open_stack.get()
    # find the available point
    point_around_list = check_exit(point=pre_node.get_items(),
                                   block_list=block_list,
                                   close_list=close_list,
                                   rows=rows,
                                   columns=columns)
    for each_point_around in point_around_list:
        # if find the end point, return all the tree
        if [rows - 1, columns - 1] == each_point_around:
            end_node = Node(position=[rows - 1, columns - 1],
                            father=pre_node,
                            end_point=[rows - 1, columns - 1],
                            model=model)
            return True, end_node
        # if not find the end point, add the current point to the tree, then return next stack
        else:
            new_node = Node(each_point_around,
                            father=pre_node,
                            end_point=[rows - 1, columns - 1],
                            model=model)
            open_stack.put(new_node)
            close_list.append(each_point_around)
    return False, open_stack


def find_path(current_node: Node, start_node=[0, 0]):
    """
    find the way from the existing tree
    :param current_node: when finding out the available way, this is the final fringe
    """
    path_list = []
    # search the father fringe
    while True:
        current_position = current_node.get_items()
        # if not find the top, keep searching
        if current_position != start_node:
            path_list.append(current_position)
            current_node = current_node.get_father_node()
            continue
        else:
            # if find the top, stop searching
            path_list.append(current_position)
            break
    # reverse the list, get the way
    path_list.reverse()
    return path_list


def A_star_search(maze: list, rows: int, columns: int, model="E"):
    """
    A star algorithm
    :param maze: the information of the maze
    :param rows: the total rows of the maze
    :param columns: the total columns of the maze
    :param model: the model used to calculate distance
    :               "E" stand for "Euclidean Distance"
    :               "M" stand for "Manhattan Distance"
    :               "C" stand for "Chebyshev Distance"
    """
    # get block info
    block_list = get_block(maze_info=maze,
                           rows=rows,
                           columns=columns)
    # create start fringe of the tree
    start_node = Node(position=[0, 0],
                      father=None,
                      end_point=[rows - 1, columns - 1],
                      model=model)
    # use priorityQueue
    open_stack = queue.PriorityQueue()
    open_stack.put(start_node)

    # create the close list, put the start point into the list
    close_list = [[0, 0]]
    status = False
    # start finding
    while not status:
        status, open_stack = search(open_stack=open_stack,
                                    block_list=block_list,
                                    close_list=close_list,
                                    rows=rows,
                                    columns=columns,
                                    model=model)

    if open_stack is None:
        # if there  is no way out, return empty list
        return []
    else:
        # if there is a way out, return that way
        path_list = find_path(current_node=open_stack)
        return path_list


def A_star_for_repeat(block_list: list, rows: int, columns: int, start_cell: list, end_cell: list, model: str,
                      close_list: list):
    # create start fringe of each A*
    start_node = Node(position=start_cell,
                      father=None,
                      end_point=end_cell,
                      model=model)

    # use priorityQueue
    open_stack = queue.PriorityQueue()
    open_stack.put(start_node)

    status = False
    # find the path using A*
    print("************************************")
    while not status:
        status, open_stack = search(open_stack=open_stack,
                                    block_list=block_list,
                                    close_list=close_list,
                                    rows=rows,
                                    columns=columns,
                                    model=model)

    print("************************************")

    if open_stack is None:
        # if there  is no way out, return empty list
        return []
    else:
        # if there is a way out, return that way
        path_list = find_path(current_node=open_stack,
                              start_node=start_cell)
        return path_list
