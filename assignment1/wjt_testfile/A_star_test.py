class Stack():
    """
    implement a stack in Python
    """

    def __int__(self):
        self.items = []

    def is_empty(self):
        """
        test if the stack is empty.
        :return: True or False
        """
        if len(self.items) == 0:
            return True
        else:
            return False

    def pop(self):
        """
        pop the first data in stack
        :return: the first data in the stack
        """
        return self.items.pop()

    def push(self, data):
        """
        push the data into the stack
        :param data: the data to be pushed
        """
        self.items.append(data)

    def size(self):
        """
        find out the size of the stack
        :return: the size of the stack
        """
        return len(self.items)


class Node():
    def __int__(self, position):
        self.position = position
        self.father_node = None

    def set_father_node(self, node_name):
        self.father_node = node_name

    def get_items(self):
        return self.position

    def get_father_node(self):
        return self.father_node


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


def check_exit(point: list, block_list: list, close_list: list, rows: int, columns: int):
    return_point_list = []
    # point_upper=[point[0]-1, point[1]]
    # point_down=[point[0]+1, point[1]]
    # point_left=[point[0], point[1]-1]
    # point_right=[point[0], point[1]+1]
    point_around = [[point[0] - 1, point[1]], [point[0] + 1, point[1]], [point[0], point[1] - 1],
                    [point[0], point[1] + 1]]
    for each_point in point_around:
        # test if point in the maze
        if 0 <= each_point[0] <= rows:
            if 0 <= each_point[1] <= columns:
                # test if the point can pass
                if each_point not in block_list:
                    # test if the point has not be used
                    if each_point not in close_list:
                        return_point_list.append(each_point)
    return return_point_list


def search(open_stack: Stack, block_list: list, close_list: list, rows: int, columns: int):
    # TODO: add situation that there is no path in the maze
    # create new stack to store all the new point
    new_stack = Stack()
    len_stack = open_stack.size()
    # iterate through the stack to find the new point
    for _ in range(len_stack):
        pre_node = open_stack.pop()
        # find the available point
        point_around_list = check_exit(point=pre_node.get_items(),
                                       block_list=block_list,
                                       close_list=close_list,
                                       rows=rows,
                                       columns=columns)
        for each_point_around in point_around_list:
            # if find the end point, return all the tree
            if [rows - 1, columns - 1] in each_point_around:
                end_node = Node(position=[rows - 1, columns - 1])
                end_node.set_father_node(pre_node)
                return True, end_node
            # if not find the end point, return next stack
            else:
                new_node = Node(each_point_around)
                new_node.set_father_node(pre_node)
                new_stack.push(new_node)
                close_list.append(each_point_around)
    return False, new_stack


def find_path(current_node: Node):
    path_list = []
    while True:
        current_position = current_node.get_items()
        if current_position != [0, 0]:
            path_list.append(current_position)
            current_node = current_node.get_father_node()
            continue
        else:
            path_list.append(current_position)
            break
    path_list = path_list.reverse()
    return path_list


def A_star_search(maze: list, rows: int, columns: int):
    block_list = get_block(maze_info=maze,
                           rows=rows,
                           columns=columns)
    start_node = Node([0, 0])
    open_stack = Stack()
    open_stack.push(start_node)
    close_list = []
    close_list.append([0, 0])
    status = False
    while not status:
        status, open_stack = search(open_stack=open_stack,
                                    block_list=block_list,
                                    close_list=close_list,
                                    rows=rows,
                                    columns=columns)
    if open_stack == None:
        return []
    else:
        path_list = find_path(current_node=open_stack)
        return path_list











