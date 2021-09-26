import queue

from A_star import A_star_for_repeat, Node, check_exit_for_repeat, find_path


def generate_path_list(re_start_cell: Node, path: list, end_point: list, model: str):
    """
    use path to generate nodes' relationship

    :param re_start_cell: the start node of the path
    :param path: the path to the goal in the current maze
    :param end_point: the goal's position in the maze
    :param model: method to calculate distance
    :return: the path list in the node format
    """
    return_cell_nodes = []
    # test if have path
    if len(path) > 0:
        # current_cell = Node(position=path[0],
        #                     father=re_start_cell,
        #                     end_point=end_point,
        #                     model=model)

        # start node
        current_cell=re_start_cell
        return_cell_nodes.append(current_cell)
        # create relationships between nodes
        for i in range(len(path) - 1):
            new_cell = Node(position=path[i+1],
                            father=current_cell,
                            end_point=end_point,
                            model=model)
            # update current node information
            current_cell = new_cell
            return_cell_nodes.append(current_cell)
    return return_cell_nodes


def add_surround_cell(maze: list, path: list, priority_queue: queue, closed_list: list, block_list: list,
                      end_point: list, surround_exist_list:list,
                      rows: int, columns: int, model: str):
    """
    update surround node data to the priority queue
    """
    for each_cell_node in path:
        # add the passed cell to the closed list
        closed_list.append(each_cell_node.get_items())
        # check which cell can be used in the surround of the current cell
        surround_cells, block_list = check_exit_for_repeat(point=each_cell_node.get_items(),
                                                           block_list=block_list,
                                                           close_list=closed_list,
                                                           rows=rows,
                                                           columns=columns,
                                                           maze=maze)
        for surround_cell in surround_cells:
            # find if surround cell is used by other cell before
            # if surround cell is used before, it means this surround cell is closer to other cell
            if surround_cell not in surround_exist_list:
                cell_node = Node(position=surround_cell,
                                 father=each_cell_node,
                                 end_point=end_point,
                                 model=model)
            # closed_list.append(surround_cell)
                # add current cell node to the priority queue
                priority_queue.put(cell_node)
                # add current cell to list to show that this surround cell has been used
                surround_exist_list.append(surround_cell)
    return priority_queue, closed_list, block_list, surround_exist_list


def check_route(path: list, block_list_info: list, closed_list: list, maze: list):
    """
    check if the route can be passed
    """
    # path=path[1:]
    for index, each_cell in enumerate(path):
        # if the location of the maze is 1, it shows that this cell is blocked, which cannot be passed
        if maze[each_cell[0]][each_cell[1]] == 1:
            # if the cell has not been recorded before, use this cell
            if each_cell not in block_list_info:
                block_list_info.append(each_cell)
            path=path[:index]
            break
        # add this cell to che closed list
        if each_cell in closed_list:
            path=path[:index]
            break
    return path


def check_hallway(maze:list, path:list, rows:int, columns:int, re_start_cell:Node):
    if len(path)==0:
        return path
    while True:
        point = path[-1]
        surround_cells=[[point[0] - 1, point[1]], [point[0] + 1, point[1]], [point[0], point[1] - 1],
                    [point[0], point[1] + 1]]
        surround_cells_copy=surround_cells.copy()
        for surround_cell in surround_cells_copy:
            if 0<=surround_cell[0]<rows:
                if 0<=surround_cell[1]<columns:
                    if maze[surround_cell[0]][surround_cell[1]]!=1:
                        continue
            surround_cells.remove(surround_cell)
        if len(surround_cells)==2:
            if len(path)==1:
                pre_cell=re_start_cell.get_father_node().get_items()
            else:
                pre_cell=path[-2]
            surround_cells.remove(pre_cell)
            new_cell=surround_cells[0]
            path.append(new_cell)
        else:
            break
    return path


def improved_repeat_A_star(start_point: list, end_point: list, maze: list, rows: int, columns: int, model="E"):
    """
    the main algorithm of improved repeat forward A*

    """
    priority_queue = queue.PriorityQueue()
    closed_list = []
    block_list = []
    surround_exist_list=[]

    final_cell_node = None

    # add start cell info
    re_start_cell = Node(position=start_point,
                         father=None,
                         end_point=end_point,
                         model=model)
    status = False

    while not status:
        # use A* to calculate a path
        path = A_star_for_repeat(block_list=block_list.copy(),
                                 close_list=closed_list.copy(),
                                 rows=rows,
                                 columns=columns,
                                 start_cell=re_start_cell.get_items(),
                                 end_cell=end_point,
                                 model=model)

        # check if the route can be pass
        route = check_route(path=path,
                            block_list_info=block_list.copy(),
                            closed_list=closed_list.copy(),
                            maze=maze)

        # go into the hallway
        route=check_hallway(maze=maze,
                            path=route,
                            rows=rows,
                            columns=columns,
                            re_start_cell=re_start_cell)

        # change cell list to cell node list
        node_path = generate_path_list(re_start_cell=re_start_cell,
                                       path=route,
                                       end_point=end_point,
                                       model=model)

        # add surround cell info to the priority queue, it can be used when choosing the start node
        priority_queue, closed_list, block_list, surround_exist_list= add_surround_cell(path=node_path,
                                                                                        priority_queue=priority_queue,
                                                                                        closed_list=closed_list.copy(),
                                                                                        block_list=block_list.copy(),
                                                                                        surround_exist_list=surround_exist_list.copy(),
                                                                                        maze=maze,
                                                                                        end_point=end_point,
                                                                                        rows=rows,
                                                                                        columns=columns,
                                                                                        model=model)
        # check if the algorithm has found the path
        if len(route)>0:
            if route[-1] == end_point:
                final_cell_node = node_path[-1]
                status = True
                continue
        # check if the maze has a path
        if priority_queue.empty():
            status = True
            continue

        # update the new restart cell
        re_start_cell = priority_queue.get()


    if final_cell_node is None:
        # if there is not a path, return no path
        return []
    else:
        # if there is a path, return the path
        return find_path(current_node=final_cell_node, start_node=start_point)