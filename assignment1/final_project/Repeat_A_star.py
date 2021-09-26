from A_star import A_star_for_repeat


def check_route(path: list, block_list: list, maze: list, rows: int, columns: int):
    for index, each_cell in enumerate(path):
        if maze[each_cell[0]][each_cell[1]] == 1:
            if each_cell not in block_list:
                block_list.append(each_cell)
            path = path[:index]
            break
        else:
            # check if the around cell of the current cell are obstacle
            # TODO the structure of the code should be optimized, there are some duplicate code down here
            # under one
            if each_cell[0] + 1 < rows:
                if maze[each_cell[0] + 1][each_cell[1]] == 1:
                    if each_cell not in block_list:
                        block_list.append([each_cell[0]+1, each_cell[1]])
            # upper one
            if each_cell[0] - 1 >= 0:
                if maze[each_cell[0] - 1][each_cell[1]] == 1:
                    if each_cell not in block_list:
                        block_list.append([each_cell[0]-1, each_cell[1]])
            # right one
            if each_cell[1] + 1 < columns:
                if maze[each_cell[0]][each_cell[1] + 1] == 1:
                    if each_cell not in block_list:
                        block_list.append([each_cell[0], each_cell[1]+1])
            # left one
            if each_cell[1] - 1 >= 0:
                if maze[each_cell[0]][each_cell[1] - 1] == 1:
                    if each_cell not in block_list:
                        block_list.append([each_cell[0], each_cell[1]-1])
    # delete the start cell
    path = path[1:]
    return path, block_list


def repeat_A_star(maze: list, rows: int, columns: int, start_cell: list, end_cell: list, model='E'):
    # record the cell that has been passed
    path_list = [start_cell]
    # record the obstacle that have been found
    block_list = []
    # record if the algorithm has found the path
    status = False

    while not status:
        found_path = A_star_for_repeat(block_list=block_list,
                                       rows=rows,
                                       columns=columns,
                                       start_cell=start_cell,
                                       end_cell=end_cell,
                                       model=model,
                                       close_list=[])
        # cannot found path, return no route
        if len(found_path) == 0 and path_list[-1] != end_cell:
            status = True
            continue
        # check if path is blocked, and find obstacles when iterating the path
        worked_path_list, block_list = check_route(path=found_path,
                                                   block_list=block_list,
                                                   maze=maze,
                                                   rows=rows,
                                                   columns=columns)
        # check if path has been found
        if len(worked_path_list)!=0:
            if worked_path_list[-1] == end_cell:
                # if the path has been found, return the found path
                path_list.extend(worked_path_list)
                status = True
                # continue
            else:
                # if the path has not been found, change start cell to current cell and re-calculate the path
                path_list.extend(worked_path_list)
                start_cell = path_list[-1]

    # test if the path is available
    if path_list[-1] != end_cell:
        return []
    else:
        return path_list
