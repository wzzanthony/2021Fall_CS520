from maze import Maze

def updateMaze(maze, position, status):


    if status==1:
        # current cell is block
        maze.set_obstacle(position[1], position[2])
        current_cell=maze.maze[position[0]][position[1]]
        children_list=maze.generate_children(current_cell)
        for children in children_list:
            position=children.get_position()
            if maze.maze[position[0]][position[1]].is_visited is True:
                current_cell=maze.maze[position[0]][position[1]]
                current_cell.num_unconfirmed_neighbours -=1
                current_cell.num_confirm_block+=1
                if current_cell.num_sensed_block==current_cell.num_confirm_block:
                    new_children_list=current_cell.generate_children()
                    for new_child in new_children_list:
                        new_position=new_child.get_position()
                        maze=updateMaze(maze, new_position, 2)
                if (current_cell.num_unconfirmed_neighbours==
                        (current_cell.num_sensed_block-current_cell.num_confirm_block)):
                    new_children_list=current_cell.generate_children()
                    for new_child in new_children_list:
                        new_position=new_child.get_position()
                        maze=updateMaze(maze, new_position,1)
    elif status==2:
        # current cell in empty
        maze.set_empty(position[0], position[1])
        current_cell = maze.maze[position[0]][position[1]]
        children_list = maze.generate_children(current_cell)


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



