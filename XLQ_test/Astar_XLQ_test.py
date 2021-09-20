import math
import queue
import random


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y



class Cell():
    """
    create Node structure to generate tree
    """

    def __init__(self, Position: list, Father_node, Goal_cell):  # position = [x, y]
        self.position = [0, 0]
        self.father_node = Father_node

        # Heuristic cost from n to Goal_cell: hn
        # self.hn = math.sqrt((position[0] - goal_cell[0]) ** 2 + (position[1] - goal_cell[1]) ** 2)        # Use Euclidean Distance
        self.hn = abs(Position[0] - Goal_cell[0]) + abs(Position[1] - Goal_cell[1])  # Use Manhattan Distance
        # self.hn = max(abs(position[0] - goal_cell[0]), abs(position[1] - goal_cell[1]))                   # Use Chebyshev Distance

        # Real cost from Start_cell to n: g(n)
        if Father_node is None:
            gn = 0
        else:
            gn = self.father_node.get_gn()

        # Total cost: fn
        self.fn = self.hn + self.gn

    def set_father_node(self, node_name):
        '''
        Set father node for the point
        :param node_name:
        :return:
        '''
        self.father_node = node_name
        if self.father_node == None:
            self.gn = 0
        else:
            self.gn += 1

    def get_position(self):
        '''
        Get position information of the node
        :return:
        '''
        return self.position

    def get_father_node(self):
        # change current node to father node
        return self.father_node

    def get_hn(self):
        return self.hn

    def get_gn(self):
        return self.gn

    def get_fn(self):
        return self.fn

    def __gt__(self, other):
        return self.fn > other.get_fn()

    def __eq__(self, other):
        return self.fn == other.get_fn()

    def __lt__(self, other):
        return self.fn < other.get_fn()


class Map():
    obstacle_num = 5000  # obstacles number

    def __init__(self, Width, Height):
        self.width = Width
        self.height = Height
        self.data = [[0 for i in range(Height)] for j in range(Width)]

    # Print out an empty maze
    def maze_show(self):
        for i in range(self.width):
            for j in range(self.height):
                print(self.data[i][j], end=' ')
            print(' ')

    # Set obstacles with '#'
    def obstacle(self, x, y):
        self.data[x][y] = "#"

    # Print out a maze with obstacles
    def maze_obstacles_show(self, Obstacle_X, Obstacle_Y):
        for num in range(self.obstacle_num):
            i = random.randint(0, (Obstacle_X - 1))
            j = random.randint(0, (Obstacle_Y - 1))
            self.obstacle(i, j)
        self.maze_show()

    # Set start cell with 'S'
    def draw_start(self, cell: Cell):
        i = cell.get_position()[0]
        j = cell.get_position()[1]
        self.data[i]][j] = "S"

    # Set goal cell with 'G'
    def draw_goal(self, cell: Cell):
        self.data[cell.position[0]][cell.position[1]] = "G"


class Astar():
    def __init__(self, Start_cell: Cell, Goal_cell: Cell, Map: Map):
        self.start_cell = Start_cell  # Start point
        self.goal_cell = Goal_cell  # Goal point
        self.current_cell = Start_cell  # Current point
        self.map = Map  # Map
        self.open_list = queue.PriorityQueue()  # (fn, [x,y])
        self.close_list = []
        self.path_list = []

    def get_cells_around(self):
        cells_around_list = []
        cell_up = Cell(Position=[self.current_cell.position[0] + 1, self.current_cell.position[1]], Father_node=,
                       Goal_cell=goal_cell)
        cell_down = Cell(Position=[self.current_cell.position[0] - 1, self.current_cell.position[1]], Father_node=,
                         Goal_cell=goal_cell)
        cell_right = Cell(Position=[self.current_cell.position[0], self.current_cell.position[1] + 1], Father_node=,
                          Goal_cell=goal_cell)
        cell_left = Cell(Position=[self.current_cell.position[0], self.current_cell.position[1] - 1], Father_node=,
                         Goal_cell=goal_cell)
        cells_around_list.extend([cell_up, cell_down, cell_right, cell_left])
        return cells_around_list

    def isin_openlist(self, Cell: Cell):
        for cnt in range(self.open_list.qsize()):
            each_cell = self.open_list.get()
            if each_cell[1] == Cell.position:
                return True
        return False

    def isin_closelist(self, Cell: Cell):
        for item in self.close_list:
            if item == Cell.position:
                return True
        return False

    def is_obstacle(self, Cell: Cell):
        if self.map.data[Cell.position[0]][Cell.position[1]] == '#':
            return True
        else:
            return False

    def get_available_cells(self, Cells_around_list: list):
        available_cells_list = []
        for each_cell in Cells_around_list:
            # Test if cell in the maze
            if 0 < each_cell.get_position()[0] <= self.map.width:
                if 0 < each_cell.get_position()[1] <= self.map.height:
                    # Test if cell can pass
                    if each_cell not in self.close_list:
                        available_cells_list.append(each_cell)
        return available_cells_list

    def search_next_cell(self, Cells_around_list: list, Available_cells_list: list):
        '''
        Find a cell with least fn in the open_list and return it
        :param Cells_around_list:
        :param Available_cells_list:
        :return:
        '''
        current_cell_fn = self.current_cell.get_fn()
        # (fn, Cell) in PriorityQueue
        self.open_list.put((current_cell_fn, self.current_cell))
        # Check if cell is adjunct to the current cell
        for cell in Cells_around_list:
            # Check if cell is available
            if cell in Available_cells_list:
                # Put available cell in the open_list
                self.open_list.put((cell.get_fn(), cell))
        # Get a cell with least fn from the PriorityQueue
        next_cell = self.open_list.get()
        return next_cell

    def get_path(self, Next_cell: Cell):
        '''
        Put
        :param Next_cell:
        :return:
        '''
        path_list = self.path_list.append(Next_cell.position)
        self.map.data[Next_cell.position[0]][Next_cell.position[1]] = '*'
        return path_list


if __name__ == '__main__':
    start_cell = Cell(Position=[0, 0], Father_node=, Goal_cell=[100, 100])
    goal_cell = Cell(Position=[100, 100], Father_node=, Goal_cell=[100, 100])
    maze = Map(101, 101)

    maze.maze_obstacles_show(101, 101)
    maze.draw_start(start_cell)
    maze.draw_goal(goal_cell)
    print('-' * 150)
    maze.maze_show()

    astar_search = Astar(Start_cell=start_cell, Goal_cell=goal_cell, Map=maze)
    cells_around_list = astar_search.get_cells_around()
    available_cells_list = astar_search.get_available_cells(Cells_around_list=cells_around_list)
    next_cell = astar_search.search_next_cell(Cells_around_list=cells_around_list,
                                              Available_cells_list=available_cells_list)
    path_list = astar_search.get_path(Next_cell=next_cell)
    maze.maze_show()
