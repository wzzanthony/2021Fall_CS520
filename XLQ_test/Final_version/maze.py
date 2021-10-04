import random

OBSTACLE = '#'
START = 'S'
GOAL = 'G'


class Cell:
    def __init__(self, position: tuple, father_node=None):  # position = (x,y)
        self.position = position
        self.father_node = father_node
        self.gn = 0 if father_node is None else self.father_node.get_gn() + 1
        self.hn = None
        self.fn = None
        # Spacial sensing
        self.num_sensed_neighbours = None
        self.num_sensed_block = None
        self.num_confirm_block = None
        self.num_confirm_empty = None
        self.num_unconfirmed_neighbours = None

    def get_position(self):
        return self.position

    def set_father_node(self, father):
        self.father_node = father

    def get_father_node(self):
        # change current node to father node
        return self.father_node

    def update_fn(self, heuristic, goal_cell: 'Cell'):
        self.hn = heuristic(self, goal_cell)
        self.fn = self.gn + self.hn

    def get_hn(self):
        '''
        the heuristic value, estimating the distance from the cell n to the goal node
        '''
        return self.hn

    def get_gn(self):
        '''
        this represents the length of the shortest path discovered from the initial search point to cell n so far
        '''
        return self.gn

    def get_fn(self):
        '''
        f(n) is defined to be g(n) + h(n), which estimates the distance from the initial search node to the final goal node through cell n
        '''
        return self.fn

    def __hash__(self):
        return hash(tuple(self.position))

    def __lt__(self, other):
        return self.position < other.position

    def __str__(self):
        return 'Cell({})'.format(self.position)

    def __repr__(self):
        return str(self)

    def __eq__(self, other: "Cell"):
        return tuple(self.position) == tuple(other.position)


class Maze:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.data = [[0 for i in range(width)] for j in range(height)]

    def initialize_maze(self, probability: float, random_seed=None):
        '''
        Initialize maze according to the density P∈(0, 0.33)
        When random_seed has been set, the maze will not change
        '''
        if random_seed:
            random.seed(random_seed)
        for i in range(self.height):
            for j in range(self.width):
                # the upper-left and lower-right corner
                if (i, j) == (0, 0):
                    self.data[i][j] = START
                elif (i, j) == (self.height - 1, self.width - 1):
                    self.data[i][j] = GOAL
                # generate obstacle
                elif random.random() <= probability:
                    self.data[i][j] = OBSTACLE

    def maze_show(self):
        for i in range(self.height):
            for j in range(self.width):
                print(self.data[i][j], end='')
            print()

    #     def obstacle
    #     def set_maze_obstacles
    #     def draw_start
    #     def draw_goal

    def position_is_valid(self, index_x, index_y):
        '''
        Check if the position (i, j) is valid, an valid position means it's in the maze that are not out of bound of the world
        '''
        return 0 <= index_x < self.height and 0 <= index_y < self.width

    def set_obstacle(self, i, j):
        '''
        Set obstacles with '#'
        '''
        self.data[i][j] = "#"

    def is_obstacle(self, i, j):
        '''
        Check if the position (i, j) is an obstacle
        '''
        return self.data[i][j] == '#'

    def generate_children(self, cell: Cell, goal_cell: Cell):
        '''
        Generate the children of n (neighbors believed or known to be unoccupied)
        '''
        dij = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        children_list = []
        # get current location, and trying to explore the adjacent cells
        i, j = cell.get_position()
        for di, dj in dij:
            ni, nj = i + di, j + dj
            # the position is valid and it's not an obstacle
            if self.position_is_valid(ni, nj) and not self.is_obstacle(ni, nj):
                child = Cell(position=(ni, nj), father_node=cell)   # set the father node simultaneously
                children_list.append(child)
        return children_list

    def generate_spacial_sensed_neighbours(self, cell:Cell):
        '''
        Generate neighbour cells that can be sensed by spacial sensing
        '''
        dij = [(1,1),(1,-1),(1,0),(0,1),(0,-1),(-1,1),(-1,-1),(-1,0)]    # eight cells can be sensed
        sensed_neighbours_list = []
        i, j = cell.get_position()
        for di, dj in dij:
            ni, nj = i + di, j + dj:
            if self.position_is_valid(ni, nj):
                each_sensed_neighbour = Cell(position=(ni, nj), father_node=cell)
                sensed_neighbours_list.append(each_sensed_neighbour)
        return sensed_neighbours_list

    def get_spacial_info(self, path_list:list):
        for cell in path_list:
            sensed_neighbours_list = self.generate_spacial_sensed_neighbours(cell)
            for neighbour in sensed_neighbours_list:
                obstacles_sum = 0
                x, y = neighbour.get_position()
                if self.is_obstacle(x,y):
                    obstacles_sum += 1
                neighbour.num_sensed_block = obstacles_sum






