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
        self.empty=False
        self.obstacle=False
        self.visited=False
        self.num_sensed_neighbours = 0
        self.num_sensed_block = 0
        self.num_confirm_block = 0
        self.num_confirm_empty = 0
        self.num_unconfirmed_neighbours = 0

    def set_confirm_block(self, num):
        self.num_confirm_block=num

    def set_confirm_empty(self, num):
        self.num_confirm_empty=num

    def change_unconfirmed_neighbours(self,num):
        self.num_unconfirmed_neighbours-=num

    def set_block(self):
        self.obstacle=True

    def set_empty(self):
        self.empty=True

    def set_visit(self):
        self.visited=True

    def is_visited(self):
        return self.visited

    def is_block(self):
        return self.obstacle

    def is_empty(self):
        return self.empty

    def have_been_visited(self):
        return self.visited

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
        self.maze= [[Cell(position=(j,i)) for i in range(width)] for j in range(height)]

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
        self.maze[i][j].set_block()

    def set_empty(self,i,j):

        self.data[i][j]="1"
        self.maze[i][j].set_empty()

    def is_obstacle(self, i, j):
        '''
        Check if the position (i, j) is an obstacle
        '''
        return self.data[i][j] == '#'

    def is_empty(self,i,j):
        return self.data[i][j]=="1"


    def generate_children(self, cell: Cell):
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

    def sense_arround(self, cell:Cell):
        # first is block, second is empty
        arround_cell=[0,0]
        dij = [(1, 1), (1, -1), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (-1, 0)]
        i,j=cell.get_position()
        for di,dj in dij:
            ni,nj=i+di,j+dj
            if self.position_is_valid(ni, nj):
                if self.is_empty(ni,nj):
                    arround_cell[1]+=1
                elif self.is_obstacle(ni,nj):
                    arround_cell[0]+=1
        return arround_cell


    def generate_spacial_sensed_neighbours(self, cell:Cell):
        '''
        Generate neighbour cells that can be sensed by spacial sensing
        '''
        dij = [(1,1),(1,-1),(1,0),(0,1),(0,-1),(-1,1),(-1,-1),(-1,0)]    # eight cells can be sensed
        sensed_neighbours_list = []
        i, j = cell.get_position()
        for di, dj in dij:
            ni, nj = i + di, j + dj
            if self.position_is_valid(ni, nj):
                # each_sensed_neighbour = Cell(position=(ni, nj), father_node=cell)
                # sensed_neighbours_list.append(each_sensed_neighbour)
                sensed_neighbours_list.append([ni,nj])
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

    def get_unconfirmed_list(self, cell: Cell):
        dij = [(1, 1), (1, -1), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (-1, 0)]
        positions = []
        x, y = cell.get_position()
        for di, dj in dij:
            ni = x + di
            nj = y + dj
            if self.position_is_valid(ni, nj) and self.is_obstacle(ni,nj) and self.is_empty(ni,nj):
                positions.append((ni, nj))
        return positions

    @staticmethod
    def get_distinct(list1, list2):
        ret=set(list1)^set(list2)
        return list(ret)

    def inferMoreEmpty(self, cell1: Cell, cell2: Cell):
        block_num1 = cell1.num_sensed_block - cell1.num_confirm_block
        block_num2 = cell2.num_sensed_block - cell2.num_confirm_block
        list1 = self.get_unconfirmed_list(cell1)
        list2 = self.get_unconfirmed_list(cell2)
        # if block_num1 == block_num2 and self.is_subset(list1, list2):
        if block_num1 == block_num2 and set(list1).issubset(list2):
            return self.get_distinct(list1, list2)
        else:
            return []





