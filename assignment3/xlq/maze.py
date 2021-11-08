import random
import numpy as np

OBSTACLE = '#'
START = 'S'
GOAL = 'G'
TARGET = 'T'


class Cell:
    def __init__(self, position: tuple, father_node=None):  # position = (x,y)
        self.position = position
        self.father_node = father_node
        self.gn = 0 if father_node is None else self.father_node.get_gn() + 1
        self.hn = None
        self.fn = None
        self.terrain = ''

    # def set_terrain(self,terrain:str):
    #     self.terrain = terrain
    #
    # def set_prob_contain_target(self,num):
    #     self.prob_contain_target=num
    #
    # def get_prob_contain_target(self):
    #     return self.prob_contain_target
    #
    # def set_prob_find_target(self,p):
    #     if self.terrain == 'flat':
    #         self.prob_find_target = p * 0.8
    #     elif self.terrain == 'hilly':
    #         self.prob_find_target = p * 0.5
    #     elif self.terrain == 'forest':
    #         self.prob_find_target = p * 0.2

    # def get_prob_find_target(self):
    #     return self.prob_find_target
    #
    def get_terrain(self):
        return self.terrain

    def set_terrain(self, terrain):
        self.terrain = terrain

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
        self.terrain_maze = [[None for i in range(width)] for j in range(height)]
        self.prob_contain_mat = None
        self.prob_find_mat = None

    def initialize_maze(self, probability: float, random_seed=None):
        '''
        Initialize maze according to the density P∈(0, 0.33)
        When random_seed has been set, the maze will not change
        '''
        if random_seed:
            random.seed(random_seed)

        for i in range(self.height):
            for j in range(self.width):
                random_num = random.random()
                # set start cell
                if (i, j) == (0, 0):
                    self.data[i][j] = START
                    if random_num <= 1/3:
                        self.terrain_maze[i][j]='flat'
                    elif 1/3 < random_num <= 2/3:
                        self.terrain_maze[i][j]='hilly'
                    elif 2/3 < random_num <= 1:
                        self.terrain_maze[i][j]='forest'
                # generate obstacle
                elif random.random() <= probability:
                    self.data[i][j] = OBSTACLE
                    self.terrain_maze[i][j]='block'

                elif probability < random.random() <= (0.3+0.7/3):
                    self.terrain_maze[i][j]='flat'

                elif (0.3+0.7/3) <= random.random() <= (0.3+1.4/3):
                    self.terrain_maze[i][j]='hilly'

                elif (0.3+1.4/3) <= random.random() <= 1:
                    self.terrain_maze[i][j]='forest'
        # set target
        while True:
            target_x = random.randint(0, self.height-1)
            target_y = random.randint(0, self.width-1)
            if self.data[target_x][target_y] == '#':
                continue
            break
        self.data[target_x][target_y] = TARGET

    def update_prob(self, cell: Cell, is_unreachable=False):
        '''
        Question 2
        '''
        x, y = cell.get_position()
        current_prob = self.prob_contain_mat[(x, y)]
        if (self.terrain_maze[x][y] == "block") or is_unreachable:
            self.prob_contain_mat /= (1 - current_prob)
            self.prob_contain_mat[(x, y)] = 0
            self.prob_find_mat[(x, y)] = 0

        elif self.terrain_maze[x][y] == "flat":
            self.prob_contain_mat /= (1 - 0.8 * current_prob)
            self.prob_contain_mat[(x, y)] = 0.2 * current_prob / (1 - 0.8 * current_prob)
            self.prob_find_mat[(x, y)] = 0.8

        elif self.terrain_maze[x][y] == "hilly":
            self.prob_contain_mat /= (1 - 0.5 * current_prob)
            self.prob_contain_mat[(x, y)] = 0.5 * current_prob / (1 - 0.5 * current_prob)
            self.prob_find_mat[(x, y)] = 0.5

        elif self.terrain_maze[x][y] == "forest":
            self.prob_contain_mat /= (1 - 0.2 * current_prob)
            self.prob_contain_mat[(x, y)] = 0.8 * current_prob / (1 - 0.2 * current_prob)
            self.prob_find_mat[(x, y)] = 0.2

    def find_next_goal(self, cell: Cell):
        prob_fix_mat = self.prob_contain_mat * self.prob_find_mat
        max_prob_x, max_prob_y = np.nonzero(prob_fix_mat == prob_fix_mat.max())
        if len(max_prob_x) == 1 and len(max_prob_y) == 1:
            return Cell(position=(max_prob_x[0], max_prob_y[0]))
        else:
            x, y = cell.get_position()
            distance_lst = (max_prob_x - x)**2 + (max_prob_y - y)**2
            min_dist = np.nonzero(distance_lst == distance_lst.min())
            if len(min_dist[0])==1:
                return Cell(position=(max_prob_x[min_dist[0][0]], max_prob_y[min_dist[0][0]]))
            else:
                len_min_dis = len(min_dist[0])
                index = random.randint(0, len_min_dis - 1)
                return Cell(position=(max_prob_x[min_dist[0][index]], max_prob_y[min_dist[0][index]]))

    def maze_show(self):
        for i in range(self.height):
            for j in range(self.width):
                print(self.data[i][j], end='')
            print()

    def exam_target(self, cell: Cell):
        x, y = cell.get_position()
        if self.position_have_target(x, y):
            prob_get_target = random.random()
            if self.terrain_maze[x][y] == 'flat':
                if prob_get_target > 0.2:
                    return True
            elif self.terrain_maze[x][y] == 'hilly':
                if prob_get_target > 0.5:
                    return  True
            elif self.terrain_maze[x][y] == 'forest':
                if prob_get_target > 0.8:
                    return True
        return False

    def position_have_target(self, index_x, index_y):
        return self.data[index_x][index_y] == TARGET

    def position_is_valid(self, i, j):
        return 0 <= i < self.height and 0 <= j < self.width

    def set_obstacle(self, i, j):
        self.data[i][j] = "#"

    def is_obstacle(self, i, j):
        return self.data[i][j] == '#'

    def is_empty(self,i,j):
        return self.data[i][j]=="1"


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


    def get_unconfirmed_list(self, cell: Cell):
        dij = [(1, 1), (1, -1), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (-1, 0)]
        positions = []
        x, y = cell.get_position()
        for di, dj in dij:
            ni = x + di
            nj = y + dj
            if self.position_is_valid(ni, nj) and (self.data[ni][nj] == 0 or self.data[ni][nj] == "G" or self.data[ni][nj] == "S"):
                positions.append((ni, nj))
        return positions






