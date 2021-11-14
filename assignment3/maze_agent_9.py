import random
import numpy as np
import random

OBSTACLE = '#'
START = 'S'
target="T"
block_poss = 0.3
flat_poss = 0.3 + 0.7 / 3
hilly_poss = 0.3 + 0.7 * 2 / 3
forest_poss = 1


class Cell:
    def __init__(self, position: tuple, father_node=None):  # position = (x,y)
        self.position = position
        self.father_node = father_node
        self.gn = 0 if father_node is None else self.father_node.get_gn() + 1
        self.hn = None
        self.fn = None

        # assignment 3
        self.terrain = ""

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
        self.maze_terrain = [[None for i in range(width)] for j in range(height)]
        self.possibility = None
        self.target_position=None
        self.target_prob_before=None
        self.target_prob_after=None

    def initialize_maze(self, random_seed=None):
        '''
        Initialize maze according to the density P∈(0, 0.33)
        When random_seed has been set, the maze will not change
        '''
        if random_seed:
            random.seed(random_seed)
        for i in range(self.height):
            for j in range(self.width):
                random_num = random.random()
                # start cannot block
                if (i, j) == (0, 0):
                    self.data[i][j] = START
                    if random_num<=1/3:
                        self.maze_terrain[i][j]="flat"
                    elif 1/3<random_num<=2/3:
                        self.maze_terrain[i][j]="hilly"
                    if 2/3<random_num<=1:
                        self.maze_terrain[i][j]="forest"
                # generate obstacle
                elif random_num <= block_poss:
                    self.data[i][j] = OBSTACLE
                    self.maze_terrain[i][j] = "block"
                elif block_poss < random_num <= flat_poss:
                    self.maze_terrain[i][j] = "flat"
                elif flat_poss < random_num <= hilly_poss:
                    self.maze_terrain[i][j] = "hilly"
                elif hilly_poss < random_num <= forest_poss:
                    self.maze_terrain[i][j] = "forest"


    def expand_available_position(self):
        self.target_prob_after=np.zeros((self.width, self.height), dtype=int)
        before_postion=np.nonzero(self.target_prob_before!=0)
        x_list, y_list = before_postion[0].tolist(), before_postion[1].tolist()
        for index, x in enumerate(x_list):
            new_position=self.expand_position(x, y_list[index])
            for index_x, index_y in new_position:
                self.target_prob_after[index_x][index_y]+=self.target_prob_before[x][y_list[index]]




    def expand_position(self, index_x, index_y):
        return_position=[]
        surround_offset=[(1, 0), (-1, 0), (0, 1), (0, -1)]
        for _x, _y in surround_offset:
            x, y=index_x+_x, index_y+_y
            if self.position_is_valid(x, y) and not self.is_obstacle(x, y):
                return_position.append((x, y))

        return return_position

    def update_target_position(self):
        while True:
            x = random.randint(0, self.height-1)
            y = random.randint(0, self.width-1)
            if self.data[x][y] == "#":
                continue
            break
        self.data[x][y] = target
        self.target_position = (x, y)

    def update_poss(self, position: tuple, is_unreachable=False):
        x,y = position
        curr_poss = self.possibility[(x, y)]
        if (self.maze_terrain[x][y] == "block") or is_unreachable:
            self.possibility /= (1 - curr_poss)
            self.possibility[(x, y)] = 0
        elif self.maze_terrain[x][y] == "flat":
            self.possibility /= (1 - 0.8 * curr_poss)
            self.possibility[(x, y)] = 0.2 * curr_poss / (1 - 0.8 * curr_poss)
        elif self.maze_terrain[x][y] == "hilly":
            self.possibility /= (1 - 0.5 * curr_poss)
            self.possibility[(x, y)] = 0.5 * curr_poss / (1 - 0.5 * curr_poss)
        elif self.maze_terrain[x][y] == "forest":
            self.possibility /= (1 - 0.2 * curr_poss)
            self.possibility[(x, y)] = 0.8 * curr_poss / (1 - 0.2 * curr_poss)

    def find_next_goal(self, start_position:tuple, status):
        if status==1:
            max_poss_posi = np.nonzero(self.possibility == self.possibility.max())
        else:
            max_poss_posi = np.nonzero(self.target_prob_after == self.target_prob_after.max())
        if len(max_poss_posi[0]) == 1:
            return max_poss_posi[0][0], max_poss_posi[1][0]
        else:
            x, y = start_position
            distance = (max_poss_posi[1] - y) ** 2 + (max_poss_posi[0] - x) ** 2
            min_dis = np.nonzero(distance == distance.min())
            if len(min_dis[0]) == 1:
                return max_poss_posi[0][min_dis[0][0]], max_poss_posi[1][min_dis[0][0]]
            else:
                len_min_dis = len(min_dis[0])
                index = random.randint(0, len_min_dis - 1)
                return max_poss_posi[0][min_dis[0][index]], max_poss_posi[1][min_dis[0][index]]

        # if status==1:
        #     max_poss_posi=np.nonzero(self.possibility==self.possibility.max())
        #     if len(max_poss_posi[0])==1:
        #         return max_poss_posi[0][0],max_poss_posi[1][0]
        #     else:
        #         x, y = start_position
        #         distance=(max_poss_posi[1]-y)**2+(max_poss_posi[0]-x)**2
        #         min_dis=np.nonzero(distance==distance.min())
        #         if len(min_dis[0])==1:
        #             return max_poss_posi[0][min_dis[0][0]], max_poss_posi[1][min_dis[0][0]]
        #         else:
        #             len_min_dis=len(min_dis[0])
        #             index=random.randint(0, len_min_dis-1)
        #             return max_poss_posi[0][min_dis[0][index]], max_poss_posi[1][min_dis[0][index]]
        # else:
        #     max_poss_posi=np.nonzero(self.target_prob_after==self.target_prob_after.max())
        #     if len(max_poss_posi[0])==1:
        #         return max_poss_posi[0][0],max_poss_posi[1][0]
        #     else:
        #         x, y = start_position
        #         distance = (max_poss_posi[1] - y) ** 2 + (max_poss_posi[0] - x) ** 2
        #         min_dis = np.nonzero(distance == distance.min())
        #         if len(min_dis[0]) == 1:
        #             return max_poss_posi[0][min_dis[0][0]], max_poss_posi[1][min_dis[0][0]]
        #         else:
        #             len_min_dis = len(min_dis[0])
        #             index = random.randint(0, len_min_dis - 1)
        #             return max_poss_posi[0][min_dis[0][index]], max_poss_posi[1][min_dis[0][index]]
        #

    def find_max_prob_posi(self, sensed_position_list):
        max_prob=0
        surround_movable_postion=[]
        surround_offset_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for sensed_position in sensed_position_list:
            available_position = []
            pro_list=[]
            for surround_offset in surround_offset_list:
                index_x, index_y=surround_offset[0]+sensed_position[0], surround_offset[1]+sensed_position[1]
                if self.position_is_valid(index_x, index_y) and not self.is_obstacle(index_x, index_y):
                    available_position.append([index_x, index_y])
            for sub_p in available_position:
                pro_list.append(self.possibility[sub_p[0]][sub_p[1]])

            if len(pro_list)==0:
                continue
            cur_pro=sum(pro_list)/len(pro_list)
            if cur_pro>max_prob:
                surround_movable_postion=[]
                surround_movable_postion.append(available_position)
            elif cur_pro==max_prob:
                surround_movable_postion.append(available_position)

        return surround_movable_postion





    def get_target(self, position:tuple):
        x,y=position
        if self.position_have_target(x, y):
            poss_get_target = random.random()
            if self.maze_terrain[x][y]=="flat":
                if poss_get_target>0.2:
                    return True
            elif self.maze_terrain[x][y]=="hilly":
                if poss_get_target>0.5:
                    return True
            elif self.maze_terrain[x][y]=="forest":
                if poss_get_target>0.8:
                    return True
        return False

    def sense_surround(self, x, y):
        sensed_info=[]
        find_status=False
        surround_offset_list=[[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for surround_offset in surround_offset_list:
            index_x=x+surround_offset[0]
            index_y=y+surround_offset[1]
            if self.position_is_valid(index_x, index_y):
                sensed_info.append((index_x, index_y))
                if self.position_have_target(index_x, index_y):
                    find_status=True
        return sensed_info, find_status

    def target_move(self):
        tar_x, tar_y=self.target_position
        new_position_list=[]
        move_surround_list=[(1, 0), (-1, 0), (0, 1), (0, -1)]
        for move_surround in move_surround_list:
            new_tar_x, new_tar_y=tar_x+move_surround[0], tar_y+move_surround[1]
            if self.position_is_valid(new_tar_x, new_tar_y):
                if not self.is_obstacle(new_tar_x, new_tar_y):
                    new_position_list.append([new_tar_x, new_tar_y])

        new_posi_index=random.randint(0, len(new_position_list)-1)
        self.data[tar_x][tar_y]=0
        self.target_position=new_position_list[new_posi_index]
        self.data[self.target_position[0]][self.target_position[1]]=target


    def position_have_target(self, index_x, index_y):
        return self.data[index_x][index_y]==target

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
                child = Cell(position=(ni, nj), father_node=cell)  # set the father node simultaneously
                children_list.append(child)
        return children_list
