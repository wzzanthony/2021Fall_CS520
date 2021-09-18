import math
import random


class Cell():
    """
    create Node structure to generate tree
    """
    def __init__(self, position:list, goal_cell:list):    # position = [x, y]
        self.position = position
        self.father_node = None
        self.gn = 0
        # self.hn = math.sqrt((position[0] - goal_cell[0]) ** 2 + (position[1] - goal_cell[1]) ** 2)        # Use Euclidean Distance
        self.hn = abs(position[0] - goal_cell[0]) + abs(position[1] - goal_cell[1])                         # Use Manhattan Distance
        # self.hn = max(abs(position[0] - goal_cell[0]), abs(position[1] - goal_cell[1]))                   # Use Chebyshev Distance
        self.fn = self.hn + self.gn

    def set_father_node(self, node_name, gn):
        # ser father node of each point
        self.father_node = node_name

    def get_position(self):
        # get position information of the node
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

    def cells_around(self):
        cells_around = [[self.position[0]+1, self.position[1]],
                        [self.position[0]-1, self.position[1]],
                        [self.position[0], self.position[1]+1],
                        [self.position[0], self.position[1]-1]]
        return cells_around




class Map():

    obstacle_num = 5000  # obstacles number

    def __init__(self,Width,Height):
        self.width = Width
        self.height = Height
        self.data = [[0 for i in range(Height)] for j in range(Width)]

    # Print out an empty maze
    def maze_show(self):
        for i in range(self.width):
            for j in range(self.height):
                print(self.data[i][j],end =' ')
            print(' ')

    # Set obstacles with '#'
    def obstacle(self,x,y):
        self.data[x][y] = "#"

    # Print out a maze with obstacles
    def maze_obstacles_show(self,Obstacle_X,Obstacle_Y):
        for num in range(self.obstacle_num):
            i = random.randint(0, (Obstacle_X - 1))
            j = random.randint(0, (Obstacle_Y - 1))
            self.obstacle(i,j)
        self.maze_show()

    # Set start cell with 'S'
    def draw_start(self,cell:Cell):
        self.data[cell.position[0]][cell.position[1]] = "S"

    # Set goal cell with 'G'
    def draw_goal(self,cell:Cell):
        self.data[cell.position[0]][cell.position[1]] = "G"



class Astar():
    def __init__(self, start_cell:list, goal_cell:list, map:Map):
        self.start_point = start_cell  # 起点
        self.end_point = goal_cell  # 终点
        self.current = 0  # 当前节点
        self.map = map  # 地图
        self.openlist = []  # 打开节点  待测试的节点
        self.closelist = []  # 已经测试过的节点
        self.path = []  # 路径  存储每次选

    def isin_openlist(self, Cell:Cell):
        for item in self.openlist:
            if item == Cell.position:
                return True
            else:
                return False

    def isin_closelist(self, Cell:Cell):
        for item in self.closelist:
            if item == Cell.position:
                return True
            else:
                return False

    def is_obstacle(self, Cell:Cell):
        if self.map.data[Cell.position[0]] [Cell.position[1]] == '#':
            return True
        else:
            return False




start_cell = Cell(position=[0, 0],goal_cell=[101, 101])
goal_cell = Cell(position=[101, 101],goal_cell=[101, 101])
maze = Map(101, 101)

maze.maze_obstacles_show(101,101)
maze.draw_start(start_cell)
maze.draw_goal(goal_cell)
print('-'*150)
maze.maze_show()

