
import random


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Maze():

    obstacle_num = 10

    def __init__(self,Width,Height):
        self.width = Width
        self.height = Height
        self.data = []
        self.data = [[0 for i in range(Height)] for j in range(Width)]   #列表推导式 创建地图的初始值为0

    # Print out an empty maze
    def maze_empty_show(self):
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
        self.maze_empty_show()

    # Set start cell with 'S'
    def draw_start(self,cell):
        self.data[cell.x][cell.y] = "S"

    # Set goal cell with 'G'
    def draw_goal(self,cell):
        self.data[cell.x][cell.y] = "G"







start_cell = Cell(0, 0)
goal_cell = Cell(9,9)
maze = Maze(10,10)

maze.maze_obstacles_show(10,10)
maze.draw_start(start_cell)
maze.draw_goal(goal_cell)
print('-'*50)
maze.maze_empty_show()

