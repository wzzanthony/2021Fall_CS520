import math
import queue
import random
import time


class Cell:

    def __init__(self, position: tuple, goal_position: tuple, father_node=None):  # position = [x, y]
        self.position = [0, 0]
        self.father_node = father_node

        # Heuristic cost from n to Goal_cell: hn

        # Use Euclidean Distance
        # self.hn = math.sqrt((position[0] - goal_cell[0]) ** 2 + (position[1] - goal_cell[1]) ** 2)

        # Use Manhattan Distance
        self.hn = abs(position[0] - goal_position[0]) + abs(position[1] - goal_position[1])

        # Use Chebyshev Distance
        # self.hn = max(abs(position[0] - goal_cell[0]), abs(position[1] - goal_cell[1]))

        # Real cost from Start_cell to n: g(n)
        if father_node is None:
            self.gn = 0
        else:
            self.gn = self.father_node.get_gn() + 1

        # Total cost: fn
        self.fn = self.hn + self.gn

    def set_father_node(self, node_name):
        '''
        Set father node for the point
        '''
        self.father_node = node_name
        if node_name == None:
            self.gn = 0
        else:
            self.gn = self.father_node.get_gn() + 1
        # # For test purpose
        print('Father node is: {}, current Gn = {}'.format(node_name,self.gn))

    def get_position(self):
        '''
        Get position information of the node
        '''
        return self.position

    def get_father_node(self):
        '''
        change current node to father node
        '''
        return self.father_node

    def get_hn(self):
        return self.hn

    def get_gn(self):
        return self.gn

    def get_fn(self):
        return self.fn

    def __hash__(self):
        return hash(tuple(self.position))

    def __eq__(self, other: "Cell"):
        return tuple(self.position) == tuple(other.position)

    def __str__(self):
        return 'Cell({})'.format(self.position)

    # For PriorityQueue
    def __lt__(self, other):
        return self.fn < other.get_fn()


class Maze:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = [[0 for i in range(height)] for j in range(width)]

    def show_maze(self):
        '''
        Print out an empty maze
        '''
        for i in range(self.width):
            for j in range(self.height):
                print(self.data[i][j], end=' ')
            print(' ')

    def obstacle(self, x, y):
        '''
        Set obstacles with '#'
        '''
        self.data[x][y] = "#"

    def set_maze_obstacles(self, obstacle_num):
        '''
        Set the obstacles, there are obstacle_num obstacles
        '''

        # Check if the obstacles number is less than total cells number
        if obstacle_num > self.width * self.height:
            msg = 'Number of obstacles is {}, which is more than total number of cells {}'.format(obstacle_num, self.width * self.height)
            raise AssertionError(msg)
        # Set obstacles
        for num in range(obstacle_num):
            i = random.randint(0, (self.width - 1))
            j = random.randint(0, (self.height - 1))
            self.obstacle(i, j)
        self.show_maze()

    def draw_start(self, cell: Cell):
        '''
        Set start cell with 'S'
        '''
        i, j = cell.get_position()
        self.data[i][j] = "S"

    def draw_goal(self, cell: Cell):
        '''
        Set goal cell with 'G'
        '''
        i, j = cell.get_position()
        self.data[i][j] = "G"

    def position_is_valid(self, i, j):
        '''
        Check if the position (i, j) is valid, an valid position means it's in the maze that
        are not out of bound of the world
        '''
        return 0 <= i < self.height and 0 <= j < self.width

    def is_obstacle(self, i, j):
        '''
        Check if the position (i, j) is an obstacle
        '''
        if self.data[i][j] == '#':
            return True
        return False

    def generate_children(self, current_cell: Cell, goal_cell: Cell):
        '''
        Generate the children of n (neighbors believed or known to be unoccupied)
        '''
        dij = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        children_list = []
        # get current location, and trying to explore the adjacent cells
        i, j = current_cell.get_position()
        for di, dj in dij:
            ni, nj = i + di, j + dj
            # the position is valid and it's not an obstacle
            if self.position_is_valid(ni, nj) and not self.is_obstacle(ni, nj):
                child = Cell(position=(ni, nj), goal_position=goal_cell.get_position(), father_node=current_cell)
                children_list.append(child)
        return children_list


class Astar():
    '''
    Astar algorithm
    '''
    def __init__(self, start_cell: Cell, goal_cell: Cell, maze: Maze):
        self.start_cell = start_cell  # Start point
        self.goal_cell = goal_cell  # Goal point
        self.maze = maze   # Whole maze

    def run_Astar(self):
        '''
        Use a priority queue, always select the cell with smallest fn to explore
        '''
        open_list = queue.PriorityQueue()
        open_list.put((self.start_cell.fn, self.start_cell))    # (fn, cell)
        closed_dict = dict()   # { current_cell : gn }
        # When open_list is not empty
        while not open_list.empty():
            fn, current_cell = open_list.get()
            # reach the goal cell
            if current_cell.get_position() == self.goal_cell.get_position():
                return self.get_path(current_cell)
            # the current cell has been visited
            if current_cell in closed_dict:
                continue
            closed_dict[current_cell] = current_cell.gn
            # Generate the children of n (neighbors believed or known to be unoccupied)
            for child_cell in self.maze.generate_children(current_cell, self.goal_cell):
                # The successors of n are the children n0 that are newly discovered, or g(n0) > g(n) + 1.

                # newly discovered, insert n0 into the open_list at priority f(n0) = g(n0) + h(n0),
                if child_cell not in closed_dict:
                    open_list.put((child_cell.fn, child_cell))
                # g(n0) > g(n) + 1, insert n0 into open_list at priority f(n0) = g(n0) + h(n0),
                elif child_cell.gn < closed_dict[child_cell]:
                    closed_dict.pop(child_cell)
                    open_list.put((child_cell.fn, child_cell))
        # If no path found
        return []

    def get_path(self, current_cell: Cell):
        '''
        Get the path
        '''
        path_list = []
        while current_cell != self.start_cell:
            path_list.append(current_cell)
            current_cell = current_cell.father_node
        path_list.append(self.start_cell)
        # reverse the list
        return path_list[::-1]



def main():
    # size of maze
    size = 101
    # number of obstacles
    num_obstacles = 2000
    # start and goal
    start_cell = Cell(position=(0, 0), goal_position=(size - 1, size - 1))
    goal_cell = Cell(position=(size - 1, size - 1), goal_position=(size - 1, size - 1))
    # Initialize maze
    maze = Maze(size, size)
    maze.set_maze_obstacles(num_obstacles)
    maze.draw_start(start_cell)
    maze.draw_goal(goal_cell)
    astar = Astar(start_cell=start_cell,goal_cell=goal_cell,maze=maze)
    # Calculate total time the A* research algorithm use
    start_time = time.time()
    path_list = astar.run_Astar()
    print("Total search time:", time.time() - start_time)

    if path_list is not []:
        print('Path:', path_list)
    else:
        print('No path found!')

if __name__ == '__main__':
    main()





