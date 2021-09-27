import random

class maze:
    def __init__(self, rows:int, columns:int):
        self.rows=rows
        self.columns=columns

    def create_maze(self, probability: float, random_seed=None):
        if random_seed:
            random.seed(random_seed)
        maze = [[0 for i in range(self.rows)] for j in range(self.columns)]
        for index_row in range(self.rows):
            for index_column in range(self.columns):
                if index_row==0 and index_column==0:
                    continue
                if index_row==(self.rows-1) and index_column==(self.columns-1):
                    continue
                if random.random()<=probability:
                    maze[index_row][index_column]=1

        return maze