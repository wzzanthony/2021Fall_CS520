import json
from algorithm_pro3 import RepeatedForwardAStar
from maze_pro3 import Maze,Cell
import torch
import time
import math

import os

def euclidean_heuristic(cell1: Cell, cell2: Cell):
    x1, y1 = cell1.position
    x2, y2 = cell2.position
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))

class CNN_net(torch.nn.Module):
    def __init__(self):
        super(CNN_net,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2,
                            out_channels=16,
                            kernel_size=(3,3),
                            stride=(1,1),
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,(3,3),(1,1),1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,(3,3),(1,1),1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128,(3,3),(1,1),1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.mlp1 = torch.nn.Linear(3*3*128,12)
        self.mlp2=torch.nn.Linear(12, 4)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = x.view(x.size(0),-1)
        # print(x.shape)
        x = self.mlp1(x)
        # print(x.shape)
        x=self.mlp2(x)
        # print(x)
        return x

model_path='D:\\520_data\\model_for_project3\\new\\epoch_13_countOfFile0_Acc_0.8989250287770236_Loss_0.2607274265625538.pth'
model=torch.load(model_path,map_location=torch.device('cpu'))
model.eval()




def main():
    json_file_path='D:\\520_data\\compare_data\\data_for_pro3.json'
    with open(json_file_path, 'r') as a:
        save_data_str=a.read()
    save_data=json.loads(save_data_str)
    for index in save_data:
        print(index)
        with open(json_file_path, 'r') as a:
            save_data_str = a.read()
        save_data = json.loads(save_data_str)
        maze_file=save_data[index]['path']
        with open(maze_file, 'r') as a:
            str_maze = a.read()
        maze = json.loads(str_maze)
        new_maze=Maze(50,50)
        new_maze.data = maze["maze"]
        new_maze.maze_terrain = maze["terrain_maze"]
        new_maze.target_position = maze["target_position"]
        sense=RepeatedForwardAStar(new_maze, euclidean_heuristic)
        start_cell = Cell((0, 0))
        count=save_data[index]['origin']['length']
        if count>10000:
            count=1000
        else:
            count*=4
        st=time.time()
        path=sense.search(start_cell, model, count)
        ed=time.time()-st
        save_data[index]['mimic']['length']=path
        save_data[index]['mimic']['time']=ed
        save_data_str=json.dumps(save_data)
        with open(json_file_path, 'w') as a:
            a.write(save_data_str)



def test():
    path='D:\\git_project\\2021Fall_CS520\\assignment4\\train_data_project_3\\Data\\origin_maze\\1.json'
    with open(path, 'r') as a:
        data=a.read()
    data=json.loads(data)
    new_maze=Maze(50,50)
    new_maze.data=data["maze"]
    new_maze.maze_terrain=data["terrain_maze"]
    new_maze.target_position=data["target_position"]
    sense=RepeatedForwardAStar(new_maze,euclidean_heuristic)
    start_cell = Cell((0, 0))
    len_path=sense.search(start_cell, model, 10000)
    print(len_path)




if __name__ == '__main__':
    main()
    # test()