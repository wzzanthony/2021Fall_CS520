import json
from algorithm_pro2 import SensingRepeatedForwardAStar
from maze_for_ori_pro2 import Maze,Cell
import torch
import time

import os



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

        self.mlp1 = torch.nn.Linear(3*3*128,4)

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
        # print(x)
        return x

model_path='D:\\520_data\\model_for_project_1\\model_project1_update\\model\\epoch_16_Acc_0.7765689643197495_Loss_0.4051238639068539.pth'
model=torch.load(model_path,map_location=torch.device('cpu'))
model.eval()




def main():
    json_file_path='D:\\520_data\\compare_data\\data_for_pro2.json'
    with open(json_file_path, 'r') as a:
        save_data_str=a.read()
    save_data=json.loads(save_data_str)
    for index in save_data:
        print(index)
        maze_file=save_data[index]['path']
        with open(maze_file, 'r') as a:
            str_maze = a.read()
        maze = json.loads(str_maze)
        new_maze=Maze(50,50)
        new_maze.data=maze
        sense=SensingRepeatedForwardAStar(new_maze)
        goal_cell = Cell((49, 49))
        start_cell = Cell((0, 0))
        count=save_data[index]['origin']['length']
        st=time.time()
        path=sense.search(start_cell, goal_cell, model,count*4)
        ed=time.time()-st
        save_data[index]['mimic']['length']=path
        save_data[index]['mimic']['time']=ed
    save_data_str=json.dumps(save_data)
    with open(json_file_path, 'w') as a:
        a.write(save_data_str)






if __name__ == '__main__':
    main()