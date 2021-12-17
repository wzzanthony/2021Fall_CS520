import torch
import tqdm
import time
import os
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset


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


def test_surround(maze, current_position):
    cd, cx,cy=current_position[0][0], current_position[1][0], current_position[2][0]
    block_list=[[1],[],[]]
    for ox,oy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx=cx+ox
        ny=cy+oy
        if nx>=50 or nx<0:
            continue
        if ny>=50 or ny <0:
            continue
        if maze[nx][ny]==1:
            block_list[1].append(nx)
            block_list[2].append(ny)
        return block_list


def get_punish_list(current_position, punishment:dict):
    cd, cx,cy=current_position[0][0], current_position[1][0], current_position[2][0]
    punish_list=[]
    for ox,oy in [(-1,0), (1,0), (0,-1), (0,1)]:
        if cx+ox<0 or cx+ox>=50 or cy+oy<0 or cy+oy>=50:
            punish_list.append(1)
        else:
            punish_list.append(punishment[cx+ox][cy+oy])
    return punish_list

def move_forward(output, current_position, maze, punishment):
    movable=False
    punish_list=get_punish_list(current_position, punishment)
    while not movable:
        cd, cx,cy=current_position[0][0], current_position[1][0], current_position[2][0]
        a = output / torch.tensor(punish_list)
        forward=torch.argmax(output/torch.tensor(punish_list)).item()
        if forward==0:
            cx-=1
        elif forward==1:
            cx+=1
        elif forward==2:
            cy-=1
        else:
            cy+=1

        if cx>=50 or cx<0:
            output[0][forward]=-float('inf')
        elif cy>=50 or cy<0:
            output[0][forward]=-float('inf')
        elif maze[cx][cy]==1:
            output[0][forward]=-float('inf')
        else:
            current_position=[[cd],[cx],[cy]]
            movable=True
    punishment[cx][cy]=punishment[cx][cy]+0.5
    return current_position, punishment



def main(maze, ori_count):

    current_position=[[0],[0],[0]]
    discovered_maze=torch.zeros((2,50,50))
    punishment=[[1 for i in range(50)] for j in range(50)]
    for i in range(ori_count):
    # i=0
    # while True:
    #     i+=1
        if current_position==[[0],[49],[49]]:
            # print(i)
            # break
            return i
        block_list=test_surround(maze, current_position)
        if len(block_list[1])!=0:
            discovered_maze[block_list]=1
        input_data=discovered_maze
        input_data[current_position]=1
        output=model(input_data.unsqueeze(dim=0))
        output=torch.softmax(output, dim=1)+1
        # print(output)
        current_position, punishment=move_forward(output,current_position, maze, punishment)
        # print(current_position)
    return False


if __name__ == '__main__':
    json_file_path='D:\\520_data\\compare_data\\data_for_pro1.json'
    with open(json_file_path, 'r') as a:
        save_data_str=a.read()
    save_data=json.loads(save_data_str)
    for index in save_data:
        print(index)
        maze_file=save_data[index]['path']
        with open(maze_file, 'r') as a:
            str_maze = a.read()
        maze = json.loads(str_maze)
        ori_count=save_data[index]['origin']['length']
        st=time.time()
        count=main(maze, ori_count*4)
        ed=time.time()-st
        save_data[index]['mimic']['length']=count
        save_data[index]['mimic']['time']=ed

    save_data_str=json.dumps(save_data)
    with open(json_file_path, 'w') as a:
        a.write(save_data_str)


