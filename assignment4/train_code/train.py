import torch
import os
import json
import numpy as np
from tqdm import tqdm
from logger import logger

from torch.utils.data import DataLoader, Dataset, TensorDataset


epochs=40
batch_size=1024 # must be power of 2
train_file_count=2000
offset=30
train_data_path='/common/home/jw1419/work/assignment4/train_data'




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


def get_train_data_list(train_data_path):
    # train_data_path='D:\\520_data\\train_data_0_1000'
    train_data_file_list=os.listdir(train_data_path)
    return train_data_file_list


def get_train_data(train_data_path, train_data_file_list,index_file, batch_size, shuffle, offset,train_file_count):
    print("Loading files************")
    train_data_x=[]
    train_data_y=[]
    if index_file>train_file_count:
        index_file=train_file_count-offset
    for train_data_file in tqdm(train_data_file_list[index_file:index_file+offset]):
        with open(os.path.join(train_data_path, train_data_file), 'r') as data_reader:
            train_data_str=data_reader.read()
        train_data_meta=json.loads(train_data_str)
        for i in train_data_meta.keys():
            input_data_1=[[0 for i1 in range(50)] for i2 in range(50)]
            one_train_data=train_data_meta[i]
            agent_x,agent_y=one_train_data["agent_position"]
            input_data_1[agent_x][agent_y]=1

            sub_train_data_x=[input_data_1, one_train_data["maze"]]
            train_data_x.append(sub_train_data_x)

            train_data_y.append(one_train_data["forward"])

    # train_data_x=np.array(train_data_x)
    # train_data_y=np.array(train_data_y)


    train_data_x=torch.tensor(train_data_x).float()
    train_data_y=torch.tensor(train_data_y).float()


    # train_data_x=torch.from_numpy(train_data_x).float()
    # train_data_x.to(device)
    # train_data_y=torch.from_numpy(train_data_y).float()
    # train_data_y.to(device)

    data_set_train=TensorDataset(train_data_x, train_data_y)
    train_data=DataLoader(dataset=data_set_train,
                          batch_size=batch_size,
                          shuffle=shuffle)
    print("Loading finish*************************")

    return train_data


def main():
    train_data_file_list=get_train_data_list(train_data_path)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=CNN_net()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print('current epoch is :', epoch)
        print('************************************')
        loss_list=[]
        acc_list=[]
        for count_index in range(train_file_count)[::offset]:
            accurancy_count=0
            accurancy_number=0
            loss_count=0
            loss_number=0
            train_data=get_train_data(train_data_path, train_data_file_list, count_index, batch_size, True, offset)
            for step, (batch_x, batch_y) in enumerate(tqdm(train_data)):
                batch_x,batch_y=batch_x.to(device), batch_y.to(device)
                output=model(batch_x)
                loss=loss_func(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accurancy_number+=torch.eq(output.argmax(1), batch_y.argmax(1)).sum().float().item()
                loss_number+=loss.item()
                accurancy_count+=len(batch_x)
                loss_count+=1
            loss_list.append(loss_number/loss_count)
            acc_list.append(accurancy_number/accurancy_count)
            logger.info("Train epoch: {}\t, index of files :{}\t, Acc :{}\t, Loss :{}".format(epoch, count_index, accurancy_number/accurancy_count, loss_number/loss_count))
        logger.info("Create model file: epoch_{}_Acc_{}_Loss_{}.pth".format(epoch, np.mean(acc_list), np.mean(loss_list)))
        torch.save(model, "/common/home/jw1419/work/assignment4/model/epoch_{}_Acc_{}_Loss_{}.pth".format(epoch, np.mean(acc_list), np.mean(loss_list)))

def test():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


def keep_training(model_path):
    train_data_file_list = get_train_data_list(train_data_path)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=torch.load(model_path)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print('current epoch is :', epoch)
        print('************************************')
        loss_list = []
        acc_list = []
        for count_index in range(train_file_count)[::offset]:
            accurancy_count = 0
            accurancy_number = 0
            loss_count = 0
            loss_number = 0
            train_data = get_train_data(train_data_path, train_data_file_list, count_index, batch_size, True, offset,train_file_count)
            for step, (batch_x, batch_y) in enumerate(tqdm(train_data)):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = loss_func(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accurancy_number += torch.eq(output.argmax(1), batch_y.argmax(1)).sum().float().item()
                loss_number += loss.item()
                accurancy_count += len(batch_x)
                loss_count += 1
            loss_list.append(loss_number / loss_count)
            acc_list.append(accurancy_number / accurancy_count)
            logger.info("Train epoch: {}\t, index of files :{}\t, Acc :{}\t, Loss :{}".format(epoch, count_index,
                                                                                              accurancy_number / accurancy_count,
                                                                                              loss_number / loss_count))
        logger.info(
            "Create model file: epoch_{}_Acc_{}_Loss_{}.pth".format(epoch, np.mean(acc_list), np.mean(loss_list)))
        torch.save(model, "/common/home/jw1419/work/assignment4/model/epoch_{}_Acc_{}_Loss_{}.pth".format(epoch,
                                                                                                          np.mean(
                                                                                                              acc_list),
                                                                                                          np.mean(
                                                                                                              loss_list)))
if __name__ == '__main__':
    model_name='epoch_30_Acc_0.7789467234192116_Loss_0.40038183451456155.pth'
    model_path=os.path.join('/common/home/jw1419/work/assignment4/model_project1', model_name)
    keep_training(model_path)


