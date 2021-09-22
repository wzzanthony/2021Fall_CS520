# Time: 2021/9/21  23:19
import random

def generate_one_maze():
    data = [[0 for i in range(101)] for j in range(101)]
    positions = [(i, j) for i in range(0, 101) for j in range(0, 101)]
    for cell in positions:
        obstacle_probability = random.randint(0, 100)
        if obstacle_probability <= 30:
            data[cell[0]][cell[1]] = "#"

    print(data)

    with open("data.txt","w") as file:
        for i in range(len(data)):
            s = str(data[i]).replace('[', '').replace(']', '')
            s = s.replace("'", '').replace(',', '').replace('\'','') + '\n'
            file.write(s)
        file.close()
        print("保存文件成功")


def generate_100_maze():
    for p in range(100):
        data = [[0 for i in range(101)] for j in range(101)]
        positions = [(i, j) for i in range(0, 101) for j in range(0, 101)]
        for cell in positions:
            obstacle_probability = random.randint(0, 100)
            if obstacle_probability <= p:
                data[cell[0]][cell[1]] = "#"

        print(data)

        with open("data.txt","a") as f:                                                   #设置文件对象
            for i in data:                                                                 #对于双层列表中的数据
                i = str(i).strip('[').strip(']').replace(',','').replace('\'','')+'\n'  #将其中每一个列表规范化成字符串
                f.write(i)
