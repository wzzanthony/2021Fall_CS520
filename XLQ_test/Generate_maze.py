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
        print("*** Success ***")


def generate_100_maze():
    for p in range(0,100,10):
        data = [[0 for i in range(101)] for j in range(101)]

        positions = [(i, j) for i in range(0, 101) for j in range(0, 101)]
        for cell in positions:
            obstacle_probability = random.randint(0, 100)
            if obstacle_probability <= p:
                data[cell[0]][cell[1]] = "#"

        with open("data.txt", "a") as file:
            for i in range(len(data)):
                s = str(data[i]).replace('[', '').replace(']', '')
                s = s.replace("'", '').replace(',', '').replace('\'', '') + '\n'
                file.write(s)
            file.close()
            print(f"*** Success {int(p/10)} ***")