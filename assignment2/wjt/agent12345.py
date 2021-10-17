import json
import os
from config import PS

from matplotlib import pyplot as plt


def json_read():
    path1 = 'data2/save.json'
    path2 = 'data3/save.json'
    path3 = 'data4/save.json'

    with open(path1, 'r') as json_reader:
        json_info1 = json_reader.read()

    with open(path2, 'r') as json_reader:
        json_info2 = json_reader.read()

    with open(path3, 'r') as json_reader:
        json_info3 = json_reader.read()

    data_123 = json.loads(json_info1)
    data_4 = json.loads(json_info3)
    data_5 = json.loads(json_info2)

    return data_123, data_4, data_5

if __name__ == '__main__':
    data_123, data_4, data_5 = json_read()

    current_file_path=os.path.dirname(os.path.abspath(__file__))
    new_dir=os.path.join(current_file_path, "data4_agent12345")
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(PS, data_123['data_time_agent1'], color='red')
    line2, = plt.plot(PS, data_123['data_time_agent2'], color='green')
    line3, = plt.plot(PS, data_123['data_time_agent3'], color='yellow')
    line4, = plt.plot(PS, data_4['data_time_agent4'], color='blue')
    line5, = plt.plot(PS, data_5['data_time_agent5'], color='orange')
    plt.xlabel('Density')
    plt.ylabel('Average time')
    plt.title('Density VS. Average time')
    plt.legend(handles=[line1, line2, line3, line4, line5],
               labels=['agent1', 'agent2', 'agent3', 'agent4', 'agent5'],
               loc='best')
    plt.savefig(os.path.join(new_dir, "1.png"))

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(PS, data_123['data_path_agent1'], color='red')
    line2, = plt.plot(PS, data_123['data_path_agent2'], color='green')
    line3, = plt.plot(PS, data_123['data_path_agent3'], color='yellow')
    line4, = plt.plot(PS, data_4['data_path_agent4'], color='blue')
    line5, = plt.plot(PS, data_5['data_path_agent5'], color='orange')
    plt.xlabel('Density')
    plt.ylabel('Total trajectory length')
    plt.title('Density VS. Total trajectory length')
    plt.legend(handles=[line1, line2, line3, line4, line5],
               labels=['agent1', 'agent2', 'agent3', 'agent4', 'agent5'],
               loc='best')
    plt.savefig(os.path.join(new_dir, "2.png"))

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(PS, data_123['final_path_agent1'], color='red')
    line2, = plt.plot(PS, data_123['final_path_agent2'], color='green')
    line3, = plt.plot(PS, data_123['final_path_agent3'], color='yellow')
    line4, = plt.plot(PS, data_4['final_path_agent4'], color='blue')
    line5, = plt.plot(PS, data_5['final_path_agent5'], color='orange')
    plt.xlabel('Density')
    plt.ylabel('Final path length through discovered gridworld')
    plt.title('Density VS. Final path length through discovered gridworld')
    plt.legend(handles=[line1, line2, line3, line4, line5],
               labels=['agent1', 'agent2', 'agent3', 'agent4', 'agent5'],
               loc='best')
    plt.savefig(os.path.join(new_dir, "3.png"))

    plt.figure(figsize=(8, 5))
    line1, = plt.plot(PS, data_123['replan_time_agent1'], color='red')
    line2, = plt.plot(PS, data_123['replan_time_agent2'], color='green')
    line3, = plt.plot(PS, data_123['replan_time_agent3'], color='yellow')
    line4, = plt.plot(PS, data_4['replan_time_agent4'], color='blue')
    line5, = plt.plot(PS, data_5['replan_time_agent5'], color='orange')
    plt.xlabel('Density')
    plt.ylabel('Re-plan time')
    plt.title('Density VS. Re-plan time')
    plt.legend(handles=[line1, line2, line3, line4, line5],
               labels=['agent1', 'agent2', 'agent3', 'agent4', 'agent5'],
               loc='best')
    plt.savefig(os.path.join(new_dir, "4.png"))