import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.dates as mdate

def draw_bar(file,name):
    plt.figure(dpi=100, figsize=(12, 14))
    plt.title(name, fontsize=20)  # 字体大小设置为25
    # plt.ylabel('month', fontsize=15)  # x轴显示“日期”，字体大小设置为10
    plt.xlabel('numbers of VOC/VOI', fontsize=15)
    date_list = []
    for i in range(len(file)):
        date = (file.loc[i])["Date"]
        date_list.append(date.strftime('%Y-%m-%d'))
    real = file["Real counts"].values.tolist()
    ocsvm = file["OCSVM"].values.tolist()
    plt.plot([], [], color='#DC143C', label='Real counts')
    plt.plot([], [], color='#00BFFF', label='OCSVM')
    width = 0.8
    x_1 = 5*np.arange(len(date_list))-width
    plt.barh(x_1,real,height=width,color='#DC143C')
    plt.barh(x_1+width,ocsvm,height=width,color = '#00BFFF')
    plt.yticks(range(0,5*len(date_list),5),date_list, rotation=0)  # 横坐标日期范围及间隔
    plt.xticks(range(0, 250, 50))
    plt.tick_params(labelsize=15)
    leg = plt.legend(ncol=1, prop={'size': 20}, loc="upper right")
    for line in leg.get_lines():
        line.set_linewidth(8)
    path = name + ".png"
    plt.savefig(path)
    plt.show()
    return

if __name__ == "__main__":
    file_name = ["Argentina", "China","Portugal"]
    for name in file_name:
        path = "./"+name+".xlsx"
        file = pd.read_excel(path)
        draw_bar(file,name)
