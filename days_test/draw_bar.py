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
    plt.xlabel('numbers of new VOC/VOI', fontsize=15)
    date_list = []
    for i in range(len(file)):
        date = (file.loc[i])["Date"]
        date_list.append(date.strftime('%Y-%m-%d'))
    real = file["Real counts"].values.tolist()
    knn = file["KNN"].values.tolist()
    lunar = file["LUNAR"].values.tolist()
    knn_lunar = file["KNN+LUNAR"].values.tolist()
    knn_knn = file["KNN+KNN"].values.tolist()
    lunar_knn = file["LUNAR+KNN"].values.tolist()
    plt.plot([], [], color='#DC143C', label='Real counts')
    plt.plot([], [], color='#6495ED', label='KNN')
    plt.plot([], [], color='#FFA07A', label='LUNAR')
    plt.plot([], [], color='#BDB76B', label='LUNAR+KNN')
    plt.plot([], [], color='#5F9EA0', label='KNN+KNN')
    plt.plot([], [], color='#FFC0CB', label='KNN+LUNAR')
    # color_l = ['#FF69B4','#6495ED','#FFA07A','#BDB76B','#5F9EA0','#87CEFA']
    width = 0.8
    x_1 = 5*np.arange(len(date_list))-2.7*width
    plt.barh(x_1,real,height=width,color='#DC143C')
    plt.barh(x_1+width,knn,height=width,color = '#6495ED')
    plt.barh(x_1+width*2,lunar,height=width,color='#FFA07A')
    plt.barh(x_1+width*3,lunar_knn,height=width,color='#BDB76B')
    plt.barh(x_1+width*4,knn_knn,height=width,color='#5F9EA0')
    plt.barh(x_1+width*5,knn_lunar,height=width,color='#FFC0CB')
    # ax = plt.gca()  # 表明设置图片的各个轴，plt.gcf()表示图片本身
    # ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))  # 横坐标标签显示的日期格式
    plt.yticks(range(0,5*len(date_list),5),date_list, rotation=0)  # 横坐标日期范围及间隔
    plt.xticks(range(0, 55, 5))
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
        path = "./结果拆分/"+name+".xlsx"
        file = pd.read_excel(path)
        draw_bar(file,name)
