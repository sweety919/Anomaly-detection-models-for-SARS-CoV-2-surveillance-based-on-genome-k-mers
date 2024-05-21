import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.dates as mdate

file = pd.read_csv("./simple.csv")
China = file.loc[(file["country"]=="China")]
clade_list = list(set(China["nextstrain_clade"].values.tolist()))


month_l = []
for t in China['collection_time'].values.tolist():
    month = t[:7]
    month_l.append(month)
China.insert(3,"collection_month",month_l)
month_list = list(set(China["collection_month"].values.tolist()))
time_clade_l = []
for i in month_list:
    for c in clade_list:
        count_num = len(China.loc[(China["collection_month"]==i)&(China["nextstrain_clade"]==c)])
        time_clade = {}
        time_clade["month"]=i
        time_clade["clade"]=c
        time_clade["count"]=count_num
        time_clade_l.append(time_clade)
time_clade_DF = pd.DataFrame(time_clade_l)

month_list = month_list+["2020-05","2020-11"]
month_array = np.array(month_list)

dataframe_clade = []
c_l = []
for c in clade_list:
    num = []
    for d in month_list:
        clade_n = time_clade_DF.loc[(time_clade_DF["clade"]==c)&(time_clade_DF["month"]==d)]
        if clade_n.values.size:
            num_ = str(clade_n["count"].tolist()).replace("[","").replace("]","").replace("'","")
            num.append(int(num_))
        else:
            num.append(0)
    dict_num = {}
    dict_num["clade"]=c
    dict_num["num"]=num
    dataframe_clade.append(dict_num)
    c_l.append(num)
dataframe_clade_ = pd.DataFrame(dataframe_clade)
# dataframe_clade_.to_excel("./china_p.xlsx",index=False)

xs = [datetime.strptime(d,'%Y-%m').date() for d in month_array]


 # 设置纵坐标，使用range()函数设置起始、结束范围及间隔步长

plt.figure(dpi=100,figsize=(10,7))
plt.title('clades count',fontsize=12)  # 字体大小设置为25
plt.xlabel('month',fontsize=10)   # x轴显示“日期”，字体大小设置为10
plt.ylabel('numbers of clades',fontsize=10)


# a_19B = [0, 0, 0, 156, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 144, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# a_20A = [0, 1, 0, 6, 2, 0, 0, 0, 2, 0, 24, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 5, 0, 0, 0, 0, 0, 49, 43, 0, 0, 0, 0, 0, 0, 0]
# a_22F = [0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 3, 0, 0]
# a_22B = [0, 0, 0, 0, 0, 46, 3, 4449, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 0, 525, 0, 0, 0, 0, 801, 0, 0, 304, 0, 0]
# a_21K = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 106, 18, 0, 0, 38, 0, 0, 59, 0, 0, 0, 0, 0, 0]
# a_20H = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
# a_22C = [0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# a_21I = [0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 40, 0, 343, 12, 0, 2, 0, 0, 1, 6, 9, 0, 0, 0, 0, 2, 0, 0, 27, 0, 0, 0]
# a_20J = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]
# a_20I = [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 15, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# a_20D = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# a_21A = [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 47, 0, 3, 87, 0, 3, 0, 0, 3, 0, 48, 0, 0, 8, 0, 109, 0, 0, 21, 0, 0, 0]
# a_22E = [0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 0, 0, 33, 0, 0]
# a_23A = [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# a_21L = [0, 0, 0, 0, 0, 22, 602, 5, 0, 0, 0, 0, 174, 0, 0, 338, 0, 0, 0, 0, 35, 0, 85, 1, 0, 37, 171, 0, 0, 7, 3, 0, 0, 2, 0, 0]
# a_20B = [48, 0, 0, 1, 2, 0, 0, 0, 5, 4, 16, 0, 0, 6, 0, 0, 0, 5, 3, 13, 0, 0, 0, 0, 21, 0, 0, 0, 2, 0, 0, 6, 0, 0, 0, 0]
# a_21J = [0, 0, 84, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0, 27, 0, 2, 14, 0, 0, 0, 0, 3, 43, 0, 0, 0, 1, 0, 11, 0, 0, 314, 0, 0, 0]
# a_22D = [0, 0, 0, 0, 0, 2, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 1, 22, 0, 0, 8, 0, 0]
# a_22A = [0, 0, 0, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0]
# a_19A = [0, 0, 0, 303, 0, 0, 0, 0, 0, 0, 126, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 254, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# a_20C = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
plt.plot( [], [], color='#8ECFC9',label='19B' )
plt.plot( [], [], color='#FFBE7A',label='20A' )
plt.plot( [], [], color='#FA7F6F',label='22F' )
plt.plot( [], [], color='#82B0D2',label='22B' )
plt.plot( [], [], color='#BEB8DC',label='21K' )
plt.plot( [], [], color='#E7DAD2',label='20H' )
plt.plot( [], [], color='#2878b5',label='22C' )
plt.plot( [], [], color='#9ac9db',label='21I' )
plt.plot( [], [], color='#f8ac8c',label='20J' )
plt.plot( [], [], color='#c82423',label='20I' )
plt.plot( [], [], color='#ff8884',label='20D' )
plt.plot( [], [], color='#F27970',label='21A' )
plt.plot( [], [], color='#BB9727',label='22E' )
plt.plot( [], [], color='#54B345',label='23A' )
plt.plot( [], [], color='#32B897',label='21L' )
plt.plot( [], [], color='#05B9E2',label='20B' )
plt.plot( [], [], color='#8983BF',label='21J' )
plt.plot( [], [], color='#C76DA2',label='22D' )
plt.plot( [], [], color='#F27970',label='22A' )
plt.plot( [], [], color='#BB9727',label='19A' )
plt.plot( [], [], color='#54B345',label='20C' )

color_l = ['#8ECFC9','#FFBE7A','#FA7F6F','#82B0D2','#BEB8DC','#E7DAD2','#2878b5','#9ac9db','#f8ac8c','#c82423','#ff8884','#F27970','#BB9727','#54B345','#32B897','#05B9E2','#8983BF','#C76DA2','#F27970','#BB9727','#54B345']
width = 20
# 将bottom_y元素都初始化为0
bottom_y = [0] * 36
for i in range(len(c_l)):
    y = c_l[i]
    plt.bar(xs, y, width, bottom=bottom_y,color=color_l[i])
    plt.show()
    # 累加数据计算新的bottom_y
    bottom_y = [a+b for a, b in zip(y, bottom_y)]

##绘图代码省略，坐标轴设置如下
ax = plt.gca()  # 表明设置图片的各个轴，plt.gcf()表示图片本身
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))  # 横坐标标签显示的日期格式
time_gap_str = ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12']
plt.xticks(time_gap_str,rotation=90)  # 横坐标日期范围及间隔
plt.yticks(range(0, 4800, 600))
plt.legend()
plt.show()
print(file)
#8ECFC9
#FFBE7A
#FA7F6F
#82B0D2
#BEB8DC
#E7DAD2
#2878b5
#9ac9db
#f8ac8c
#c82423
#ff8884
#F27970
#BB9727
#54B345
#32B897
#05B9E2
#8983BF
#C76DA2
#A1A9D0
#F0988C
#B883D4
#9E9E9E
#CFEAF1
#C4A5DE
#F6CAE5
#96CCCB


# plt.plot( [], [], color='#8ECFC9',label='19A' )
# plt.plot( [], [], color='#FFBE7A',label='19B' )
# plt.plot( [], [], color='#FA7F6F',label='20A' )
# plt.plot( [], [], color='#82B0D2',label='20B' )
# plt.plot( [], [], color='#BEB8DC',label='20C' )
# plt.plot( [], [], color='#E7DAD2',label='20D' )
# plt.plot( [], [], color='#2878b5',label='20E' )
# plt.plot( [], [], color='#9ac9db',label='20G' )
# plt.plot( [], [], color='#f8ac8c',label='20H' )
# plt.plot( [], [], color='#c82423',label='20I' )
# plt.plot( [], [], color='#ff8884',label='20J' )
# plt.plot( [], [], color='#F6CAE5',label='21A' )
# plt.plot( [], [], color='#96CCCB',label='21B' )
# plt.plot( [], [], color='#0c84c6',label='21C' )
# plt.plot( [], [], color='#f74d4d',label='21D' )
# plt.plot( [], [], color='#ffa510',label='21F' )
# plt.plot( [], [], color='#002c53',label='21G' )
# plt.plot( [], [], color='#41b7ac',label='21H' )
# plt.plot( [], [], color='#F27970',label='21I' )
# plt.plot( [], [], color='#BB9727',label='21J' )
# plt.plot( [], [], color='#54B345',label='21K' )
# plt.plot( [], [], color='#32B897',label='21L' )
# plt.plot( [], [], color='#05B9E2',label='21M' )
# plt.plot( [], [], color='#8983BF',label='22A' )
# plt.plot( [], [], color='#C76DA2',label='22B' )
# plt.plot( [], [], color='#A1A9D0',label='22C' )
# plt.plot( [], [], color='#F0988C',label='22D' )
# plt.plot( [], [], color='#B883D4',label='22E' )
# plt.plot( [], [], color='#9E9E9E',label='22F' )
# plt.plot( [], [], color='#CFEAF1',label='23A' )
# plt.plot( [], [], color='#C4A5DE',label='23C' )
# color_l = ['#8ECFC9','#FFBE7A','#FA7F6F','#82B0D2','#BEB8DC','#E7DAD2','#2878b5','#9ac9db','#f8ac8c','#c82423','#ff8884','#F6CAE5','#96CCCB','#0c84c6','#f74d4d','#ffa510','#002c53','#41b7ac','#F27970','#BB9727','#54B345','#32B897','#05B9E2','#8983BF','#C76DA2','#A1A9D0','#F0988C','#B883D4','#9E9E9E','#CFEAF1','#C4A5DE']
