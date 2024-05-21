import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.dates as mdate

# 三个国家所有的毒株
# clade_list = ['19A','19B','20A','20B','20C','20D','20E','20G','20H','20I','20J','21A','21B','21C','21D','21F','21G','21H','21I','21J','21K','21L','21M','22A','22B','22C','22D','22E','22F','23A','23C']
month_all = ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12']
clade_list = ['19A','19B','20A','20B','20C','20D','20G','20H','20I','20J','21A','21C','21D','21F','21G','21H','21I','21J','21K','21L','21M','22A','22B','22C','22D','22E','22F','23A','23C']

file = pd.read_csv("./simple.csv")
# China = file.loc[(file["country"]=="China")]
Argentina = file.loc[(file["country"]=="Argentina")]
# Portugal = file.loc[(file["country"]=="Portugal")]

month_l = []
for t in Argentina['collection_time'].values.tolist():
    month = t[:7]
    month_l.append(month)
Argentina.insert(3,"collection_month",month_l)
time_clade_l = []
for i in month_all:
    for c in clade_list:
        count_num = len(Argentina.loc[(Argentina["collection_month"]==i)&(Argentina["nextstrain_clade"]==c)])
        time_clade = {}
        time_clade["month"]=i
        time_clade["clade"]=c
        time_clade["count"]=count_num
        time_clade_l.append(time_clade)
time_clade_DF = pd.DataFrame(time_clade_l)
dataframe_clade = []
c_l = []
for c in clade_list:
    q=time_clade_DF.loc[(time_clade_DF["clade"] == c)]["count"].tolist()
    c_l.append(q)


month_array = np.array(month_all)
xs = [datetime.strptime(d,'%Y-%m').date() for d in month_array]
plt.figure(dpi=300,figsize=(12,16))
plt.title('Clade count of Argentina',fontsize=28)  # 字体大小设置为25
plt.xlabel('month',fontsize=20)   # x轴显示“日期”，字体大小设置为10
plt.ylabel('numbers of clades',fontsize=20)

plt.plot( [], [], color='#8ECFC9',label='19A' )
plt.plot( [], [], color='#FFBE7A',label='19B' )
plt.plot( [], [], color='#FA7F6F',label='20A' )
plt.plot( [], [], color='#82B0D2',label='20B' )
plt.plot( [], [], color='#BEB8DC',label='20C' )
plt.plot( [], [], color='#E7DAD2',label='20D' )
# plt.plot( [], [], color='#2878b5',label='20E' )
plt.plot( [], [], color='#9ac9db',label='20G' )
plt.plot( [], [], color='#f8ac8c',label='20H' )
plt.plot( [], [], color='#c82423',label='20I' )
plt.plot( [], [], color='#ff8884',label='20J' )
plt.plot( [], [], color='#F6CAE5',label='21A' )
# plt.plot( [], [], color='#96CCCB',label='21B' )
plt.plot( [], [], color='#0c84c6',label='21C' )
plt.plot( [], [], color='#f74d4d',label='21D' )
plt.plot( [], [], color='#ffa510',label='21F' )
plt.plot( [], [], color='#002c53',label='21G' )
plt.plot( [], [], color='#41b7ac',label='21H' )
plt.plot( [], [], color='#F27970',label='21I' )
plt.plot( [], [], color='#BB9727',label='21J' )
plt.plot( [], [], color='#54B345',label='21K' )
plt.plot( [], [], color='#32B897',label='21L' )
plt.plot( [], [], color='#05B9E2',label='21M' )
plt.plot( [], [], color='#8983BF',label='22A' )
plt.plot( [], [], color='#C76DA2',label='22B' )
plt.plot( [], [], color='#A1A9D0',label='22C' )
plt.plot( [], [], color='#F0988C',label='22D' )
plt.plot( [], [], color='#B883D4',label='22E' )
plt.plot( [], [], color='#9E9E9E',label='22F' )
plt.plot( [], [], color='#CFEAF1',label='23A' )
plt.plot( [], [], color='#C4A5DE',label='23C' )
color_l = ['#8ECFC9','#FFBE7A','#FA7F6F','#82B0D2','#BEB8DC','#E7DAD2','#9ac9db','#f8ac8c','#c82423','#ff8884','#F6CAE5','#0c84c6','#f74d4d','#ffa510','#002c53','#41b7ac','#F27970','#BB9727','#54B345','#32B897','#05B9E2','#8983BF','#C76DA2','#A1A9D0','#F0988C','#B883D4','#9E9E9E','#CFEAF1','#C4A5DE']

bottom_y = [0] * 36
for i in range(len(c_l)):
    y = c_l[i]
    plt.bar(xs, y, width=25, bottom=bottom_y,color=color_l[i])
    # 累加数据计算新的bottom_y
    bottom_y = [a+b for a, b in zip(y, bottom_y)]


month_all_ = []
for i in month_all:
    date_month_ = {"01": "January", "02": "February", "03": "March", "04": "April",
                   "05": "May", "06": "June", "07": "July", "08": "August",
                   "09": "September", "10": "October", "11": "November", "12": "December"}
    month = date_month_[i[5:]]
    year = i[:4]
    month_name = month+" "+year
    month_all_.append(month_name)

ax = plt.gca()  # 表明设置图片的各个轴，plt.gcf()表示图片本身
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))  # 横坐标标签显示的日期格式
plt.xticks(xs,month_all_,rotation=90)  # 横坐标日期范围及间隔
plt.yticks(range(0,5000,200))
plt.tick_params(labelsize=14)
leg = plt.legend(ncol=3,prop={'size':24},loc="upper left")
for line in leg.get_lines():
    line.set_linewidth(8)
plt.savefig('./Argentina.png')
plt.show()
