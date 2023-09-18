import pandas as pd
from datetime import datetime,timedelta

def record_first_date(country,clade):
    select = file.loc[(file["country"]==country)&(file["nextstrain_clade"]==clade)]
    select_first = select[select["c_t_num"]==select["c_t_num"].min()]
    date = str((select_first["collection_time"].values.tolist())[0]).replace("[","").replace("]","").replace("'","")
    return date


def count_sample(country,date,clade): # 训练集30天测试集1天
    num_sample_1 = len(file.loc[(file["country"]==country)&(file["collection_time"]==date)]) # 变体出现当天样本数量
    num_clade = len(file.loc[(file["country"]==country)&(file["collection_time"]==date)&(file["nextstrain_clade"]==clade)]) #变体出现当天变体数量
    date_ = datetime.strptime(date, '%Y-%m-%d') # 时间格式，变体出现当日
    date_1_ = date_+timedelta(days=-1) #时间格式，变体出现前一日
    date_1 = date_1_.strftime('%Y-%m-%d')
    date_30_ = date_+timedelta(days=-30) #时间格式，变体出现前30日
    date_30 = date_30_.strftime('%Y-%m-%d')
    date_31_ = date_+timedelta(days=-31) #时间格式，变体出现前31日
    date_31 = date_31_.strftime('%Y-%m-%d')
    num_sample_2 = len(file.loc[(file["country"] == country) & (file["collection_time"] == date_1)]) # 变体出现前一天样本数量
    date_num = int(date.replace("-",""))
    date_1_num = int(date_1.replace("-",""))
    date_30_num = int(date_30.replace("-",""))
    date_31_num = int(date_31.replace("-",""))
    num_sample_3 = len(file.loc[(file["country"] == country) & (file["c_t_num"]<date_num)&(file["c_t_num"]>=date_30_num)]) # 变体出现前30天样本数量
    num_sample_4 = len(file.loc[(file["country"] == country) & (file["c_t_num"]<date_1_num)&(file["c_t_num"]>=date_31_num)]) # 变体出现前一天的30天前的样本数量
    num_dict = {}
    num_dict["country"]=country
    num_dict["clade"]=clade
    num_dict["date"]=date
    num_dict["变体出现当天变体数量"] = num_clade
    num_dict["变体出现当天样本数量"] = num_sample_1
    num_dict["变体出现前一天样本数量"] = num_sample_2
    num_dict["变体出现前30天样本数量"] = num_sample_3
    num_dict["变体出现前一天的30天前的样本数量"] = num_sample_4
    return num_dict

def count_sample_2(country,date,clade): # 训练集30天测试集7天
    date_ = datetime.strptime(date, '%Y-%m-%d') # 时间格式，变体出现当日
    date_1_ = date_+timedelta(days=-3) #时间格式，变体出现前3日
    date_1 = date_1_.strftime('%Y-%m-%d')
    date_2_ = date_+timedelta(days=3) #时间格式，变体出现后3日
    date_2 = date_2_.strftime('%Y-%m-%d')
    date_30_ = date_+timedelta(days=-33) #时间格式，训练集第一天
    date_30 = date_30_.strftime('%Y-%m-%d')
    date_1_num = int(date_1.replace("-",""))
    date_2_num = int(date_2.replace("-",""))
    date_30_num = int(date_30.replace("-",""))
    num_sample_1 = len(file.loc[(file["country"]==country)&(file["c_t_num"]<=date_2_num)&(file["c_t_num"]>=date_1_num)]) # 测试集样本数量
    num_clade = len(file.loc[(file["country"]==country)&(file["c_t_num"]<=date_2_num)&(file["c_t_num"]>=date_1_num)&(file["nextstrain_clade"]==clade)]) #测试集变体数量
    num_sample_2 = len(file.loc[(file["country"] == country) & (file["c_t_num"]<date_1_num)&(file["c_t_num"]>=date_30_num)]) # 变体出现前一天样本数量
    num_dict = {}
    num_dict["country"]=country
    num_dict["clade"]=clade
    num_dict["date"]=date
    num_dict["测试集变体数量"] = num_clade
    num_dict["测试集样本数量"] = num_sample_1
    num_dict["训练集30天样本数量"] = num_sample_2
    return num_dict


if __name__ == "__main__":
    file = pd.read_csv("./counry_all_kmers_info_clear_.csv")
    time_file = pd.read_excel("./voci_country.xlsx")
    # time_file = pd.read_excel("./time_list.xlsx")
    # time_file['date_str'] = time_file['date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    count_list = []
    for i in range(len(time_file)):
        row = time_file.loc[i]
        country = row["country"]
        clade = row["clade"]
        date = record_first_date(country,clade)
        # num_dict = count_sample(country,date,clade)
        num_dict = count_sample_2(country, date, clade)
        count_list.append(num_dict)
    count_frame = pd.DataFrame(count_list)
    count_frame.to_csv("./pre_count_all(30-7).csv",index=False)

