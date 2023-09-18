import pandas as pd

#part1
file = pd.read_csv("./1.csv")
i_list = []
for i in range(len(file)):
    if file.iloc[i,5]=="classify_target":
        i_list.append(i)
    else:
        continue

file_1 = file.drop(i_list,axis=0)
file_2 = file_1.drop(file_1.columns[0],axis=1)
file_2.to_csv("./counry_all_kmers_info.csv",index=False)
print(file)

#part2
file_ = pd.read_csv("./counry_all_kmers_info.csv")
print(file_["collection_time"].dtypes)
collection_time = file_["collection_time"].values.tolist()
collection_time_num = []
for i in collection_time:
    date = str(i).replace("[","").replace("]","").replace("'","").replace('"',"")
    date = int(date.replace("-",""))
    collection_time_num.append(date)
file_.insert(3,"c_t_num",collection_time_num)
file_.sort_values(by="c_t_num",inplace=True,ascending=True)
file_clear_1 = file.drop(file[file['c_t_num']<1000000].index)# 删除没有具体日期的行
file_clear_2 = file_clear_1.dropna(subset=["nextstrain_clade"])
file_clear_2.to_csv("./counry_all_kmers_info_clear.csv",index=False)


# part3
file = pd.read_csv("./counry_all_kmers_info_clear.csv")
country_list = []
name_list = file["name"].values.tolist()
for i in name_list:
    name = str(i).replace("[","").replace("]","").replace("'","").replace('"',"")
    a = name.split("/")[1]
    if a == "Portugal":
        b = "Portugal"
    elif a== "Argentina":
        b = "Argentina"
    else:
        b = "China"
    country_list.append(b)
file.insert(0,"country",country_list)
# file_simple = file[["country","name","accession_id","collection_time","c_t_num","who_clade","classify_target","nextstrain_clade"]]
# c = country_list.count("Argentina")
file.to_csv("./counry_all_kmers_info_clear_.csv",index=False)
print(file)