import pandas as pd

file = pd.read_csv("./Portugal_info_kmers.csv")
c_t = file["collection_time"].values.tolist()
collection_date = []
for i in c_t:
    i = str(i).replace("[","").replace("]","").replace("'","")
    a = i.split(" ")[0]
    date = a.split("(")[1]
    collection_date.append(date)
file = file.drop("collection_time",axis=1)
file.insert(2,"collection_time",collection_date)
print(file)


# 调整一下China文件列的顺序，与另外两个文件保持一致
# file = pd.read_csv("./China_info_kmers_target.csv")
# # file = file.drop(file.columns[1],axis=1)
# columns = list(file)
# columns.insert(6, columns.pop(columns.index('nextstrain_clade')))
# file = file.loc[:, columns]
# file.to_csv("./China_info_kmers_target_1.csv",index=False)