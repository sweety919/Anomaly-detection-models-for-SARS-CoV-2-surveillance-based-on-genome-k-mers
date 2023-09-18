import pandas as pd

# file = pd.read_csv("./China_info_kmers.csv")
# # file = file.replace("21G, 21G","21G").replace("20J, 20J","20J").replace("21J, 21J","21J")
# clade = file["nextstrain_clade"].values.tolist()
# clade = list(set(clade))
# frame = pd.DataFrame()
# frame.insert(0,"clade",clade)
# # file.to_csv("./Argentina_info_kmers_1.csv",index=False)
# print(clade)

file = pd.read_csv("./Argentina_info_kmers.csv")
clade_file = pd.read_excel("./clade_who.xlsx")
who_clade = []
classify_target = []
clade = file["nextstrain_clade"].values.tolist()

for i in clade:
    i = str(i).replace("[","").replace("]","").replace("'","")
    who_clade_ = str(clade_file.loc[(clade_file["clade"]==i)]["who"].tolist()).replace("'","").replace("[","").replace("]","")
    who_clade.append(who_clade_)
    if who_clade_=="none":
        target = "neutral"
    else:
        target = "positive"
    classify_target.append(target)

file.insert(4,"who_clade",who_clade)
file.insert(5,"classify_target",classify_target)

file=file[~file['who_clade'].isin(["pass"])]

file.to_csv("./Argentina_info_kmers_target.csv",index=False)