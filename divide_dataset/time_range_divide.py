import pandas as pd

# file = pd.read_csv("./counry_all_kmers_info_clear.csv")
file = pd.read_csv("D:/work files/stacking_model/total_just_voc_positive.csv")
time_set_1_train = file.loc[(file["c_t_num"]<=20201200)]
time_set_1_test = file.loc[(file["c_t_num"]<=20201231)&(file["c_t_num"]>20201200)]
time_set_2_train = file.loc[(file["c_t_num"]<=20211200)&(file["c_t_num"]>20210000)]
time_set_2_test = file.loc[(file["c_t_num"]<=20211231)&(file["c_t_num"]>20211200)]
time_set_3_train = file.loc[(file["c_t_num"]<=20221200)&(file["c_t_num"]>20220000)]
time_set_3_test = file.loc[(file["c_t_num"]<=20221231)&(file["c_t_num"]>20221200)]

a_1 = [set(time_set_1_train["nextstrain_clade"].tolist())]
a_2 = [set(time_set_1_test["nextstrain_clade"].tolist())]
b_1 = [set(time_set_2_train["nextstrain_clade"].tolist())]
b_2 = [set(time_set_2_test["nextstrain_clade"].tolist())]
c_1 = [set(time_set_3_train["nextstrain_clade"].tolist())]
c_2 = [set(time_set_3_test["nextstrain_clade"].tolist())]


# time_set_1_train.to_csv("./step1_dataset/time_set_1_train.csv",index=False)
# time_set_1_test.to_csv("./step1_dataset/time_set_1_test.csv",index=False)
# time_set_2_train.to_csv("./step1_dataset/time_set_2_train.csv",index=False)
# time_set_2_test.to_csv("./step1_dataset/time_set_2_test.csv",index=False)
# time_set_3_train.to_csv("./step1_dataset/time_set_3_train.csv",index=False)
# time_set_3_test.to_csv("./step1_dataset/time_set_3_test.csv",index=False)
time_set_1_train.to_csv("./重跑（仅VOC为positive）/step1_dataset/time_set_1_train.csv",index=False)
time_set_1_test.to_csv("./重跑（仅VOC为positive）/step1_dataset/time_set_1_test.csv",index=False)
time_set_2_train.to_csv("./重跑（仅VOC为positive）/step1_dataset/time_set_2_train.csv",index=False)
time_set_2_test.to_csv("./重跑（仅VOC为positive）/step1_dataset/time_set_2_test.csv",index=False)
time_set_3_train.to_csv("./重跑（仅VOC为positive）/step1_dataset/time_set_3_train.csv",index=False)
time_set_3_test.to_csv("./重跑（仅VOC为positive）/step1_dataset/time_set_3_test.csv",index=False)
print(file)