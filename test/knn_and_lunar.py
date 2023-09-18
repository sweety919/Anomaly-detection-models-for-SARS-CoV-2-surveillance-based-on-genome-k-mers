import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from pyod.models.knn import KNN
from pyod.models.lunar import LUNAR
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score,matthews_corrcoef



def prepare_file(country,date):
    that_day = file.loc[(file["country"]==country)&(file["collection_time"]==date)]
    date_ = datetime.strptime(date, '%Y-%m-%d')
    date_30_ = date_+timedelta(days=-30) #时间格式，变体出现前30日
    date_30 = date_30_.strftime('%Y-%m-%d')
    date_num = int(date.replace("-", ""))
    date_30_num = int(date_30.replace("-", ""))
    before_that_day = file.loc[(file["country"] == country) & (file["c_t_num"]<date_num)&(file["c_t_num"]>=date_30_num)]
    return that_day,before_that_day # 测试集、训练集

def prepare_file_2(country,date):
    date_ = datetime.strptime(date, '%Y-%m-%d')  # 时间格式，变体出现当日
    date_1_ = date_ + timedelta(days=-3)  # 时间格式，变体出现前3日
    date_1 = date_1_.strftime('%Y-%m-%d')
    date_2_ = date_ + timedelta(days=3)  # 时间格式，变体出现后3日
    date_2 = date_2_.strftime('%Y-%m-%d')
    date_30_ = date_ + timedelta(days=-33)  # 时间格式，训练集第一天
    date_30 = date_30_.strftime('%Y-%m-%d')
    date_1_num = int(date_1.replace("-", ""))
    date_2_num = int(date_2.replace("-", ""))
    date_30_num = int(date_30.replace("-", ""))
    test_days = file.loc[(file["country"] == country) & (file["c_t_num"] <= date_2_num) & (
                file["c_t_num"] >= date_1_num)]
    train_days = file.loc[(file["country"] == country) & (file["c_t_num"] < date_1_num) & (
                file["c_t_num"] >= date_30_num)]
    return test_days,train_days

def prepare_input_train(file):
    input_array = file.iloc[:,8:]
    input_array.fillna(0,inplace = True)
    clade_list = file["nextstrain_clade"].values.tolist()
    return input_array,clade_list

def prepare_input_test(file,clade_list):
    input_array = file.iloc[:,8:]
    input_array.fillna(0,inplace = True)
    input_array = np.array(input_array)
    target_list_ = file["nextstrain_clade"].values.tolist()
    target_list = []
    classify_list = file["classify_target"].values.tolist()
    for i in range(len(target_list_)):
        target = str(target_list_[i]).replace("[","").replace("]","").replace("'","")
        classify = str(classify_list[i]).replace("[","").replace("]","").replace("'","")
        if target in clade_list:
            a=0
        else:
            if classify=="positive":
                a=1
            else:
                a=0
        target_list.append(a)
    return input_array,target_list

def knn_model(train_set,test_set,test_target):
    model = KNN()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    if (matrix[1, 1] + matrix[1, 0]) == 0:
        specifity = 0
    else:
        specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    metrics_dict = {}
    metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    metrics_dict["specifity"] = specifity
    try:
        auc_value = roc_auc_score(test_target, pred_list)
        metrics_dict["auc"] = auc_value
    except ValueError:
        pass
    metrics_dict["condusion_matrix"] = matrix
    metrics_dict["result"] = pred_list
    metrics_dict["pred_v_num"] = (pred_list.tolist()).count(1)
    return metrics_dict

def lunar_model(train_set,test_set,test_target):
    model = LUNAR()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    if (matrix[1, 1] + matrix[1, 0]) == 0:
        specifity = 0
    else:
        specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    metrics_dict = {}
    metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    metrics_dict["specifity"] = specifity
    try:
        auc_value = roc_auc_score(test_target, pred_list)
        metrics_dict["auc"] = auc_value
    except ValueError:
        pass
    metrics_dict["condusion_matrix"] = matrix
    metrics_dict["result"] = pred_list
    metrics_dict["pred_v_num"] = (pred_list.tolist()).count(1)
    return metrics_dict


if __name__ == "__main__":
    file = pd.read_csv("./counry_all_kmers_info_clear_.csv")
    info_file = pd.read_csv("./pre_count_all.csv")
    knn_l = []
    lunar_l = []
    for i in range(len(info_file)):
        row = info_file.loc[i]
        country = row["country"]
        clade = row["clade"]
        date = row["date"]
        # test_file,train_file=prepare_file(country,date)
        test_file, train_file = prepare_file_2(country, date)  #测试集7天训练集30天
        train_set,clade_list= prepare_input_train(train_file)
        test_set,test_target = prepare_input_test(test_file,clade_list)
        if len(train_set)==0:
            continue
        elif (len(train_set)<=5)&(len(train_set)<len(test_set)):  # 防止knn划分不了聚类
            continue
        else:
            print(i, "+1")
            metric_knn = knn_model(train_set, test_set, test_target)
            knn_l.append(metric_knn)
            print(i, "+2")
            metric_lunar = lunar_model(train_set, test_set, test_target)
            lunar_l.append(metric_lunar)

    knn_f = pd.DataFrame(knn_l)
    lunar_f = pd.DataFrame(lunar_l)
    knn_f.to_csv("./test_knn0830(n=2).csv")
    lunar_f.to_csv("./test_lunar0830.csv")
    print(file)
