import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from pyod.models.knn import KNN
from pyod.models.lunar import LUNAR
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score,matthews_corrcoef


file = pd.read_csv("D:/work files/stacking_model/test/counry_all_kmers_info_clear_.csv")

def prepare_data(country,date): # 训练集30天测试集1天
    date_30 = date + timedelta(days=-30)
    date_num = int(date.strftime('%Y-%m-%d').replace("-",""))
    date_30_num = int(date_30.strftime('%Y-%m-%d').replace("-",""))
    test_sample = file.loc[(file["c_t_num"]==date_num)&(file["country"]==country)]
    test_sample_num = len(test_sample)
    train_sample = file.loc[(file["c_t_num"]<date_num)&(file["c_t_num"]>=date_30_num)&(file["country"]==country)]
    train_sample_num = len(train_sample)
    sample_count = {}
    sample_count["country"] = country
    sample_count["date"] = date.strftime('%Y-%m-%d')
    sample_count["test set"]=test_sample_num
    sample_count["train set"]=train_sample_num
    return sample_count,test_sample,train_sample


def prepare_input_train(date_file):
    input_array = date_file.iloc[:,8:]
    input_array.fillna(0,inplace = True)
    clade_list = date_file["nextstrain_clade"].values.tolist()
    return input_array,clade_list

def prepare_input_test(date_file,clade_list):
    input_array = date_file.iloc[:,8:]
    input_array.fillna(0,inplace = True)
    input_array = np.array(input_array)
    target_list_ = date_file["nextstrain_clade"].values.tolist()
    target_list = []
    classify_list = date_file["classify_target"].values.tolist()
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

def knn_model(train_set,test_set,test_target,country,date):
    model = KNN(n_neighbors=5)
    model.fit(train_set)
    pred_list = model.predict(test_set)
    matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    if (matrix[1, 1] + matrix[1, 0]) == 0:
        specifity = 0
    else:
        specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    metrics_dict = {}
    metrics_dict["country"]=country
    metrics_dict["date"] = date.strftime('%Y-%m-%d')
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

def lunar_model(train_set,test_set,test_target,country,date):
    model = LUNAR()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    if (matrix[1, 1] + matrix[1, 0]) == 0:
        specifity = 0
    else:
        specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    metrics_dict = {}
    metrics_dict["country"]=country
    metrics_dict["date"] = date.strftime('%Y-%m-%d')
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
    country_list = ["China","Argentina","Portugal"]

    for country in country_list:
        knn_l = []
        lunar_l = []
        sample_list = []
        path1 = country+"_days.xlsx"
        day_file = pd.read_excel(path1)
        for i in range(len(day_file)):
            date = (day_file.loc[i])["date"]
            sample_count, test_sample, train_sample = prepare_data(country,date)
            train_set, clade_list = prepare_input_train(train_sample)
            test_set, test_target = prepare_input_test(test_sample, clade_list)
            if len(train_sample) == 0:
                continue
            elif len(test_sample)==0:
                continue
            else:
                print(i, "+1")
                metric_knn = knn_model(train_set, test_set, test_target,country,date)
                knn_l.append(metric_knn)
                print(i, "+2")
                metric_lunar = lunar_model(train_set, test_set, test_target,country,date)
                lunar_l.append(metric_lunar)

        knn_f = pd.DataFrame(knn_l)
        lunar_f = pd.DataFrame(lunar_l)
        path2_1 = country + "_0904+knn.csv"
        path2_2 = country + "_0904+lunar.csv"
        knn_f.to_csv(path2_1)
        lunar_f.to_csv(path2_2)

    print(file)
