import pandas as pd
import numpy as np
from datetime import datetime,timedelta

from pyod.models.knn import KNN
from pyod.models.lunar import LUNAR
from pyod.models.ocsvm import OCSVM
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


def prepare_input_train(country,date):
    pre_file = file.loc[(file["country"] == country) & (file["c_t_num"] <= int(date.strftime('%Y-%m-%d').replace("-", "")))]
    pre_file = pre_file.reset_index(drop=True)
    neutral_list = []
    for i in range(len(pre_file)):
        a = pre_file.loc[i,"classify_target"]
        if a == "neutral":
            neutral_list.append((file.iloc[i]).tolist())
        else:
            continue
    neutral_list_frame = pd.DataFrame(neutral_list)
    input_array = neutral_list_frame.iloc[:,8:]
    input_array.fillna(0,inplace = True)
    input_array = np.array(input_array)
    return input_array


def prepare_input_test(frame):
    input_array = frame.iloc[:, 8:]
    input_array.fillna(0, inplace=True)
    input_array = np.array(input_array)
    target_list_ = frame["classify_target"].values.tolist()
    target_list = []
    for i in target_list_:
        target = str(i).replace("[", "").replace("]", "").replace("'", "")
        if target == "neutral":
            a = 0
        else:
            a = 1
        target_list.append(a)
    return input_array, target_list


def ocsvm_model(train_set,test_set,test_target,country,date):
    model = OCSVM()
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
    metrics_dict["real_v_num"] = test_target.count(1)
    return metrics_dict


if __name__ == "__main__":

    country_list = ["China","Argentina","Portugal"]

    for country in country_list:
        whole_metrics = []
        sample_list = []
        path1 = country+"_days.xlsx"
        day_file = pd.read_excel(path1)
        for i in range(len(day_file)):
            date = (day_file.loc[i])["date"]
            sample_count, test_sample, train_sample = prepare_data(country,date)
            sample_list.append(sample_count)
            neutral_train = prepare_input_train(country,date)
            test_data,test_target = prepare_input_test(test_sample)
            metric = ocsvm_model(neutral_train,test_data,test_target,country,date)
            whole_metrics.append(metric)
        whole_metrics_f = pd.DataFrame(whole_metrics)
        path2_1 = country + "_0906+ocsvm.csv"
        whole_metrics_f.to_csv(path2_1)
    print(file)